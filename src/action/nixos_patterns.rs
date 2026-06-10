// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! NixOS-Specific Action Patterns
//!
//! This module provides NixOS-aware action execution with:
//! - Generation-based rollback support
//! - Φ-gated confirmation for dangerous operations
//! - Command classification and safety scoring
//! - JSON output mode for structured results
//!
//! ## Design Philosophy
//!
//! NixOS is uniquely rollback-friendly. Every `nixos-rebuild switch` creates
//! a new generation that can be reverted. This module leverages that for safe
//! action execution with automatic rollback on failure.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::action::nixos_patterns::{NixOSExecutor, NixOSCommand};
//!
//! let mut executor = NixOSExecutor::new();
//!
//! // Capture current state before action
//! executor.capture_generation().await?;
//!
//! // Execute with Φ-awareness
//! let result = executor.execute_with_rollback(
//!     NixOSCommand::RebuildSwitch { flake: None },
//!     0.55,  // Current Φ level
//! ).await?;
//!
//! match result {
//!     ExecutionResult::Success(output) => println!("Done: {}", output),
//!     ExecutionResult::PendingConfirmation { .. } => println!("Needs confirmation"),
//!     ExecutionResult::RolledBack { error } => println!("Failed and rolled back: {}", error),
//! }
//! ```

use crate::consciousness::phi_attention::{ActionType, ConsciousnessThresholds, PhiAwareScoring};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::process::Stdio;
use tokio::process::Command;
use tracing::{info, warn};

/// NixOS-specific commands with structured parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NixOSCommand {
    /// nixos-rebuild switch (system-wide change)
    RebuildSwitch {
        /// Optional flake reference
        flake: Option<String>,
        /// Additional arguments
        extra_args: Vec<String>,
    },
    /// nixos-rebuild test (temporary, no boot entry)
    RebuildTest {
        flake: Option<String>,
        extra_args: Vec<String>,
    },
    /// nixos-rebuild boot (next reboot only)
    RebuildBoot {
        flake: Option<String>,
        extra_args: Vec<String>,
    },
    /// nix-env -i (user package install)
    EnvInstall {
        /// Package names or attributes
        packages: Vec<String>,
    },
    /// nix-env -e (user package remove)
    EnvRemove { packages: Vec<String> },
    /// nix-env --rollback (user profile rollback)
    EnvRollback,
    /// nix search (package search)
    Search {
        query: String,
        /// Use JSON output
        json: bool,
    },
    /// nix-channel operations
    Channel { operation: ChannelOperation },
    /// nix flake operations
    Flake { operation: FlakeOperation },
    /// home-manager switch
    HomeManagerSwitch { flake: Option<String> },
    /// nix-collect-garbage
    CollectGarbage {
        /// Delete older than N days
        older_than_days: Option<u32>,
        /// Delete everything not currently referenced
        delete_all: bool,
    },
    /// Custom command with safety classification
    Custom {
        command: String,
        args: Vec<String>,
        safety_level: SafetyLevel,
    },
}

/// Channel operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChannelOperation {
    /// nix-channel --update
    Update { channel: Option<String> },
    /// nix-channel --add
    Add { url: String, name: String },
    /// nix-channel --remove
    Remove { name: String },
    /// nix-channel --list
    List,
}

/// Flake operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FlakeOperation {
    /// nix flake update
    Update { inputs: Vec<String> },
    /// nix flake lock
    Lock { inputs: Vec<String> },
    /// nix flake show
    Show,
    /// nix flake check
    Check,
}

/// Safety levels for commands
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SafetyLevel {
    /// Read-only operations (search, list, show)
    ReadOnly,
    /// User-level modifications (nix-env)
    UserModify,
    /// System-level modifications (nixos-rebuild test)
    SystemModify,
    /// System-critical modifications (nixos-rebuild switch)
    SystemCritical,
    /// Destructive operations (garbage collection)
    Destructive,
}

impl SafetyLevel {
    /// Convert to Φ action type for threshold checking
    pub fn to_action_type(&self) -> ActionType {
        match self {
            Self::ReadOnly => ActionType::BasicQuery,
            Self::UserModify => ActionType::StateModifying,
            Self::SystemModify => ActionType::SystemCritical,
            Self::SystemCritical => ActionType::SystemCritical,
            Self::Destructive => ActionType::Irreversible,
        }
    }

    /// Get required Φ threshold
    pub fn required_phi(&self) -> f32 {
        ConsciousnessThresholds::default().threshold_for(self.to_action_type())
    }
}

impl NixOSCommand {
    /// Get the safety level of this command
    pub fn safety_level(&self) -> SafetyLevel {
        match self {
            Self::Search { .. } => SafetyLevel::ReadOnly,
            Self::Channel {
                operation: ChannelOperation::List,
            } => SafetyLevel::ReadOnly,
            Self::Flake {
                operation: FlakeOperation::Show,
            } => SafetyLevel::ReadOnly,
            Self::Flake {
                operation: FlakeOperation::Check,
            } => SafetyLevel::ReadOnly,

            Self::EnvInstall { .. } => SafetyLevel::UserModify,
            Self::EnvRemove { .. } => SafetyLevel::UserModify,
            Self::EnvRollback => SafetyLevel::UserModify,
            Self::Channel {
                operation: ChannelOperation::Update { .. },
            } => SafetyLevel::UserModify,
            Self::Channel {
                operation: ChannelOperation::Add { .. },
            } => SafetyLevel::UserModify,
            Self::Channel {
                operation: ChannelOperation::Remove { .. },
            } => SafetyLevel::UserModify,
            Self::Flake {
                operation: FlakeOperation::Update { .. },
            } => SafetyLevel::UserModify,
            Self::Flake {
                operation: FlakeOperation::Lock { .. },
            } => SafetyLevel::UserModify,
            Self::HomeManagerSwitch { .. } => SafetyLevel::UserModify,

            Self::RebuildTest { .. } => SafetyLevel::SystemModify,
            Self::RebuildBoot { .. } => SafetyLevel::SystemModify,

            Self::RebuildSwitch { .. } => SafetyLevel::SystemCritical,

            Self::CollectGarbage { .. } => SafetyLevel::Destructive,

            Self::Custom { safety_level, .. } => *safety_level,
        }
    }

    /// Get the rollback command if available
    pub fn rollback_command(&self) -> Option<NixOSCommand> {
        match self {
            Self::RebuildSwitch { .. } | Self::RebuildTest { .. } | Self::RebuildBoot { .. } => {
                Some(NixOSCommand::Custom {
                    command: "nixos-rebuild".to_string(),
                    args: vec!["switch".to_string(), "--rollback".to_string()],
                    safety_level: SafetyLevel::SystemCritical,
                })
            }
            Self::EnvInstall { .. } | Self::EnvRemove { .. } => Some(NixOSCommand::EnvRollback),
            Self::HomeManagerSwitch { .. } => {
                // Home-manager rollback is more complex
                Some(NixOSCommand::Custom {
                    command: "sh".to_string(),
                    args: vec![
                        "-c".to_string(),
                        "home-manager generations | head -2 | tail -1 | awk '{print $NF}' | xargs -I {} {}/activate".to_string(),
                    ],
                    safety_level: SafetyLevel::UserModify,
                })
            }
            _ => None,
        }
    }

    /// Convert to shell command and arguments
    pub fn to_command(&self) -> (String, Vec<String>) {
        match self {
            Self::RebuildSwitch { flake, extra_args } => {
                let mut args = vec!["switch".to_string()];
                if let Some(f) = flake {
                    args.push("--flake".to_string());
                    args.push(f.clone());
                }
                args.extend(extra_args.clone());
                ("nixos-rebuild".to_string(), args)
            }
            Self::RebuildTest { flake, extra_args } => {
                let mut args = vec!["test".to_string()];
                if let Some(f) = flake {
                    args.push("--flake".to_string());
                    args.push(f.clone());
                }
                args.extend(extra_args.clone());
                ("nixos-rebuild".to_string(), args)
            }
            Self::RebuildBoot { flake, extra_args } => {
                let mut args = vec!["boot".to_string()];
                if let Some(f) = flake {
                    args.push("--flake".to_string());
                    args.push(f.clone());
                }
                args.extend(extra_args.clone());
                ("nixos-rebuild".to_string(), args)
            }
            Self::EnvInstall { packages } => {
                let mut args = vec!["-iA".to_string()];
                for pkg in packages {
                    args.push(format!("nixpkgs.{pkg}"));
                }
                ("nix-env".to_string(), args)
            }
            Self::EnvRemove { packages } => {
                let mut args = vec!["-e".to_string()];
                args.extend(packages.clone());
                ("nix-env".to_string(), args)
            }
            Self::EnvRollback => ("nix-env".to_string(), vec!["--rollback".to_string()]),
            Self::Search { query, json } => {
                let mut args = vec!["search".to_string(), "nixpkgs".to_string(), query.clone()];
                if *json {
                    args.push("--json".to_string());
                }
                ("nix".to_string(), args)
            }
            Self::Channel { operation } => match operation {
                ChannelOperation::Update { channel } => {
                    let mut args = vec!["--update".to_string()];
                    if let Some(ch) = channel {
                        args.push(ch.clone());
                    }
                    ("nix-channel".to_string(), args)
                }
                ChannelOperation::Add { url, name } => (
                    "nix-channel".to_string(),
                    vec!["--add".to_string(), url.clone(), name.clone()],
                ),
                ChannelOperation::Remove { name } => (
                    "nix-channel".to_string(),
                    vec!["--remove".to_string(), name.clone()],
                ),
                ChannelOperation::List => ("nix-channel".to_string(), vec!["--list".to_string()]),
            },
            Self::Flake { operation } => match operation {
                FlakeOperation::Update { inputs } => {
                    let mut args = vec!["flake".to_string(), "update".to_string()];
                    args.extend(inputs.clone());
                    ("nix".to_string(), args)
                }
                FlakeOperation::Lock { inputs } => {
                    let mut args = vec!["flake".to_string(), "lock".to_string()];
                    for input in inputs {
                        args.push("--update-input".to_string());
                        args.push(input.clone());
                    }
                    ("nix".to_string(), args)
                }
                FlakeOperation::Show => (
                    "nix".to_string(),
                    vec!["flake".to_string(), "show".to_string()],
                ),
                FlakeOperation::Check => (
                    "nix".to_string(),
                    vec!["flake".to_string(), "check".to_string()],
                ),
            },
            Self::HomeManagerSwitch { flake } => {
                let mut args = vec!["switch".to_string()];
                if let Some(f) = flake {
                    args.push("--flake".to_string());
                    args.push(f.clone());
                }
                ("home-manager".to_string(), args)
            }
            Self::CollectGarbage {
                older_than_days,
                delete_all,
            } => {
                let mut args = vec!["-d".to_string()];
                if let Some(days) = older_than_days {
                    args.push("--delete-older-than".to_string());
                    args.push(format!("{days}d"));
                }
                if *delete_all {
                    args.push("--delete-old".to_string());
                }
                ("nix-collect-garbage".to_string(), args)
            }
            Self::Custom { command, args, .. } => (command.clone(), args.clone()),
        }
    }
}

/// Result of NixOS command execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionResult {
    /// Command succeeded
    Success {
        stdout: String,
        stderr: String,
        execution_time_ms: u64,
    },
    /// Command requires user confirmation (low Φ)
    PendingConfirmation {
        command: NixOSCommand,
        phi: f32,
        required_phi: f32,
        confidence: String,
    },
    /// Command failed but was rolled back
    RolledBack {
        error: String,
        rollback_output: String,
    },
    /// Command failed and rollback also failed
    FailedNoRollback {
        error: String,
        rollback_error: Option<String>,
    },
    /// Command was blocked by safety policy
    Blocked {
        reason: String,
        safety_level: SafetyLevel,
    },
}

/// NixOS-aware command executor with Φ integration
pub struct NixOSExecutor {
    /// Current system generation (for rollback)
    current_generation: Option<u32>,
    /// Consciousness thresholds
    thresholds: ConsciousnessThresholds,
    /// Execution history for learning
    history: VecDeque<ExecutionRecord>,
    /// Enable dry-run mode (don't actually execute)
    dry_run: bool,
}

/// Record of an execution for learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRecord {
    pub command: NixOSCommand,
    pub phi_at_execution: f32,
    pub result: ExecutionResult,
    pub timestamp_ms: u64,
}

impl Default for NixOSExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl NixOSExecutor {
    /// Create a new NixOS executor
    pub fn new() -> Self {
        Self {
            current_generation: None,
            thresholds: ConsciousnessThresholds::default(),
            history: VecDeque::new(),
            dry_run: false,
        }
    }

    /// Enable dry-run mode (log but don't execute)
    pub fn with_dry_run(mut self, dry_run: bool) -> Self {
        self.dry_run = dry_run;
        self
    }

    /// Set custom consciousness thresholds
    pub fn with_thresholds(mut self, thresholds: ConsciousnessThresholds) -> Self {
        self.thresholds = thresholds;
        self
    }

    /// Capture the current NixOS generation for rollback
    pub async fn capture_generation(&mut self) -> anyhow::Result<u32> {
        let output = Command::new("nixos-rebuild")
            .args(["list-generations"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await?;

        let stdout = String::from_utf8_lossy(&output.stdout);

        // Parse the current generation from output
        // Format: "  42   2024-01-15 14:30:00   (current)"
        for line in stdout.lines() {
            if line.contains("(current)") {
                if let Some(gen_str) = line.split_whitespace().next() {
                    if let Ok(r#gen) = gen_str.trim().parse::<u32>() {
                        self.current_generation = Some(r#gen);
                        info!(generation = r#gen, "Captured current NixOS generation");
                        return Ok(r#gen);
                    }
                }
            }
        }

        Err(anyhow::anyhow!("Could not determine current generation"))
    }

    /// Execute a NixOS command with Φ-aware safety
    pub async fn execute(&mut self, command: NixOSCommand, phi: f32) -> ExecutionResult {
        let safety = command.safety_level();
        let required_phi = safety.required_phi();

        // Check if Φ is sufficient
        if phi < required_phi {
            let confidence = PhiAwareScoring::confidence_level(phi);
            return ExecutionResult::PendingConfirmation {
                command,
                phi,
                required_phi,
                confidence: confidence.recommendation().to_string(),
            };
        }

        // Execute the command
        let (cmd, args) = command.to_command();

        info!(
            command = %cmd,
            args = ?args,
            phi = %phi,
            safety = ?safety,
            "Executing NixOS command"
        );

        if self.dry_run {
            return ExecutionResult::Success {
                stdout: format!("[DRY-RUN] Would execute: {} {}", cmd, args.join(" ")),
                stderr: String::new(),
                execution_time_ms: 0,
            };
        }

        let start = std::time::Instant::now();

        let result = Command::new(&cmd)
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await;

        let elapsed = start.elapsed().as_millis() as u64;

        match result {
            Ok(output) if output.status.success() => {
                let exec_result = ExecutionResult::Success {
                    stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                    stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                    execution_time_ms: elapsed,
                };

                self.record_execution(&command, phi, &exec_result);
                exec_result
            }
            Ok(output) => {
                let error = String::from_utf8_lossy(&output.stderr).to_string();
                warn!(error = %error, "Command failed, attempting rollback");

                // Attempt rollback
                if let Some(rollback_cmd) = command.rollback_command() {
                    let (rb_cmd, rb_args) = rollback_cmd.to_command();
                    let rb_result = Command::new(&rb_cmd)
                        .args(&rb_args)
                        .stdout(Stdio::piped())
                        .stderr(Stdio::piped())
                        .output()
                        .await;

                    match rb_result {
                        Ok(rb_output) if rb_output.status.success() => {
                            let exec_result = ExecutionResult::RolledBack {
                                error,
                                rollback_output: String::from_utf8_lossy(&rb_output.stdout)
                                    .to_string(),
                            };
                            self.record_execution(&command, phi, &exec_result);
                            exec_result
                        }
                        Ok(rb_output) => {
                            let exec_result = ExecutionResult::FailedNoRollback {
                                error,
                                rollback_error: Some(
                                    String::from_utf8_lossy(&rb_output.stderr).to_string(),
                                ),
                            };
                            self.record_execution(&command, phi, &exec_result);
                            exec_result
                        }
                        Err(e) => {
                            let exec_result = ExecutionResult::FailedNoRollback {
                                error,
                                rollback_error: Some(e.to_string()),
                            };
                            self.record_execution(&command, phi, &exec_result);
                            exec_result
                        }
                    }
                } else {
                    let exec_result = ExecutionResult::FailedNoRollback {
                        error,
                        rollback_error: None,
                    };
                    self.record_execution(&command, phi, &exec_result);
                    exec_result
                }
            }
            Err(e) => {
                let exec_result = ExecutionResult::FailedNoRollback {
                    error: e.to_string(),
                    rollback_error: None,
                };
                self.record_execution(&command, phi, &exec_result);
                exec_result
            }
        }
    }

    /// Force execution even if Φ is low (for confirmed actions)
    pub async fn execute_confirmed(&mut self, command: NixOSCommand, phi: f32) -> ExecutionResult {
        // Skip Φ check since user confirmed
        let (cmd, args) = command.to_command();

        info!(
            command = %cmd,
            args = ?args,
            phi = %phi,
            confirmed = true,
            "Executing confirmed NixOS command"
        );

        if self.dry_run {
            return ExecutionResult::Success {
                stdout: format!("[DRY-RUN] Would execute: {} {}", cmd, args.join(" ")),
                stderr: String::new(),
                execution_time_ms: 0,
            };
        }

        let start = std::time::Instant::now();

        let result = Command::new(&cmd)
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await;

        let elapsed = start.elapsed().as_millis() as u64;

        match result {
            Ok(output) if output.status.success() => ExecutionResult::Success {
                stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                execution_time_ms: elapsed,
            },
            Ok(output) => ExecutionResult::FailedNoRollback {
                error: String::from_utf8_lossy(&output.stderr).to_string(),
                rollback_error: None,
            },
            Err(e) => ExecutionResult::FailedNoRollback {
                error: e.to_string(),
                rollback_error: None,
            },
        }
    }

    /// Record an execution for history/learning
    fn record_execution(&mut self, command: &NixOSCommand, phi: f32, result: &ExecutionResult) {
        let record = ExecutionRecord {
            command: command.clone(),
            phi_at_execution: phi,
            result: result.clone(),
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        };
        self.history.push_back(record);

        // Keep history bounded
        if self.history.len() > 1000 {
            self.history.pop_front();
        }
    }

    /// Get execution history
    pub fn history(&self) -> &VecDeque<ExecutionRecord> {
        &self.history
    }

    /// Get success rate for a command type
    pub fn success_rate(&self, safety_level: SafetyLevel) -> Option<f32> {
        let matching: Vec<_> = self
            .history
            .iter()
            .filter(|r| r.command.safety_level() == safety_level)
            .collect();

        if matching.is_empty() {
            return None;
        }

        let successes = matching
            .iter()
            .filter(|r| matches!(r.result, ExecutionResult::Success { .. }))
            .count();

        Some(successes as f32 / matching.len() as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_command_safety_levels() {
        // Read-only
        let search = NixOSCommand::Search {
            query: "vim".to_string(),
            json: false,
        };
        assert_eq!(search.safety_level(), SafetyLevel::ReadOnly);

        // User modify
        let install = NixOSCommand::EnvInstall {
            packages: vec!["vim".to_string()],
        };
        assert_eq!(install.safety_level(), SafetyLevel::UserModify);

        // System critical
        let rebuild = NixOSCommand::RebuildSwitch {
            flake: None,
            extra_args: vec![],
        };
        assert_eq!(rebuild.safety_level(), SafetyLevel::SystemCritical);

        // Destructive
        let gc = NixOSCommand::CollectGarbage {
            older_than_days: Some(7),
            delete_all: false,
        };
        assert_eq!(gc.safety_level(), SafetyLevel::Destructive);
    }

    #[test]
    fn test_command_to_shell() {
        let install = NixOSCommand::EnvInstall {
            packages: vec!["vim".to_string(), "git".to_string()],
        };
        let (cmd, args) = install.to_command();
        assert_eq!(cmd, "nix-env");
        assert_eq!(args, vec!["-iA", "nixpkgs.vim", "nixpkgs.git"]);

        let search = NixOSCommand::Search {
            query: "editor".to_string(),
            json: true,
        };
        let (cmd, args) = search.to_command();
        assert_eq!(cmd, "nix");
        assert_eq!(args, vec!["search", "nixpkgs", "editor", "--json"]);
    }

    #[test]
    fn test_rollback_commands() {
        let rebuild = NixOSCommand::RebuildSwitch {
            flake: None,
            extra_args: vec![],
        };
        assert!(rebuild.rollback_command().is_some());

        let install = NixOSCommand::EnvInstall {
            packages: vec!["vim".to_string()],
        };
        assert!(install.rollback_command().is_some());

        let search = NixOSCommand::Search {
            query: "vim".to_string(),
            json: false,
        };
        assert!(search.rollback_command().is_none());
    }

    #[test]
    fn test_safety_to_phi() {
        assert_eq!(SafetyLevel::ReadOnly.required_phi(), 0.2);
        assert_eq!(SafetyLevel::UserModify.required_phi(), 0.3);
        assert_eq!(SafetyLevel::SystemCritical.required_phi(), 0.4);
        assert_eq!(SafetyLevel::Destructive.required_phi(), 0.6);
    }

    #[tokio::test]
    async fn test_dry_run_execution() {
        let mut executor = NixOSExecutor::new().with_dry_run(true);

        let search = NixOSCommand::Search {
            query: "vim".to_string(),
            json: false,
        };
        let result = executor.execute(search, 0.5).await;

        match result {
            ExecutionResult::Success { stdout, .. } => {
                assert!(stdout.contains("[DRY-RUN]"));
            }
            _ => panic!("Expected success"),
        }
    }

    #[tokio::test]
    async fn test_phi_gating() {
        let mut executor = NixOSExecutor::new().with_dry_run(true);

        // Try to execute system-critical with low Φ
        let rebuild = NixOSCommand::RebuildSwitch {
            flake: None,
            extra_args: vec![],
        };
        let result = executor.execute(rebuild, 0.2).await;

        match result {
            ExecutionResult::PendingConfirmation {
                phi, required_phi, ..
            } => {
                assert_eq!(phi, 0.2);
                assert!(required_phi > phi);
            }
            _ => panic!("Expected pending confirmation"),
        }
    }
}
