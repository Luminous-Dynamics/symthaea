//! NixOS-Specific Action Patterns
//!
//! Provides NixOS-aware action execution with:
//! - Generation-based rollback support
//! - Φ-gated confirmation for dangerous operations
//! - Command classification and safety scoring
//! - JSON output mode for structured results

use crate::traits::{ActionType, ConsciousnessThresholds, PhiAwareScoring};
use serde::{Deserialize, Serialize};
use std::process::Stdio;
use tokio::process::Command;
use tracing::{info, warn};

/// NixOS-specific commands with structured parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NixOSCommand {
    /// nixos-rebuild switch (system-wide change)
    RebuildSwitch {
        flake: Option<String>,
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
    EnvInstall { packages: Vec<String> },
    /// nix-env -e (user package remove)
    EnvRemove { packages: Vec<String> },
    /// nix-env --rollback (user profile rollback)
    EnvRollback,
    /// nix search (package search)
    Search { query: String, json: bool },
    /// nix-channel operations
    Channel { operation: ChannelOperation },
    /// nix flake operations
    Flake { operation: FlakeOperation },
    /// home-manager switch
    HomeManagerSwitch { flake: Option<String> },
    /// nix-collect-garbage
    CollectGarbage {
        older_than_days: Option<u32>,
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
    Update { channel: Option<String> },
    Add { url: String, name: String },
    Remove { name: String },
    List,
}

/// Flake operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FlakeOperation {
    Update { inputs: Vec<String> },
    Lock { inputs: Vec<String> },
    Show,
    Check,
}

/// Safety levels for commands
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SafetyLevel {
    ReadOnly,
    UserModify,
    SystemModify,
    SystemCritical,
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
            Self::EnvInstall { .. } | Self::EnvRemove { .. } => {
                Some(NixOSCommand::EnvRollback)
            }
            Self::HomeManagerSwitch { .. } => {
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
                    args.push(format!("nixpkgs.{}", pkg));
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
                    args.push(format!("{}d", days));
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
    Success {
        stdout: String,
        stderr: String,
        execution_time_ms: u64,
    },
    PendingConfirmation {
        command: NixOSCommand,
        phi: f32,
        required_phi: f32,
        confidence: String,
    },
    RolledBack {
        error: String,
        rollback_output: String,
    },
    FailedNoRollback {
        error: String,
        rollback_error: Option<String>,
    },
    Blocked {
        reason: String,
        safety_level: SafetyLevel,
    },
}

/// NixOS-aware command executor with Φ integration
pub struct NixOSExecutor {
    current_generation: Option<u32>,
    thresholds: ConsciousnessThresholds,
    history: Vec<ExecutionRecord>,
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
    pub fn new() -> Self {
        Self {
            current_generation: None,
            thresholds: ConsciousnessThresholds::default(),
            history: Vec::new(),
            dry_run: false,
        }
    }

    pub fn with_dry_run(mut self, dry_run: bool) -> Self {
        self.dry_run = dry_run;
        self
    }

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

        for line in stdout.lines() {
            if line.contains("(current)") {
                if let Some(gen_str) = line.split_whitespace().next() {
                    if let Ok(gen) = gen_str.trim().parse::<u32>() {
                        self.current_generation = Some(gen);
                        info!(generation = gen, "Captured current NixOS generation");
                        return Ok(gen);
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

        if phi < required_phi {
            let confidence = PhiAwareScoring::confidence_level(phi);
            return ExecutionResult::PendingConfirmation {
                command,
                phi,
                required_phi,
                confidence: confidence.recommendation().to_string(),
            };
        }

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
        self.history.push(record);

        if self.history.len() > 1000 {
            self.history.remove(0);
        }
    }

    pub fn history(&self) -> &[ExecutionRecord] {
        &self.history
    }

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
        let search = NixOSCommand::Search {
            query: "vim".to_string(),
            json: false,
        };
        assert_eq!(search.safety_level(), SafetyLevel::ReadOnly);

        let install = NixOSCommand::EnvInstall {
            packages: vec!["vim".to_string()],
        };
        assert_eq!(install.safety_level(), SafetyLevel::UserModify);

        let rebuild = NixOSCommand::RebuildSwitch {
            flake: None,
            extra_args: vec![],
        };
        assert_eq!(rebuild.safety_level(), SafetyLevel::SystemCritical);

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

    #[test]
    fn test_rebuild_with_flake_to_command() {
        let cmd = NixOSCommand::RebuildSwitch {
            flake: Some(".#myhost".to_string()),
            extra_args: vec!["--show-trace".to_string()],
        };
        let (bin, args) = cmd.to_command();
        assert_eq!(bin, "nixos-rebuild");
        assert!(args.contains(&"switch".to_string()));
        assert!(args.contains(&"--flake".to_string()));
        assert!(args.contains(&".#myhost".to_string()));
        assert!(args.contains(&"--show-trace".to_string()));
    }

    #[test]
    fn test_rebuild_test_and_boot_to_command() {
        let test_cmd = NixOSCommand::RebuildTest {
            flake: None,
            extra_args: vec![],
        };
        let (bin, args) = test_cmd.to_command();
        assert_eq!(bin, "nixos-rebuild");
        assert_eq!(args[0], "test");

        let boot_cmd = NixOSCommand::RebuildBoot {
            flake: None,
            extra_args: vec![],
        };
        let (bin, args) = boot_cmd.to_command();
        assert_eq!(bin, "nixos-rebuild");
        assert_eq!(args[0], "boot");
    }

    #[test]
    fn test_channel_operations_to_command() {
        let list = NixOSCommand::Channel {
            operation: ChannelOperation::List,
        };
        let (bin, args) = list.to_command();
        assert_eq!(bin, "nix-channel");
        assert!(args.contains(&"--list".to_string()));

        let update = NixOSCommand::Channel {
            operation: ChannelOperation::Update {
                channel: Some("nixos".into()),
            },
        };
        let (bin, args) = update.to_command();
        assert_eq!(bin, "nix-channel");
        assert!(args.contains(&"--update".to_string()));
        assert!(args.contains(&"nixos".to_string()));

        let add = NixOSCommand::Channel {
            operation: ChannelOperation::Add {
                url: "https://nixos.org/channels/nixpkgs-unstable".into(),
                name: "nixpkgs".into(),
            },
        };
        let (bin, args) = add.to_command();
        assert_eq!(bin, "nix-channel");
        assert!(args.contains(&"--add".to_string()));

        let remove = NixOSCommand::Channel {
            operation: ChannelOperation::Remove {
                name: "nixpkgs".into(),
            },
        };
        let (bin, args) = remove.to_command();
        assert_eq!(bin, "nix-channel");
        assert!(args.contains(&"--remove".to_string()));
    }

    #[test]
    fn test_flake_operations_to_command() {
        let update = NixOSCommand::Flake {
            operation: FlakeOperation::Update {
                inputs: vec!["nixpkgs".into()],
            },
        };
        let (bin, args) = update.to_command();
        assert_eq!(bin, "nix");
        assert!(args.contains(&"flake".to_string()));
        assert!(args.contains(&"update".to_string()));
        assert!(args.contains(&"nixpkgs".to_string()));

        let lock = NixOSCommand::Flake {
            operation: FlakeOperation::Lock {
                inputs: vec!["nixpkgs".into()],
            },
        };
        let (bin, args) = lock.to_command();
        assert_eq!(bin, "nix");
        assert!(args.contains(&"lock".to_string()));
        assert!(args.contains(&"--update-input".to_string()));
    }

    #[test]
    fn test_home_manager_to_command() {
        let hm = NixOSCommand::HomeManagerSwitch {
            flake: Some(".".into()),
        };
        let (bin, args) = hm.to_command();
        assert_eq!(bin, "home-manager");
        assert!(args.contains(&"switch".to_string()));
        assert!(args.contains(&"--flake".to_string()));
    }

    #[test]
    fn test_gc_to_command() {
        let gc = NixOSCommand::CollectGarbage {
            older_than_days: Some(30),
            delete_all: false,
        };
        let (bin, args) = gc.to_command();
        assert_eq!(bin, "nix-collect-garbage");
        assert!(args.contains(&"--delete-older-than".to_string()));
        assert!(args.contains(&"30d".to_string()));

        let gc_all = NixOSCommand::CollectGarbage {
            older_than_days: None,
            delete_all: true,
        };
        let (_, args) = gc_all.to_command();
        assert!(args.contains(&"--delete-old".to_string()));
    }

    #[test]
    fn test_custom_command_safety() {
        let custom = NixOSCommand::Custom {
            command: "echo".to_string(),
            args: vec!["hello".to_string()],
            safety_level: SafetyLevel::ReadOnly,
        };
        assert_eq!(custom.safety_level(), SafetyLevel::ReadOnly);
        let (bin, args) = custom.to_command();
        assert_eq!(bin, "echo");
        assert_eq!(args, vec!["hello"]);
    }

    #[test]
    fn test_all_safety_levels_complete() {
        // Verify every safety level maps to a valid action type
        for level in [
            SafetyLevel::ReadOnly,
            SafetyLevel::UserModify,
            SafetyLevel::SystemModify,
            SafetyLevel::SystemCritical,
            SafetyLevel::Destructive,
        ] {
            let phi = level.required_phi();
            assert!(
                (0.0..=1.0).contains(&phi),
                "Phi for {:?} out of range: {}",
                level,
                phi
            );
        }
    }

    #[test]
    fn test_rollback_all_rebuild_variants() {
        let switch = NixOSCommand::RebuildSwitch {
            flake: None,
            extra_args: vec![],
        };
        let test = NixOSCommand::RebuildTest {
            flake: None,
            extra_args: vec![],
        };
        let boot = NixOSCommand::RebuildBoot {
            flake: None,
            extra_args: vec![],
        };
        let hm = NixOSCommand::HomeManagerSwitch { flake: None };
        let gc = NixOSCommand::CollectGarbage {
            older_than_days: None,
            delete_all: false,
        };

        assert!(switch.rollback_command().is_some());
        assert!(test.rollback_command().is_some());
        assert!(boot.rollback_command().is_some());
        assert!(hm.rollback_command().is_some());
        assert!(
            gc.rollback_command().is_none(),
            "GC should not have rollback"
        );
    }

    #[tokio::test]
    async fn test_dry_run_skips_history() {
        // Dry-run returns before recording to history (by design)
        let mut executor = NixOSExecutor::new().with_dry_run(true);
        assert!(executor.history().is_empty());

        let cmd = NixOSCommand::Search {
            query: "test".into(),
            json: false,
        };
        executor.execute(cmd, 0.5).await;
        assert!(
            executor.history().is_empty(),
            "Dry-run should not record to history"
        );
    }

    #[tokio::test]
    async fn test_phi_gated_skips_history() {
        // Phi-gated PendingConfirmation also doesn't record
        let mut executor = NixOSExecutor::new().with_dry_run(true);
        let cmd = NixOSCommand::RebuildSwitch {
            flake: None,
            extra_args: vec![],
        };
        executor.execute(cmd, 0.1).await; // phi=0.1 < 0.4 required
        assert!(
            executor.history().is_empty(),
            "Phi-gated should not record to history"
        );
    }

    #[test]
    fn test_success_rate_no_history() {
        let executor = NixOSExecutor::new();
        assert!(executor.success_rate(SafetyLevel::ReadOnly).is_none());
        assert!(executor.success_rate(SafetyLevel::UserModify).is_none());
    }

    #[test]
    fn test_command_serde_roundtrip() {
        let cmd = NixOSCommand::RebuildSwitch {
            flake: Some(".#host".into()),
            extra_args: vec!["--show-trace".into()],
        };
        let json = serde_json::to_string(&cmd).unwrap();
        let restored: NixOSCommand = serde_json::from_str(&json).unwrap();
        let (bin, args) = restored.to_command();
        assert_eq!(bin, "nixos-rebuild");
        assert!(args.contains(&".#host".to_string()));
    }
}
