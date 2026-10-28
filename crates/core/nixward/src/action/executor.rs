// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! NixOS-Specific Action Patterns
//!
//! Provides NixOS-aware action execution with:
//! - Generation-based rollback support
//! - Φ-gated confirmation for dangerous operations
//! - Command classification and safety scoring
//! - JSON output mode for structured results

use super::authorization::{
    ActionScope, AuthorizationAttempt, AuthorizationAuthority, CommandBinding,
    ExecutionAuthorizationLedger, ExecutionPermit,
};
use crate::traits::{ActionType, ConsciousnessThresholds, PhiAwareScoring};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::future::Future;
use std::io;
use std::pin::Pin;
use std::process::{Output, Stdio};
use std::sync::Arc;
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
    /// Create a Custom command with auto-classified safety level.
    ///
    /// Uses `classify_command_destructiveness` from `phi_gate` to infer
    /// the safety level from the command string, rather than requiring
    /// the caller to specify it manually.
    pub fn custom_auto(command: &str, args: Vec<String>) -> Self {
        let full_cmd = if args.is_empty() {
            command.to_string()
        } else {
            format!("{} {}", command, args.join(" "))
        };
        let safety_level = super::phi_gate::classify_command_destructiveness(&full_cmd);
        Self::Custom {
            command: command.to_string(),
            args,
            safety_level,
        }
    }

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
                let flake_extra = if flake.is_some() { 2 } else { 0 };
                let mut args = Vec::with_capacity(1 + flake_extra + extra_args.len());
                args.push("switch".to_string());
                if let Some(f) = flake {
                    args.push("--flake".to_string());
                    args.push(f.clone());
                }
                args.extend(extra_args.iter().cloned());
                ("nixos-rebuild".to_string(), args)
            }
            Self::RebuildTest { flake, extra_args } => {
                let flake_extra = if flake.is_some() { 2 } else { 0 };
                let mut args = Vec::with_capacity(1 + flake_extra + extra_args.len());
                args.push("test".to_string());
                if let Some(f) = flake {
                    args.push("--flake".to_string());
                    args.push(f.clone());
                }
                args.extend(extra_args.iter().cloned());
                ("nixos-rebuild".to_string(), args)
            }
            Self::RebuildBoot { flake, extra_args } => {
                let flake_extra = if flake.is_some() { 2 } else { 0 };
                let mut args = Vec::with_capacity(1 + flake_extra + extra_args.len());
                args.push("boot".to_string());
                if let Some(f) = flake {
                    args.push("--flake".to_string());
                    args.push(f.clone());
                }
                args.extend(extra_args.iter().cloned());
                ("nixos-rebuild".to_string(), args)
            }
            Self::EnvInstall { packages } => {
                let mut args = Vec::with_capacity(1 + packages.len());
                args.push("-iA".to_string());
                for pkg in packages {
                    args.push(format!("nixpkgs.{pkg}"));
                }
                ("nix-env".to_string(), args)
            }
            Self::EnvRemove { packages } => {
                let mut args = Vec::with_capacity(1 + packages.len());
                args.push("-e".to_string());
                args.extend(packages.iter().cloned());
                ("nix-env".to_string(), args)
            }
            Self::EnvRollback => ("nix-env".to_string(), vec!["--rollback".to_string()]),
            Self::Search { query, json } => {
                let cap = if *json { 4 } else { 3 };
                let mut args = Vec::with_capacity(cap);
                args.push("search".to_string());
                args.push("nixpkgs".to_string());
                args.push(query.clone());
                if *json {
                    args.push("--json".to_string());
                }
                ("nix".to_string(), args)
            }
            Self::Channel { operation } => match operation {
                ChannelOperation::Update { channel } => {
                    let cap = if channel.is_some() { 2 } else { 1 };
                    let mut args = Vec::with_capacity(cap);
                    args.push("--update".to_string());
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
                    let mut args = Vec::with_capacity(2 + inputs.len());
                    args.push("flake".to_string());
                    args.push("update".to_string());
                    args.extend(inputs.iter().cloned());
                    ("nix".to_string(), args)
                }
                FlakeOperation::Lock { inputs } => {
                    let mut args = Vec::with_capacity(2 + 2 * inputs.len());
                    args.push("flake".to_string());
                    args.push("lock".to_string());
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
                let cap = if flake.is_some() { 3 } else { 1 };
                let mut args = Vec::with_capacity(cap);
                args.push("switch".to_string());
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
                let cap = 1
                    + if older_than_days.is_some() { 2 } else { 0 }
                    + if *delete_all { 1 } else { 0 };
                let mut args = Vec::with_capacity(cap);
                args.push("-d".to_string());
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

/// Injectable boundary for external process invocation.
///
/// Production uses Tokio's process runner. Tests can provide a mock to prove
/// denied authorization decisions never invoke the process boundary.
pub trait NixOSCommandRunner: Send + Sync {
    fn run<'a>(
        &'a self,
        executable: &'a str,
        args: &'a [String],
    ) -> Pin<Box<dyn Future<Output = io::Result<Output>> + Send + 'a>>;
}

struct TokioNixOSCommandRunner;

impl NixOSCommandRunner for TokioNixOSCommandRunner {
    fn run<'a>(
        &'a self,
        executable: &'a str,
        args: &'a [String],
    ) -> Pin<Box<dyn Future<Output = io::Result<Output>> + Send + 'a>> {
        Box::pin(async move {
            Command::new(executable)
                .args(args)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output()
                .await
        })
    }
}

/// NixOS-aware command executor with Φ integration
pub struct NixOSExecutor {
    current_generation: Option<u32>,
    thresholds: ConsciousnessThresholds,
    history: VecDeque<ExecutionRecord>,
    dry_run: bool,
    runner: Arc<dyn NixOSCommandRunner>,
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
            history: VecDeque::with_capacity(1000),
            dry_run: false,
            runner: Arc::new(TokioNixOSCommandRunner),
        }
    }

    /// Replace the process runner. Primarily useful for deterministic tests.
    pub fn with_runner(mut self, runner: Arc<dyn NixOSCommandRunner>) -> Self {
        self.runner = runner;
        self
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
        // Route even observational subprocesses through the injectable runner
        // so tests can qualify the complete executor process boundary.
        let args = vec!["list-generations".to_owned()];
        let output = self.runner.run("nixos-rebuild", &args).await?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow::anyhow!(
                "nixos-rebuild list-generations exited unsuccessfully: {}",
                stderr.trim()
            ));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);

        for line in stdout.lines() {
            if line.contains("(current)")
                && let Some(gen_str) = line.split_whitespace().next()
                && let Ok(r#gen) = gen_str.trim().parse::<u32>()
            {
                self.current_generation = Some(r#gen);
                info!(generation = r#gen, "Captured current NixOS generation");
                return Ok(r#gen);
            }
        }

        Err(anyhow::anyhow!("Could not determine current generation"))
    }

    /// Execute a NixOS command, gated on `phi` clearing the command's safety
    /// tier threshold (see `SafetyLevel::required_phi`).
    ///
    /// IMPORTANT: `phi` is NOT measured by this function or anywhere else in
    /// this crate — it is whatever the caller passes in. This is a real,
    /// useful tier-based gate (ReadOnly < UserModify < SystemModify <
    /// SystemCritical < Destructive), but it is only as honest as its caller:
    /// a caller that hardcodes a constant here (e.g. `execute(cmd, 0.7)`) has
    /// built a rubber stamp, not a safety check. Callers MUST derive `phi`
    /// from a real confirmation source (an explicit human approval, a
    /// deliberately conservative CLI default, etc.) — never a fixed literal
    /// chosen to clear every tier. If the confirmation already happened
    /// upstream (e.g. a human approved this exact command via a TUI prompt),
    /// call `execute_confirmed` instead, which says so honestly rather than
    /// re-checking a synthetic score. See
    /// SYMTHAEA_NIXOS_MANAGEMENT_IMPROVEMENT_PLAN_2026-07-26.md Phase 1.
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

        let result = self.runner.run(&cmd, &args).await;

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
                    let rb_result = self.runner.run(&rb_cmd, &rb_args).await;

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

    /// Execute a typed NixOS command only after consuming a matching one-shot permit.
    ///
    /// This is the permit-checked path for typed built-in commands. The action
    /// scope is derived from this command's safety tier; callers cannot supply a
    /// weaker scope. Custom commands are rejected here until a separate
    /// command-classification policy can validate them independently.
    ///
    /// IMPORTANT: this method does not remove the legacy `execute_confirmed`
    /// escape hatch. Until all real dispatch call sites are migrated or the
    /// escape hatch is narrowed, the repository is not globally fail-closed.
    pub async fn execute_with_permit(
        &mut self,
        command: NixOSCommand,
        phi: f32,
        ledger: &mut ExecutionAuthorizationLedger,
        permit: &ExecutionPermit,
        principal: &str,
        authority: AuthorizationAuthority,
    ) -> ExecutionResult {
        let safety_level = command.safety_level();

        if matches!(&command, NixOSCommand::Custom { .. }) {
            let _ = ledger.burn(permit);
            return ExecutionResult::Blocked {
                reason: "custom command requires a separately audited authorization policy"
                    .to_string(),
                safety_level,
            };
        }

        let scope = match safety_level {
            SafetyLevel::ReadOnly => ActionScope::ReadOnly,
            SafetyLevel::UserModify => ActionScope::UserModify,
            SafetyLevel::SystemModify => ActionScope::SystemModify,
            SafetyLevel::SystemCritical => ActionScope::SystemCritical,
            SafetyLevel::Destructive => ActionScope::Destructive,
        };
        let (executable, args) = command.to_command();

        let authorization = CommandBinding::new(executable, args)
            .and_then(|binding| {
                AuthorizationAttempt::new(binding, principal.to_owned(), scope, authority)
            })
            .and_then(|attempt| ledger.consume(permit, &attempt));

        if let Err(error) = authorization {
            // consume() already burns a permit on ledger-level denials; this
            // also burns it when binding/attempt construction failed earlier.
            let _ = ledger.burn(permit);
            return ExecutionResult::Blocked {
                reason: format!("one-shot authorization denied before dispatch: {error}"),
                safety_level,
            };
        }

        // No await or command transformation occurs between capability
        // consumption and entering the dispatch routine. The immutable command
        // value used to verify argv is the same value handed to the runner.
        self.execute_confirmed(command, phi).await
    }

    /// Execute unconditionally, bypassing the tier-threshold check in
    /// `execute()`. `phi` is recorded for telemetry only (`ExecutionRecord`),
    /// not checked. Only call this when the command was already confirmed by
    /// a real gate elsewhere (e.g. an explicit human approval) — this
    /// function performs no safety check of its own.
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

        // Keep the confirmed path on the same injectable process boundary as
        // ordinary execution. Permit-path tests must never fall through to a
        // real OS process, and every dispatch seam must be observable by the
        // runner used for enforcement qualification.
        let result = self.runner.run(&cmd, &args).await;

        let elapsed = start.elapsed().as_millis() as u64;

        let exec_result = match result {
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
        };

        // Confirmed live execution is still execution evidence, even though
        // the caller supplied the upstream authorization. Keep dry-run out of
        // this history until its outcome has a distinct non-execution type.
        self.record_execution(&command, phi, &exec_result);
        exec_result
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
        self.history.push_back(record);

        if self.history.len() > 1000 {
            self.history.pop_front();
        }
    }

    pub fn history(&self) -> &VecDeque<ExecutionRecord> {
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
#[cfg(unix)]
    #[derive(Clone)]
    struct CountingRunner {
        invocations: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    #[cfg(unix)]
    impl NixOSCommandRunner for CountingRunner {
        fn run<'a>(
            &'a self,
            _executable: &'a str,
            _args: &'a [String],
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<
                        Output = std::io::Result<std::process::Output>,
                    > + Send
                    + 'a,
            >,
        > {
            self.invocations
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Box::pin(async {
                use std::os::unix::process::ExitStatusExt;
                Ok(std::process::Output {
                    status: std::process::ExitStatus::from_raw(0),
                    stdout: Vec::new(),
                    stderr: Vec::new(),
                })
            })
        }
    }


    /// Test-only effect adapter: records the exact invocation and creates
    /// an independent filesystem witness when dispatch crosses this injected
    /// runner boundary. This never launches a real NixOS command.
    #[cfg(unix)]
    struct EffectWitnessRunner {
        invocations: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        observed_calls: std::sync::Arc<std::sync::Mutex<Vec<(String, Vec<String>)>>>,
        expected_executable: String,
        expected_args: Vec<String>,
        witness_path: std::path::PathBuf,
    }

    #[cfg(unix)]
    impl NixOSCommandRunner for EffectWitnessRunner {
        fn run<'a>(
            &'a self,
            executable: &'a str,
            args: &'a [String],
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<
                        Output = std::io::Result<std::process::Output>,
                    > + Send
                    + 'a,
            >,
        > {
            self.invocations
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.observed_calls
                .lock()
                .expect("effect witness call log lock")
                .push((executable.to_owned(), args.to_vec()));
            let expected_executable = self.expected_executable.clone();
            let expected_args = self.expected_args.clone();
            let observed_executable = executable.to_owned();
            let observed_args = args.to_vec();
            let witness_path = self.witness_path.clone();
            Box::pin(async move {
                // Harness-sensitivity control: do not create a witness unless
                // the adapter receives the exact executable and ordered argv.
                if observed_executable != expected_executable || observed_args != expected_args {
                    return Err(std::io::Error::other(
                        "effect witness adapter received unexpected executable/argv",
                    ));
                }
                // Spawn a harmless child-process fixture and pass the path as
                // an argv element, not shell-interpolated command text. It
                // produces a real temporary-file witness but never launches
                // nixos-rebuild or mutates host configuration.
                let output = std::process::Command::new("sh")
                    .arg("-c")
                    .arg("printf '%s' 'authorized-effect-witness' > \"$1\"")
                    .arg("effect-witness")
                    .arg(&witness_path)
                    .output()?;
                if !output.status.success() {
                    return Err(std::io::Error::other(
                        "temporary child-process witness command failed",
                    ));
                }
                Ok(output)
            })
        }
    }

    #[cfg(unix)]
    struct StaticOutputRunner {
        invocations: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        success: bool,
        stdout: Vec<u8>,
        stderr: Vec<u8>,
    }

    #[cfg(unix)]
    impl NixOSCommandRunner for StaticOutputRunner {
        fn run<'a>(
            &'a self,
            executable: &'a str,
            args: &'a [String],
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<
                        Output = std::io::Result<std::process::Output>,
                    > + Send
                    + 'a,
            >,
        > {
            assert_eq!(executable, "nixos-rebuild");
            assert_eq!(args, &["list-generations".to_owned()]);
            self.invocations
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let success = self.success;
            let stdout = self.stdout.clone();
            let stderr = self.stderr.clone();
            Box::pin(async move {
                use std::os::unix::process::ExitStatusExt;
                let status = if success {
                    std::process::ExitStatus::from_raw(0)
                } else {
                    std::process::ExitStatus::from_raw(1 << 8)
                };
                Ok(std::process::Output {
                    status,
                    stdout,
                    stderr,
                })
            })
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn capture_generation_uses_runner_and_rejects_failed_command_output() {
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };

        let valid_invocations = Arc::new(AtomicUsize::new(0));
        let valid_runner = Arc::new(StaticOutputRunner {
            invocations: Arc::clone(&valid_invocations),
            success: true,
            stdout: b"42 2026-10-24 12:00:00 (current) NixOS\n".to_vec(),
            stderr: Vec::new(),
        });
        let mut valid_executor = NixOSExecutor::new().with_runner(valid_runner);
        assert_eq!(valid_executor.capture_generation().await.unwrap(), 42);
        assert_eq!(valid_executor.current_generation, Some(42));
        assert_eq!(valid_invocations.load(Ordering::SeqCst), 1);

        // A failed subprocess must not be accepted as fresh pre-state even
        // if stdout happens to contain a parseable '(current)' row.
        let failed_invocations = Arc::new(AtomicUsize::new(0));
        let failed_runner = Arc::new(StaticOutputRunner {
            invocations: Arc::clone(&failed_invocations),
            success: false,
            stdout: b"99 2026-10-24 12:00:00 (current) NixOS\n".to_vec(),
            stderr: b"failed to inspect generations".to_vec(),
        });
        let mut failed_executor = NixOSExecutor::new().with_runner(failed_runner);
        let error = failed_executor.capture_generation().await.unwrap_err();
        assert!(error.to_string().contains("exited unsuccessfully"));
        assert_eq!(failed_executor.current_generation, None);
        assert_eq!(failed_invocations.load(Ordering::SeqCst), 1);
    }

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

    #[cfg(unix)]
    #[tokio::test]
    async fn test_execute_with_permit_accepts_exact_typed_command_once() {
        use super::super::authorization::{
            ActionScope, AuthorizationAuthority, AuthorizationRequest, CommandBinding,
            ExecutionAuthorizationLedger,
        };
        use std::time::Duration;

        let command = NixOSCommand::RebuildSwitch {
            flake: Some(".#host".to_string()),
            extra_args: vec![],
        };
        let (executable, args) = command.to_command();
        let binding = CommandBinding::new(executable, args).unwrap();
        let request = AuthorizationRequest::new(
            binding,
            "operator:alice",
            ActionScope::SystemCritical,
            AuthorizationAuthority::HumanOperator,
            "exec-test-001",
            Duration::from_secs(5),
        )
        .unwrap();
        let mut ledger = ExecutionAuthorizationLedger::default();
        let permit = ledger.issue(request).unwrap();
        let invocations = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let runner = std::sync::Arc::new(CountingRunner {
            invocations: std::sync::Arc::clone(&invocations),
        });
        let mut executor = NixOSExecutor::new().with_runner(runner);

        let result = executor
            .execute_with_permit(
                command.clone(),
                0.0,
                &mut ledger,
                &permit,
                "operator:alice",
                AuthorizationAuthority::HumanOperator,
            )
            .await;
        assert!(matches!(result, ExecutionResult::Success { .. }));
        assert_eq!(invocations.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(executor.history().len(), 1);
        assert!(matches!(
            executor.history().back().map(|entry| &entry.result),
            Some(ExecutionResult::Success { .. })
        ));

        let replay = executor
            .execute_with_permit(
                command,
                0.0,
                &mut ledger,
                &permit,
                "operator:alice",
                AuthorizationAuthority::HumanOperator,
            )
            .await;
        assert!(matches!(
            replay,
            ExecutionResult::Blocked { ref reason, .. } if reason.contains("already consumed")
        ));
        assert_eq!(invocations.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(executor.history().len(), 1);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_execute_with_permit_blocks_command_substitution() {
        use super::super::authorization::{
            ActionScope, AuthorizationAuthority, AuthorizationRequest, CommandBinding,
            ExecutionAuthorizationLedger,
        };
        use std::time::Duration;

        let approved = NixOSCommand::RebuildSwitch {
            flake: Some(".#approved".to_string()),
            extra_args: vec![],
        };
        let (executable, args) = approved.to_command();
        let request = AuthorizationRequest::new(
            CommandBinding::new(executable, args).unwrap(),
            "operator:alice",
            ActionScope::SystemCritical,
            AuthorizationAuthority::HumanOperator,
            "exec-test-002",
            Duration::from_secs(5),
        )
        .unwrap();
        let mut ledger = ExecutionAuthorizationLedger::default();
        let permit = ledger.issue(request).unwrap();
        let invocations = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let runner = std::sync::Arc::new(CountingRunner {
            invocations: std::sync::Arc::clone(&invocations),
        });
        let mut executor = NixOSExecutor::new().with_runner(runner);
        let substituted = NixOSCommand::RebuildSwitch {
            flake: Some(".#attacker-controlled".to_string()),
            extra_args: vec![],
        };

        let result = executor
            .execute_with_permit(
                substituted,
                1.0,
                &mut ledger,
                &permit,
                "operator:alice",
                AuthorizationAuthority::HumanOperator,
            )
            .await;
        assert!(matches!(
            result,
            ExecutionResult::Blocked { ref reason, .. } if reason.contains("differ from the permit")
        ));
        assert_eq!(invocations.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_execute_with_permit_rejects_unclassified_custom_command() {
        use super::super::authorization::{
            ActionScope, AuthorizationAuthority, AuthorizationRequest, CommandBinding,
            ExecutionAuthorizationLedger,
        };
        use std::time::Duration;

        let command = NixOSCommand::Custom {
            command: "sh".to_string(),
            args: vec!["-c".to_string(), "touch /tmp/should-not-run".to_string()],
            safety_level: SafetyLevel::ReadOnly,
        };
        let (executable, args) = command.to_command();
        let request = AuthorizationRequest::new(
            CommandBinding::new(executable, args).unwrap(),
            "operator:alice",
            ActionScope::ReadOnly,
            AuthorizationAuthority::ReadOnlyPolicy,
            "exec-test-003",
            Duration::from_secs(5),
        )
        .unwrap();
        let mut ledger = ExecutionAuthorizationLedger::default();
        let permit = ledger.issue(request).unwrap();
        let invocations = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let runner = std::sync::Arc::new(CountingRunner {
            invocations: std::sync::Arc::clone(&invocations),
        });
        let mut executor = NixOSExecutor::new().with_runner(runner);

        let result = executor
            .execute_with_permit(
                command,
                1.0,
                &mut ledger,
                &permit,
                "operator:alice",
                AuthorizationAuthority::ReadOnlyPolicy,
            )
            .await;
        assert!(matches!(
            result,
            ExecutionResult::Blocked { ref reason, .. } if reason.contains("separately audited")
        ));
        assert_eq!(invocations.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_malformed_identity_attempt_burns_the_permit() {
        use super::super::authorization::{
            ActionScope, AuthorizationAuthority, AuthorizationRequest, CommandBinding,
            ExecutionAuthorizationLedger,
        };
        use std::time::Duration;

        let command = NixOSCommand::RebuildSwitch {
            flake: Some(".#host".to_string()),
            extra_args: vec![],
        };
        let (executable, args) = command.to_command();
        let request = AuthorizationRequest::new(
            CommandBinding::new(executable, args).unwrap(),
            "operator:alice",
            ActionScope::SystemCritical,
            AuthorizationAuthority::HumanOperator,
            "exec-test-004",
            Duration::from_secs(5),
        )
        .unwrap();
        let mut ledger = ExecutionAuthorizationLedger::default();
        let permit = ledger.issue(request).unwrap();
        let invocations = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let runner = std::sync::Arc::new(CountingRunner {
            invocations: std::sync::Arc::clone(&invocations),
        });
        let mut executor = NixOSExecutor::new().with_runner(runner);

        let malformed = executor
            .execute_with_permit(
                command.clone(),
                1.0,
                &mut ledger,
                &permit,
                "",
                AuthorizationAuthority::HumanOperator,
            )
            .await;
        assert!(matches!(
            malformed,
            ExecutionResult::Blocked { ref reason, .. } if reason.contains("principal")
        ));

        let replay = executor
            .execute_with_permit(
                command,
                1.0,
                &mut ledger,
                &permit,
                "operator:alice",
                AuthorizationAuthority::HumanOperator,
            )
            .await;
        assert!(matches!(
            replay,
            ExecutionResult::Blocked { ref reason, .. } if reason.contains("already consumed")
        ));
        assert_eq!(invocations.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_permit_gate_controls_independent_effect_witness() {
        use super::super::authorization::{
            ActionScope, AuthorizationAuthority, AuthorizationRequest, CommandBinding,
            ExecutionAuthorizationLedger,
        };
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
            Mutex,
        };
        use std::time::Duration;

        let temp = tempfile::tempdir().unwrap();
        let denied_witness = temp.path().join("denied-effect.witness");
        let allowed_witness = temp.path().join("allowed-effect.witness");

        let approved = NixOSCommand::RebuildSwitch {
            flake: Some(".#approved".to_owned()),
            extra_args: Vec::new(),
        };
        let (executable, args) = approved.to_command();
        let request = AuthorizationRequest::new(
            CommandBinding::new(executable, args).unwrap(),
            "operator:alice",
            ActionScope::SystemCritical,
            AuthorizationAuthority::HumanOperator,
            "effect-witness-denied-001",
            Duration::from_secs(5),
        )
        .unwrap();
        let mut denied_ledger = ExecutionAuthorizationLedger::default();
        let denied_permit = denied_ledger.issue(request).unwrap();
        let denied_invocations = Arc::new(AtomicUsize::new(0));
        let denied_calls = Arc::new(Mutex::new(Vec::new()));
        let denied_runner = Arc::new(EffectWitnessRunner {
            invocations: Arc::clone(&denied_invocations),
            observed_calls: Arc::clone(&denied_calls),
            expected_executable: "nixos-rebuild".to_owned(),
            expected_args: vec!["switch".to_owned(), "--flake".to_owned(), ".#approved".to_owned()],
            witness_path: denied_witness.clone(),
        });
        let mut denied_executor = NixOSExecutor::new().with_runner(denied_runner);

        // Mutate the exact flake after issuance. The permit must fail before
        // the runner boundary, leaving no call record and no witness file.
        let substituted = NixOSCommand::RebuildSwitch {
            flake: Some(".#attacker-controlled".to_owned()),
            extra_args: Vec::new(),
        };
        let denied = denied_executor
            .execute_with_permit(
                substituted,
                1.0,
                &mut denied_ledger,
                &denied_permit,
                "operator:alice",
                AuthorizationAuthority::HumanOperator,
            )
            .await;
        assert!(matches!(denied, ExecutionResult::Blocked { .. }));
        assert_eq!(denied_invocations.load(Ordering::SeqCst), 0);
        assert!(denied_calls.lock().unwrap().is_empty());
        assert!(!denied_witness.exists());

        // A fresh permit for the exact command crosses the runner exactly
        // once and creates an independently observable test artifact.
        let (executable, args) = approved.to_command();
        let request = AuthorizationRequest::new(
            CommandBinding::new(executable, args).unwrap(),
            "operator:alice",
            ActionScope::SystemCritical,
            AuthorizationAuthority::HumanOperator,
            "effect-witness-allowed-001",
            Duration::from_secs(5),
        )
        .unwrap();
        let mut allowed_ledger = ExecutionAuthorizationLedger::default();
        let allowed_permit = allowed_ledger.issue(request).unwrap();
        let allowed_invocations = Arc::new(AtomicUsize::new(0));
        let allowed_calls = Arc::new(Mutex::new(Vec::new()));
        let allowed_runner = Arc::new(EffectWitnessRunner {
            invocations: Arc::clone(&allowed_invocations),
            observed_calls: Arc::clone(&allowed_calls),
            expected_executable: "nixos-rebuild".to_owned(),
            expected_args: vec!["switch".to_owned(), "--flake".to_owned(), ".#approved".to_owned()],
            witness_path: allowed_witness.clone(),
        });
        let mut allowed_executor = NixOSExecutor::new().with_runner(allowed_runner);

        let first = allowed_executor
            .execute_with_permit(
                approved.clone(),
                0.0,
                &mut allowed_ledger,
                &allowed_permit,
                "operator:alice",
                AuthorizationAuthority::HumanOperator,
            )
            .await;
        assert!(matches!(first, ExecutionResult::Success { .. }));
        assert_eq!(allowed_invocations.load(Ordering::SeqCst), 1);
        assert_eq!(
            std::fs::read(&allowed_witness).unwrap(),
            b"authorized-effect-witness"
        );
        assert_eq!(
            *allowed_calls.lock().unwrap(),
            vec![(
                "nixos-rebuild".to_owned(),
                vec![
                    "switch".to_owned(),
                    "--flake".to_owned(),
                    ".#approved".to_owned(),
                ],
            )]
        );

        // Replaying the consumed permit must not create or overwrite another
        // effect witness; the original witness and single invocation remain.
        let replay = allowed_executor
            .execute_with_permit(
                approved,
                0.0,
                &mut allowed_ledger,
                &allowed_permit,
                "operator:alice",
                AuthorizationAuthority::HumanOperator,
            )
            .await;
        assert!(matches!(replay, ExecutionResult::Blocked { .. }));
        assert_eq!(allowed_invocations.load(Ordering::SeqCst), 1);
        assert_eq!(allowed_calls.lock().unwrap().len(), 1);
        assert_eq!(
            std::fs::read(&allowed_witness).unwrap(),
            b"authorized-effect-witness"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn test_effect_witness_fails_closed_on_unexpected_adapter_argv() {
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
            Mutex,
        };

        let temp = tempfile::tempdir().unwrap();
        let witness_path = temp.path().join("unexpected-argv.witness");
        let invocations = Arc::new(AtomicUsize::new(0));
        let observed_calls = Arc::new(Mutex::new(Vec::new()));
        let runner = EffectWitnessRunner {
            invocations: Arc::clone(&invocations),
            observed_calls: Arc::clone(&observed_calls),
            expected_executable: "nixos-rebuild".to_owned(),
            expected_args: vec!["switch".to_owned(), "--flake".to_owned(), ".#expected".to_owned()],
            witness_path: witness_path.clone(),
        };
        let unexpected_args = vec!["switch".to_owned(), "--flake".to_owned(), ".#unexpected".to_owned()];

        let result = runner.run("nixos-rebuild", &unexpected_args).await;
        assert!(result.is_err());
        assert_eq!(invocations.load(Ordering::SeqCst), 1);
        assert_eq!(observed_calls.lock().unwrap().len(), 1);
        assert!(!witness_path.exists());
    }

    #[tokio::test]
    async fn test_execute_confirmed_bypasses_phi_check() {
        // execute_confirmed must run a command that execute() would gate,
        // documenting that this path is reserved for already-approved
        // actions rather than performing its own safety check.
        let mut executor = NixOSExecutor::new().with_dry_run(true);

        let rebuild = NixOSCommand::RebuildSwitch {
            flake: None,
            extra_args: vec![],
        };
        // phi=0.0 is below every tier's threshold, including Destructive's
        // 0.6 — execute() would refuse this; execute_confirmed must not.
        let result = executor.execute_confirmed(rebuild, 0.0).await;

        match result {
            ExecutionResult::Success { stdout, .. } => {
                assert!(stdout.contains("[DRY-RUN]"));
            }
            other => panic!("Expected dry-run success, got {:?}", other),
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
    fn test_custom_auto_classify_search() {
        let cmd =
            NixOSCommand::custom_auto("nix", vec!["search".into(), "nixpkgs".into(), "vim".into()]);
        assert_eq!(cmd.safety_level(), SafetyLevel::ReadOnly);
    }

    #[test]
    fn test_custom_auto_classify_rebuild() {
        let cmd = NixOSCommand::custom_auto("nixos-rebuild", vec!["switch".into()]);
        assert_eq!(cmd.safety_level(), SafetyLevel::SystemCritical);
    }

    #[test]
    fn test_custom_auto_classify_gc() {
        let cmd = NixOSCommand::custom_auto("nix-collect-garbage", vec!["-d".into()]);
        assert_eq!(cmd.safety_level(), SafetyLevel::Destructive);
    }

    #[test]
    fn test_custom_auto_classify_unknown() {
        // Unknown commands default to SystemCritical (conservative)
        let cmd = NixOSCommand::custom_auto("some-unknown-tool", vec![]);
        assert_eq!(cmd.safety_level(), SafetyLevel::SystemCritical);
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
    fn test_history_ring_buffer_eviction() {
        let mut executor = NixOSExecutor::new();
        // Fill beyond 1000 entries
        for i in 0..1005 {
            let record = ExecutionRecord {
                command: NixOSCommand::Search {
                    query: format!("pkg{i}"),
                    json: false,
                },
                phi_at_execution: 0.5,
                result: ExecutionResult::Success {
                    stdout: String::new(),
                    stderr: String::new(),
                    execution_time_ms: 0,
                },
                timestamp_ms: i as u64,
            };
            executor.history.push_back(record);
            if executor.history.len() > 1000 {
                executor.history.pop_front();
            }
        }
        assert_eq!(executor.history.len(), 1000);
        // Oldest entry should be pkg5 (0..4 evicted)
        if let NixOSCommand::Search { query, .. } = &executor.history[0].command {
            assert_eq!(query, "pkg5");
        } else {
            panic!("Expected Search command");
        }
    }

    #[test]
    fn test_to_command_capacity_hints() {
        // Ensure pre-allocated capacity is sufficient (no reallocation panics)
        let cmd = NixOSCommand::RebuildSwitch {
            flake: Some(".#host".into()),
            extra_args: vec!["--show-trace".into(), "--verbose".into()],
        };
        let (_, args) = cmd.to_command();
        assert_eq!(args.len(), 5); // switch, --flake, .#host, --show-trace, --verbose

        let cmd = NixOSCommand::CollectGarbage {
            older_than_days: Some(30),
            delete_all: true,
        };
        let (_, args) = cmd.to_command();
        assert_eq!(args.len(), 4); // -d, --delete-older-than, 30d, --delete-old
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
