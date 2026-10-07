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

use crate::action::authorization::NixLocalExecutionAuthorityV1;
use crate::action::execution_witness::NixLiveExecutionWitnessV1;
#[cfg(feature = "systemd-observer")]
use crate::action::NixSystemdReadOnlyObserverV1;
use crate::action::service_domain::{NixServiceOperationKindV1, NixServiceOperationV1};
use crate::action::service_manager::ServiceManager;
use crate::action::service_state::NixServiceObservedStateV1;
use crate::traits::{ActionType, ConsciousnessThresholds, PhiAwareScoring};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::process::Stdio;
use tokio::process::Command;
use tracing::{info, warn};

#[derive(Debug, Deserialize)]
struct NixOSGenerationRecordV1 {
    generation: u32,
    current: bool,
}

const SERVICE_PRE_STATE_IDENTITY_PREFIX_V1: &str = "nixward-service-pre-state-v1";

fn parse_generation_pre_state_identity(identity: &str) -> Result<u32, String> {
    const PREFIX: &str = "generation:";
    let raw = identity
        .strip_prefix(PREFIX)
        .ok_or_else(|| format!("unsupported pre-state identity format: {identity}"))?;
    if raw.is_empty() {
        return Err("generation pre-state identity has no generation number".to_string());
    }
    raw.parse::<u32>()
        .map_err(|_| format!("invalid generation pre-state identity: {identity}"))
}

fn parse_service_pre_state_identity(
    identity: &str,
) -> Result<(u64, String, String), String> {
    let mut parts = identity.split('|');
    if parts.next() != Some(SERVICE_PRE_STATE_IDENTITY_PREFIX_V1)
        || parts.clone().count() != 3
    {
        return Err(format!("unsupported service pre-state identity format: {identity}"));
    }

    let generation = parts
        .next()
        .and_then(|part| part.strip_prefix("generation="))
        .ok_or_else(|| format!("service pre-state identity missing generation: {identity}"))?;
    if generation == "none" {
        return Err(
            "service execution authority requires a bound NixOS generation".to_string(),
        );
    }
    let generation = generation
        .parse::<u64>()
        .map_err(|_| format!("invalid service pre-state generation: {identity}"))?;

    let unit = parts
        .next()
        .and_then(|part| part.strip_prefix("unit="))
        .ok_or_else(|| format!("service pre-state identity missing unit: {identity}"))?
        .to_string();
    let typed = NixServiceOperationV1::new(&unit, NixServiceOperationKindV1::Start)
        .map_err(|error| format!("invalid service pre-state unit: {error}"))?;
    if typed.unit() != unit {
        return Err(format!(
            "service pre-state identity unit is not canonical: {unit}"
        ));
    }

    let digest = parts
        .next()
        .and_then(|part| part.strip_prefix("state="))
        .ok_or_else(|| format!("service pre-state identity missing state digest: {identity}"))?;
    if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!(
            "service pre-state identity has invalid state digest: {identity}"
        ));
    }

    Ok((generation, unit, digest.to_string()))
}


fn validate_service_pre_state_observation(
    identity: &str,
    command_unit: &str,
    actual_generation: u64,
    observed: &NixServiceObservedStateV1,
) -> Result<(), String> {
    let (expected_generation, expected_unit, expected_state_digest) =
        parse_service_pre_state_identity(identity)?;

    if expected_generation != actual_generation {
        return Err(format!(
            "execution authority is stale: approved generation={} but current generation={}",
            expected_generation, actual_generation
        ));
    }
    if expected_unit != command_unit {
        return Err(format!(
            "service execution authority unit mismatch: approved={} command={}",
            expected_unit, command_unit
        ));
    }
    if observed.unit() != command_unit {
        return Err(format!(
            "service observation canonical identity mismatch: requested={} observed={}",
            command_unit,
            observed.unit()
        ));
    }

    let actual_state_digest = observed
        .digest()
        .map_err(|error| format!("could not digest current service pre-state: {error}"))?;
    if actual_state_digest != expected_state_digest {
        return Err(format!(
            "service execution authority is stale: approved pre-state digest={} but current digest={}",
            expected_state_digest, actual_state_digest
        ));
    }
    Ok(())
}

fn parse_current_generation(stdout: &str) -> Result<u32, String> {
    let records: Vec<NixOSGenerationRecordV1> = serde_json::from_str(stdout)
        .map_err(|error| format!("invalid nixos-rebuild generation JSON: {error}"))?;
    let mut current = records.iter().filter(|record| record.current);
    let record = current.next().ok_or_else(|| {
        "nixos-rebuild generation JSON contained no current generation".to_string()
    })?;
    if current.next().is_some() {
        return Err(
            "nixos-rebuild generation JSON contained multiple current generations".to_string(),
        );
    }
    Ok(record.generation)
}

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
    /// nix-env --switch-generation for the system profile.
    EnvSwitchGeneration { generation: u32 },
    /// nix-env --delete-generations +N for the system profile.
    EnvDeleteGenerations { keep_last: usize },
    /// nix-env --delete-generations Nd for the system profile.
    EnvDeleteGenerationsOlderThan { days: u32 },
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
    /// Exact typed systemd service lifecycle operation.
    Service {
        operation: NixServiceOperationKindV1,
        unit: String,
    },
    /// Exact NixOS configuration option mutation, followed by a fixed switch.
    ///
    /// The expected configuration digest binds the authorization to the file state
    /// that was observed when the mutation was proposed.
    ConfigPatch {
        option_path: String,
        value: String,
        expected_config_digest: String,
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
    Init { template: Option<String> },
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
    /// Create a legacy Custom command with auto-classified safety level.
    ///
    /// Custom commands are retained only for compatibility and dry-run
    /// presentation. Non-dry-run execution fails closed because the arbitrary
    /// executable/argv pair cannot establish a bounded effect identity.
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

    /// Validate command-specific invariants that must hold even after
    /// confirmation. Typed service commands must remain canonical at the
    /// final execution boundary.
    pub(crate) fn validate_shape(&self) -> Result<(), String> {
        match self {
            Self::Service { operation, unit } => {
                let typed = NixServiceOperationV1::new(unit.clone(), *operation)
                    .map_err(|error| error.to_string())?;
                if typed.unit() != unit {
                    return Err(
                        "typed service unit must be canonical at the command boundary".to_string(),
                    );
                }
                Ok(())
            }
            Self::ConfigPatch {
                option_path,
                value,
                expected_config_digest,
            } => {
                if option_path.trim().is_empty() {
                    return Err("config patch option path must not be blank".to_string());
                }
                if value.trim().is_empty() {
                    return Err("config patch value must not be blank".to_string());
                }
                if expected_config_digest.len() != 64
                    || !expected_config_digest
                        .bytes()
                        .all(|b| b.is_ascii_hexdigit())
                {
                    return Err(
                        "config patch expected config digest must be 64 hex characters".to_string(),
                    );
                }
                Ok(())
            }

            _ => Ok(()),
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
            Self::Flake {
                operation: FlakeOperation::Init { .. },
            } => SafetyLevel::UserModify,
            Self::HomeManagerSwitch { .. } => SafetyLevel::UserModify,

            Self::RebuildTest { .. } => SafetyLevel::SystemModify,
            Self::RebuildBoot { .. } => SafetyLevel::SystemModify,

            Self::RebuildSwitch { .. } => SafetyLevel::SystemCritical,

            Self::CollectGarbage { .. } => SafetyLevel::Destructive,

            Self::Service { .. } => SafetyLevel::SystemModify,

            Self::ConfigPatch { .. } => SafetyLevel::SystemCritical,

            Self::Custom { safety_level, .. } => *safety_level,
        }
    }

    /// Get the rollback command if available
    pub fn rollback_command(&self) -> Option<NixOSCommand> {
        match self {
            Self::RebuildSwitch { .. } | Self::RebuildTest { .. } | Self::RebuildBoot { .. } => {
                Some(NixOSCommand::RebuildSwitch {
                    flake: None,
                    extra_args: vec!["--rollback".to_string()],
                })
            }
            Self::EnvInstall { .. } | Self::EnvRemove { .. } => {
                Some(NixOSCommand::EnvRollback)
            }
            // The previous implementation used a shell pipeline to locate and
            // activate an older Home Manager generation. That was an arbitrary
            // shell effect and cannot cross a typed execution boundary safely.
            // Fail closed until a dedicated typed rollback operation exists.
            Self::HomeManagerSwitch { .. } => None,
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
            Self::EnvSwitchGeneration { generation } => (
                "nix-env".to_string(),
                vec![
                    "--switch-generation".to_string(),
                    generation.to_string(),
                    "-p".to_string(),
                    "/nix/var/nix/profiles/system".to_string(),
                ],
            ),
            Self::EnvDeleteGenerations { keep_last } => (
                "nix-env".to_string(),
                vec![
                    "--delete-generations".to_string(),
                    format!("+{keep_last}"),
                    "-p".to_string(),
                    "/nix/var/nix/profiles/system".to_string(),
                ],
            ),
            Self::EnvDeleteGenerationsOlderThan { days } => (
                "nix-env".to_string(),
                vec![
                    "--delete-generations".to_string(),
                    format!("{days}d"),
                    "-p".to_string(),
                    "/nix/var/nix/profiles/system".to_string(),
                ],
            ),
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
                FlakeOperation::Init { template } => {
                    let mut args = vec!["flake".to_string(), "init".to_string()];
                    if let Some(template) = template {
                        args.push("--template".to_string());
                        args.push(template.clone());
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
            Self::Service { operation, unit } => (
                "systemctl".to_string(),
                vec![
                    match operation {
                        NixServiceOperationKindV1::Start => "start",
                        NixServiceOperationKindV1::Stop => "stop",
                        NixServiceOperationKindV1::Restart => "restart",
                        NixServiceOperationKindV1::Reload => "reload",
                        NixServiceOperationKindV1::Enable => "enable",
                        NixServiceOperationKindV1::Disable => "disable",
                    }
                    .to_string(),
                    unit.clone(),
                ],
            ),
            Self::ConfigPatch { .. } => ("nixos-rebuild".to_string(), vec!["switch".to_string()]),
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

enum ExecutionBasisV1 {
    Phi {
        phi: f32,
    },
    LiveAuthority {
        intent_digest: String,
        approval_request_id: String,
        projection_digest: String,
    },
}

/// Free-form Custom commands are legacy compatibility data, not typed effect
/// semantics. They may be previewed, but a non-dry-run executor must not treat
/// an arbitrary executable + argv pair as sufficiently constrained to authorize
/// or dispatch an effect. This deliberately avoids an incomplete command-name
/// or shell-text blacklist: an arbitrary wrapper/helper could otherwise invoke
/// systemctl without containing "systemctl" at the top level.
///
/// Typed NixOSCommand variants are the only non-dry-run effect representation.
fn legacy_effect_requires_typed_authority(command: &NixOSCommand) -> bool {
    matches!(
        command,
        NixOSCommand::Service { .. } | NixOSCommand::Custom { .. }
    )
}

/// In-memory provenance for an execution that crossed the live Nixward
/// authority boundary. This is separate from ExecutionRecord, whose
/// phi_at_execution field is legacy telemetry.
#[derive(Debug, Clone)]
pub struct AuthorizedExecutionRecordV1 {
    command: NixOSCommand,
    action_intent_digest: String,
    approval_request_id: String,
    projection_digest: String,
    pre_state_identity: Option<String>,
    result: ExecutionResult,
    timestamp_ms: u64,
}

impl AuthorizedExecutionRecordV1 {
    pub fn command(&self) -> &NixOSCommand {
        &self.command
    }

    pub fn action_intent_digest(&self) -> &str {
        &self.action_intent_digest
    }

    pub fn approval_request_id(&self) -> &str {
        &self.approval_request_id
    }

    pub fn projection_digest(&self) -> &str {
        &self.projection_digest
    }

    pub fn pre_state_identity(&self) -> Option<&str> {
        self.pre_state_identity.as_deref()
    }

    pub fn result(&self) -> &ExecutionResult {
        &self.result
    }

    pub fn timestamp_ms(&self) -> u64 {
        self.timestamp_ms
    }
}

/// NixOS-aware command executor with Φ integration
pub struct NixOSExecutor {
    current_generation: Option<u32>,
    thresholds: ConsciousnessThresholds,
    history: VecDeque<ExecutionRecord>,
    authorized_history: VecDeque<AuthorizedExecutionRecordV1>,
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
            history: VecDeque::with_capacity(1000),
            authorized_history: VecDeque::with_capacity(1000),
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
            .args(["list-generations", "--json"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await?;

        if !output.status.success() {
            return Err(anyhow::anyhow!(
                "nixos-rebuild list-generations --json failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let generation = parse_current_generation(&stdout)
            .map_err(|error| anyhow::anyhow!("Could not determine current generation: {error}"))?;
        self.current_generation = Some(generation);
        info!(generation, "Captured current NixOS generation");
        Ok(generation)
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
        if let Err(reason) = command.validate_shape() {
            return ExecutionResult::Blocked {
                reason,
                safety_level: safety,
            };
        }
        let required_phi = safety.required_phi();

        if !self.dry_run && legacy_effect_requires_typed_authority(&command) {
            return ExecutionResult::Blocked {
                reason: "free-form Custom commands are not execution authority; use a typed Nixward command".to_string(),
                safety_level: safety,
            };
        }

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

    /// Execute only with a live Nixward execution-authority object.
    ///
    /// The authority object is consumed by value and validates that the command
    /// exactly matches the action intent whose local approval was consumed. Phi is
    /// deliberately not an input to this authority path.
    pub async fn execute_authorized(
        &mut self,
        command: NixOSCommand,
        authority: NixLocalExecutionAuthorityV1,
    ) -> ExecutionResult {
        let safety = command.safety_level();
        if let Err(reason) = command.validate_shape() {
            return ExecutionResult::Blocked {
                reason,
                safety_level: safety,
            };
        }
        if let Err(error) = authority.validate_command(&command) {
            return ExecutionResult::Blocked {
                reason: format!("execution authority rejected command: {error}"),
                safety_level: safety,
            };
        }
        if let Err(reason) = self
            .validate_authorized_pre_state_identity(&authority, &command)
            .await
        {
            return ExecutionResult::Blocked {
                reason,
                safety_level: safety,
            };
        }

        if let NixOSCommand::ConfigPatch {
            option_path,
            value,
            expected_config_digest,
        } = &command
        {
            if !self.dry_run {
                let writer = super::config_writer::ConfigWriter::new();
                let patch = match writer.set_option(option_path, value) {
                    Ok(patch) => patch,
                    Err(error) => {
                        return ExecutionResult::FailedNoRollback {
                            error: format!("config patch preparation failed: {error}"),
                            rollback_error: None,
                        };
                    }
                };
                if let Err(error) = writer.apply_patch_if_current(&patch, expected_config_digest) {
                    return ExecutionResult::FailedNoRollback {
                        error: format!("config patch currentness/write failed: {error}"),
                        rollback_error: None,
                    };
                }
            }
        }

        let intent_digest = authority
            .action_intent_digest()
            .unwrap_or_else(|_| "<invalid-intent>".to_string());
        let approval_request_id = authority.approval_request_id().to_string();
        let projection_digest = authority.projection_digest().to_string();
        let pre_state_identity = authority.pre_state_identity().map(str::to_owned);

        let result = self
            .execute_confirmed_inner(
                command.clone(),
                ExecutionBasisV1::LiveAuthority {
                    intent_digest: intent_digest.clone(),
                    approval_request_id: approval_request_id.clone(),
                    projection_digest: projection_digest.clone(),
                },
            )
            .await;
        self.record_authorized_execution(
            command,
            intent_digest,
            approval_request_id,
            projection_digest,
            pre_state_identity,
            &result,
        );
        result
    }
    /// Revalidate the state identity bound into live authority immediately before dispatch.
    ///
    /// Ordinary commands retain the existing NixOS-generation binding. Typed service
    /// commands additionally bind the exact observed service pre-state digest and must
    /// re-observe that same canonical unit immediately before dispatch.
    async fn validate_authorized_pre_state_identity(
        &mut self,
        authority: &NixLocalExecutionAuthorityV1,
        command: &NixOSCommand,
    ) -> Result<(), String> {
        let identity = authority
            .pre_state_identity()
            .ok_or_else(|| "execution authority has no bound pre-state identity".to_string())?;

        if let NixOSCommand::Service { unit, .. } = command {
            if self.dry_run {
                return Ok(());
            }

            let actual_generation = self
                .capture_generation()
                .await
                .map_err(|error| {
                    format!("could not revalidate current NixOS generation: {error}")
                })?;
            let observed = ServiceManager::observed_state(unit)
                .map_err(|error| format!("could not revalidate service pre-state: {error}"))?;
            validate_service_pre_state_observation(
                identity,
                unit,
                u64::from(actual_generation),
                &observed,
            )?;

            #[cfg(feature = "systemd-observer")]
            self.validate_authorized_service_definition_content(&authority, unit)
                .await?;

            #[cfg(not(feature = "systemd-observer"))]
            return Err(
                "typed Service execution requires the systemd read-only observer capability"
                    .to_string(),
            );

            #[cfg(feature = "systemd-observer")]
            return Ok(());
        }

        let expected_generation = parse_generation_pre_state_identity(identity)?;
        if self.dry_run {
            return Ok(());
        }

        let actual_generation = self
            .capture_generation()
            .await
            .map_err(|error| format!("could not revalidate current NixOS generation: {error}"))?;
        if actual_generation != expected_generation {
            return Err(format!(
                "execution authority is stale: approved generation={} but current generation={}",
                expected_generation, actual_generation
            ));
        }
        Ok(())
    }

    #[cfg(feature = "systemd-observer")]
    async fn validate_authorized_service_definition_content(
        &mut self,
        authority: &NixLocalExecutionAuthorityV1,
        unit: &str,
    ) -> Result<(), String> {
        if self.dry_run {
            return Ok(());
        }

        let expected_digest = authority
            .service_definition_content_digest()
            .ok_or_else(|| {
                "Service execution authority has no bound definition-content commitment"
                    .to_string()
            })?;

        let observer = NixSystemdReadOnlyObserverV1::connect_system()
            .await
            .map_err(|error| {
                format!(
                    "could not connect read-only systemd observer for definition revalidation: {error}"
                )
            })?;

        let content = observer
            .capture_service_definition_content(unit)
            .await
            .map_err(|error| {
                format!("could not revalidate service definition content: {error}")
            })?;
        let actual_digest = content
            .digest()
            .map_err(|error| {
                format!("could not digest revalidated service definition content: {error}")
            })?;

        if actual_digest != expected_digest {
            return Err(format!(
                "service definition content is stale or changed since approval: approved={} current={}",
                expected_digest, actual_digest
            ));
        }

        Ok(())
    }

    /// Execute unconditionally, bypassing the tier-threshold check in
    /// `execute()`. `phi` is recorded for telemetry only (`ExecutionRecord`),
    /// not checked. Only call this when the command was already confirmed by
    /// a real gate elsewhere (e.g. an explicit human approval) — this
    /// function performs no safety check of its own.
    pub async fn execute_confirmed(&mut self, command: NixOSCommand, phi: f32) -> ExecutionResult {
        let safety = command.safety_level();
        if let Err(reason) = command.validate_shape() {
            return ExecutionResult::Blocked {
                reason,
                safety_level: safety,
            };
        }
        if matches!(command, NixOSCommand::ConfigPatch { .. }) {
            return ExecutionResult::Blocked {
                reason: "ConfigPatch requires a live Nixward execution authority".to_string(),
                safety_level: safety,
            };
        }
        if !self.dry_run && legacy_effect_requires_typed_authority(&command) {
            return ExecutionResult::Blocked {
                reason: "free-form Custom commands and service effects require typed Nixward execution authority; Phi confirmation is not execution authority".to_string(),
                safety_level: safety,
            };
        }
        self.execute_confirmed_inner(command, ExecutionBasisV1::Phi { phi })
            .await
    }

    async fn execute_confirmed_inner(
        &mut self,
        command: NixOSCommand,
        basis: ExecutionBasisV1,
    ) -> ExecutionResult {
        let safety = command.safety_level();

        // Keep the invariant at the final shared dispatch helper too:
        // future Phi-based callers must not accidentally acquire service
        // execution merely by reaching this private function. Live authority
        // is the sole non-dry-run service execution basis.
        if !self.dry_run
            && matches!(&basis, ExecutionBasisV1::Phi { .. })
            && legacy_effect_requires_typed_authority(&command)
        {
            return ExecutionResult::Blocked {
                reason: "free-form Custom commands and service effects require typed Nixward execution authority; Phi confirmation is not execution authority".to_string(),
                safety_level: safety,
            };
        }

        let (cmd, args) = command.to_command();

        match &basis {
            ExecutionBasisV1::Phi { phi } => {
                info!(
                    command = %cmd,
                    args = ?args,
                    phi = %phi,
                    confirmed = true,
                    "Executing confirmed NixOS command"
                );
            }
            ExecutionBasisV1::LiveAuthority {
                intent_digest,
                approval_request_id,
                projection_digest,
            } => {
                info!(
                    command = %cmd,
                    args = ?args,
                    intent = %intent_digest,
                    approval_request = %approval_request_id,
                    projection = %projection_digest,
                    confirmed = true,
                    "Executing command through live Nixward authority"
                );
            }
        }

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
        self.history.push_back(record);

        if self.history.len() > 1000 {
            self.history.pop_front();
        }
    }

    fn record_authorized_execution(
        &mut self,
        command: NixOSCommand,
        action_intent_digest: String,
        approval_request_id: String,
        projection_digest: String,
        pre_state_identity: Option<String>,
        result: &ExecutionResult,
    ) {
        self.authorized_history
            .push_back(AuthorizedExecutionRecordV1 {
                command,
                action_intent_digest,
                approval_request_id,
                projection_digest,
                pre_state_identity,
                result: result.clone(),
                timestamp_ms: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0),
            });

        if self.authorized_history.len() > 1000 {
            self.authorized_history.pop_front();
        }
    }

    pub fn history(&self) -> &VecDeque<ExecutionRecord> {
        &self.history
    }

    pub fn authorized_history(&self) -> &VecDeque<AuthorizedExecutionRecordV1> {
        &self.authorized_history
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
    fn service_pre_state_identity_parser_is_strict_and_unit_bound() {
        let identity = "nixward-service-pre-state-v1|generation=42|unit=nginx.service|state=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let (generation, unit, digest) = parse_service_pre_state_identity(identity).unwrap();
        assert_eq!(generation, Some(42));
        assert_eq!(unit, "nginx.service");
        assert_eq!(digest.len(), 64);
        assert!(parse_service_pre_state_identity(
            "nixward-service-pre-state-v1|generation=none|unit=nginx.service|state=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
        )
        .is_err());

        assert!(parse_service_pre_state_identity(
            "nixward-service-pre-state-v1|generation=42|unit=nginx|state=0123"
        )
        .is_err());
        assert!(parse_service_pre_state_identity(
            "nixward-service-pre-state-v1|generation=42|unit=nginx.service|state=not-a-digest"
        )
        .is_err());
    }

    #[test]
    fn service_pre_state_observation_rejects_generation_change() {
        let state = NixServiceObservedStateV1::parse_systemd_properties(
            "nginx.service",
            concat!(
                "Id=nginx.service\n",
                "Names=nginx.service\n",
                "LoadState=loaded\n",
                "ActiveState=active\n",
                "SubState=running\n",
                "UnitFileState=enabled\n",
            ),
        )
        .unwrap();
        let identity = state.execution_pre_state_identity(42).unwrap();

        let error =
            validate_service_pre_state_observation(&identity, "nginx.service", 43, &state)
                .unwrap_err();
        assert!(error.contains("approved generation=42"));
    }

    #[test]
    fn service_pre_state_observation_rejects_state_transition_same_generation() {
        let active = NixServiceObservedStateV1::parse_systemd_properties(
            "nginx.service",
            concat!(
                "Id=nginx.service\n",
                "Names=nginx.service\n",
                "LoadState=loaded\n",
                "ActiveState=active\n",
                "SubState=running\n",
                "UnitFileState=enabled\n",
            ),
        )
        .unwrap();
        let failed = NixServiceObservedStateV1::parse_systemd_properties(
            "nginx.service",
            concat!(
                "Id=nginx.service\n",
                "Names=nginx.service\n",
                "LoadState=loaded\n",
                "ActiveState=failed\n",
                "SubState=failed\n",
                "UnitFileState=enabled\n",
            ),
        )
        .unwrap();
        let identity = active.execution_pre_state_identity(42).unwrap();

        let error =
            validate_service_pre_state_observation(&identity, "nginx.service", 42, &failed)
                .unwrap_err();
        assert!(error.contains("approved pre-state digest"));
    }

    #[test]
    fn service_pre_state_observation_rejects_cross_unit_use() {
        let nginx = NixServiceObservedStateV1::parse_systemd_properties(
            "nginx.service",
            concat!(
                "Id=nginx.service\n",
                "Names=nginx.service\n",
                "LoadState=loaded\n",
                "ActiveState=active\n",
                "SubState=running\n",
                "UnitFileState=enabled\n",
            ),
        )
        .unwrap();
        let identity = nginx.execution_pre_state_identity(42).unwrap();

        let error =
            validate_service_pre_state_observation(&identity, "sshd.service", 42, &nginx)
                .unwrap_err();
        assert!(error.contains("unit mismatch"));
    }

    #[test]
    fn generation_pre_state_identity_parser_is_strict() {
        assert_eq!(
            parse_generation_pre_state_identity("generation:42").unwrap(),
            42
        );
        assert!(parse_generation_pre_state_identity("generation:").is_err());
        assert!(parse_generation_pre_state_identity("generation:-1").is_err());
        assert!(parse_generation_pre_state_identity("host:workstation").is_err());
    }

    #[test]
    fn current_generation_parser_accepts_structured_json() {
        let json =
            r#"[{"generation": 874, "current": true}, {"generation": 873, "current": false}]"#;
        assert_eq!(parse_current_generation(json).unwrap(), 874);
    }

    #[test]
    fn current_generation_parser_rejects_missing_current_generation() {
        let json = r#"[{"generation": 874, "current": false}]"#;
        assert!(parse_current_generation(json).is_err());
    }

    #[test]
    fn current_generation_parser_rejects_multiple_current_generations() {
        let json =
            r#"[{"generation": 874, "current": true}, {"generation": 873, "current": true}]"#;
        assert!(parse_current_generation(json).is_err());
    }

    #[test]
    fn current_generation_parser_rejects_malformed_json() {
        assert!(parse_current_generation("{not-json}").is_err());
    }

    #[test]
    fn authorized_history_records_non_phi_provenance() {
        let mut executor = NixOSExecutor::new().with_dry_run(true);
        let command = NixOSCommand::Service {
            operation: NixServiceOperationKindV1::Restart,
            unit: "nginx.service".to_string(),
        };
        let result = ExecutionResult::Success {
            stdout: "dry-run".to_string(),
            stderr: String::new(),
            execution_time_ms: 0,
        };

        executor.record_authorized_execution(
            command.clone(),
            "intent-123".to_string(),
            "request-123".to_string(),
            "projection-123".to_string(),
            Some("generation:42".to_string()),
            &result,
        );

        let record = executor.authorized_history().front().unwrap();
        assert!(matches!(
            record.command(),
            NixOSCommand::Service {
                operation: NixServiceOperationKindV1::Restart,
                unit
            } if unit == "nginx.service"
        ));
        assert_eq!(record.action_intent_digest(), "intent-123");
        assert_eq!(record.approval_request_id(), "request-123");
        assert_eq!(record.projection_digest(), "projection-123");
        assert_eq!(record.pre_state_identity(), Some("generation:42"));
        assert!(matches!(record.result(), ExecutionResult::Success { .. }));
        assert!(record.timestamp_ms() > 0);
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

    #[test]
    fn config_patch_command_has_fixed_scope_and_argv() {
        let command = NixOSCommand::ConfigPatch {
            option_path: "services.nginx.enable".to_string(),
            value: "true".to_string(),
            expected_config_digest: "ab".repeat(32),
        };
        assert_eq!(command.safety_level(), SafetyLevel::SystemCritical);
        assert!(command.validate_shape().is_ok());
        let (bin, args) = command.to_command();
        assert_eq!(bin, "nixos-rebuild");
        assert_eq!(args, vec!["switch"]);
    }

    #[test]
    fn invalid_config_patch_digest_is_blocked() {
        let command = NixOSCommand::ConfigPatch {
            option_path: "services.nginx.enable".to_string(),
            value: "true".to_string(),
            expected_config_digest: "not-a-digest".to_string(),
        };
        assert!(command.validate_shape().is_err());
    }

    #[test]
    fn typed_service_command_has_fixed_scope_and_argv() {
        let command = NixOSCommand::Service {
            operation: NixServiceOperationKindV1::Restart,
            unit: "nginx.service".to_string(),
        };
        assert_eq!(command.safety_level(), SafetyLevel::SystemModify);
        assert!(command.validate_shape().is_ok());
        let (bin, args) = command.to_command();
        assert_eq!(bin, "systemctl");
        assert_eq!(args, vec!["restart", "nginx.service"]);
    }

    #[tokio::test]
    async fn execute_rejects_valid_service_without_live_authority_even_at_high_phi() {
        let mut executor = NixOSExecutor::new();
        let command = NixOSCommand::Service {
            operation: NixServiceOperationKindV1::Restart,
            unit: "nginx.service".to_string(),
        };

        let result = executor.execute(command, 1.0).await;

        assert!(matches!(
            result,
            ExecutionResult::Blocked {
                safety_level: SafetyLevel::SystemModify,
                reason
            } if reason.contains("live Nixward execution authority")
        ));
    }

    #[tokio::test]
    async fn execute_confirmed_rejects_valid_service_without_live_authority() {
        let mut executor = NixOSExecutor::new();
        let command = NixOSCommand::Service {
            operation: NixServiceOperationKindV1::Start,
            unit: "nginx.service".to_string(),
        };

        let result = executor.execute_confirmed(command, 1.0).await;

        assert!(matches!(
            result,
            ExecutionResult::Blocked {
                safety_level: SafetyLevel::SystemModify,
                reason
            } if reason.contains("live Nixward execution authority")
        ));
    }

    #[tokio::test]
    async fn executor_preserves_dry_run_service_preview_without_authority() {
        let mut executor = NixOSExecutor::new().with_dry_run(true);
        let command = NixOSCommand::Service {
            operation: NixServiceOperationKindV1::Stop,
            unit: "nginx.service".to_string(),
        };

        let result = executor.execute(command, 1.0).await;

        assert!(matches!(
            result,
            ExecutionResult::Success { stdout, .. }
                if stdout.contains("[DRY-RUN] Would execute: systemctl stop nginx.service")
        ));
    }

    #[tokio::test]
    async fn legacy_custom_systemctl_cannot_execute_without_live_authority() {
        let mut executor = NixOSExecutor::new();
        let command = NixOSCommand::Custom {
            command: "systemctl".to_string(),
            args: vec!["restart".to_string(), "nginx.service".to_string()],
            safety_level: SafetyLevel::SystemModify,
        };

        let result = executor.execute(command, 1.0).await;

        assert!(matches!(
            result,
            ExecutionResult::Blocked {
                safety_level: SafetyLevel::SystemModify,
                reason
            } if reason.contains("typed Nixward command")
        ));
    }

    #[tokio::test]
    async fn arbitrary_custom_wrapper_cannot_reach_service_effects() {
        let mut executor = NixOSExecutor::new();
        for (command, args) in [
            ("sh", vec!["-c".to_string(), "systemctl restart nginx.service".to_string()]),
            ("env", vec!["systemctl".to_string(), "restart".to_string(), "nginx.service".to_string()]),
        ] {
            let result = executor
                .execute(
                    NixOSCommand::Custom {
                        command: command.to_string(),
                        args,
                        safety_level: SafetyLevel::SystemModify,
                    },
                    1.0,
                )
                .await;
            assert!(matches!(
                result,
                ExecutionResult::Blocked {
                    safety_level: SafetyLevel::SystemModify,
                    reason
                } if reason.contains("typed Nixward command")
            ));
        }
    }

    #[tokio::test]
    async fn custom_commands_remain_previewable_in_dry_run() {
        let mut executor = NixOSExecutor::new().with_dry_run(true);
        let result = executor
            .execute(
                NixOSCommand::Custom {
                    command: "sh".to_string(),
                    args: vec!["-c".to_string(), "systemctl restart nginx.service".to_string()],
                    safety_level: SafetyLevel::SystemModify,
                },
                1.0,
            )
            .await;
        assert!(matches!(
            result,
            ExecutionResult::Success { stdout, .. }
                if stdout.contains("[DRY-RUN] Would execute: sh -c systemctl restart nginx.service")
        ));
    }

    #[test]
    fn service_pre_state_digest_identity_changes_when_observed_state_changes() {
        let active = NixServiceObservedStateV1::parse_systemd_properties(
            "nginx.service",
            concat!(
                "Id=nginx.service\n",
                "Names=nginx.service\n",
                "LoadState=loaded\n",
                "ActiveState=active\n",
                "SubState=running\n",
                "UnitFileState=enabled\n",
            ),
        )
        .unwrap();
        let failed = NixServiceObservedStateV1::parse_systemd_properties(
            "nginx.service",
            concat!(
                "Id=nginx.service\n",
                "Names=nginx.service\n",
                "LoadState=loaded\n",
                "ActiveState=failed\n",
                "SubState=failed\n",
                "UnitFileState=enabled\n",
            ),
        )
        .unwrap();

        let a = active.execution_pre_state_identity(42).unwrap();
        let b = failed.execution_pre_state_identity(42).unwrap();

        assert_ne!(a, b);
        assert_eq!(parse_service_pre_state_identity(&a).unwrap().0, 42);
        assert_eq!(parse_service_pre_state_identity(&b).unwrap().0, 42);
    }

    #[tokio::test]
    async fn invalid_typed_service_is_blocked_even_when_confirmed() {
        let mut executor = NixOSExecutor::new().with_dry_run(true);
        let command = NixOSCommand::Service {
            operation: NixServiceOperationKindV1::Restart,
            unit: "nginx*.service".to_string(),
        };
        let result = executor.execute_confirmed(command, 1.0).await;
        assert!(matches!(
            result,
            ExecutionResult::Blocked {
                safety_level: SafetyLevel::SystemModify,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn test_execute_confirmed_rejects_config_patch_without_authority() {
        let mut executor = NixOSExecutor::new().with_dry_run(true);
        let command = NixOSCommand::ConfigPatch {
            option_path: "services.nginx.enable".to_string(),
            value: "true".to_string(),
            expected_config_digest: "ab".repeat(32),
        };
        let result = executor.execute_confirmed(command, 1.0).await;
        assert!(matches!(
            result,
            ExecutionResult::Blocked {
                safety_level: SafetyLevel::SystemCritical,
                ..
            }
        ));
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
        assert!(
            hm.rollback_command().is_none(),
            "Home Manager rollback must not use an arbitrary shell command"
        );
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
