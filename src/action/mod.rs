// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Action-level safety primitives: declarative policy, sandboxed paths, and a safe action IR.
//!
//! These types are intentionally minimal so they can be integrated without pulling in heavy
//! executors. They model the Phase 1/2 security scaffolding from the v1.2 plan.
//!
//! ## Modules
//!
//! - `nixos_patterns`: NixOS-specific command patterns with rollback support

pub mod bindings;
pub mod nixos_patterns;
pub mod primitives;

pub use nixos_patterns::{
    ChannelOperation, ExecutionRecord as NixOSExecutionRecord,
    ExecutionResult as NixOSExecutionResult, FlakeOperation, NixOSCommand, NixOSExecutor,
    SafetyLevel,
};

use serde::{Deserialize, Serialize};
pub use symthaea_dream::CausalLink;
use symthaea_dream::{DreamEngine, DreamEngineConfig, DreamableAction};

impl DreamableAction for ActionIR {
    fn perturb(&self, seed: u64) -> Self {
        match self {
            ActionIR::RunCommand {
                program,
                args,
                env,
                working_dir,
            } => {
                let mut new_args = args.clone();
                if seed % 2 == 0 {
                    if new_args.is_empty() {
                        new_args.push("--help".to_string());
                    } else if program == "rm" {
                        return ActionIR::RunCommand {
                            program: "ls".to_string(),
                            args: new_args,
                            env: env.clone(),
                            working_dir: working_dir.clone(),
                        };
                    }
                }

                ActionIR::RunCommand {
                    program: program.clone(),
                    args: new_args,
                    env: env.clone(),
                    working_dir: working_dir.clone(),
                }
            }
            ActionIR::WriteFile {
                path,
                content,
                create_dirs,
            } => {
                let mut new_path = path.clone();
                if let Some(ext) = path.extension() {
                    let mut new_ext = ext.to_os_string();
                    new_ext.push(".dream");
                    new_path.set_extension(new_ext);
                } else {
                    new_path.set_extension("dream");
                }

                ActionIR::WriteFile {
                    path: new_path,
                    content: content.clone(),
                    create_dirs: *create_dirs,
                }
            }
            ActionIR::DeleteFile { path } => ActionIR::RunCommand {
                program: "ls".to_string(),
                args: vec!["-l".to_string(), path.to_string_lossy().to_string()],
                env: BTreeMap::new(),
                working_dir: None,
            },
            ActionIR::Sequence(actions) => {
                let idx = (seed as usize) % actions.len().max(1);
                let mut new_actions = actions.clone();
                if !new_actions.is_empty() {
                    new_actions[idx] = new_actions[idx].perturb(seed.wrapping_add(1));
                }
                ActionIR::Sequence(new_actions)
            }
            ActionIR::ReadSensor {
                sensor_id,
                channels,
            } => ActionIR::ReadSensor {
                sensor_id: sensor_id.clone(),
                channels: channels.clone(),
            },
            ActionIR::WriteServo { servo_id, value } => ActionIR::WriteServo {
                servo_id: *servo_id,
                value: value + ((seed % 10) as f32 / 100.0),
            },
            ActionIR::SwarmGossip { topic, payload } => ActionIR::SwarmGossip {
                topic: topic.clone(),
                payload: payload.clone(),
            },
            ActionIR::WasmSandbox {
                module_path,
                function_name,
                input_data,
            } => ActionIR::WasmSandbox {
                module_path: module_path.clone(),
                function_name: function_name.clone(),
                input_data: input_data.clone(),
            },
            ActionIR::NoOp => ActionIR::NoOp,
            ActionIR::ReadFile { path, encoding } => ActionIR::ReadFile {
                path: path.clone(),
                encoding: encoding.clone(),
            },
            ActionIR::CreateDirectory { path, recursive } => ActionIR::CreateDirectory {
                path: path.clone(),
                recursive: *recursive,
            },
            ActionIR::ListDirectory { path, recursive } => ActionIR::ListDirectory {
                path: path.clone(),
                recursive: *recursive,
            },
        }
    }

    fn predict_outcome(&self, state: &[f32]) -> Vec<f32> {
        let score = match self.destructiveness() {
            DestructivenessLevel::ReadOnly => 0.8,
            DestructivenessLevel::Reversible => 0.6,
            DestructivenessLevel::NeedsConfirmation => 0.4,
            DestructivenessLevel::Destructive => 0.1,
        };
        // Mix with current state context
        state
            .iter()
            .map(|&s| (s * 0.5 + score * 0.5).clamp(-1.0, 1.0))
            .collect()
    }

    fn magnitude(&self) -> f32 {
        match self.destructiveness() {
            DestructivenessLevel::Destructive => 1.0,
            DestructivenessLevel::NeedsConfirmation => 0.7,
            DestructivenessLevel::Reversible => 0.3,
            DestructivenessLevel::ReadOnly => 0.05,
        }
    }
}

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Complete security policy bundle (TOML-ready).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyBundle {
    pub version: String,
    pub name: String,
    pub capabilities: Capabilities,
    pub budgets: Budgets,
}

impl PolicyBundle {
    /// Restrictive default: allow only basic read/list operations and a tiny write sandbox.
    pub fn restrictive() -> Self {
        Self {
            version: "1.0.0".into(),
            name: "restrictive".into(),
            capabilities: Capabilities {
                shell: ShellCapabilities {
                    allowed_programs: ["nix", "ls", "cat", "echo"]
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                    blocked_programs: BTreeSet::new(),
                    budget_per_hour: 100,
                    allowed_env: BTreeMap::new(),
                    min_phi: 0.5,
                },
                filesystem: FilesystemCapabilities {
                    read_patterns: vec!["/tmp/symthaea/".into()],
                    write_patterns: vec!["/tmp/symthaea/".into()],
                    max_write_bytes: 10 * 1024 * 1024, // 10 MB
                },
                network: NetworkCapabilities {
                    allowed_hosts: vec![],
                    allowed_ports: vec![],
                    enabled: false,
                },
                min_phi: 0.3,
            },
            budgets: Budgets {
                shell_commands_per_session: 100,
                file_writes_per_session: 50,
                bytes_written_per_session: 50 * 1024 * 1024,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capabilities {
    pub shell: ShellCapabilities,
    pub filesystem: FilesystemCapabilities,
    pub network: NetworkCapabilities,
    /// Minimum Phi level required for any non-readonly action.
    #[serde(default = "default_min_phi")]
    pub min_phi: f64,
}

fn default_min_phi() -> f64 {
    0.3
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShellCapabilities {
    pub allowed_programs: BTreeSet<String>,
    pub blocked_programs: BTreeSet<String>,
    pub budget_per_hour: u32,
    pub allowed_env: BTreeMap<String, String>,
    /// Minimum Phi level required for shell execution.
    #[serde(default = "default_shell_phi")]
    pub min_phi: f64,
}

fn default_shell_phi() -> f64 {
    0.5
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilesystemCapabilities {
    pub read_patterns: Vec<String>,
    pub write_patterns: Vec<String>,
    pub max_write_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkCapabilities {
    pub allowed_hosts: Vec<String>,
    pub allowed_ports: Vec<u16>,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Budgets {
    pub shell_commands_per_session: u32,
    pub file_writes_per_session: u32,
    pub bytes_written_per_session: u64,
}

/// Secure sandbox root for path validation.
#[derive(Debug, Clone)]
pub struct SandboxRoot {
    root: PathBuf,
}

impl SandboxRoot {
    /// Create a sandbox rooted at `/tmp/symthaea/{session}`.
    pub fn new(session_id: &str) -> std::io::Result<Self> {
        let path = PathBuf::from(format!("/tmp/symthaea/{session_id}"));
        std::fs::create_dir_all(&path)?;
        let canonical = path.canonicalize()?;
        Ok(Self { root: canonical })
    }

    /// Create a sandbox at a specific path.
    pub fn at(path: PathBuf) -> std::io::Result<Self> {
        std::fs::create_dir_all(&path)?;
        let canonical = path.canonicalize()?;
        Ok(Self { root: canonical })
    }

    /// Validate a path is inside the sandbox after canonicalization.
    ///
    /// Relative paths are resolved against the sandbox root. This allows
    /// commands like `cargo check` with `working_dir: "."` to work correctly
    /// when the sandbox root is the project directory.
    pub fn validate(&self, requested: &Path) -> Result<PathBuf, PolicyViolation> {
        // Resolve relative paths against the sandbox root
        let requested = if requested.is_absolute() {
            requested.to_path_buf()
        } else {
            self.root.join(requested)
        };
        let requested = requested.as_path();

        let canonical = if requested.exists() {
            requested
                .canonicalize()
                .map_err(|e| PolicyViolation::SandboxEscape(e.to_string()))?
        } else {
            let mut ancestor = requested.parent();
            let mut found = None;
            while let Some(parent) = ancestor {
                if parent.exists() {
                    found = Some(parent);
                    break;
                }
                ancestor = parent.parent();
            }

            let base = found.ok_or_else(|| {
                PolicyViolation::SandboxEscape(format!(
                    "cannot canonicalize {}",
                    requested.display()
                ))
            })?;
            let base_canon = base
                .canonicalize()
                .map_err(|e| PolicyViolation::SandboxEscape(e.to_string()))?;
            let suffix = requested.strip_prefix(base).unwrap_or(requested);
            base_canon.join(suffix)
        };

        let canonical = normalize_path(&canonical);

        if !canonical.starts_with(&self.root) {
            return Err(PolicyViolation::SandboxEscape(format!(
                "path {} escapes sandbox {}",
                canonical.display(),
                self.root.display()
            )));
        }

        Ok(canonical)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
}

/// Safe intermediate representation for actions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionIR {
    ReadFile {
        path: PathBuf,
        encoding: Option<String>,
    },
    WriteFile {
        path: PathBuf,
        content: Vec<u8>,
        create_dirs: bool,
    },
    DeleteFile {
        path: PathBuf,
    },
    CreateDirectory {
        path: PathBuf,
        recursive: bool,
    },
    ListDirectory {
        path: PathBuf,
        recursive: bool,
    },
    RunCommand {
        program: String,
        args: Vec<String>,
        env: BTreeMap<String, String>,
        working_dir: Option<PathBuf>,
    },
    /// HAL: Read from a physical sensor (IMU, Power, etc.)
    ReadSensor {
        sensor_id: String,
        channels: Vec<String>,
    },
    /// HAL: Write to a physical servo or PWM channel
    WriteServo {
        servo_id: u32,
        value: f32,
    },
    /// Swarm: Broadcast a message (optimization, research, or curriculum) to the swarm.
    SwarmGossip {
        topic: String,
        payload: Vec<u8>,
    },
    /// Forge: Execute a pre-compiled .wasm optimization in a sandbox for verification.
    WasmSandbox {
        module_path: PathBuf,
        function_name: String,
        input_data: Vec<u8>,
    },
    Sequence(Vec<ActionIR>),
    NoOp,
}

impl ActionIR {
    /// Whether the action can be rolled back automatically.
    pub fn is_reversible(&self) -> bool {
        match self {
            ActionIR::ReadFile { .. } => true,
            ActionIR::ListDirectory { .. } => true,
            ActionIR::NoOp => true,
            ActionIR::WriteFile { .. } => true,
            ActionIR::DeleteFile { .. } => true,
            ActionIR::CreateDirectory { .. } => true,
            ActionIR::ReadSensor { .. } => true,
            ActionIR::WriteServo { .. } => true,
            ActionIR::SwarmGossip { .. } => true,
            ActionIR::WasmSandbox { .. } => true,
            ActionIR::Sequence(actions) => actions.iter().all(|a| a.is_reversible()),
            ActionIR::RunCommand { .. } => false,
        }
    }

    /// Classify risk tier for budgeting/logging.
    pub fn risk_tier(&self) -> RiskTier {
        match self {
            ActionIR::ReadFile { .. }
            | ActionIR::ListDirectory { .. }
            | ActionIR::NoOp
            | ActionIR::ReadSensor { .. }
            | ActionIR::WasmSandbox { .. } => RiskTier::Low,
            ActionIR::WriteFile { .. }
            | ActionIR::DeleteFile { .. }
            | ActionIR::CreateDirectory { .. } => RiskTier::Medium,
            ActionIR::WriteServo { .. }
            | ActionIR::SwarmGossip { .. }
            | ActionIR::RunCommand { .. } => RiskTier::High,
            ActionIR::Sequence(actions) => actions
                .iter()
                .map(|a| a.risk_tier())
                .max()
                .unwrap_or(RiskTier::Low),
        }
    }

    /// Classify destructiveness level for shell sidecar confirmation flow.
    /// More granular than is_reversible() - distinguishes between read-only,
    /// reversible, needs-confirmation, and destructive operations.
    pub fn destructiveness(&self) -> DestructivenessLevel {
        match self {
            ActionIR::ReadFile { .. }
            | ActionIR::ListDirectory { .. }
            | ActionIR::NoOp
            | ActionIR::ReadSensor { .. }
            | ActionIR::WasmSandbox { .. } => DestructivenessLevel::ReadOnly,
            ActionIR::WriteFile { .. } | ActionIR::CreateDirectory { .. } => {
                DestructivenessLevel::Reversible
            }
            ActionIR::DeleteFile { .. } => DestructivenessLevel::Destructive,
            ActionIR::WriteServo { .. } | ActionIR::SwarmGossip { .. } => {
                DestructivenessLevel::NeedsConfirmation
            }
            ActionIR::RunCommand { program, args, .. } => {
                classify_command_destructiveness(program, args)
            }
            ActionIR::Sequence(actions) => actions
                .iter()
                .map(|a| a.destructiveness())
                .max()
                .unwrap_or(DestructivenessLevel::ReadOnly),
        }
    }

    /// Get rollback hint for the shell sidecar to display
    pub fn rollback_hint(&self) -> Option<String> {
        match self {
            ActionIR::RunCommand { program, args, .. } => get_rollback_hint(program, args),
            ActionIR::WriteFile { path, .. } => {
                Some(format!("Restore from backup: {}.bak", path.display()))
            }
            ActionIR::DeleteFile { path } => Some(format!(
                "File will be backed up before deletion: {}",
                path.display()
            )),
            _ => None,
        }
    }

    /// Validate against policy and sandbox.
    pub fn validate(
        &self,
        policy: &PolicyBundle,
        sandbox: &SandboxRoot,
        current_phi: f64,
    ) -> Result<(), PolicyViolation> {
        // Consciousness Gate: Check global min_phi for any mutation
        if self.risk_tier() > RiskTier::Low && current_phi < policy.capabilities.min_phi {
            return Err(PolicyViolation::PhiTooLow {
                required: policy.capabilities.min_phi,
                actual: current_phi,
            });
        }

        match self {
            ActionIR::ReadFile { path, .. } | ActionIR::ListDirectory { path, .. } => {
                let canonical = sandbox.validate(path)?;
                ensure_pattern(
                    &canonical,
                    &policy.capabilities.filesystem.read_patterns,
                    sandbox,
                    AccessKind::Read,
                )?;
            }
            ActionIR::WriteFile { path, content, .. } => {
                let canonical = sandbox.validate(path)?;
                ensure_pattern(
                    &canonical,
                    &policy.capabilities.filesystem.write_patterns,
                    sandbox,
                    AccessKind::Write,
                )?;
                if content.len() as u64 > policy.capabilities.filesystem.max_write_bytes {
                    return Err(PolicyViolation::WriteTooLarge(content.len()));
                }
            }
            ActionIR::DeleteFile { path } | ActionIR::CreateDirectory { path, .. } => {
                let canonical = sandbox.validate(path)?;
                ensure_pattern(
                    &canonical,
                    &policy.capabilities.filesystem.write_patterns,
                    sandbox,
                    AccessKind::Write,
                )?;
            }
            ActionIR::ReadSensor { .. }
            | ActionIR::WriteServo { .. }
            | ActionIR::SwarmGossip { .. }
            | ActionIR::WasmSandbox { .. } => {
                // HAL, Swarm, and Forge primitives are currently allowed by default if they pass the global Phi gate
            }
            ActionIR::RunCommand {
                program,
                env,
                working_dir,
                ..
            } => {
                // Specific gate for shell execution
                if current_phi < policy.capabilities.shell.min_phi {
                    return Err(PolicyViolation::PhiTooLow {
                        required: policy.capabilities.shell.min_phi,
                        actual: current_phi,
                    });
                }

                if policy.capabilities.shell.blocked_programs.contains(program) {
                    return Err(PolicyViolation::ProgramBlocked(program.clone()));
                }
                if !policy.capabilities.shell.allowed_programs.contains(program) {
                    return Err(PolicyViolation::ProgramNotAllowed(program.clone()));
                }
                validate_env_overrides(env, &policy.capabilities.shell.allowed_env)?;
                if let Some(dir) = working_dir {
                    let canonical = sandbox.validate(dir)?;
                    ensure_pattern(
                        &canonical,
                        &policy.capabilities.filesystem.read_patterns,
                        sandbox,
                        AccessKind::Read,
                    )?;
                }
            }
            ActionIR::Sequence(actions) => {
                for action in actions {
                    action.validate(policy, sandbox, current_phi)?;
                }
            }
            ActionIR::NoOp => {}
        }
        Ok(())
    }
}

/// Risk tier for auditing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RiskTier {
    Low,
    Medium,
    High,
}

/// Destructiveness level for command classification (Shell Sidecar support).
/// Provides granular classification beyond binary is_reversible().
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Serialize, Deserialize)]
pub enum DestructivenessLevel {
    /// Read-only operations: nix search, cat, ls, nix flake show
    #[default]
    ReadOnly,
    /// Reversible operations: nix-env -i (can uninstall), file writes with backup
    Reversible,
    /// Needs confirmation but has rollback: nixos-rebuild switch, systemctl restart
    NeedsConfirmation,
    /// Destructive/irreversible: nix-collect-garbage -d, rm -rf, format operations
    Destructive,
}

impl DestructivenessLevel {
    /// Whether this level requires explicit user confirmation
    pub fn requires_confirmation(&self) -> bool {
        matches!(self, Self::NeedsConfirmation | Self::Destructive)
    }

    /// Whether a rollback plan is available
    pub fn has_rollback(&self) -> bool {
        matches!(self, Self::Reversible | Self::NeedsConfirmation)
    }

    /// Get human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            Self::ReadOnly => "Read-only operation, no system changes",
            Self::Reversible => "Reversible operation, can be undone",
            Self::NeedsConfirmation => "System change with rollback available",
            Self::Destructive => "Irreversible operation, cannot be undone",
        }
    }
}

/// Capability tier for remote command execution surfaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RemoteCommandCapability {
    /// Safe observational commands with no intended side effects.
    ReadOnly,
    /// Explicitly recognized mutating commands.
    Mutating,
}

/// Classify a parsed command against the remote execution capability allowlist.
///
/// Unknown commands are rejected so network-facing execution surfaces do not silently expand.
pub fn classify_remote_command_capability(
    program: &str,
    args: &[String],
) -> Result<RemoteCommandCapability, String> {
    let program = program.to_lowercase();
    let arg0 = args.first().map(|s| s.to_lowercase());
    let arg1 = args.get(1).map(|s| s.to_lowercase());

    match program.as_str() {
        "pwd" | "echo" | "true" | "false" | "date" | "whoami" | "id" | "uname" | "hostname"
        | "printenv" | "env" | "ls" | "cat" | "head" | "tail" | "stat" | "which" | "rg"
        | "grep" | "find" | "journalctl" | "ps" | "df" | "du" | "free" | "sleep" => {
            Ok(RemoteCommandCapability::ReadOnly)
        }
        "git" => match arg0.as_deref() {
            Some("status" | "diff" | "show" | "log" | "rev-parse" | "branch") => {
                Ok(RemoteCommandCapability::ReadOnly)
            }
            Some(
                "commit" | "push" | "apply" | "checkout" | "switch" | "merge" | "rebase" | "pull"
                | "reset" | "clean",
            ) => Ok(RemoteCommandCapability::Mutating),
            Some(other) => Err(format!(
                "git subcommand '{other}' is not allowed over remote execution"
            )),
            None => Err("git requires a subcommand".to_string()),
        },
        "nix" => match (arg0.as_deref(), arg1.as_deref()) {
            (Some("search"), _)
            | (Some("eval"), _)
            | (Some("path-info"), _)
            | (Some("flake"), Some("show" | "metadata")) => Ok(RemoteCommandCapability::ReadOnly),
            (Some("profile"), Some("install" | "remove")) => Ok(RemoteCommandCapability::Mutating),
            _ => Err("nix command is not in the remote execution allowlist".to_string()),
        },
        "nix-env" => match arg0.as_deref() {
            Some("-q" | "--query") => Ok(RemoteCommandCapability::ReadOnly),
            Some("-i" | "--install" | "-e" | "--uninstall") => {
                Ok(RemoteCommandCapability::Mutating)
            }
            _ => Err("nix-env command is not in the remote execution allowlist".to_string()),
        },
        "systemctl" => match arg0.as_deref() {
            Some("status" | "show" | "list-units" | "list-unit-files" | "is-active") => {
                Ok(RemoteCommandCapability::ReadOnly)
            }
            Some("restart" | "start" | "stop" | "disable" | "enable") => {
                Ok(RemoteCommandCapability::Mutating)
            }
            _ => Err("systemctl verb is not in the remote execution allowlist".to_string()),
        },
        "nixos-rebuild" => match arg0.as_deref() {
            Some("dry-run" | "build") => Ok(RemoteCommandCapability::ReadOnly),
            Some("test" | "switch" | "boot") => Ok(RemoteCommandCapability::Mutating),
            _ => Err("nixos-rebuild verb is not in the remote execution allowlist".to_string()),
        },
        "mkdir" | "touch" | "cp" | "mv" | "chmod" | "chown" | "rm" | "dd" | "mkfs" | "fdisk"
        | "parted" | "shred" | "wipefs" => Ok(RemoteCommandCapability::Mutating),
        _ => Err(format!(
            "program '{program}' is not in the remote execution capability allowlist"
        )),
    }
}

/// Validation errors.
#[derive(Debug, Clone)]
pub enum PolicyViolation {
    SandboxEscape(String),
    ReadNotAllowed(PathBuf),
    WriteNotAllowed(PathBuf),
    WriteTooLarge(usize),
    ProgramBlocked(String),
    ProgramNotAllowed(String),
    EnvNotAllowed(String),
    EnvValueMismatch {
        key: String,
        expected: String,
        actual: String,
    },
    ShellBudgetExceeded {
        allowed: u32,
        attempted: u32,
    },
    ShellHourlyBudgetExceeded {
        allowed: u32,
        attempted: u32,
    },
    FileWriteBudgetExceeded {
        allowed: u32,
        attempted: u32,
    },
    BytesWrittenBudgetExceeded {
        allowed: u64,
        attempted: u64,
    },
    PhiTooLow {
        required: f64,
        actual: f64,
    },
}

#[derive(Debug, Clone, Copy)]
enum AccessKind {
    Read,
    Write,
}

fn ensure_pattern(
    path: &Path,
    patterns: &[String],
    sandbox: &SandboxRoot,
    access: AccessKind,
) -> Result<(), PolicyViolation> {
    let path_str = path.to_string_lossy();
    let root_str = sandbox.root().to_string_lossy();

    // Simple prefix-based allowlist; replace with glob if needed.
    if patterns.iter().any(|p| {
        let normalized = if p.ends_with("**") {
            p.trim_end_matches("**")
        } else {
            p
        };

        if normalized.starts_with('/') {
            path_str.starts_with(normalized)
        } else {
            // Treat non-absolute pattern as relative to sandbox root.
            let candidate = format!("{}/{}", root_str, normalized.trim_start_matches("./"));
            path_str.starts_with(&candidate)
        }
    }) {
        Ok(())
    } else {
        match access {
            AccessKind::Read => Err(PolicyViolation::ReadNotAllowed(path.to_path_buf())),
            AccessKind::Write => Err(PolicyViolation::WriteNotAllowed(path.to_path_buf())),
        }
    }
}

fn validate_env_overrides(
    env: &BTreeMap<String, String>,
    allowed: &BTreeMap<String, String>,
) -> Result<(), PolicyViolation> {
    if env.is_empty() {
        return Ok(());
    }

    if allowed.is_empty() {
        return Err(PolicyViolation::EnvNotAllowed(
            "environment overrides disabled by policy".to_string(),
        ));
    }

    for (key, value) in env {
        match allowed.get(key) {
            Some(allowed_value) if allowed_value == "*" || allowed_value == value => {}
            Some(allowed_value) => {
                return Err(PolicyViolation::EnvValueMismatch {
                    key: key.clone(),
                    expected: allowed_value.clone(),
                    actual: value.clone(),
                });
            }
            None => return Err(PolicyViolation::EnvNotAllowed(key.clone())),
        }
    }

    Ok(())
}

fn normalize_path(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(Path::new("/")),
            Component::CurDir => {}
            Component::ParentDir => {
                let _ = normalized.pop();
            }
            Component::Normal(part) => normalized.push(part),
        }
    }
    normalized
}

/// Outcome of executing an action.
#[derive(Debug, Clone, PartialEq)]
pub enum ActionOutcome {
    Success,
    FileContent(Vec<u8>),
    DirectoryListing(Vec<PathBuf>),
    SensorData {
        sensor_id: String,
        values: BTreeMap<String, f32>,
    },
    ServoStatus {
        servo_id: u32,
        current_value: f32,
    },
    WasmResult {
        output: Vec<u8>,
        logs: Vec<String>,
    },
    SimulatedCommand {
        program: String,
        args: Vec<String>,
    },
    CommandOutput {
        stdout: Vec<u8>,
        stderr: Vec<u8>,
        exit_code: i32,
    },
}

/// Execution error.
#[derive(Debug, Error)]
pub enum ExecutionError {
    #[error("policy violation: {0:?}")]
    Policy(PolicyViolation),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("unsupported: {0}")]
    Unsupported(String),
}

/// Action-level errors for simulation and planning.
#[derive(Debug, Clone, Error)]
pub enum ActionError {
    #[error("unexpected simulation outcome: {0}")]
    UnexpectedOutcome(String),
    #[error("action validation failed: {0}")]
    ValidationFailed(String),
    #[error("simulation mismatch: expected {expected}, got {actual}")]
    SimulationMismatch { expected: String, actual: String },
}

impl From<PolicyViolation> for ExecutionError {
    fn from(err: PolicyViolation) -> Self {
        ExecutionError::Policy(err)
    }
}

/// Execution mode for commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    /// Do not spawn real commands; return a simulated outcome.
    Simulated,
    /// Spawn real commands for allowed programs (use with care).
    Real,
}

/// Execution record for telemetry.
#[derive(Debug, Clone)]
pub struct ExecutionRecord {
    pub action: ActionIR,
    pub outcome: ActionOutcome,
    pub rollback: Option<RollbackStep>,
}

/// Rollback step for reversible actions.
#[derive(Debug, Clone)]
pub enum RollbackStep {
    RestoreFile { path: PathBuf, content: Vec<u8> },
    DeleteFile { path: PathBuf },
}

/// Outcome plus rollback info.
#[derive(Debug, Clone)]
pub struct ExecutionOutcome {
    pub action: ActionIR,
    pub outcome: ActionOutcome,
    pub rollback: Option<RollbackStep>,
}

/// Minimal executor that runs validated actions with telemetry and rollback hooks.
pub struct SimpleExecutor {
    mode: ExecutionMode,
    log: Vec<ExecutionRecord>,
    budget: BudgetTracker,
    pub dream_engine: DreamEngine<ActionIR>,
}

impl SimpleExecutor {
    /// Simulated executor (default, safe).
    pub fn new() -> Self {
        Self {
            mode: ExecutionMode::Simulated,
            log: Vec::new(),
            budget: BudgetTracker::new(),
            dream_engine: DreamEngine::new(DreamEngineConfig::default()),
        }
    }

    /// Enable real command execution for allowed programs.
    pub fn with_real_commands() -> Self {
        Self {
            mode: ExecutionMode::Real,
            log: Vec::new(),
            budget: BudgetTracker::new(),
            dream_engine: DreamEngine::new(DreamEngineConfig::default()),
        }
    }

    /// Construct executor from env flag: set SYMTHAEA_ALLOW_REAL_EXEC=1 to enable real commands.
    pub fn from_env() -> Self {
        match env::var("SYMTHAEA_ALLOW_REAL_EXEC") {
            Ok(val) if val == "1" => Self {
                mode: ExecutionMode::Real,
                log: Vec::new(),
                budget: BudgetTracker::new(),
                dream_engine: DreamEngine::new(DreamEngineConfig::default()),
            },
            _ => Self {
                mode: ExecutionMode::Simulated,
                log: Vec::new(),
                budget: BudgetTracker::new(),
                dream_engine: DreamEngine::new(DreamEngineConfig::default()),
            },
        }
    }

    /// Inspect current execution mode.
    pub fn mode(&self) -> ExecutionMode {
        self.mode
    }

    /// Access execution log (telemetry).
    pub fn telemetry(&self) -> &[ExecutionRecord] {
        &self.log
    }

    /// Roll back the last n reversible actions.
    pub fn rollback_last(&mut self, n: usize) -> Result<(), ExecutionError> {
        for _ in 0..n {
            if let Some(record) = self.log.pop() {
                if let Some(step) = record.rollback {
                    Self::apply_rollback(step)?;
                }
            }
        }
        Ok(())
    }

    /// Check if accumulated wisdom suggests a better action
    pub fn consult_wisdom(&self, _action: &ActionIR) -> Option<ActionIR> {
        let current_context = vec![0.0; 64];

        for w in self.dream_engine.wisdom() {
            if w.context_state == current_context {
                if w.confidence > 0.3 {
                    return Some(w.better_action.clone());
                } else {
                    println!(
                        "   (Wisdom found but confidence {:.2} <= 0.3)",
                        w.confidence
                    );
                }
            } else {
                println!(
                    "   (Wisdom found but context mismatch: len={})",
                    w.context_state.len()
                );
            }
        }
        None
    }

    pub fn execute(
        &mut self,
        action: &ActionIR,
        policy: &PolicyBundle,
        sandbox: &SandboxRoot,
        current_phi: f64,
    ) -> Result<ExecutionOutcome, ExecutionError> {
        // 0. Consult Wisdom (The "Pre-flight Check")
        // If the dream engine suggests a better path, we take it.
        let (final_action, wisdom_used) = if let Some(better_action) = self.consult_wisdom(action) {
            println!(
                "✨ Wisdom Intervention: Substituted dangerous action with safer alternative."
            );
            (better_action, true)
        } else {
            (action.clone(), false)
        };

        // 0.5. PRECOGNITION (The "Causal Veto")
        // Before executing, simulate the outcome in working memory.
        let state_dummy = vec![0.0; 64];
        let prediction = self
            .dream_engine
            .predict_outcome_distribution(&state_dummy, &final_action);

        if prediction.failure_probability > 0.8 {
            tracing::warn!(target: "symthaea::action", 
                "CAUSAL VETO: Simulation predicts {:.1}% failure probability. Aborting action.", 
                prediction.failure_probability * 100.0);
            return Err(ExecutionError::Unsupported(format!(
                "Causal Veto: Predicted failure probability {:.1}% is too high.",
                prediction.failure_probability * 100.0
            )));
        } else if prediction.failure_probability > 0.4 {
            tracing::info!(target: "symthaea::action", 
                "PRECOGNITION: Proceeding with caution. Predicted failure probability: {:.1}%", 
                prediction.failure_probability * 100.0);
        }

        // 1. Validate
        final_action.validate(policy, sandbox, current_phi)?;

        self.enforce_budgets(&final_action, policy)?;

        let rollback = Self::prepare_rollback(&final_action);

        let outcome: ActionOutcome = match &final_action {
            ActionIR::ReadFile { path, .. } => {
                let canonical = sandbox.validate(path).map_err(ExecutionError::Policy)?;
                let data = std::fs::read(&canonical)?;
                ActionOutcome::FileContent(data)
            }
            ActionIR::WriteFile {
                path,
                content,
                create_dirs,
            } => {
                let canonical = sandbox.validate(path).map_err(ExecutionError::Policy)?;
                if *create_dirs {
                    if let Some(parent) = canonical.parent() {
                        std::fs::create_dir_all(parent)?;
                    }
                }
                std::fs::write(&canonical, content)?;
                ActionOutcome::Success
            }
            ActionIR::DeleteFile { path } => {
                let canonical = sandbox.validate(path).map_err(ExecutionError::Policy)?;
                if canonical.exists() {
                    std::fs::remove_file(&canonical)?;
                }
                ActionOutcome::Success
            }
            ActionIR::CreateDirectory { path, recursive } => {
                let canonical = sandbox.validate(path).map_err(ExecutionError::Policy)?;
                if *recursive {
                    std::fs::create_dir_all(&canonical)?;
                } else {
                    std::fs::create_dir(&canonical)?;
                }
                ActionOutcome::Success
            }
            ActionIR::ListDirectory { path, recursive } => {
                let canonical = sandbox.validate(path).map_err(ExecutionError::Policy)?;
                let mut entries = Vec::new();
                if *recursive {
                    let mut stack = vec![canonical.clone()];
                    while let Some(dir) = stack.pop() {
                        for entry in std::fs::read_dir(&dir)? {
                            let e = entry?;
                            let p = e.path();
                            if p.is_dir() {
                                stack.push(p.clone());
                            }
                            entries.push(p);
                        }
                    }
                } else {
                    for entry in std::fs::read_dir(&canonical)? {
                        let e = entry?;
                        entries.push(e.path());
                    }
                }
                ActionOutcome::DirectoryListing(entries)
            }
            ActionIR::ReadSensor {
                sensor_id,
                channels,
            } => match self.mode {
                ExecutionMode::Simulated => {
                    let mut values = BTreeMap::new();
                    for chan in channels {
                        values.insert(chan.clone(), 0.0);
                    }
                    ActionOutcome::SensorData {
                        sensor_id: sensor_id.clone(),
                        values,
                    }
                }
                ExecutionMode::Real => {
                    // Placeholder for real HAL call
                    ActionOutcome::Success
                }
            },
            ActionIR::WriteServo { servo_id, value } => match self.mode {
                ExecutionMode::Simulated => ActionOutcome::ServoStatus {
                    servo_id: *servo_id,
                    current_value: *value,
                },
                ExecutionMode::Real => {
                    // Placeholder for real HAL call
                    ActionOutcome::Success
                }
            },
            ActionIR::SwarmGossip { topic, .. } => match self.mode {
                ExecutionMode::Simulated => {
                    println!("📢 Swarm Gossip: Broadcasted to topic '{}'", topic);
                    ActionOutcome::Success
                }
                ExecutionMode::Real => {
                    // Placeholder for real Swarm call (Iroh/Holochain)
                    ActionOutcome::Success
                }
            },
            ActionIR::WasmSandbox {
                module_path,
                function_name,
                input_data: _,
            } => match self.mode {
                ExecutionMode::Simulated => {
                    println!(
                        "🧪 WASM Sandbox: Simulating verification of module {:?}::{}()",
                        module_path, function_name
                    );
                    ActionOutcome::WasmResult {
                        output: vec![1], // 1 = success in our protocol
                        logs: vec!["Simulation: Verification successful.".to_string()],
                    }
                }
                ExecutionMode::Real => {
                    #[cfg(feature = "wasm-sandbox")]
                    {
                        use wasmtime::*;
                        let engine = Engine::default();
                        let module = Module::from_file(&engine, module_path)
                            .map_err(|e| ExecutionError::Unsupported(e.to_string()))?;
                        let mut store = Store::new(&engine, ());
                        let instance = Instance::new(&mut store, &module, &[])
                            .map_err(|e| ExecutionError::Unsupported(e.to_string()))?;
                        let func = instance
                            .get_typed_func::<(), i32>(&mut store, function_name)
                            .map_err(|e| ExecutionError::Unsupported(e.to_string()))?;
                        let res = func
                            .call(&mut store, ())
                            .map_err(|e| ExecutionError::Unsupported(e.to_string()))?;

                        ActionOutcome::WasmResult {
                            output: vec![res as u8],
                            logs: vec![format!(
                                "Wasmtime: Function {} returned {}",
                                function_name, res
                            )],
                        }
                    }
                    #[cfg(not(feature = "wasm-sandbox"))]
                    {
                        ActionOutcome::WasmResult {
                            output: vec![0],
                            logs: vec!["Error: wasm-sandbox feature not enabled.".to_string()],
                        }
                    }
                }
            },
            ActionIR::RunCommand {
                program,
                args,
                env,
                working_dir,
            } => match self.mode {
                ExecutionMode::Simulated => ActionOutcome::SimulatedCommand {
                    program: program.clone(),
                    args: args.clone(),
                },
                ExecutionMode::Real => {
                    let mut cmd = Command::new(program);
                    cmd.args(args);
                    cmd.envs(env);
                    if let Some(dir) = working_dir {
                        let canonical = sandbox.validate(dir).map_err(ExecutionError::Policy)?;
                        cmd.current_dir(canonical);
                    }
                    let output = cmd.output()?;
                    ActionOutcome::CommandOutput {
                        stdout: output.stdout,
                        stderr: output.stderr,
                        exit_code: output.status.code().unwrap_or(-1),
                    }
                }
            },
            ActionIR::Sequence(actions) => {
                let mut last = ActionOutcome::Success;
                for act in actions {
                    last = self.execute(act, policy, sandbox, current_phi)?.outcome;
                }
                last
            }
            ActionIR::NoOp => ActionOutcome::Success,
        };

        let record = ExecutionRecord {
            action: final_action.clone(),
            outcome: outcome.clone(),
            rollback,
        };
        self.log.push(record.clone());

        // Record to dream engine (learning from experience)
        // We use a dummy state because SimpleExecutor is stateless,
        // but this allows the DreamEngine to start accumulating action-outcome pairs.
        let state_dummy = vec![0.0; 64];
        let outcome_vec = self.vectorize_outcome(&outcome);
        // Surprise heuristic: 0.0 for success, 1.0 for error or dangerous simulated command
        let surprise = match &outcome {
            ActionOutcome::CommandOutput { exit_code, .. } if *exit_code != 0 => 1.0,
            ActionOutcome::SimulatedCommand { program, .. } if program == "rm" => 1.0,
            _ => 0.0,
        };
        // Only record if we didn't just use wisdom (prevent circular reinforcement)
        if !wisdom_used {
            self.dream_engine
                .record(&state_dummy, final_action.clone(), &outcome_vec, surprise);
        }

        Ok(ExecutionOutcome {
            action: record.action,
            outcome: record.outcome,
            rollback: record.rollback,
        })
    }

    fn vectorize_outcome(&self, outcome: &ActionOutcome) -> Vec<f32> {
        let score = match outcome {
            ActionOutcome::Success => 1.0,
            ActionOutcome::FileContent(_) => 0.9,
            ActionOutcome::DirectoryListing(_) => 0.8,
            ActionOutcome::SensorData { .. } => 0.85,
            ActionOutcome::ServoStatus { .. } => 0.7,
            ActionOutcome::SimulatedCommand { program, .. } => {
                if program == "ls" || program == "cat" || program == "nix" {
                    0.8
                } else if program == "rm" {
                    0.2
                } else {
                    0.5
                }
            }
            ActionOutcome::CommandOutput { exit_code, .. } => {
                if *exit_code == 0 {
                    1.0
                } else {
                    0.1
                }
            }
            ActionOutcome::WasmResult { output, .. } => {
                if !output.is_empty() && output[0] == 1 {
                    1.0
                } else {
                    0.2
                }
            }
        };
        vec![score; 64]
    }

    fn prepare_rollback(action: &ActionIR) -> Option<RollbackStep> {
        match action {
            ActionIR::WriteFile { path, .. } => {
                if path.exists() {
                    if let Ok(content) = std::fs::read(path) {
                        return Some(RollbackStep::RestoreFile {
                            path: path.clone(),
                            content,
                        });
                    }
                }
                None
            }
            ActionIR::DeleteFile { path } => {
                if path.exists() {
                    if let Ok(content) = std::fs::read(path) {
                        return Some(RollbackStep::RestoreFile {
                            path: path.clone(),
                            content,
                        });
                    }
                }
                None
            }
            _ => None,
        }
    }

    fn apply_rollback(step: RollbackStep) -> Result<(), ExecutionError> {
        match step {
            RollbackStep::RestoreFile { path, content } => {
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(path, content)?;
            }
            RollbackStep::DeleteFile { path } => {
                if path.exists() {
                    std::fs::remove_file(path)?;
                }
            }
        }
        Ok(())
    }

    fn enforce_budgets(
        &mut self,
        action: &ActionIR,
        policy: &PolicyBundle,
    ) -> Result<(), ExecutionError> {
        match action {
            ActionIR::RunCommand { .. } => {
                self.budget.check_shell_budget(policy)?;
                self.budget.record_shell_command();
            }
            ActionIR::WriteFile { content, .. } => {
                self.budget
                    .check_file_write_budget(1, content.len() as u64, policy)?;
                self.budget.record_file_write(content.len() as u64);
            }
            ActionIR::DeleteFile { .. } | ActionIR::CreateDirectory { .. } => {
                self.budget.check_file_write_budget(1, 0, policy)?;
                self.budget.record_file_write(0);
            }
            ActionIR::Sequence(_)
            | ActionIR::ReadFile { .. }
            | ActionIR::ListDirectory { .. }
            | ActionIR::ReadSensor { .. }
            | ActionIR::WriteServo { .. }
            | ActionIR::SwarmGossip { .. }
            | ActionIR::WasmSandbox { .. } => {}
            ActionIR::NoOp => {}
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct BudgetTracker {
    shell_commands_used: u32,
    shell_window_start: Instant,
    shell_window_used: u32,
    file_writes_used: u32,
    bytes_written: u64,
}

impl BudgetTracker {
    fn new() -> Self {
        Self {
            shell_commands_used: 0,
            shell_window_start: Instant::now(),
            shell_window_used: 0,
            file_writes_used: 0,
            bytes_written: 0,
        }
    }

    fn check_shell_budget(&mut self, policy: &PolicyBundle) -> Result<(), ExecutionError> {
        let allowed = policy.budgets.shell_commands_per_session;
        let attempted = self.shell_commands_used.saturating_add(1);
        if allowed > 0 && attempted > allowed {
            return Err(ExecutionError::Policy(
                PolicyViolation::ShellBudgetExceeded { allowed, attempted },
            ));
        }

        if self.shell_window_start.elapsed() >= Duration::from_secs(3600) {
            self.shell_window_start = Instant::now();
            self.shell_window_used = 0;
        }
        let hourly_allowed = policy.capabilities.shell.budget_per_hour;
        let hourly_attempted = self.shell_window_used.saturating_add(1);
        if hourly_allowed > 0 && hourly_attempted > hourly_allowed {
            return Err(ExecutionError::Policy(
                PolicyViolation::ShellHourlyBudgetExceeded {
                    allowed: hourly_allowed,
                    attempted: hourly_attempted,
                },
            ));
        }

        Ok(())
    }

    fn record_shell_command(&mut self) {
        self.shell_commands_used = self.shell_commands_used.saturating_add(1);
        self.shell_window_used = self.shell_window_used.saturating_add(1);
    }

    fn check_file_write_budget(
        &self,
        write_ops: u32,
        bytes: u64,
        policy: &PolicyBundle,
    ) -> Result<(), ExecutionError> {
        let allowed_ops = policy.budgets.file_writes_per_session;
        let attempted_ops = self.file_writes_used.saturating_add(write_ops);
        if allowed_ops > 0 && attempted_ops > allowed_ops {
            return Err(ExecutionError::Policy(
                PolicyViolation::FileWriteBudgetExceeded {
                    allowed: allowed_ops,
                    attempted: attempted_ops,
                },
            ));
        }

        let allowed_bytes = policy.budgets.bytes_written_per_session;
        let attempted_bytes = self.bytes_written.saturating_add(bytes);
        if allowed_bytes > 0 && attempted_bytes > allowed_bytes {
            return Err(ExecutionError::Policy(
                PolicyViolation::BytesWrittenBudgetExceeded {
                    allowed: allowed_bytes,
                    attempted: attempted_bytes,
                },
            ));
        }

        Ok(())
    }

    fn record_file_write(&mut self, bytes: u64) {
        self.file_writes_used = self.file_writes_used.saturating_add(1);
        self.bytes_written = self.bytes_written.saturating_add(bytes);
    }
}

// ============================================================================
// COMMAND DESTRUCTIVENESS CLASSIFICATION (Shell Sidecar Support)
// ============================================================================

/// Classify a shell command's destructiveness level for the sidecar confirmation flow.
/// This provides NixOS-aware classification that goes beyond simple pattern matching.
pub fn classify_command_destructiveness(program: &str, args: &[String]) -> DestructivenessLevel {
    let args_str = args.join(" ");
    let full_cmd = format!("{program} {args_str}");

    // Destructive commands (no rollback possible)
    let destructive_patterns = [
        // Garbage collection
        ("nix-collect-garbage", "-d"),
        ("nix", "store gc"),
        ("nix", "store delete"),
        // File destruction
        ("rm", "-rf"),
        ("rm", "-r"),
        ("shred", ""),
        ("wipefs", ""),
        // Disk operations
        ("dd", "of="),
        ("mkfs", ""),
        ("fdisk", ""),
        ("parted", ""),
        // Database drops
        ("dropdb", ""),
        ("drop database", ""),
    ];

    for (prog, pattern) in &destructive_patterns {
        if program.contains(prog) && (pattern.is_empty() || args_str.contains(pattern)) {
            return DestructivenessLevel::Destructive;
        }
    }

    // NeedsConfirmation commands (have rollback via generations/snapshots)
    let confirmation_patterns = [
        // NixOS system changes (rollback via generations)
        ("nixos-rebuild", "switch"),
        ("nixos-rebuild", "boot"),
        ("nixos-rebuild", "test"),
        // Service management
        ("systemctl", "restart"),
        ("systemctl", "stop"),
        ("systemctl", "disable"),
        // Package profile changes
        ("nix", "profile install"),
        ("nix", "profile remove"),
        ("nix-env", "-e"),
        ("nix-env", "--uninstall"),
        // Network configuration
        ("nmcli", "connection delete"),
        ("ip", "link set"),
    ];

    for (prog, pattern) in &confirmation_patterns {
        if program.contains(prog) && args_str.contains(pattern) {
            return DestructivenessLevel::NeedsConfirmation;
        }
    }

    // Reversible commands (can be undone)
    let reversible_patterns = [
        // Package installation (can uninstall)
        ("nix-env", "-i"),
        ("nix-env", "--install"),
        ("nix", "profile install"),
        // File modifications (can restore)
        ("cp", ""),
        ("mv", ""),
        ("chmod", ""),
        ("chown", ""),
        // Git operations
        ("git", "commit"),
        ("git", "push"),
    ];

    for (prog, pattern) in &reversible_patterns {
        if program.contains(prog) && (pattern.is_empty() || args_str.contains(pattern)) {
            return DestructivenessLevel::Reversible;
        }
    }

    // Read-only commands
    let readonly_patterns = [
        // Nix queries
        ("nix", "search"),
        ("nix", "show"),
        ("nix", "flake show"),
        ("nix", "flake metadata"),
        ("nix", "eval"),
        ("nix-env", "-q"),
        ("nix-env", "--query"),
        // File reading
        ("cat", ""),
        ("less", ""),
        ("head", ""),
        ("tail", ""),
        ("ls", ""),
        ("find", ""),
        // Text output (no filesystem/state effect at all -- already grouped with
        // cat/ls/head/tail as ReadOnly in classify_remote_command_capability above,
        // this list had just never been updated to match)
        ("echo", ""),
        ("grep", ""),
        ("rg", ""),
        // System info
        ("systemctl", "status"),
        ("journalctl", ""),
        ("uname", ""),
        ("hostname", ""),
        ("df", ""),
        ("free", ""),
        ("ps", ""),
        ("top", ""),
        ("htop", ""),
    ];

    for (prog, pattern) in &readonly_patterns {
        if program.contains(prog) && (pattern.is_empty() || args_str.contains(pattern)) {
            return DestructivenessLevel::ReadOnly;
        }
    }

    // Default: if unknown, be cautious and require confirmation
    // This is the safe default for the shell sidecar
    if full_cmd.contains("sudo") || full_cmd.contains("doas") {
        DestructivenessLevel::NeedsConfirmation
    } else {
        DestructivenessLevel::NeedsConfirmation
    }
}

/// Parse a command line into a program and arguments without invoking a shell.
///
/// Shell control operators and expansion syntax are rejected up front so callers can safely
/// execute the returned program/args pair with `std::process::Command`.
pub fn parse_command_line(command: &str) -> Result<(String, Vec<String>), String> {
    let trimmed = command.trim();
    if trimmed.is_empty() {
        return Err("command is empty".to_string());
    }

    const DISALLOWED_CHARS: [char; 10] = ['|', '&', ';', '<', '>', '$', '`', '(', ')', '\n'];
    if trimmed.contains('\r') || trimmed.chars().any(|c| DISALLOWED_CHARS.contains(&c)) {
        return Err("shell metacharacters are not allowed".to_string());
    }

    let mut lexer = shlex::Shlex::new(trimmed);
    let parts: Vec<String> = lexer.by_ref().collect();
    if lexer.had_error {
        return Err("command contains invalid shell quoting".to_string());
    }

    let mut iter = parts.into_iter();
    let program = iter.next().ok_or_else(|| "command is empty".to_string())?;
    let args: Vec<String> = iter.collect();

    Ok((program, args))
}

/// Get rollback hint for a command to display in the shell sidecar
pub fn get_rollback_hint(program: &str, args: &[String]) -> Option<String> {
    let args_str = args.join(" ");

    // NixOS-specific rollback hints
    if program.contains("nixos-rebuild")
        && (args_str.contains("switch") || args_str.contains("boot"))
    {
        return Some("nixos-rebuild switch --rollback".to_string());
    }

    if program.contains("nix-env") && (args_str.contains("-i") || args_str.contains("--install")) {
        // Extract package name from args
        let pkg = args.last().map(|s| s.as_str()).unwrap_or("PACKAGE");
        return Some(format!("nix-env -e {pkg}"));
    }

    if program.contains("nix") && args_str.contains("profile install") {
        return Some("nix profile rollback".to_string());
    }

    if program.contains("systemctl") {
        if args_str.contains("stop") {
            let service = args.last().map(|s| s.as_str()).unwrap_or("SERVICE");
            return Some(format!("systemctl start {service}"));
        }
        if args_str.contains("disable") {
            let service = args.last().map(|s| s.as_str()).unwrap_or("SERVICE");
            return Some(format!("systemctl enable {service}"));
        }
    }

    if program.contains("nix-collect-garbage")
        || (program.contains("nix") && args_str.contains("gc"))
    {
        return Some("WARNING: Garbage collection cannot be undone. Old generations will be permanently deleted.".to_string());
    }

    None
}

#[cfg(test)]
mod remote_command_capability_tests {
    use super::{RemoteCommandCapability, classify_remote_command_capability};

    #[test]
    fn test_remote_command_capability_allows_read_only_git_status() {
        let capability = classify_remote_command_capability("git", &["status".into()])
            .expect("git status should be allowed");
        assert_eq!(capability, RemoteCommandCapability::ReadOnly);
    }

    #[test]
    fn test_remote_command_capability_marks_mutating_touch() {
        let capability = classify_remote_command_capability("touch", &["/tmp/file".into()])
            .expect("touch should be recognized");
        assert_eq!(capability, RemoteCommandCapability::Mutating);
    }

    #[test]
    fn test_remote_command_capability_rejects_unknown_program() {
        let err = classify_remote_command_capability("python", &["-c".into(), "print(1)".into()])
            .expect_err("python should not be allowlisted");
        assert!(err.contains("allowlist"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restrictive_policy_allows_sandbox_reads() {
        let policy = PolicyBundle::restrictive();
        let sandbox = SandboxRoot::new("test_restrictive").unwrap();
        let path = sandbox.root().join("file.txt");
        std::fs::write(&path, b"ok").unwrap();

        let action = ActionIR::ReadFile {
            path: path.clone(),
            encoding: None,
        };
        assert!(action.validate(&policy, &sandbox, 1.0).is_ok());
    }

    #[test]
    fn restrictive_policy_blocks_outside_paths() {
        let policy = PolicyBundle::restrictive();
        let sandbox = SandboxRoot::new("test_block").unwrap();
        let action = ActionIR::ReadFile {
            path: PathBuf::from("/etc/passwd"),
            encoding: None,
        };
        let err = action.validate(&policy, &sandbox, 1.0).unwrap_err();
        matches!(
            err,
            PolicyViolation::SandboxEscape(_) | PolicyViolation::ReadNotAllowed(_)
        );
    }

    #[test]
    fn command_validation_respects_allowlist() {
        let policy = PolicyBundle::restrictive();
        let sandbox = SandboxRoot::new("test_cmd").unwrap();

        let allowed = ActionIR::RunCommand {
            program: "nix".into(),
            args: vec![],
            env: BTreeMap::new(),
            working_dir: None,
        };
        assert!(allowed.validate(&policy, &sandbox, 1.0).is_ok());

        let blocked = ActionIR::RunCommand {
            program: "rm".into(),
            args: vec!["-rf".into(), "/".into()],
            env: BTreeMap::new(),
            working_dir: None,
        };
        assert!(blocked.validate(&policy, &sandbox, 1.0).is_err());
    }

    #[test]
    fn command_validation_blocks_env_overrides() {
        let policy = PolicyBundle::restrictive();
        let sandbox = SandboxRoot::new("test_env").unwrap();

        let mut env = BTreeMap::new();
        env.insert("PATH".to_string(), "/tmp/symthaea/bin".to_string());
        let action = ActionIR::RunCommand {
            program: "echo".into(),
            args: vec!["hi".into()],
            env,
            working_dir: None,
        };
        assert!(action.validate(&policy, &sandbox, 1.0).is_err());
    }

    #[test]
    fn validate_allows_nested_paths_with_missing_parents() {
        let policy = PolicyBundle::restrictive();
        let sandbox = SandboxRoot::new("test_nested").unwrap();

        let nested_path = sandbox.root().join("deep/nested/path.txt");
        let action = ActionIR::WriteFile {
            path: nested_path,
            content: b"ok".to_vec(),
            create_dirs: true,
        };
        assert!(action.validate(&policy, &sandbox, 1.0).is_ok());
    }

    #[test]
    fn validate_blocks_parent_traversal() {
        let policy = PolicyBundle::restrictive();
        let sandbox = SandboxRoot::new("test_traversal").unwrap();

        let escaped = sandbox.root().join("dir/../../etc/passwd");
        let action = ActionIR::ReadFile {
            path: escaped,
            encoding: None,
        };
        assert!(action.validate(&policy, &sandbox, 1.0).is_err());
    }

    #[test]
    fn simple_executor_read_write_roundtrip() -> Result<(), ActionError> {
        let policy = PolicyBundle::restrictive();
        let sandbox = SandboxRoot::new("test_exec_rw")
            .map_err(|e| ActionError::ValidationFailed(e.to_string()))?;
        let path = sandbox.root().join("note.txt");
        let mut executor = SimpleExecutor::new();

        let write = ActionIR::WriteFile {
            path: path.clone(),
            content: b"hello".to_vec(),
            create_dirs: true,
        };
        executor
            .execute(&write, &policy, &sandbox, 1.0)
            .map_err(|e| ActionError::ValidationFailed(e.to_string()))?;

        let read = ActionIR::ReadFile {
            path: path.clone(),
            encoding: None,
        };
        let result = executor
            .execute(&read, &policy, &sandbox, 1.0)
            .map_err(|e| ActionError::ValidationFailed(e.to_string()))?;
        match result.outcome {
            ActionOutcome::FileContent(data) => {
                assert_eq!(data, b"hello");
                Ok(())
            }
            other => Err(ActionError::UnexpectedOutcome(format!("{:?}", other))),
        }
    }

    #[test]
    fn simple_executor_simulates_commands() -> Result<(), ActionError> {
        let policy = PolicyBundle::restrictive();
        let sandbox = SandboxRoot::new("test_exec_cmd")
            .map_err(|e| ActionError::ValidationFailed(e.to_string()))?;
        let mut executor = SimpleExecutor::new();

        let action = ActionIR::RunCommand {
            program: "nix".into(),
            args: vec!["search".into(), "nixpkgs".into(), "vim".into()],
            env: BTreeMap::new(),
            working_dir: None,
        };

        let outcome = executor
            .execute(&action, &policy, &sandbox, 1.0)
            .map_err(|e| ActionError::ValidationFailed(e.to_string()))?;
        match outcome.outcome {
            ActionOutcome::SimulatedCommand { program, args } => {
                assert_eq!(program, "nix");
                assert!(args.contains(&"vim".to_string()));
                Ok(())
            }
            other => Err(ActionError::SimulationMismatch {
                expected: "SimulatedCommand".to_string(),
                actual: format!("{:?}", other),
            }),
        }
    }

    #[test]
    fn simple_executor_enforces_shell_budget() {
        let mut policy = PolicyBundle::restrictive();
        policy.budgets.shell_commands_per_session = 1;
        let sandbox = SandboxRoot::new("test_exec_budget").unwrap();
        let mut executor = SimpleExecutor::new();

        let action = ActionIR::RunCommand {
            program: "echo".into(),
            args: vec!["first".into()],
            env: BTreeMap::new(),
            working_dir: None,
        };

        let _ = executor.execute(&action, &policy, &sandbox, 1.0);
        let err = executor
            .execute(&action, &policy, &sandbox, 1.0)
            .unwrap_err();
        assert!(matches!(
            err,
            ExecutionError::Policy(PolicyViolation::ShellBudgetExceeded { .. })
        ));
    }
}
