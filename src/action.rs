//! Action-level safety primitives: declarative policy, sandboxed paths, and a safe action IR.
//!
//! These types are intentionally minimal so they can be integrated without pulling in heavy
//! executors. They model the Phase 1/2 security scaffolding from the v1.2 plan.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::path::{Path, PathBuf};
use thiserror::Error;
use std::process::Command;

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
                    allowed_programs: ["nix", "ls", "cat", "echo"].iter().map(|s| s.to_string()).collect(),
                    blocked_programs: BTreeSet::new(),
                    budget_per_hour: 100,
                    allowed_env: BTreeMap::new(),
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShellCapabilities {
    pub allowed_programs: BTreeSet<String>,
    pub blocked_programs: BTreeSet<String>,
    pub budget_per_hour: u32,
    pub allowed_env: BTreeMap<String, String>,
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

    /// Validate a path is inside the sandbox after canonicalization.
    pub fn validate(&self, requested: &Path) -> Result<PathBuf, PolicyViolation> {
        if !requested.is_absolute() {
            return Err(PolicyViolation::SandboxEscape(format!(
                "path must be absolute: {}",
                requested.display()
            )));
        }

        let canonical = if requested.exists() {
            requested
                .canonicalize()
                .map_err(|e| PolicyViolation::SandboxEscape(e.to_string()))?
        } else if let Some(parent) = requested.parent() {
            parent
                .canonicalize()
                .map_err(|e| PolicyViolation::SandboxEscape(e.to_string()))?
                .join(requested.file_name().unwrap_or_default())
        } else {
            return Err(PolicyViolation::SandboxEscape(format!(
                "cannot canonicalize {}",
                requested.display()
            )));
        };

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
    ReadFile { path: PathBuf, encoding: Option<String> },
    WriteFile { path: PathBuf, content: Vec<u8>, create_dirs: bool },
    DeleteFile { path: PathBuf },
    CreateDirectory { path: PathBuf, recursive: bool },
    ListDirectory { path: PathBuf, recursive: bool },
    RunCommand {
        program: String,
        args: Vec<String>,
        env: BTreeMap<String, String>,
        working_dir: Option<PathBuf>,
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
            ActionIR::Sequence(actions) => actions.iter().all(|a| a.is_reversible()),
            ActionIR::RunCommand { .. } => false,
        }
    }

    /// Classify risk tier for budgeting/logging.
    pub fn risk_tier(&self) -> RiskTier {
        match self {
            ActionIR::ReadFile { .. } | ActionIR::ListDirectory { .. } | ActionIR::NoOp => RiskTier::Low,
            ActionIR::WriteFile { .. }
            | ActionIR::DeleteFile { .. }
            | ActionIR::CreateDirectory { .. } => RiskTier::Medium,
            ActionIR::RunCommand { .. } => RiskTier::High,
            ActionIR::Sequence(actions) => actions
                .iter()
                .map(|a| a.risk_tier())
                .max()
                .unwrap_or(RiskTier::Low),
        }
    }

    /// Validate against policy and sandbox.
    pub fn validate(&self, policy: &PolicyBundle, sandbox: &SandboxRoot) -> Result<(), PolicyViolation> {
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
            ActionIR::RunCommand { program, .. } => {
                if policy.capabilities.shell.blocked_programs.contains(program) {
                    return Err(PolicyViolation::ProgramBlocked(program.clone()));
                }
                if !policy.capabilities.shell.allowed_programs.contains(program) {
                    return Err(PolicyViolation::ProgramNotAllowed(program.clone()));
                }
            }
            ActionIR::Sequence(actions) => {
                for action in actions {
                    action.validate(policy, sandbox)?;
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

/// Validation errors.
#[derive(Debug, Clone)]
pub enum PolicyViolation {
    SandboxEscape(String),
    ReadNotAllowed(PathBuf),
    WriteNotAllowed(PathBuf),
    WriteTooLarge(usize),
    ProgramBlocked(String),
    ProgramNotAllowed(String),
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

/// Outcome of executing an action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionOutcome {
    Success,
    FileContent(Vec<u8>),
    DirectoryListing(Vec<PathBuf>),
    SimulatedCommand { program: String, args: Vec<String> },
    CommandOutput { stdout: Vec<u8>, stderr: Vec<u8>, exit_code: i32 },
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
}

impl SimpleExecutor {
    /// Simulated executor (default, safe).
    pub fn new() -> Self {
        Self { mode: ExecutionMode::Simulated, log: Vec::new() }
    }

    /// Enable real command execution for allowed programs.
    pub fn with_real_commands() -> Self {
        Self { mode: ExecutionMode::Real, log: Vec::new() }
    }

    /// Construct executor from env flag: set SYMTHAEA_ALLOW_REAL_EXEC=1 to enable real commands.
    pub fn from_env() -> Self {
        match env::var("SYMTHAEA_ALLOW_REAL_EXEC") {
            Ok(val) if val == "1" => Self { mode: ExecutionMode::Real, log: Vec::new() },
            _ => Self { mode: ExecutionMode::Simulated, log: Vec::new() },
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

    pub fn execute(
        &mut self,
        action: &ActionIR,
        policy: &PolicyBundle,
        sandbox: &SandboxRoot,
    ) -> Result<ExecutionOutcome, ExecutionError> {
        // Validate first
        action.validate(policy, sandbox)?;

        let rollback = Self::prepare_rollback(action);

        let outcome: ActionOutcome = match action {
            ActionIR::ReadFile { path, .. } => {
                let data = std::fs::read(path)?;
                ActionOutcome::FileContent(data)
            }
            ActionIR::WriteFile { path, content, create_dirs } => {
                if *create_dirs {
                    if let Some(parent) = path.parent() {
                        std::fs::create_dir_all(parent)?;
                    }
                }
                std::fs::write(path, content)?;
                ActionOutcome::Success
            }
            ActionIR::DeleteFile { path } => {
                if path.exists() {
                    std::fs::remove_file(path)?;
                }
                ActionOutcome::Success
            }
            ActionIR::CreateDirectory { path, recursive } => {
                if *recursive {
                    std::fs::create_dir_all(path)?;
                } else {
                    std::fs::create_dir(path)?;
                }
                ActionOutcome::Success
            }
            ActionIR::ListDirectory { path, recursive } => {
                let mut entries = Vec::new();
                if *recursive {
                    let mut stack = vec![path.clone()];
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
                    for entry in std::fs::read_dir(path)? {
                        let e = entry?;
                        entries.push(e.path());
                    }
                }
                ActionOutcome::DirectoryListing(entries)
            }
            ActionIR::RunCommand { program, args, env, working_dir } => {
                match self.mode {
                    ExecutionMode::Simulated => ActionOutcome::SimulatedCommand {
                        program: program.clone(),
                        args: args.clone(),
                    },
                    ExecutionMode::Real => {
                        let mut cmd = Command::new(program);
                        cmd.args(args);
                        cmd.envs(env);
                        if let Some(dir) = working_dir {
                            cmd.current_dir(dir);
                        }
                        let output = cmd.output()?;
                        ActionOutcome::CommandOutput {
                            stdout: output.stdout,
                            stderr: output.stderr,
                            exit_code: output.status.code().unwrap_or(-1),
                        }
                    }
                }
            }
            ActionIR::Sequence(actions) => {
                let mut last = ActionOutcome::Success;
                for act in actions {
                    last = self.execute(act, policy, sandbox)?.outcome;
                }
                last
            }
            ActionIR::NoOp => ActionOutcome::Success,
        };

        let record = ExecutionRecord {
            action: action.clone(),
            outcome: outcome.clone(),
            rollback,
        };
        self.log.push(record.clone());

        Ok(ExecutionOutcome {
            action: record.action,
            outcome: record.outcome,
            rollback: record.rollback,
        })
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

        let action = ActionIR::ReadFile { path: path.clone(), encoding: None };
        assert!(action.validate(&policy, &sandbox).is_ok());
    }

    #[test]
    fn restrictive_policy_blocks_outside_paths() {
        let policy = PolicyBundle::restrictive();
        let sandbox = SandboxRoot::new("test_block").unwrap();
        let action = ActionIR::ReadFile { path: PathBuf::from("/etc/passwd"), encoding: None };
        let err = action.validate(&policy, &sandbox).unwrap_err();
        matches!(err, PolicyViolation::SandboxEscape(_) | PolicyViolation::ReadNotAllowed(_));
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
        assert!(allowed.validate(&policy, &sandbox).is_ok());

        let blocked = ActionIR::RunCommand {
            program: "rm".into(),
            args: vec!["-rf".into(), "/".into()],
            env: BTreeMap::new(),
            working_dir: None,
        };
        assert!(blocked.validate(&policy, &sandbox).is_err());
    }

    #[test]
    fn simple_executor_read_write_roundtrip() {
        let policy = PolicyBundle::restrictive();
        let sandbox = SandboxRoot::new("test_exec_rw").unwrap();
        let path = sandbox.root().join("note.txt");
        let mut executor = SimpleExecutor::new();

        let write = ActionIR::WriteFile {
            path: path.clone(),
            content: b"hello".to_vec(),
            create_dirs: true,
        };
        executor.execute(&write, &policy, &sandbox).expect("write should succeed");

        let read = ActionIR::ReadFile { path: path.clone(), encoding: None };
        let result = executor.execute(&read, &policy, &sandbox).unwrap();
        match result.outcome {
            ActionOutcome::FileContent(data) => assert_eq!(data, b"hello"),
            other => panic!("unexpected outcome: {:?}", other),
        }
    }

    #[test]
    fn simple_executor_simulates_commands() {
        let policy = PolicyBundle::restrictive();
        let sandbox = SandboxRoot::new("test_exec_cmd").unwrap();
        let mut executor = SimpleExecutor::new();

        let action = ActionIR::RunCommand {
            program: "nix".into(),
            args: vec!["search".into(), "nixpkgs".into(), "vim".into()],
            env: BTreeMap::new(),
            working_dir: None,
        };

        let outcome = executor.execute(&action, &policy, &sandbox).unwrap();
        match outcome.outcome {
            ActionOutcome::SimulatedCommand { program, args } => {
                assert_eq!(program, "nix");
                assert!(args.contains(&"vim".to_string()));
            }
            other => panic!("expected simulated command, got {:?}", other),
        }
    }
}
