// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! K3: Dry-Run Sandbox
//!
//! Provides scoped command testing without affecting the actual system by default.
//! This is not a security boundary; real command execution is opt-in.

use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

/// Sandbox for isolated command execution
pub struct Sandbox {
    /// Sandbox root directory
    root: PathBuf,
    /// Environment variables
    env: HashMap<String, String>,
    /// Allowed commands
    allowed_commands: Vec<String>,
    /// Timeout for sandbox operations
    timeout: Duration,
    /// Whether sandbox is initialized
    initialized: bool,
    /// Captured outputs
    outputs: Vec<SandboxOutput>,
    /// Simulation mode (no real execution)
    simulation_only: bool,
    /// Require explicit opt-in before running real commands
    real_execution_enabled: bool,
}

impl Sandbox {
    /// Create a new sandbox
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let root =
            std::env::temp_dir().join(format!("symthaea-sandbox-{}-{}", std::process::id(), id));

        Self {
            root,
            env: HashMap::new(),
            allowed_commands: default_allowed_commands(),
            timeout: Duration::from_secs(60),
            initialized: false,
            outputs: Vec::new(),
            simulation_only: false,
            real_execution_enabled: false,
        }
    }

    /// Create with custom root
    pub fn with_root(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            env: HashMap::new(),
            allowed_commands: default_allowed_commands(),
            timeout: Duration::from_secs(60),
            initialized: false,
            outputs: Vec::new(),
            simulation_only: false,
            real_execution_enabled: false,
        }
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Enable simulation-only mode (no real execution)
    pub fn simulation_only(mut self) -> Self {
        self.simulation_only = true;
        self
    }

    /// Explicitly allow real command execution (not a security boundary).
    pub fn enable_real_execution(mut self) -> Self {
        self.real_execution_enabled = true;
        self
    }

    /// Add allowed command
    pub fn allow_command(&mut self, cmd: impl Into<String>) {
        self.allowed_commands.push(cmd.into());
    }

    /// Whether this sandbox is configured for simulation-only execution.
    pub fn is_simulation_only(&self) -> bool {
        self.simulation_only
    }

    /// Whether real execution has been explicitly enabled.
    pub fn is_real_execution_enabled(&self) -> bool {
        self.real_execution_enabled
    }

    /// Set environment variable
    pub fn set_env(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.env.insert(key.into(), value.into());
    }

    /// Initialize the sandbox
    pub fn init(&mut self) -> Result<(), SandboxError> {
        if self.initialized {
            return Ok(());
        }

        // Create sandbox directory structure
        fs::create_dir_all(&self.root).map_err(|e| SandboxError::InitFailed(e.to_string()))?;

        fs::create_dir_all(self.root.join("etc"))
            .map_err(|e| SandboxError::InitFailed(e.to_string()))?;

        fs::create_dir_all(self.root.join("nix/store"))
            .map_err(|e| SandboxError::InitFailed(e.to_string()))?;

        fs::create_dir_all(self.root.join("tmp"))
            .map_err(|e| SandboxError::InitFailed(e.to_string()))?;

        // Create minimal /etc/nixos for dry-run
        let nixos_dir = self.root.join("etc/nixos");
        fs::create_dir_all(&nixos_dir).map_err(|e| SandboxError::InitFailed(e.to_string()))?;

        self.initialized = true;
        Ok(())
    }

    /// Run a command in the sandbox
    pub fn run(&mut self, command: &str, args: &[&str]) -> Result<SandboxResult, SandboxError> {
        if !self.initialized {
            self.init()?;
        }

        // Check if command is allowed
        if !self.is_command_allowed(command) {
            return Err(SandboxError::CommandNotAllowed(command.to_string()));
        }

        let start = Instant::now();

        // In simulation mode, return simulated output
        if self.simulation_only {
            return Ok(self.simulate_command(command, args));
        }

        if !self.real_execution_enabled {
            return Err(SandboxError::RealExecutionDisabled);
        }

        let result = self.run_command_with_timeout(command, args, start)?;

        self.outputs.push(SandboxOutput {
            result: result.clone(),
            timestamp: Instant::now(),
        });

        Ok(result)
    }

    /// Run a NixOS dry-run rebuild
    pub fn nixos_dry_run(&mut self, config_path: &Path) -> Result<SandboxResult, SandboxError> {
        if !self.initialized {
            self.init()?;
        }

        // Copy config to sandbox
        let sandbox_config = self.root.join("etc/nixos/configuration.nix");
        fs::copy(config_path, &sandbox_config)
            .map_err(|e| SandboxError::ExecutionFailed(e.to_string()))?;

        if self.simulation_only {
            return Ok(SandboxResult {
                command: format!("nixos-rebuild dry-build -I nixos-config={}", sandbox_config.display()),
                exit_code: 0,
                stdout: "[Simulated] Would build configuration:\n  - No errors detected\n  - Estimated build time: ~2 minutes".to_string(),
                stderr: String::new(),
                elapsed: Duration::from_millis(100),
                simulated: true,
            });
        }

        if !self.real_execution_enabled {
            return Err(SandboxError::RealExecutionDisabled);
        }

        // Run nixos-rebuild dry-build
        let start = Instant::now();

        self.run_command_with_timeout(
            "nixos-rebuild",
            &[
                "dry-build",
                "-I",
                &format!("nixos-config={}", sandbox_config.display()),
            ],
            start,
        )
    }

    /// Evaluate a Nix expression in sandbox
    pub fn nix_eval(&mut self, expr: &str) -> Result<SandboxResult, SandboxError> {
        if self.simulation_only {
            return Ok(SandboxResult {
                command: format!("nix-instantiate --eval -E '{expr}'"),
                exit_code: 0,
                stdout: "[Simulated] Nix expression valid".to_string(),
                stderr: String::new(),
                elapsed: Duration::from_millis(50),
                simulated: true,
            });
        }

        self.run("nix-instantiate", &["--eval", "-E", expr])
    }

    /// Check Nix syntax
    pub fn nix_syntax_check(&mut self, file: &Path) -> Result<SandboxResult, SandboxError> {
        if self.simulation_only {
            return Ok(SandboxResult {
                command: format!("nix-instantiate --parse {}", file.display()),
                exit_code: 0,
                stdout: "[Simulated] Syntax OK".to_string(),
                stderr: String::new(),
                elapsed: Duration::from_millis(30),
                simulated: true,
            });
        }

        self.run("nix-instantiate", &["--parse", &file.to_string_lossy()])
    }

    fn run_command_with_timeout(
        &self,
        command: &str,
        args: &[&str],
        start: Instant,
    ) -> Result<SandboxResult, SandboxError> {
        let mut child = Command::new(command)
            .args(args)
            .current_dir(&self.root)
            .envs(&self.env)
            .env("HOME", self.root.join("home"))
            .env("TMPDIR", self.root.join("tmp"))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| SandboxError::ExecutionFailed(e.to_string()))?;

        let mut stdout = child
            .stdout
            .take()
            .ok_or_else(|| SandboxError::ExecutionFailed("Missing stdout pipe".to_string()))?;
        let mut stderr = child
            .stderr
            .take()
            .ok_or_else(|| SandboxError::ExecutionFailed("Missing stderr pipe".to_string()))?;

        let stdout_handle = thread::spawn(move || {
            let mut stdout_buf = Vec::new();
            let _ = stdout.read_to_end(&mut stdout_buf);
            stdout_buf
        });
        let stderr_handle = thread::spawn(move || {
            let mut stderr_buf = Vec::new();
            let _ = stderr.read_to_end(&mut stderr_buf);
            stderr_buf
        });

        // Avoid signal-handler based timeout machinery here; some sandboxed
        // environments reject the internal wakeup fd writes used by
        // `wait-timeout`, which turns an ordinary child wait into a process
        // abort. A small polling loop is slower but robust and sufficient for
        // these short-lived verification commands.
        let poll_interval = Duration::from_millis(10);
        let deadline = start + self.timeout;
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) => {
                    if Instant::now() >= deadline {
                        let _ = child.kill();
                        let _ = child.wait();
                        let _ = stdout_handle.join();
                        let _ = stderr_handle.join();
                        return Err(SandboxError::Timeout);
                    }
                    thread::sleep(poll_interval);
                }
                Err(e) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = stdout_handle.join();
                    let _ = stderr_handle.join();
                    return Err(SandboxError::ExecutionFailed(e.to_string()));
                }
            }
        };

        let stdout_buf = stdout_handle
            .join()
            .map_err(|_| SandboxError::ExecutionFailed("stdout reader panicked".to_string()))?;
        let stderr_buf = stderr_handle
            .join()
            .map_err(|_| SandboxError::ExecutionFailed("stderr reader panicked".to_string()))?;

        Ok(SandboxResult {
            command: format!("{} {}", command, args.join(" ")),
            exit_code: status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&stdout_buf).to_string(),
            stderr: String::from_utf8_lossy(&stderr_buf).to_string(),
            elapsed: start.elapsed(),
            simulated: false,
        })
    }

    /// Get all sandbox outputs
    pub fn outputs(&self) -> &[SandboxOutput] {
        &self.outputs
    }

    /// Clear sandbox outputs
    pub fn clear_outputs(&mut self) {
        self.outputs.clear();
    }

    /// Clean up sandbox
    pub fn cleanup(&mut self) -> Result<(), SandboxError> {
        if self.root.exists() {
            fs::remove_dir_all(&self.root)
                .map_err(|e| SandboxError::CleanupFailed(e.to_string()))?;
        }
        self.initialized = false;
        Ok(())
    }

    /// Get sandbox root path
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn is_command_allowed(&self, command: &str) -> bool {
        // Extract base command name
        let base = Path::new(command)
            .file_name()
            .map(|s| s.to_string_lossy())
            .unwrap_or_default();

        self.allowed_commands
            .iter()
            .any(|c| c == command || c == base.as_ref())
    }

    fn simulate_command(&self, command: &str, args: &[&str]) -> SandboxResult {
        let full_cmd = format!("{} {}", command, args.join(" "));

        // Simulate common commands
        let (exit_code, stdout, stderr) = match command {
            "nix" | "nix-build" | "nix-shell" | "nix-env" => (
                0,
                format!("[Simulated] {full_cmd} would execute successfully"),
                String::new(),
            ),
            "nixos-rebuild" => {
                if args.contains(&"dry-build") || args.contains(&"dry-run") {
                    (
                        0,
                        "[Simulated] Dry run successful\n  - Configuration valid\n  - No errors"
                            .to_string(),
                        String::new(),
                    )
                } else {
                    (
                        0,
                        "[Simulated] Build would succeed".to_string(),
                        String::new(),
                    )
                }
            }
            "nix-instantiate" => {
                if args.contains(&"--parse") {
                    (0, "[Simulated] Syntax OK".to_string(), String::new())
                } else if args.contains(&"--eval") {
                    (0, "[Simulated] Expression valid".to_string(), String::new())
                } else {
                    (0, format!("[Simulated] {full_cmd}"), String::new())
                }
            }
            _ => (
                0,
                format!("[Simulated] {full_cmd} completed"),
                String::new(),
            ),
        };

        SandboxResult {
            command: full_cmd,
            exit_code,
            stdout,
            stderr,
            elapsed: Duration::from_millis(10),
            simulated: true,
        }
    }
}

impl Default for Sandbox {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        // Best-effort cleanup
        let _ = self.cleanup();
    }
}

/// Result of a sandbox operation
#[derive(Debug, Clone)]
pub struct SandboxResult {
    /// Command that was run
    pub command: String,
    /// Exit code
    pub exit_code: i32,
    /// Standard output
    pub stdout: String,
    /// Standard error
    pub stderr: String,
    /// Execution time
    pub elapsed: Duration,
    /// Whether this was simulated
    pub simulated: bool,
}

impl SandboxResult {
    /// Check if command succeeded
    pub fn success(&self) -> bool {
        self.exit_code == 0
    }

    /// Get combined output
    pub fn combined_output(&self) -> String {
        if self.stderr.is_empty() {
            self.stdout.clone()
        } else if self.stdout.is_empty() {
            self.stderr.clone()
        } else {
            format!("{}\n{}", self.stdout, self.stderr)
        }
    }
}

/// Recorded sandbox output
#[derive(Debug, Clone)]
pub struct SandboxOutput {
    /// The result
    pub result: SandboxResult,
    /// When it was recorded
    pub timestamp: Instant,
}

/// Sandbox errors
#[derive(Debug, Clone)]
pub enum SandboxError {
    /// Sandbox initialization failed
    InitFailed(String),
    /// Command not allowed in sandbox
    CommandNotAllowed(String),
    /// Command execution failed
    ExecutionFailed(String),
    /// Real execution is disabled (simulation only)
    RealExecutionDisabled,
    /// Timeout exceeded
    Timeout,
    /// Cleanup failed
    CleanupFailed(String),
}

impl std::fmt::Display for SandboxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InitFailed(e) => write!(f, "Sandbox init failed: {e}"),
            Self::CommandNotAllowed(cmd) => write!(f, "Command not allowed: {cmd}"),
            Self::ExecutionFailed(e) => write!(f, "Execution failed: {e}"),
            Self::RealExecutionDisabled => write!(f, "Real execution disabled"),
            Self::Timeout => write!(f, "Sandbox operation timed out"),
            Self::CleanupFailed(e) => write!(f, "Cleanup failed: {e}"),
        }
    }
}

impl std::error::Error for SandboxError {}

/// Default allowed commands in sandbox
fn default_allowed_commands() -> Vec<String> {
    vec![
        // Nix commands
        "nix".to_string(),
        "nix-build".to_string(),
        "nix-shell".to_string(),
        "nix-env".to_string(),
        "nix-instantiate".to_string(),
        "nix-store".to_string(),
        "nixos-rebuild".to_string(),
        "nixos-option".to_string(),
        // Safe system commands
        "cat".to_string(),
        "ls".to_string(),
        "head".to_string(),
        "tail".to_string(),
        "grep".to_string(),
        "find".to_string(),
        "echo".to_string(),
        "true".to_string(),
        "false".to_string(),
    ]
}

/// Sandbox configuration builder
pub struct SandboxConfig {
    root: Option<PathBuf>,
    timeout: Duration,
    allowed_commands: Vec<String>,
    simulation_only: bool,
    real_execution_enabled: bool,
}

impl SandboxConfig {
    pub fn new() -> Self {
        Self {
            root: None,
            timeout: Duration::from_secs(60),
            allowed_commands: default_allowed_commands(),
            simulation_only: false,
            real_execution_enabled: false,
        }
    }

    pub fn root(mut self, path: impl Into<PathBuf>) -> Self {
        self.root = Some(path.into());
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn allow(mut self, command: impl Into<String>) -> Self {
        self.allowed_commands.push(command.into());
        self
    }

    pub fn simulation_only(mut self) -> Self {
        self.simulation_only = true;
        self
    }

    pub fn enable_real_execution(mut self) -> Self {
        self.real_execution_enabled = true;
        self
    }

    pub fn build(self) -> Sandbox {
        let mut sandbox = if let Some(root) = self.root {
            Sandbox::with_root(root)
        } else {
            Sandbox::new()
        };

        sandbox.timeout = self.timeout;
        sandbox.allowed_commands = self.allowed_commands;
        sandbox.simulation_only = self.simulation_only;
        sandbox.real_execution_enabled = self.real_execution_enabled;

        sandbox
    }
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sandbox_creation() {
        let sandbox = Sandbox::new();
        assert!(!sandbox.initialized);
        assert!(!sandbox.allowed_commands.is_empty());
    }

    #[test]
    fn test_simulation_mode() {
        let mut sandbox = Sandbox::new().simulation_only();

        let result = sandbox.run("nix-build", &["--version"]).unwrap();
        assert!(result.simulated);
        assert!(result.success());
        assert!(result.stdout.contains("[Simulated]"));
    }

    #[test]
    fn test_command_allowlist() {
        let sandbox = Sandbox::new();

        assert!(sandbox.is_command_allowed("nix"));
        assert!(sandbox.is_command_allowed("nix-build"));
        assert!(sandbox.is_command_allowed("ls"));
        assert!(!sandbox.is_command_allowed("rm"));
        assert!(!sandbox.is_command_allowed("dd"));
    }

    #[test]
    fn test_sandbox_config_builder() {
        let sandbox = SandboxConfig::new()
            .timeout(Duration::from_secs(30))
            .simulation_only()
            .allow("custom-cmd")
            .build();

        assert!(sandbox.simulation_only);
        assert!(sandbox.allowed_commands.contains(&"custom-cmd".to_string()));
    }

    #[test]
    fn test_real_execution_disabled_by_default() {
        let mut sandbox = Sandbox::new();
        let err = sandbox.run("nix-build", &["--version"]).unwrap_err();
        assert!(matches!(err, SandboxError::RealExecutionDisabled));
    }

    #[test]
    fn test_nixos_dry_run_simulation() {
        let mut sandbox = Sandbox::new().simulation_only();
        sandbox.init().unwrap();

        // Create a dummy config
        let config_path = sandbox.root().join("test-config.nix");
        fs::write(&config_path, "{ }").unwrap();

        let result = sandbox.nixos_dry_run(&config_path).unwrap();
        assert!(result.simulated);
        assert!(result.success());
    }

    // ---- SandboxError variant construction and Display/Debug tests ----

    #[test]
    fn test_error_init_failed_display() {
        let err = SandboxError::InitFailed("permission denied".to_string());
        let msg = format!("{}", err);
        assert!(!msg.is_empty());
        assert!(msg.contains("Sandbox init failed"));
        assert!(msg.contains("permission denied"));
    }

    #[test]
    fn test_error_command_not_allowed_display() {
        let err = SandboxError::CommandNotAllowed("rm".to_string());
        let msg = format!("{}", err);
        assert!(!msg.is_empty());
        assert!(msg.contains("Command not allowed"));
        assert!(msg.contains("rm"));
    }

    #[test]
    fn test_error_execution_failed_display() {
        let err = SandboxError::ExecutionFailed("segfault".to_string());
        let msg = format!("{}", err);
        assert!(!msg.is_empty());
        assert!(msg.contains("Execution failed"));
        assert!(msg.contains("segfault"));
    }

    #[test]
    fn test_error_real_execution_disabled_display() {
        let err = SandboxError::RealExecutionDisabled;
        let msg = format!("{}", err);
        assert!(!msg.is_empty());
        assert!(msg.contains("Real execution disabled"));
    }

    #[test]
    fn test_error_timeout_display() {
        let err = SandboxError::Timeout;
        let msg = format!("{}", err);
        assert!(!msg.is_empty());
        assert!(msg.contains("timed out"));
    }

    #[test]
    fn test_error_cleanup_failed_display() {
        let err = SandboxError::CleanupFailed("device busy".to_string());
        let msg = format!("{}", err);
        assert!(!msg.is_empty());
        assert!(msg.contains("Cleanup failed"));
        assert!(msg.contains("device busy"));
    }

    #[test]
    fn test_error_debug_formatting_all_variants() {
        let variants: Vec<SandboxError> = vec![
            SandboxError::InitFailed("init reason".to_string()),
            SandboxError::CommandNotAllowed("dangerous-cmd".to_string()),
            SandboxError::ExecutionFailed("exec reason".to_string()),
            SandboxError::RealExecutionDisabled,
            SandboxError::Timeout,
            SandboxError::CleanupFailed("cleanup reason".to_string()),
        ];

        for err in &variants {
            let debug = format!("{:?}", err);
            assert!(!debug.is_empty(), "Debug output should be non-empty");
        }
    }

    #[test]
    fn test_error_clone_all_variants() {
        let variants: Vec<SandboxError> = vec![
            SandboxError::InitFailed("clone test".to_string()),
            SandboxError::CommandNotAllowed("clone test".to_string()),
            SandboxError::ExecutionFailed("clone test".to_string()),
            SandboxError::RealExecutionDisabled,
            SandboxError::Timeout,
            SandboxError::CleanupFailed("clone test".to_string()),
        ];

        for err in &variants {
            let cloned = err.clone();
            assert_eq!(format!("{}", err), format!("{}", cloned));
            assert_eq!(format!("{:?}", err), format!("{:?}", cloned));
        }
    }

    #[test]
    fn test_error_implements_std_error() {
        let err = SandboxError::Timeout;
        // Verify the std::error::Error trait is implemented
        let std_err: &dyn std::error::Error = &err;
        // source() should return None since there are no nested errors
        assert!(std_err.source().is_none());
        // Display via std::error::Error should match Display directly
        assert_eq!(format!("{}", std_err), format!("{}", err));
    }

    #[test]
    fn test_error_display_messages_are_distinct() {
        let init = format!("{}", SandboxError::InitFailed("x".to_string()));
        let not_allowed = format!("{}", SandboxError::CommandNotAllowed("x".to_string()));
        let exec = format!("{}", SandboxError::ExecutionFailed("x".to_string()));
        let disabled = format!("{}", SandboxError::RealExecutionDisabled);
        let timeout = format!("{}", SandboxError::Timeout);
        let cleanup = format!("{}", SandboxError::CleanupFailed("x".to_string()));

        // All error messages should be unique
        let messages = [&init, &not_allowed, &exec, &disabled, &timeout, &cleanup];
        for (i, a) in messages.iter().enumerate() {
            for (j, b) in messages.iter().enumerate() {
                if i != j {
                    assert_ne!(
                        a, b,
                        "Error messages for variants {} and {} should differ",
                        i, j
                    );
                }
            }
        }
    }

    #[test]
    fn test_error_command_not_allowed_triggered_by_run() {
        let mut sandbox = Sandbox::new().simulation_only();
        let err = sandbox.run("rm", &["-rf", "/"]).unwrap_err();
        assert!(matches!(err, SandboxError::CommandNotAllowed(ref cmd) if cmd == "rm"));
        let msg = format!("{}", err);
        assert!(msg.contains("rm"));
    }

    #[test]
    fn test_error_real_execution_disabled_triggered_by_run() {
        // Default sandbox (not simulation, not real-execution-enabled)
        let mut sandbox = Sandbox::new();
        let err = sandbox.run("nix", &["--version"]).unwrap_err();
        assert!(matches!(err, SandboxError::RealExecutionDisabled));
    }

    #[test]
    fn test_real_execution_handles_large_stdout() {
        let root = tempfile::tempdir().unwrap();
        let script = root.path().join("emit.sh");
        fs::write(
            &script,
            "#!/bin/sh\ncount=0\nwhile [ \"$count\" -lt 5000 ]; do\n  echo line-$count\n  count=$((count + 1))\ndone\n",
        )
        .unwrap();

        let mut sandbox = SandboxConfig::new()
            .root(root.path())
            .allow("sh")
            .enable_real_execution()
            .timeout(Duration::from_secs(5))
            .build();
        sandbox.init().unwrap();

        let result = sandbox.run("sh", &[script.to_str().unwrap()]).unwrap();
        assert!(result.success());
        assert!(result.stdout.contains("line-0"));
        assert!(result.stdout.contains("line-4999"));
    }

    #[test]
    fn test_error_with_empty_inner_message() {
        // Ensure variants with empty strings still produce non-empty Display output
        let variants = vec![
            SandboxError::InitFailed(String::new()),
            SandboxError::CommandNotAllowed(String::new()),
            SandboxError::ExecutionFailed(String::new()),
            SandboxError::CleanupFailed(String::new()),
        ];

        for err in &variants {
            let msg = format!("{}", err);
            assert!(
                !msg.is_empty(),
                "Display should be non-empty even with empty inner string"
            );
            // The prefix part of the message should still be present
            assert!(
                msg.len() > 5,
                "Display should contain a meaningful prefix: got '{}'",
                msg
            );
        }
    }
}
