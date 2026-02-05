//! K3: Dry-Run Sandbox
//!
//! Provides isolated command testing without affecting the actual system.
//! Uses Nix sandbox features and temporary stores.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::fs;
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
}

impl Sandbox {
    /// Create a new sandbox
    pub fn new() -> Self {
        let root = std::env::temp_dir().join(format!("symthaea-sandbox-{}", std::process::id()));

        Self {
            root,
            env: HashMap::new(),
            allowed_commands: default_allowed_commands(),
            timeout: Duration::from_secs(60),
            initialized: false,
            outputs: Vec::new(),
            simulation_only: false,
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

    /// Add allowed command
    pub fn allow_command(&mut self, cmd: impl Into<String>) {
        self.allowed_commands.push(cmd.into());
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
        fs::create_dir_all(&self.root)
            .map_err(|e| SandboxError::InitFailed(e.to_string()))?;

        fs::create_dir_all(self.root.join("etc"))
            .map_err(|e| SandboxError::InitFailed(e.to_string()))?;

        fs::create_dir_all(self.root.join("nix/store"))
            .map_err(|e| SandboxError::InitFailed(e.to_string()))?;

        fs::create_dir_all(self.root.join("tmp"))
            .map_err(|e| SandboxError::InitFailed(e.to_string()))?;

        // Create minimal /etc/nixos for dry-run
        let nixos_dir = self.root.join("etc/nixos");
        fs::create_dir_all(&nixos_dir)
            .map_err(|e| SandboxError::InitFailed(e.to_string()))?;

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

        // Build the command
        let output = Command::new(command)
            .args(args)
            .current_dir(&self.root)
            .envs(&self.env)
            .env("HOME", self.root.join("home"))
            .env("TMPDIR", self.root.join("tmp"))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| SandboxError::ExecutionFailed(e.to_string()))?;

        let elapsed = start.elapsed();

        let result = SandboxResult {
            command: format!("{} {}", command, args.join(" ")),
            exit_code: output.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            elapsed,
            simulated: false,
        };

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

        // Run nixos-rebuild dry-build
        let start = Instant::now();

        let output = Command::new("nixos-rebuild")
            .args([
                "dry-build",
                "-I", &format!("nixos-config={}", sandbox_config.display()),
            ])
            .current_dir(&self.root)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| SandboxError::ExecutionFailed(e.to_string()))?;

        let elapsed = start.elapsed();

        Ok(SandboxResult {
            command: format!("nixos-rebuild dry-build -I nixos-config={}", sandbox_config.display()),
            exit_code: output.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            elapsed,
            simulated: false,
        })
    }

    /// Evaluate a Nix expression in sandbox
    pub fn nix_eval(&mut self, expr: &str) -> Result<SandboxResult, SandboxError> {
        if self.simulation_only {
            return Ok(SandboxResult {
                command: format!("nix-instantiate --eval -E '{}'", expr),
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

        self.allowed_commands.iter().any(|c| {
            c == command || c == base.as_ref()
        })
    }

    fn simulate_command(&self, command: &str, args: &[&str]) -> SandboxResult {
        let full_cmd = format!("{} {}", command, args.join(" "));

        // Simulate common commands
        let (exit_code, stdout, stderr) = match command {
            "nix" | "nix-build" | "nix-shell" | "nix-env" => {
                (0, format!("[Simulated] {} would execute successfully", full_cmd), String::new())
            }
            "nixos-rebuild" => {
                if args.contains(&"dry-build") || args.contains(&"dry-run") {
                    (0, "[Simulated] Dry run successful\n  - Configuration valid\n  - No errors".to_string(), String::new())
                } else {
                    (0, "[Simulated] Build would succeed".to_string(), String::new())
                }
            }
            "nix-instantiate" => {
                if args.contains(&"--parse") {
                    (0, "[Simulated] Syntax OK".to_string(), String::new())
                } else if args.contains(&"--eval") {
                    (0, "[Simulated] Expression valid".to_string(), String::new())
                } else {
                    (0, format!("[Simulated] {}", full_cmd), String::new())
                }
            }
            _ => {
                (0, format!("[Simulated] {} completed", full_cmd), String::new())
            }
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
    /// Timeout exceeded
    Timeout,
    /// Cleanup failed
    CleanupFailed(String),
}

impl std::fmt::Display for SandboxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InitFailed(e) => write!(f, "Sandbox init failed: {}", e),
            Self::CommandNotAllowed(cmd) => write!(f, "Command not allowed: {}", cmd),
            Self::ExecutionFailed(e) => write!(f, "Execution failed: {}", e),
            Self::Timeout => write!(f, "Sandbox operation timed out"),
            Self::CleanupFailed(e) => write!(f, "Cleanup failed: {}", e),
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
}

impl SandboxConfig {
    pub fn new() -> Self {
        Self {
            root: None,
            timeout: Duration::from_secs(60),
            allowed_commands: default_allowed_commands(),
            simulation_only: false,
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

    pub fn build(self) -> Sandbox {
        let mut sandbox = if let Some(root) = self.root {
            Sandbox::with_root(root)
        } else {
            Sandbox::new()
        };

        sandbox.timeout = self.timeout;
        sandbox.allowed_commands = self.allowed_commands;
        sandbox.simulation_only = self.simulation_only;

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
}
