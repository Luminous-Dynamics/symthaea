// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Remote command execution for Sovereign RDP support sessions.
//!
//! Consciousness-gated + MFDI-gated + command allowlist/denylist.
//! All output privacy-scrubbed. All commands logged to session recording.

use std::collections::HashSet;
use std::io::Read;
use std::process::Stdio;
use std::thread;
use std::time::Duration;
use tracing::{info, warn};
use wait_timeout::ChildExt;

use super::rdp_protocol_ext::RemoteExecMessage;
use super::rdp_session::SessionPermission;

/// Dangerous commands that require elevated identity (E3+).
const DESTRUCTIVE_COMMANDS: &[&str] = &[
    "rm",
    "rmdir",
    "dd",
    "mkfs",
    "format",
    "fdisk",
    "parted",
    "shred",
    "wipefs",
    "kill",
    "killall",
    "systemctl stop",
    "systemctl disable",
    "reboot",
    "shutdown",
    "halt",
    "poweroff",
];

/// Commands always blocked regardless of permission.
const BLOCKED_COMMANDS: &[&str] = &["curl | sh", "wget | sh", "eval", "exec"];

/// Gate for remote command execution.
pub struct RemoteExecGate {
    /// Command allowlist (if non-empty, only these are permitted).
    pub allowlist: HashSet<String>,
    /// Additional denylist entries.
    pub denylist: HashSet<String>,
    /// Minimum session permission for execution.
    pub min_permission: SessionPermission,
    /// Minimum consciousness phi for execution.
    pub min_phi: f32,
    /// Default command timeout.
    pub default_timeout: Duration,
}

impl Default for RemoteExecGate {
    fn default() -> Self {
        Self {
            allowlist: HashSet::new(),
            denylist: HashSet::new(),
            min_permission: SessionPermission::Full,
            min_phi: 0.3,
            default_timeout: Duration::from_secs(30),
        }
    }
}

/// Result of a permission check.
#[derive(Debug, Clone)]
pub enum ExecPermission {
    /// Command is allowed.
    Allowed,
    /// Command requires elevated identity (E3+).
    RequiresElevated { reason: String },
    /// Command is blocked.
    Blocked { reason: String },
}

impl RemoteExecGate {
    /// Check if a command is permitted.
    pub fn check_permission(
        &self,
        command: &str,
        session_permission: SessionPermission,
        phi: f32,
    ) -> ExecPermission {
        // Check session permission level.
        if session_permission < self.min_permission {
            return ExecPermission::Blocked {
                reason: format!(
                    "Session permission {:?} below required {:?}",
                    session_permission, self.min_permission
                ),
            };
        }

        // Check consciousness.
        if phi < self.min_phi {
            return ExecPermission::Blocked {
                reason: format!(
                    "Consciousness {:.2} below threshold {:.2}",
                    phi, self.min_phi
                ),
            };
        }

        let parsed = match crate::action::parse_command_line(command) {
            Ok(parsed) => parsed,
            Err(err) => {
                return ExecPermission::Blocked { reason: err };
            }
        };
        let (program, args) = parsed;
        let capability = match crate::action::classify_remote_command_capability(&program, &args) {
            Ok(capability) => capability,
            Err(err) => {
                return ExecPermission::Blocked { reason: err };
            }
        };
        let cmd_lower = format!("{} {}", program, args.join(" ")).to_lowercase();

        // Always-blocked commands.
        for blocked in BLOCKED_COMMANDS {
            if cmd_lower.contains(blocked) {
                return ExecPermission::Blocked {
                    reason: format!("Command pattern '{}' is always blocked", blocked),
                };
            }
        }

        // Custom denylist.
        for denied in &self.denylist {
            if cmd_lower.contains(&denied.to_lowercase()) {
                return ExecPermission::Blocked {
                    reason: format!("Command matches denylist entry '{}'", denied),
                };
            }
        }

        // Allowlist (if non-empty, only listed commands allowed).
        if !self.allowlist.is_empty() {
            let cmd_base = program.to_lowercase();
            if !self.allowlist.contains(&cmd_base) {
                return ExecPermission::Blocked {
                    reason: format!("Command '{}' not in allowlist", cmd_base),
                };
            }
        }

        if capability == crate::action::RemoteCommandCapability::Mutating {
            return ExecPermission::RequiresElevated {
                reason: "Mutating commands require elevated approval and are disabled by default"
                    .to_string(),
            };
        }

        // Destructive commands require elevated identity.
        for dangerous in DESTRUCTIVE_COMMANDS {
            if cmd_lower.starts_with(dangerous) || cmd_lower.contains(&format!(" {}", dangerous)) {
                return ExecPermission::RequiresElevated {
                    reason: format!(
                        "'{}' is a destructive command, requires E3+ identity",
                        dangerous
                    ),
                };
            }
        }

        ExecPermission::Allowed
    }

    /// Execute a command with timeout. Returns output.
    ///
    /// Caller is responsible for permission checking before calling this.
    pub fn execute(
        &self,
        command: &str,
        working_dir: Option<&str>,
        timeout: Option<Duration>,
    ) -> RemoteExecMessage {
        let timeout = timeout.unwrap_or(self.default_timeout);
        let exec_id = {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64
        };

        let (program, args) = match crate::action::parse_command_line(command) {
            Ok(parsed) => parsed,
            Err(err) => {
                warn!(target: "symthaea_rdp_audit", event = "command_parse_failed", detail = %err);
                return RemoteExecMessage::Output {
                    exec_id,
                    stdout: String::new(),
                    stderr: format!("Execution failed: {}", err),
                    exit_code: None,
                };
            }
        };
        match crate::action::classify_remote_command_capability(&program, &args) {
            Ok(crate::action::RemoteCommandCapability::ReadOnly) => {}
            Ok(crate::action::RemoteCommandCapability::Mutating) => {
                let reason =
                    "Mutating commands are not executable without elevated approval".to_string();
                warn!(
                    target: "symthaea_rdp_audit",
                    event = "command_blocked",
                    detail = %reason,
                    command = %command
                );
                return RemoteExecMessage::Output {
                    exec_id,
                    stdout: String::new(),
                    stderr: reason,
                    exit_code: None,
                };
            }
            Err(err) => {
                warn!(
                    target: "symthaea_rdp_audit",
                    event = "command_blocked",
                    detail = %err,
                    command = %command
                );
                return RemoteExecMessage::Output {
                    exec_id,
                    stdout: String::new(),
                    stderr: err,
                    exit_code: None,
                };
            }
        }
        info!(
            target: "symthaea_rdp_audit",
            event = "command_execute_attempt",
            command = %command
        );

        let mut cmd = std::process::Command::new(&program);
        cmd.args(&args);
        if let Some(dir) = working_dir {
            cmd.current_dir(dir);
        }

        let mut child = match cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn() {
            Ok(child) => child,
            Err(e) => {
                return RemoteExecMessage::Output {
                    exec_id,
                    stdout: String::new(),
                    stderr: format!("Execution failed: {}", e),
                    exit_code: None,
                };
            }
        };

        let stdout = match child.stdout.take() {
            Some(stdout) => stdout,
            None => {
                return RemoteExecMessage::Output {
                    exec_id,
                    stdout: String::new(),
                    stderr: "Execution failed: missing stdout pipe".to_string(),
                    exit_code: None,
                };
            }
        };
        let stderr = match child.stderr.take() {
            Some(stderr) => stderr,
            None => {
                return RemoteExecMessage::Output {
                    exec_id,
                    stdout: String::new(),
                    stderr: "Execution failed: missing stderr pipe".to_string(),
                    exit_code: None,
                };
            }
        };

        let stdout_handle = thread::spawn(move || {
            let mut stdout = stdout;
            let mut buf = Vec::new();
            let _ = stdout.read_to_end(&mut buf);
            buf
        });
        let stderr_handle = thread::spawn(move || {
            let mut stderr = stderr;
            let mut buf = Vec::new();
            let _ = stderr.read_to_end(&mut buf);
            buf
        });

        let status = match child.wait_timeout(timeout) {
            Ok(Some(status)) => status,
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = stdout_handle.join();
                let _ = stderr_handle.join();
                return RemoteExecMessage::Output {
                    exec_id,
                    stdout: String::new(),
                    stderr: format!("Execution timed out after {:?}", timeout),
                    exit_code: None,
                };
            }
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = stdout_handle.join();
                let _ = stderr_handle.join();
                return RemoteExecMessage::Output {
                    exec_id,
                    stdout: String::new(),
                    stderr: format!("Execution failed: {}", e),
                    exit_code: None,
                };
            }
        };

        let stdout = match stdout_handle.join() {
            Ok(buf) => String::from_utf8_lossy(&buf).to_string(),
            Err(_) => String::new(),
        };
        let stderr = match stderr_handle.join() {
            Ok(buf) => String::from_utf8_lossy(&buf).to_string(),
            Err(_) => String::new(),
        };

        // Truncate output to prevent abuse (1MB max per stream).
        let stdout = if stdout.len() > 1_048_576 {
            format!("{}... [truncated at 1MB]", &stdout[..1_048_576])
        } else {
            stdout
        };
        let stderr = if stderr.len() > 1_048_576 {
            format!("{}... [truncated at 1MB]", &stderr[..1_048_576])
        } else {
            stderr
        };

        RemoteExecMessage::Output {
            exec_id,
            stdout,
            stderr,
            exit_code: status.code(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_command_allowed() {
        let gate = RemoteExecGate::default();
        let result = gate.check_permission("ls -la", SessionPermission::Full, 0.5);
        assert!(matches!(result, ExecPermission::Allowed));
    }

    #[test]
    fn test_destructive_requires_elevated() {
        let gate = RemoteExecGate::default();
        let result = gate.check_permission("rm -rf /tmp/test", SessionPermission::Full, 0.5);
        assert!(matches!(result, ExecPermission::RequiresElevated { .. }));
    }

    #[test]
    fn test_blocked_command() {
        let gate = RemoteExecGate::default();
        // BLOCKED_COMMANDS checks substring containment: "curl | sh" must appear literally.
        let result = gate.check_permission("curl | sh", SessionPermission::Full, 0.5);
        assert!(matches!(result, ExecPermission::Blocked { .. }));
    }

    #[test]
    fn test_shell_metacharacters_blocked() {
        let gate = RemoteExecGate::default();
        let result = gate.check_permission("ls; rm -rf /tmp/test", SessionPermission::Full, 0.5);
        assert!(matches!(result, ExecPermission::Blocked { .. }));
    }

    #[test]
    fn test_low_consciousness_blocks() {
        let gate = RemoteExecGate::default();
        let result = gate.check_permission("ls", SessionPermission::Full, 0.1);
        assert!(matches!(result, ExecPermission::Blocked { .. }));
    }

    #[test]
    fn test_view_only_blocks() {
        let gate = RemoteExecGate::default();
        let result = gate.check_permission("ls", SessionPermission::ViewOnly, 0.5);
        assert!(matches!(result, ExecPermission::Blocked { .. }));
    }

    #[test]
    fn test_allowlist_enforcement() {
        let mut gate = RemoteExecGate::default();
        gate.allowlist.insert("ls".to_string());
        gate.allowlist.insert("cat".to_string());

        let allowed = gate.check_permission("ls -la", SessionPermission::Full, 0.5);
        assert!(matches!(allowed, ExecPermission::Allowed));

        let blocked = gate.check_permission("whoami", SessionPermission::Full, 0.5);
        assert!(matches!(blocked, ExecPermission::Blocked { .. }));
    }

    #[test]
    fn test_execute_simple_command() {
        let gate = RemoteExecGate::default();
        let result = gate.execute("echo hello", None, None);
        match result {
            RemoteExecMessage::Output {
                stdout, exit_code, ..
            } => {
                assert!(stdout.contains("hello"));
                assert_eq!(exit_code, Some(0));
            }
            _ => panic!("Expected Output"),
        }
    }

    #[test]
    fn test_execute_timeout() {
        let gate = RemoteExecGate::default();
        let result = gate.execute("sleep 1", None, Some(Duration::from_millis(10)));
        match result {
            RemoteExecMessage::Output {
                stderr, exit_code, ..
            } => {
                assert!(stderr.contains("timed out"));
                assert_eq!(exit_code, None);
            }
            _ => panic!("Expected Output"),
        }
    }

    #[test]
    fn test_mutating_command_requires_elevated() {
        let gate = RemoteExecGate::default();
        let result = gate.check_permission("touch /tmp/file", SessionPermission::Full, 0.5);
        assert!(matches!(result, ExecPermission::RequiresElevated { .. }));
    }

    #[test]
    fn test_unknown_program_blocked() {
        let gate = RemoteExecGate::default();
        let result = gate.check_permission("python -c 'print(1)'", SessionPermission::Full, 0.5);
        assert!(matches!(result, ExecPermission::Blocked { .. }));
    }
}
