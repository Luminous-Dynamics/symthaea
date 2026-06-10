// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Systemd Service Management Actions
//!
//! Wraps `systemctl` commands for service lifecycle management.
//! All operations produce `NixOSCommand` values routed through
//! the Φ-gated executor — the service manager itself does NOT
//! execute commands directly.

use super::executor::{NixOSCommand, SafetyLevel};
use std::process::Command;

/// Manages systemd services: start, stop, restart, enable, disable.
pub struct ServiceManager;

/// Result of a service query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceStatus {
    /// Unit name.
    pub name: String,
    /// Whether the service is currently running.
    pub active: bool,
    /// Whether the service is enabled (starts on boot).
    pub enabled: bool,
    /// Active state string (e.g. "active", "inactive", "failed").
    pub active_state: String,
    /// Sub-state string (e.g. "running", "dead", "exited").
    pub sub_state: String,
}

impl ServiceManager {
    /// Generate a command to start a service.
    pub fn start(service: &str) -> NixOSCommand {
        NixOSCommand::Custom {
            command: "systemctl".to_string(),
            args: vec!["start".to_string(), Self::normalize_name(service)],
            safety_level: SafetyLevel::SystemModify,
        }
    }

    /// Generate a command to stop a service.
    pub fn stop(service: &str) -> NixOSCommand {
        NixOSCommand::Custom {
            command: "systemctl".to_string(),
            args: vec!["stop".to_string(), Self::normalize_name(service)],
            safety_level: SafetyLevel::SystemModify,
        }
    }

    /// Generate a command to restart a service.
    pub fn restart(service: &str) -> NixOSCommand {
        NixOSCommand::Custom {
            command: "systemctl".to_string(),
            args: vec!["restart".to_string(), Self::normalize_name(service)],
            safety_level: SafetyLevel::SystemModify,
        }
    }

    /// Generate a command to reload a service (without full restart).
    pub fn reload(service: &str) -> NixOSCommand {
        NixOSCommand::Custom {
            command: "systemctl".to_string(),
            args: vec!["reload".to_string(), Self::normalize_name(service)],
            safety_level: SafetyLevel::SystemModify,
        }
    }

    /// Generate a command to enable a service (start on boot).
    ///
    /// Note: on NixOS this is typically done declaratively. This is for
    /// imperative service management or user services.
    pub fn enable(service: &str) -> NixOSCommand {
        NixOSCommand::Custom {
            command: "systemctl".to_string(),
            args: vec!["enable".to_string(), Self::normalize_name(service)],
            safety_level: SafetyLevel::SystemModify,
        }
    }

    /// Generate a command to disable a service.
    pub fn disable(service: &str) -> NixOSCommand {
        NixOSCommand::Custom {
            command: "systemctl".to_string(),
            args: vec!["disable".to_string(), Self::normalize_name(service)],
            safety_level: SafetyLevel::SystemModify,
        }
    }

    /// Query the current status of a service (read-only, runs directly).
    pub fn status(service: &str) -> Result<ServiceStatus, std::io::Error> {
        let unit = Self::normalize_name(service);

        let output = Command::new("systemctl")
            .args([
                "show",
                &unit,
                "--no-pager",
                "--property=ActiveState,SubState,UnitFileState",
            ])
            .output()?;

        if !output.status.success() {
            return Err(std::io::Error::other(format!(
                "systemctl show failed for '{}': {}",
                unit,
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut active_state = String::new();
        let mut sub_state = String::new();
        let mut enabled = false;
        let mut parsed_fields = 0u32;

        for line in stdout.lines() {
            if let Some((key, value)) = line.split_once('=') {
                match key {
                    "ActiveState" => {
                        active_state = value.to_string();
                        parsed_fields += 1;
                    }
                    "SubState" => {
                        sub_state = value.to_string();
                        parsed_fields += 1;
                    }
                    "UnitFileState" => {
                        enabled = value == "enabled";
                        parsed_fields += 1;
                    }
                    _ => {}
                }
            }
        }

        if parsed_fields == 0 {
            return Err(std::io::Error::other(format!(
                "systemctl show returned no parseable properties for '{}'",
                unit
            )));
        }

        Ok(ServiceStatus {
            name: unit,
            active: active_state == "active",
            enabled,
            active_state,
            sub_state,
        })
    }

    /// Check if a service is running (read-only, runs directly).
    pub fn is_active(service: &str) -> Result<bool, std::io::Error> {
        let output = Command::new("systemctl")
            .args(["is-active", "--quiet", &Self::normalize_name(service)])
            .status()?;
        Ok(output.success())
    }

    /// Ensure service name ends with ".service" if no suffix given.
    fn normalize_name(service: &str) -> String {
        if service.contains('.') {
            service.to_string()
        } else {
            format!("{service}.service")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_name() {
        assert_eq!(ServiceManager::normalize_name("nginx"), "nginx.service");
        assert_eq!(
            ServiceManager::normalize_name("nginx.service"),
            "nginx.service"
        );
        assert_eq!(ServiceManager::normalize_name("foo.socket"), "foo.socket");
    }

    #[test]
    fn test_start_command() {
        let cmd = ServiceManager::start("nginx");
        let (bin, args) = cmd.to_command();
        assert_eq!(bin, "systemctl");
        assert_eq!(args, vec!["start", "nginx.service"]);
        assert_eq!(cmd.safety_level(), SafetyLevel::SystemModify);
    }

    #[test]
    fn test_stop_command() {
        let cmd = ServiceManager::stop("nginx");
        let (bin, args) = cmd.to_command();
        assert_eq!(bin, "systemctl");
        assert_eq!(args, vec!["stop", "nginx.service"]);
    }

    #[test]
    fn test_restart_command() {
        let cmd = ServiceManager::restart("postgresql");
        let (bin, args) = cmd.to_command();
        assert_eq!(bin, "systemctl");
        assert_eq!(args, vec!["restart", "postgresql.service"]);
    }

    #[test]
    fn test_enable_command() {
        let cmd = ServiceManager::enable("sshd");
        let (bin, args) = cmd.to_command();
        assert_eq!(bin, "systemctl");
        assert_eq!(args, vec!["enable", "sshd.service"]);
    }

    #[test]
    fn test_disable_command() {
        let cmd = ServiceManager::disable("sshd");
        let (bin, args) = cmd.to_command();
        assert_eq!(bin, "systemctl");
        assert_eq!(args, vec!["disable", "sshd.service"]);
    }

    #[test]
    fn test_reload_command() {
        let cmd = ServiceManager::reload("nginx");
        let (bin, args) = cmd.to_command();
        assert_eq!(bin, "systemctl");
        assert_eq!(args, vec!["reload", "nginx.service"]);
    }

    #[test]
    fn test_normalize_preserves_socket_suffix() {
        assert_eq!(ServiceManager::normalize_name("cups.socket"), "cups.socket");
        assert_eq!(
            ServiceManager::normalize_name("sshd.service"),
            "sshd.service"
        );
        assert_eq!(ServiceManager::normalize_name("tmp.mount"), "tmp.mount");
        assert_eq!(
            ServiceManager::normalize_name("fstrim.timer"),
            "fstrim.timer"
        );
    }

    #[test]
    fn test_all_commands_safety_level() {
        let cmds = [
            ServiceManager::start("nginx"),
            ServiceManager::stop("nginx"),
            ServiceManager::restart("nginx"),
            ServiceManager::reload("nginx"),
            ServiceManager::enable("nginx"),
            ServiceManager::disable("nginx"),
        ];
        for cmd in &cmds {
            assert_eq!(
                cmd.safety_level(),
                SafetyLevel::SystemModify,
                "All service lifecycle commands should be SystemModify"
            );
        }
    }

    #[test]
    fn test_commands_use_normalized_names() {
        let cmd = ServiceManager::stop("docker");
        let (_, args) = cmd.to_command();
        assert_eq!(args[1], "docker.service");

        let cmd = ServiceManager::stop("docker.socket");
        let (_, args) = cmd.to_command();
        assert_eq!(args[1], "docker.socket");
    }

    #[test]
    fn test_service_status_struct() {
        let status = ServiceStatus {
            name: "nginx.service".to_string(),
            active: true,
            enabled: true,
            active_state: "active".to_string(),
            sub_state: "running".to_string(),
        };
        assert!(status.active);
        assert!(status.enabled);
        assert_eq!(status.active_state, "active");

        // Inactive service
        let status = ServiceStatus {
            name: "stopped.service".to_string(),
            active: false,
            enabled: false,
            active_state: "inactive".to_string(),
            sub_state: "dead".to_string(),
        };
        assert!(!status.active);
        assert!(!status.enabled);
    }
}
