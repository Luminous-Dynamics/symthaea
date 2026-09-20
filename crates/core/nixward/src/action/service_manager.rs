// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Systemd Service Management Actions
//!
//! Provides normalized typed lifecycle operations plus read-only status queries.
//! Lifecycle mutations are represented as structured `NixOSCommand::Service`
//! values and are executed by the normal Nixward execution/authority layers;
//! the service manager does not execute those mutations directly.

use super::executor::{
    NixOSCommand, ServiceOperation, validate_service_unit_name,
};
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
    fn lifecycle(operation: ServiceOperation, service: &str) -> NixOSCommand {
        NixOSCommand::Service {
            operation,
            unit: Self::normalize_name(service),
        }
    }

    /// Generate a typed command to start a service.
    pub fn start(service: &str) -> NixOSCommand {
        Self::lifecycle(ServiceOperation::Start, service)
    }

    /// Generate a typed command to stop a service.
    pub fn stop(service: &str) -> NixOSCommand {
        Self::lifecycle(ServiceOperation::Stop, service)
    }

    /// Generate a typed command to restart a service.
    pub fn restart(service: &str) -> NixOSCommand {
        Self::lifecycle(ServiceOperation::Restart, service)
    }

    /// Generate a typed command to reload a service (without full restart).
    pub fn reload(service: &str) -> NixOSCommand {
        Self::lifecycle(ServiceOperation::Reload, service)
    }

    /// Generate a typed command to enable a service (start on boot).
    ///
    /// Note: on NixOS this is typically done declaratively. This is for
    /// imperative service management or user services.
    pub fn enable(service: &str) -> NixOSCommand {
        Self::lifecycle(ServiceOperation::Enable, service)
    }

    /// Generate a typed command to disable a service.
    pub fn disable(service: &str) -> NixOSCommand {
        Self::lifecycle(ServiceOperation::Disable, service)
    }

    /// Query the current status of a service (read-only, runs directly).
    pub fn status(service: &str) -> Result<ServiceStatus, std::io::Error> {
        let unit = Self::validated_name(service)?;

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
        let unit = Self::validated_name(service)?;
        let output = Command::new("systemctl")
            .args(["is-active", "--quiet", &unit])
            .status()?;
        Ok(output.success())
    }

    /// Ensure service name ends with ".service" if no suffix is supplied.
    fn normalize_name(service: &str) -> String {
        if service.contains('.') {
            service.to_string()
        } else {
            format!("{service}.service")
        }
    }

    fn validated_name(service: &str) -> Result<String, std::io::Error> {
        let unit = Self::normalize_name(service);
        validate_service_unit_name(&unit)
            .map_err(|reason| std::io::Error::new(std::io::ErrorKind::InvalidInput, reason))?;
        Ok(unit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::executor::SafetyLevel;

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
        assert!(cmd.validate_shape().is_ok());
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
            assert!(cmd.validate_shape().is_ok());
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
    fn test_invalid_unit_is_rejected_by_typed_and_read_only_paths() {
        let cmd = ServiceManager::restart("nginx*.service");
        assert!(cmd.validate_shape().is_err());

        let err = ServiceManager::validated_name("--now.service").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);

        let err = ServiceManager::validated_name("foo/bar.service").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
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
