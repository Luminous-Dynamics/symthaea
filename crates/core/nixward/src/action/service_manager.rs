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
use super::service_state::{
    NixServiceEnablementEvidenceV1, NixServiceObservedStateV1, NixServiceOperationCapabilitiesV1,
};
use super::systemd_transport::observe_service_properties;

/// Manages systemd services: start, stop, restart, enable, disable.
pub struct ServiceManager;

/// Compatibility/diagnostic result of a service query.
///
/// This lossy representation is not governed pre-state evidence and must not
/// be used to authorize, bind, or execute a Nix service effect. Governed
/// pre-state uses `NixServiceObservedStateV1` instead.
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

    /// Query compatibility/diagnostic status of a service (read-only).
    ///
    /// This API intentionally remains lossy for existing CLI consumers. Its
    /// result is not valid governed pre-state and must not cross into the
    /// authorization/effect-binding path.
    pub fn status(service: &str) -> Result<ServiceStatus, std::io::Error> {
        let observed = Self::observed_state(service)?;

        let active_state = match observed.active_state() {
            super::service_state::ServiceActiveStateV1::Active => "active",
            super::service_state::ServiceActiveStateV1::Reloading => "reloading",
            super::service_state::ServiceActiveStateV1::Inactive => "inactive",
            super::service_state::ServiceActiveStateV1::Failed => "failed",
            super::service_state::ServiceActiveStateV1::Activating => "activating",
            super::service_state::ServiceActiveStateV1::Deactivating => "deactivating",
            super::service_state::ServiceActiveStateV1::Maintenance => "maintenance",
            super::service_state::ServiceActiveStateV1::Refreshing => "refreshing",
        };

        let enabled = matches!(
            observed.unit_file_state(),
            super::service_state::ServiceUnitFileStateV1::Enabled
        );

        Ok(ServiceStatus {
            name: observed.unit().to_string(),
            active: matches!(
                observed.active_state(),
                super::service_state::ServiceActiveStateV1::Active
            ),
            enabled,
            active_state: active_state.to_string(),
            sub_state: observed.sub_state().to_string(),
        })
    }

    /// Observe the exact governed systemd pre-state projection.
    ///
    /// Unlike the legacy status() compatibility API, this path never
    /// collapses UnitFileState to a boolean or accepts partial observations.
    /// It is evidence only; it does not authorize or execute an effect.
    pub fn observed_state(service: &str) -> Result<NixServiceObservedStateV1, std::io::Error> {
        let unit = Self::normalize_name(service);

        let properties = observe_service_properties(&unit)?;
        NixServiceObservedStateV1::parse_systemd_properties(&unit, &properties)
            .map_err(|error| std::io::Error::other(format!(
                "invalid governed systemd observation for '{}': {error}",
                unit
            )))
    }

    /// Observe unit-file enablement evidence from the same exact
    /// governed pre-state. This remains distinct from lifecycle capability
    /// facts and carries no operation authorization.
    pub fn observed_state_with_enablement_evidence(
        service: &str,
    ) -> Result<(NixServiceObservedStateV1, NixServiceEnablementEvidenceV1), std::io::Error> {
        let state = Self::observed_state(service)?;
        let evidence = NixServiceEnablementEvidenceV1::from_observed_state(&state)
            .map_err(|error| std::io::Error::other(format!(
                "invalid governed enablement evidence for '{}': {error}",
                state.unit()
            )))?;
        Ok((state, evidence))
    }

    /// Observe operation capability facts from the same atomic systemd
    /// property projection as the governed pre-state.
    ///
    /// These facts describe systemd's current capability surface; they are
    /// observational evidence only and never imply authorization.
    pub fn observed_state_with_capabilities(
        service: &str,
    ) -> Result<(NixServiceObservedStateV1, NixServiceOperationCapabilitiesV1), std::io::Error> {
        let unit = Self::normalize_name(service);
        let properties = observe_service_properties(&unit)?;
        NixServiceObservedStateV1::parse_systemd_observation(&unit, &properties)
            .map_err(|error| std::io::Error::other(format!(
                "invalid governed systemd observation for '{}': {error}",
                unit
            )))
    }

    /// Check if a service is active through the single governed observation path.
    ///
    /// This remains a compatibility/diagnostic boolean and is not governed
    /// pre-state evidence. Unknown or malformed observations now fail closed
    /// instead of collapsing to false.
    pub fn is_active(service: &str) -> Result<bool, std::io::Error> {
        let observed = Self::observed_state(service)?;
        Ok(matches!(
            observed.active_state(),
            super::service_state::ServiceActiveStateV1::Active
        ))
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
