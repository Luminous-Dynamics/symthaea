// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! What-If Simulation with Dry-Run Preview
//!
//! Provides simulation of NixOS commands without actual execution:
//! - Analyzes what would change
//! - Shows package additions/removals
//! - Displays service changes
//! - Estimates rebuild time
//! - Warns about potential issues

use std::collections::HashSet;

/// Result of a what-if simulation
#[derive(Debug, Clone)]
pub struct WhatIfResult {
    /// Original command
    pub command: String,

    /// What the command would do (high-level summary)
    pub summary: String,

    /// Packages that would be added
    pub packages_added: Vec<String>,

    /// Packages that would be removed
    pub packages_removed: Vec<String>,

    /// Services that would be started/enabled
    pub services_enabled: Vec<String>,

    /// Services that would be stopped/disabled
    pub services_disabled: Vec<String>,

    /// Files that would be modified
    pub files_modified: Vec<String>,

    /// Estimated time for the operation
    pub estimated_time: Option<String>,

    /// Whether a reboot would be required
    pub reboot_required: bool,

    /// Warnings about the operation
    pub warnings: Vec<String>,

    /// The dry-run command to use
    pub dry_run_command: Option<String>,

    /// Confidence in the simulation (0.0 - 1.0)
    pub confidence: f64,

    /// Whether this is reversible
    pub reversible: bool,

    /// Rollback command if applicable
    pub rollback_command: Option<String>,
}

impl Default for WhatIfResult {
    fn default() -> Self {
        Self {
            command: String::new(),
            summary: String::new(),
            packages_added: Vec::new(),
            packages_removed: Vec::new(),
            services_enabled: Vec::new(),
            services_disabled: Vec::new(),
            files_modified: Vec::new(),
            estimated_time: None,
            reboot_required: false,
            warnings: Vec::new(),
            dry_run_command: None,
            confidence: 0.5,
            reversible: true,
            rollback_command: None,
        }
    }
}

/// What-if simulator
pub struct WhatIfSimulator {
    /// Known packages for validation
    known_packages: HashSet<String>,
    /// Known services for validation
    known_services: HashSet<String>,
}

impl Default for WhatIfSimulator {
    fn default() -> Self {
        Self::new()
    }
}

impl WhatIfSimulator {
    pub fn new() -> Self {
        Self {
            known_packages: HashSet::new(),
            known_services: Self::common_services(),
        }
    }

    /// Set known packages (from flake context)
    pub fn set_known_packages(&mut self, packages: HashSet<String>) {
        self.known_packages = packages;
    }

    /// Set known services (from flake context)
    pub fn set_known_services(&mut self, services: HashSet<String>) {
        self.known_services = services;
    }

    fn common_services() -> HashSet<String> {
        [
            "nginx",
            "postgresql",
            "mysql",
            "redis",
            "docker",
            "sshd",
            "openssh",
            "firewall",
            "fail2ban",
            "wireguard",
            "xserver",
            "pipewire",
            "pulseaudio",
            "networkmanager",
            "bluetooth",
            "printing",
            "cups",
            "avahi",
            "tailscale",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    /// Simulate a command and return what would happen
    pub fn simulate(&self, command: &str) -> WhatIfResult {
        let parts: Vec<&str> = command.split_whitespace().collect();
        let cmd = parts.first().copied().unwrap_or("");

        match cmd {
            "install" | "add" => self.simulate_install(command, &parts[1..]),
            "remove" | "uninstall" => self.simulate_remove(command, &parts[1..]),
            "enable" => self.simulate_enable(command, &parts[1..]),
            "disable" => self.simulate_disable(command, &parts[1..]),
            "nixos-rebuild" => self.simulate_rebuild(command, &parts[1..]),
            "nix-collect-garbage" => self.simulate_garbage_collect(command, &parts[1..]),
            "nix" => self.simulate_nix_command(command, &parts[1..]),
            _ => self.simulate_generic(command),
        }
    }

    fn simulate_install(&self, command: &str, args: &[&str]) -> WhatIfResult {
        let packages: Vec<String> = args
            .iter()
            .filter(|a| !a.starts_with('-'))
            .map(|s| s.to_string())
            .collect();

        let mut warnings = Vec::new();

        // Check if any packages are already installed
        for pkg in &packages {
            if self.known_packages.contains(pkg) {
                warnings.push(format!("{pkg} is already installed"));
            }
        }

        WhatIfResult {
            command: command.to_string(),
            summary: format!("Install {} package(s)", packages.len()),
            packages_added: packages.clone(),
            estimated_time: Some(self.estimate_install_time(packages.len())),
            warnings,
            dry_run_command: Some(format!(
                "nix-env -iA nixpkgs.{} --dry-run",
                packages.join(" nixpkgs.")
            )),
            confidence: 0.9,
            reversible: true,
            rollback_command: Some(format!("nix-env -e {}", packages.join(" "))),
            files_modified: vec!["/etc/nixos/configuration.nix".to_string()],
            ..Default::default()
        }
    }

    fn simulate_remove(&self, command: &str, args: &[&str]) -> WhatIfResult {
        let packages: Vec<String> = args
            .iter()
            .filter(|a| !a.starts_with('-'))
            .map(|s| s.to_string())
            .collect();

        let mut warnings = Vec::new();

        // Check if packages are actually installed
        for pkg in &packages {
            if !self.known_packages.contains(pkg) {
                warnings.push(format!("{pkg} may not be installed"));
            }
        }

        // Warn about critical packages
        let critical = ["bash", "coreutils", "systemd", "nix", "linux"];
        for pkg in &packages {
            if critical.iter().any(|c| pkg.contains(c)) {
                warnings.push(format!("WARNING: {pkg} is a critical system package!"));
            }
        }

        WhatIfResult {
            command: command.to_string(),
            summary: format!("Remove {} package(s)", packages.len()),
            packages_removed: packages.clone(),
            estimated_time: Some("~1-2 min".to_string()),
            warnings,
            dry_run_command: Some(format!("nix-env -e {} --dry-run", packages.join(" "))),
            confidence: 0.85,
            reversible: true,
            rollback_command: Some(format!(
                "nix-env -iA nixpkgs.{}",
                packages.join(" nixpkgs.")
            )),
            files_modified: vec!["/etc/nixos/configuration.nix".to_string()],
            ..Default::default()
        }
    }

    fn simulate_enable(&self, command: &str, args: &[&str]) -> WhatIfResult {
        let services: Vec<String> = args
            .iter()
            .filter(|a| !a.starts_with('-'))
            .map(|s| s.to_string())
            .collect();

        let mut warnings = Vec::new();

        // Check if service is known
        for svc in &services {
            if !self.known_services.contains(svc) {
                warnings.push(format!("{svc} is not a recognized service"));
            }
        }

        WhatIfResult {
            command: command.to_string(),
            summary: format!("Enable {} service(s)", services.len()),
            services_enabled: services.clone(),
            estimated_time: Some("~2-5 min (with rebuild)".to_string()),
            warnings,
            dry_run_command: Some("nixos-rebuild dry-activate".to_string()),
            confidence: 0.8,
            reversible: true,
            rollback_command: Some(format!("disable {}", services.join(" "))),
            files_modified: vec!["/etc/nixos/configuration.nix".to_string()],
            ..Default::default()
        }
    }

    fn simulate_disable(&self, command: &str, args: &[&str]) -> WhatIfResult {
        let services: Vec<String> = args
            .iter()
            .filter(|a| !a.starts_with('-'))
            .map(|s| s.to_string())
            .collect();

        let mut warnings = Vec::new();

        // Warn about critical services
        let critical = ["sshd", "networking", "systemd", "nix-daemon"];
        for svc in &services {
            if critical.iter().any(|c| svc.contains(c)) {
                warnings.push(format!(
                    "WARNING: Disabling {svc} may affect system accessibility!"
                ));
            }
        }

        WhatIfResult {
            command: command.to_string(),
            summary: format!("Disable {} service(s)", services.len()),
            services_disabled: services.clone(),
            estimated_time: Some("~2-5 min (with rebuild)".to_string()),
            warnings,
            dry_run_command: Some("nixos-rebuild dry-activate".to_string()),
            confidence: 0.8,
            reversible: true,
            rollback_command: Some(format!("enable {}", services.join(" "))),
            files_modified: vec!["/etc/nixos/configuration.nix".to_string()],
            ..Default::default()
        }
    }

    fn simulate_rebuild(&self, command: &str, args: &[&str]) -> WhatIfResult {
        let action = args.first().copied().unwrap_or("switch");

        let (summary, reboot_required, time_estimate) = match action {
            "switch" => (
                "Rebuild and switch to new configuration",
                false,
                "~2-10 min",
            ),
            "boot" => (
                "Rebuild and set as boot default (needs reboot)",
                true,
                "~2-10 min",
            ),
            "test" => (
                "Rebuild and activate temporarily (no boot entry)",
                false,
                "~2-10 min",
            ),
            "dry-run" | "dry-activate" => ("Simulate rebuild (no changes)", false, "~30 sec"),
            _ => ("Rebuild configuration", false, "~2-10 min"),
        };

        let mut warnings = Vec::new();

        if action == "switch" {
            warnings.push(
                "Running services will be restarted if their configuration changed".to_string(),
            );
        }

        if args.contains(&"--upgrade") {
            warnings.push(
                "Channels will be updated first, which may bring in new package versions"
                    .to_string(),
            );
        }

        WhatIfResult {
            command: command.to_string(),
            summary: summary.to_string(),
            reboot_required,
            estimated_time: Some(time_estimate.to_string()),
            warnings,
            dry_run_command: Some("nixos-rebuild dry-activate".to_string()),
            confidence: 0.95,
            reversible: true,
            rollback_command: Some("nixos-rebuild switch --rollback".to_string()),
            files_modified: vec![
                "/etc/nixos/configuration.nix".to_string(),
                "/nix/var/nix/profiles/system".to_string(),
            ],
            ..Default::default()
        }
    }

    fn simulate_garbage_collect(&self, command: &str, args: &[&str]) -> WhatIfResult {
        let delete_old = args.iter().any(|a| *a == "-d" || *a == "--delete-old");

        let mut warnings = Vec::new();

        if delete_old {
            warnings.push(
                "This will delete ALL old generations - you won't be able to rollback!".to_string(),
            );
            warnings.push("Consider using --delete-older-than 30d instead".to_string());
        }

        WhatIfResult {
            command: command.to_string(),
            summary: if delete_old {
                "Delete all old generations and collect garbage".to_string()
            } else {
                "Collect garbage (unreachable store paths)".to_string()
            },
            estimated_time: Some("~1-30 min (depends on store size)".to_string()),
            warnings,
            dry_run_command: Some("nix-collect-garbage --dry-run".to_string()),
            confidence: 0.9,
            reversible: !delete_old,
            rollback_command: None,
            ..Default::default()
        }
    }

    fn simulate_nix_command(&self, command: &str, args: &[&str]) -> WhatIfResult {
        let subcommand = args.first().copied().unwrap_or("");

        match subcommand {
            "build" => WhatIfResult {
                command: command.to_string(),
                summary: "Build a derivation".to_string(),
                estimated_time: Some("varies (seconds to hours)".to_string()),
                dry_run_command: Some(command.replace("nix build", "nix build --dry-run")),
                confidence: 0.9,
                reversible: true,
                ..Default::default()
            },
            "run" => WhatIfResult {
                command: command.to_string(),
                summary: "Build and run a package".to_string(),
                estimated_time: Some("varies".to_string()),
                confidence: 0.7,
                reversible: true,
                ..Default::default()
            },
            "shell" => WhatIfResult {
                command: command.to_string(),
                summary: "Enter a shell with specified packages".to_string(),
                estimated_time: Some("~1-5 min (if not cached)".to_string()),
                confidence: 0.9,
                reversible: true,
                ..Default::default()
            },
            "flake" if args.len() > 1 && args[1] == "update" => WhatIfResult {
                command: command.to_string(),
                summary: "Update flake inputs".to_string(),
                estimated_time: Some("~10-60 sec".to_string()),
                warnings: vec!["This may bring in breaking changes from inputs".to_string()],
                dry_run_command: Some("nix flake update --dry-run".to_string()),
                confidence: 0.8,
                reversible: true,
                rollback_command: Some("git checkout flake.lock".to_string()),
                files_modified: vec!["flake.lock".to_string()],
                ..Default::default()
            },
            _ => self.simulate_generic(command),
        }
    }

    fn simulate_generic(&self, command: &str) -> WhatIfResult {
        WhatIfResult {
            command: command.to_string(),
            summary: "Execute command".to_string(),
            confidence: 0.3,
            warnings: vec!["Cannot predict effects of this command".to_string()],
            ..Default::default()
        }
    }

    fn estimate_install_time(&self, package_count: usize) -> String {
        match package_count {
            0 => "instant".to_string(),
            1 => "~30 sec - 2 min".to_string(),
            2..=5 => "~1-5 min".to_string(),
            6..=10 => "~3-10 min".to_string(),
            _ => "~5-20 min".to_string(),
        }
    }

    /// Format result for terminal display
    pub fn format_for_terminal(&self, result: &WhatIfResult) -> Vec<String> {
        let mut lines = Vec::new();

        // Header
        lines.push(format!("=== What-If: {} ===", result.summary));
        lines.push(format!("Confidence: {:.0}%", result.confidence * 100.0));

        // Packages
        if !result.packages_added.is_empty() {
            lines.push(format!(
                "+ Packages to add: {}",
                result.packages_added.join(", ")
            ));
        }
        if !result.packages_removed.is_empty() {
            lines.push(format!(
                "- Packages to remove: {}",
                result.packages_removed.join(", ")
            ));
        }

        // Services
        if !result.services_enabled.is_empty() {
            lines.push(format!(
                "▶ Services to enable: {}",
                result.services_enabled.join(", ")
            ));
        }
        if !result.services_disabled.is_empty() {
            lines.push(format!(
                "■ Services to disable: {}",
                result.services_disabled.join(", ")
            ));
        }

        // Files
        if !result.files_modified.is_empty() {
            lines.push(format!(
                "📄 Files affected: {}",
                result.files_modified.join(", ")
            ));
        }

        // Time estimate
        if let Some(ref time) = result.estimated_time {
            lines.push(format!("⏱ Estimated time: {time}"));
        }

        // Reboot
        if result.reboot_required {
            lines.push("🔄 Reboot required".to_string());
        }

        // Reversibility
        if result.reversible {
            if let Some(ref cmd) = result.rollback_command {
                lines.push(format!("↩ Rollback: {cmd}"));
            } else {
                lines.push("↩ Reversible".to_string());
            }
        } else {
            lines.push("⚠ NOT REVERSIBLE".to_string());
        }

        // Warnings
        for warning in &result.warnings {
            lines.push(format!("⚠ {warning}"));
        }

        // Dry-run command
        if let Some(ref cmd) = result.dry_run_command {
            lines.push(format!("🔍 Dry-run: {cmd}"));
        }

        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulate_install() {
        let sim = WhatIfSimulator::new();
        let result = sim.simulate("install firefox vim");

        assert_eq!(result.packages_added, vec!["firefox", "vim"]);
        assert!(result.reversible);
    }

    #[test]
    fn test_simulate_rebuild() {
        let sim = WhatIfSimulator::new();
        let result = sim.simulate("nixos-rebuild switch");

        assert!(result.summary.contains("switch"));
        assert!(!result.reboot_required);
    }

    #[test]
    fn test_garbage_collect_warning() {
        let sim = WhatIfSimulator::new();
        let result = sim.simulate("nix-collect-garbage -d");

        assert!(!result.reversible);
        assert!(!result.warnings.is_empty());
    }
}
