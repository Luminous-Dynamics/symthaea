// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Systemd service and unit observation.
//!
//! Parses diagnostic output from `systemctl` to observe service lists and
//! failed units. Governed service pre-state is obtained through the typed
//! `NixServiceObservedStateV1` path in the action layer, not this module.

use std::process::Command;

/// Observes systemd unit states, dependencies, and resource usage.
pub struct SystemdObserver;

/// Diagnostic information about a systemd unit.
///
/// This representation is intentionally compatibility/diagnostic-only. It is
/// lossy and must not be used as governed pre-state, authorization input,
/// effect-binding input, or execution authority. Governed service pre-state
/// uses `NixServiceObservedStateV1` instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnitInfo {
    /// Unit name (e.g. "nginx.service").
    pub name: String,
    /// Load state: "loaded", "not-found", "masked", etc.
    pub load_state: String,
    /// Active state: "active", "inactive", "failed", "activating", etc.
    pub active_state: String,
    /// Sub-state: "running", "dead", "exited", "waiting", etc.
    pub sub_state: String,
    /// Human-readable description.
    pub description: String,
}

impl SystemdObserver {
    /// List all service units by running
    /// `systemctl list-units --type=service --no-pager --plain --no-legend`.
    ///
    /// Output format per line:
    /// ```text
    /// nginx.service loaded active running A high performance web server
    /// sshd.service  loaded active running OpenSSH Daemon
    /// ```
    pub fn list_units() -> Result<Vec<UnitInfo>, std::io::Error> {
        let output = Command::new("systemctl")
            .args([
                "list-units",
                "--type=service",
                "--no-pager",
                "--plain",
                "--no-legend",
            ])
            .output()?;

        if !output.status.success() {
            return Err(std::io::Error::other(format!(
                "systemctl list-units failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Self::parse_unit_list(&stdout)
    }

    /// List only failed units by running
    /// `systemctl list-units --type=service --state=failed --no-pager --plain --no-legend`.
    pub fn failed_units() -> Result<Vec<UnitInfo>, std::io::Error> {
        let output = Command::new("systemctl")
            .args([
                "list-units",
                "--type=service",
                "--state=failed",
                "--no-pager",
                "--plain",
                "--no-legend",
            ])
            .output()?;

        if !output.status.success() {
            return Err(std::io::Error::other(format!(
                "systemctl list-units --state=failed failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Self::parse_unit_list(&stdout)
    }

    // ---- Parsing helpers ----

    /// Parse `systemctl list-units --plain --no-legend` output.
    ///
    /// Each line has exactly 4 fixed-width fields followed by the description:
    /// `<UNIT> <LOAD> <ACTIVE> <SUB> <DESCRIPTION...>`
    pub fn parse_unit_list(output: &str) -> Result<Vec<UnitInfo>, std::io::Error> {
        let mut units = Vec::new();

        for line in output.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let tokens: Vec<&str> = trimmed.split_whitespace().collect();
            if tokens.len() < 4 {
                continue;
            }

            let name = tokens[0].to_string();
            let load_state = tokens[1].to_string();
            let active_state = tokens[2].to_string();
            let sub_state = tokens[3].to_string();
            let description = if tokens.len() > 4 {
                tokens[4..].join(" ")
            } else {
                String::new()
            };

            units.push(UnitInfo {
                name,
                load_state,
                active_state,
                sub_state,
                description,
            });
        }

        Ok(units)
    }

}


#[cfg(test)]
mod tests {
    use super::*;

    const MOCK_LIST_UNITS: &str = "\
  nginx.service                loaded active running  A high performance web server
  sshd.service                 loaded active running  OpenSSH Daemon
  postgresql.service           loaded active running  PostgreSQL database server
  failed-thing.service         loaded failed failed   A broken service
";

    #[test]
    fn test_parse_unit_list() {
        let units = SystemdObserver::parse_unit_list(MOCK_LIST_UNITS).unwrap();
        assert_eq!(units.len(), 4);

        assert_eq!(units[0].name, "nginx.service");
        assert_eq!(units[0].load_state, "loaded");
        assert_eq!(units[0].active_state, "active");
        assert_eq!(units[0].sub_state, "running");
        assert_eq!(units[0].description, "A high performance web server");

        assert_eq!(units[1].name, "sshd.service");
        assert_eq!(units[1].sub_state, "running");

        assert_eq!(units[3].name, "failed-thing.service");
        assert_eq!(units[3].active_state, "failed");
        assert_eq!(units[3].sub_state, "failed");
    }

    #[test]
    fn test_parse_unit_list_empty() {
        let units = SystemdObserver::parse_unit_list("").unwrap();
        assert!(units.is_empty());
    }

    #[test]
    fn test_parse_failed_units() {
        let mock_failed = "\
  failed-thing.service   loaded failed failed   A broken service
";
        let units = SystemdObserver::parse_unit_list(mock_failed).unwrap();
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].active_state, "failed");
    }

    #[test]
    fn test_parse_unit_list_short_line_skipped() {
        let output = "too short\nfoo\n  a.service loaded active running OK\n";
        let units = SystemdObserver::parse_unit_list(output).unwrap();
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].name, "a.service");
    }

    #[test]
    fn test_parse_unit_list_no_description() {
        let output = "  bare.service loaded active exited\n";
        let units = SystemdObserver::parse_unit_list(output).unwrap();
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].description, "");
        assert_eq!(units[0].sub_state, "exited");
    }

    #[test]
    fn test_parse_unit_list_mixed_states() {
        let output = "\
  a.service loaded active   running   Service A
  b.service loaded inactive dead      Service B
  c.service loaded failed   failed    Service C
  d.service masked inactive dead      Masked D
";
        let units = SystemdObserver::parse_unit_list(output).unwrap();
        assert_eq!(units.len(), 4);
        assert_eq!(units[0].active_state, "active");
        assert_eq!(units[1].active_state, "inactive");
        assert_eq!(units[2].active_state, "failed");
        assert_eq!(units[3].load_state, "masked");
    }
}
