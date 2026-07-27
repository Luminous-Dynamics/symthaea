// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Systemd service and unit observation.
//!
//! Parses output from `systemctl` to observe service states, failed units,
//! and individual unit status. This is the primary source for the "services"
//! dimension of the system state snapshot.

use std::process::Command;

/// Observes systemd unit states, dependencies, and resource usage.
pub struct SystemdObserver;

/// Information about a single systemd unit.
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

    /// Get the status of a specific unit.
    pub fn unit_status(unit: &str) -> Result<UnitInfo, std::io::Error> {
        let output = Command::new("systemctl")
            .args([
                "show",
                unit,
                "--no-pager",
                "--property=Id,LoadState,ActiveState,SubState,Description",
            ])
            .output()?;

        if !output.status.success() {
            return Err(std::io::Error::other(format!(
                "systemctl show {} failed: {}",
                unit,
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Self::parse_unit_show(&stdout)
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

    /// Parse `systemctl show --property=...` output.
    ///
    /// Format:
    /// ```text
    /// Id=nginx.service
    /// LoadState=loaded
    /// ActiveState=active
    /// SubState=running
    /// Description=A high performance web server
    /// ```
    pub fn parse_unit_show(output: &str) -> Result<UnitInfo, std::io::Error> {
        let mut name = String::new();
        let mut load_state = String::new();
        let mut active_state = String::new();
        let mut sub_state = String::new();
        let mut description = String::new();

        for line in output.lines() {
            let trimmed = line.trim();
            if let Some((key, value)) = trimmed.split_once('=') {
                match key {
                    "Id" => name = value.to_string(),
                    "LoadState" => load_state = value.to_string(),
                    "ActiveState" => active_state = value.to_string(),
                    "SubState" => sub_state = value.to_string(),
                    "Description" => description = value.to_string(),
                    _ => {}
                }
            }
        }

        if name.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "could not parse unit name from systemctl show output",
            ));
        }

        Ok(UnitInfo {
            name,
            load_state,
            active_state,
            sub_state,
            description,
        })
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

    const MOCK_SHOW_OUTPUT: &str = "\
Id=nginx.service
LoadState=loaded
ActiveState=active
SubState=running
Description=A high performance web server and reverse proxy
";

    #[test]
    fn test_parse_unit_show() {
        let unit = SystemdObserver::parse_unit_show(MOCK_SHOW_OUTPUT).unwrap();
        assert_eq!(unit.name, "nginx.service");
        assert_eq!(unit.load_state, "loaded");
        assert_eq!(unit.active_state, "active");
        assert_eq!(unit.sub_state, "running");
        assert_eq!(
            unit.description,
            "A high performance web server and reverse proxy"
        );
    }

    #[test]
    fn test_parse_unit_show_missing_id() {
        let bad = "LoadState=loaded\nActiveState=active\n";
        let result = SystemdObserver::parse_unit_show(bad);
        assert!(result.is_err());
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
    fn test_parse_unit_show_extra_keys_ignored() {
        let output = "Id=test.service\nLoadState=loaded\nActiveState=active\nSubState=running\nDescription=Test\nExtraKey=ignored\n";
        let unit = SystemdObserver::parse_unit_show(output).unwrap();
        assert_eq!(unit.name, "test.service");
    }

    #[test]
    fn test_parse_unit_show_empty_values() {
        let output = "Id=test.service\nLoadState=\nActiveState=\nSubState=\nDescription=\n";
        let unit = SystemdObserver::parse_unit_show(output).unwrap();
        assert_eq!(unit.name, "test.service");
        assert_eq!(unit.load_state, "");
    }

    #[test]
    fn test_parse_unit_show_empty_output() {
        let result = SystemdObserver::parse_unit_show("");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_unit_list_mixed_states() {
        let output = "\
  a.service loaded active   running   Service A\n\
  b.service loaded inactive dead      Service B\n\
  c.service loaded failed   failed    Service C\n\
  d.service masked inactive dead      Masked D\n";
        let units = SystemdObserver::parse_unit_list(output).unwrap();
        assert_eq!(units.len(), 4);
        assert_eq!(units[0].active_state, "active");
        assert_eq!(units[1].active_state, "inactive");
        assert_eq!(units[2].active_state, "failed");
        assert_eq!(units[3].load_state, "masked");
    }
}
