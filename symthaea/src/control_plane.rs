// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shared control-plane policy and audit helpers.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs::{OpenOptions, create_dir_all};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub const SERVICE_PROTOCOL_VERSION: u32 = 1;
pub const MAX_REQUEST_LINE_BYTES: usize = 64 * 1024;

pub fn service_readonly_programs() -> Vec<String> {
    [
        "cat",
        "date",
        "df",
        "du",
        "echo",
        "env",
        "find",
        "free",
        "git diff|log|show|status|branch|rev-parse",
        "grep",
        "head",
        "hostname",
        "id",
        "journalctl",
        "ls",
        "nix search|eval|path-info|flake show|flake metadata",
        "nix-env -q|--query",
        "nixos-rebuild dry-run|build",
        "printenv",
        "ps",
        "pwd",
        "rg",
        "sleep",
        "stat",
        "systemctl status|show|list-units|list-unit-files|is-active",
        "tail",
        "true",
        "false",
        "uname",
        "which",
        "whoami",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

pub fn service_known_not_implemented_request_types() -> Vec<String> {
    ["gui_widget_change", "parse_nix_config"]
        .into_iter()
        .map(str::to_string)
        .collect()
}

pub fn parse_bearer_token(value: &str) -> Option<&str> {
    value.strip_prefix("Bearer ")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub timestamp: DateTime<Utc>,
    pub source: String,
    pub event: String,
    pub subject: String,
    pub detail: String,
}

pub struct AuditLog {
    recent: Mutex<VecDeque<AuditEvent>>,
    file_path: Option<PathBuf>,
    max_recent: usize,
}

impl AuditLog {
    pub fn new(file_path: Option<PathBuf>, max_recent: usize) -> Self {
        Self {
            recent: Mutex::new(VecDeque::new()),
            file_path,
            max_recent,
        }
    }

    pub fn file_path(&self) -> Option<&Path> {
        self.file_path.as_deref()
    }

    pub fn record(
        &self,
        source: impl Into<String>,
        event: impl Into<String>,
        subject: impl Into<String>,
        detail: impl Into<String>,
    ) -> std::io::Result<AuditEvent> {
        let entry = AuditEvent {
            timestamp: Utc::now(),
            source: source.into(),
            event: event.into(),
            subject: subject.into(),
            detail: detail.into(),
        };

        {
            let mut recent = self.recent.lock().expect("audit recent log poisoned");
            recent.push_front(entry.clone());
            while recent.len() > self.max_recent {
                recent.pop_back();
            }
        }

        if let Some(path) = &self.file_path {
            if let Some(parent) = path.parent() {
                create_dir_all(parent)?;
            }
            let mut file = OpenOptions::new().create(true).append(true).open(path)?;
            serde_json::to_writer(&mut file, &entry).map_err(std::io::Error::other)?;
            file.write_all(b"\n")?;
        }

        Ok(entry)
    }

    pub fn list(&self, limit: usize) -> Vec<AuditEvent> {
        let recent = self.recent.lock().expect("audit recent log poisoned");
        recent.iter().take(limit).cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AuditLog, SERVICE_PROTOCOL_VERSION, parse_bearer_token,
        service_known_not_implemented_request_types,
    };

    #[test]
    fn test_parse_bearer_token() {
        assert_eq!(parse_bearer_token("Bearer abc"), Some("abc"));
        assert_eq!(parse_bearer_token("Basic abc"), None);
    }

    #[test]
    fn test_audit_log_keeps_recent_entries() {
        let log = AuditLog::new(None, 2);
        log.record("service", "a", "x", "1").expect("record 1");
        log.record("service", "b", "y", "2").expect("record 2");
        log.record("service", "c", "z", "3").expect("record 3");
        let events = log.list(10);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event, "c");
        assert_eq!(events[1].event, "b");
        assert_eq!(SERVICE_PROTOCOL_VERSION, 1);
    }

    #[test]
    fn test_known_not_implemented_requests_are_stable() {
        assert_eq!(
            service_known_not_implemented_request_types(),
            vec![
                "gui_widget_change".to_string(),
                "parse_nix_config".to_string()
            ]
        );
    }
}
