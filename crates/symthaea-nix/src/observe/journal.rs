// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Systemd journal log observation and streaming.
//!
//! Parses output from `journalctl` to observe recent log entries, per-unit logs,
//! and error patterns. Provides structured journal entries for anomaly detection
//! in the world model.

use std::process::Command;
use tracing::debug;

/// Observes and streams journal log entries for anomaly detection.
pub struct JournalObserver;

/// A single parsed journal entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JournalEntry {
    /// Timestamp string as printed by journalctl (e.g. "Jan 15 10:30:00").
    pub timestamp: String,
    /// The systemd unit or syslog identifier that produced this entry.
    pub unit: String,
    /// Syslog priority level (0=emerg, 1=alert, ..., 6=info, 7=debug).
    pub priority: u8,
    /// The log message text.
    pub message: String,
}

impl JournalObserver {
    /// Retrieve the most recent `lines` journal entries by running
    /// `journalctl -n <lines> --no-pager -o short`.
    ///
    /// Output format per line:
    /// ```text
    /// Jan 15 10:30:00 hostname unitname[pid]: message text here
    /// ```
    pub fn recent_entries(lines: usize) -> Result<Vec<JournalEntry>, std::io::Error> {
        let output = Command::new("journalctl")
            .args(["-n", &lines.to_string(), "--no-pager", "-o", "short"])
            .output()?;

        if !output.status.success() {
            return Err(std::io::Error::other(format!(
                "journalctl failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Self::parse_short_output(&stdout, 6) // default priority = info
    }

    /// Retrieve recent logs for a specific unit.
    pub fn unit_logs(unit: &str, lines: usize) -> Result<Vec<JournalEntry>, std::io::Error> {
        let output = Command::new("journalctl")
            .args([
                "-u",
                unit,
                "-n",
                &lines.to_string(),
                "--no-pager",
                "-o",
                "short",
            ])
            .output()?;

        if !output.status.success() {
            return Err(std::io::Error::other(format!(
                "journalctl -u {} failed: {}",
                unit,
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Self::parse_short_output(&stdout, 6)
    }

    /// Retrieve error-level (priority <= 3) entries since a duration ago.
    ///
    /// The `duration` parameter is a systemd time span, e.g. "1h", "30min", "2d".
    pub fn errors_since(duration: &str) -> Result<Vec<JournalEntry>, std::io::Error> {
        let since = format!("-{duration}");
        let output = Command::new("journalctl")
            .args(["--since", &since, "-p", "err", "--no-pager", "-o", "short"])
            .output()?;

        if !output.status.success() {
            return Err(std::io::Error::other(format!(
                "journalctl --since {} -p err failed: {}",
                since,
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Self::parse_short_output(&stdout, 3) // error priority
    }

    // ---- Parsing helpers ----

    /// Parse journalctl `-o short` output.
    ///
    /// Each line typically looks like:
    /// ```text
    /// Jan 15 10:30:00 myhostname nginx[1234]: 127.0.0.1 GET /index.html 200
    /// Jan 15 10:30:01 myhostname kernel: some kernel message
    /// ```
    ///
    /// The `default_priority` is assigned to all entries from this batch since
    /// the short format does not include priority in the output.
    pub fn parse_short_output(
        output: &str,
        default_priority: u8,
    ) -> Result<Vec<JournalEntry>, std::io::Error> {
        let mut entries = Vec::new();

        for line in output.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            // Skip journalctl header lines like "-- Journal begins at ..."
            // or "-- No entries --"
            if trimmed.starts_with("-- ") {
                continue;
            }

            if let Some(entry) = Self::parse_short_line(trimmed, default_priority) {
                entries.push(entry);
            }
        }

        Ok(entries)
    }

    /// Parse a single journalctl short-format line.
    ///
    /// Expected format: `<Month> <Day> <Time> <hostname> <unit>[<pid>]: <message>`
    /// or: `<Month> <Day> <Time> <hostname> <unit>: <message>`
    fn parse_short_line(line: &str, default_priority: u8) -> Option<JournalEntry> {
        // The timestamp is the first 3 tokens: "Jan 15 10:30:00"
        let tokens: Vec<&str> = line.splitn(5, char::is_whitespace).collect();
        let tokens: Vec<&str> = tokens.into_iter().filter(|t| !t.is_empty()).collect();

        if tokens.len() < 5 {
            // Try a simpler split: might be a short line
            return Self::parse_short_line_fallback(line, default_priority);
        }

        let timestamp = format!("{} {} {}", tokens[0], tokens[1], tokens[2]);
        // tokens[3] is hostname
        let remainder = tokens[4]; // "unit[pid]: message" or "unit: message"

        // Split on first ": " to separate unit from message
        let (unit_part, message) = if let Some(idx) = remainder.find(": ") {
            (&remainder[..idx], remainder[idx + 2..].to_string())
        } else {
            (remainder, String::new())
        };

        // Strip [pid] from unit name if present
        let unit = if let Some(bracket_idx) = unit_part.find('[') {
            unit_part[..bracket_idx].to_string()
        } else {
            unit_part.to_string()
        };

        Some(JournalEntry {
            timestamp,
            unit,
            priority: default_priority,
            message,
        })
    }

    /// Fallback parser for lines that don't match the standard format.
    fn parse_short_line_fallback(line: &str, default_priority: u8) -> Option<JournalEntry> {
        if line.trim().is_empty() {
            return None;
        }

        debug!(
            line_preview = &line[..line.len().min(80)],
            "Journal line did not match standard short format, using fallback parser"
        );

        Some(JournalEntry {
            timestamp: String::new(),
            unit: "unknown".to_string(),
            priority: default_priority,
            message: line.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MOCK_JOURNAL_OUTPUT: &str = "\
Jan 15 10:30:00 myhost nginx[1234]: 127.0.0.1 GET /index.html 200
Jan 15 10:30:01 myhost kernel: Linux version 6.1.72
Jan 15 10:30:02 myhost sshd[5678]: Accepted publickey for user from 10.0.0.1
Jan 15 10:30:03 myhost systemd[1]: Started nginx.service - A high performance web server
";

    #[test]
    fn test_parse_short_output() {
        let entries = JournalObserver::parse_short_output(MOCK_JOURNAL_OUTPUT, 6).unwrap();
        assert_eq!(entries.len(), 4);

        assert_eq!(entries[0].timestamp, "Jan 15 10:30:00");
        assert_eq!(entries[0].unit, "nginx");
        assert_eq!(entries[0].message, "127.0.0.1 GET /index.html 200");
        assert_eq!(entries[0].priority, 6);

        assert_eq!(entries[1].unit, "kernel");
        assert_eq!(entries[1].message, "Linux version 6.1.72");

        assert_eq!(entries[2].unit, "sshd");
        assert!(entries[2].message.contains("Accepted publickey"));

        assert_eq!(entries[3].unit, "systemd");
    }

    #[test]
    fn test_parse_short_output_empty() {
        let entries = JournalObserver::parse_short_output("", 6).unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_parse_short_output_skips_header() {
        let output = "-- Journal begins at Mon 2024-01-01 00:00:00 UTC. --\n\
                       Jan 15 10:30:00 myhost nginx[1234]: test message\n";
        let entries = JournalObserver::parse_short_output(output, 6).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].unit, "nginx");
    }

    #[test]
    fn test_parse_error_entries() {
        let output = "\
Jan 15 10:30:00 myhost postgresql[999]: FATAL: could not open relation mapping file
Jan 15 10:30:01 myhost kernel: BUG: unable to handle page fault at 0xdeadbeef
";
        let entries = JournalObserver::parse_short_output(output, 3).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].priority, 3);
        assert!(entries[0].message.contains("FATAL"));
        assert!(entries[1].message.contains("BUG"));
    }

    #[test]
    fn test_parse_unit_without_pid() {
        let output = "Feb 01 09:00:00 myhost kernel: some kernel message\n";
        let entries = JournalObserver::parse_short_output(output, 6).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].unit, "kernel");
        assert_eq!(entries[0].message, "some kernel message");
    }

    #[test]
    fn test_parse_short_line_fallback() {
        // Lines that are too short for standard parsing use fallback
        let output = "short\n";
        let entries = JournalObserver::parse_short_output(output, 6).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].unit, "unknown");
        assert_eq!(entries[0].message, "short");
        assert!(entries[0].timestamp.is_empty());
    }

    #[test]
    fn test_parse_no_entries_marker() {
        let output = "-- No entries --\n";
        let entries = JournalObserver::parse_short_output(output, 6).unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_parse_mixed_valid_and_fallback() {
        let output = "Jan 15 10:30:00 myhost nginx[1234]: valid entry\n\
                       orphan line\n";
        let entries = JournalObserver::parse_short_output(output, 6).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].unit, "nginx");
        assert_eq!(entries[1].unit, "unknown"); // fallback
    }
}
