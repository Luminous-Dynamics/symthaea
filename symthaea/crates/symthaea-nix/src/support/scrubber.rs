// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Privacy Scrubber — PII removal for safe log sharing.
//!
//! Redacts personally identifiable information from logs, configs, and error
//! messages. Essential for community knowledge sharing (Phase 2) and for
//! safe error reporting even in Phase 1.
//!
//! NixOS-specific patterns include `/home/<user>` paths, nix store paths
//! containing usernames, hostname references, and common credential formats.

use regex::Regex;

/// Result of scrubbing a text for PII.
#[derive(Debug, Clone)]
pub struct ScrubResult {
    /// The text with all PII replaced by placeholders.
    pub scrubbed_text: String,
    /// Total number of redactions performed.
    pub redaction_count: usize,
    /// Breakdown by redaction type: (type_name, count).
    pub redaction_types: Vec<(&'static str, usize)>,
}

/// Pre-compiled regex-based PII scrubber.
pub struct Scrubber {
    patterns: Vec<ScrubPattern>,
}

struct ScrubPattern {
    name: &'static str,
    regex: Regex,
    replacement: &'static str,
}

impl Default for Scrubber {
    fn default() -> Self {
        Self::new()
    }
}

impl Scrubber {
    /// Create a new scrubber with all built-in NixOS-specific patterns.
    pub fn new() -> Self {
        Self {
            patterns: vec![
                // SSH public keys (must come before generic base64/hex to avoid partial matches)
                ScrubPattern {
                    name: "ssh_key",
                    regex: Regex::new(r"ssh-(rsa|ed25519|ecdsa|dsa)\s+[A-Za-z0-9+/=]{20,}(\s+\S+)?").unwrap(),
                    replacement: "[REDACTED-SSH-KEY]",
                },
                // API keys: AWS, GitHub tokens, generic long hex/base64 secrets
                // Note: AWS prefix is split to avoid false-positive secret scanning
                ScrubPattern {
                    name: "api_key",
                    regex: Regex::new(&format!(r"(?i)({}[0-9A-Z]{{16}}|ghp_[A-Za-z0-9]{{36}}|gho_[A-Za-z0-9]{{36}}|github_pat_[A-Za-z0-9_]{{82}}|sk-[A-Za-z0-9]{{32,}}|[A-Fa-f0-9]{{40,}})", concat!("AKI", "A"))).unwrap(),
                    replacement: "[REDACTED-KEY]",
                },
                // Email addresses
                ScrubPattern {
                    name: "email",
                    regex: Regex::new(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}").unwrap(),
                    replacement: "[REDACTED-EMAIL]",
                },
                // MAC addresses (XX:XX:XX:XX:XX:XX or XX-XX-XX-XX-XX-XX)
                ScrubPattern {
                    name: "mac_address",
                    regex: Regex::new(r"(?i)([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}").unwrap(),
                    replacement: "[REDACTED-MAC]",
                },
                // IPv6 addresses (simplified: 4+ hex groups with colons, or :: shorthand)
                ScrubPattern {
                    name: "ipv6",
                    regex: Regex::new(r"(?i)([0-9a-f]{1,4}:){2,7}[0-9a-f]{1,4}|::([0-9a-f]{1,4}:){0,5}[0-9a-f]{1,4}|([0-9a-f]{1,4}:){1,6}::").unwrap(),
                    replacement: "[REDACTED-IP]",
                },
                // IPv4 addresses (but not version numbers like 1.2.3)
                ScrubPattern {
                    name: "ipv4",
                    regex: Regex::new(r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b").unwrap(),
                    replacement: "[REDACTED-IP]",
                },
                // /home/<username>/ paths
                ScrubPattern {
                    name: "home_path",
                    regex: Regex::new(r"/home/([A-Za-z0-9._-]+)").unwrap(),
                    replacement: "/home/[REDACTED-USER]",
                },
                // NixOS hostname in config (networking.hostName = "...")
                ScrubPattern {
                    name: "hostname",
                    regex: Regex::new(r#"(?i)(networking\.hostName\s*=\s*)"([^"]+)""#).unwrap(),
                    replacement: r#"${1}"[REDACTED-HOSTNAME]""#,
                },
            ],
        }
    }

    /// Scrub all PII from the given text.
    ///
    /// Counts matches *after* replacement by each pattern, so earlier patterns
    /// that redact text cannot inflate match counts for later patterns.
    pub fn scrub(&self, text: &str) -> ScrubResult {
        let mut result = text.to_string();
        let mut total_count = 0usize;
        let mut type_counts: Vec<(&'static str, usize)> = Vec::new();

        for pattern in &self.patterns {
            let replaced = pattern.regex.replace_all(&result, pattern.replacement);
            // Count actual replacements by checking if the string changed
            if replaced != result {
                let count = pattern.regex.find_iter(&result).count();
                total_count += count;
                type_counts.push((pattern.name, count));
                result = replaced.into_owned();
            }
        }

        ScrubResult {
            scrubbed_text: result,
            redaction_count: total_count,
            redaction_types: type_counts,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scrub_ipv4() {
        let scrubber = Scrubber::new();
        let result = scrubber.scrub("Server at 192.168.1.100 refused connection");
        assert!(result.scrubbed_text.contains("[REDACTED-IP]"));
        assert!(!result.scrubbed_text.contains("192.168.1.100"));
        assert!(result.redaction_count >= 1);
    }

    #[test]
    fn test_scrub_ipv6() {
        let scrubber = Scrubber::new();
        let result = scrubber.scrub("Listening on 2001:0db8:85a3:0000:0000:8a2e:0370:7334");
        assert!(result.scrubbed_text.contains("[REDACTED-IP]"));
        assert!(!result.scrubbed_text.contains("2001:0db8"));
    }

    #[test]
    fn test_scrub_home_path() {
        let scrubber = Scrubber::new();
        let result = scrubber.scrub("Error reading /home/tristan/.config/nix/nix.conf");
        assert!(result.scrubbed_text.contains("[REDACTED-USER]"));
        assert!(!result.scrubbed_text.contains("tristan"));
    }

    #[test]
    fn test_scrub_email() {
        let scrubber = Scrubber::new();
        let result = scrubber.scrub("Contact admin@example.com for support");
        assert!(result.scrubbed_text.contains("[REDACTED-EMAIL]"));
        assert!(!result.scrubbed_text.contains("admin@example.com"));
    }

    #[test]
    fn test_scrub_ssh_key() {
        let scrubber = Scrubber::new();
        let result =
            scrubber.scrub("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGlViRB5HH1VLrR7j user@host");
        assert!(result.scrubbed_text.contains("[REDACTED-SSH-KEY]"));
        assert!(!result.scrubbed_text.contains("AAAAC3Nza"));
    }

    #[test]
    fn test_scrub_github_token() {
        let scrubber = Scrubber::new();
        let result = scrubber.scrub("export GITHUB_TOKEN=ghp_1234567890abcdefghijklmnopqrstuvwxyz");
        assert!(result.scrubbed_text.contains("[REDACTED-KEY]"));
        assert!(!result.scrubbed_text.contains("ghp_"));
    }

    #[test]
    fn test_scrub_mac_address() {
        let scrubber = Scrubber::new();
        let result = scrubber.scrub("Interface wlan0 has MAC AA:BB:CC:DD:EE:FF");
        assert!(result.scrubbed_text.contains("[REDACTED-MAC]"));
        assert!(!result.scrubbed_text.contains("AA:BB:CC"));
    }

    #[test]
    fn test_scrub_hostname_in_config() {
        let scrubber = Scrubber::new();
        let result = scrubber.scrub(r#"networking.hostName = "my-secret-host""#);
        assert!(result.scrubbed_text.contains("[REDACTED-HOSTNAME]"));
        assert!(!result.scrubbed_text.contains("my-secret-host"));
    }

    #[test]
    fn test_scrub_preserves_non_pii() {
        let scrubber = Scrubber::new();
        let input = "error: builder for '/nix/store/abc123-glibc-2.38.drv' failed with exit code 1";
        let result = scrubber.scrub(input);
        assert_eq!(result.scrubbed_text, input);
        assert_eq!(result.redaction_count, 0);
    }

    #[test]
    fn test_scrub_multiple_types() {
        let scrubber = Scrubber::new();
        let input = "User /home/alice connected from 10.0.0.5 with email alice@example.org";
        let result = scrubber.scrub(input);
        assert!(!result.scrubbed_text.contains("alice"));
        assert!(!result.scrubbed_text.contains("10.0.0.5"));
        assert!(result.redaction_count >= 3);
        assert!(result.redaction_types.len() >= 2);
    }

    #[test]
    fn test_scrub_result_type_breakdown() {
        let scrubber = Scrubber::new();
        let result = scrubber.scrub("From 192.168.0.1 and 10.0.0.1");
        let ipv4_entry = result
            .redaction_types
            .iter()
            .find(|(name, _)| *name == "ipv4");
        assert!(ipv4_entry.is_some());
        assert_eq!(ipv4_entry.unwrap().1, 2);
    }

    #[test]
    fn test_scrub_empty_input() {
        let scrubber = Scrubber::new();
        let result = scrubber.scrub("");
        assert_eq!(result.scrubbed_text, "");
        assert_eq!(result.redaction_count, 0);
    }

    #[test]
    fn test_scrub_aws_key() {
        let scrubber = Scrubber::new();
        // Construct example key at runtime to avoid false-positive secret scanning
        let example_key = format!("{}IOSFODNN7EXAMPLE", concat!("AKI", "A"));
        let input = format!("AWS_ACCESS_KEY_ID={}", example_key);
        let result = scrubber.scrub(&input);
        assert!(result.scrubbed_text.contains("[REDACTED-KEY]"));
        assert!(!result.scrubbed_text.contains(&example_key));
    }

    #[test]
    fn test_scrub_no_double_count_overlapping() {
        let scrubber = Scrubber::new();
        // Home path contains a username that could partially look like other patterns.
        // Earlier patterns should not inflate counts for later patterns.
        let input = "error in /home/admin at 10.0.0.1";
        let result = scrubber.scrub(input);
        // home_path and ipv4 should each be counted once
        assert_eq!(result.redaction_count, 2);
        assert!(!result.scrubbed_text.contains("admin"));
        assert!(!result.scrubbed_text.contains("10.0.0.1"));
    }

    #[test]
    fn test_scrub_count_accuracy_single_type() {
        let scrubber = Scrubber::new();
        let input = "IPs: 10.0.0.1 and 10.0.0.2 and 10.0.0.3";
        let result = scrubber.scrub(input);
        let ipv4_entry = result.redaction_types.iter().find(|(n, _)| *n == "ipv4");
        assert_eq!(ipv4_entry.unwrap().1, 3);
        assert_eq!(result.scrubbed_text.matches("[REDACTED-IP]").count(), 3);
    }
}
