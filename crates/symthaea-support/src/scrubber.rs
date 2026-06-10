// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Privacy Scrubber — Deep PII removal from text and JSON

use crate::types::ScrubResult;
use regex::Regex;
use std::collections::HashMap;

pub struct PrivacyScrubber {
    ip_regex: Regex,
    path_regex: Regex,
    email_regex: Regex,
    key_regex: Regex,
}

impl PrivacyScrubber {
    pub fn new() -> Self {
        Self {
            ip_regex: Regex::new(r"\b(?:\d{1,3}\.){3}\d{1,3}\b").unwrap(),
            path_regex: Regex::new(
                r"(?:/home/\w+[\w/.-]*|/Users/\w+[\w/.-]*|C:\\Users\\\w+[\w\\.-]*)",
            )
            .unwrap(),
            email_regex: Regex::new(r"\b[\w.+-]+@[\w.-]+\.\w{2,}\b").unwrap(),
            key_regex: Regex::new(r"\b(?:sk-|pk-|api-|token-|key-)[a-zA-Z0-9]{16,}\b").unwrap(),
        }
    }

    /// Scrub all PII from text, replacing with tagged placeholders.
    pub fn scrub(&self, text: &str) -> ScrubResult {
        let mut result = text.to_string();
        let mut redaction_types: HashMap<String, usize> = HashMap::new();
        let mut count = 0;

        // Order matters: scrub more specific patterns first
        let patterns: Vec<(&str, &Regex)> = vec![
            ("key", &self.key_regex),
            ("email", &self.email_regex),
            ("path", &self.path_regex),
            ("ip", &self.ip_regex),
        ];

        for (kind, regex) in patterns {
            let mut idx = 0;
            let scrubbed = regex.replace_all(&result, |_: &regex::Captures| {
                idx += 1;
                count += 1;
                *redaction_types.entry(kind.to_string()).or_insert(0) += 1;
                format!("[REDACTED_{}_{idx}]", kind.to_uppercase())
            });
            result = scrubbed.into_owned();
        }

        ScrubResult {
            scrubbed_text: result,
            redaction_count: count,
            redaction_types,
        }
    }

    /// Scrub structured JSON findings.
    pub fn scrub_json(&self, json: &str) -> ScrubResult {
        // Parse, walk all string values, scrub each one
        match serde_json::from_str::<serde_json::Value>(json) {
            Ok(value) => {
                let scrubbed_str = serde_json::to_string(&value).unwrap_or_default();
                self.scrub(&scrubbed_str)
            }
            Err(_) => self.scrub(json),
        }
    }
}

impl Default for PrivacyScrubber {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scrub_removes_ips() {
        let scrubber = PrivacyScrubber::new();
        let result = scrubber.scrub("Server at 192.168.1.1 is down, also check 10.0.0.5");
        assert!(!result.scrubbed_text.contains("192.168.1.1"));
        assert!(!result.scrubbed_text.contains("10.0.0.5"));
        assert!(result.scrubbed_text.contains("[REDACTED_IP_"));
        assert_eq!(result.redaction_types["ip"], 2);
    }

    #[test]
    fn scrub_removes_paths() {
        let scrubber = PrivacyScrubber::new();
        let result = scrubber.scrub("Config at /home/tstoltz/.config/mycelix/conductor.yaml");
        assert!(!result.scrubbed_text.contains("/home/tstoltz"));
        assert!(result.scrubbed_text.contains("[REDACTED_PATH_"));
    }

    #[test]
    fn scrub_removes_emails() {
        let scrubber = PrivacyScrubber::new();
        let result = scrubber.scrub("Contact admin@example.com for help");
        assert!(!result.scrubbed_text.contains("admin@example.com"));
        assert!(result.scrubbed_text.contains("[REDACTED_EMAIL_"));
        assert_eq!(result.redaction_types["email"], 1);
    }

    #[test]
    fn scrub_removes_api_keys() {
        let scrubber = PrivacyScrubber::new();
        let result = scrubber.scrub("Using key sk-abcdefghijklmnopqr for auth");
        assert!(!result.scrubbed_text.contains("sk-abcdefghijklmnopqr"));
        assert!(result.scrubbed_text.contains("[REDACTED_KEY_"));
        assert_eq!(result.redaction_types["key"], 1);
    }

    #[test]
    fn scrub_json_works() {
        let scrubber = PrivacyScrubber::new();
        let json = r#"{"host":"192.168.1.1","user":"admin@example.com"}"#;
        let result = scrubber.scrub_json(json);
        assert!(!result.scrubbed_text.contains("192.168.1.1"));
        assert!(!result.scrubbed_text.contains("admin@example.com"));
        assert!(result.redaction_count >= 2);
    }

    #[test]
    fn scrub_counts_are_correct() {
        let scrubber = PrivacyScrubber::new();
        let result = scrubber
            .scrub("IPs: 1.2.3.4 and 5.6.7.8, email: user@test.com, key: api-ABCDEFGHIJKLMNOP");
        assert_eq!(result.redaction_count, 4);
        assert_eq!(result.redaction_types["ip"], 2);
        assert_eq!(result.redaction_types["email"], 1);
        assert_eq!(result.redaction_types["key"], 1);
    }

    #[test]
    fn text_without_pii_is_unchanged() {
        let scrubber = PrivacyScrubber::new();
        let input = "The holochain conductor failed to start due to a configuration error";
        let result = scrubber.scrub(input);
        assert_eq!(result.scrubbed_text, input);
        assert_eq!(result.redaction_count, 0);
        assert!(result.redaction_types.is_empty());
    }
}
