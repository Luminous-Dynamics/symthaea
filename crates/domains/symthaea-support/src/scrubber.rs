// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Privacy Scrubber — Deep PII removal from text and JSON

use crate::types::ScrubResult;
use regex::Regex;
use std::collections::HashMap;

pub struct PrivacyScrubber {
    ip_regex: Regex,
    ipv6_regex: Regex,
    mac_regex: Regex,
    path_regex: Regex,
    email_regex: Regex,
    key_regex: Regex,
    cloud_key_regex: Regex,
    jwt_regex: Regex,
    phone_regex: Regex,
    ssn_regex: Regex,
    credit_card_regex: Regex,
}

impl PrivacyScrubber {
    pub fn new() -> Self {
        Self {
            ip_regex: Regex::new(r"\b(?:\d{1,3}\.){3}\d{1,3}\b").unwrap(),
            // Uncompressed-ish form requires >=4 colon-separated hex groups
            // (not just 2-3, which would false-positive on HH:MM:SS
            // timestamps like "14:23:07" -- every digit is valid hex), or
            // an explicit "::" compression marker for the compressed form
            // (e.g. 2001:db8::1).
            ipv6_regex: Regex::new(
                r"\b(?:(?:[0-9a-fA-F]{1,4}:){3,7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:)+:[0-9a-fA-F]{0,4}(?::[0-9a-fA-F]{1,4})*)\b",
            )
            .unwrap(),
            mac_regex: Regex::new(r"\b[0-9a-fA-F]{2}(?:[:-][0-9a-fA-F]{2}){5}\b").unwrap(),
            path_regex: Regex::new(
                r"(?:/home/\w+[\w/.-]*|/Users/\w+[\w/.-]*|C:\\Users\\\w+[\w\\.-]*)",
            )
            .unwrap(),
            email_regex: Regex::new(r"\b[\w.+-]+@[\w.-]+\.\w{2,}\b").unwrap(),
            key_regex: Regex::new(r"\b(?:sk-|pk-|api-|token-|key-)[a-zA-Z0-9]{16,}\b").unwrap(),
            // Common cloud/service credential formats that don't use the
            // generic sk-/pk-/api-/token-/key- prefix convention above.
            cloud_key_regex: Regex::new(
                r"\b(?:AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{36,}|xox[baprs]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{35})\b",
            )
            .unwrap(),
            jwt_regex: Regex::new(
                r"\beyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b",
            )
            .unwrap(),
            // Loose international/US phone format: optional +country code,
            // then 3 groups of digits separated by space/dot/dash.
            phone_regex: Regex::new(
                r"\b(?:\+\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b",
            )
            .unwrap(),
            ssn_regex: Regex::new(r"\b\d{3}-\d{2}-\d{4}\b").unwrap(),
            // Issuer-prefix patterns rather than a bare 13-19-digit run,
            // which would false-positive on timestamps, byte counts, and
            // other long numeric IDs common in technical/log text. Optional
            // space/dash separators every 4 digits (or Amex's 4-6-5
            // grouping) match common formatting. Alternatives, in order:
            // Visa, Mastercard, Amex, Discover.
            credit_card_regex: Regex::new(
                r"\b(?:4\d{3}(?:[ -]?\d{4}){2}[ -]?\d{1,4}|5[1-5]\d{2}(?:[ -]?\d{4}){3}|3[47]\d{2}[ -]?\d{6}[ -]?\d{5}|6(?:011|5\d{2})(?:[ -]?\d{4}){3})\b",
            )
            .unwrap(),
        }
    }

    /// Scrub all PII from text, replacing with tagged placeholders.
    pub fn scrub(&self, text: &str) -> ScrubResult {
        let mut result = text.to_string();
        let mut redaction_types: HashMap<String, usize> = HashMap::new();
        let mut count = 0;

        // Order matters: scrub more specific/structured patterns before
        // looser ones that could otherwise partially match inside them
        // (e.g. credit_card's bare-digit-run pattern must run after ip/mac/
        // phone/ssn, all of which are digit runs with their own separators).
        let patterns: Vec<(&str, &Regex)> = vec![
            ("jwt", &self.jwt_regex),
            ("cloud_key", &self.cloud_key_regex),
            ("key", &self.key_regex),
            ("email", &self.email_regex),
            ("path", &self.path_regex),
            ("mac", &self.mac_regex),
            ("ipv6", &self.ipv6_regex),
            ("ip", &self.ip_regex),
            ("ssn", &self.ssn_regex),
            ("phone", &self.phone_regex),
            ("credit_card", &self.credit_card_regex),
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

    #[test]
    fn timestamps_are_not_false_flagged_as_ipv6() {
        let scrubber = PrivacyScrubber::new();
        let result = scrubber.scrub("Request logged at 14:23:07 and again at 09:15:42");
        assert_eq!(
            result.scrubbed_text,
            "Request logged at 14:23:07 and again at 09:15:42"
        );
        assert_eq!(result.redaction_count, 0);
    }

    #[test]
    fn scrub_removes_ipv6() {
        let scrubber = PrivacyScrubber::new();
        let result = scrubber.scrub("Node reachable at 2001:0db8:85a3:0000:0000:8a2e:0370:7334");
        assert!(!result.scrubbed_text.contains("2001:0db8"));
        assert!(result.scrubbed_text.contains("[REDACTED_IPV6_"));

        let compressed = scrubber.scrub("Loopback is ::1 and node is at 2001:db8::1");
        assert!(!compressed.scrubbed_text.contains("2001:db8::1"));
    }

    #[test]
    fn scrub_removes_mac_addresses() {
        let scrubber = PrivacyScrubber::new();
        let result = scrubber.scrub("Interface eth0 has MAC 00:1B:44:11:3A:B7");
        assert!(!result.scrubbed_text.contains("00:1B:44:11:3A:B7"));
        assert!(result.scrubbed_text.contains("[REDACTED_MAC_"));
    }

    #[test]
    fn scrub_removes_cloud_provider_keys() {
        let scrubber = PrivacyScrubber::new();
        let aws = scrubber.scrub("Using AKIAIOSFODNN7EXAMPLE for S3 access");
        assert!(!aws.scrubbed_text.contains("AKIAIOSFODNN7EXAMPLE"));
        assert!(aws.scrubbed_text.contains("[REDACTED_CLOUD_KEY_"));

        let gh = scrubber.scrub("token: ghp_1234567890abcdefghijklmnopqrstuvwxyzAB");
        assert!(
            !gh.scrubbed_text
                .contains("ghp_1234567890abcdefghijklmnopqrstuvwxyzAB")
        );
    }

    #[test]
    fn scrub_removes_jwt_tokens() {
        let scrubber = PrivacyScrubber::new();
        let jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dQw4w9WgXcQ";
        let result = scrubber.scrub(&format!("Authorization: Bearer {jwt}"));
        assert!(!result.scrubbed_text.contains(jwt));
        assert!(result.scrubbed_text.contains("[REDACTED_JWT_"));
    }

    #[test]
    fn scrub_removes_phone_numbers() {
        let scrubber = PrivacyScrubber::new();
        let result = scrubber.scrub("Call me at 555-867-5309 or +1 555-867-5309");
        assert!(!result.scrubbed_text.contains("555-867-5309"));
        assert!(result.scrubbed_text.contains("[REDACTED_PHONE_"));
    }

    #[test]
    fn scrub_removes_ssn() {
        let scrubber = PrivacyScrubber::new();
        let result = scrubber.scrub("SSN on file: 123-45-6789");
        assert!(!result.scrubbed_text.contains("123-45-6789"));
        assert!(result.scrubbed_text.contains("[REDACTED_SSN_"));
    }

    #[test]
    fn scrub_removes_credit_card_numbers() {
        let scrubber = PrivacyScrubber::new();
        let visa = scrubber.scrub("Card on file: 4111 1111 1111 1111");
        assert!(!visa.scrubbed_text.contains("4111 1111 1111 1111"));
        assert!(visa.scrubbed_text.contains("[REDACTED_CREDIT_CARD_"));

        let amex = scrubber.scrub("Amex: 3782 822463 10005");
        assert!(!amex.scrubbed_text.contains("3782 822463 10005"));
    }

    #[test]
    fn long_non_card_numbers_are_not_false_flagged() {
        let scrubber = PrivacyScrubber::new();
        // A byte count / generic large ID that happens to be long, but
        // doesn't match any known card-issuer prefix.
        let result = scrubber.scrub("Processed 9876543210987654321 bytes total");
        assert_eq!(result.redaction_types.get("credit_card"), None);
    }
}
