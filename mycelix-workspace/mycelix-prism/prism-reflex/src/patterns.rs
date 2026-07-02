// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Threat pattern definitions for the reflex arc.
//!
//! Uses multi-word phrases to minimize false positives on legitimate content.
//! Single common words (attack, bypass, exploit) are NOT included because they
//! appear frequently in security documentation, changelogs, and educational content.

use prism_common::ThreatType;
use serde::{Deserialize, Serialize};

/// A detected threat match from the Aho-Corasick scan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatMatch {
    pub threat_type: ThreatType,
    pub keyword: String,
    pub confidence: f32,
}

/// Build the keyword list and pattern map for Aho-Corasick.
pub fn build_patterns() -> (Vec<String>, Vec<(ThreatType, f32)>) {
    let patterns: &[(&[&str], ThreatType, f32)] = &[
        // Adversarial — prompt injection / override attempts
        // Multi-word phrases only to avoid false positives on "bypass" in docs
        (
            &[
                "ignore previous instructions",
                "ignore all instructions",
                "disregard all previous",
                "forget everything above",
                "new instructions:",
                "you are now",
                "jailbreak",
                "override safety",
                "override the safety",
                "override all security",
                "bypass security",
                "bypass the filter",
                "bypass restrictions",
            ],
            ThreatType::Adversarial,
            0.8,
        ),
        // Harmful — multi-word phrases for real threats
        (
            &[
                "how to exploit",
                "how to attack",
                "how to hack",
                "how to destroy",
                "build a weapon",
                "create malware",
                "create ransomware",
                "launch a ddos",
                "sql injection attack",
                "buffer overflow exploit",
                "remote code execution",
                "privilege escalation attack",
            ],
            ThreatType::Harmful,
            0.7,
        ),
        // Deceptive — social engineering
        (
            &[
                "pretend you are",
                "roleplay as",
                "hypothetically imagine",
                "for educational purposes only",
                "in a fictional scenario",
                "act as if you were",
            ],
            ThreatType::Deceptive,
            0.5,
        ),
        // Boundary — credential/privacy probing
        // "password" alone triggers on every login form — use phrases instead
        (
            &[
                "enter your password",
                "enter your ssn",
                "your social security number",
                "your credit card number",
                "your bank account",
                "your api key",
                "your secret key",
                "your private key",
                "send me your password",
                "provide your credentials",
            ],
            ThreatType::Boundary,
            0.6,
        ),
    ];

    let mut keywords = Vec::new();
    let mut pattern_map = Vec::new();

    for &(kws, threat_type, weight) in patterns {
        for kw in kws {
            keywords.push(kw.to_string());
            pattern_map.push((threat_type, weight));
        }
    }

    (keywords, pattern_map)
}
