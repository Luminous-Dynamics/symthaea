// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Triage Engine — HDC-based ticket classification and prioritization

use crate::types::*;

/// Triage engine for classifying and prioritizing support tickets.
#[derive(Debug)]
pub struct TriageEngine {
    category_keywords: Vec<(SupportCategory, Vec<&'static str>)>,
}

impl TriageEngine {
    pub fn new() -> Self {
        Self {
            category_keywords: vec![
                (
                    SupportCategory::Network,
                    vec![
                        "network",
                        "wifi",
                        "dns",
                        "firewall",
                        "port",
                        "connectivity",
                        "ping",
                        "latency",
                    ],
                ),
                (
                    SupportCategory::Hardware,
                    vec![
                        "hardware", "disk", "memory", "cpu", "gpu", "device", "driver", "usb",
                    ],
                ),
                (
                    SupportCategory::Software,
                    vec![
                        "software", "install", "update", "crash", "bug", "error", "version",
                    ],
                ),
                (
                    SupportCategory::Holochain,
                    vec![
                        "holochain",
                        "conductor",
                        "dht",
                        "gossip",
                        "zome",
                        "happ",
                        "agent",
                    ],
                ),
                (
                    SupportCategory::Mycelix,
                    vec![
                        "mycelix",
                        "bridge",
                        "commons",
                        "civic",
                        "governance",
                        "federation",
                    ],
                ),
                (
                    SupportCategory::Security,
                    vec![
                        "security",
                        "password",
                        "encryption",
                        "key",
                        "certificate",
                        "auth",
                        "permission",
                    ],
                ),
                (
                    SupportCategory::General,
                    vec!["help", "question", "how", "what", "support"],
                ),
            ],
        }
    }

    /// Triage a support ticket based on title and description text.
    pub fn triage(&self, title: &str, description: &str) -> TriageResult {
        let text = format!("{} {}", title, description).to_lowercase();

        // Category classification via keyword matching
        let mut best_category = SupportCategory::General;
        let mut best_score = 0usize;

        for (category, keywords) in &self.category_keywords {
            let score: usize = keywords.iter().filter(|kw| text.contains(*kw)).count();
            if score > best_score {
                best_score = score;
                best_category = category.clone();
            }
        }

        // Priority heuristic based on urgency keywords
        let priority = if text.contains("critical")
            || text.contains("emergency")
            || text.contains("down")
            || text.contains("outage")
        {
            TicketPriority::Critical
        } else if text.contains("urgent")
            || text.contains("broken")
            || text.contains("cannot")
            || text.contains("blocked")
        {
            TicketPriority::High
        } else if text.contains("slow")
            || text.contains("intermittent")
            || text.contains("sometimes")
        {
            TicketPriority::Medium
        } else {
            TicketPriority::Low
        };

        let confidence = if best_score > 2 {
            0.9
        } else if best_score > 0 {
            0.6
        } else {
            0.3
        };

        TriageResult {
            suggested_priority: priority,
            suggested_category: best_category,
            similar_tickets: vec![],
            confidence,
            phi: 0.0, // Populated by caller with actual Phi from Symthaea
            suggested_articles: vec![],
            epistemic_status: if confidence > 0.7 {
                EpistemicStatus::Probable
            } else {
                EpistemicStatus::Uncertain
            },
        }
    }
}

impl Default for TriageEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triage_network_issue() {
        let engine = TriageEngine::new();
        let result = engine.triage(
            "Cannot connect to network",
            "WiFi keeps dropping, DNS resolution fails",
        );
        assert_eq!(result.suggested_category, SupportCategory::Network);
    }

    #[test]
    fn triage_critical_priority() {
        let engine = TriageEngine::new();
        let result = engine.triage("System down", "Critical outage, nothing works");
        assert_eq!(result.suggested_priority, TicketPriority::Critical);
    }

    #[test]
    fn triage_holochain_issue() {
        let engine = TriageEngine::new();
        let result = engine.triage(
            "Conductor error",
            "Holochain conductor fails to start, DHT not syncing",
        );
        assert_eq!(result.suggested_category, SupportCategory::Holochain);
    }

    #[test]
    fn triage_general_fallback() {
        let engine = TriageEngine::new();
        let result = engine.triage("Hello", "I have a question about something");
        assert_eq!(result.suggested_category, SupportCategory::General);
    }

    #[test]
    fn triage_low_priority_default() {
        let engine = TriageEngine::new();
        let result = engine.triage("Minor issue", "Small cosmetic problem");
        assert_eq!(result.suggested_priority, TicketPriority::Low);
    }

    #[test]
    fn triage_confidence_scales_with_matches() {
        let engine = TriageEngine::new();
        let high = engine.triage(
            "Network DNS firewall port",
            "connectivity ping latency issues",
        );
        let low = engine.triage("Problem", "Something is wrong");
        assert!(high.confidence > low.confidence);
    }
}
