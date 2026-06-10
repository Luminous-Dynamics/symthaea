// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Diagnostic Planner — Orders diagnostics by expected information gain

use crate::types::*;

pub struct DiagnosticPlanner;

impl DiagnosticPlanner {
    pub fn new() -> Self {
        Self
    }

    /// Plan diagnostics based on reported symptoms.
    /// Orders by expected information gain (highest first).
    pub fn plan_diagnostics(&self, symptoms: &[String]) -> DiagnosticPlan {
        let text = symptoms.join(" ").to_lowercase();
        let mut steps = Vec::new();

        // Always start with service status check (highest baseline info gain)
        steps.push(DiagnosticStep {
            diagnostic_type: DiagnosticType::ServiceStatus,
            description: "Check status of all core services".to_string(),
            expected_info_gain: 0.8,
            autonomous_capable: true,
        });

        if text.contains("network")
            || text.contains("connection")
            || text.contains("dns")
            || text.contains("timeout")
        {
            steps.push(DiagnosticStep {
                diagnostic_type: DiagnosticType::NetworkCheck,
                description: "Run network connectivity and DNS resolution tests".to_string(),
                expected_info_gain: 0.9,
                autonomous_capable: true,
            });
        }

        if text.contains("disk")
            || text.contains("space")
            || text.contains("storage")
            || text.contains("full")
        {
            steps.push(DiagnosticStep {
                diagnostic_type: DiagnosticType::DiskSpace,
                description: "Check disk space usage across partitions".to_string(),
                expected_info_gain: 0.85,
                autonomous_capable: true,
            });
        }

        if text.contains("memory")
            || text.contains("oom")
            || text.contains("swap")
            || text.contains("ram")
        {
            steps.push(DiagnosticStep {
                diagnostic_type: DiagnosticType::MemoryUsage,
                description: "Analyze memory usage and swap activity".to_string(),
                expected_info_gain: 0.75,
                autonomous_capable: true,
            });
        }

        if text.contains("holochain")
            || text.contains("conductor")
            || text.contains("dht")
            || text.contains("gossip")
        {
            steps.push(DiagnosticStep {
                diagnostic_type: DiagnosticType::HolochainHealth,
                description: "Check Holochain conductor health, peer count, and gossip status"
                    .to_string(),
                expected_info_gain: 0.95,
                autonomous_capable: true,
            });
        }

        // Sort by expected info gain (descending)
        steps.sort_by(|a, b| {
            b.expected_info_gain
                .partial_cmp(&a.expected_info_gain)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let confidence = if steps.len() > 2 {
            0.8
        } else if steps.len() > 1 {
            0.6
        } else {
            0.4
        };

        DiagnosticPlan {
            steps,
            expected_resolution_confidence: confidence,
        }
    }
}

impl Default for DiagnosticPlanner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn network_symptoms_produce_network_check_first() {
        let planner = DiagnosticPlanner::new();
        let plan = planner.plan_diagnostics(&[
            "network timeout".to_string(),
            "dns resolution fails".to_string(),
        ]);
        assert_eq!(plan.steps[0].diagnostic_type, DiagnosticType::NetworkCheck);
    }

    #[test]
    fn empty_symptoms_still_has_service_status() {
        let planner = DiagnosticPlanner::new();
        let plan = planner.plan_diagnostics(&[]);
        assert_eq!(plan.steps.len(), 1);
        assert_eq!(plan.steps[0].diagnostic_type, DiagnosticType::ServiceStatus);
    }

    #[test]
    fn multiple_symptoms_produce_multiple_steps() {
        let planner = DiagnosticPlanner::new();
        let plan = planner.plan_diagnostics(&[
            "network timeout".to_string(),
            "disk full".to_string(),
            "memory oom".to_string(),
        ]);
        // Should have ServiceStatus + NetworkCheck + DiskSpace + MemoryUsage = 4 steps
        assert!(plan.steps.len() >= 4);
    }

    #[test]
    fn steps_sorted_by_info_gain_descending() {
        let planner = DiagnosticPlanner::new();
        let plan = planner.plan_diagnostics(&[
            "network issues".to_string(),
            "disk full".to_string(),
            "memory oom".to_string(),
            "holochain conductor".to_string(),
        ]);
        for i in 0..plan.steps.len() - 1 {
            assert!(
                plan.steps[i].expected_info_gain >= plan.steps[i + 1].expected_info_gain,
                "Steps not sorted: {} >= {} failed at index {}",
                plan.steps[i].expected_info_gain,
                plan.steps[i + 1].expected_info_gain,
                i
            );
        }
    }
}
