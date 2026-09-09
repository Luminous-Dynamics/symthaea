// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Diagnostic Planner — legacy symptom heuristics plus model-based information gain.
//!
//! `plan_diagnostics` preserves the existing lightweight symptom router. Its
//! `expected_info_gain` values are heuristic priorities, not entropy-derived
//! information gain. For true expected information gain, use
//! `rank_modelled_tests`, which delegates to the Bayesian diagnostic-belief kernel.

use crate::diagnostic_beliefs::{
    rank_tests_by_information_gain, DiagnosticBeliefError, DiagnosticTestModelV1,
    ExpectedInformationGainV1, HypothesisDistributionV1,
};
use crate::types::*;

pub struct DiagnosticPlanner;

impl DiagnosticPlanner {
    pub fn new() -> Self {
        Self
    }

    /// Plan diagnostics from reported symptoms using the legacy fixed-priority
    /// heuristic. This remains intentionally cheap and backwards-compatible.
    pub fn plan_diagnostics(&self, symptoms: &[String]) -> DiagnosticPlan {
        let text = symptoms.join(" ").to_lowercase();
        let mut steps = Vec::new();

        // Heuristic priority only; not a measured Shannon information gain.
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

    /// Rank tests using actual expected entropy reduction under an explicit
    /// hypothesis prior and P(outcome | hypothesis) likelihood models.
    pub fn rank_modelled_tests(
        &self,
        prior: &HypothesisDistributionV1,
        tests: &[DiagnosticTestModelV1],
    ) -> Result<Vec<ExpectedInformationGainV1>, DiagnosticBeliefError> {
        rank_tests_by_information_gain(prior, tests)
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
    use crate::diagnostic_beliefs::{DiagnosticOutcomeId, DiagnosticTestId, HypothesisId};
    use std::collections::BTreeMap;

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
        assert!(plan.steps.len() >= 4);
    }

    #[test]
    fn legacy_steps_remain_sorted_by_heuristic_priority() {
        let planner = DiagnosticPlanner::new();
        let plan = planner.plan_diagnostics(&[
            "network issues".to_string(),
            "disk full".to_string(),
            "memory oom".to_string(),
            "holochain conductor".to_string(),
        ]);
        for i in 0..plan.steps.len() - 1 {
            assert!(plan.steps[i].expected_info_gain >= plan.steps[i + 1].expected_info_gain);
        }
    }

    #[test]
    fn modelled_path_uses_true_information_gain() {
        let dns = HypothesisId("dns".into());
        let server = HypothesisId("server".into());
        let prior = HypothesisDistributionV1::from_weights([
            (dns.clone(), 1.0),
            (server.clone(), 1.0),
        ])
        .unwrap();
        let test = DiagnosticTestModelV1 {
            id: DiagnosticTestId("dns-query".into()),
            description: "Query configured resolver".into(),
            outcomes: BTreeMap::from([
                (
                    DiagnosticOutcomeId("fail".into()),
                    BTreeMap::from([(dns.clone(), 0.9), (server.clone(), 0.1)]),
                ),
                (
                    DiagnosticOutcomeId("pass".into()),
                    BTreeMap::from([(dns, 0.1), (server, 0.9)]),
                ),
            ]),
        };

        let ranked = DiagnosticPlanner::new()
            .rank_modelled_tests(&prior, &[test])
            .unwrap();
        assert_eq!(ranked.len(), 1);
        assert!(ranked[0].information_gain_bits > 0.0);
        assert!(ranked[0].expected_posterior_entropy_bits < ranked[0].prior_entropy_bits);
    }
}
