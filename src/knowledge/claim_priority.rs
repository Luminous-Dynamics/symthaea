// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Claim Prioritizer — Active Inference EFE for Knowledge Validation
//!
//! Uses the Expected Free Energy (EFE) decomposition from Active Inference
//! to prioritize which claims should be fact-checked first.
//!
//! EFE(claim) = epistemic_value + pragmatic_value
//!
//! - **Epistemic value**: How much uncertainty would be reduced by validating
//!   this claim? High uncertainty in important domains = high epistemic value.
//! - **Pragmatic value**: What are the consequences of this claim being wrong?
//!   Health/safety claims have higher pragmatic value than trivia.
//!
//! The prioritizer respects the thermodynamic energy budget — validation
//! has a real cost, so we can't check everything.
//!
//! ## Science
//!
//! - Friston et al. (2015): Active inference and epistemic value
//! - Parr & Friston (2019): Generalised free energy and active inference
//! - arXiv 2504.14898 (2025): EFE-based planning as variational inference

use serde::{Deserialize, Serialize};

/// A claim pending validation with its computed priority.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrioritizedClaim {
    /// The claim text
    pub claim: String,
    /// Domain of the claim (e.g. "health", "physics", "governance")
    pub domain: String,
    /// Epistemic value: uncertainty reduction from validating (0.0-1.0)
    pub epistemic_value: f64,
    /// Pragmatic value: consequence of the claim being wrong (0.0-1.0)
    pub pragmatic_value: f64,
    /// Combined EFE priority (higher = validate first)
    pub priority: f64,
}

/// Configuration for claim prioritization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrioritizerConfig {
    /// Weight for epistemic component in combined priority (default: 0.6)
    pub epistemic_weight: f64,
    /// Weight for pragmatic component (default: 0.4)
    pub pragmatic_weight: f64,
    /// Maximum claims to prioritize per cycle (energy budget)
    pub max_per_cycle: usize,
    /// Minimum priority threshold to submit for validation
    pub min_priority_threshold: f64,
}

impl Default for PrioritizerConfig {
    fn default() -> Self {
        Self {
            epistemic_weight: 0.6,
            pragmatic_weight: 0.4,
            max_per_cycle: 4,
            min_priority_threshold: 0.2,
        }
    }
}

/// High-consequence domains and their pragmatic value multipliers.
const HIGH_STAKES_DOMAINS: &[(&str, f64)] = &[
    ("health", 0.95),
    ("safety", 0.95),
    ("medical", 0.90),
    ("governance", 0.80),
    ("legal", 0.80),
    ("financial", 0.75),
    ("environmental", 0.70),
    ("education", 0.60),
];

/// Prioritizes claims for validation using Active Inference EFE.
#[derive(Debug, Clone)]
pub struct ClaimPrioritizer {
    config: PrioritizerConfig,
    /// Domain-level uncertainty estimates (0.0 = certain, 1.0 = unknown)
    domain_uncertainty: std::collections::HashMap<String, f64>,
}

impl ClaimPrioritizer {
    pub fn new(config: PrioritizerConfig) -> Self {
        Self {
            config,
            domain_uncertainty: std::collections::HashMap::new(),
        }
    }

    /// Update uncertainty estimate for a domain after a fact-check result.
    pub fn update_domain_uncertainty(&mut self, domain: &str, was_correct: bool) {
        let current = self.domain_uncertainty.get(domain).copied().unwrap_or(0.5);
        let update = if was_correct {
            // Correct fact-check reduces uncertainty
            current * 0.95
        } else {
            // Incorrect raises uncertainty
            (current + 0.1).min(1.0)
        };
        self.domain_uncertainty.insert(domain.to_string(), update);
    }

    /// Compute epistemic value for a claim in a domain.
    /// High uncertainty domains yield high epistemic value.
    fn epistemic_value(&self, domain: &str) -> f64 {
        self.domain_uncertainty.get(domain).copied().unwrap_or(0.5)
    }

    /// Compute pragmatic value based on domain stakes.
    fn pragmatic_value(&self, domain: &str) -> f64 {
        let domain_lower = domain.to_lowercase();
        for (key, value) in HIGH_STAKES_DOMAINS {
            if domain_lower.contains(key) {
                return *value;
            }
        }
        0.3 // Default for unrecognized domains
    }

    /// Infer domain from claim text (simple keyword matching).
    pub fn infer_domain(claim: &str) -> String {
        let lower = claim.to_lowercase();
        for (keyword, _) in HIGH_STAKES_DOMAINS {
            if lower.contains(keyword) {
                return keyword.to_string();
            }
        }
        "general".to_string()
    }

    /// Prioritize a batch of claims, returning them sorted by priority (highest first).
    ///
    /// Only returns claims above the minimum priority threshold,
    /// limited by the per-cycle energy budget.
    pub fn prioritize(&self, claims: &[String]) -> Vec<PrioritizedClaim> {
        let mut prioritized: Vec<PrioritizedClaim> = claims
            .iter()
            .map(|claim| {
                let domain = Self::infer_domain(claim);
                let epistemic = self.epistemic_value(&domain);
                let pragmatic = self.pragmatic_value(&domain);
                let priority = epistemic * self.config.epistemic_weight
                    + pragmatic * self.config.pragmatic_weight;
                PrioritizedClaim {
                    claim: claim.clone(),
                    domain,
                    epistemic_value: epistemic,
                    pragmatic_value: pragmatic,
                    priority,
                }
            })
            .filter(|p| p.priority >= self.config.min_priority_threshold)
            .collect();

        prioritized.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap());
        prioritized.truncate(self.config.max_per_cycle);
        prioritized
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_claims_prioritized_higher() {
        let prioritizer = ClaimPrioritizer::new(PrioritizerConfig::default());
        let claims = vec![
            "The weather is nice today in Paris.".to_string(),
            "This medication can cause liver damage.".to_string(),
        ];
        let result = prioritizer.prioritize(&claims);
        assert!(!result.is_empty());
        // Health claim should be first (higher pragmatic value)
        assert!(result[0].claim.contains("medication"));
    }

    #[test]
    fn test_high_uncertainty_domain_gets_priority() {
        let mut prioritizer = ClaimPrioritizer::new(PrioritizerConfig::default());
        prioritizer
            .domain_uncertainty
            .insert("speculative".to_string(), 0.9);
        prioritizer
            .domain_uncertainty
            .insert("general".to_string(), 0.1);

        let epistemic_high = prioritizer.epistemic_value("speculative");
        let epistemic_low = prioritizer.epistemic_value("general");
        assert!(epistemic_high > epistemic_low);
    }

    #[test]
    fn test_max_per_cycle_respected() {
        let config = PrioritizerConfig {
            max_per_cycle: 2,
            ..Default::default()
        };
        let prioritizer = ClaimPrioritizer::new(config);
        let claims: Vec<String> = (0..10)
            .map(|i| format!("Health claim number {} about safety concerns.", i))
            .collect();
        let result = prioritizer.prioritize(&claims);
        assert!(result.len() <= 2);
    }

    #[test]
    fn test_domain_inference() {
        assert_eq!(
            ClaimPrioritizer::infer_domain("This is about health risks"),
            "health"
        );
        assert_eq!(
            ClaimPrioritizer::infer_domain("A legal precedent was set"),
            "legal"
        );
        assert_eq!(
            ClaimPrioritizer::infer_domain("Random fact about cats"),
            "general"
        );
    }

    #[test]
    fn test_uncertainty_update_correct_reduces() {
        let mut prioritizer = ClaimPrioritizer::new(PrioritizerConfig::default());
        prioritizer
            .domain_uncertainty
            .insert("test".to_string(), 0.8);
        prioritizer.update_domain_uncertainty("test", true);
        assert!(prioritizer.domain_uncertainty["test"] < 0.8);
    }

    #[test]
    fn test_uncertainty_update_incorrect_increases() {
        let mut prioritizer = ClaimPrioritizer::new(PrioritizerConfig::default());
        prioritizer
            .domain_uncertainty
            .insert("test".to_string(), 0.5);
        prioritizer.update_domain_uncertainty("test", false);
        assert!(prioritizer.domain_uncertainty["test"] > 0.5);
    }
}
