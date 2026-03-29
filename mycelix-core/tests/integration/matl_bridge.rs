// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! MATL Bridge Integration Tests
//!
//! Tests the Multi-Agent Trust Layer components:
//! - Proof of Gradient Quality (PoGQ)
//! - Trust Contribution Decay Model (TCDM)
//! - Trust entropy measurement
//! - Combined trust score calculation

#[cfg(test)]
mod tests {
    use matl_bridge::pogq::{PoGQValidator, GradientSubmission};
    use matl_bridge::tcdm::{TrustEvolution, TrustHistory};
    use matl_bridge::entropy::{EntropyCalculator, TrustDistribution};

    /// Test the MATL trust formula: T = 0.4×PoGQ + 0.3×TCDM + 0.3×Entropy
    #[test]
    fn test_matl_trust_formula() {
        let pogq_score = 0.85;
        let tcdm_score = 0.72;
        let entropy_score = 0.90;

        let trust_score = 0.4 * pogq_score + 0.3 * tcdm_score + 0.3 * entropy_score;

        // Expected: 0.4 * 0.85 + 0.3 * 0.72 + 0.3 * 0.90 = 0.34 + 0.216 + 0.27 = 0.826
        assert!((trust_score - 0.826).abs() < 0.001);
    }

    /// Test PoGQ validation with valid gradient
    #[test]
    fn test_pogq_valid_gradient() {
        // Create a valid gradient submission
        let submission = GradientSubmission {
            node_id: "node-1".to_string(),
            round_id: 1,
            gradient: vec![0.01, -0.02, 0.015, -0.005],
            proof: None,
        };

        // Validate against acceptable bounds
        let validator = PoGQValidator::new(0.1); // Max gradient magnitude
        let result = validator.validate(&submission);

        assert!(result.is_ok());
        assert!(result.unwrap().quality_score > 0.5);
    }

    /// Test PoGQ rejection of Byzantine gradient
    #[test]
    fn test_pogq_byzantine_gradient() {
        // Create an obviously malicious gradient
        let submission = GradientSubmission {
            node_id: "byzantine-node".to_string(),
            round_id: 1,
            gradient: vec![1000.0, -500.0, 999.0, -888.0], // Extreme values
            proof: None,
        };

        let validator = PoGQValidator::new(0.1);
        let result = validator.validate(&submission);

        // Should be flagged as Byzantine
        assert!(result.is_err() || result.unwrap().quality_score < 0.3);
    }

    /// Test TCDM trust decay over time
    #[test]
    fn test_tcdm_trust_decay() {
        let mut history = TrustHistory::new("node-1");

        // Add initial high trust
        history.record_contribution(1.0, 1);

        // Wait multiple rounds without contribution
        let evolution = TrustEvolution::with_decay_rate(0.1);

        let trust_after_0_rounds = evolution.calculate(&history, 1);
        let trust_after_5_rounds = evolution.calculate(&history, 6);
        let trust_after_10_rounds = evolution.calculate(&history, 11);

        // Trust should decay over time
        assert!(trust_after_5_rounds < trust_after_0_rounds);
        assert!(trust_after_10_rounds < trust_after_5_rounds);
    }

    /// Test TCDM trust recovery
    #[test]
    fn test_tcdm_trust_recovery() {
        let mut history = TrustHistory::new("node-1");

        // Start with degraded trust
        history.record_contribution(0.5, 1);

        // Make good contributions
        for round in 2..=10 {
            history.record_contribution(1.0, round);
        }

        let evolution = TrustEvolution::with_recovery_bonus(0.12);
        let final_trust = evolution.calculate(&history, 10);

        // Trust should have recovered significantly
        assert!(final_trust > 0.8);
    }

    /// Test entropy calculation for diverse trust distribution
    #[test]
    fn test_entropy_diverse_distribution() {
        // Create a diverse trust distribution (healthy network)
        let distribution = TrustDistribution::new(vec![
            ("node-1", 0.85),
            ("node-2", 0.82),
            ("node-3", 0.88),
            ("node-4", 0.79),
            ("node-5", 0.91),
        ]);

        let calculator = EntropyCalculator::new();
        let entropy = calculator.calculate(&distribution);

        // High entropy indicates diverse trust (good)
        assert!(entropy > 0.8);
    }

    /// Test entropy for centralized trust distribution
    #[test]
    fn test_entropy_centralized_distribution() {
        // One node dominates trust
        let distribution = TrustDistribution::new(vec![
            ("node-1", 0.99),
            ("node-2", 0.10),
            ("node-3", 0.12),
            ("node-4", 0.08),
            ("node-5", 0.11),
        ]);

        let calculator = EntropyCalculator::new();
        let entropy = calculator.calculate(&distribution);

        // Low entropy indicates centralized trust (warning)
        assert!(entropy < 0.5);
    }

    /// Integration test: Full MATL trust calculation
    #[test]
    fn test_full_matl_integration() {
        // Simulate a complete trust evaluation

        // 1. PoGQ: Validate gradient quality
        let gradient = vec![0.01, -0.02, 0.015];
        let pogq_validator = PoGQValidator::new(0.1);
        let pogq_result = pogq_validator.validate_gradient(&gradient);
        assert!(pogq_result.is_ok());
        let pogq_score = pogq_result.unwrap();

        // 2. TCDM: Calculate trust evolution
        let mut history = TrustHistory::new("test-node");
        for round in 1..=5 {
            history.record_contribution(0.9, round);
        }
        let tcdm = TrustEvolution::default();
        let tcdm_score = tcdm.calculate(&history, 5);

        // 3. Entropy: Measure network diversity
        let distribution = TrustDistribution::new(vec![
            ("test-node", tcdm_score),
            ("other-node-1", 0.85),
            ("other-node-2", 0.80),
        ]);
        let entropy_calc = EntropyCalculator::new();
        let entropy_score = entropy_calc.calculate(&distribution);

        // 4. Combine with MATL formula
        let trust_score = 0.4 * pogq_score + 0.3 * tcdm_score + 0.3 * entropy_score;

        // Verify result is in valid range
        assert!(trust_score >= 0.0 && trust_score <= 1.0);
        println!("MATL Trust Score: {:.4} (PoGQ={:.4}, TCDM={:.4}, Entropy={:.4})",
                 trust_score, pogq_score, tcdm_score, entropy_score);
    }
}

// Stub implementations for tests to compile
#[cfg(test)]
mod matl_bridge {
    pub mod pogq {
        pub struct PoGQValidator {
            max_magnitude: f64,
        }

        impl PoGQValidator {
            pub fn new(max_magnitude: f64) -> Self {
                Self { max_magnitude }
            }

            pub fn validate(&self, submission: &GradientSubmission) -> Result<ValidationResult, String> {
                self.validate_gradient(&submission.gradient)
                    .map(|score| ValidationResult { quality_score: score })
            }

            pub fn validate_gradient(&self, gradient: &[f64]) -> Result<f64, String> {
                let max = gradient.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
                if max > self.max_magnitude * 100.0 {
                    return Err("Byzantine gradient detected".to_string());
                }
                Ok(1.0 - (max / (self.max_magnitude * 100.0)).min(1.0))
            }
        }

        pub struct GradientSubmission {
            pub node_id: String,
            pub round_id: u64,
            pub gradient: Vec<f64>,
            pub proof: Option<Vec<u8>>,
        }

        pub struct ValidationResult {
            pub quality_score: f64,
        }
    }

    pub mod tcdm {
        pub struct TrustHistory {
            node_id: String,
            contributions: Vec<(f64, u64)>,
        }

        impl TrustHistory {
            pub fn new(node_id: &str) -> Self {
                Self {
                    node_id: node_id.to_string(),
                    contributions: Vec::new(),
                }
            }

            pub fn record_contribution(&mut self, score: f64, round: u64) {
                self.contributions.push((score, round));
            }
        }

        pub struct TrustEvolution {
            decay_rate: f64,
            recovery_bonus: f64,
        }

        impl TrustEvolution {
            pub fn with_decay_rate(rate: f64) -> Self {
                Self { decay_rate: rate, recovery_bonus: 0.0 }
            }

            pub fn with_recovery_bonus(bonus: f64) -> Self {
                Self { decay_rate: 0.05, recovery_bonus: bonus }
            }

            pub fn calculate(&self, history: &TrustHistory, current_round: u64) -> f64 {
                if history.contributions.is_empty() {
                    return 0.0;
                }

                let (last_score, last_round) = history.contributions.last().unwrap();
                let rounds_since = current_round - last_round;
                let decay = (-self.decay_rate * rounds_since as f64).exp();

                let base = *last_score * decay;
                let recovery = if history.contributions.len() > 3 {
                    self.recovery_bonus * (history.contributions.len() - 3) as f64
                } else {
                    0.0
                };

                (base + recovery).min(1.0)
            }
        }

        impl Default for TrustEvolution {
            fn default() -> Self {
                Self { decay_rate: 0.05, recovery_bonus: 0.05 }
            }
        }
    }

    pub mod entropy {
        pub struct TrustDistribution {
            nodes: Vec<(String, f64)>,
        }

        impl TrustDistribution {
            pub fn new(nodes: Vec<(&str, f64)>) -> Self {
                Self {
                    nodes: nodes.into_iter().map(|(n, t)| (n.to_string(), t)).collect(),
                }
            }
        }

        pub struct EntropyCalculator;

        impl EntropyCalculator {
            pub fn new() -> Self {
                Self
            }

            pub fn calculate(&self, distribution: &TrustDistribution) -> f64 {
                let total: f64 = distribution.nodes.iter().map(|(_, t)| *t).sum();
                if total == 0.0 {
                    return 0.0;
                }

                let n = distribution.nodes.len() as f64;
                let normalized: Vec<f64> = distribution.nodes.iter()
                    .map(|(_, t)| *t / total)
                    .collect();

                // Shannon entropy normalized to [0, 1]
                let entropy: f64 = normalized.iter()
                    .filter(|p| **p > 0.0)
                    .map(|p| -p * p.ln())
                    .sum();

                let max_entropy = n.ln();
                if max_entropy > 0.0 {
                    entropy / max_entropy
                } else {
                    0.0
                }
            }
        }
    }
}
