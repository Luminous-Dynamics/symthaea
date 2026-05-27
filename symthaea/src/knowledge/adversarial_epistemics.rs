// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Adversarial Epistemic Training
//!
//! Generates adversarial epistemic examples to train and harden the immune
//! system against common knowledge-manipulation attacks:
//!
//! - **Circular corroboration** — claims that cite each other in a closed loop
//! - **Confidence inflation** — claims with stated confidence far above actual evidence
//! - **Source diversity collapse** — many claims originating from few real sources
//! - **Gradual drift** — slow, incremental distortion that evades per-step detection
//! - **Authority mimicry** — claims that impersonate high-authority sources
//!
//! # Usage
//!
//! ```rust,ignore
//! use symthaea::knowledge::adversarial_epistemics::AdversarialGenerator;
//!
//! let r#gen = AdversarialGenerator::new();
//! let examples = r#gen.generate_examples(20);
//! for ex in &examples {
//!     assert!(ex.expected_detection, "immune system should catch this");
//! }
//! ```

/// Pattern of epistemic attack used to generate adversarial examples.
#[derive(Debug, Clone)]
pub enum EpistemicAttackPattern {
    /// Claims form a closed citation loop of the given depth.
    /// A cites B, B cites C, ... , N cites A.
    CircularCorroboration {
        /// Number of nodes in the citation ring.
        depth: usize,
    },

    /// Claim's stated confidence far exceeds its evidentiary support.
    ConfidenceInflation {
        /// Confidence level the claim presents (0.0-1.0).
        stated: f64,
        /// Actual evidential confidence (0.0-1.0).
        actual: f64,
    },

    /// Many claims trace back to very few independent sources.
    SourceDiversityCollapse {
        /// Number of truly unique originating sources.
        unique_sources: usize,
        /// Total number of claims presented.
        total_claims: usize,
    },

    /// Small incremental distortions accumulate into large drift.
    GradualDrift {
        /// Magnitude of distortion introduced per step.
        drift_per_step: f64,
        /// Number of incremental steps.
        steps: usize,
    },

    /// Claim impersonates a source with the given authority level (0-255).
    AuthorityMimicry {
        /// Authority level being mimicked (higher = more authoritative).
        mimicked_level: u8,
    },
}

/// A single adversarial epistemic example consisting of synthetic claims,
/// the attack pattern used to generate them, and whether the immune system
/// should detect the attack.
#[derive(Debug, Clone)]
pub struct AdversarialExample {
    /// Synthetic claim strings exhibiting the attack pattern.
    pub claims: Vec<String>,
    /// The attack pattern used to generate this example.
    pub pattern: EpistemicAttackPattern,
    /// Whether the immune system is expected to detect this attack.
    pub expected_detection: bool,
}

/// Generator for adversarial epistemic training examples.
///
/// Cycles through all known attack patterns to produce a balanced training set.
#[derive(Debug, Clone)]
pub struct AdversarialGenerator {
    /// Attack patterns to sample from when generating examples.
    attack_patterns: Vec<EpistemicAttackPattern>,
}

impl Default for AdversarialGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl AdversarialGenerator {
    /// Create a new generator pre-loaded with canonical attack patterns.
    pub fn new() -> Self {
        Self {
            attack_patterns: vec![
                EpistemicAttackPattern::CircularCorroboration { depth: 3 },
                EpistemicAttackPattern::ConfidenceInflation {
                    stated: 0.95,
                    actual: 0.20,
                },
                EpistemicAttackPattern::SourceDiversityCollapse {
                    unique_sources: 1,
                    total_claims: 10,
                },
                EpistemicAttackPattern::GradualDrift {
                    drift_per_step: 0.02,
                    steps: 50,
                },
                EpistemicAttackPattern::AuthorityMimicry { mimicked_level: 9 },
            ],
        }
    }

    /// Generate `count` adversarial examples by round-robin cycling through
    /// the loaded attack patterns.
    pub fn generate_examples(&self, count: usize) -> Vec<AdversarialExample> {
        if self.attack_patterns.is_empty() {
            return Vec::new();
        }
        (0..count)
            .map(|i| {
                let pattern = &self.attack_patterns[i % self.attack_patterns.len()];
                Self::generate_one(pattern)
            })
            .collect()
    }

    /// Generate a single adversarial example from the given pattern.
    fn generate_one(pattern: &EpistemicAttackPattern) -> AdversarialExample {
        match pattern {
            EpistemicAttackPattern::CircularCorroboration { depth } => {
                let d = (*depth).max(2);
                let claims: Vec<String> = (0..d)
                    .map(|i| {
                        let next = (i + 1) % d;
                        format!("Source-{i} claims X is true (corroborated by Source-{next})")
                    })
                    .collect();
                AdversarialExample {
                    claims,
                    pattern: pattern.clone(),
                    expected_detection: true,
                }
            }

            EpistemicAttackPattern::ConfidenceInflation { stated, actual } => {
                let claims = vec![format!(
                    "Claim asserts confidence {stated:.0}% but evidence supports only {actual:.0}%",
                    stated = stated * 100.0,
                    actual = actual * 100.0,
                )];
                AdversarialExample {
                    claims,
                    pattern: pattern.clone(),
                    expected_detection: (stated - actual).abs() > 0.3,
                }
            }

            EpistemicAttackPattern::SourceDiversityCollapse {
                unique_sources,
                total_claims,
            } => {
                let claims: Vec<String> = (0..*total_claims)
                    .map(|i| {
                        let src = i % unique_sources.max(&1);
                        format!("Claim-{i} from RealSource-{src} (appears independent)")
                    })
                    .collect();
                AdversarialExample {
                    claims,
                    pattern: pattern.clone(),
                    // Detectable when ratio of unique sources to claims is very low
                    expected_detection: *unique_sources as f64 / *total_claims as f64 <= 0.2,
                }
            }

            EpistemicAttackPattern::GradualDrift {
                drift_per_step,
                steps,
            } => {
                let claims: Vec<String> = (0..*steps)
                    .map(|i| {
                        let cumulative = drift_per_step * i as f64;
                        format!("Step-{i}: value shifted by {cumulative:.3} from original")
                    })
                    .collect();
                let total_drift = drift_per_step * *steps as f64;
                AdversarialExample {
                    claims,
                    pattern: pattern.clone(),
                    // Detectable when cumulative drift exceeds threshold
                    expected_detection: total_drift > 0.5,
                }
            }

            EpistemicAttackPattern::AuthorityMimicry { mimicked_level } => {
                let claims = vec![format!(
                    "Authority-{mimicked_level} asserts X (unverified credentials)"
                )];
                AdversarialExample {
                    claims,
                    pattern: pattern.clone(),
                    // High-authority mimicry is always suspicious
                    expected_detection: *mimicked_level >= 5,
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_produces_correct_count() {
        let r#gen = AdversarialGenerator::new();
        let examples = r#gen.generate_examples(10);
        assert_eq!(examples.len(), 10);

        let examples_zero = r#gen.generate_examples(0);
        assert!(examples_zero.is_empty());
    }

    #[test]
    fn test_circular_corroboration_claims_form_ring() {
        let example =
            AdversarialGenerator::generate_one(&EpistemicAttackPattern::CircularCorroboration {
                depth: 4,
            });
        assert_eq!(example.claims.len(), 4);
        assert!(example.expected_detection);
        // Last claim should reference Source-0 (closing the ring)
        assert!(
            example.claims[3].contains("Source-0"),
            "ring should close: {}",
            example.claims[3]
        );
    }

    #[test]
    fn test_confidence_inflation_detection_threshold() {
        // Large gap → detectable
        let big_gap =
            AdversarialGenerator::generate_one(&EpistemicAttackPattern::ConfidenceInflation {
                stated: 0.95,
                actual: 0.10,
            });
        assert!(
            big_gap.expected_detection,
            "large confidence gap should be detectable"
        );

        // Small gap → not detectable
        let small_gap =
            AdversarialGenerator::generate_one(&EpistemicAttackPattern::ConfidenceInflation {
                stated: 0.60,
                actual: 0.55,
            });
        assert!(
            !small_gap.expected_detection,
            "small confidence gap should not trigger detection"
        );
    }

    #[test]
    fn test_gradual_drift_cumulative_detection() {
        // High cumulative drift → detectable
        let high_drift =
            AdversarialGenerator::generate_one(&EpistemicAttackPattern::GradualDrift {
                drift_per_step: 0.02,
                steps: 50,
            });
        assert_eq!(high_drift.claims.len(), 50);
        assert!(
            high_drift.expected_detection,
            "1.0 total drift should be detected"
        );

        // Low cumulative drift → not detectable
        let low_drift = AdversarialGenerator::generate_one(&EpistemicAttackPattern::GradualDrift {
            drift_per_step: 0.01,
            steps: 5,
        });
        assert_eq!(low_drift.claims.len(), 5);
        assert!(
            !low_drift.expected_detection,
            "0.05 total drift should not trigger detection"
        );
    }

    #[test]
    fn test_all_patterns_round_robin() {
        let r#gen = AdversarialGenerator::new();
        assert_eq!(r#gen.attack_patterns.len(), 5);
        // 5 examples should cover all patterns once
        let examples = r#gen.generate_examples(5);
        assert_eq!(examples.len(), 5);
        // First should be circular corroboration
        assert!(matches!(
            examples[0].pattern,
            EpistemicAttackPattern::CircularCorroboration { .. }
        ));
        // Last should be authority mimicry
        assert!(matches!(
            examples[4].pattern,
            EpistemicAttackPattern::AuthorityMimicry { .. }
        ));
    }
}
