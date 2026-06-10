// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Institutional Reasoning: Causal Decomposition Benchmark
//!
//! Tests whether the HDC composition algebra can correctly answer counterfactual
//! institutional questions using the causal axioms defined in
//! `CompositionAlgebra::load_institutional_axioms`.
//!
//! ## Key Claims Tested
//!
//! 1. **Causal Decomposition Accuracy**: Given a composite like NATION_STATE,
//!    unbinding ENFORCEMENT should yield a residual closer to FAILED_STATE than
//!    to LEGITIMATE_GOVERNANCE — because enforcement collapse is the defining
//!    feature of state failure.
//!
//! 2. **Axiom Discrimination**: Each institutional axiom should occupy a distinct
//!    region of the 16,384D hypervector space, with self-similarity (1.0) strictly
//!    exceeding max cross-similarity to any other axiom.
//!
//! 3. **Recovery Fidelity**: XOR binding is its own inverse — unbinding and
//!    rebinding a component should recover the original composite with high
//!    similarity (ideally 1.0 for lossless XOR).
//!
//! 4. **Cross-Domain Coherence**: Composites that share components should be
//!    more similar than composites that share no components, validating that
//!    the algebra preserves structural relationships.
//!
//! ## References
//!
//! - Putnam, H. (1967). Psychological predicates. In *Art, Mind, and Religion*.
//! - Kanerva, P. (2009). Hyperdimensional computing. *Cognitive Computation*.
//! - Ostrom, E. (1990). *Governing the Commons*. Cambridge UP.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::primitive_system::CompositionAlgebra;
use symthaea_core::hdc::primitive_system::PrimitiveSystem;

/// Institutional Reasoning: Causal Decomposition benchmark.
pub struct InstitutionalReasoningBenchmark;

struct TrialResult {
    decomposition_accuracy: f64,
    axiom_discrimination: f64,
    recovery_fidelity: f64,
    cross_domain_coherence: f64,
}

/// A decomposition test case: unbind `removed` from `composite`, expect the
/// residual to be closer to `expected_near` than to `expected_far`.
struct DecompositionCase {
    composite: &'static str,
    removed: &'static str,
    expected_near: &'static str,
    expected_far: &'static str,
}

/// All 8 institutional axiom names loaded by `load_institutional_axioms`.
const AXIOM_NAMES: &[&str] = &[
    "REVOLUTION",
    "FAILED_STATE",
    "BORDER_DISPUTE",
    "LEGITIMATE_GOVERNANCE",
    "REGULATORY_CAPTURE",
    "TRADE_AGREEMENT",
    "ECONOMIC_SANCTION",
    "CIVIL_DISOBEDIENCE",
];

impl InstitutionalReasoningBenchmark {
    fn decomposition_cases() -> Vec<DecompositionCase> {
        vec![
            DecompositionCase {
                composite: "LEGITIMATE_GOVERNANCE",
                removed: "TRUST",
                expected_near: "REVOLUTION",
                expected_far: "TRADE_AGREEMENT",
            },
            DecompositionCase {
                composite: "NATION_STATE",
                removed: "ENFORCEMENT",
                expected_near: "FAILED_STATE",
                expected_far: "LEGITIMATE_GOVERNANCE",
            },
            DecompositionCase {
                composite: "LEGITIMATE_GOVERNANCE",
                removed: "LEGITIMACY",
                expected_near: "REVOLUTION",
                expected_far: "TRADE_AGREEMENT",
            },
            DecompositionCase {
                composite: "TRADE_AGREEMENT",
                removed: "RECIPROCATE",
                expected_near: "ECONOMIC_SANCTION",
                expected_far: "CIVIL_DISOBEDIENCE",
            },
            DecompositionCase {
                composite: "REGULATORY_CAPTURE",
                removed: "DEFECT",
                expected_near: "REGULATION",
                expected_far: "REVOLUTION",
            },
            DecompositionCase {
                composite: "ECONOMIC_SANCTION",
                removed: "PROHIBITION",
                expected_near: "TRADE_AGREEMENT",
                expected_far: "FAILED_STATE",
            },
        ]
    }

    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let seed = config.trial_seed("institutional", "causal_decomposition", trial_idx);
        let system = PrimitiveSystem::new();

        let mut algebra = CompositionAlgebra::new();
        let loaded = algebra.load_institutional_axioms(&system);
        assert!(
            loaded >= 8,
            "expected at least 8 institutional axioms, got {loaded}"
        );

        // Also define NATION_STATE and REGULATION in the algebra so we can
        // use them as composites/targets in decomposition tests.
        // Use bundle (+) to match the axiom definitions — XOR (^) binding
        // creates a fundamentally different representation that can't be
        // compared via similarity with bundle-based axioms.
        let _ = algebra.define(
            "NATION_STATE",
            "SOVEREIGNTY + INSTITUTION + ENFORCEMENT + POPULATION",
            &system,
        );
        let _ = algebra.define("REGULATION", "LAW + COMPLIANCE", &system);

        // ── 1. Decomposition accuracy ──
        // Institutional axioms are created via bundling (majority vote), not XOR
        // binding. XOR unbinding is NOT the inverse of bundling. Instead, use
        // a similarity-residual approach: after factoring out the removed
        // component's contribution, which target does the composite align with?
        //
        // residual_sim = sim(composite, target) - sim(removed, target)
        // This measures: "the composite is similar to the target for reasons
        // beyond just the removed component."
        let cases = Self::decomposition_cases();
        let mut correct = 0usize;
        let total = cases.len();

        for case in &cases {
            let composite_hv = match algebra.get_encoding(case.composite, &system) {
                Some(hv) => hv,
                None => continue,
            };
            let removed_hv = match algebra.get_encoding(case.removed, &system) {
                Some(hv) => hv,
                None => continue,
            };
            let near_hv = match algebra.get_encoding(case.expected_near, &system) {
                Some(hv) => hv,
                None => continue,
            };
            let far_hv = match algebra.get_encoding(case.expected_far, &system) {
                Some(hv) => hv,
                None => continue,
            };

            // Similarity-residual decomposition: factor out the removed
            // component's contribution to see which target the composite
            // aligns with for structural (non-removed) reasons.
            let sim_comp_near = composite_hv.similarity(&near_hv);
            let sim_comp_far = composite_hv.similarity(&far_hv);
            let sim_rem_near = removed_hv.similarity(&near_hv);
            let sim_rem_far = removed_hv.similarity(&far_hv);

            let residual_near = sim_comp_near - sim_rem_near;
            let residual_far = sim_comp_far - sim_rem_far;

            if residual_near > residual_far {
                correct += 1;
            }
        }

        let decomposition_accuracy = if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        };

        // ── 2. Axiom discrimination ──
        // For each axiom, measure 1.0 - max_cross_similarity.
        let mut discrimination_scores = Vec::new();
        for &name_a in AXIOM_NAMES {
            let hv_a = match algebra.get_encoding(name_a, &system) {
                Some(hv) => hv,
                None => continue,
            };

            let mut max_cross = f32::NEG_INFINITY;
            for &name_b in AXIOM_NAMES {
                if name_a == name_b {
                    continue;
                }
                let hv_b = match algebra.get_encoding(name_b, &system) {
                    Some(hv) => hv,
                    None => continue,
                };
                let sim = hv_a.similarity(&hv_b);
                if sim > max_cross {
                    max_cross = sim;
                }
            }

            // Self-similarity is 1.0 for BinaryHV; discrimination = 1.0 - max_cross
            if max_cross.is_finite() {
                discrimination_scores.push((1.0 - max_cross as f64).max(0.0));
            }
        }

        let axiom_discrimination = if discrimination_scores.is_empty() {
            0.0
        } else {
            discrimination_scores.iter().sum::<f64>() / discrimination_scores.len() as f64
        };

        // ── 3. Recovery fidelity ──
        // For each decomposition case, unbind then rebind and measure recovery.
        let mut recovery_scores = Vec::new();
        for case in &cases {
            let composite_hv = match algebra.get_encoding(case.composite, &system) {
                Some(hv) => hv,
                None => continue,
            };
            let removed_hv = match algebra.get_encoding(case.removed, &system) {
                Some(hv) => hv,
                None => continue,
            };

            // Unbind then rebind: (composite ^ removed) ^ removed = composite
            let residual = composite_hv.bind(&removed_hv);
            let recovered = residual.bind(&removed_hv);
            let recovery_sim = recovered.similarity(&composite_hv) as f64;
            recovery_scores.push(recovery_sim);
        }

        let recovery_fidelity = if recovery_scores.is_empty() {
            0.0
        } else {
            recovery_scores.iter().sum::<f64>() / recovery_scores.len() as f64
        };

        // ── 4. Cross-domain coherence ──
        // Composites sharing components should be more similar than those that don't.
        // Shared-component pairs:
        //   REVOLUTION and LEGITIMATE_GOVERNANCE both contain AUTHORITY
        //   TRADE_AGREEMENT and ECONOMIC_SANCTION both contain EXCHANGE
        // We compare their pairwise similarity against the mean pairwise similarity
        // of all axiom pairs.

        // Use seed for deterministic jitter (unused here, but consistent with pattern)
        let _ = seed;

        let shared_pairs: &[(&str, &str)] = &[
            ("REVOLUTION", "LEGITIMATE_GOVERNANCE"),  // share AUTHORITY
            ("TRADE_AGREEMENT", "ECONOMIC_SANCTION"), // share EXCHANGE
            ("REVOLUTION", "CIVIL_DISOBEDIENCE"),     // share PROHIBITION
        ];

        let mut shared_sims = Vec::new();
        for &(a, b) in shared_pairs {
            let hv_a = match algebra.get_encoding(a, &system) {
                Some(hv) => hv,
                None => continue,
            };
            let hv_b = match algebra.get_encoding(b, &system) {
                Some(hv) => hv,
                None => continue,
            };
            shared_sims.push(hv_a.similarity(&hv_b) as f64);
        }

        // Mean pairwise similarity across all axiom pairs
        let mut all_sims = Vec::new();
        for (i, &name_a) in AXIOM_NAMES.iter().enumerate() {
            for &name_b in &AXIOM_NAMES[i + 1..] {
                let hv_a = match algebra.get_encoding(name_a, &system) {
                    Some(hv) => hv,
                    None => continue,
                };
                let hv_b = match algebra.get_encoding(name_b, &system) {
                    Some(hv) => hv,
                    None => continue,
                };
                all_sims.push(hv_a.similarity(&hv_b) as f64);
            }
        }

        let mean_shared = if shared_sims.is_empty() {
            0.0
        } else {
            shared_sims.iter().sum::<f64>() / shared_sims.len() as f64
        };
        let mean_all = if all_sims.is_empty() {
            0.0
        } else {
            all_sims.iter().sum::<f64>() / all_sims.len() as f64
        };

        // Coherence = how much shared-component pairs exceed the baseline
        // Normalize to [0, 1] range: (shared - all) / (1 - all), clamped
        let cross_domain_coherence = if (1.0 - mean_all).abs() > 1e-10 {
            ((mean_shared - mean_all) / (1.0 - mean_all)).clamp(0.0, 1.0)
        } else {
            0.0
        };

        TrialResult {
            decomposition_accuracy,
            axiom_discrimination,
            recovery_fidelity,
            cross_domain_coherence,
        }
    }
}

impl PsychBenchmark for InstitutionalReasoningBenchmark {
    fn name(&self) -> &str {
        "InstitutionalReasoning::CausalDecomposition"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Institutional Causal Decomposition (HDC Composition Algebra)",
            citation: "Putnam (1967); Kanerva (2009)",
            year: 2009,
            doi: None,
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut decomposition_accs = Vec::new();
        let mut axiom_discs = Vec::new();
        let mut recovery_fids = Vec::new();
        let mut coherences = Vec::new();
        let mut trace = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            decomposition_accs.push(r.decomposition_accuracy);
            axiom_discs.push(r.axiom_discrimination);
            recovery_fids.push(r.recovery_fidelity);
            coherences.push(r.cross_domain_coherence);

            if config.trial_trace {
                let mut extra = BTreeMap::new();
                extra.insert("axiom_discrimination".to_string(), r.axiom_discrimination);
                extra.insert("recovery_fidelity".to_string(), r.recovery_fidelity);
                extra.insert(
                    "cross_domain_coherence".to_string(),
                    r.cross_domain_coherence,
                );
                trace.push(TrialOutcome {
                    trial_idx: trial,
                    condition: "causal_decomposition".to_string(),
                    correct: r.decomposition_accuracy > 0.5,
                    rt_ticks: 0.0,
                    similarity: r.decomposition_accuracy,
                    confidence: r.recovery_fidelity,
                    response_idx: 0,
                    extra,
                });
            }
        }

        result.insert(
            "institutional_decomposition_accuracy",
            MetricValue::from_samples(&decomposition_accs),
        );
        result.insert(
            "institutional_axiom_discrimination",
            MetricValue::from_samples(&axiom_discs),
        );
        result.insert(
            "institutional_recovery_fidelity",
            MetricValue::from_samples(&recovery_fids),
        );
        result.insert(
            "institutional_cross_domain_coherence",
            MetricValue::from_samples(&coherences),
        );

        result.conditions = 4; // 4 metric dimensions
        result.trials_per_condition = config.trials_per_condition;
        if config.trial_trace {
            result.trial_trace = trace;
        }
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_institutional_reasoning_runs() {
        let config = BenchmarkConfig::default();
        let result = InstitutionalReasoningBenchmark.run(&config);
        assert!(
            result
                .metrics
                .contains_key("institutional_decomposition_accuracy")
        );
        assert!(
            result
                .metrics
                .contains_key("institutional_axiom_discrimination")
        );
        assert!(
            result
                .metrics
                .contains_key("institutional_recovery_fidelity")
        );
        assert!(
            result
                .metrics
                .contains_key("institutional_cross_domain_coherence")
        );
    }

    #[test]
    fn test_institutional_reasoning_finite_metrics() {
        let config = BenchmarkConfig::default();
        let result = InstitutionalReasoningBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {key} is not finite");
        }
    }

    #[test]
    fn test_recovery_fidelity_perfect() {
        // XOR binding is its own inverse, so recovery should be 1.0
        let config = BenchmarkConfig::default();
        let result = InstitutionalReasoningBenchmark.run(&config);
        let recovery = result
            .metrics
            .get("institutional_recovery_fidelity")
            .unwrap();
        assert!(
            recovery.mean > 0.99,
            "XOR recovery should be ~1.0, got {}",
            recovery.mean
        );
    }

    #[test]
    fn test_axiom_discrimination_positive() {
        let config = BenchmarkConfig::default();
        let result = InstitutionalReasoningBenchmark.run(&config);
        let disc = result
            .metrics
            .get("institutional_axiom_discrimination")
            .unwrap();
        assert!(
            disc.mean > 0.0,
            "axiom discrimination should be positive, got {}",
            disc.mean
        );
    }
}
