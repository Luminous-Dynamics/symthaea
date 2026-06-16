// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Institutional Reasoning: Weighted Decomposition Benchmark
//!
//! Compares decomposition accuracy between weighted and unweighted bundling.
//! Tests whether causal salience weights (e.g., AUTHORITY:3 vs PROHIBITION:1)
//! improve the system's ability to decompose institutional composites.
//!
//! ## Key Claims Tested
//!
//! 1. **Weighted Decomposition Accuracy**: Does weighted bundling produce
//!    better decomposition results than uniform bundling?
//! 2. **Weight Sensitivity**: Does removing a high-weight component cause
//!    a larger similarity shift than removing a low-weight one?
//! 3. **Weighted vs Unweighted Delta**: The accuracy difference between
//!    weighted and unweighted approaches.
//!
//! ## References
//!
//! - Kanerva, P. (2009). Hyperdimensional computing. *Cognitive Computation*.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::primitive_system::{CompositionAlgebra, PrimitiveSystem};

/// Institutional Reasoning: Weighted Decomposition benchmark.
pub struct WeightedDecompositionBenchmark;

struct TrialResult {
    weighted_accuracy: f64,
    weight_sensitivity: f64,
    weighted_vs_unweighted_delta: f64,
}

struct DecompositionCase {
    composite: &'static str,
    removed: &'static str,
    expected_near: &'static str,
    expected_far: &'static str,
}

impl WeightedDecompositionBenchmark {
    fn cases() -> Vec<DecompositionCase> {
        vec![
            DecompositionCase {
                composite: "LEGITIMATE_GOVERNANCE",
                removed: "TRUST",
                expected_near: "REVOLUTION",
                expected_far: "TRADE_AGREEMENT",
            },
            DecompositionCase {
                composite: "LEGITIMATE_GOVERNANCE",
                removed: "LEGITIMACY",
                expected_near: "REVOLUTION",
                expected_far: "TRADE_AGREEMENT",
            },
            DecompositionCase {
                // ARMS_EMBARGO(SANCTION:5,ENFORCEMENT:2,PROHIBITION:1) -PROHIBITION
                // Residual dominated by SANCTION+ENFORCEMENT → nearest ECONOMIC_SANCTION
                composite: "ARMS_EMBARGO",
                removed: "PROHIBITION",
                expected_near: "ECONOMIC_SANCTION",
                expected_far: "CIVIL_DISOBEDIENCE",
            },
            DecompositionCase {
                composite: "ECONOMIC_SANCTION",
                removed: "PROHIBITION",
                expected_near: "TRADE_AGREEMENT",
                expected_far: "FAILED_STATE",
            },
            DecompositionCase {
                composite: "DEMOCRATIC_ELECTION",
                removed: "COOPERATE",
                expected_near: "LEGITIMATE_GOVERNANCE",
                expected_far: "FAILED_STATE",
            },
            DecompositionCase {
                composite: "CORRUPTION",
                removed: "DEFECT",
                expected_near: "TRADE_AGREEMENT",
                expected_far: "FAILED_STATE",
            },
        ]
    }

    fn eval_accuracy(
        algebra: &CompositionAlgebra,
        system: &PrimitiveSystem,
        cases: &[DecompositionCase],
    ) -> f64 {
        let mut correct = 0usize;
        let mut total = 0usize;

        for case in cases {
            let near_hv = match algebra.get_encoding(case.expected_near, system) {
                Some(hv) => hv,
                None => continue,
            };
            let far_hv = match algebra.get_encoding(case.expected_far, system) {
                Some(hv) => hv,
                None => continue,
            };

            let residual = match algebra.query_decomposition(case.composite, case.removed, system) {
                Ok((_name, _sim, hv)) => hv,
                Err(_) => continue,
            };

            total += 1;
            let sim_near = residual.similarity(&near_hv);
            let sim_far = residual.similarity(&far_hv);
            if sim_near > sim_far {
                correct += 1;
            }
        }

        if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        }
    }

    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let _seed = config.trial_seed("institutional", "weighted_decomposition", trial_idx);
        let system = PrimitiveSystem::new();
        let cases = Self::cases();

        // ── Weighted accuracy ──
        let mut weighted_algebra = CompositionAlgebra::new();
        weighted_algebra.load_institutional_axioms_weighted(&system);
        let weighted_accuracy = Self::eval_accuracy(&weighted_algebra, &system, &cases);

        // ── Unweighted accuracy ──
        let mut unweighted_algebra = CompositionAlgebra::new();
        unweighted_algebra.load_institutional_axioms(&system);
        let unweighted_accuracy = Self::eval_accuracy(&unweighted_algebra, &system, &cases);

        // ── Weight sensitivity ──
        // For each axiom, compare removing its highest-weight component vs
        // its lowest-weight component. The delta should be larger for high-weight.
        // Uses multiple axioms for a robust estimate.
        let sensitivity_cases: &[(&str, &str, &str)] = &[
            // (axiom, high-weight component, low-weight component)
            ("LEGITIMATE_GOVERNANCE", "LEGITIMACY", "TRUST"), // w5 vs w1
            ("REVOLUTION", "AUTHORITY", "PROHIBITION"),       // w5 vs w1
            ("TRADE_AGREEMENT", "EXCHANGE", "TREATY"),        // w5 vs w1
            ("ECONOMIC_SANCTION", "SANCTION", "PROHIBITION"), // w5 vs w2
            ("DEMOCRATIC_ELECTION", "LEGITIMACY", "COOPERATE"), // w5 vs w1
            ("CORRUPTION", "DEFECT", "AUTHORITY"),            // w5 vs w1
        ];

        let mut sensitivity_scores = Vec::new();
        for &(axiom, high_comp, low_comp) in sensitivity_cases {
            let comp_hv = match weighted_algebra.get_encoding(axiom, &system) {
                Some(hv) => hv,
                None => continue,
            };
            let high_res = weighted_algebra.query_decomposition(axiom, high_comp, &system);
            let low_res = weighted_algebra.query_decomposition(axiom, low_comp, &system);
            if let (Ok((_n1, _s1, high_hv)), Ok((_n2, _s2, low_hv))) = (high_res, low_res) {
                let delta_high = 1.0 - high_hv.similarity(&comp_hv) as f64;
                let delta_low = 1.0 - low_hv.similarity(&comp_hv) as f64;
                sensitivity_scores.push((delta_high - delta_low).max(0.0));
            }
        }

        let weight_sensitivity = if sensitivity_scores.is_empty() {
            0.0
        } else {
            sensitivity_scores.iter().sum::<f64>() / sensitivity_scores.len() as f64
        };

        let weighted_vs_unweighted_delta = weighted_accuracy - unweighted_accuracy;

        TrialResult {
            weighted_accuracy,
            weight_sensitivity,
            weighted_vs_unweighted_delta,
        }
    }
}

impl PsychBenchmark for WeightedDecompositionBenchmark {
    fn name(&self) -> &str {
        "InstitutionalReasoning::WeightedDecomposition"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Weighted HDC Decomposition Analysis",
            citation: "Kanerva, P. (2009). Hyperdimensional computing.",
            year: 2009,
            doi: None,
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut accuracies = Vec::new();
        let mut sensitivities = Vec::new();
        let mut deltas = Vec::new();
        let mut trace = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            accuracies.push(r.weighted_accuracy);
            sensitivities.push(r.weight_sensitivity);
            deltas.push(r.weighted_vs_unweighted_delta);

            if config.trial_trace {
                let mut extra = BTreeMap::new();
                extra.insert("weight_sensitivity".to_string(), r.weight_sensitivity);
                extra.insert(
                    "weighted_vs_unweighted_delta".to_string(),
                    r.weighted_vs_unweighted_delta,
                );
                trace.push(TrialOutcome {
                    trial_idx: trial,
                    condition: "weighted_decomposition".to_string(),
                    correct: r.weighted_accuracy > 0.5,
                    rt_ticks: 0.0,
                    similarity: r.weighted_accuracy,
                    confidence: r.weight_sensitivity,
                    response_idx: 0,
                    extra,
                });
            }
        }

        result.insert(
            "weighted_decomposition_accuracy",
            MetricValue::from_samples(&accuracies),
        );
        result.insert(
            "weight_sensitivity",
            MetricValue::from_samples(&sensitivities),
        );
        result.insert(
            "weighted_vs_unweighted_delta",
            MetricValue::from_samples(&deltas),
        );

        result.conditions = 3;
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
    fn test_weighted_decomposition_runs() {
        let config = BenchmarkConfig::default();
        let result = WeightedDecompositionBenchmark.run(&config);
        assert!(
            result
                .metrics
                .contains_key("weighted_decomposition_accuracy")
        );
        assert!(result.metrics.contains_key("weight_sensitivity"));
        assert!(result.metrics.contains_key("weighted_vs_unweighted_delta"));
    }

    #[test]
    fn test_weighted_decomposition_finite_metrics() {
        let config = BenchmarkConfig::default();
        let result = WeightedDecompositionBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {key} is not finite");
        }
    }

    #[test]
    fn test_weighted_per_case_diagnostic() {
        let system = PrimitiveSystem::new();
        let mut algebra = CompositionAlgebra::new();
        algebra.load_institutional_axioms_weighted(&system);

        let cases = WeightedDecompositionBenchmark::cases();
        for case in &cases {
            let near_hv = match algebra.get_encoding(case.expected_near, &system) {
                Some(hv) => hv,
                None => {
                    eprintln!("[ERR] missing near encoding: {}", case.expected_near);
                    continue;
                }
            };
            let far_hv = match algebra.get_encoding(case.expected_far, &system) {
                Some(hv) => hv,
                None => {
                    eprintln!("[ERR] missing far encoding: {}", case.expected_far);
                    continue;
                }
            };
            let residual = match algebra.query_decomposition(case.composite, case.removed, &system)
            {
                Ok((_name, _sim, hv)) => hv,
                Err(e) => {
                    eprintln!("[ERR] {}-{}: {e}", case.composite, case.removed);
                    continue;
                }
            };
            let sim_near = residual.similarity(&near_hv);
            let sim_far = residual.similarity(&far_hv);
            let pass = if sim_near > sim_far { "PASS" } else { "FAIL" };
            // Also show query_decomposition nearest for failing cases
            let (nearest, nearest_sim, _) = algebra
                .query_decomposition(case.composite, case.removed, &system)
                .unwrap();
            eprintln!(
                "[{pass}] {}-{} => near({})={:.4}, far({})={:.4}, actual_nearest={} sim={:.4}",
                case.composite,
                case.removed,
                case.expected_near,
                sim_near,
                case.expected_far,
                sim_far,
                nearest,
                nearest_sim,
            );
        }
    }

    #[test]
    fn test_print_weighted_decomposition_scores() {
        let config = BenchmarkConfig::default();
        let result = WeightedDecompositionBenchmark.run(&config);
        eprintln!("\n═══ Weighted Decomposition Benchmark Scores ═══");
        for (key, val) in &result.metrics {
            eprintln!("  {key:<40} mean={:.4}  sd={:.4}", val.mean, val.std_dev);
        }
        eprintln!("═════════════════════════════════════════════════\n");
    }
}
