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
                composite: "TRADE_AGREEMENT",
                removed: "RECIPROCATE",
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
        // Compare: does removing a high-weight component cause a larger
        // similarity shift than removing a low-weight one?
        // High-weight removals: LEGITIMACY from LEGITIMATE_GOVERNANCE (weight 3)
        // Low-weight removals: COOPERATE from DEMOCRATIC_ELECTION (weight 1)
        let mut sensitivity_pairs = Vec::new();

        // High-weight removal
        if let (Some(comp_hv), Ok((_n, _s, res_hv))) = (
            weighted_algebra.get_encoding("LEGITIMATE_GOVERNANCE", &system),
            weighted_algebra.query_decomposition("LEGITIMATE_GOVERNANCE", "LEGITIMACY", &system),
        ) {
            let delta_high = 1.0 - res_hv.similarity(&comp_hv) as f64;
            sensitivity_pairs.push(("high", delta_high));
        }

        // Low-weight removal
        if let (Some(comp_hv), Ok((_n, _s, res_hv))) = (
            weighted_algebra.get_encoding("DEMOCRATIC_ELECTION", &system),
            weighted_algebra.query_decomposition("DEMOCRATIC_ELECTION", "COOPERATE", &system),
        ) {
            let delta_low = 1.0 - res_hv.similarity(&comp_hv) as f64;
            sensitivity_pairs.push(("low", delta_low));
        }

        let weight_sensitivity = if sensitivity_pairs.len() == 2 {
            let high = sensitivity_pairs
                .iter()
                .find(|(l, _)| *l == "high")
                .map(|(_, v)| *v)
                .unwrap_or(0.0);
            let low = sensitivity_pairs
                .iter()
                .find(|(l, _)| *l == "low")
                .map(|(_, v)| *v)
                .unwrap_or(0.0);
            (high - low).max(0.0)
        } else {
            0.0
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
        assert!(result
            .metrics
            .contains_key("weighted_decomposition_accuracy"));
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
}
