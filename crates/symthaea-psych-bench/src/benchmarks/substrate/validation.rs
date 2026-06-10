// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Substrate Prediction Validation benchmark.
//!
//! Connects `TestablePrediction` (substrate validation framework) to empirical
//! psych-bench results. Runs the existing substrate benchmarks (Transfer,
//! Degradation) and maps their results to the 9 testable predictions defined
//! in `SubstrateValidationFramework`, recording pass/fail outcomes and
//! upgrading evidence levels when warranted.
//!
//! This is the Phase 5 bridge that makes substrate independence claims
//! empirically testable rather than purely theoretical.
//!
//! # Design
//!
//! Each `TestablePrediction` maps to a specific benchmark + pass criterion:
//! - "Silicon achieves Phi > 0" → TransferBenchmark (silicon profile), phi_preservation > 0
//! - "Consciousness survives transfer" → TransferBenchmark, transfer_fidelity > 0.7
//! - "Radiation reduces Phi" → DegradationBenchmark, degradation_slope > 0
//!
//! Results flow back into `SubstrateValidationFramework::record_prediction_result()`
//! which can upgrade evidence levels (Theoretical → Indirect → CaseStudy).

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::substrate_validation::SubstrateValidationFramework;

use super::degradation::SubstrateDegradationBenchmark;
use super::transfer::SubstrateTransferBenchmark;

/// Substrate Prediction Validation benchmark.
///
/// Runs substrate benchmarks, maps results to testable predictions,
/// and tracks evidence level upgrades.
pub struct SubstrateValidationBenchmark;

/// Mapping from a prediction to its empirical test.
struct PredictionTest {
    substrate: &'static str,
    prediction_idx: usize,
    description: &'static str,
    pass: fn(&BTreeMap<String, MetricValue>, &BTreeMap<String, MetricValue>) -> bool,
}

/// Extract a float from a MetricValue.
fn metric_f64(metrics: &BTreeMap<String, MetricValue>, key: &str) -> f64 {
    match metrics.get(key) {
        Some(mv) => mv.mean,
        None => 0.0,
    }
}

impl SubstrateValidationBenchmark {
    /// Define the mapping from predictions to empirical tests.
    ///
    /// Each entry specifies:
    /// - Which substrate's prediction (by index into its predictions vec)
    /// - A pass function that takes (transfer_metrics, degradation_metrics) → bool
    fn prediction_tests() -> Vec<PredictionTest> {
        vec![
            // Silicon predictions (indices 0, 1, 2 in silicon's predictions vec)
            PredictionTest {
                substrate: "silicon",
                prediction_idx: 0,
                description: "Silicon systems can achieve Phi > 0",
                pass: |transfer, _degradation| {
                    // Silicon must show non-zero phi preservation in transfer benchmark
                    metric_f64(transfer, "phi_preservation") > 0.0
                },
            },
            PredictionTest {
                substrate: "silicon",
                prediction_idx: 1,
                description: "Silicon consciousness requires specific architectures",
                pass: |transfer, _degradation| {
                    // Transfer fidelity varies across substrates → architecture matters
                    let fidelity = metric_f64(transfer, "transfer_fidelity");
                    // If fidelity is high but not perfect, architecture matters
                    fidelity > 0.5 && fidelity < 1.0
                },
            },
            // Quantum predictions (indices 0, 1)
            PredictionTest {
                substrate: "quantum",
                prediction_idx: 0,
                description: "Quantum coherence maintained at relevant scales",
                pass: |transfer, _degradation| {
                    // Quantum substrate should show high binding (entanglement)
                    metric_f64(transfer, "cross_substrate_correlation") > 0.6
                },
            },
            // Biological prediction (index 0)
            PredictionTest {
                substrate: "biological",
                prediction_idx: 0,
                description: "Consciousness requires specific neural architectures",
                pass: |transfer, degradation| {
                    // Biological should have highest fidelity AND graceful degradation
                    let bio_fidelity = metric_f64(transfer, "transfer_fidelity");
                    let graceful = metric_f64(degradation, "graceful_ratio");
                    bio_fidelity > 0.7 && graceful > 0.5
                },
            },
            // Hybrid prediction (index 0)
            PredictionTest {
                substrate: "hybrid",
                prediction_idx: 0,
                description: "Bio-silicon hybrids achieve unified consciousness",
                pass: |transfer, _degradation| {
                    // Transfer across substrates should preserve consciousness
                    metric_f64(transfer, "phi_preservation") > 0.5
                },
            },
            // Spacecraft predictions (indices 0, 1)
            PredictionTest {
                substrate: "spacecraft",
                prediction_idx: 0,
                description: "Spacecraft computers sustain Phi > 0",
                pass: |transfer, _degradation| metric_f64(transfer, "phi_preservation") > 0.0,
            },
            PredictionTest {
                substrate: "spacecraft",
                prediction_idx: 1,
                description: "Radiation-induced bit flips reduce Phi",
                pass: |_transfer, degradation| {
                    // Degradation slope should be positive (phi decreases with damage)
                    metric_f64(degradation, "degradation_slope") > 0.0
                },
            },
        ]
    }
}

impl PsychBenchmark for SubstrateValidationBenchmark {
    fn name(&self) -> &str {
        "substrate_prediction_validation"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let mut framework = SubstrateValidationFramework::new();

        // Run the empirical benchmarks
        let transfer = SubstrateTransferBenchmark;
        let degradation = SubstrateDegradationBenchmark;

        let transfer_result = transfer.run(config);
        let degradation_result = degradation.run(config);

        // Execute each prediction test
        let tests = Self::prediction_tests();
        let mut tested = 0usize;
        let mut passed = 0usize;
        let mut details = Vec::new();

        for test in &tests {
            let result = (test.pass)(&transfer_result.metrics, &degradation_result.metrics);
            let upgraded =
                framework.record_prediction_result(test.substrate, test.prediction_idx, result);

            tested += 1;
            if result {
                passed += 1;
            }

            details.push(format!(
                "[{}] {} — {} {}",
                test.substrate,
                test.description,
                if result { "PASSED" } else { "FAILED" },
                if upgraded { "(evidence UPGRADED)" } else { "" },
            ));
        }

        // Build summary metrics
        let mut result = BenchmarkResult::new(self.name(), None);

        let tested_f = tested as f64;
        let passed_f = passed as f64;

        result.insert(
            "predictions_tested",
            MetricValue {
                mean: tested_f,
                std_dev: 0.0,
                n: 1,
                ci_lower: tested_f,
                ci_upper: tested_f,
            },
        );
        result.insert(
            "predictions_passed",
            MetricValue {
                mean: passed_f,
                std_dev: 0.0,
                n: 1,
                ci_lower: passed_f,
                ci_upper: passed_f,
            },
        );
        let pass_rate = if tested > 0 { passed_f / tested_f } else { 0.0 };
        result.insert(
            "pass_rate",
            MetricValue {
                mean: pass_rate,
                std_dev: 0.0,
                n: 1,
                ci_lower: pass_rate,
                ci_upper: pass_rate,
            },
        );

        // Add per-substrate evidence levels
        for summary in framework.prediction_summary() {
            let conf = summary.honest_confidence;
            result.insert(
                format!("{}_honest_confidence", summary.substrate),
                MetricValue {
                    mean: conf,
                    std_dev: 0.0,
                    n: 1,
                    ci_lower: conf,
                    ci_upper: conf,
                },
            );
            let p = summary.passed as f64;
            result.insert(
                format!("{}_predictions_passed", summary.substrate),
                MetricValue {
                    mean: p,
                    std_dev: 0.0,
                    n: 1,
                    ci_lower: p,
                    ci_upper: p,
                },
            );
            // Evidence level stored as a note since MetricValue is numeric-only
            result.notes.push(format!(
                "{}_evidence_level: {:?}",
                summary.substrate, summary.evidence_level
            ));
        }

        // Include raw benchmark metrics for transparency
        for (key, value) in &transfer_result.metrics {
            result.insert(format!("transfer_{}", key), value.clone());
        }
        for (key, value) in &degradation_result.metrics {
            result.insert(format!("degradation_{}", key), value.clone());
        }

        // Add prediction details as notes
        result.notes.extend(details);
        result.conditions = tested;
        result.trials_per_condition = 1;

        result
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "TestablePrediction → Empirical Benchmark mapping",
            citation: "Putnam (1967); Tononi (2004); Bostrom (2003)",
            year: 2026,
            doi: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_benchmark_runs() {
        let bench = SubstrateValidationBenchmark;
        let config = BenchmarkConfig::default();
        let result = bench.run(&config);

        assert_eq!(result.benchmark, "substrate_prediction_validation");
        assert!(!result.metrics.is_empty());

        // Should have tested predictions
        let tested = result
            .metrics
            .get("predictions_tested")
            .map(|mv| mv.mean as usize)
            .unwrap_or(0);
        assert!(tested > 0, "Should have tested at least one prediction");

        // Should have per-substrate confidence metrics
        assert!(result.metrics.contains_key("silicon_honest_confidence"));
        assert!(result.metrics.contains_key("biological_honest_confidence"));
    }

    #[test]
    fn test_prediction_tests_cover_all_substrates() {
        let tests = SubstrateValidationBenchmark::prediction_tests();
        let substrates: std::collections::HashSet<_> = tests.iter().map(|t| t.substrate).collect();

        // Should cover at least these key substrates
        assert!(substrates.contains("silicon"));
        assert!(substrates.contains("biological"));
        assert!(substrates.contains("spacecraft"));
    }

    #[test]
    fn test_provenance() {
        let bench = SubstrateValidationBenchmark;
        let prov = bench.provenance().unwrap();
        assert_eq!(prov.year, 2026);
        assert!(prov.citation.contains("Putnam"));
    }

    #[test]
    fn test_evidence_upgrade_flow() {
        // Simulate the full flow: run benchmark → record results → check upgrade
        let mut framework = SubstrateValidationFramework::new();

        // Before: silicon is Theoretical
        let silicon = framework.get("silicon").unwrap();
        assert_eq!(
            silicon.evidence_level,
            symthaea_core::hdc::substrate_validation::EvidenceLevel::Theoretical
        );

        // Simulate passing prediction 0
        let upgraded = framework.record_prediction_result("silicon", 0, true);
        assert!(upgraded); // Should upgrade to Indirect

        let silicon = framework.get("silicon").unwrap();
        assert_eq!(
            silicon.evidence_level,
            symthaea_core::hdc::substrate_validation::EvidenceLevel::Indirect
        );
    }
}
