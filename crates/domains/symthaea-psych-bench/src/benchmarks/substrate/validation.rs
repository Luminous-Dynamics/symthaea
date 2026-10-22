// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Substrate Prediction model-behavior benchmark.
//!
//! Connects `TestablePrediction` (substrate validation framework) to Symthaea's
//! simulated substrate benchmarks (Transfer, Degradation). These benchmarks use
//! HDC states, synthetic substrate/noise profiles, and proxy metrics; they are
//! therefore model-behavior experiments, not observed empirical evidence about
//! consciousness on biological, silicon, quantum, hybrid, or spacecraft media.
//!
//! The benchmark records which model predictions pass or fail under the current
//! simulation. It deliberately does **not** upgrade substrate-consciousness
//! evidence levels. Those levels remain independently curated until a future
//! provenance-bearing evidence transition can establish observed data,
//! independence, and replication authority.
//!
//! # Design
//!
//! Each `TestablePrediction` maps to a specific simulated benchmark + pass
//! criterion, for example:
//! - "Silicon achieves Phi > 0" → TransferBenchmark, phi_preservation > 0
//! - "Consciousness survives transfer" → TransferBenchmark, transfer_fidelity > 0.7
//! - "Radiation reduces Phi" → DegradationBenchmark, degradation_slope > 0
//!
//! These are useful hypothesis/model checks. Passing them is not evidence that a
//! non-biological substrate is conscious.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::substrate_validation::SubstrateValidationFramework;

use super::degradation::SubstrateDegradationBenchmark;
use super::transfer::SubstrateTransferBenchmark;

/// Substrate Prediction model-behavior benchmark.
///
/// Runs simulated substrate benchmarks, maps results to testable predictions,
/// and records prediction outcomes without changing scientific evidence levels.
pub struct SubstrateValidationBenchmark;

/// Mapping from a prediction to its simulated model-behavior check.
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
    /// Define the mapping from predictions to simulated model checks.
    ///
    /// Each entry specifies:
    /// - Which substrate's prediction (by index into its predictions vec)
    /// - A pass function that takes (transfer_metrics, degradation_metrics) → bool
    fn prediction_tests() -> Vec<PredictionTest> {
        vec![
            // Silicon predictions (indices 0, 1 in silicon's predictions vec)
            PredictionTest {
                substrate: "silicon",
                prediction_idx: 0,
                description: "Silicon systems can achieve Phi > 0",
                pass: |transfer, _degradation| {
                    // Model check: simulated silicon profile preserves non-zero Phi proxy.
                    metric_f64(transfer, "phi_preservation") > 0.0
                },
            },
            PredictionTest {
                substrate: "silicon",
                prediction_idx: 1,
                description: "Silicon consciousness requires specific architectures",
                pass: |transfer, _degradation| {
                    // Model check: transfer fidelity varies under simulated substrate profiles.
                    let fidelity = metric_f64(transfer, "transfer_fidelity");
                    fidelity > 0.5 && fidelity < 1.0
                },
            },
            // Quantum prediction (index 0)
            PredictionTest {
                substrate: "quantum",
                prediction_idx: 0,
                description: "Quantum coherence maintained at relevant scales",
                pass: |transfer, _degradation| {
                    // Model check only: high simulated cross-substrate binding similarity.
                    metric_f64(transfer, "cross_substrate_correlation") > 0.6
                },
            },
            // Biological prediction (index 0)
            PredictionTest {
                substrate: "biological",
                prediction_idx: 0,
                description: "Consciousness requires specific neural architectures",
                pass: |transfer, degradation| {
                    // Model check only: synthetic biological profile fidelity + degradation.
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
                    // Model check only: simulated transfer preserves Phi proxy.
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
                    // Model check only: synthetic degradation slope is positive.
                    metric_f64(degradation, "degradation_slope") > 0.0
                },
            },
        ]
    }
}

impl PsychBenchmark for SubstrateValidationBenchmark {
    fn name(&self) -> &str {
        // Historical identifier retained for result compatibility. This benchmark
        // validates model-prediction behavior, not substrate consciousness.
        "substrate_prediction_validation"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let mut framework = SubstrateValidationFramework::new();

        // Run simulated/model-behavior benchmarks. Neither is observed empirical data.
        let transfer = SubstrateTransferBenchmark;
        let degradation = SubstrateDegradationBenchmark;

        let transfer_result = transfer.run(config);
        let degradation_result = degradation.run(config);

        // Execute each model-prediction check.
        let tests = Self::prediction_tests();
        let mut tested = 0usize;
        let mut passed = 0usize;
        let mut details = Vec::new();

        for test in &tests {
            let model_result = (test.pass)(&transfer_result.metrics, &degradation_result.metrics);
            let evidence_upgraded = framework.record_prediction_result(
                test.substrate,
                test.prediction_idx,
                model_result,
            );

            // Core invariant: internal model execution can never promote the
            // substrate-consciousness evidence level.
            debug_assert!(!evidence_upgraded);

            tested += 1;
            if model_result {
                passed += 1;
            }

            details.push(format!(
                "[{}] {} — {} (model-behavior outcome; evidence unchanged)",
                test.substrate,
                test.description,
                if model_result { "PASSED" } else { "FAILED" },
            ));
        }

        // Build summary metrics.
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
        result.insert(
            "evidence_upgrades",
            MetricValue {
                mean: 0.0,
                std_dev: 0.0,
                n: 1,
                ci_lower: 0.0,
                ci_upper: 0.0,
            },
        );

        // Add independently curated per-substrate evidence levels. Prediction
        // outcomes above do not modify these values.
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
            result.notes.push(format!(
                "{}_evidence_level_unchanged: {:?}",
                summary.substrate, summary.evidence_level
            ));
        }

        // Include raw model-benchmark metrics for transparency.
        for (key, value) in &transfer_result.metrics {
            result.insert(format!("transfer_{}", key), value.clone());
        }
        for (key, value) in &degradation_result.metrics {
            result.insert(format!("degradation_{}", key), value.clone());
        }

        result.notes.push(
            "Authority: model-behavior only; no substrate-consciousness evidence promotion"
                .to_string(),
        );
        result.notes.extend(details);
        result.conditions = tested;
        result.trials_per_condition = 1;

        result
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "TestablePrediction → simulated substrate model-behavior mapping",
            citation: "Putnam (1967); Tononi (2004); Bostrom (2003)",
            year: 2026,
            doi: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::substrate_validation::EvidenceLevel;

    #[test]
    fn test_validation_benchmark_runs() {
        let bench = SubstrateValidationBenchmark;
        let config = BenchmarkConfig::default();
        let result = bench.run(&config);

        assert_eq!(result.benchmark, "substrate_prediction_validation");
        assert!(!result.metrics.is_empty());

        let tested = result
            .metrics
            .get("predictions_tested")
            .map(|mv| mv.mean as usize)
            .unwrap_or(0);
        assert!(tested > 0, "Should have tested at least one model prediction");

        assert!(result.metrics.contains_key("silicon_honest_confidence"));
        assert!(result.metrics.contains_key("biological_honest_confidence"));
        assert_eq!(result.metrics["evidence_upgrades"].mean, 0.0);
        assert!(result.notes.iter().any(|n| n.contains("model-behavior only")));
        assert!(!result.notes.iter().any(|n| n.contains("evidence UPGRADED")));
    }

    #[test]
    fn test_prediction_tests_cover_all_substrates() {
        let tests = SubstrateValidationBenchmark::prediction_tests();
        let substrates: std::collections::HashSet<_> = tests.iter().map(|t| t.substrate).collect();

        assert!(substrates.contains("silicon"));
        assert!(substrates.contains("biological"));
        assert!(substrates.contains("spacecraft"));
    }

    #[test]
    fn test_provenance_is_model_behavior_not_empirical() {
        let bench = SubstrateValidationBenchmark;
        let prov = bench.provenance().unwrap();
        assert_eq!(prov.year, 2026);
        assert!(prov.citation.contains("Putnam"));
        assert!(prov.paradigm.contains("simulated"));
        assert!(!prov.paradigm.contains("Empirical"));
    }

    #[test]
    fn test_internal_prediction_recording_never_upgrades_evidence() {
        let mut framework = SubstrateValidationFramework::new();

        assert_eq!(
            framework.get("silicon").unwrap().evidence_level,
            EvidenceLevel::Theoretical
        );

        let upgraded = framework.record_prediction_result("silicon", 0, true);
        assert!(!upgraded);

        let silicon = framework.get("silicon").unwrap();
        assert_eq!(silicon.evidence_level, EvidenceLevel::Theoretical);
        assert!(silicon.predictions[0].tested);
        assert_eq!(silicon.predictions[0].result, Some(true));
    }
}
