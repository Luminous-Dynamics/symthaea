// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Antagonist Profiles Benchmark — Named Receptor Blockade Validation
//!
//! Validates named antagonist convenience methods (D2, GABA-A, 5-HT2A)
//! produce expected behavioral effects and wear off over time.
//!
//! Science: Stahl (2013) — Essential Psychopharmacology; Seeman (2002) — D2 receptor occupancy.

use crate::harness::PsychBenchmark;
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use std::collections::BTreeMap;
use symthaea_neuromodulators::{NeuromodulatorBath, NeuromodulatorInputs};

/// Observation cycles per condition.
const N_CYCLES: usize = 150;
/// Warmup before measurement.
const WARMUP: usize = 30;
/// Injection half-life.
const HALF_LIFE: u32 = 50;

/// Neutral inputs for simulation.
fn neutral_inputs() -> NeuromodulatorInputs {
    NeuromodulatorInputs {
        prediction_error: 0.2,
        surprise: false,
        reward_signal: 0.0,
        coherence: 0.6,
        arousal: 0.4,
        binding_strength: 0.5,
        epistemic_confidence: 0.5,
        flow_active: false,
        consciousness_level: None,
        moral_signal: None,
    }
}

/// Run baseline condition returning (flexibility, ei_ratio, confidence_delta).
fn run_baseline() -> (f64, f64, f64) {
    let mut bath = NeuromodulatorBath::default();
    let inputs = neutral_inputs();

    for _ in 0..WARMUP {
        bath.update(&inputs);
    }

    let mut flexibility_sum = 0.0_f64;
    let mut ei_sum = 0.0_f64;
    let mut confidence_sum = 0.0_f64;

    for _ in 0..N_CYCLES {
        bath.update(&inputs);
        flexibility_sum += bath.behavioral_flexibility() as f64;
        ei_sum += bath.ei_ratio() as f64;
        confidence_sum += bath.confidence_delta() as f64;
    }

    let n = N_CYCLES as f64;
    (flexibility_sum / n, ei_sum / n, confidence_sum / n)
}

/// Antagonist Profiles Benchmark.
///
/// Tests D2, GABA-A, and 5-HT2A antagonist methods for expected
/// behavioral effects, wear-off dynamics, and agonist+antagonist interaction.
pub struct AntagonistProfilesBenchmark;

impl PsychBenchmark for AntagonistProfilesBenchmark {
    fn name(&self) -> &str {
        "AntagonistProfiles"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let inputs = neutral_inputs();
        let mut trace = Vec::new();
        let mut trial_idx = 0usize;

        let (base_flexibility, base_ei, base_confidence) = run_baseline();

        // Condition 1: D2 antagonist → reduced behavioral flexibility
        let d2_flexibility = {
            let mut bath = NeuromodulatorBath::default();
            for _ in 0..WARMUP {
                bath.update(&inputs);
            }
            bath.inject_d2_antagonist(0.4, HALF_LIFE);
            let mut sum = 0.0_f64;
            for _ in 0..N_CYCLES {
                bath.update(&inputs);
                sum += bath.behavioral_flexibility() as f64;
            }
            sum / N_CYCLES as f64
        };

        // Condition 2: GABA-A antagonist → raised E/I ratio
        let gaba_a_ei = {
            let mut bath = NeuromodulatorBath::default();
            for _ in 0..WARMUP {
                bath.update(&inputs);
            }
            bath.inject_gaba_a_antagonist(0.4, HALF_LIFE);
            let mut sum = 0.0_f64;
            for _ in 0..N_CYCLES {
                bath.update(&inputs);
                sum += bath.ei_ratio() as f64;
            }
            sum / N_CYCLES as f64
        };

        // Condition 3: 5-HT2A antagonist → reduced serotonin-dependent confidence
        // consciousness_modulation() depends on ACh/NE, not 5-HT, so we track
        // confidence_delta() which is directly driven by 5-HT effective level.
        let sht2a_confidence = {
            let mut bath = NeuromodulatorBath::default();
            for _ in 0..WARMUP {
                bath.update(&inputs);
            }
            bath.inject_sht2a_antagonist(0.4, HALF_LIFE);
            let mut sum = 0.0_f64;
            for _ in 0..N_CYCLES {
                bath.update(&inputs);
                sum += bath.confidence_delta() as f64;
            }
            sum / N_CYCLES as f64
        };

        // Condition 4: Antagonist wears off — after enough cycles, effect should diminish
        let wearoff = {
            let mut bath = NeuromodulatorBath::default();
            for _ in 0..WARMUP {
                bath.update(&inputs);
            }
            bath.inject_d2_antagonist(0.3, 20); // short half-life
            // Run past expiration
            for _ in 0..200 {
                bath.update(&inputs);
            }
            // Measure post-wearoff flexibility
            let mut sum = 0.0_f64;
            for _ in 0..50 {
                bath.update(&inputs);
                sum += bath.behavioral_flexibility() as f64;
            }
            sum / 50.0
        };

        // Condition 5: Concurrent agonist + antagonist
        let concurrent = {
            let mut bath = NeuromodulatorBath::default();
            for _ in 0..WARMUP {
                bath.update(&inputs);
            }
            bath.inject("da", 0.5, 100); // DA agonist
            bath.inject_d2_antagonist(0.3, 100); // D2 antagonist
            let mut sum = 0.0_f64;
            for _ in 0..N_CYCLES {
                bath.update(&inputs);
                sum += bath.behavioral_flexibility() as f64;
            }
            sum / N_CYCLES as f64
        };

        let mut result = BenchmarkResult::new(self.name(), None);

        // D2 reduces flexibility
        let d2_effect = base_flexibility - d2_flexibility;
        result.insert(
            "d2_flexibility_reduction",
            MetricValue::from_samples(&[d2_effect]),
        );
        result.insert(
            "d2_flexibility_mean",
            MetricValue::from_samples(&[d2_flexibility]),
        );
        if config.trial_trace {
            trace.push(TrialOutcome {
                trial_idx,
                condition: "d2_antagonist".to_string(),
                correct: d2_effect.abs() > 0.0,
                rt_ticks: 0.0,
                similarity: d2_effect.abs(),
                confidence: 0.0,
                response_idx: 0,
                extra: BTreeMap::new(),
            });
            trial_idx += 1;
        }

        // GABA-A raises E/I
        let gaba_effect = gaba_a_ei - base_ei;
        result.insert(
            "gaba_a_ei_increase",
            MetricValue::from_samples(&[gaba_effect]),
        );
        result.insert("gaba_a_ei_mean", MetricValue::from_samples(&[gaba_a_ei]));
        if config.trial_trace {
            trace.push(TrialOutcome {
                trial_idx,
                condition: "gaba_a_antagonist".to_string(),
                correct: gaba_effect.abs() > 0.0,
                rt_ticks: 0.0,
                similarity: gaba_effect.abs(),
                confidence: 0.0,
                response_idx: 0,
                extra: BTreeMap::new(),
            });
            trial_idx += 1;
        }

        // 5-HT2A reduces consciousness
        let sht2a_effect = base_confidence - sht2a_confidence;
        result.insert(
            "sht2a_confidence_reduction",
            MetricValue::from_samples(&[sht2a_effect]),
        );
        result.insert(
            "sht2a_confidence_mean",
            MetricValue::from_samples(&[sht2a_confidence]),
        );
        if config.trial_trace {
            trace.push(TrialOutcome {
                trial_idx,
                condition: "sht2a_antagonist".to_string(),
                correct: sht2a_effect.abs() > 0.0,
                rt_ticks: 0.0,
                similarity: sht2a_effect.abs(),
                confidence: 0.0,
                response_idx: 0,
                extra: BTreeMap::new(),
            });
            trial_idx += 1;
        }

        // Wearoff: flexibility should recover toward baseline
        let wearoff_recovery = wearoff - d2_flexibility; // should be positive
        result.insert(
            "wearoff_recovery",
            MetricValue::from_samples(&[wearoff_recovery]),
        );
        result.insert("wearoff_flexibility", MetricValue::from_samples(&[wearoff]));
        if config.trial_trace {
            trace.push(TrialOutcome {
                trial_idx,
                condition: "wearoff".to_string(),
                correct: wearoff_recovery.abs() > 0.0,
                rt_ticks: 0.0,
                similarity: wearoff_recovery.abs(),
                confidence: 0.0,
                response_idx: 0,
                extra: BTreeMap::new(),
            });
            trial_idx += 1;
        }

        // Concurrent: partial attenuation of agonist by antagonist
        result.insert(
            "concurrent_flexibility",
            MetricValue::from_samples(&[concurrent]),
        );
        if config.trial_trace {
            let concurrent_effect = (concurrent - base_flexibility).abs();
            trace.push(TrialOutcome {
                trial_idx,
                condition: "concurrent".to_string(),
                correct: concurrent_effect > 0.0,
                rt_ticks: 0.0,
                similarity: concurrent_effect,
                confidence: 0.0,
                response_idx: 0,
                extra: BTreeMap::new(),
            });
            trial_idx += 1;
        }

        result.insert(
            "baseline_flexibility",
            MetricValue::from_samples(&[base_flexibility]),
        );
        result.insert("baseline_ei", MetricValue::from_samples(&[base_ei]));
        result.insert(
            "baseline_consciousness",
            MetricValue::from_samples(&[base_confidence]),
        );

        if config.trial_trace {
            result.trial_trace = trace;
        }
        let _ = trial_idx;
        result.conditions = 6;
        result.trials_per_condition = 1;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> BenchmarkConfig {
        BenchmarkConfig {
            seed: 42,
            ..Default::default()
        }
    }

    #[test]
    fn test_antagonist_profiles_runs() {
        let bench = AntagonistProfilesBenchmark;
        let result = bench.run(&config());
        assert_eq!(result.benchmark, "AntagonistProfiles");
        assert!(result.metrics.len() >= 10);
    }

    #[test]
    fn test_d2_reduces_flexibility() {
        let bench = AntagonistProfilesBenchmark;
        let result = bench.run(&config());
        let reduction = result.metrics.get("d2_flexibility_reduction").unwrap().mean;
        assert!(
            reduction > 0.0,
            "D2 antagonist should reduce behavioral flexibility: delta={reduction}"
        );
    }

    #[test]
    fn test_gaba_a_raises_ei() {
        let bench = AntagonistProfilesBenchmark;
        let result = bench.run(&config());
        let increase = result.metrics.get("gaba_a_ei_increase").unwrap().mean;
        assert!(
            increase > 0.0,
            "GABA-A antagonist should raise E/I ratio: delta={increase}"
        );
    }

    #[test]
    fn test_sht2a_reduces_confidence() {
        let bench = AntagonistProfilesBenchmark;
        let result = bench.run(&config());
        let reduction = result
            .metrics
            .get("sht2a_confidence_reduction")
            .unwrap()
            .mean;
        assert!(
            reduction > 0.0,
            "5-HT2A antagonist should reduce confidence delta: delta={reduction}"
        );
    }

    #[test]
    fn test_antagonist_wears_off() {
        let bench = AntagonistProfilesBenchmark;
        let result = bench.run(&config());
        let recovery = result.metrics.get("wearoff_recovery").unwrap().mean;
        assert!(
            recovery > 0.0,
            "D2 antagonist should wear off: recovery delta={recovery}"
        );
    }

    #[test]
    fn test_concurrent_agonist_antagonist() {
        let bench = AntagonistProfilesBenchmark;
        let result = bench.run(&config());
        let concurrent = result.metrics.get("concurrent_flexibility").unwrap().mean;
        assert!(
            concurrent.is_finite(),
            "Concurrent agonist+antagonist should produce finite output: {concurrent}"
        );
    }

    #[test]
    fn test_all_metrics_finite() {
        let bench = AntagonistProfilesBenchmark;
        let result = bench.run(&config());
        for (name, metric) in &result.metrics {
            assert!(
                metric.mean.is_finite(),
                "Metric '{name}' should be finite, got {}",
                metric.mean
            );
        }
    }
}
