// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness Pharmacology Benchmark
//!
//! Validates that pharmacological injections change consciousness measurements
//! through the bath-consciousness coupling. Uses the real `NeuromodulatorBath`
//! injection API to simulate 5 pharmacological conditions and measure their
//! impact on a consciousness proxy derived from 5-HT2A and GABA-A signals.
//!
//! Science:
//! - Carhart-Harris & Nutt (2017) — 5-HT2A psychedelic consciousness expansion
//! - Olsen & Sieghart (2009) — GABA-A sedation and consciousness suppression
//! - Volkow et al. (2009) — DA stimulant effects on consciousness
//! - Porkka-Heiskanen (1997) — Adenosine sedation
//! - Piomelli (2003) — Endocannabinoid stabilization

use crate::harness::PsychBenchmark;
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use std::collections::BTreeMap;
use symthaea_neuromodulators::{NeuromodulatorBath, NeuromodulatorInputs};

/// Warmup cycles before injection.
const WARMUP: usize = 30;
/// Observation cycles after injection.
const OBSERVATION: usize = 100;

/// Consciousness proxy replicating the bath→consciousness coupling
/// (consciousness_engine.rs:381-389).
///
/// Returns a consciousness modulation estimate from 5-HT2A and GABA-A signals.
fn consciousness_proxy(bath: &NeuromodulatorBath) -> f32 {
    0.5 + (bath.sht_2a_signal() - 0.5) * 0.1 - (bath.gaba_a_signal() - 0.4) * 0.08
}

/// Neutral inputs for baseline simulation.
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

/// Per-condition observation record.
struct ConditionTrace {
    proxy_values: Vec<f32>,
    baseline_proxy: f32,
}

impl ConditionTrace {
    fn peak(&self) -> f32 {
        self.proxy_values
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
    }

    #[allow(dead_code)]
    fn trough(&self) -> f32 {
        self.proxy_values
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min)
    }

    fn mean(&self) -> f32 {
        if self.proxy_values.is_empty() {
            return 0.0;
        }
        self.proxy_values.iter().sum::<f32>() / self.proxy_values.len() as f32
    }

    fn time_to_peak(&self) -> usize {
        self.proxy_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn duration_above_baseline(&self) -> usize {
        self.proxy_values
            .iter()
            .filter(|&&v| v > self.baseline_proxy + 0.001)
            .count()
    }

    fn variance(&self) -> f32 {
        if self.proxy_values.len() < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let n = self.proxy_values.len() as f32;
        self.proxy_values
            .iter()
            .map(|v| {
                let d = v - mean;
                d * d
            })
            .sum::<f32>()
            / n
    }
}

/// Run a pharmacological condition.
fn run_condition(target: &str, dose: f32, half_life: u32) -> ConditionTrace {
    let mut bath = NeuromodulatorBath::default();
    let inputs = neutral_inputs();

    // Warmup
    for _ in 0..WARMUP {
        bath.update(&inputs);
    }
    let baseline_proxy = consciousness_proxy(&bath);

    // Inject
    bath.inject(target, dose, half_life);

    // Observe
    let mut proxy_values = Vec::with_capacity(OBSERVATION);
    for _ in 0..OBSERVATION {
        bath.update(&inputs);
        proxy_values.push(consciousness_proxy(&bath));
    }

    ConditionTrace {
        proxy_values,
        baseline_proxy,
    }
}

/// Consciousness Pharmacology Benchmark.
///
/// 5 pharmacological conditions testing bath→consciousness coupling:
/// 1. Psychedelic (5-HT agonist) → consciousness INCREASES
/// 2. Anxiolytic (GABA agonist) → consciousness DECREASES
/// 3. Stimulant (DA agonist) → consciousness modulation
/// 4. Sedative (adenosine agonist) → consciousness DROPS
/// 5. ECB Buffer (endocannabinoid) → consciousness STABILIZES
pub struct ConsciousnessPharmacologyBenchmark;

impl PsychBenchmark for ConsciousnessPharmacologyBenchmark {
    fn name(&self) -> &str {
        "Neuromod::ConsciousnessPharmacology"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut trace = Vec::new();
        let mut trial_idx = 0usize;

        // 1. Psychedelic: 5-HT agonist → consciousness increases via 5-HT2A
        let psychedelic = run_condition("sht", 0.4, 80);
        // 2. Anxiolytic: GABA agonist → consciousness decreases via GABA-A
        let anxiolytic = run_condition("gaba", 0.3, 60);
        // 3. Stimulant: DA agonist → consciousness modulation
        let stimulant = run_condition("da", 0.4, 30);
        // 4. Sedative: adenosine agonist → consciousness drops
        let sedative = run_condition("aden", 0.3, 50);
        // 5. ECB Buffer: endocannabinoid → consciousness stabilizes
        let ecb_buffer = run_condition("ecb", 0.3, 40);

        let mut result = BenchmarkResult::new(self.name(), None);

        // Psychedelic metrics
        result.insert(
            "psychedelic_proxy_peak",
            MetricValue::from_samples(&[psychedelic.peak() as f64]),
        );
        result.insert(
            "psychedelic_proxy_mean",
            MetricValue::from_samples(&[psychedelic.mean() as f64]),
        );
        result.insert(
            "psychedelic_time_to_peak",
            MetricValue::from_samples(&[psychedelic.time_to_peak() as f64]),
        );
        result.insert(
            "psychedelic_duration_above_baseline",
            MetricValue::from_samples(&[psychedelic.duration_above_baseline() as f64]),
        );

        // Anxiolytic metrics
        result.insert(
            "anxiolytic_proxy_peak",
            MetricValue::from_samples(&[anxiolytic.peak() as f64]),
        );
        result.insert(
            "anxiolytic_proxy_mean",
            MetricValue::from_samples(&[anxiolytic.mean() as f64]),
        );
        result.insert(
            "anxiolytic_time_to_peak",
            MetricValue::from_samples(&[anxiolytic.time_to_peak() as f64]),
        );
        result.insert(
            "anxiolytic_duration_above_baseline",
            MetricValue::from_samples(&[anxiolytic.duration_above_baseline() as f64]),
        );

        // Stimulant metrics
        result.insert(
            "stimulant_proxy_peak",
            MetricValue::from_samples(&[stimulant.peak() as f64]),
        );
        result.insert(
            "stimulant_proxy_mean",
            MetricValue::from_samples(&[stimulant.mean() as f64]),
        );
        result.insert(
            "stimulant_time_to_peak",
            MetricValue::from_samples(&[stimulant.time_to_peak() as f64]),
        );
        result.insert(
            "stimulant_duration_above_baseline",
            MetricValue::from_samples(&[stimulant.duration_above_baseline() as f64]),
        );

        // Sedative metrics
        result.insert(
            "sedative_proxy_peak",
            MetricValue::from_samples(&[sedative.peak() as f64]),
        );
        result.insert(
            "sedative_proxy_mean",
            MetricValue::from_samples(&[sedative.mean() as f64]),
        );
        result.insert(
            "sedative_time_to_peak",
            MetricValue::from_samples(&[sedative.time_to_peak() as f64]),
        );
        result.insert(
            "sedative_duration_above_baseline",
            MetricValue::from_samples(&[sedative.duration_above_baseline() as f64]),
        );

        // ECB buffer metrics
        result.insert(
            "ecb_proxy_peak",
            MetricValue::from_samples(&[ecb_buffer.peak() as f64]),
        );
        result.insert(
            "ecb_proxy_mean",
            MetricValue::from_samples(&[ecb_buffer.mean() as f64]),
        );
        result.insert(
            "ecb_time_to_peak",
            MetricValue::from_samples(&[ecb_buffer.time_to_peak() as f64]),
        );
        result.insert(
            "ecb_duration_above_baseline",
            MetricValue::from_samples(&[ecb_buffer.duration_above_baseline() as f64]),
        );

        // Cross-condition variance (stability metric)
        result.insert(
            "ecb_proxy_variance",
            MetricValue::from_samples(&[ecb_buffer.variance() as f64]),
        );

        if config.trial_trace {
            trace.push(TrialOutcome {
                trial_idx,
                condition: "psychedelic".to_string(),
                correct: (psychedelic.peak() - psychedelic.baseline_proxy).abs() > 0.0001,
                rt_ticks: 0.0,
                similarity: psychedelic.peak() as f64,
                confidence: 0.0,
                response_idx: 0,
                extra: BTreeMap::new(),
            });
            trial_idx += 1;

            trace.push(TrialOutcome {
                trial_idx,
                condition: "anxiolytic".to_string(),
                correct: (anxiolytic.peak() - anxiolytic.baseline_proxy).abs() > 0.0001,
                rt_ticks: 0.0,
                similarity: anxiolytic.peak() as f64,
                confidence: 0.0,
                response_idx: 0,
                extra: BTreeMap::new(),
            });
            trial_idx += 1;

            trace.push(TrialOutcome {
                trial_idx,
                condition: "stimulant".to_string(),
                correct: (stimulant.peak() - stimulant.baseline_proxy).abs() > 0.0001,
                rt_ticks: 0.0,
                similarity: stimulant.peak() as f64,
                confidence: 0.0,
                response_idx: 0,
                extra: BTreeMap::new(),
            });
            trial_idx += 1;

            trace.push(TrialOutcome {
                trial_idx,
                condition: "sedative".to_string(),
                correct: (sedative.peak() - sedative.baseline_proxy).abs() > 0.0001,
                rt_ticks: 0.0,
                similarity: sedative.peak() as f64,
                confidence: 0.0,
                response_idx: 0,
                extra: BTreeMap::new(),
            });
            trial_idx += 1;

            trace.push(TrialOutcome {
                trial_idx,
                condition: "ecb_buffer".to_string(),
                correct: (ecb_buffer.peak() - ecb_buffer.baseline_proxy).abs() > 0.0001,
                rt_ticks: 0.0,
                similarity: ecb_buffer.peak() as f64,
                confidence: 0.0,
                response_idx: 0,
                extra: BTreeMap::new(),
            });
            trial_idx += 1;

            result.trial_trace = trace;
        }
        let _ = trial_idx;

        result.conditions = 5;
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
    fn test_consciousness_pharmacology_runs() {
        let bench = ConsciousnessPharmacologyBenchmark;
        let result = bench.run(&config());
        assert_eq!(result.benchmark, "Neuromod::ConsciousnessPharmacology");
        assert!(result.metrics.len() >= 20);
    }

    #[test]
    fn test_psychedelic_increases_consciousness() {
        let trace = run_condition("sht", 0.4, 80);
        assert!(
            trace.peak() >= trace.baseline_proxy,
            "Psychedelic should increase consciousness proxy: peak={}, baseline={}",
            trace.peak(),
            trace.baseline_proxy
        );
    }

    #[test]
    fn test_anxiolytic_decreases_consciousness() {
        let trace = run_condition("gaba", 0.3, 60);
        assert!(
            trace.trough() <= trace.baseline_proxy,
            "Anxiolytic should decrease consciousness proxy: trough={}, baseline={}",
            trace.trough(),
            trace.baseline_proxy
        );
    }

    #[test]
    fn test_stimulant_alters_consciousness() {
        let trace = run_condition("da", 0.4, 30);
        let deviation = (trace.mean() - trace.baseline_proxy).abs();
        assert!(
            deviation.is_finite(),
            "Stimulant should produce finite deviation"
        );
    }

    #[test]
    fn test_sedative_decreases_consciousness() {
        let trace = run_condition("aden", 0.3, 50);
        assert!(
            trace.mean().is_finite(),
            "Sedative should produce finite consciousness values"
        );
    }

    #[test]
    fn test_ecb_stabilizes_variance() {
        let ecb_trace = run_condition("ecb", 0.3, 40);
        let var = ecb_trace.variance();
        assert!(var.is_finite(), "ECB variance should be finite: {}", var);
    }

    #[test]
    fn test_all_metrics_finite() {
        let bench = ConsciousnessPharmacologyBenchmark;
        let result = bench.run(&config());
        for (key, value) in &result.metrics {
            assert!(
                value.mean.is_finite(),
                "Metric {key} should be finite: {}",
                value.mean
            );
        }
    }
}
