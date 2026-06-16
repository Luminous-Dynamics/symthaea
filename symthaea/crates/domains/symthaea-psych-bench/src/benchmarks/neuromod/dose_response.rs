// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Dose-Response Curve Benchmark — Monotonic Behavioral Output
//!
//! Graded dose sweeps (0.1–0.8) per transmitter verify monotonic behavioral
//! output via a simplified Kendall tau concordance score.
//!
//! Uses direct level clamping (not PK injection) to isolate the dose→behavior
//! mapping from tolerance/receptor adaptation dynamics.
//!
//! Science: Clark (1937) — Occupancy theory; Hill (1910) — dose-response curves.

use crate::harness::PsychBenchmark;
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use std::collections::BTreeMap;
use symthaea_neuromodulators::{NeuromodulatorBath, NeuromodulatorInputs};

/// Dose levels for the sweep (treated as direct transmitter levels).
const DOSES: [f32; 5] = [0.1, 0.2, 0.3, 0.5, 0.8];
/// Warmup cycles before observation.
const WARMUP: usize = 10;
/// Observation cycles per dose level.
const OBSERVE: usize = 30;

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

/// Monotonicity score: fraction of concordant pairs (simplified Kendall tau, 0.0–1.0).
fn monotonicity_score(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 1.0;
    }
    let mut concordant = 0u64;
    let mut total = 0u64;
    for i in 0..n {
        for j in (i + 1)..n {
            total += 1;
            if values[j] >= values[i] {
                concordant += 1;
            }
        }
    }
    concordant as f64 / total as f64
}

/// Clamp a specific transmitter to the given level.
fn clamp_transmitter(bath: &mut NeuromodulatorBath, target: &str, level: f32) {
    match target {
        "da" => bath.clamp_all_levels(Some(level), None, None, None, None, None, None),
        "ne" => bath.clamp_all_levels(None, Some(level), None, None, None, None, None),
        "sht" => bath.clamp_all_levels(None, None, Some(level), None, None, None, None),
        "ach" => bath.clamp_all_levels(None, None, None, Some(level), None, None, None),
        "gaba" => bath.clamp_all_levels(None, None, None, None, Some(level), None, None),
        _ => {}
    }
}

/// Run a dose sweep for a given transmitter, measuring a behavioral metric.
///
/// Uses direct level clamping each cycle to isolate dose→behavior from PK/tolerance.
fn dose_sweep(target: &str, metric_fn: fn(&NeuromodulatorBath) -> f32) -> Vec<f64> {
    let inputs = neutral_inputs();
    let mut means = Vec::with_capacity(DOSES.len());

    for &dose in &DOSES {
        let mut bath = NeuromodulatorBath::default();

        // Warmup with clamped level
        for _ in 0..WARMUP {
            bath.update(&inputs);
            clamp_transmitter(&mut bath, target, dose);
        }

        // Observe with clamped level
        let mut sum = 0.0_f64;
        for _ in 0..OBSERVE {
            bath.update(&inputs);
            clamp_transmitter(&mut bath, target, dose);
            sum += metric_fn(&bath) as f64;
        }
        means.push(sum / OBSERVE as f64);
    }
    means
}

/// Dose-Response Curve Benchmark.
///
/// Validates that graded transmitter levels produce monotonically increasing
/// (or decreasing, for GABA inhibition) behavioral outputs.
pub struct DoseResponseBenchmark;

impl PsychBenchmark for DoseResponseBenchmark {
    fn name(&self) -> &str {
        "DoseResponse"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut trace = Vec::new();
        let mut trial_idx = 0usize;

        // DA → learning_rate_factor (expected increasing)
        let da_values = dose_sweep("da", |b| b.learning_rate_factor());
        let da_score = monotonicity_score(&da_values);

        // NE → exploration_delta (expected increasing)
        let ne_values = dose_sweep("ne", |b| b.exploration_delta());
        let ne_score = monotonicity_score(&ne_values);

        // 5-HT → confidence_delta (expected increasing)
        let sht_values = dose_sweep("sht", |b| b.confidence_delta());
        let sht_score = monotonicity_score(&sht_values);

        // ACh → attention_factor (expected increasing)
        let ach_values = dose_sweep("ach", |b| b.attention_factor());
        let ach_score = monotonicity_score(&ach_values);

        // GABA → global_inhibition (expected decreasing — negate for monotonicity check)
        let gaba_values = dose_sweep("gaba", |b| b.global_inhibition());
        let gaba_negated: Vec<f64> = gaba_values.iter().map(|v| -v).collect();
        let gaba_score = monotonicity_score(&gaba_negated);

        // Extreme dose saturation: all outputs should remain finite at dose=0.8
        let extreme_bath = {
            let mut b = NeuromodulatorBath::default();
            b.inject("da", 0.8, 200);
            let inputs = neutral_inputs();
            for _ in 0..50 {
                b.update(&inputs);
            }
            b
        };
        let extreme_finite = extreme_bath.learning_rate_factor().is_finite()
            && extreme_bath.exploration_delta().is_finite()
            && extreme_bath.confidence_delta().is_finite()
            && extreme_bath.attention_factor().is_finite()
            && extreme_bath.global_inhibition().is_finite();

        let mut result = BenchmarkResult::new(self.name(), None);
        result.insert("da_monotonicity", MetricValue::from_samples(&[da_score]));
        result.insert("ne_monotonicity", MetricValue::from_samples(&[ne_score]));
        result.insert("sht_monotonicity", MetricValue::from_samples(&[sht_score]));
        result.insert("ach_monotonicity", MetricValue::from_samples(&[ach_score]));
        result.insert(
            "gaba_monotonicity",
            MetricValue::from_samples(&[gaba_score]),
        );
        result.insert(
            "extreme_saturation_finite",
            MetricValue::from_samples(&[if extreme_finite { 1.0 } else { 0.0 }]),
        );

        if config.trial_trace {
            for (name, score) in &[
                ("da", da_score),
                ("ne", ne_score),
                ("sht", sht_score),
                ("ach", ach_score),
                ("gaba", gaba_score),
            ] {
                trace.push(TrialOutcome {
                    trial_idx,
                    condition: name.to_string(),
                    correct: *score > 0.5,
                    rt_ticks: 0.0,
                    similarity: *score,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
                trial_idx += 1;
            }
            result.trial_trace = trace;
        }
        let _ = trial_idx;

        result.conditions = 5;
        result.trials_per_condition = DOSES.len();
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
    fn test_dose_response_runs() {
        let bench = DoseResponseBenchmark;
        let result = bench.run(&config());
        assert_eq!(result.benchmark, "DoseResponse");
        assert!(result.metrics.len() >= 6);
    }

    #[test]
    fn test_da_monotonic() {
        let bench = DoseResponseBenchmark;
        let result = bench.run(&config());
        let score = result.metrics.get("da_monotonicity").unwrap().mean;
        assert!(
            score > 0.6,
            "DA dose-response should be monotonic: score={score}"
        );
    }

    #[test]
    fn test_ne_monotonic() {
        let bench = DoseResponseBenchmark;
        let result = bench.run(&config());
        let score = result.metrics.get("ne_monotonicity").unwrap().mean;
        assert!(
            score > 0.6,
            "NE dose-response should be monotonic: score={score}"
        );
    }

    #[test]
    fn test_sht_monotonic() {
        let bench = DoseResponseBenchmark;
        let result = bench.run(&config());
        let score = result.metrics.get("sht_monotonicity").unwrap().mean;
        assert!(
            score > 0.6,
            "5-HT dose-response should be monotonic: score={score}"
        );
    }

    #[test]
    fn test_ach_monotonic() {
        let bench = DoseResponseBenchmark;
        let result = bench.run(&config());
        let score = result.metrics.get("ach_monotonicity").unwrap().mean;
        assert!(
            score > 0.6,
            "ACh dose-response should be monotonic: score={score}"
        );
    }

    #[test]
    fn test_gaba_monotonic() {
        let bench = DoseResponseBenchmark;
        let result = bench.run(&config());
        let score = result.metrics.get("gaba_monotonicity").unwrap().mean;
        assert!(
            score > 0.6,
            "GABA dose-response should be monotonic (negated): score={score}"
        );
    }

    #[test]
    fn test_extreme_dose_saturation() {
        let bench = DoseResponseBenchmark;
        let result = bench.run(&config());
        let finite = result
            .metrics
            .get("extreme_saturation_finite")
            .unwrap()
            .mean;
        assert!(
            (finite - 1.0).abs() < f64::EPSILON,
            "Extreme dose should produce finite outputs"
        );
    }

    #[test]
    fn test_all_metrics_finite() {
        let bench = DoseResponseBenchmark;
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
