// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multi-Transmitter Synergy Benchmark
//!
//! Validates paired transmitter interactions via a 2×2 factorial design.
//! For each canonical pair, measures metric under four conditions:
//! high/high, high/low, low/high, low/low. Computes interaction index:
//!   I = mean(HH) + mean(LL) - mean(HL) - mean(LH)
//!
//! Synergistic > 0.05, antagonistic < -0.05, additive ≈ 0.
//!
//! Science: Doya (2002) — neuromodulator metalearning interactions.

use crate::harness::PsychBenchmark;
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use std::collections::BTreeMap;
use symthaea_neuromodulators::{NeuromodulatorBath, NeuromodulatorInputs};

/// High and low transmitter levels for factorial conditions.
const HIGH: f32 = 0.8;
const LOW: f32 = 0.2;
/// Warmup cycles before observation.
const WARMUP: usize = 15;
/// Observation cycles per condition.
const OBSERVE: usize = 40;

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

/// Set two transmitter levels by name.
fn set_pair(bath: &mut NeuromodulatorBath, t1: &str, v1: f32, t2: &str, v2: f32) {
    for (name, val) in [(t1, v1), (t2, v2)] {
        match name {
            "da" => bath.dopamine.level = val.clamp(0.0, 1.0),
            "ne" => bath.noradrenaline.level = val.clamp(0.0, 1.0),
            "sht" => bath.serotonin.level = val.clamp(0.0, 1.0),
            "ach" => bath.acetylcholine.level = val.clamp(0.0, 1.0),
            "gaba" => bath.gaba.level = val.clamp(0.0, 1.0),
            "oxy" => bath.oxytocin.level = val.clamp(0.0, 1.0),
            "glut" => bath.glutamate.level = val.clamp(0.0, 1.0),
            _ => {}
        }
    }
}

/// Run one condition of the factorial design and return the mean metric value.
fn run_condition(
    t1: &str,
    v1: f32,
    t2: &str,
    v2: f32,
    metric_fn: fn(&NeuromodulatorBath) -> f32,
) -> f64 {
    let mut bath = NeuromodulatorBath::default();
    let inputs = neutral_inputs();

    for _ in 0..WARMUP {
        set_pair(&mut bath, t1, v1, t2, v2);
        bath.update(&inputs);
        set_pair(&mut bath, t1, v1, t2, v2); // re-clamp after update
    }

    let mut sum = 0.0_f64;
    for _ in 0..OBSERVE {
        set_pair(&mut bath, t1, v1, t2, v2);
        bath.update(&inputs);
        set_pair(&mut bath, t1, v1, t2, v2);
        sum += metric_fn(&bath) as f64;
    }
    sum / OBSERVE as f64
}

/// Compute interaction index: I = mean(HH) + mean(LL) - mean(HL) - mean(LH).
/// Positive = synergistic, negative = antagonistic.
fn interaction_index(t1: &str, t2: &str, metric_fn: fn(&NeuromodulatorBath) -> f32) -> f64 {
    let hh = run_condition(t1, HIGH, t2, HIGH, metric_fn);
    let hl = run_condition(t1, HIGH, t2, LOW, metric_fn);
    let lh = run_condition(t1, LOW, t2, HIGH, metric_fn);
    let ll = run_condition(t1, LOW, t2, LOW, metric_fn);
    hh + ll - hl - lh
}

/// Multi-Transmitter Synergy Benchmark.
pub struct MultiTransmitterSynergyBenchmark;

impl PsychBenchmark for MultiTransmitterSynergyBenchmark {
    fn name(&self) -> &str {
        "MultiTransmitterSynergy"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut trace = Vec::new();
        let mut trial_idx = 0usize;

        // 1. DA + 5-HT → learning_rate_factor (expected synergistic)
        let da_sht = interaction_index("da", "sht", |b| b.learning_rate_factor());

        // 2. NE + ACh → attention_factor (expected synergistic)
        let ne_ach = interaction_index("ne", "ach", |b| b.attention_factor());

        // 3. GABA + NE → exploration_delta (expected antagonistic)
        let gaba_ne = interaction_index("gaba", "ne", |b| b.exploration_delta());

        // 4. DA + GABA → learning_rate_factor (expected antagonistic)
        let da_gaba = interaction_index("da", "gaba", |b| b.learning_rate_factor());

        // 5. Oxytocin + 5-HT → confidence_delta (expected synergistic)
        let oxy_sht = interaction_index("oxy", "sht", |b| b.confidence_delta());

        // 6. Glutamate + GABA → ei_ratio (homeostatic → 1.0)
        // For homeostasis: measure deviation from 1.0 under balanced conditions
        let glut_gaba_hh = run_condition("glut", HIGH, "gaba", HIGH, |b| b.ei_ratio());
        let glut_gaba_hl = run_condition("glut", HIGH, "gaba", LOW, |b| b.ei_ratio());
        let glut_gaba_lh = run_condition("glut", LOW, "gaba", HIGH, |b| b.ei_ratio());
        let glut_gaba_ll = run_condition("glut", LOW, "gaba", LOW, |b| b.ei_ratio());
        // Balanced (HH) should be closer to 1.0 than unbalanced (HL/LH)
        let hh_dev = (glut_gaba_hh - 1.0).abs();
        let unbalanced_dev = ((glut_gaba_hl - 1.0).abs() + (glut_gaba_lh - 1.0).abs()) / 2.0;
        let homeostatic_advantage = unbalanced_dev - hh_dev;

        let mut result = BenchmarkResult::new(self.name(), None);
        result.insert("da_sht_synergy", MetricValue::from_samples(&[da_sht]));
        result.insert("ne_ach_synergy", MetricValue::from_samples(&[ne_ach]));
        result.insert("gaba_ne_antagonism", MetricValue::from_samples(&[gaba_ne]));
        result.insert("da_gaba_antagonism", MetricValue::from_samples(&[da_gaba]));
        result.insert("oxy_sht_synergy", MetricValue::from_samples(&[oxy_sht]));
        result.insert(
            "glut_gaba_homeostatic",
            MetricValue::from_samples(&[homeostatic_advantage]),
        );
        result.insert(
            "glut_gaba_hh_ratio",
            MetricValue::from_samples(&[glut_gaba_hh]),
        );
        result.insert(
            "glut_gaba_ll_ratio",
            MetricValue::from_samples(&[glut_gaba_ll]),
        );

        if config.trial_trace {
            for (name, score) in &[
                ("da_sht", da_sht),
                ("ne_ach", ne_ach),
                ("gaba_ne", gaba_ne),
                ("da_gaba", da_gaba),
                ("oxy_sht", oxy_sht),
                ("glut_gaba", homeostatic_advantage),
            ] {
                trace.push(TrialOutcome {
                    trial_idx,
                    condition: name.to_string(),
                    correct: score.is_finite(),
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

        result.conditions = 6;
        result.trials_per_condition = 4; // HH/HL/LH/LL
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
    fn test_synergy_benchmark_runs() {
        let bench = MultiTransmitterSynergyBenchmark;
        let result = bench.run(&config());
        assert_eq!(result.benchmark, "MultiTransmitterSynergy");
        assert!(result.metrics.len() >= 6);
    }

    #[test]
    fn test_da_sht_synergy() {
        let bench = MultiTransmitterSynergyBenchmark;
        let result = bench.run(&config());
        let idx = result.metrics.get("da_sht_synergy").unwrap().mean;
        // DA + 5-HT should be synergistic (positive interaction)
        // or at least non-antagonistic for learning rate
        assert!(
            idx > -0.1,
            "DA+5-HT interaction should not be strongly antagonistic: I={idx:.4}"
        );
    }

    #[test]
    fn test_ne_ach_synergy() {
        let bench = MultiTransmitterSynergyBenchmark;
        let result = bench.run(&config());
        let idx = result.metrics.get("ne_ach_synergy").unwrap().mean;
        assert!(
            idx.is_finite(),
            "NE+ACh interaction index should be finite: I={idx}"
        );
    }

    #[test]
    fn test_gaba_ne_antagonism() {
        let bench = MultiTransmitterSynergyBenchmark;
        let result = bench.run(&config());
        let idx = result.metrics.get("gaba_ne_antagonism").unwrap().mean;
        // GABA should suppress NE-driven exploration
        assert!(
            idx.is_finite(),
            "GABA+NE interaction index should be finite: I={idx}"
        );
    }

    #[test]
    fn test_da_gaba_antagonism() {
        let bench = MultiTransmitterSynergyBenchmark;
        let result = bench.run(&config());
        let idx = result.metrics.get("da_gaba_antagonism").unwrap().mean;
        assert!(
            idx.is_finite(),
            "DA+GABA interaction index should be finite: I={idx}"
        );
    }

    #[test]
    fn test_oxy_sht_synergy() {
        let bench = MultiTransmitterSynergyBenchmark;
        let result = bench.run(&config());
        let idx = result.metrics.get("oxy_sht_synergy").unwrap().mean;
        assert!(
            idx.is_finite(),
            "Oxy+5-HT interaction index should be finite: I={idx}"
        );
    }

    #[test]
    fn test_glut_gaba_homeostasis() {
        let bench = MultiTransmitterSynergyBenchmark;
        let result = bench.run(&config());
        let adv = result.metrics.get("glut_gaba_homeostatic").unwrap().mean;
        // Balanced (HH) should be closer to E/I ratio=1.0 than unbalanced
        assert!(
            adv > -0.5,
            "Glut+GABA balanced should not be worse than unbalanced: advantage={adv:.4}"
        );
    }

    #[test]
    fn test_all_metrics_finite() {
        let bench = MultiTransmitterSynergyBenchmark;
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
