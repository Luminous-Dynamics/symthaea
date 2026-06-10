// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Proprioceptive Drift (Rubber Hand Illusion).
//!
//! Synchronous multisensory stimulation creates body ownership over an
//! external object. The brain updates proprioceptive representation toward
//! the visual input. Tests embodied consciousness via multisensory integration.
//!
//! HDC implementation: "body part" HVs (proprioceptive hand position),
//! "visual input" HVs (rubber hand position), and "tactile" HVs (touch events)
//! are created. Multisensory binding bundles visual+tactile with temporal
//! synchrony weighting — synchronous stimulation produces high binding
//! similarity, asynchronous produces low. Proprioceptive drift is modeled as
//! EMA shift of the proprioceptive HV toward the visual HV when binding
//! strength exceeds threshold. Ownership emerges when drift exceeds threshold.
//!
//! Human baselines (Botvinick & Cohen, 1998; Tsakiris & Haggard, 2005):
//! - synchronous_drift: 0.25 (SD≈0.10) — drift magnitude (normalized 0-1)
//! - asynchronous_drift: 0.05 (SD≈0.04) — drift in asynchronous control
//! - ownership_rate: 0.75 (SD≈0.15) — fraction of trials with ownership
//! - drift_difference: 0.20 (SD≈0.08) — synchronous minus asynchronous

use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Proprioceptive Drift (Rubber Hand Illusion) benchmark.
pub struct ProprioceptiveDriftBenchmark;

struct TrialResult {
    synchronous_drift: f64,
    asynchronous_drift: f64,
    ownership_rate: f64,
    drift_difference: f64,
}

impl ProprioceptiveDriftBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let diff_model = difficulty_model_for(self.name());
        let dim = config.dimension;
        let seed = config.trial_seed("motor", "proprioceptive_drift", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        // Body representation HVs
        let proprioceptive_hand = ContinuousHV::random(dim, seed.wrapping_add(1));
        let visual_rubber_hand = ContinuousHV::random(dim, seed.wrapping_add(2));

        // Tactile event prototypes (synchronous touch at multiple locations)
        let num_touch_events: u64 = 12;
        let tactile_events: Vec<ContinuousHV> = (0..num_touch_events)
            .map(|i| ContinuousHV::random(dim, seed.wrapping_add(100 + i * 7)))
            .collect();

        // Noise scaling from difficulty and time pressure
        let noise_level: f32 =
            (0.15 + config.time_pressure as f32 * 0.10 + config.effective_noise() as f32 * 0.3)
                * diff_model.signal_multiplier(config.difficulty) as f32;

        // ── Synchronous condition ──
        // Visual and tactile are temporally aligned → strong multisensory binding
        let drift_rate: f32 = 0.08; // EMA rate for proprioceptive update
        let ownership_threshold: f32 = 0.15; // drift beyond which ownership emerges
        let num_stimulation_steps: usize = 20;

        let mut proprio_sync = proprioceptive_hand.clone();
        let mut sync_ownership_count = 0u32;
        let sync_sub_trials: usize = 5;

        for _sub in 0..sync_sub_trials {
            proprio_sync = proprioceptive_hand.clone();

            for step in 0..num_stimulation_steps {
                let touch_idx = step % tactile_events.len();
                let tactile = &tactile_events[touch_idx];

                // Synchronous binding: visual + tactile bundled together (high coherence)
                // Temporal synchrony weight: 1.0 for synchronous
                let sync_weight: f32 = 0.9;
                let multisensory_bind = ContinuousHV::weighted_bundle(
                    &[&visual_rubber_hand, tactile],
                    &[sync_weight, sync_weight],
                );

                // Binding strength: how coherent is the multisensory representation?
                let binding_strength = multisensory_bind.similarity(&visual_rubber_hand).max(0.0);

                // Proprioceptive drift: shift toward visual when binding is strong
                xor_shift(&mut rng);
                let noise = (rng % 10000) as f32 / 10000.0 * noise_level;
                let effective_rate = drift_rate * binding_strength + noise * 0.1;

                // EMA update: proprioceptive representation drifts toward visual
                proprio_sync = ContinuousHV::weighted_bundle(
                    &[&proprio_sync, &visual_rubber_hand],
                    &[1.0 - effective_rate, effective_rate],
                );
            }

            // Measure drift: how far did proprioceptive move toward visual?
            let original_sim = proprioceptive_hand.similarity(&visual_rubber_hand);
            let drifted_sim = proprio_sync.similarity(&visual_rubber_hand);
            let drift = (drifted_sim - original_sim).max(0.0);

            if drift > ownership_threshold {
                sync_ownership_count += 1;
            }

            // Re-seed for sub-trial variation
            xor_shift(&mut rng);
        }

        // Final synchronous drift magnitude
        let original_sim = proprioceptive_hand.similarity(&visual_rubber_hand);
        let sync_drifted_sim = proprio_sync.similarity(&visual_rubber_hand);
        let sync_drift = (sync_drifted_sim - original_sim).max(0.0) as f64;

        // ── Asynchronous condition ──
        // Visual and tactile are temporally misaligned → weak multisensory binding
        let mut proprio_async = proprioceptive_hand.clone();
        let mut _async_ownership_count = 0u32;

        for _sub in 0..sync_sub_trials {
            proprio_async = proprioceptive_hand.clone();

            for step in 0..num_stimulation_steps {
                // Desynchronized: tactile event paired with wrong visual timing
                // Use a different (offset) tactile event to break temporal coherence
                let touch_idx = (step + 5) % tactile_events.len();
                let tactile = &tactile_events[touch_idx];

                // Asynchronous binding: weaker weight represents temporal mismatch
                let async_weight: f32 = 0.25;
                let multisensory_bind = ContinuousHV::weighted_bundle(
                    &[&visual_rubber_hand, tactile],
                    &[async_weight, 1.0 - async_weight],
                );

                // Binding strength is much lower for asynchronous
                let binding_strength = multisensory_bind.similarity(&visual_rubber_hand).max(0.0);

                xor_shift(&mut rng);
                let noise = (rng % 10000) as f32 / 10000.0 * noise_level;
                let effective_rate = drift_rate * binding_strength * 0.3 + noise * 0.05;

                // Minimal drift for asynchronous condition
                proprio_async = ContinuousHV::weighted_bundle(
                    &[&proprio_async, &visual_rubber_hand],
                    &[1.0 - effective_rate, effective_rate],
                );
            }

            let original_sim_a = proprioceptive_hand.similarity(&visual_rubber_hand);
            let drifted_sim_a = proprio_async.similarity(&visual_rubber_hand);
            let drift_a = (drifted_sim_a - original_sim_a).max(0.0);

            if drift_a > ownership_threshold {
                _async_ownership_count += 1;
            }

            xor_shift(&mut rng);
        }

        let async_drifted_sim = proprio_async.similarity(&visual_rubber_hand);
        let async_drift = (async_drifted_sim - original_sim).max(0.0) as f64;

        let ownership_rate = sync_ownership_count as f64 / sync_sub_trials as f64;
        let drift_difference = (sync_drift - async_drift).max(0.0);

        TrialResult {
            synchronous_drift: sync_drift,
            asynchronous_drift: async_drift,
            ownership_rate,
            drift_difference,
        }
    }
}

impl PsychBenchmark for ProprioceptiveDriftBenchmark {
    fn name(&self) -> &str {
        "Motor::ProprioceptiveDrift"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Rubber Hand Illusion (Proprioceptive Drift)",
            citation: "Botvinick & Cohen (1998)",
            year: 1998,
            doi: Some("10.1038/35784"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut sync_drifts = Vec::new();
        let mut async_drifts = Vec::new();
        let mut ownership_rates = Vec::new();
        let mut drift_diffs = Vec::new();
        let mut trace = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            sync_drifts.push(r.synchronous_drift);
            async_drifts.push(r.asynchronous_drift);
            ownership_rates.push(r.ownership_rate);
            drift_diffs.push(r.drift_difference);

            if config.trial_trace {
                let mut extra = BTreeMap::new();
                extra.insert("async_drift".to_string(), r.asynchronous_drift);
                extra.insert("ownership".to_string(), r.ownership_rate);
                trace.push(TrialOutcome {
                    trial_idx: trial,
                    condition: "proprioceptive_drift".to_string(),
                    correct: r.synchronous_drift > r.asynchronous_drift,
                    rt_ticks: 0.0,
                    similarity: r.synchronous_drift,
                    confidence: r.ownership_rate,
                    response_idx: 0,
                    extra,
                });
            }
        }

        result.insert("synchronous_drift", MetricValue::from_samples(&sync_drifts));
        result.insert(
            "asynchronous_drift",
            MetricValue::from_samples(&async_drifts),
        );
        result.insert(
            "ownership_rate",
            MetricValue::from_samples(&ownership_rates),
        );
        result.insert("drift_difference", MetricValue::from_samples(&drift_diffs));

        result.conditions = 2; // synchronous vs asynchronous
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
    fn test_proprioceptive_drift_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = ProprioceptiveDriftBenchmark.run(&config);
        assert!(result.metrics.contains_key("synchronous_drift"));
        assert!(result.metrics.contains_key("asynchronous_drift"));
        assert!(result.metrics.contains_key("ownership_rate"));
        assert!(result.metrics.contains_key("drift_difference"));
    }

    #[test]
    fn test_proprioceptive_drift_finite_metrics() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = ProprioceptiveDriftBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_proprioceptive_drift_key_metric_bounded() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = ProprioceptiveDriftBenchmark.run(&config);
        let sync = result.metrics["synchronous_drift"].mean;
        let async_d = result.metrics["asynchronous_drift"].mean;
        assert!(
            sync >= 0.0 && sync <= 1.0,
            "synchronous_drift ({:.3}) out of [0,1]",
            sync
        );
        assert!(
            async_d >= 0.0 && async_d <= 1.0,
            "asynchronous_drift ({:.3}) out of [0,1]",
            async_d
        );
        // Synchronous should produce more drift than asynchronous
        assert!(
            sync >= async_d - 0.05,
            "synchronous ({:.3}) should >= asynchronous ({:.3})",
            sync,
            async_d
        );
    }

    #[test]
    fn test_proprioceptive_drift_time_pressure() {
        let base = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            time_pressure: 0.0,
            ..Default::default()
        };
        let pressed = BenchmarkConfig {
            time_pressure: 1.0,
            ..base.clone()
        };
        let r_base = ProprioceptiveDriftBenchmark.run(&base);
        let r_press = ProprioceptiveDriftBenchmark.run(&pressed);
        // Under time pressure, drift metrics should still be finite and non-negative
        let drift_base = r_base.metrics["synchronous_drift"].mean;
        let drift_press = r_press.metrics["synchronous_drift"].mean;
        assert!(
            drift_base.is_finite() && drift_press.is_finite(),
            "drift should be finite under time pressure: base={:.3}, pressed={:.3}",
            drift_base,
            drift_press
        );
    }
}
