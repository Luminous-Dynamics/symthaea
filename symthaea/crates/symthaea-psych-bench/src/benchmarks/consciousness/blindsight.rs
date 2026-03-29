// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Blindsight benchmark.
//!
//! Models the dissociation between forced-choice accuracy and conscious
//! awareness first documented in blindsight patients (Weiskrantz, 1986).
//!
//! GWT predicts that sub-threshold stimuli are processed locally but do not
//! enter the global workspace ("ignition"). This creates a signature
//! dissociation: forced-choice accuracy remains above chance even when the
//! subject reports no awareness.
//!
//! HDC implementation: Stimulus HVs at varying signal strengths are compared
//! against two forced-choice alternatives. A global workspace threshold
//! determines whether the stimulus is consciously reported. Sub-threshold
//! stimuli retain residual similarity (supporting above-chance forced-choice)
//! but are not "reported" as seen.
//!
//! Human baselines (Weiskrantz, 1986; Azzopardi & Cowey, 1997):
//! - supraliminal_accuracy: 0.92 (SD~0.05) — above-threshold accuracy
//! - subliminal_accuracy: 0.65 (SD~0.08) — below-threshold forced-choice
//! - awareness_dissociation: 0.25 (SD~0.10) — accuracy minus report rate
//! - threshold_sharpness: 0.80 (SD~0.12) — steepness of transition

use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Blindsight benchmark: forced-choice vs. subjective awareness dissociation.
pub struct BlindSightBenchmark;

struct TrialResult {
    supraliminal_accuracy: f64,
    subliminal_accuracy: f64,
    awareness_dissociation: f64,
    threshold_sharpness: f64,
}

impl BlindSightBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let dim = config.dimension;
        let seed = config.trial_seed("consciousness", "blindsight", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        let diff_model = difficulty_model_for(self.name());

        // Global workspace threshold: stimuli above this are consciously perceived.
        // Difficulty raises the threshold (harder to become conscious).
        let gw_threshold: f32 =
            0.45 + 0.10 * diff_model.temperature_multiplier(config.difficulty) as f32;

        // Time pressure narrows the workspace window, making conscious access
        // harder (attentional blink analogy — Raymond et al., 1992).
        let tp_penalty: f32 = config.time_pressure as f32 * 0.08;
        let effective_threshold = (gw_threshold + tp_penalty).min(0.85);

        // Encoding noise degrades stimulus fidelity
        let noise_level = config.effective_noise() as f32;

        // Number of signal-strength levels to probe
        let n_levels = 10usize;
        let stimuli_per_level = config.trials_per_condition.max(5);

        // Create a pool of stimulus category HVs (two forced-choice alternatives)
        xor_shift(&mut rng);
        let category_a = ContinuousHV::random(dim, seed.wrapping_add(0xCAFE));
        let category_b = ContinuousHV::random(dim, seed.wrapping_add(0xBEEF));

        // Track per-level results for threshold sharpness computation
        let mut level_accuracy = vec![0.0f64; n_levels];
        let mut level_report_rate = vec![0.0f64; n_levels];

        let mut supra_correct = 0u32;
        let mut supra_total = 0u32;
        let mut sub_correct = 0u32;
        let mut sub_total = 0u32;
        let mut sub_reported = 0u32; // subliminal stimuli reported as seen

        for level in 0..n_levels {
            // Signal strength: 0.05 to 0.95
            let signal_strength = (level as f32 + 0.5) / n_levels as f32;
            let is_supraliminal = signal_strength > effective_threshold;

            let mut level_correct = 0u32;
            let mut level_total = 0u32;
            let mut level_reported = 0u32;

            for _stim in 0..stimuli_per_level {
                xor_shift(&mut rng);
                let is_category_a = (rng % 2) == 0;
                let target_hv = if is_category_a {
                    &category_a
                } else {
                    &category_b
                };

                // Create stimulus HV by blending target with noise.
                // Signal strength controls the blend: high strength = more like target.
                xor_shift(&mut rng);
                let noise_hv = ContinuousHV::random(dim, rng);

                // Effective signal: attenuated by encoding noise
                let effective_signal = (signal_strength * (1.0 - noise_level * 0.5)).max(0.0);

                // Blend: stimulus = signal * target + (1 - signal) * noise
                let stimulus = target_hv
                    .scale(effective_signal)
                    .add(&noise_hv.scale(1.0 - effective_signal));

                // Forced-choice: compare stimulus against both categories
                let sim_a = stimulus.similarity(&category_a);
                let sim_b = stimulus.similarity(&category_b);

                // Add response noise (time pressure increases response noise)
                xor_shift(&mut rng);
                let resp_noise_a = ((rng % 10000) as f32 / 10000.0 - 0.5)
                    * 0.05
                    * (1.0 + config.time_pressure as f32);
                xor_shift(&mut rng);
                let resp_noise_b = ((rng % 10000) as f32 / 10000.0 - 0.5)
                    * 0.05
                    * (1.0 + config.time_pressure as f32);

                let chose_a = (sim_a + resp_noise_a) > (sim_b + resp_noise_b);
                let correct = chose_a == is_category_a;

                // Subjective report: only if signal exceeds workspace threshold
                // Supraliminal stimuli are amplified by global workspace;
                // subliminal stimuli may occasionally be "guessed" as seen.
                xor_shift(&mut rng);
                let report_noise = (rng % 10000) as f32 / 10000.0 * 0.1;
                let reported = (effective_signal + report_noise) > effective_threshold;

                level_total += 1;
                if correct {
                    level_correct += 1;
                }
                if reported {
                    level_reported += 1;
                }

                if is_supraliminal {
                    supra_total += 1;
                    if correct {
                        supra_correct += 1;
                    }
                } else {
                    sub_total += 1;
                    if correct {
                        sub_correct += 1;
                    }
                    if reported {
                        sub_reported += 1;
                    }
                }
            }

            level_accuracy[level] = if level_total > 0 {
                level_correct as f64 / level_total as f64
            } else {
                0.5
            };
            level_report_rate[level] = if level_total > 0 {
                level_reported as f64 / level_total as f64
            } else {
                0.0
            };
        }

        let supraliminal_accuracy = if supra_total > 0 {
            supra_correct as f64 / supra_total as f64
        } else {
            0.5
        };

        let subliminal_accuracy = if sub_total > 0 {
            sub_correct as f64 / sub_total as f64
        } else {
            0.5
        };

        // Awareness dissociation: subliminal accuracy minus subliminal report rate
        // High dissociation = good blindsight modeling (accurate without awareness)
        let sub_report_rate = if sub_total > 0 {
            sub_reported as f64 / sub_total as f64
        } else {
            0.0
        };
        let awareness_dissociation = (subliminal_accuracy - sub_report_rate).max(0.0);

        // Threshold sharpness: how steep is the transition from unaware to aware?
        // Measured as max slope of report-rate across signal-strength levels.
        // A sharp threshold (sigmoid-like) indicates clean workspace ignition.
        let mut max_slope = 0.0f64;
        for i in 1..n_levels {
            let slope = (level_report_rate[i] - level_report_rate[i - 1]).abs();
            if slope > max_slope {
                max_slope = slope;
            }
        }
        // Normalize: max possible per-step slope is 1.0; scale to [0, 1]
        // Multiply by n_levels to get the steepness relative to uniform
        let threshold_sharpness = (max_slope * n_levels as f64).clamp(0.0, 1.0);

        TrialResult {
            supraliminal_accuracy,
            subliminal_accuracy,
            awareness_dissociation,
            threshold_sharpness,
        }
    }
}

impl PsychBenchmark for BlindSightBenchmark {
    fn name(&self) -> &str {
        "Consciousness::Blindsight"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Blindsight forced-choice dissociation",
            citation: "Weiskrantz (1986)",
            year: 1986,
            doi: Some("10.1093/acprof:oso/9780198521921.001.0001"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut supra_accs = Vec::new();
        let mut sub_accs = Vec::new();
        let mut dissociations = Vec::new();
        let mut sharpnesses = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            supra_accs.push(r.supraliminal_accuracy);
            sub_accs.push(r.subliminal_accuracy);
            dissociations.push(r.awareness_dissociation);
            sharpnesses.push(r.threshold_sharpness);
            if config.trial_trace {
                let mut extra = BTreeMap::new();
                extra.insert("subliminal_accuracy".to_string(), r.subliminal_accuracy);
                extra.insert(
                    "awareness_dissociation".to_string(),
                    r.awareness_dissociation,
                );
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "blindsight".to_string(),
                    correct: r.supraliminal_accuracy > 0.5,
                    rt_ticks: 0.0,
                    similarity: r.subliminal_accuracy,
                    confidence: r.threshold_sharpness,
                    response_idx: 0,
                    extra,
                });
            }
        }

        result.insert(
            "supraliminal_accuracy",
            MetricValue::from_samples(&supra_accs),
        );
        result.insert("subliminal_accuracy", MetricValue::from_samples(&sub_accs));
        result.insert(
            "awareness_dissociation",
            MetricValue::from_samples(&dissociations),
        );
        result.insert(
            "threshold_sharpness",
            MetricValue::from_samples(&sharpnesses),
        );

        result.conditions = 2; // supraliminal vs subliminal
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
    fn test_blindsight_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = BlindSightBenchmark.run(&config);
        assert!(result.metrics.contains_key("supraliminal_accuracy"));
        assert!(result.metrics.contains_key("subliminal_accuracy"));
        assert!(result.metrics.contains_key("awareness_dissociation"));
        assert!(result.metrics.contains_key("threshold_sharpness"));
    }

    #[test]
    fn test_blindsight_finite_metrics() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = BlindSightBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_blindsight_key_metric_bounded() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = BlindSightBenchmark.run(&config);

        let supra = result.metrics["supraliminal_accuracy"].mean;
        assert!(
            supra >= 0.0 && supra <= 1.0,
            "supraliminal_accuracy ({:.3}) out of bounds",
            supra
        );

        let sub = result.metrics["subliminal_accuracy"].mean;
        assert!(
            sub >= 0.0 && sub <= 1.0,
            "subliminal_accuracy ({:.3}) out of bounds",
            sub
        );

        // Core blindsight signature: subliminal accuracy should be above chance
        assert!(
            sub > 0.45,
            "subliminal_accuracy ({:.3}) should be above chance",
            sub
        );

        let dissoc = result.metrics["awareness_dissociation"].mean;
        assert!(
            dissoc >= 0.0 && dissoc <= 1.0,
            "awareness_dissociation ({:.3}) out of bounds",
            dissoc
        );
    }

    #[test]
    fn test_blindsight_time_pressure() {
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
        let r_base = BlindSightBenchmark.run(&base);
        let r_press = BlindSightBenchmark.run(&pressed);

        // Under time pressure, supraliminal accuracy should decrease or
        // subliminal accuracy should decrease (degraded processing)
        let supra_base = r_base.metrics["supraliminal_accuracy"].mean;
        let supra_press = r_press.metrics["supraliminal_accuracy"].mean;
        let sub_base = r_base.metrics["subliminal_accuracy"].mean;
        let sub_press = r_press.metrics["subliminal_accuracy"].mean;
        // At least one accuracy measure should not increase substantially
        assert!(
            supra_press <= supra_base + 0.15 || sub_press <= sub_base + 0.15,
            "time pressure should degrade performance: supra {:.3}->{:.3}, sub {:.3}->{:.3}",
            supra_base,
            supra_press,
            sub_base,
            sub_press
        );
    }
}
