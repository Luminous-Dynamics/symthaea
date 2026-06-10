// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Reading the Mind in the Eyes (RME).
//!
//! Tests the ability to infer emotional/mental states from eye region images.
//! Each trial presents an "eye region" (simulated via HDC) with a target
//! mental state and three foils. The participant selects the most similar
//! mental state label.
//!
//! HDC implementation: Eye-region stimuli are ContinuousHVs blending emotional
//! valence, arousal, and social cues. Mental state labels are prototype HVs.
//! Accuracy reflects the ability to discriminate subtle emotional differences
//! in high-dimensional space.
//!
//! Human baselines (Baron-Cohen et al., 2001; Vellante et al., 2013):
//! - rme_accuracy: 0.72 (SD≈0.09) — 4-AFC accuracy
//! - easy_accuracy: 0.85 (SD≈0.08) — easy items
//! - hard_accuracy: 0.60 (SD≈0.12) — hard items

use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// RME benchmark.
pub struct RmeBenchmark;

struct TrialResult {
    rme_accuracy: f64,
    easy_accuracy: f64,
    hard_accuracy: f64,
    rt_ticks: f64,
}

impl RmeBenchmark {
    fn run_trial(
        &self,
        config: &BenchmarkConfig,
        trial_idx: usize,
        trace: &mut Vec<TrialOutcome>,
        global_trial_idx: &mut usize,
    ) -> TrialResult {
        let diff_model = difficulty_model_for(self.name());
        let dim = config.dimension;
        let seed = config.trial_seed("social", "rme", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        // Mental state prototypes (36 items in the original RME test)
        let n_states = 20;
        let states: Vec<ContinuousHV> = (0..n_states)
            .map(|i| ContinuousHV::random(dim, seed.wrapping_add(100 + i as u64 * 77)))
            .collect();

        let n_items = 36; // Standard RME has 36 items
        let easy_rate = 0.50; // 50% easy, 50% hard

        // Time pressure: reduces deliberation time (Baron-Cohen et al., 2001).
        // Difficulty increases decision noise (harder to discriminate subtle emotions).
        let difficulty_noise: f32 = config.difficulty as f32 * 0.20;
        let noise_level: f32 = 0.25 + config.time_pressure as f32 * 0.15 + difficulty_noise;
        // Encoding noise adds per-comparison perceptual noise (individual differences
        // in emotion recognition; Schlegel et al., 2017 GERT).
        let enc_noise: f32 = config.effective_noise() as f32;

        // Social coherence: when enabled, boosts emotion recognition
        let social_bonus: f32 = if config.enable_social { 0.25 } else { 0.0 };

        let mut total_correct = 0u32;
        let mut easy_correct = 0u32;
        let mut easy_total = 0u32;
        let mut hard_correct = 0u32;
        let mut hard_total = 0u32;
        let mut rt_sum = 0.0f64;

        for item in 0..n_items {
            xor_shift(&mut rng);
            let is_easy = (rng % 100) as f64 / 100.0 < easy_rate;

            // Target mental state
            xor_shift(&mut rng);
            let target_idx = (rng % n_states as u64) as usize;
            let target = &states[target_idx];

            // Generate stimulus (eye region) — blend of target + noise
            xor_shift(&mut rng);
            let stimulus_noise = ContinuousHV::random(dim, rng);
            let signal_weight = (if is_easy { 0.65 } else { 0.40 })
                * diff_model.signal_multiplier(config.difficulty) as f32;
            let stimulus = ContinuousHV::weighted_bundle(
                &[target, &stimulus_noise],
                &[signal_weight, 1.0 - signal_weight],
            );

            // Generate foils: 3 distractor mental states
            let mut foil_indices = Vec::with_capacity(3);
            while foil_indices.len() < 3 {
                xor_shift(&mut rng);
                let f_idx = (rng % n_states as u64) as usize;
                if f_idx != target_idx && !foil_indices.contains(&f_idx) {
                    foil_indices.push(f_idx);
                }
            }

            // For hard items, foils are more similar to target (semantic neighbors)
            let foils: Vec<ContinuousHV> = foil_indices
                .iter()
                .map(|&fi| {
                    if is_easy {
                        states[fi].clone()
                    } else {
                        // Hard foils: blend with target (more confusable)
                        ContinuousHV::weighted_bundle(&[&states[fi], target], &[0.70, 0.30])
                    }
                })
                .collect();

            // 4-AFC: compare stimulus to target + 3 foils
            // Per-option encoding noise (hash-based deterministic per item)
            let target_enc_noise = {
                let ns = seed.wrapping_add(6000 + item as u64 * 17);
                ((ns.wrapping_mul(0x9E3779B97F4A7C15) >> 33) as f32 / (1u64 << 31) as f32) - 0.5
            };
            let target_sim =
                stimulus.similarity(target) + target_enc_noise * enc_noise * 0.12 + social_bonus;
            let foil_sims: Vec<f32> = foils
                .iter()
                .enumerate()
                .map(|(fi, f)| {
                    let foil_enc_noise = {
                        let ns = seed.wrapping_add(6100 + item as u64 * 17 + fi as u64 * 31);
                        ((ns.wrapping_mul(0x9E3779B97F4A7C15) >> 33) as f32 / (1u64 << 31) as f32)
                            - 0.5
                    };
                    stimulus.similarity(f) + foil_enc_noise * enc_noise * 0.12
                })
                .collect();

            // Add noise to all choices
            xor_shift(&mut rng);
            let target_noised = target_sim + (rng % 10000) as f32 / 10000.0 * noise_level;

            // Attention lapse model: on a lapse trial, the subject randomly guesses
            // among the 4 choices. trial_idx * n_items + item gives a unique id per item.
            let item_global_idx = trial_idx * n_items + item;
            let chose_target = if config.should_lapse("rme", item_global_idx) {
                // Random guess: chosen_idx 0 = target, 1-3 = foils
                let chosen_idx = config.check_choice(0, 4, "rme", item_global_idx);
                chosen_idx == 0
            } else {
                let mut result = true;
                for &fsim in &foil_sims {
                    xor_shift(&mut rng);
                    let foil_noised = fsim + (rng % 10000) as f32 / 10000.0 * noise_level;
                    if foil_noised > target_noised {
                        result = false;
                        break;
                    }
                }
                result
            };

            if chose_target {
                total_correct += 1;
                if is_easy {
                    easy_correct += 1;
                } else {
                    hard_correct += 1;
                }
            }

            if is_easy {
                easy_total += 1;
            } else {
                hard_total += 1;
            }

            // RT: harder items take longer
            let base_rt = if is_easy { 4.0 } else { 6.0 };
            let tp_speedup = config.time_pressure * 1.5;
            let item_rt = (base_rt - tp_speedup).max(1.0);
            rt_sum += item_rt;

            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: *global_trial_idx,
                    condition: if is_easy {
                        "easy".to_string()
                    } else {
                        "hard".to_string()
                    },
                    correct: chose_target,
                    rt_ticks: item_rt,
                    similarity: target_sim as f64,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
                *global_trial_idx += 1;
            }

            let _ = item; // suppress unused warning
        }

        let rme_acc = total_correct as f64 / n_items as f64;
        let easy_acc = if easy_total > 0 {
            easy_correct as f64 / easy_total as f64
        } else {
            0.0
        };
        let hard_acc = if hard_total > 0 {
            hard_correct as f64 / hard_total as f64
        } else {
            0.0
        };
        let mean_rt = rt_sum / n_items as f64;

        TrialResult {
            rme_accuracy: rme_acc,
            easy_accuracy: easy_acc,
            hard_accuracy: hard_acc,
            rt_ticks: mean_rt,
        }
    }
}

impl PsychBenchmark for RmeBenchmark {
    fn name(&self) -> &str {
        "Social::RME"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Reading the Mind in the Eyes",
            citation: "Baron-Cohen et al. (2001)",
            year: 2001,
            doi: Some("10.1111/j.1469-7610.2001.tb01739.x"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut accs = Vec::new();
        let mut easy_accs = Vec::new();
        let mut hard_accs = Vec::new();
        let mut rts = Vec::new();
        let mut trace = Vec::new();
        let mut global_trial_idx = 0usize;

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial, &mut trace, &mut global_trial_idx);
            accs.push(r.rme_accuracy);
            easy_accs.push(r.easy_accuracy);
            hard_accs.push(r.hard_accuracy);
            rts.push(r.rt_ticks);
        }

        result.insert("rme_accuracy", MetricValue::from_samples(&accs));
        result.insert("easy_accuracy", MetricValue::from_samples(&easy_accs));
        result.insert("hard_accuracy", MetricValue::from_samples(&hard_accs));
        result.insert("rt_ticks", MetricValue::from_samples(&rts));

        if config.trial_trace {
            result.trial_trace = trace;
        }

        result.conditions = 2; // easy vs hard
        result.trials_per_condition = config.trials_per_condition;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rme_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = RmeBenchmark.run(&config);
        assert!(result.metrics.contains_key("rme_accuracy"));
        assert!(result.metrics.contains_key("easy_accuracy"));
        assert!(result.metrics.contains_key("hard_accuracy"));
        assert!(result.metrics.contains_key("rt_ticks"));
    }

    #[test]
    fn test_rme_finite_metrics() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = RmeBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_rme_accuracy_bounded() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = RmeBenchmark.run(&config);
        let acc = result.metrics["rme_accuracy"].mean;
        assert!(
            acc >= 0.0 && acc <= 1.0,
            "accuracy ({:.3}) out of bounds",
            acc
        );
        // Should be above chance (0.25 for 4-AFC) given non-trivial dim
        assert!(acc > 0.25, "accuracy ({:.3}) should be above chance", acc);
    }

    #[test]
    fn test_rme_time_pressure() {
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
        let r_base = RmeBenchmark.run(&base);
        let r_press = RmeBenchmark.run(&pressed);
        let rt_base = r_base.metrics["rt_ticks"].mean;
        let rt_press = r_press.metrics["rt_ticks"].mean;
        assert!(
            rt_press <= rt_base + 0.5,
            "time pressure should reduce RT: base={:.2}, pressed={:.2}",
            rt_base,
            rt_press
        );
    }
}
