// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Emotional Stroop Task.
//!
//! Name the ink color of emotionally valenced words. Negative words slow
//! responses and reduce accuracy due to attention capture by threat content.
//!
//! HDC implementation: word meaning (valence) and ink color encoded as
//! separate HDC vectors. Negative valence creates interference by partially
//! activating competing response channels via affect-attention interaction.
//!
//! Human baselines (Williams et al. 1996; Bar-Haim et al. 2007):
//! - neutral_accuracy: 0.95 (SD≈0.03)
//! - negative_accuracy: 0.88 (SD≈0.06)
//! - emotional_interference: 0.07 (SD≈0.04)

use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Emotional Stroop benchmark.
pub struct EmotionalStroopBenchmark;

struct TrialResult {
    neutral_accuracy: f64,
    negative_accuracy: f64,
    emotional_interference: f64,
    neutral_rts: Vec<f64>,
    negative_rts: Vec<f64>,
}

impl EmotionalStroopBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let diff_model = difficulty_model_for(self.name());
        let dim = config.dimension;
        let seed = config.trial_seed("affect", "emotional_stroop", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        // 4 color HVs (the response set)
        let color_hvs: Vec<ContinuousHV> = (0..4)
            .map(|i| ContinuousHV::random(dim, seed.wrapping_add(100 + i)))
            .collect();

        // Emotional interference: negative valence attracts attention,
        // partially activating a competing color response
        let emotional_weight: f32 =
            0.28 * diff_model.interference_multiplier(config.difficulty) as f32;
        // Time pressure: base 0.25 yields ~10% emotional interference (Williams et al., 1996);
        // +0.15/unit raises temperature, modeling noisier color selection under SAT (Heitz, 2014).
        let temperature: f64 = (0.25 + config.time_pressure * 0.15)
            * diff_model.temperature_multiplier(config.difficulty);
        let trials_per_condition = 40;

        let mut neutral_correct = 0u32;
        let mut negative_correct = 0u32;
        let mut neutral_rts = Vec::new();
        let mut negative_rts = Vec::new();

        for trial in 0..(trials_per_condition * 2) {
            let is_negative = trial % 2 == 0;

            // Pick random ink color (correct answer)
            xor_shift(&mut rng);
            let ink_idx = (rng % 4) as usize;

            let combined = if is_negative {
                // Negative word: valence captures attention, interferes with color naming
                // Pick a random "competing" color activation from emotional capture
                xor_shift(&mut rng);
                let mut competing_idx = (rng % 3) as usize;
                if competing_idx >= ink_idx {
                    competing_idx += 1;
                }
                // combined = ink_color + emotional_weight * competing_color
                let ink_act = &color_hvs[ink_idx];
                let competing_act = color_hvs[competing_idx].scale(emotional_weight);
                ContinuousHV::bundle(&[ink_act, &competing_act])
            } else {
                // Neutral word: minimal interference
                let noise = ContinuousHV::random(dim, seed.wrapping_add(2000 + trial as u64));
                let noise_act = noise.scale(emotional_weight * 0.2);
                ContinuousHV::bundle(&[&color_hvs[ink_idx], &noise_act])
            };

            // Compute similarity to each color candidate
            // Encoding noise adds per-comparison noise that degrades color
            // discrimination (Lu & Dosher, 1998 noise exclusion model).
            let enc_noise = config.effective_noise() as f32;
            let sims: Vec<f64> = color_hvs
                .iter()
                .enumerate()
                .map(|(ci, c)| {
                    let raw = combined.similarity(c);
                    let noise_seed = seed.wrapping_add(5000 + trial as u64 * 7 + ci as u64 * 31);
                    let noise_val = ((noise_seed.wrapping_mul(0x9E3779B97F4A7C15) >> 33) as f32
                        / (1u64 << 31) as f32)
                        - 0.5;
                    (raw + noise_val * enc_noise * 0.15) as f64
                })
                .collect();

            // Softmax response selection
            let max_sim = sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_sims: Vec<f64> = sims
                .iter()
                .map(|s| ((s - max_sim) / temperature).exp())
                .collect();
            let exp_sum: f64 = exp_sims.iter().sum();

            xor_shift(&mut rng);
            let r = (rng % 10000) as f64 / 10000.0;
            let mut cumsum = 0.0;
            let mut response_idx = 0;
            for (i, e) in exp_sims.iter().enumerate() {
                cumsum += e / exp_sum;
                if r < cumsum {
                    response_idx = i;
                    break;
                }
            }

            // Attention lapse: negative-valence trials capture attention more,
            // increasing lapse vulnerability (Williams et al., 1996). The boost
            // scales with base lapse_rate so it only activates during reliability testing.
            let unique_trial = trial_idx * (trials_per_condition * 2) + trial;
            let emotional_boost = if is_negative {
                config.lapse_rate * 0.6
            } else {
                0.0
            };
            let effective_lapse = config.lapse_rate + emotional_boost;
            let lapse_seed = config.trial_seed("emotional_stroop", "lapse", unique_trial);
            response_idx = if (lapse_seed % 10000) as f64 / 10000.0 < effective_lapse {
                let h = config.trial_seed("emotional_stroop", "lapse_choice", unique_trial);
                h as usize % 4
            } else {
                response_idx
            };

            // RT proxy: deliberation ticks based on decision margin
            let probs: Vec<f64> = exp_sims.iter().map(|e| e / exp_sum).collect();
            let mut sorted_probs = probs.clone();
            sorted_probs.sort_by(|a, b| b.total_cmp(a));
            let decision_margin = (sorted_probs[0] - sorted_probs[1]).clamp(0.0, 1.0);
            let rt_ticks = 8.0 + (1.0 - decision_margin) * 12.0;

            let correct = response_idx == ink_idx;
            if is_negative {
                if correct {
                    negative_correct += 1;
                }
                negative_rts.push(rt_ticks);
            } else {
                if correct {
                    neutral_correct += 1;
                }
                neutral_rts.push(rt_ticks);
            }
        }

        let neut_acc = neutral_correct as f64 / trials_per_condition as f64;
        let neg_acc = negative_correct as f64 / trials_per_condition as f64;

        TrialResult {
            neutral_accuracy: neut_acc,
            negative_accuracy: neg_acc,
            emotional_interference: neut_acc - neg_acc,
            neutral_rts,
            negative_rts,
        }
    }
}

impl PsychBenchmark for EmotionalStroopBenchmark {
    fn name(&self) -> &str {
        "Affect::EmotionalStroop"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Emotional Stroop",
            citation: "Williams et al. (1996)",
            year: 1996,
            doi: Some("10.1037/0033-2909.120.1.3"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut neutral_accs = Vec::new();
        let mut negative_accs = Vec::new();
        let mut interferences = Vec::new();
        let mut all_neutral_rts = Vec::new();
        let mut all_negative_rts = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            neutral_accs.push(r.neutral_accuracy);
            negative_accs.push(r.negative_accuracy);
            interferences.push(r.emotional_interference);
            all_neutral_rts.extend_from_slice(&r.neutral_rts);
            all_negative_rts.extend_from_slice(&r.negative_rts);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "emotional_stroop".to_string(),
                    correct: r.neutral_accuracy > 0.5,
                    rt_ticks: r.neutral_rts.first().copied().unwrap_or(0.0),
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("neutral_accuracy", MetricValue::from_samples(&neutral_accs));
        result.insert(
            "negative_accuracy",
            MetricValue::from_samples(&negative_accs),
        );
        result.insert(
            "emotional_interference",
            MetricValue::from_samples(&interferences),
        );
        result.insert(
            "neutral::rt_ticks",
            MetricValue::from_samples(&all_neutral_rts),
        );
        result.insert(
            "negative::rt_ticks",
            MetricValue::from_samples(&all_negative_rts),
        );

        result.conditions = 2;
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
    fn test_emotional_stroop_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = EmotionalStroopBenchmark.run(&config);
        assert!(result.metrics.contains_key("neutral_accuracy"));
        assert!(result.metrics.contains_key("negative_accuracy"));
        assert!(result.metrics.contains_key("emotional_interference"));
    }

    #[test]
    fn test_emotional_stroop_finite_metrics() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = EmotionalStroopBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_emotional_interference_direction() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = EmotionalStroopBenchmark.run(&config);
        let neutral = result.metrics["neutral_accuracy"].mean;
        let negative = result.metrics["negative_accuracy"].mean;
        assert!(
            neutral >= negative - 0.05,
            "neutral ({:.3}) should be >= negative ({:.3})",
            neutral,
            negative
        );
    }
}
