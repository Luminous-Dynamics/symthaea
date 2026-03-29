// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Stroop Color-Word Interference Test.
//!
//! Tests inhibitory control (executive attention). The system must identify
//! the ink color of a color word while suppressing the word's semantic content.
//!
//! HDC implementation: models the Stroop effect as activation competition.
//! Reading is "automatic" — the word's semantic meaning activates its
//! corresponding color representation. The system must select the ink color
//! despite interference from the word's automatic color activation.
//!
//! Conditions:
//! - Congruent: word "RED" in red ink (word reinforces ink)
//! - Incongruent: word "RED" in blue ink (word competes with ink)
//! - Neutral: word "XXX" in red ink (no semantic activation)
//!
//! Human baselines (MacLeod, 1991; Stroop, 1935):
//! - congruent_accuracy: 0.98
//! - incongruent_accuracy: 0.88
//! - stroop_effect: 0.10 (congruent - incongruent accuracy)
//! - neutral_accuracy: 0.95

use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Stroop Color-Word Interference benchmark.
pub struct StroopBenchmark;

#[derive(Clone, Copy)]
enum Condition {
    Congruent,
    Incongruent,
    Neutral,
}

impl StroopBenchmark {
    fn run_trial_with_difficulty(
        &self,
        config: &BenchmarkConfig,
        trial_idx: usize,
        diff_model: &crate::harness::difficulty::DifficultyModel,
        trace: &mut Vec<TrialOutcome>,
        global_trial_idx: &mut usize,
    ) -> TrialResult {
        let dim = config.dimension;
        let seed = config.trial_seed("executive", "stroop", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        // 4 color representation HVs: red, blue, green, yellow
        let color_hvs: Vec<ContinuousHV> = (0..4)
            .map(|i| ContinuousHV::random(dim, seed.wrapping_add(100 + i)))
            .collect();

        // Reading automaticity: how strongly the word activates its color.
        // Difficulty amplifies interference (reading automaticity).
        // 0.35 calibrated to produce ~10% Stroop effect (MacLeod, 1991).
        // Higher values (0.45) produced superhuman interference resistance.
        let base_automaticity: f32 = 0.35;
        let reading_automaticity: f32 = (base_automaticity
            * diff_model.interference_multiplier(config.difficulty) as f32)
            .min(0.95);

        // Decision temperature: controls stochasticity of response selection.
        // Difficulty increases temperature (more stochastic responses).
        let base_temperature: f64 = 0.25 + config.time_pressure * 0.15;
        let temperature: f64 =
            base_temperature * diff_model.temperature_multiplier(config.difficulty);

        let trials_per_condition = 40;
        let mut congruent_correct = 0u32;
        let mut incongruent_correct = 0u32;
        let mut neutral_correct = 0u32;
        let mut cong_rts = Vec::new();
        let mut incong_rts = Vec::new();
        let mut neut_rts = Vec::new();
        let collect_trace = config.trial_trace;

        for trial in 0..(trials_per_condition * 3) {
            let condition = match trial % 3 {
                0 => Condition::Congruent,
                1 => Condition::Incongruent,
                _ => Condition::Neutral,
            };

            // Pick a random ink color (the correct answer)
            xor_shift(&mut rng);
            let ink_idx = (rng % 4) as usize;

            // Build the combined activation:
            // ink_activation (strength 1.0) + word_activation (strength reading_automaticity)
            let combined = match condition {
                Condition::Congruent => color_hvs[ink_idx].scale(1.0 + reading_automaticity),
                Condition::Incongruent => {
                    xor_shift(&mut rng);
                    let mut word_idx = (rng % 3) as usize;
                    if word_idx >= ink_idx {
                        word_idx += 1;
                    }
                    let ink_act = &color_hvs[ink_idx];
                    let word_act = color_hvs[word_idx].scale(reading_automaticity);
                    ContinuousHV::bundle(&[ink_act, &word_act])
                }
                Condition::Neutral => {
                    let noise = ContinuousHV::random(dim, seed.wrapping_add(1000 + trial as u64));
                    let noise_act = noise.scale(reading_automaticity * 0.3);
                    ContinuousHV::bundle(&[&color_hvs[ink_idx], &noise_act])
                }
            };

            // Compute similarity to each color candidate
            // Encoding noise adds per-comparison noise (individual differences in
            // perceptual discrimination; Lu & Dosher, 1998 noise exclusion model).
            let enc_noise = config.effective_noise() as f32;
            let sims: Vec<f64> = color_hvs
                .iter()
                .enumerate()
                .map(|(ci, c)| {
                    let raw = combined.similarity(c);
                    // Per-comparison noise: hash-based deterministic noise per color×trial
                    let noise_seed = seed.wrapping_add(5000 + trial as u64 * 7 + ci as u64 * 31);
                    let noise_val = ((noise_seed.wrapping_mul(0x9E3779B97F4A7C15) >> 33) as f32
                        / (1u64 << 31) as f32)
                        - 0.5;
                    (raw + noise_val * enc_noise * 0.15) as f64
                })
                .collect();

            // Softmax response selection with temperature
            let max_sim = sims.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_sims: Vec<f64> = sims
                .iter()
                .map(|s| ((s - max_sim) / temperature).exp())
                .collect();
            let exp_sum: f64 = exp_sims.iter().sum();

            // Sample from softmax distribution
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

            // Attention lapse: incongruent trials are more vulnerable because
            // response conflict demands sustained attention (Botvinick et al., 2001).
            let unique_trial = trial_idx * (trials_per_condition * 3) + trial;
            let conflict_boost = match condition {
                Condition::Incongruent => config.lapse_rate * 0.6,
                Condition::Neutral => config.lapse_rate * 0.2,
                Condition::Congruent => 0.0,
            };
            let effective_lapse = config.lapse_rate + conflict_boost;
            let lapse_seed = config.trial_seed("stroop", "lapse", unique_trial);
            response_idx = if (lapse_seed % 10000) as f64 / 10000.0 < effective_lapse {
                let h = config.trial_seed("stroop", "lapse_choice", unique_trial);
                h as usize % 4
            } else {
                response_idx
            };

            // RT proxy: deliberation ticks based on decision margin
            let decision_margin = (sims[ink_idx] - max_sim + sims[ink_idx]).abs()
                / (sims.iter().sum::<f64>() + 1e-10);
            let rt_ticks = 8.0 + (1.0 - decision_margin) * 12.0;

            let correct = response_idx == ink_idx;
            let confidence = exp_sims[response_idx] / exp_sum;
            let similarity = sims[ink_idx];

            match condition {
                Condition::Congruent => {
                    if correct {
                        congruent_correct += 1;
                    }
                    cong_rts.push(rt_ticks);
                }
                Condition::Incongruent => {
                    if correct {
                        incongruent_correct += 1;
                    }
                    incong_rts.push(rt_ticks);
                }
                Condition::Neutral => {
                    if correct {
                        neutral_correct += 1;
                    }
                    neut_rts.push(rt_ticks);
                }
            }

            if collect_trace {
                let cond_name = match condition {
                    Condition::Congruent => "congruent",
                    Condition::Incongruent => "incongruent",
                    Condition::Neutral => "neutral",
                };
                trace.push(TrialOutcome {
                    trial_idx: *global_trial_idx,
                    condition: cond_name.to_string(),
                    correct,
                    rt_ticks,
                    similarity,
                    confidence,
                    response_idx,
                    extra: BTreeMap::new(),
                });
                *global_trial_idx += 1;
            }
        }

        let cong_acc = congruent_correct as f64 / trials_per_condition as f64;
        let incong_acc = incongruent_correct as f64 / trials_per_condition as f64;
        let neut_acc = neutral_correct as f64 / trials_per_condition as f64;

        TrialResult {
            congruent_accuracy: cong_acc,
            incongruent_accuracy: incong_acc,
            neutral_accuracy: neut_acc,
            stroop_effect: cong_acc - incong_acc,
            congruent_rt: cong_rts,
            incongruent_rt: incong_rts,
            neutral_rt: neut_rts,
        }
    }
}

struct TrialResult {
    congruent_accuracy: f64,
    incongruent_accuracy: f64,
    neutral_accuracy: f64,
    stroop_effect: f64,
    congruent_rt: Vec<f64>,
    incongruent_rt: Vec<f64>,
    neutral_rt: Vec<f64>,
}

impl PsychBenchmark for StroopBenchmark {
    fn name(&self) -> &str {
        "Executive::Stroop"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Stroop Color-Word",
            citation: "Stroop (1935)",
            year: 1935,
            doi: Some("10.1037/h0054651"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let diff_model = difficulty_model_for(self.name());

        let mut cong = Vec::new();
        let mut incong = Vec::new();
        let mut neutral = Vec::new();
        let mut effect = Vec::new();
        let mut all_cong_rt = Vec::new();
        let mut all_incong_rt = Vec::new();
        let mut all_neut_rt = Vec::new();
        let mut trace = Vec::new();
        let mut global_trial_idx = 0usize;

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial_with_difficulty(
                config,
                trial,
                &diff_model,
                &mut trace,
                &mut global_trial_idx,
            );
            cong.push(r.congruent_accuracy);
            incong.push(r.incongruent_accuracy);
            neutral.push(r.neutral_accuracy);
            effect.push(r.stroop_effect);
            all_cong_rt.extend_from_slice(&r.congruent_rt);
            all_incong_rt.extend_from_slice(&r.incongruent_rt);
            all_neut_rt.extend_from_slice(&r.neutral_rt);
        }

        result.insert("congruent_accuracy", MetricValue::from_samples(&cong));
        result.insert("incongruent_accuracy", MetricValue::from_samples(&incong));
        result.insert("neutral_accuracy", MetricValue::from_samples(&neutral));
        result.insert("stroop_effect", MetricValue::from_samples(&effect));

        // RT metrics (tick-based)
        result.insert(
            "congruent::rt_ticks",
            MetricValue::from_samples(&all_cong_rt),
        );
        result.insert(
            "incongruent::rt_ticks",
            MetricValue::from_samples(&all_incong_rt),
        );
        result.insert("neutral::rt_ticks", MetricValue::from_samples(&all_neut_rt));

        if config.trial_trace {
            result.trial_trace = trace;
        }

        result.conditions = 3;
        result.trials_per_condition = config.trials_per_condition;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stroop_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = StroopBenchmark.run(&config);
        assert!(result.metrics.contains_key("congruent_accuracy"));
        assert!(result.metrics.contains_key("incongruent_accuracy"));
        assert!(result.metrics.contains_key("neutral_accuracy"));
        assert!(result.metrics.contains_key("stroop_effect"));
    }

    #[test]
    fn test_stroop_finite_metrics() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = StroopBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_stroop_effect_direction() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 20,
            ..Default::default()
        };
        let result = StroopBenchmark.run(&config);
        let cong = result.metrics["congruent_accuracy"].mean;
        let incong = result.metrics["incongruent_accuracy"].mean;
        // Congruent should be easier than incongruent
        assert!(
            cong >= incong - 0.05,
            "congruent ({:.3}) should be >= incongruent ({:.3})",
            cong,
            incong
        );
    }
}
