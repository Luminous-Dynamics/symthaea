// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Eriksen Flanker Task.
//!
//! Tests selective attention and response inhibition. The system must identify
//! the direction of a central target arrow flanked by congruent or incongruent
//! distractors.
//!
//! HDC implementation: models the flanker effect as attention leakage.
//! The target and flanker directions are combined as a weighted sum, where
//! flanker weight represents imperfect spatial attention filtering.
//! Response is selected via softmax over similarities to direction candidates.
//!
//! Conditions:
//! - Congruent: >>>>> (flankers reinforce target)
//! - Incongruent: >><>> (flankers compete with target)
//! - Neutral: --<-- (flankers carry no directional information)
//!
//! Human baselines (Eriksen & Eriksen, 1974; Ridderinkhof et al., 2021):
//! - congruent_accuracy: 0.97
//! - incongruent_accuracy: 0.90
//! - flanker_effect: 0.07 (congruent - incongruent accuracy)

use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Eriksen Flanker Task benchmark.
pub struct FlankerBenchmark;

#[derive(Clone, Copy)]
enum Condition {
    Congruent,
    Incongruent,
    Neutral,
}

impl FlankerBenchmark {
    fn run_trial(
        &self,
        config: &BenchmarkConfig,
        trial_idx: usize,
        trace: &mut Vec<TrialOutcome>,
        global_trial_idx: &mut usize,
    ) -> TrialResult {
        let diff_model = difficulty_model_for(self.name());
        let dim = config.dimension;
        let seed = config.trial_seed("executive", "flanker", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        // Direction HVs: left, right
        let dir_left = ContinuousHV::random(dim, seed.wrapping_add(100));
        let dir_right = ContinuousHV::random(dim, seed.wrapping_add(101));
        let dir_neutral = ContinuousHV::random(dim, seed.wrapping_add(102));
        let directions = [&dir_left, &dir_right];

        // Attention leakage: how much flanker direction bleeds into the
        // response activation. In humans, spatial filtering is imperfect,
        // especially at close spacing (Eriksen & Eriksen, 1974).
        let attention_leak: f32 = 0.35;

        // Decision temperature: controls stochasticity of response selection.
        // Time pressure: base 0.25 matches ~10% flanker interference (Eriksen & Eriksen, 1974);
        // +0.15/unit reflects boundary collapse under speed emphasis (Ratcliff & McKoon, 2008 DDM).
        let temperature: f64 = (0.25 + config.time_pressure * 0.15)
            * diff_model.temperature_multiplier(config.difficulty);

        let trials_per_condition = 40;
        let mut congruent_correct = 0u32;
        let mut incongruent_correct = 0u32;
        let mut neutral_correct = 0u32;
        let mut cong_rts = Vec::new();
        let mut incong_rts = Vec::new();
        let mut neut_rts = Vec::new();

        for trial in 0..(trials_per_condition * 3) {
            let condition = match trial % 3 {
                0 => Condition::Congruent,
                1 => Condition::Incongruent,
                _ => Condition::Neutral,
            };

            // Random target direction (0=left, 1=right)
            xor_shift(&mut rng);
            let target_idx = (rng % 2) as usize;
            let target_dir = directions[target_idx];

            // Build combined activation: target + attention_leak * flanker
            let combined = match condition {
                Condition::Congruent => {
                    // Flankers same direction: reinforcement
                    target_dir.scale(1.0 + attention_leak)
                }
                Condition::Incongruent => {
                    // Flankers opposite direction: competition
                    let flanker_dir = directions[1 - target_idx];
                    let flanker_act = flanker_dir.scale(attention_leak);
                    ContinuousHV::bundle(&[target_dir, &flanker_act])
                }
                Condition::Neutral => {
                    // Flankers non-directional: mild noise
                    let noise_act = dir_neutral.scale(attention_leak * 0.3);
                    ContinuousHV::bundle(&[target_dir, &noise_act])
                }
            };

            // Compute similarity to each direction candidate
            // Encoding noise degrades direction discrimination
            let noise_degrade = config.effective_noise() as f32 * 0.4;
            let sim_left = (combined.similarity(&dir_left) * (1.0 - noise_degrade)) as f64;
            let sim_right = (combined.similarity(&dir_right) * (1.0 - noise_degrade)) as f64;
            let sims = [sim_left, sim_right];

            // Softmax response selection with temperature
            let max_sim = sim_left.max(sim_right);
            let exp_sims: Vec<f64> = sims
                .iter()
                .map(|s| ((s - max_sim) / temperature).exp())
                .collect();
            let exp_sum: f64 = exp_sims.iter().sum();

            xor_shift(&mut rng);
            let r = (rng % 10000) as f64 / 10000.0;
            let response_idx = if r < exp_sims[0] / exp_sum { 0 } else { 1 };

            // RT proxy: deliberation ticks based on decision margin
            let margin = (sim_left - sim_right).abs();
            let rt_ticks = 5.0 + (1.0 - margin) * 8.0;

            let correct = response_idx == target_idx;
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

            if config.trial_trace {
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
                    similarity: 0.0,
                    confidence: 0.0,
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
            flanker_effect: cong_acc - incong_acc,
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
    flanker_effect: f64,
    congruent_rt: Vec<f64>,
    incongruent_rt: Vec<f64>,
    neutral_rt: Vec<f64>,
}

impl PsychBenchmark for FlankerBenchmark {
    fn name(&self) -> &str {
        "Executive::Flanker"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Eriksen Flanker Task",
            citation: "Eriksen & Eriksen (1974)",
            year: 1974,
            doi: Some("10.3758/BF03203267"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

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
            let r = self.run_trial(config, trial, &mut trace, &mut global_trial_idx);
            cong.push(r.congruent_accuracy);
            incong.push(r.incongruent_accuracy);
            neutral.push(r.neutral_accuracy);
            effect.push(r.flanker_effect);
            all_cong_rt.extend_from_slice(&r.congruent_rt);
            all_incong_rt.extend_from_slice(&r.incongruent_rt);
            all_neut_rt.extend_from_slice(&r.neutral_rt);
        }

        result.insert("congruent_accuracy", MetricValue::from_samples(&cong));
        result.insert("incongruent_accuracy", MetricValue::from_samples(&incong));
        result.insert("neutral_accuracy", MetricValue::from_samples(&neutral));
        result.insert("flanker_effect", MetricValue::from_samples(&effect));

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
    fn test_flanker_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = FlankerBenchmark.run(&config);
        assert!(result.metrics.contains_key("congruent_accuracy"));
        assert!(result.metrics.contains_key("incongruent_accuracy"));
        assert!(result.metrics.contains_key("neutral_accuracy"));
        assert!(result.metrics.contains_key("flanker_effect"));
    }

    #[test]
    fn test_flanker_finite_metrics() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = FlankerBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_flanker_effect_direction() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 20,
            ..Default::default()
        };
        let result = FlankerBenchmark.run(&config);
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
