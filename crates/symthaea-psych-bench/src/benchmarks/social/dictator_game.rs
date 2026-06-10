// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Dictator Game.
//!
//! One player (the dictator) decides how to split a resource with a passive
//! recipient who has no power to reject. Pure self-interest predicts zero
//! giving, but humans typically give ~28% on average.
//!
//! HDC implementation: Generous and selfish prototypes. Context conditions
//! (anonymous, observed, earned, charity) modulate the fairness norm.
//! This is the purest test of intrinsic altruism without strategic incentives.
//!
//! Human baselines (Engel, 2011 meta-analysis of 616 treatments):
//! - mean_offer: 0.28 (SD~0.13) — mean fraction given
//! - positive_offer_rate: 0.64 (SD~0.12) — fraction offering > 0
//! - fairness_offers: 0.17 (SD~0.08) — fraction offering exactly 50%

use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Dictator Game benchmark.
pub struct DictatorGameBenchmark;

struct TrialResult {
    mean_offer: f64,
    positive_offer_rate: f64,
    fairness_offers: f64,
    generosity_index: f64,
    context_effect: f64,
    rt_ticks: f64,
}

impl DictatorGameBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let diff_model = difficulty_model_for(self.name());
        let dim = config.dimension;
        let seed = config.trial_seed("social", "dictator_game", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        // Prototypes
        let generous_proto = ContinuousHV::random(dim, seed.wrapping_add(1));
        let selfish_proto = ContinuousHV::random(dim, seed.wrapping_add(2));

        let noise_level: f32 = (0.12 + config.time_pressure as f32 * 0.12)
            * diff_model.interference_multiplier(config.difficulty) as f32;

        // Social cognition: fairness norms and empathy increase giving
        let social_bonus: f32 = if config.enable_social { 0.18 } else { 0.0 };

        // Context conditions with different fairness norm strengths
        struct ContextCondition {
            name: &'static str,
            generous_weight: f32,
            extra_bonus: f32,
        }
        let conditions = [
            ContextCondition {
                name: "anonymous",
                generous_weight: 0.50,
                extra_bonus: 0.0,
            },
            ContextCondition {
                name: "observed",
                generous_weight: 0.50,
                extra_bonus: 0.12, // social accountability
            },
            ContextCondition {
                name: "earned",
                generous_weight: 0.35, // entitlement reduces giving
                extra_bonus: -0.08,
            },
            ContextCondition {
                name: "charity",
                generous_weight: 0.60, // warm glow boosts giving
                extra_bonus: 0.10,
            },
        ];

        let trials_per_condition = 10;
        let mut offers = Vec::new();
        let mut anon_offers = Vec::new();
        let mut observed_offers = Vec::new();
        let mut rt_sum = 0.0f64;
        let mut total_decisions = 0u32;

        for condition in &conditions {
            for _ in 0..trials_per_condition {
                let scenario_hv = ContinuousHV::weighted_bundle(
                    &[&generous_proto, &selfish_proto],
                    &[condition.generous_weight, 1.0 - condition.generous_weight],
                );

                let noise_degrade = config.effective_noise() as f32 * 0.3;
                let generous_sim = scenario_hv.similarity(&generous_proto) * (1.0 - noise_degrade)
                    + social_bonus
                    + condition.extra_bonus;
                let selfish_sim = scenario_hv.similarity(&selfish_proto) * (1.0 - noise_degrade);

                xor_shift(&mut rng);
                let noise = (rng % 10000) as f32 / 10000.0 * noise_level - noise_level / 2.0;

                // Offer = proportion given, mapped from similarity difference
                let raw_offer = ((generous_sim - selfish_sim + noise + 0.5) / 1.0) as f64;
                // Quantize to 0%, 10%, ..., 50%
                let offer = (raw_offer * 5.0).round().clamp(0.0, 5.0) / 10.0;

                offers.push(offer);
                if condition.name == "anonymous" {
                    anon_offers.push(offer);
                }
                if condition.name == "observed" {
                    observed_offers.push(offer);
                }

                // RT: lower for extreme decisions (selfish or generous), higher for conflicted
                let conflict = (offer - 0.25).abs();
                let base_rt = 3.0 + (0.25 - conflict).max(0.0) as f64 * 8.0;
                let tp_speedup = config.time_pressure * 1.0;
                rt_sum += (base_rt - tp_speedup).max(1.0);
                total_decisions += 1;
            }
        }

        let mean_offer = if !offers.is_empty() {
            offers.iter().sum::<f64>() / offers.len() as f64
        } else {
            0.0
        };

        let positive_offer_rate = if !offers.is_empty() {
            offers.iter().filter(|&&o| o > 0.0).count() as f64 / offers.len() as f64
        } else {
            0.0
        };

        let fairness_offers = if !offers.is_empty() {
            offers.iter().filter(|&&o| (o - 0.5).abs() < 0.01).count() as f64 / offers.len() as f64
        } else {
            0.0
        };

        // Generosity index: mean offer given that offer > 0
        let positive_offers: Vec<f64> = offers.iter().copied().filter(|&o| o > 0.0).collect();
        let generosity_index = if !positive_offers.is_empty() {
            positive_offers.iter().sum::<f64>() / positive_offers.len() as f64
        } else {
            0.0
        };

        // Context effect: observed - anonymous
        let mean_anon = if !anon_offers.is_empty() {
            anon_offers.iter().sum::<f64>() / anon_offers.len() as f64
        } else {
            0.0
        };
        let mean_observed = if !observed_offers.is_empty() {
            observed_offers.iter().sum::<f64>() / observed_offers.len() as f64
        } else {
            0.0
        };
        let context_effect = mean_observed - mean_anon;

        TrialResult {
            mean_offer,
            positive_offer_rate,
            fairness_offers,
            generosity_index,
            context_effect,
            rt_ticks: rt_sum / total_decisions.max(1) as f64,
        }
    }
}

impl PsychBenchmark for DictatorGameBenchmark {
    fn name(&self) -> &str {
        "Social::DictatorGame"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Dictator Game",
            citation: "Engel (2011)",
            year: 2011,
            doi: Some("10.1027/1618-3169/a000087"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut mean_offers = Vec::new();
        let mut positive_rates = Vec::new();
        let mut fairness_rates = Vec::new();
        let mut generosity_indices = Vec::new();
        let mut context_effects = Vec::new();
        let mut rts = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            mean_offers.push(r.mean_offer);
            positive_rates.push(r.positive_offer_rate);
            fairness_rates.push(r.fairness_offers);
            generosity_indices.push(r.generosity_index);
            context_effects.push(r.context_effect);
            rts.push(r.rt_ticks);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "dictator_game".to_string(),
                    correct: r.mean_offer > 0.0,
                    rt_ticks: r.rt_ticks,
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("mean_offer", MetricValue::from_samples(&mean_offers));
        result.insert(
            "positive_offer_rate",
            MetricValue::from_samples(&positive_rates),
        );
        result.insert(
            "fairness_offers",
            MetricValue::from_samples(&fairness_rates),
        );
        result.insert(
            "generosity_index",
            MetricValue::from_samples(&generosity_indices),
        );
        result.insert(
            "context_effect",
            MetricValue::from_samples(&context_effects),
        );
        result.insert("rt_ticks", MetricValue::from_samples(&rts));

        result.conditions = 4; // 4 context conditions
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
    fn test_dictator_game_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = DictatorGameBenchmark.run(&config);
        assert!(result.metrics.contains_key("mean_offer"));
        assert!(result.metrics.contains_key("positive_offer_rate"));
        assert!(result.metrics.contains_key("fairness_offers"));
        assert!(result.metrics.contains_key("generosity_index"));
        assert!(result.metrics.contains_key("rt_ticks"));
    }

    #[test]
    fn test_dictator_game_finite_metrics() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = DictatorGameBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_dictator_game_offer_bounded() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = DictatorGameBenchmark.run(&config);
        let offer = result.metrics["mean_offer"].mean;
        assert!(
            offer >= 0.0 && offer <= 0.5,
            "mean offer ({:.3}) out of bounds [0, 0.5]",
            offer
        );
    }

    #[test]
    fn test_dictator_game_positive_rate_bounded() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = DictatorGameBenchmark.run(&config);
        let rate = result.metrics["positive_offer_rate"].mean;
        assert!(
            rate >= 0.0 && rate <= 1.0,
            "positive offer rate ({:.3}) out of bounds",
            rate
        );
    }

    #[test]
    fn test_dictator_game_time_pressure() {
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
        let r_base = DictatorGameBenchmark.run(&base);
        let r_press = DictatorGameBenchmark.run(&pressed);
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
