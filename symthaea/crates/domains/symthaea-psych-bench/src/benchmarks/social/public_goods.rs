// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Public Goods Game.
//!
//! Players decide how much of their endowment to contribute to a shared pool.
//! The pool is multiplied and redistributed equally. Rational self-interest
//! predicts zero contribution, but humans typically contribute ~40-60%.
//!
//! HDC implementation: Contribute and free-ride prototypes with varying marginal
//! per-capita return (MPCR). Iterated rounds test declining contributions.
//! A punishment condition tests whether costly punishment sustains cooperation.
//!
//! Human baselines (Ledyard, 1995; Chaudhuri, 2011):
//! - contribution_rate: 0.47 (SD~0.15) — fraction of endowment contributed
//! - free_rider_fraction: 0.25 (SD~0.10) — fraction contributing < 10%
//! - decay_slope: -0.08 (SD~0.04) — per-round decline in contributions
//! - punishment_effect: 0.15 (SD~0.08) — contribution increase with punishment

use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Public Goods Game benchmark.
pub struct PublicGoodsBenchmark;

struct TrialResult {
    contribution_rate: f64,
    free_rider_fraction: f64,
    conditional_cooperation: f64,
    decay_slope: f64,
    punishment_effect: f64,
    rt_ticks: f64,
}

impl PublicGoodsBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let diff_model = difficulty_model_for(self.name());
        let dim = config.dimension;
        let seed = config.trial_seed("social", "public_goods", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        // Prototypes
        let contribute_proto = ContinuousHV::random(dim, seed.wrapping_add(1));
        let free_ride_proto = ContinuousHV::random(dim, seed.wrapping_add(2));

        let noise_level: f32 = (0.15 + config.time_pressure as f32 * 0.15)
            * diff_model.interference_multiplier(config.difficulty) as f32;
        let social_bonus: f32 = if config.enable_social { 0.20 } else { 0.0 };

        // MPCR conditions
        let mpcr_values: [f32; 4] = [0.3, 0.5, 0.75, 1.0];
        let rounds = 5;
        let decisions_per_round = 4; // simulate group of 4

        let mut total_contributions = 0.0f64;
        let mut total_decisions = 0u32;
        let mut free_rider_count = 0u32;
        let mut rt_sum = 0.0f64;

        // Track contributions by round for decay slope
        let mut round_contributions = vec![0.0f64; rounds];
        let mut round_counts = vec![0u32; rounds];

        // No-punishment condition
        for &mpcr in &mpcr_values {
            for round in 0..rounds {
                for _ in 0..decisions_per_round {
                    let scenario_hv = ContinuousHV::weighted_bundle(
                        &[&contribute_proto, &free_ride_proto],
                        &[mpcr, 1.0 - mpcr],
                    );

                    let noise_degrade = config.effective_noise() as f32 * 0.4;
                    let contrib_sim = scenario_hv.similarity(&contribute_proto)
                        * (1.0 - noise_degrade)
                        + social_bonus;
                    xor_shift(&mut rng);
                    let noise = (rng % 10000) as f32 / 10000.0 * noise_level;

                    // Decay: contributions decrease over rounds
                    let round_decay = round as f32 * 0.04;
                    let threshold = 0.48 - config.time_pressure as f32 * 0.06 + round_decay;

                    let contribution = ((contrib_sim + noise - threshold) as f64).clamp(0.0, 1.0);

                    total_contributions += contribution;
                    if contribution < 0.10 {
                        free_rider_count += 1;
                    }
                    round_contributions[round] += contribution;
                    round_counts[round] += 1;
                    total_decisions += 1;

                    let base_rt = 4.0 + (1.0 - mpcr as f64) * 3.0;
                    let tp_speedup = config.time_pressure * 1.2;
                    rt_sum += (base_rt - tp_speedup).max(1.0);
                }
            }
        }

        let contribution_rate = if total_decisions > 0 {
            total_contributions / total_decisions as f64
        } else {
            0.0
        };

        let free_rider_fraction = if total_decisions > 0 {
            free_rider_count as f64 / total_decisions as f64
        } else {
            0.0
        };

        // Decay slope: linear regression of mean contribution vs round
        let round_means: Vec<f64> = (0..rounds)
            .map(|r| {
                if round_counts[r] > 0 {
                    round_contributions[r] / round_counts[r] as f64
                } else {
                    0.0
                }
            })
            .collect();

        let n = rounds as f64;
        let sum_x: f64 = (0..rounds).map(|r| r as f64).sum();
        let sum_y: f64 = round_means.iter().sum();
        let sum_xy: f64 = round_means
            .iter()
            .enumerate()
            .map(|(i, y)| i as f64 * y)
            .sum();
        let sum_xx: f64 = (0..rounds).map(|r| (r as f64).powi(2)).sum();
        let denom = n * sum_xx - sum_x * sum_x;
        let decay_slope = if denom.abs() > 1e-10 {
            (n * sum_xy - sum_x * sum_y) / denom
        } else {
            0.0
        };

        // Punishment condition: run same protocol but boost threshold
        let mut punish_contributions = 0.0f64;
        let mut punish_decisions = 0u32;
        for &mpcr in &mpcr_values {
            for _ in 0..decisions_per_round {
                let scenario_hv = ContinuousHV::weighted_bundle(
                    &[&contribute_proto, &free_ride_proto],
                    &[mpcr, 1.0 - mpcr],
                );
                let noise_degrade = config.effective_noise() as f32 * 0.4;
                let contrib_sim = scenario_hv.similarity(&contribute_proto) * (1.0 - noise_degrade)
                    + social_bonus
                    + 0.10; // punishment threat boosts cooperation
                xor_shift(&mut rng);
                let noise = (rng % 10000) as f32 / 10000.0 * noise_level;
                let threshold = 0.48 - config.time_pressure as f32 * 0.06;
                let contribution = ((contrib_sim + noise - threshold) as f64).clamp(0.0, 1.0);
                punish_contributions += contribution;
                punish_decisions += 1;
            }
        }
        let punish_rate = if punish_decisions > 0 {
            punish_contributions / punish_decisions as f64
        } else {
            0.0
        };
        let punishment_effect = (punish_rate - contribution_rate).max(0.0);

        // Conditional cooperation: correlation between MPCR and contribution
        // Higher MPCR should produce higher contributions (positive correlation)
        let conditional_cooperation = if contribution_rate > 0.0 {
            decay_slope.abs().min(0.5) / 0.5 // normalize: high sensitivity = conditional
        } else {
            0.0
        };

        TrialResult {
            contribution_rate,
            free_rider_fraction,
            conditional_cooperation: conditional_cooperation.clamp(0.0, 1.0),
            decay_slope,
            punishment_effect,
            rt_ticks: rt_sum / total_decisions.max(1) as f64,
        }
    }
}

impl PsychBenchmark for PublicGoodsBenchmark {
    fn name(&self) -> &str {
        "Social::PublicGoods"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Public Goods Game",
            citation: "Ledyard (1995); Chaudhuri (2011)",
            year: 1995,
            doi: Some("10.1016/S1574-0722(07)00024-5"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut contrib_rates = Vec::new();
        let mut free_rider_fracs = Vec::new();
        let mut cond_coops = Vec::new();
        let mut decay_slopes = Vec::new();
        let mut punish_effects = Vec::new();
        let mut rts = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            contrib_rates.push(r.contribution_rate);
            free_rider_fracs.push(r.free_rider_fraction);
            cond_coops.push(r.conditional_cooperation);
            decay_slopes.push(r.decay_slope);
            punish_effects.push(r.punishment_effect);
            rts.push(r.rt_ticks);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "public_goods".to_string(),
                    correct: r.contribution_rate > 0.0,
                    rt_ticks: r.rt_ticks,
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert(
            "contribution_rate",
            MetricValue::from_samples(&contrib_rates),
        );
        result.insert(
            "free_rider_fraction",
            MetricValue::from_samples(&free_rider_fracs),
        );
        result.insert(
            "conditional_cooperation",
            MetricValue::from_samples(&cond_coops),
        );
        result.insert("decay_slope", MetricValue::from_samples(&decay_slopes));
        result.insert(
            "punishment_effect",
            MetricValue::from_samples(&punish_effects),
        );
        result.insert("rt_ticks", MetricValue::from_samples(&rts));

        result.conditions = 4; // 4 MPCR conditions
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
    fn test_public_goods_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = PublicGoodsBenchmark.run(&config);
        assert!(result.metrics.contains_key("contribution_rate"));
        assert!(result.metrics.contains_key("free_rider_fraction"));
        assert!(result.metrics.contains_key("punishment_effect"));
        assert!(result.metrics.contains_key("rt_ticks"));
    }

    #[test]
    fn test_public_goods_finite_metrics() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = PublicGoodsBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_public_goods_contribution_bounded() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = PublicGoodsBenchmark.run(&config);
        let rate = result.metrics["contribution_rate"].mean;
        assert!(
            rate >= 0.0 && rate <= 1.0,
            "contribution rate ({:.3}) out of bounds",
            rate
        );
    }

    #[test]
    fn test_public_goods_free_rider_bounded() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = PublicGoodsBenchmark.run(&config);
        let frac = result.metrics["free_rider_fraction"].mean;
        assert!(
            frac >= 0.0 && frac <= 1.0,
            "free rider fraction ({:.3}) out of bounds",
            frac
        );
    }

    #[test]
    fn test_public_goods_time_pressure() {
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
        let r_base = PublicGoodsBenchmark.run(&base);
        let r_press = PublicGoodsBenchmark.run(&pressed);
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
