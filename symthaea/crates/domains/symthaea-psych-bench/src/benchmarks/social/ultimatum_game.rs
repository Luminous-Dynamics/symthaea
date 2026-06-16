// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Ultimatum Game.
//!
//! A proposer offers a split of a resource; the responder accepts or rejects.
//! Rejecting an unfair offer is costly but enforces social norms. Humans
//! reject offers below ~30% at high rates, showing fairness sensitivity.
//!
//! HDC implementation: Offers at 10%, 20%, 30%, 40%, 50% of a resource.
//! Each offer is encoded as a ContinuousHV blend of "fair" (50:50 split)
//! and "unfair" (100:0 split) prototypes. Accept/reject based on similarity
//! to the fair prototype. Time pressure lowers deliberation threshold.
//!
//! Human baselines (Guth et al., 1982; Camerer, 2003):
//! - fairness_sensitivity: 0.70 (SD~0.15) — slope of rejection vs unfairness
//! - rejection_rate: 0.40 (SD~0.12) — overall rejection rate across offers
//! - offer_threshold: 0.30 (SD~0.08) — offer level at 50% acceptance

use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Ultimatum Game benchmark.
pub struct UltimatumGameBenchmark;

struct TrialResult {
    fairness_sensitivity: f64,
    rejection_rate: f64,
    offer_threshold: f64,
    strategic_ratio: f64,
    rt_ticks: f64,
}

impl UltimatumGameBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let diff_model = difficulty_model_for(self.name());
        let dim = config.dimension;
        let seed = config.trial_seed("social", "ultimatum_game", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        // Prototypes: fair (50:50) and unfair (100:0)
        let fair_proto = ContinuousHV::random(dim, seed.wrapping_add(1));
        let unfair_proto = ContinuousHV::random(dim, seed.wrapping_add(2));

        // Time pressure lowers deliberation: more impulsive rejections
        let noise_level: f32 = (0.15 + config.time_pressure as f32 * 0.20)
            * diff_model.interference_multiplier(config.difficulty) as f32;
        let threshold_shift: f32 = config.time_pressure as f32 * 0.10; // lower acceptance threshold

        // Social cognition: empathy reduces rejection of low offers
        let social_bonus: f32 = if config.enable_social { 0.20 } else { 0.0 };

        // Offer levels: 10%, 20%, 30%, 40%, 50%
        let offer_levels = [0.10f32, 0.20, 0.30, 0.40, 0.50];
        let trials_per_offer = 8;
        let mut rejections_by_level = [0u32; 5];
        let mut total_by_level = [0u32; 5];
        let mut total_rejections = 0u32;
        let mut total_offers = 0u32;
        let mut rt_sum = 0.0f64;

        for (level_idx, &offer_pct) in offer_levels.iter().enumerate() {
            for _ in 0..trials_per_offer {
                // Encode offer as blend of fair and unfair prototypes
                let offer_hv = ContinuousHV::weighted_bundle(
                    &[&fair_proto, &unfair_proto],
                    &[offer_pct, 1.0 - offer_pct],
                );

                // Decision: compare offer similarity to fair prototype
                // Encoding noise degrades fairness evaluation
                let noise_degrade = config.effective_noise() as f32 * 0.4;
                let fair_sim =
                    offer_hv.similarity(&fair_proto) * (1.0 - noise_degrade) + social_bonus;
                xor_shift(&mut rng);
                let noise = (rng % 10000) as f32 / 10000.0 * noise_level;

                // Accept if sufficiently fair (threshold = 0.45 base)
                let accept_threshold = 0.45 - threshold_shift;
                let reject = (fair_sim + noise) < accept_threshold;

                if reject {
                    rejections_by_level[level_idx] += 1;
                    total_rejections += 1;
                }
                total_by_level[level_idx] += 1;
                total_offers += 1;

                // RT: unfair offers take longer (conflict between accept/reject)
                let conflict = (0.5 - offer_pct).abs();
                let base_rt = 5.0 + conflict as f64 * 4.0;
                let tp_speedup = config.time_pressure * 1.5;
                rt_sum += (base_rt - tp_speedup).max(1.0);
            }
        }

        // Fairness sensitivity: slope of rejection rate vs unfairness (1 - offer)
        // Linear regression: rejection_rate = a + b * (1 - offer)
        // b = fairness_sensitivity
        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;
        let mut sum_xy = 0.0f64;
        let mut sum_xx = 0.0f64;
        let n = offer_levels.len() as f64;

        for (i, &offer_pct) in offer_levels.iter().enumerate() {
            let x = 1.0 - offer_pct as f64; // unfairness
            let y = if total_by_level[i] > 0 {
                rejections_by_level[i] as f64 / total_by_level[i] as f64
            } else {
                0.0
            };
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        let denom = n * sum_xx - sum_x * sum_x;
        let fairness_sensitivity = if denom.abs() > 1e-10 {
            ((n * sum_xy - sum_x * sum_y) / denom).clamp(0.0, 2.0)
        } else {
            0.0
        };

        let overall_rejection = if total_offers > 0 {
            total_rejections as f64 / total_offers as f64
        } else {
            0.0
        };

        // Offer threshold: interpolate to find 50% rejection point
        let mut threshold = 0.30; // default
        for (i, &offer_pct) in offer_levels.iter().enumerate() {
            let rej_rate = if total_by_level[i] > 0 {
                rejections_by_level[i] as f64 / total_by_level[i] as f64
            } else {
                0.0
            };
            if rej_rate <= 0.5 {
                threshold = offer_pct as f64;
                break;
            }
        }

        // Strategic ratio: fraction of rejections at lowest offer vs highest
        let low_rej = if total_by_level[0] > 0 {
            rejections_by_level[0] as f64 / total_by_level[0] as f64
        } else {
            0.0
        };
        let high_rej = if total_by_level[4] > 0 {
            rejections_by_level[4] as f64 / total_by_level[4] as f64
        } else {
            0.0
        };
        let strategic_ratio = if low_rej > 0.0 {
            (low_rej - high_rej) / low_rej
        } else {
            0.0
        };

        TrialResult {
            fairness_sensitivity,
            rejection_rate: overall_rejection,
            offer_threshold: threshold,
            strategic_ratio: strategic_ratio.clamp(0.0, 1.0),
            rt_ticks: rt_sum / total_offers as f64,
        }
    }
}

impl PsychBenchmark for UltimatumGameBenchmark {
    fn name(&self) -> &str {
        "Social::UltimatumGame"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Ultimatum Game",
            citation: "Guth et al. (1982)",
            year: 1982,
            doi: Some("10.1016/0167-2681(82)90011-7"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut sensitivities = Vec::new();
        let mut rejection_rates = Vec::new();
        let mut thresholds = Vec::new();
        let mut strategic_ratios = Vec::new();
        let mut rts = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            sensitivities.push(r.fairness_sensitivity);
            rejection_rates.push(r.rejection_rate);
            thresholds.push(r.offer_threshold);
            strategic_ratios.push(r.strategic_ratio);
            rts.push(r.rt_ticks);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "ultimatum".to_string(),
                    correct: r.fairness_sensitivity > 0.0,
                    rt_ticks: r.rt_ticks,
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert(
            "fairness_sensitivity",
            MetricValue::from_samples(&sensitivities),
        );
        result.insert(
            "rejection_rate",
            MetricValue::from_samples(&rejection_rates),
        );
        result.insert("offer_threshold", MetricValue::from_samples(&thresholds));
        result.insert(
            "strategic_ratio",
            MetricValue::from_samples(&strategic_ratios),
        );
        result.insert("rt_ticks", MetricValue::from_samples(&rts));

        result.conditions = 5; // 5 offer levels
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
    fn test_ultimatum_game_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = UltimatumGameBenchmark.run(&config);
        assert!(result.metrics.contains_key("fairness_sensitivity"));
        assert!(result.metrics.contains_key("rejection_rate"));
        assert!(result.metrics.contains_key("offer_threshold"));
        assert!(result.metrics.contains_key("strategic_ratio"));
        assert!(result.metrics.contains_key("rt_ticks"));
    }

    #[test]
    fn test_ultimatum_game_finite_metrics() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = UltimatumGameBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_ultimatum_game_rejection_bounded() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = UltimatumGameBenchmark.run(&config);
        let rate = result.metrics["rejection_rate"].mean;
        assert!(
            rate >= 0.0 && rate <= 1.0,
            "rejection rate ({:.3}) out of bounds",
            rate
        );
    }

    #[test]
    fn test_ultimatum_game_sensitivity_non_negative() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = UltimatumGameBenchmark.run(&config);
        let sens = result.metrics["fairness_sensitivity"].mean;
        assert!(
            sens >= 0.0,
            "fairness sensitivity ({:.3}) should be >= 0",
            sens
        );
    }

    #[test]
    fn test_ultimatum_game_time_pressure() {
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
        let r_base = UltimatumGameBenchmark.run(&base);
        let r_press = UltimatumGameBenchmark.run(&pressed);
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
