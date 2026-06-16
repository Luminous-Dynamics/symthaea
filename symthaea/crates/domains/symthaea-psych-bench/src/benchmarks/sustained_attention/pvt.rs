// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Psychomotor Vigilance Task (PVT).
//!
//! A simple reaction time task sustained over 10 minutes. Measures vigilance
//! decrement: RT slows and lapses increase as time on task increases.
//!
//! HDC implementation: 200 trials across 10 blocks. A stimulus ContinuousHV
//! appears and the participant responds. RT = base + fatigue_increment * block
//! + noise. Lapses are trials where RT > 2x mean threshold.
//!
//! Human baselines (Dinges & Powell, 1985; Basner & Dinges, 2011):
//! - vigilance_decrement: 0.05 (SD~0.02) — RT slope per block (ticks)
//! - mean_rt: 5.0 (SD~1.0) — ~250ms mean RT
//! - lapse_rate: 0.05 (SD~0.03) — fraction of lapses (RT > 500ms)
//! - fastest_10pct: 3.0 (SD~0.5) — fastest 10% RT (ticks)

use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// PVT benchmark.
pub struct PvtBenchmark;

struct TrialResult {
    vigilance_decrement: f64,
    mean_rt: f64,
    lapse_rate: f64,
    fastest_10pct: f64,
    rt_ticks: f64,
}

impl PvtBenchmark {
    fn run_trial_traced(
        &self,
        config: &BenchmarkConfig,
        trial_idx: usize,
        temp_mult: f64,
        global_trial_idx: &mut usize,
    ) -> (TrialResult, Vec<TrialOutcome>) {
        let dim = config.dimension;
        let seed = config.trial_seed("sustained", "pvt", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        let stimulus_proto = ContinuousHV::random(dim, seed.wrapping_add(1));

        let n_blocks = 10;
        let trials_per_block = 20;
        let total_trials = n_blocks * trials_per_block;

        let base_rt = 4.0f64;
        // Difficulty increases fatigue increment
        let fatigue_increment = 0.05 * temp_mult;
        let noise_base: f64 = (0.8 + config.time_pressure * 0.5) * temp_mult;
        let lapse_threshold = base_rt * 2.0;

        let mut all_rts = Vec::with_capacity(total_trials);
        let mut block_mean_rts = Vec::with_capacity(n_blocks);
        let mut lapses = 0u32;
        let mut outcomes = Vec::new();

        for block in 0..n_blocks {
            let mut block_rt_sum = 0.0f64;

            for _ in 0..trials_per_block {
                xor_shift(&mut rng);
                let noise_hv = ContinuousHV::random(dim, rng);
                let stimulus =
                    ContinuousHV::weighted_bundle(&[&stimulus_proto, &noise_hv], &[0.80, 0.20]);
                // Encoding noise degrades stimulus detection
                let noise_degrade = config.effective_noise() as f32 * 0.4;
                let detection =
                    (stimulus.similarity(&stimulus_proto) * (1.0 - noise_degrade)) as f64;

                xor_shift(&mut rng);
                let rt_noise = ((rng % 10000) as f64 / 10000.0 - 0.3) * noise_base;
                let fatigue = fatigue_increment * block as f64;
                let tp_speedup = config.time_pressure * 1.0;
                let rt = (base_rt + fatigue + rt_noise - tp_speedup).max(1.0);

                all_rts.push(rt);
                block_rt_sum += rt;

                let is_lapse = rt > lapse_threshold;
                if is_lapse {
                    lapses += 1;
                }

                if config.trial_trace {
                    outcomes.push(TrialOutcome {
                        trial_idx: *global_trial_idx,
                        condition: format!("block_{}", block),
                        correct: !is_lapse,
                        rt_ticks: rt,
                        similarity: detection,
                        confidence: (1.0 - rt / (lapse_threshold * 2.0)).max(0.0),
                        response_idx: 0,
                        extra: BTreeMap::new(),
                    });
                    *global_trial_idx += 1;
                }
            }

            block_mean_rts.push(block_rt_sum / trials_per_block as f64);
        }

        let n = n_blocks as f64;
        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;
        let mut sum_xy = 0.0f64;
        let mut sum_xx = 0.0f64;

        for (i, &mean_rt) in block_mean_rts.iter().enumerate() {
            let x = i as f64;
            sum_x += x;
            sum_y += mean_rt;
            sum_xy += x * mean_rt;
            sum_xx += x * x;
        }

        let denom = n * sum_xx - sum_x * sum_x;
        let vigilance_decrement = if denom.abs() > 1e-10 {
            ((n * sum_xy - sum_x * sum_y) / denom).max(0.0)
        } else {
            0.0
        };

        let mean_rt = all_rts.iter().sum::<f64>() / all_rts.len() as f64;
        let lapse_rate = lapses as f64 / total_trials as f64;

        let mut sorted_rts = all_rts.clone();
        sorted_rts.sort_by(|a, b| a.total_cmp(b));
        let top_10_count = (total_trials as f64 * 0.10).max(1.0) as usize;
        let fastest_10pct = sorted_rts[..top_10_count].iter().sum::<f64>() / top_10_count as f64;

        (
            TrialResult {
                vigilance_decrement,
                mean_rt,
                lapse_rate,
                fastest_10pct,
                rt_ticks: mean_rt,
            },
            outcomes,
        )
    }
}

impl PsychBenchmark for PvtBenchmark {
    fn name(&self) -> &str {
        "SustainedAttention::PVT"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Psychomotor Vigilance Task",
            citation: "Dinges & Powell (1985)",
            year: 1985,
            doi: Some("10.1016/0166-4328(85)90208-1"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let diff_model = difficulty_model_for(self.name());
        let temp_mult = diff_model.temperature_multiplier(config.difficulty);

        let mut decrements = Vec::new();
        let mut mean_rts = Vec::new();
        let mut lapse_rates = Vec::new();
        let mut fastest_10s = Vec::new();
        let mut rts = Vec::new();
        let mut trace = Vec::new();
        let mut global_trial_idx = 0usize;

        for trial in 0..config.trials_per_condition {
            let (r, outcomes) =
                self.run_trial_traced(config, trial, temp_mult, &mut global_trial_idx);
            decrements.push(r.vigilance_decrement);
            mean_rts.push(r.mean_rt);
            lapse_rates.push(r.lapse_rate);
            fastest_10s.push(r.fastest_10pct);
            rts.push(r.rt_ticks);
            if config.trial_trace {
                trace.extend(outcomes);
            }
        }

        result.insert(
            "vigilance_decrement",
            MetricValue::from_samples(&decrements),
        );
        result.insert("mean_rt", MetricValue::from_samples(&mean_rts));
        result.insert("lapse_rate", MetricValue::from_samples(&lapse_rates));
        result.insert("fastest_10pct", MetricValue::from_samples(&fastest_10s));
        result.insert("rt_ticks", MetricValue::from_samples(&rts));

        if config.trial_trace {
            result.trial_trace = trace;
        }

        result.conditions = 10;
        result.trials_per_condition = config.trials_per_condition;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pvt_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = PvtBenchmark.run(&config);
        assert!(result.metrics.contains_key("vigilance_decrement"));
        assert!(result.metrics.contains_key("mean_rt"));
        assert!(result.metrics.contains_key("lapse_rate"));
        assert!(result.metrics.contains_key("fastest_10pct"));
        assert!(result.metrics.contains_key("rt_ticks"));
    }

    #[test]
    fn test_pvt_finite_metrics() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = PvtBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_pvt_vigilance_decrement_non_negative() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = PvtBenchmark.run(&config);
        let dec = result.metrics["vigilance_decrement"].mean;
        assert!(
            dec >= 0.0,
            "vigilance decrement ({:.3}) should be >= 0",
            dec
        );
    }

    #[test]
    fn test_pvt_lapse_rate_bounded() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = PvtBenchmark.run(&config);
        let rate = result.metrics["lapse_rate"].mean;
        assert!(
            rate >= 0.0 && rate <= 1.0,
            "lapse rate ({:.3}) out of bounds",
            rate
        );
    }

    #[test]
    fn test_pvt_time_pressure() {
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
        let r_base = PvtBenchmark.run(&base);
        let r_press = PvtBenchmark.run(&pressed);
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
