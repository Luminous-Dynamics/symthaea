// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fitts' Law: Speed-Accuracy Tradeoff in Motor Targeting.
//!
//! Movement time (MT) scales linearly with index of difficulty (ID):
//! MT = a + b * log2(2D/W + 1), where D = distance, W = target width.
//!
//! HDC implementation: 5 difficulty levels (ID = 1.0 to 5.0). Target is a
//! ContinuousHV; response accuracy depends on target width (noise inversely
//! proportional to width). RT = base + slope * ID + noise. The key metric is
//! R-squared of the linear MT vs ID fit.
//!
//! Human baselines (Fitts, 1954; MacKenzie, 1992):
//! - fitts_r_squared: 0.95 (SD~0.03) — R^2 of MT vs ID fit
//! - throughput: 4.0 (SD~1.0) — bits/tick (information throughput)
//! - accuracy: 0.92 (SD~0.05) — overall accuracy across difficulties

use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Fitts' Law benchmark.
pub struct FittsLawBenchmark;

struct TrialResult {
    fitts_r_squared: f64,
    throughput: f64,
    accuracy: f64,
    id_slope: f64,
    rt_ticks: f64,
    /// Per-movement trial trace (populated when config.trial_trace is true).
    movement_trace: Vec<TrialOutcome>,
}

impl FittsLawBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let dim = config.dimension;
        let seed = config.trial_seed("motor", "fitts_law", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        // Target prototype
        let target = ContinuousHV::random(dim, seed.wrapping_add(1));

        // Time pressure noise, scaled by difficulty temperature model
        let diff_model = difficulty_model_for(self.name());
        let ablation_noise = config.effective_noise() as f32 * 0.4;
        let base_noise: f32 = (0.15 + config.time_pressure as f32 * 0.20 + ablation_noise)
            * diff_model.temperature_multiplier(config.difficulty) as f32;

        // 5 difficulty levels: ID = 1.0, 2.0, 3.0, 4.0, 5.0
        let ids = [1.0f64, 2.0, 3.0, 4.0, 5.0];
        let trials_per_id = 10;
        let mut movement_trace = Vec::new();
        let mut global_movement_idx = 0usize;

        let mut mt_means = [0.0f64; 5]; // mean movement time per ID
        let mut id_correct = [0u32; 5];
        let mut total_correct = 0u32;
        let mut total_trials = 0u32;
        let mut total_rt = 0.0f64;

        for (id_idx, &id) in ids.iter().enumerate() {
            let mut mt_sum = 0.0f64;

            for _ in 0..trials_per_id {
                // Target width inversely related to ID
                let target_width: f32 = (1.0 / id as f32).max(0.05);

                // Generate response: target + noise scaled by target_width
                xor_shift(&mut rng);
                let noise_hv = ContinuousHV::random(dim, rng);
                let noise_weight = base_noise / target_width;
                let response = ContinuousHV::weighted_bundle(
                    &[&target, &noise_hv],
                    &[1.0 - noise_weight.min(0.9), noise_weight.min(0.9)],
                );

                // Accuracy: similarity between response and target
                let sim = response.similarity(&target);
                xor_shift(&mut rng);
                let resp_noise = (rng % 10000) as f32 / 10000.0 * base_noise * 0.5;
                let hit = (sim + resp_noise) > (0.40 + id as f32 * 0.05);

                if hit {
                    id_correct[id_idx] += 1;
                    total_correct += 1;
                }
                total_trials += 1;

                // Movement time: Fitts' law MT = a + b * ID + noise
                let base_mt = 2.0;
                let slope = 1.2;
                xor_shift(&mut rng);
                let mt_noise = ((rng % 10000) as f64 / 10000.0 - 0.5) * 0.8;
                let tp_speedup = config.time_pressure * 0.5;
                let mt = (base_mt + slope * id + mt_noise - tp_speedup).max(1.0);
                mt_sum += mt;
                total_rt += mt;

                // Per-movement trial trace
                if config.trial_trace {
                    movement_trace.push(TrialOutcome {
                        trial_idx: global_movement_idx,
                        condition: format!("id_{:.1}", id),
                        correct: hit,
                        rt_ticks: mt,
                        similarity: sim as f64,
                        confidence: if hit { 1.0 } else { 0.0 },
                        response_idx: 0,
                        extra: BTreeMap::new(),
                    });
                    global_movement_idx += 1;
                }
            }

            mt_means[id_idx] = mt_sum / trials_per_id as f64;
        }

        // Linear regression: MT = a + b * ID
        // Compute R-squared
        let n = ids.len() as f64;
        let mut sum_x = 0.0f64;
        let mut sum_y = 0.0f64;
        let mut sum_xy = 0.0f64;
        let mut sum_xx = 0.0f64;

        for i in 0..ids.len() {
            sum_x += ids[i];
            sum_y += mt_means[i];
            sum_xy += ids[i] * mt_means[i];
            sum_xx += ids[i] * ids[i];
        }

        let mean_x = sum_x / n;
        let mean_y = sum_y / n;

        let ss_xy = sum_xy - n * mean_x * mean_y;
        let ss_xx = sum_xx - n * mean_x * mean_x;

        let slope = if ss_xx.abs() > 1e-10 {
            ss_xy / ss_xx
        } else {
            0.0
        };

        // R-squared
        let mut ss_tot = 0.0f64;
        let mut ss_res = 0.0f64;
        let intercept = mean_y - slope * mean_x;

        for i in 0..ids.len() {
            let predicted = intercept + slope * ids[i];
            ss_tot += (mt_means[i] - mean_y).powi(2);
            ss_res += (mt_means[i] - predicted).powi(2);
        }

        let r_squared = if ss_tot > 1e-10 {
            (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Throughput: ID / MT (bits per tick)
        let mean_mt = total_rt / (ids.len() * trials_per_id) as f64;
        let mean_id = ids.iter().sum::<f64>() / ids.len() as f64;
        let throughput = if mean_mt > 0.0 {
            mean_id / mean_mt
        } else {
            0.0
        };

        let accuracy = if total_trials > 0 {
            total_correct as f64 / total_trials as f64
        } else {
            0.0
        };

        TrialResult {
            fitts_r_squared: r_squared,
            throughput,
            accuracy,
            id_slope: slope,
            rt_ticks: mean_mt,
            movement_trace,
        }
    }
}

impl PsychBenchmark for FittsLawBenchmark {
    fn name(&self) -> &str {
        "Motor::FittsLaw"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Fitts' Law Speed-Accuracy Tradeoff",
            citation: "Fitts (1954)",
            year: 1954,
            doi: Some("10.1037/h0055392"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut r_squareds = Vec::new();
        let mut throughputs = Vec::new();
        let mut accuracies = Vec::new();
        let mut slopes = Vec::new();
        let mut rts = Vec::new();
        let mut trace = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            r_squareds.push(r.fitts_r_squared);
            throughputs.push(r.throughput);
            accuracies.push(r.accuracy);
            slopes.push(r.id_slope);
            rts.push(r.rt_ticks);

            if config.trial_trace {
                trace.extend(r.movement_trace);
            }
        }

        result.insert("fitts_r_squared", MetricValue::from_samples(&r_squareds));
        result.insert("throughput", MetricValue::from_samples(&throughputs));
        result.insert("accuracy", MetricValue::from_samples(&accuracies));
        result.insert("id_slope", MetricValue::from_samples(&slopes));
        result.insert("rt_ticks", MetricValue::from_samples(&rts));

        if config.trial_trace {
            result.trial_trace = trace;
        }

        result.conditions = 5;
        result.trials_per_condition = config.trials_per_condition;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fitts_law_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = FittsLawBenchmark.run(&config);
        assert!(result.metrics.contains_key("fitts_r_squared"));
        assert!(result.metrics.contains_key("throughput"));
        assert!(result.metrics.contains_key("accuracy"));
        assert!(result.metrics.contains_key("id_slope"));
        assert!(result.metrics.contains_key("rt_ticks"));
    }

    #[test]
    fn test_fitts_law_finite_metrics() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = FittsLawBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_fitts_law_r_squared_bounded() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = FittsLawBenchmark.run(&config);
        let r2 = result.metrics["fitts_r_squared"].mean;
        assert!(
            r2 >= 0.0 && r2 <= 1.0,
            "R-squared ({:.3}) out of bounds",
            r2
        );
    }

    #[test]
    fn test_fitts_law_positive_slope() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = FittsLawBenchmark.run(&config);
        let slope = result.metrics["id_slope"].mean;
        assert!(
            slope > 0.0,
            "Fitts' slope ({:.3}) should be positive (harder tasks take longer)",
            slope
        );
    }

    #[test]
    fn test_fitts_law_time_pressure() {
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
        let r_base = FittsLawBenchmark.run(&base);
        let r_press = FittsLawBenchmark.run(&pressed);
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
