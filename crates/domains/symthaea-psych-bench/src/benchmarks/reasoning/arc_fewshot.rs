// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! ARC Few-Shot Scaling benchmark (SYNTHETIC — tests HDC bundling, not learning).
//!
//! **Honesty note:** This benchmark uses procedurally generated grid tasks.
//! It tests how HDC majority-vote bundling improves with more examples — a
//! property of the encoding algebra, not of learning or generalization.
//! **FIXED (2026-07-18)**: the 2-AFC scoring used to use random BinaryHV
//! distractors (chance = ~50%) — see `arc_dataset.rs`'s retraction note for
//! why that's a discriminability artifact, not a fair baseline (on real
//! ARC-AGI data it measured 99.0% vs. 13.8-67.8% under fair distractors). Now
//! uses `arc_dataset::fair_distractor_grid` (a generic wrong transform of the
//! test input) instead. The z-scores still reflect bundling efficiency, not
//! few-shot learning ability.
//!
//! Tests how transfer accuracy scales with the number of training examples
//! (1, 2, 3, 4, 5). This reveals the learning curve of HDC rule bundling:
//! does consensus improve monotonically with more examples, or is there
//! a saturation point?
//!
//! For each task, generates 5 training pairs showing the same transform.
//! Tests transfer at each k (1..=5) by bundling only the first k rules.
//!
//! Human baselines (estimated from Chollet 2019; few-shot learning lit):
//! - accuracy_1shot: ~0.60 (SD~0.15) — single example
//! - accuracy_5shot: ~0.85 (SD~0.10) — five examples
//! - saturation_point: ~3 (SD~1) — where gains plateau
//! - learning_rate: ~0.06 (SD~0.03) — accuracy gain per example

use crate::benchmarks::reasoning::arc_dataset::fair_distractor_grid;
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::binary_grid_encoder::BinaryGridEncoder;
use symthaea_core::hdc::grid_encoder::GridEncoder;

/// ARC few-shot scaling benchmark.
pub struct ArcFewShotBenchmark;

const MAX_SHOTS: usize = 5;

fn xor_shift(s: &mut u64) {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
}

fn gen_grid(rng: &mut u64, size: usize, num_colors: u8) -> Vec<Vec<u8>> {
    let mut grid = vec![vec![0u8; size]; size];
    for row in grid.iter_mut() {
        for cell in row.iter_mut() {
            xor_shift(rng);
            *cell = (*rng % num_colors as u64) as u8;
        }
    }
    grid
}

#[derive(Debug, Clone, Copy)]
enum TaskType {
    ColorFill,
    Translation,
    ColorReplacement,
    Reflection,
}

const TASK_TYPES: [TaskType; 4] = [
    TaskType::ColorFill,
    TaskType::Translation,
    TaskType::ColorReplacement,
    TaskType::Reflection,
];

fn apply_transform(grid: &[Vec<u8>], tt: TaskType, param: u64, num_colors: u8) -> Vec<Vec<u8>> {
    match tt {
        TaskType::ColorFill => {
            let color = (param % num_colors as u64) as u8;
            let top = (param / 7 % 3) as usize;
            let left = (param / 11 % 3) as usize;
            GridEncoder::fill_region(grid, top, left, top + 1, left + 1, color)
        }
        TaskType::Translation => {
            let dx = ((param % 3) as i32) - 1;
            let dy = ((param / 3 % 3) as i32) - 1;
            let fill = (param / 9 % num_colors as u64) as u8;
            GridEncoder::translate_grid(grid, dx, dy, fill)
        }
        TaskType::ColorReplacement => {
            let from = (param % num_colors as u64) as u8;
            let to = ((param / 7 + 1) % num_colors as u64) as u8;
            GridEncoder::color_replace(grid, from, to)
        }
        TaskType::Reflection => {
            if param % 2 == 0 {
                GridEncoder::reflect_x(grid)
            } else {
                GridEncoder::reflect_y(grid)
            }
        }
    }
}

struct TrialResult {
    /// Accuracy at each shot count: [1-shot, 2-shot, 3-shot, 4-shot, 5-shot]
    accuracy_per_shot: [f64; MAX_SHOTS],
    /// Similarity at each shot count
    similarity_per_shot: [f64; MAX_SHOTS],
    /// Linear slope of accuracy vs shot count
    learning_rate: f64,
    /// Shot count where accuracy first exceeds 90% of max (or MAX_SHOTS if never)
    saturation_point: f64,
    rt_ticks: f64,
}

impl ArcFewShotBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let _dim = config.dimension;
        let seed = config.trial_seed("reasoning", "arc_fewshot", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let grid_size = 5;
        let num_colors: u8 = 6;
        let tasks_per_type = 4;

        let pressure = config.time_pressure;
        let noise_weight = 0.04 + pressure * 0.12;
        let tick_scale = 1.0 - pressure * 0.4;

        let encoder = BinaryGridEncoder::new(grid_size, grid_size, num_colors as usize, seed);

        let mut hits_per_shot = [0u32; MAX_SHOTS];
        let mut total_per_shot = [0u32; MAX_SHOTS];
        let mut sim_per_shot = [0.0f64; MAX_SHOTS];
        let mut sim_count = [0u32; MAX_SHOTS];
        let mut total_ticks = 0.0f64;
        let mut total_tasks = 0u32;

        for &tt in &TASK_TYPES {
            for _task_i in 0..tasks_per_type {
                xor_shift(&mut rng);
                let task_param = rng;

                // Generate MAX_SHOTS training pairs
                let mut train_rules = Vec::new();
                for _pair_i in 0..MAX_SHOTS {
                    xor_shift(&mut rng);
                    let input = gen_grid(&mut rng, grid_size, num_colors);
                    let output = apply_transform(&input, tt, task_param, num_colors);
                    let in_hv = encoder.encode_grid(&input);
                    let out_hv = encoder.encode_grid(&output);
                    let mut rule = encoder.encode_rule(&in_hv, &out_hv);

                    if noise_weight > 0.0 {
                        xor_shift(&mut rng);
                        rule = rule.add_noise(noise_weight as f32, rng);
                    }
                    train_rules.push(rule);
                }

                // Generate test pair
                xor_shift(&mut rng);
                let test_input = gen_grid(&mut rng, grid_size, num_colors);
                let test_output = apply_transform(&test_input, tt, task_param, num_colors);
                let test_in_hv = encoder.encode_grid(&test_input);
                let test_out_hv = encoder.encode_grid(&test_output);

                // 2-AFC distractor: fair (equally structured), not random noise
                xor_shift(&mut rng);
                let distractor_grid = fair_distractor_grid(&test_input, &test_output)
                    .unwrap_or_else(|| test_input.clone());
                let distractor = encoder.encode_grid(&distractor_grid);

                // Test at each shot count (1..=5)
                for k in 1..=MAX_SHOTS {
                    let consensus = encoder.bundle_rules(&train_rules[..k]);
                    let predicted = encoder.apply_rule(&test_in_hv, &consensus);

                    let pred_sim = predicted.similarity(&test_out_hv) as f64;
                    let dist_sim = predicted.similarity(&distractor) as f64;

                    let shot_idx = k - 1;
                    sim_per_shot[shot_idx] += pred_sim;
                    sim_count[shot_idx] += 1;
                    total_per_shot[shot_idx] += 1;
                    if pred_sim > dist_sim {
                        hits_per_shot[shot_idx] += 1;
                    }
                }

                xor_shift(&mut rng);
                total_ticks += (4.0 + (rng % 5) as f64) * tick_scale;
                total_tasks += 1;
            }
        }

        let accuracy_per_shot: [f64; MAX_SHOTS] = std::array::from_fn(|i| {
            if total_per_shot[i] > 0 {
                hits_per_shot[i] as f64 / total_per_shot[i] as f64
            } else {
                0.0
            }
        });
        let similarity_per_shot: [f64; MAX_SHOTS] = std::array::from_fn(|i| {
            if sim_count[i] > 0 {
                sim_per_shot[i] / sim_count[i] as f64
            } else {
                0.0
            }
        });

        // Linear regression slope: accuracy = slope * k + intercept
        // Using least squares: slope = (sum(x*y) - n*mean_x*mean_y) / (sum(x²) - n*mean_x²)
        let n = MAX_SHOTS as f64;
        let mean_x = (1.0 + n) / 2.0; // mean of 1..5 = 3
        let mean_y: f64 = accuracy_per_shot.iter().sum::<f64>() / n;
        let mut sum_xy = 0.0f64;
        let mut sum_xx = 0.0f64;
        for k in 1..=MAX_SHOTS {
            let x = k as f64;
            sum_xy += x * accuracy_per_shot[k - 1];
            sum_xx += x * x;
        }
        let learning_rate = if (sum_xx - n * mean_x * mean_x).abs() > 1e-10 {
            (sum_xy - n * mean_x * mean_y) / (sum_xx - n * mean_x * mean_x)
        } else {
            0.0
        };

        // Saturation point: first k where accuracy >= 90% of max accuracy
        let max_acc = accuracy_per_shot.iter().cloned().fold(0.0f64, f64::max);
        let threshold = max_acc * 0.9;
        let saturation_point = accuracy_per_shot
            .iter()
            .position(|&a| a >= threshold)
            .map(|i| (i + 1) as f64) // 1-indexed shot count
            .unwrap_or(MAX_SHOTS as f64);

        let rt_ticks = if total_tasks > 0 {
            total_ticks / total_tasks as f64
        } else {
            0.0
        };

        TrialResult {
            accuracy_per_shot,
            similarity_per_shot,
            learning_rate,
            saturation_point,
            rt_ticks,
        }
    }
}

impl PsychBenchmark for ArcFewShotBenchmark {
    fn name(&self) -> &str {
        "Reasoning::ArcFewShot"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Synthetic HDC bundling efficiency (few-shot scaling)",
            citation: "Chollet (2019); Lake et al. (2015)",
            year: 2019,
            doi: Some("10.48550/arXiv.1911.01547"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut accs_per_shot: [Vec<f64>; MAX_SHOTS] = Default::default();
        let mut sims_per_shot: [Vec<f64>; MAX_SHOTS] = Default::default();
        let mut learning_rates = Vec::new();
        let mut saturation_points = Vec::new();
        let mut rts = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            for i in 0..MAX_SHOTS {
                accs_per_shot[i].push(r.accuracy_per_shot[i]);
                sims_per_shot[i].push(r.similarity_per_shot[i]);
            }
            learning_rates.push(r.learning_rate);
            saturation_points.push(r.saturation_point);
            rts.push(r.rt_ticks);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "arc_fewshot".to_string(),
                    correct: r.accuracy_per_shot[4] > 0.5,
                    rt_ticks: r.rt_ticks,
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        for k in 1..=MAX_SHOTS {
            result.insert(
                format!("accuracy_{}shot", k),
                MetricValue::from_samples(&accs_per_shot[k - 1]),
            );
            result.insert(
                format!("similarity_{}shot", k),
                MetricValue::from_samples(&sims_per_shot[k - 1]),
            );
        }
        result.insert("learning_rate", MetricValue::from_samples(&learning_rates));
        result.insert(
            "saturation_point",
            MetricValue::from_samples(&saturation_points),
        );
        result.insert("rt_ticks", MetricValue::from_samples(&rts));

        // Few-shot gain: 5-shot minus 1-shot accuracy
        let gains: Vec<f64> = accs_per_shot[0]
            .iter()
            .zip(accs_per_shot[4].iter())
            .map(|(one, five)| five - one)
            .collect();
        result.insert("fewshot_gain", MetricValue::from_samples(&gains));

        result.conditions = MAX_SHOTS; // 5 shot counts
        result.trials_per_condition = config.trials_per_condition;
        if config.trial_trace {
            result.trial_trace = trace;
        }
        result.notes.push(
            "SYNTHETIC: Uses procedural grid transforms, not Chollet's real ARC. \
             Tests HDC bundling efficiency (majority-vote consensus), not few-shot \
             learning. 2-AFC uses a fair (equally structured) distractor, not random \
             noise (fixed 2026-07-18). Z-scores reflect bundling quality."
                .to_string(),
        );
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> BenchmarkConfig {
        BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        }
    }

    #[test]
    fn test_fewshot_runs_with_metrics() {
        let result = ArcFewShotBenchmark.run(&test_config());
        assert!(result.metrics.contains_key("accuracy_1shot"));
        assert!(result.metrics.contains_key("accuracy_5shot"));
        assert!(result.metrics.contains_key("learning_rate"));
        assert!(result.metrics.contains_key("saturation_point"));
        assert!(result.metrics.contains_key("fewshot_gain"));
        assert!(result.metrics.contains_key("rt_ticks"));
    }

    #[test]
    fn test_all_metrics_finite() {
        let result = ArcFewShotBenchmark.run(&test_config());
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_similarity_improves_with_shots() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = ArcFewShotBenchmark.run(&config);
        let sim_1 = result.metrics["similarity_1shot"].mean;
        let sim_5 = result.metrics["similarity_5shot"].mean;
        // More examples should generally improve similarity (or at least not hurt badly)
        // Allow some tolerance since HDC bundling can be noisy
        assert!(
            sim_5 >= sim_1 - 0.1,
            "5-shot similarity should not be much worse than 1-shot: 1-shot={}, 5-shot={}",
            sim_1,
            sim_5
        );
    }

    #[test]
    fn test_saturation_bounded() {
        let result = ArcFewShotBenchmark.run(&test_config());
        let sat = result.metrics["saturation_point"].mean;
        assert!(
            sat >= 1.0 && sat <= 5.0,
            "Saturation should be in [1,5], got {}",
            sat
        );
    }

    #[test]
    fn test_learning_rate_finite() {
        let result = ArcFewShotBenchmark.run(&test_config());
        let lr = result.metrics["learning_rate"].mean;
        assert!(lr.is_finite(), "learning_rate should be finite");
        // Learning rate is slope of accuracy vs shot count — can be positive or negative
        assert!(
            lr > -0.5 && lr < 0.5,
            "learning_rate out of reasonable range: {}",
            lr
        );
    }

    #[test]
    fn test_provenance() {
        let prov = ArcFewShotBenchmark.provenance().unwrap();
        assert!(prov.citation.contains("Chollet"));
        assert!(prov.citation.contains("Lake"));
    }

    #[test]
    fn test_deterministic() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 2,
            seed: 42,
            ..Default::default()
        };
        let r1 = ArcFewShotBenchmark.run(&config);
        let r2 = ArcFewShotBenchmark.run(&config);
        assert_eq!(
            r1.metrics["learning_rate"].mean,
            r2.metrics["learning_rate"].mean
        );
    }
}
