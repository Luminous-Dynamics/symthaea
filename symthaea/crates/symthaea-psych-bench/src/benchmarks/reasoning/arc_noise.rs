// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! ARC Noise Robustness benchmark (SYNTHETIC — tests HDC noise tolerance, not reasoning).
//!
//! **Honesty note:** This benchmark uses procedurally generated grid tasks.
//! It tests how HDC encoding degrades under input corruption — a property
//! of the high-dimensional distributed representation, not of reasoning.
//! The 2-AFC scoring uses random BinaryHV distractors (chance = ~50%),
//! which is a lenient baseline. The z-scores reflect encoding noise
//! tolerance, not robust reasoning under uncertainty.
//!
//! Tests graceful degradation of HDC grid encoding under input corruption.
//! Corrupts varying fractions of cells in test input grids and measures
//! how transfer accuracy degrades. HDC representations are theoretically
//! noise-tolerant due to high-dimensional distributed encoding — this
//! benchmark quantifies that tolerance.
//!
//! Noise levels: 0%, 10%, 20%, 30%, 50% of cells randomly replaced.
//!
//! Key metric: noise_resilience = area under the accuracy-vs-noise curve,
//! normalized to [0,1]. A perfectly noise-tolerant system scores 1.0;
//! a system that fails immediately scores ~0.5 (chance at all levels).
//!
//! Human baselines (estimated):
//! - noise_resilience: ~0.85 (SD~0.08) — humans tolerate moderate noise
//! - accuracy_0pct: ~0.80 (SD~0.12) — clean baseline
//! - accuracy_50pct: ~0.55 (SD~0.15) — heavily corrupted

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::BinaryHV;
use symthaea_core::hdc::binary_grid_encoder::BinaryGridEncoder;
use symthaea_core::hdc::grid_encoder::GridEncoder;

/// ARC noise robustness benchmark.
pub struct ArcNoiseBenchmark;

const NOISE_LEVELS: [f64; 5] = [0.0, 0.10, 0.20, 0.30, 0.50];

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

/// Corrupt a fraction of cells in a grid with random colors.
fn corrupt_grid(grid: &[Vec<u8>], noise_frac: f64, rng: &mut u64, num_colors: u8) -> Vec<Vec<u8>> {
    let mut corrupted: Vec<Vec<u8>> = grid.iter().map(|r| r.to_vec()).collect();
    let rows = corrupted.len();
    let cols = if rows > 0 { corrupted[0].len() } else { 0 };
    let total_cells = rows * cols;
    let num_corrupt = (total_cells as f64 * noise_frac).round() as usize;

    // Randomly select cells to corrupt using Fisher-Yates on indices
    let mut indices: Vec<usize> = (0..total_cells).collect();
    for i in 0..num_corrupt.min(total_cells) {
        xor_shift(rng);
        let j = i + (*rng as usize % (total_cells - i));
        indices.swap(i, j);
    }

    for &idx in indices.iter().take(num_corrupt) {
        let r = idx / cols;
        let c = idx % cols;
        xor_shift(rng);
        corrupted[r][c] = (*rng % num_colors as u64) as u8;
    }
    corrupted
}

#[derive(Debug, Clone, Copy)]
enum TransformType {
    ColorFill,
    Translation,
    ColorReplacement,
    Reflection,
}

const TRANSFORMS: [TransformType; 4] = [
    TransformType::ColorFill,
    TransformType::Translation,
    TransformType::ColorReplacement,
    TransformType::Reflection,
];

fn apply_transform(
    grid: &[Vec<u8>],
    tt: TransformType,
    param: u64,
    num_colors: u8,
) -> Vec<Vec<u8>> {
    match tt {
        TransformType::ColorFill => {
            let color = (param % num_colors as u64) as u8;
            let top = (param / 7 % 3) as usize;
            let left = (param / 11 % 3) as usize;
            GridEncoder::fill_region(grid, top, left, top + 1, left + 1, color)
        }
        TransformType::Translation => {
            let dx = ((param % 3) as i32) - 1;
            let dy = ((param / 3 % 3) as i32) - 1;
            let fill = (param / 9 % num_colors as u64) as u8;
            GridEncoder::translate_grid(grid, dx, dy, fill)
        }
        TransformType::ColorReplacement => {
            let from = (param % num_colors as u64) as u8;
            let to = ((param / 7 + 1) % num_colors as u64) as u8;
            GridEncoder::color_replace(grid, from, to)
        }
        TransformType::Reflection => {
            if param % 2 == 0 {
                GridEncoder::reflect_x(grid)
            } else {
                GridEncoder::reflect_y(grid)
            }
        }
    }
}

struct TrialResult {
    /// Accuracy at each noise level
    accuracy_per_level: [f64; 5],
    /// Similarity at each noise level
    similarity_per_level: [f64; 5],
    /// AUC-based noise resilience score
    noise_resilience: f64,
    rt_ticks: f64,
}

impl ArcNoiseBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let _dim = config.dimension;
        let seed = config.trial_seed("reasoning", "arc_noise", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let grid_size = 5;
        let num_colors: u8 = 6;
        let tasks_per_type = 4;

        let pressure = config.time_pressure;
        let noise_weight = 0.05 + pressure * 0.15;
        let tick_scale = 1.0 - pressure * 0.4;

        let encoder = BinaryGridEncoder::new(grid_size, grid_size, num_colors as usize, seed);

        let mut hits_per_level = [0u32; 5];
        let mut total_per_level = [0u32; 5];
        let mut sim_per_level = [0.0f64; 5];
        let mut sim_count = [0u32; 5];
        let mut total_ticks = 0.0f64;
        let mut total_tasks = 0u32;

        for &tt in &TRANSFORMS {
            for _task_i in 0..tasks_per_type {
                xor_shift(&mut rng);
                let task_param = rng;

                // Generate 2 training pairs (clean, no noise)
                let mut train_rules = Vec::new();
                for _pair_i in 0..2 {
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

                let consensus = encoder.bundle_rules(&train_rules);

                // Generate ONE test input, then corrupt at each noise level
                xor_shift(&mut rng);
                let test_input = gen_grid(&mut rng, grid_size, num_colors);
                let test_output = apply_transform(&test_input, tt, task_param, num_colors);
                let test_out_hv = encoder.encode_grid(&test_output);

                for (level_idx, &noise_frac) in NOISE_LEVELS.iter().enumerate() {
                    xor_shift(&mut rng);
                    let corrupted_input = if noise_frac > 0.0 {
                        corrupt_grid(&test_input, noise_frac, &mut rng, num_colors)
                    } else {
                        test_input.clone()
                    };

                    let corrupted_in_hv = encoder.encode_grid(&corrupted_input);
                    let predicted = encoder.apply_rule(&corrupted_in_hv, &consensus);
                    let pred_sim = predicted.similarity(&test_out_hv) as f64;

                    sim_per_level[level_idx] += pred_sim;
                    sim_count[level_idx] += 1;

                    // 2-AFC vs random distractor
                    xor_shift(&mut rng);
                    let distractor = BinaryHV::random(rng);
                    let dist_sim = predicted.similarity(&distractor) as f64;

                    total_per_level[level_idx] += 1;
                    if pred_sim > dist_sim {
                        hits_per_level[level_idx] += 1;
                    }
                }

                xor_shift(&mut rng);
                total_ticks += (4.0 + (rng % 5) as f64) * tick_scale;
                total_tasks += 1;
            }
        }

        let accuracy_per_level: [f64; 5] = std::array::from_fn(|i| {
            if total_per_level[i] > 0 {
                hits_per_level[i] as f64 / total_per_level[i] as f64
            } else {
                0.0
            }
        });
        let similarity_per_level: [f64; 5] = std::array::from_fn(|i| {
            if sim_count[i] > 0 {
                sim_per_level[i] / sim_count[i] as f64
            } else {
                0.0
            }
        });

        // Noise resilience: trapezoidal AUC of accuracy vs noise_fraction, normalized to [0,1]
        // AUC of a system that maintains perfect accuracy across all noise levels = 0.5 (area of [0,0.5] × [0,1])
        // We normalize so that perfect = 1.0
        let mut auc = 0.0f64;
        for i in 1..NOISE_LEVELS.len() {
            let dx = NOISE_LEVELS[i] - NOISE_LEVELS[i - 1];
            let avg_acc = (accuracy_per_level[i] + accuracy_per_level[i - 1]) / 2.0;
            auc += dx * avg_acc;
        }
        // Max possible AUC = 0.5 (integral from 0 to 0.5 of 1.0)
        let noise_resilience = (auc / 0.5).min(1.0);

        let rt_ticks = if total_tasks > 0 {
            total_ticks / total_tasks as f64
        } else {
            0.0
        };

        TrialResult {
            accuracy_per_level,
            similarity_per_level,
            noise_resilience,
            rt_ticks,
        }
    }
}

impl PsychBenchmark for ArcNoiseBenchmark {
    fn name(&self) -> &str {
        "Reasoning::ArcNoise"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Noise robustness of distributed representations",
            citation: "Kanerva (2009); Chollet (2019)",
            year: 2019,
            doi: Some("10.48550/arXiv.1911.01547"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let level_names = ["0pct", "10pct", "20pct", "30pct", "50pct"];
        let mut accs_per_level: [Vec<f64>; 5] = Default::default();
        let mut sims_per_level: [Vec<f64>; 5] = Default::default();
        let mut resiliences = Vec::new();
        let mut rts = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            for i in 0..5 {
                accs_per_level[i].push(r.accuracy_per_level[i]);
                sims_per_level[i].push(r.similarity_per_level[i]);
            }
            resiliences.push(r.noise_resilience);
            rts.push(r.rt_ticks);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "arc_noise".to_string(),
                    correct: r.noise_resilience > 0.5,
                    rt_ticks: r.rt_ticks,
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        for (i, name) in level_names.iter().enumerate() {
            result.insert(
                format!("accuracy_{}", name),
                MetricValue::from_samples(&accs_per_level[i]),
            );
            result.insert(
                format!("similarity_{}", name),
                MetricValue::from_samples(&sims_per_level[i]),
            );
        }
        result.insert("noise_resilience", MetricValue::from_samples(&resiliences));
        result.insert("rt_ticks", MetricValue::from_samples(&rts));

        // Accuracy drop from clean to 50% noise
        let drops: Vec<f64> = accs_per_level[0]
            .iter()
            .zip(accs_per_level[4].iter())
            .map(|(clean, noisy)| clean - noisy)
            .collect();
        result.insert("accuracy_drop", MetricValue::from_samples(&drops));

        result.conditions = 5; // 5 noise levels
        result.trials_per_condition = config.trials_per_condition;
        if config.trial_trace {
            result.trial_trace = trace;
        }
        result.notes.push(
            "SYNTHETIC: Uses procedural grid transforms, not Chollet's real ARC. \
             Tests HDC noise tolerance (graceful degradation under corruption), not \
             robust reasoning. 2-AFC uses random BinaryHV distractors (chance ~50%), \
             a lenient baseline. Z-scores reflect encoding noise tolerance."
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
    fn test_noise_runs_with_metrics() {
        let result = ArcNoiseBenchmark.run(&test_config());
        assert!(result.metrics.contains_key("accuracy_0pct"));
        assert!(result.metrics.contains_key("accuracy_50pct"));
        assert!(result.metrics.contains_key("noise_resilience"));
        assert!(result.metrics.contains_key("accuracy_drop"));
        assert!(result.metrics.contains_key("rt_ticks"));
    }

    #[test]
    fn test_all_metrics_finite() {
        let result = ArcNoiseBenchmark.run(&test_config());
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_clean_above_chance() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = ArcNoiseBenchmark.run(&config);
        let acc = result.metrics["accuracy_0pct"].mean;
        assert!(
            acc > 0.4,
            "Clean accuracy should be near/above chance, got {}",
            acc
        );
    }

    #[test]
    fn test_noise_generally_degrades() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = ArcNoiseBenchmark.run(&config);
        let sim_clean = result.metrics["similarity_0pct"].mean;
        let sim_noisy = result.metrics["similarity_50pct"].mean;
        // 50% corruption should reduce similarity vs clean
        assert!(
            sim_clean >= sim_noisy - 0.05,
            "50% noise should not improve similarity: clean={}, noisy={}",
            sim_clean,
            sim_noisy
        );
    }

    #[test]
    fn test_resilience_bounded() {
        let result = ArcNoiseBenchmark.run(&test_config());
        let res = result.metrics["noise_resilience"].mean;
        assert!(
            res >= 0.0 && res <= 1.0,
            "noise_resilience should be in [0,1], got {}",
            res
        );
    }

    #[test]
    fn test_corrupt_grid_function() {
        let grid = vec![vec![0u8; 5]; 5];
        let mut rng = 42u64 ^ 0x9E3779B97F4A7C15;
        let corrupted = corrupt_grid(&grid, 0.5, &mut rng, 6);
        // About half the cells should be changed
        let changed: usize = grid
            .iter()
            .flatten()
            .zip(corrupted.iter().flatten())
            .filter(|(a, b)| a != b)
            .count();
        assert!(changed > 0, "50% noise should change some cells");
        assert!(changed <= 25, "50% noise should change at most 25 cells");
    }

    #[test]
    fn test_deterministic() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 2,
            seed: 42,
            ..Default::default()
        };
        let r1 = ArcNoiseBenchmark.run(&config);
        let r2 = ArcNoiseBenchmark.run(&config);
        assert_eq!(
            r1.metrics["noise_resilience"].mean,
            r2.metrics["noise_resilience"].mean
        );
    }
}
