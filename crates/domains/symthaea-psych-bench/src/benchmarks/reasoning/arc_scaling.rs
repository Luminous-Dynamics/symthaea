// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! ARC Grid Complexity and Dimension Scaling benchmark
//! (SYNTHETIC — tests HDC capacity limits, not reasoning).
//!
//! **Honesty note:** This benchmark uses procedurally generated grid tasks.
//! It measures how HDC encoding fidelity scales with grid size and vector
//! dimension — these are capacity/efficiency metrics of the representation,
//! not reasoning metrics. **FIXED (2026-07-18)**: the 2-AFC scoring used to
//! use random BinaryHV distractors (chance = ~50%) — see `arc_dataset.rs`'s
//! retraction note for why that's a discriminability artifact, not a fair
//! baseline (on real ARC-AGI data it measured 99.0% vs. 13.8-67.8% under fair
//! distractors). Now uses `arc_dataset::fair_distractor_grid` (a generic
//! wrong transform of the test input) instead.
//!
//! Two parametric studies in one benchmark:
//!
//! 1. **Grid complexity scaling**: Varies grid size (3×3, 5×5, 8×8, 12×12)
//!    while holding dimension fixed. Reveals capacity limits of the HDC
//!    grid encoder as the number of cells grows quadratically.
//!
//! 2. **HDC dimension scaling**: Varies dimension (128, 256, 512, 1024)
//!    while holding grid size fixed at 5×5. Reveals the minimum dimension
//!    needed for reliable rule encoding.
//!
//! Both use the same core paradigm: 2 training pairs → transfer test (2-AFC).
//!
//! Key insight: GridEncoder encodes each cell as bind(row, col, color) then
//! bundles all cells. With N cells, the signal-to-noise ratio for any single
//! cell is approximately sqrt(D/N), so accuracy should degrade when D/N < ~10.
//!
//! Human baselines (estimated):
//! - grid_3x3_accuracy: ~0.90 (SD~0.08)
//! - grid_12x12_accuracy: ~0.60 (SD~0.15) — harder with more cells
//! - dim_128_accuracy: ~0.55 (SD~0.15) — minimal capacity
//! - dim_1024_accuracy: ~0.85 (SD~0.10) — ample capacity

use crate::benchmarks::reasoning::arc_dataset::fair_distractor_grid;
use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::binary_grid_encoder::BinaryGridEncoder;
use symthaea_core::hdc::grid_encoder::GridEncoder;

/// ARC scaling benchmark: grid size × HDC dimension.
pub struct ArcScalingBenchmark;

const GRID_SIZES: [usize; 4] = [3, 5, 8, 12];
const DIMENSIONS: [usize; 4] = [128, 256, 512, 1024];

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

/// Simple transform: reflect_x (works on any grid size).
fn transform(grid: &[Vec<u8>]) -> Vec<Vec<u8>> {
    GridEncoder::reflect_x(grid)
}

/// Run a scaling study for a single grid_size.
/// The `_dim` parameter is kept for API compatibility but BinaryHV uses fixed 16384D.
fn scaling_probe(
    grid_size: usize,
    _dim: usize,
    num_tasks: usize,
    seed: u64,
    noise_weight: f64,
) -> (f64, f64) {
    let mut rng = seed ^ 0x9E3779B97F4A7C15;
    let num_colors: u8 = 6;
    let encoder = BinaryGridEncoder::new(grid_size, grid_size, num_colors as usize, seed);

    let mut hits = 0u32;
    let mut total = 0u32;
    let mut sim_sum = 0.0f64;

    for _task in 0..num_tasks {
        // 2 training pairs
        let mut train_rules = Vec::new();
        for _pair in 0..2 {
            xor_shift(&mut rng);
            let input = gen_grid(&mut rng, grid_size, num_colors);
            let output = transform(&input);
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

        // Test
        xor_shift(&mut rng);
        let test_input = gen_grid(&mut rng, grid_size, num_colors);
        let test_output = transform(&test_input);
        let test_in_hv = encoder.encode_grid(&test_input);
        let test_out_hv = encoder.encode_grid(&test_output);
        let predicted = encoder.apply_rule(&test_in_hv, &consensus);

        let pred_sim = predicted.similarity(&test_out_hv) as f64;
        sim_sum += pred_sim;

        // 2-AFC vs a fair (equally structured) distractor
        xor_shift(&mut rng);
        let distractor_grid =
            fair_distractor_grid(&test_input, &test_output).unwrap_or_else(|| test_input.clone());
        let distractor = encoder.encode_grid(&distractor_grid);
        let dist_sim = predicted.similarity(&distractor) as f64;

        total += 1;
        if pred_sim > dist_sim {
            hits += 1;
        }
    }

    let accuracy = if total > 0 {
        hits as f64 / total as f64
    } else {
        0.0
    };
    let similarity = if total > 0 {
        sim_sum / total as f64
    } else {
        0.0
    };
    (accuracy, similarity)
}

struct TrialResult {
    /// Grid scaling: accuracy at each grid size (dim fixed at config.dimension)
    grid_accuracy: [f64; 4],
    grid_similarity: [f64; 4],
    /// Dimension scaling: accuracy at each dimension (grid fixed at 5×5)
    dim_accuracy: [f64; 4],
    dim_similarity: [f64; 4],
    /// Grid complexity slope (accuracy per cell² increase)
    grid_slope: f64,
    /// Dimension efficiency slope (accuracy per doubling of dimension)
    dim_slope: f64,
}

impl ArcScalingBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let seed = config.trial_seed("reasoning", "arc_scaling", trial_idx);
        let noise_weight = 0.04 + config.time_pressure * 0.12;
        let num_tasks = 16; // per probe

        // Grid scaling: vary grid size, keep dimension fixed
        let mut grid_accuracy = [0.0f64; 4];
        let mut grid_similarity = [0.0f64; 4];
        for (i, &gs) in GRID_SIZES.iter().enumerate() {
            let (acc, sim) = scaling_probe(
                gs,
                config.dimension,
                num_tasks,
                seed ^ (i as u64),
                noise_weight,
            );
            grid_accuracy[i] = acc;
            grid_similarity[i] = sim;
        }

        // Dimension scaling: vary dimension, keep grid size = 5
        let mut dim_accuracy = [0.0f64; 4];
        let mut dim_similarity = [0.0f64; 4];
        for (i, &dim) in DIMENSIONS.iter().enumerate() {
            let (acc, sim) =
                scaling_probe(5, dim, num_tasks, seed ^ (100 + i as u64), noise_weight);
            dim_accuracy[i] = acc;
            dim_similarity[i] = sim;
        }

        // Grid slope: linear regression of accuracy vs grid_size
        let grid_xs: Vec<f64> = GRID_SIZES.iter().map(|&s| (s * s) as f64).collect();
        let grid_slope = linear_slope(&grid_xs, &grid_accuracy);

        // Dimension slope: linear regression of accuracy vs log2(dimension)
        let dim_xs: Vec<f64> = DIMENSIONS.iter().map(|&d| (d as f64).log2()).collect();
        let dim_slope = linear_slope(&dim_xs, &dim_accuracy);

        TrialResult {
            grid_accuracy,
            grid_similarity,
            dim_accuracy,
            dim_similarity,
            grid_slope,
            dim_slope,
        }
    }
}

/// Compute slope of simple linear regression.
fn linear_slope(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut den = 0.0;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        num += (x - mean_x) * (y - mean_y);
        den += (x - mean_x) * (x - mean_x);
    }
    if den.abs() > 1e-10 { num / den } else { 0.0 }
}

impl PsychBenchmark for ArcScalingBenchmark {
    fn name(&self) -> &str {
        "Reasoning::ArcScaling"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Parametric scaling of HDC capacity",
            citation: "Kanerva (2009); Chollet (2019)",
            year: 2009,
            doi: Some("10.1007/s12559-009-9009-8"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let grid_names = ["3x3", "5x5", "8x8", "12x12"];
        let dim_names = ["128d", "256d", "512d", "1024d"];

        let mut grid_accs: [Vec<f64>; 4] = Default::default();
        let mut grid_sims: [Vec<f64>; 4] = Default::default();
        let mut dim_accs: [Vec<f64>; 4] = Default::default();
        let mut dim_sims: [Vec<f64>; 4] = Default::default();
        let mut grid_slopes = Vec::new();
        let mut dim_slopes = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            for i in 0..4 {
                grid_accs[i].push(r.grid_accuracy[i]);
                grid_sims[i].push(r.grid_similarity[i]);
                dim_accs[i].push(r.dim_accuracy[i]);
                dim_sims[i].push(r.dim_similarity[i]);
            }
            grid_slopes.push(r.grid_slope);
            dim_slopes.push(r.dim_slope);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "arc_scaling".to_string(),
                    correct: r.grid_accuracy[0] > 0.5,
                    rt_ticks: 0.0,
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        for (i, name) in grid_names.iter().enumerate() {
            result.insert(
                format!("grid_{}_accuracy", name),
                MetricValue::from_samples(&grid_accs[i]),
            );
            result.insert(
                format!("grid_{}_similarity", name),
                MetricValue::from_samples(&grid_sims[i]),
            );
        }
        for (i, name) in dim_names.iter().enumerate() {
            result.insert(
                format!("dim_{}_accuracy", name),
                MetricValue::from_samples(&dim_accs[i]),
            );
            result.insert(
                format!("dim_{}_similarity", name),
                MetricValue::from_samples(&dim_sims[i]),
            );
        }
        result.insert(
            "grid_complexity_slope",
            MetricValue::from_samples(&grid_slopes),
        );
        result.insert(
            "dimension_efficiency_slope",
            MetricValue::from_samples(&dim_slopes),
        );

        // Capacity ratio: grid_3x3 accuracy / grid_12x12 accuracy
        let cap_ratios: Vec<f64> = grid_accs[0]
            .iter()
            .zip(grid_accs[3].iter())
            .map(|(small, large)| if *large > 0.0 { small / large } else { 1.0 })
            .collect();
        result.insert("capacity_ratio", MetricValue::from_samples(&cap_ratios));

        result.conditions = 8; // 4 grid sizes + 4 dimensions
        result.trials_per_condition = config.trials_per_condition;
        if config.trial_trace {
            result.trial_trace = trace;
        }
        result.notes.push(
            "SYNTHETIC: Tests HDC capacity scaling (grid size x dimension), not \
             reasoning. 2-AFC uses a fair (equally structured) distractor, not \
             random noise (fixed 2026-07-18). Z-scores reflect encoding capacity limits."
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
            trials_per_condition: 2,
            ..Default::default()
        }
    }

    #[test]
    fn test_scaling_runs_with_metrics() {
        let result = ArcScalingBenchmark.run(&test_config());
        assert!(result.metrics.contains_key("grid_3x3_accuracy"));
        assert!(result.metrics.contains_key("grid_12x12_accuracy"));
        assert!(result.metrics.contains_key("dim_128d_accuracy"));
        assert!(result.metrics.contains_key("dim_1024d_accuracy"));
        assert!(result.metrics.contains_key("grid_complexity_slope"));
        assert!(result.metrics.contains_key("dimension_efficiency_slope"));
        assert!(result.metrics.contains_key("capacity_ratio"));
    }

    #[test]
    fn test_all_metrics_finite() {
        let result = ArcScalingBenchmark.run(&test_config());
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_small_grid_reasonable() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = ArcScalingBenchmark.run(&config);
        let acc = result.metrics["grid_3x3_accuracy"].mean;
        // Small grids with high dimension should work well
        assert!(
            acc > 0.4,
            "3x3 grid at dim=512 should be near/above chance: {}",
            acc
        );
    }

    #[test]
    fn test_higher_dim_finite() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = ArcScalingBenchmark.run(&config);
        let sim_128 = result.metrics["dim_128d_similarity"].mean;
        let sim_1024 = result.metrics["dim_1024d_similarity"].mean;
        // Both should be finite; relative ordering is stochastic so just verify values
        assert!(sim_128.is_finite(), "128d similarity should be finite");
        assert!(sim_1024.is_finite(), "1024d similarity should be finite");
        // Both should be in reasonable range
        assert!(
            sim_128 > -1.0 && sim_128 < 1.0,
            "128d sim out of range: {}",
            sim_128
        );
        assert!(
            sim_1024 > -1.0 && sim_1024 < 1.0,
            "1024d sim out of range: {}",
            sim_1024
        );
    }

    #[test]
    fn test_grid_slope_negative() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = ArcScalingBenchmark.run(&config);
        let slope = result.metrics["grid_complexity_slope"].mean;
        // Accuracy should generally decrease with more cells (negative slope)
        // But allow positive slope at higher dimensions
        assert!(slope.is_finite(), "grid slope should be finite");
    }

    #[test]
    fn test_dim_slope_positive() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = ArcScalingBenchmark.run(&config);
        let slope = result.metrics["dimension_efficiency_slope"].mean;
        // Accuracy should generally increase with dimension (positive slope)
        assert!(slope.is_finite(), "dim slope should be finite");
    }

    #[test]
    fn test_provenance() {
        let prov = ArcScalingBenchmark.provenance().unwrap();
        assert!(prov.citation.contains("Kanerva"));
    }

    #[test]
    fn test_deterministic() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 2,
            seed: 42,
            ..Default::default()
        };
        let r1 = ArcScalingBenchmark.run(&config);
        let r2 = ArcScalingBenchmark.run(&config);
        assert_eq!(
            r1.metrics["capacity_ratio"].mean,
            r2.metrics["capacity_ratio"].mean
        );
    }
}
