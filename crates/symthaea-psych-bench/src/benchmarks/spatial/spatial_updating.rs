// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Spatial Path Updating benchmark.
//!
//! Morrow et al. (1989) demonstrated that spatial representations are updated
//! during narrative comprehension: participants track their position through
//! described environments, and accuracy decreases as paths become more complex
//! (more turns, more steps).
//!
//! HDC implementation: Grid locations are encoded compositionally using two
//! basis HVs (pos_x, pos_y): location(x,y) = permute(pos_x, x) ⊕ permute(pos_y, y).
//! This means locations that differ by a known displacement are related by a
//! fixed HDC transformation, so walking a path via HDC operations can be compared
//! against the target location encoded the same way. Accuracy drops with path
//! length/complexity because cumulative XOR binding noise degrades the match.
//!
//! Human baselines (Morrow et al., 1989; Rieser, 1989):
//! - updating_accuracy: 0.85 (SD~0.08) -- mean accuracy across path lengths
//! - complexity_slope: 0.06 (SD~0.03) -- accuracy decrease per additional step
//! - simple_accuracy: 0.95 (SD~0.03) -- accuracy on 1-2 step paths
//! - complex_accuracy: 0.70 (SD~0.10) -- accuracy on 5+ step paths

use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::BinaryHV;

/// Spatial Path Updating benchmark (Morrow et al., 1989).
pub struct SpatialPathUpdatingBenchmark;

struct TrialResult {
    updating_accuracy: f64,
    complexity_slope: f64,
    simple_accuracy: f64,
    complex_accuracy: f64,
    mean_rt: f64,
}

impl SpatialPathUpdatingBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let seed = config.trial_seed("spatial", "spatial_updating", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        let diff_model = difficulty_model_for(self.name());
        let tp_noise: f32 = (0.05 + config.time_pressure as f32 * 0.10)
            / diff_model.signal_multiplier(config.difficulty) as f32;
        let noise_degrade = config.effective_noise() as f32 * 0.30;

        // Compositional grid coordinate system.
        // Each grid location (x, y) is encoded as:
        //   location(x, y) = permute(pos_x_basis, x) ⊕ permute(pos_y_basis, y)
        // This means locations are derived from coordinates, not random, so HDC
        // path arithmetic can be compared against the target location.
        let grid_size = 5usize;
        let pos_x_basis = BinaryHV::random(seed.wrapping_add(1000));
        let pos_y_basis = BinaryHV::random(seed.wrapping_add(2000));

        // Encode location from (x, y) coordinates using compositional binding.
        let encode_location = |x: usize, y: usize| -> BinaryHV {
            let px = pos_x_basis.permute(x + 1); // +1 so permute(0) != identity
            let py = pos_y_basis.permute(y + 1);
            px.bind(&py)
        };

        // Pre-build all grid locations using the compositional scheme.
        let mut locations = Vec::with_capacity(grid_size * grid_size);
        for y in 0..grid_size {
            for x in 0..grid_size {
                locations.push(encode_location(x, y));
            }
        }

        // Direction delta HVs: encode displacement from current to next cell.
        // direction_delta(dx, dy) = encode_location(cx+dx, cy+dy) ⊕ encode_location(cx, cy)
        // = permute(pos_x, cx+dx) ⊕ permute(pos_y, cy+dy) ⊕ permute(pos_x, cx) ⊕ permute(pos_y, cy)
        // The cx/cy terms cancel via XOR self-inverse when we apply it:
        //   location(cx+dx, cy+dy) = location(cx, cy) ⊕ direction_delta(dx, dy)  [approx]
        // This is only exact when there is no cross-coupling; in practice the
        // permute shifts are independent so the cancellation is clean.
        //
        // N=(0,-1), S=(0,+1), E=(+1,0), W=(-1,0) in grid coords (y increases southward)
        let dir_deltas: [(i32, i32); 4] = [(0, -1), (0, 1), (1, 0), (-1, 0)];

        // Pre-compute direction delta HVs from the center of the grid so they
        // capture the local displacement structure.
        let cx0 = grid_size / 2;
        let cy0 = grid_size / 2;
        let center_hv = encode_location(cx0, cy0);
        let dir_hvs: [BinaryHV; 4] = std::array::from_fn(|d| {
            let (dx, dy) = dir_deltas[d];
            let nx = (cx0 as i32 + dx).clamp(0, grid_size as i32 - 1) as usize;
            let ny = (cy0 as i32 + dy).clamp(0, grid_size as i32 - 1) as usize;
            let neighbor_hv = encode_location(nx, ny);
            // delta = neighbor ⊕ center (XOR self-inverse recovers neighbor when applied to center)
            neighbor_hv.bind(&center_hv)
        });

        // Test at different path complexities (1 to 7 steps)
        let max_steps = 7;
        let mut complexity_data: Vec<(f64, f64)> = Vec::new(); // (n_steps_norm, accuracy)
        let mut rt_sum = 0.0f64;
        let mut rt_count = 0u32;

        for n_steps in 1..=max_steps {
            let n_presentations = 25;
            let mut correct = 0u32;

            for _pres in 0..n_presentations {
                xor_shift(&mut rng);

                // Start at center of grid.
                let mut pos_x: i32 = (grid_size / 2) as i32;
                let mut pos_y: i32 = (grid_size / 2) as i32;

                // Walk the path: apply each direction delta to the current
                // position HV. Because location(x+dx, y+dy) ≈ location(x, y) ⊕ dir_delta,
                // we can compose steps as successive XOR applications.
                let mut current_hv = encode_location(pos_x as usize, pos_y as usize);

                for _step in 0..n_steps {
                    xor_shift(&mut rng);
                    let dir_idx = (rng % 4) as usize;

                    // Update ground-truth grid position (clamped to grid).
                    let (dx, dy) = dir_deltas[dir_idx];
                    pos_x = (pos_x + dx).clamp(0, grid_size as i32 - 1);
                    pos_y = (pos_y + dy).clamp(0, grid_size as i32 - 1);

                    // Apply the direction delta HV to advance the HDC position.
                    // current ⊕ dir_delta ≈ encode_location(new_x, new_y)
                    current_hv = current_hv.bind(&dir_hvs[dir_idx]);
                }

                // Ground-truth target index.
                let target_idx = pos_y as usize * grid_size + pos_x as usize;

                // Find the most similar precomputed location HV.
                let mut best_sim = f32::MIN;
                let mut best_idx = 0usize;
                for (i, loc) in locations.iter().enumerate() {
                    let sim = current_hv.similarity(loc) * (1.0 - noise_degrade);
                    if sim > best_sim {
                        best_sim = sim;
                        best_idx = i;
                    }
                }

                // Add decision noise.
                xor_shift(&mut rng);
                let noise = ((rng % 10000) as f32 / 10000.0 - 0.5)
                    * tp_noise
                    * diff_model.temperature_multiplier(config.difficulty) as f32;

                // Correct if the best-matching location is the target and noise
                // doesn't flip the decision.
                let margin = best_sim - {
                    // Second-best similarity for signal-to-noise check.
                    locations
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| *i != best_idx)
                        .map(|(_, loc)| current_hv.similarity(loc))
                        .fold(f32::MIN, f32::max)
                };
                if best_idx == target_idx && margin + noise > 0.0 {
                    correct += 1;
                }

                // RT: longer paths take more time.
                let base_rt = 2.0 + n_steps as f64 * 1.5;
                let tp_speedup = config.time_pressure * 1.0;
                rt_sum += (base_rt - tp_speedup).max(1.0);
                rt_count += 1;
            }

            let accuracy = correct as f64 / n_presentations as f64;
            let steps_norm = n_steps as f64 / max_steps as f64;
            complexity_data.push((steps_norm, accuracy));
        }

        // Overall accuracy
        let n = complexity_data.len() as f64;
        let updating_accuracy: f64 = complexity_data.iter().map(|d| d.1).sum::<f64>() / n;

        // Simple vs complex accuracy
        let simple_accuracy = if complexity_data.len() >= 2 {
            (complexity_data[0].1 + complexity_data[1].1) / 2.0
        } else {
            complexity_data[0].1
        };
        let complex_accuracy = if complexity_data.len() >= 2 {
            let n_complex = 2.min(complexity_data.len());
            complexity_data[complexity_data.len() - n_complex..]
                .iter()
                .map(|d| d.1)
                .sum::<f64>()
                / n_complex as f64
        } else {
            complexity_data.last().unwrap().1
        };

        // Complexity slope: accuracy decrease per unit normalized complexity
        let mean_steps: f64 = complexity_data.iter().map(|d| d.0).sum::<f64>() / n;
        let mean_acc: f64 = complexity_data.iter().map(|d| d.1).sum::<f64>() / n;
        let cov: f64 = complexity_data
            .iter()
            .map(|d| (d.0 - mean_steps) * (d.1 - mean_acc))
            .sum::<f64>();
        let var_steps: f64 = complexity_data
            .iter()
            .map(|d| (d.0 - mean_steps).powi(2))
            .sum::<f64>();
        let raw_slope = if var_steps > 1e-10 {
            -(cov / var_steps)
        } else {
            0.0
        };
        let complexity_slope = raw_slope.clamp(0.0, 1.0);

        let mean_rt = if rt_count > 0 {
            rt_sum / rt_count as f64
        } else {
            3.0
        };

        TrialResult {
            updating_accuracy: updating_accuracy.clamp(0.0, 1.0),
            complexity_slope,
            simple_accuracy: simple_accuracy.clamp(0.0, 1.0),
            complex_accuracy: complex_accuracy.clamp(0.0, 1.0),
            mean_rt,
        }
    }
}

impl PsychBenchmark for SpatialPathUpdatingBenchmark {
    fn name(&self) -> &str {
        "Spatial::PathUpdating"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Spatial Path Updating",
            citation: "Morrow et al. (1989)",
            year: 1989,
            doi: Some("10.1037/0278-7393.15.5.786"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut accuracies = Vec::new();
        let mut slopes = Vec::new();
        let mut simples = Vec::new();
        let mut complexes = Vec::new();
        let mut rts = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            accuracies.push(r.updating_accuracy);
            slopes.push(r.complexity_slope);
            simples.push(r.simple_accuracy);
            complexes.push(r.complex_accuracy);
            rts.push(r.mean_rt);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "path_updating".to_string(),
                    correct: r.updating_accuracy > 0.5,
                    rt_ticks: r.mean_rt,
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("updating_accuracy", MetricValue::from_samples(&accuracies));
        result.insert("complexity_slope", MetricValue::from_samples(&slopes));
        result.insert("simple_accuracy", MetricValue::from_samples(&simples));
        result.insert("complex_accuracy", MetricValue::from_samples(&complexes));
        result.insert("rt_ticks", MetricValue::from_samples(&rts));

        result.conditions = 7; // path lengths 1-7
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
    fn test_spatial_updating_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = SpatialPathUpdatingBenchmark.run(&config);
        assert!(result.metrics.contains_key("updating_accuracy"));
        assert!(result.metrics.contains_key("complexity_slope"));
        assert!(result.metrics.contains_key("simple_accuracy"));
        assert!(result.metrics.contains_key("complex_accuracy"));
    }

    #[test]
    fn test_spatial_updating_finite_metrics() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = SpatialPathUpdatingBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_spatial_updating_metrics_bounded() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = SpatialPathUpdatingBenchmark.run(&config);
        let acc = result.metrics["updating_accuracy"].mean;
        assert!(
            acc >= 0.0 && acc <= 1.0,
            "updating_accuracy ({:.3}) out of bounds",
            acc
        );
        let slope = result.metrics["complexity_slope"].mean;
        assert!(
            slope >= 0.0 && slope <= 1.0,
            "complexity_slope ({:.3}) out of bounds",
            slope
        );
    }

    #[test]
    fn test_simple_better_than_complex() {
        // Simple paths should be easier than complex ones
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 15,
            ..Default::default()
        };
        let result = SpatialPathUpdatingBenchmark.run(&config);
        let simple = result.metrics["simple_accuracy"].mean;
        let complex = result.metrics["complex_accuracy"].mean;
        assert!(
            simple >= complex - 0.05,
            "Simple ({:.3}) should be >= complex ({:.3})",
            simple,
            complex
        );
    }

    #[test]
    fn test_spatial_updating_provenance() {
        let prov = SpatialPathUpdatingBenchmark.provenance().unwrap();
        assert_eq!(prov.year, 1989);
        assert!(prov.citation.contains("Morrow"));
    }

    #[test]
    fn test_accuracy_values_meaningful() {
        // Verify that the HDC coordinate encoding produces genuine signal (not ~4% random)
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = SpatialPathUpdatingBenchmark.run(&config);
        let simple = result.metrics["simple_accuracy"].mean;
        let updating = result.metrics["updating_accuracy"].mean;
        // With compositional encoding, 1-step accuracy should be well above chance (1/25 = 0.04)
        assert!(
            simple > 0.3,
            "simple_accuracy ({:.3}) should be well above chance (0.04), compositional encoding must work",
            simple
        );
        assert!(
            updating > 0.1,
            "updating_accuracy ({:.3}) should be above chance (0.04)",
            updating
        );
        eprintln!(
            "PathUpdating: simple={:.3}, complex={:.3}, overall={:.3}",
            simple, result.metrics["complex_accuracy"].mean, updating,
        );
    }
}
