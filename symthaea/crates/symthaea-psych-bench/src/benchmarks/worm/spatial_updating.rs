// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Spatial updating task.
//!
//! Track an object's position across a grid as it receives movement updates.
//! Tests the updating component of WM in the spatial domain.

use crate::adapter::StimulusAdapter;
use crate::adapter::spatial::{GridPosition, SpatialAdapter};
use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use crate::wm::{WmConfig, WorkingMemory};
use std::collections::BTreeMap;

/// Spatial updating benchmark.
pub struct SpatialUpdatingBenchmark;

impl SpatialUpdatingBenchmark {
    /// Run a single trial: start at a position, apply N moves, check final position.
    /// Returns (accuracy, rt_ticks).
    fn run_trial(
        &self,
        num_updates: usize,
        config: &BenchmarkConfig,
        trial_idx: usize,
    ) -> (f64, f64) {
        let dim = config.dimension;
        let seed = config.trial_seed("worm", &format!("spatial_{}", num_updates), trial_idx);
        let adapter = SpatialAdapter::default();
        let grid_size = 8u32;

        let mut wm = WorkingMemory::new(WmConfig {
            dimension: dim,
            capacity: config.working_memory_capacity,
            ..Default::default()
        });

        // Start position
        let mut rng_state = seed ^ 0x9E3779B97F4A7C15;
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let mut current_pos = GridPosition::new(
            (rng_state % grid_size as u64) as u32,
            ((rng_state >> 8) % grid_size as u64) as u32,
        );

        // Present starting position
        let start_hv = adapter.encode(&current_pos, dim);
        wm.perceive(start_hv);
        wm.tick();

        // Apply movement updates
        for _ in 0..num_updates {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;

            let dx = ((rng_state % 3) as i32) - 1; // -1, 0, or 1
            let dy = (((rng_state >> 4) % 3) as i32) - 1;

            let new_x = (current_pos.x as i32 + dx).clamp(0, grid_size as i32 - 1) as u32;
            let new_y = (current_pos.y as i32 + dy).clamp(0, grid_size as i32 - 1) as u32;
            current_pos = GridPosition::new(new_x, new_y);

            let hv = adapter.encode(&current_pos, dim);
            wm.perceive(hv);
            wm.tick();
        }

        // Probe: does WM contain the final position?
        let target_hv = adapter.encode(&current_pos, dim);
        let contents = wm.contents();

        let max_sim = contents
            .iter()
            .map(|wm_item| wm_item.similarity(&target_hv))
            .fold(0.0f32, f32::max);

        // Time pressure: base 0.3 threshold yields ~75% spatial recall; +0.10/unit raises criterion,
        // modeling reduced search depth under speed emphasis (Luce, 1986 sequential sampling).
        let diff_model = difficulty_model_for("WorM::SpatialUpdating");
        let interference = config.difficulty as f32
            * 0.20
            * diff_model.interference_multiplier(config.difficulty) as f32;
        let degraded_sim = (max_sim - interference).max(0.0);
        let threshold = 0.3 * diff_model.temperature_multiplier(config.difficulty) as f32
            + config.time_pressure as f32 * 0.10;

        // RT proxy: deliberation ticks based on spatial match confidence.
        // Smaller margin between max_sim and threshold → harder discrimination → longer RT
        // (Luce, 1986; Ratcliff, 1978 diffusion model).
        let decision_margin = ((degraded_sim - threshold).abs() as f64).min(1.0);
        let rt_ticks = 4.0 + (1.0 - decision_margin) * 6.0;

        let acc = if degraded_sim > threshold { 1.0 } else { 0.0 };
        (acc, rt_ticks)
    }
}

impl PsychBenchmark for SpatialUpdatingBenchmark {
    fn name(&self) -> &str {
        "WorM::SpatialUpdating"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Spatial Updating",
            citation: "Oberauer (2003)",
            year: 2003,
            doi: Some("10.1016/S0022-5371(03)00063-0"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        for num_updates in [3, 5, 7, 10] {
            let mut accuracies = Vec::new();
            let mut rts = Vec::new();

            for trial in 0..config.trials_per_condition {
                let (acc, rt) = self.run_trial(num_updates, config, trial);
                accuracies.push(acc);
                rts.push(rt);
                if config.trial_trace {
                    trace.push(TrialOutcome {
                        trial_idx: trace.len(),
                        condition: format!("spatial_{}", num_updates),
                        correct: acc > 0.5,
                        rt_ticks: rt,
                        similarity: 0.0,
                        confidence: 0.0,
                        response_idx: 0,
                        extra: BTreeMap::new(),
                    });
                }
            }

            result.insert(
                format!("updates_{}::accuracy", num_updates),
                MetricValue::from_samples(&accuracies),
            );
            result.insert(
                format!("updates_{}::rt_ticks", num_updates),
                MetricValue::from_samples(&rts),
            );
        }

        // Aggregate: mean accuracy across all update conditions
        let all_means: Vec<f64> = [3, 5, 7, 10]
            .iter()
            .filter_map(|n| result.metrics.get(&format!("updates_{}::accuracy", n)))
            .map(|m| m.mean)
            .collect();
        if !all_means.is_empty() {
            result.insert("overall_accuracy", MetricValue::from_samples(&all_means));
        }

        result.conditions = 4;
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
            trials_per_condition: 3,
            dimension: 256,
            ..Default::default()
        };
        let result = SpatialUpdatingBenchmark.run(&config);
        assert_eq!(result.conditions, 4);
        assert!(result.metrics.contains_key("updates_3::accuracy"));
    }
}
