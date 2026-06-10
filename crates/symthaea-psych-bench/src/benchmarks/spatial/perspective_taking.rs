// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Perspective Taking benchmark.
//!
//! Kozhevnikov & Hegarty (2001) showed that spatial perspective-taking error
//! increases with the angular disparity between the imagined and actual
//! viewpoints. Participants must mentally rotate their reference frame to
//! judge spatial relationships from another perspective.
//!
//! HDC implementation: A spatial layout is encoded from perspective 0 as a
//! set of object-location bindings. To represent perspective k (rotated by
//! angle θ), ALL location HVs are rotated by the same cyclic permutation shift
//! k, and a new scene is built: scene_k = Σ obj_i ⊕ permute(loc_i, k). To
//! query what location an object is at from perspective k, we unbind it from
//! scene_k: scene_k ⊕ obj_q ≈ permute(loc_q, k) (approximate, with bundling
//! noise from other objects). Error increases with number of objects (bundling
//! noise) and with angle (permutation distributes uniformly, so the angular
//! effect models RT rather than accuracy degradation directly).
//!
//! Human baselines (Kozhevnikov & Hegarty, 2001; Hegarty & Waller, 2004):
//! - perspective_accuracy: 0.75 (SD~0.12) -- mean accuracy across angles
//! - angular_error_slope: 0.60 (SD~0.10) -- normalized error increase with angle
//! - small_angle_accuracy: 0.90 (SD~0.05) -- accuracy at <90 degree rotations
//! - large_angle_accuracy: 0.60 (SD~0.15) -- accuracy at >90 degree rotations

use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::BinaryHV;

/// Perspective Taking benchmark (Kozhevnikov & Hegarty, 2001).
pub struct PerspectiveTakingBenchmark;

struct TrialResult {
    perspective_accuracy: f64,
    angular_error_slope: f64,
    small_angle_accuracy: f64,
    large_angle_accuracy: f64,
    mean_rt: f64,
}

impl PerspectiveTakingBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let seed = config.trial_seed("spatial", "perspective_taking", trial_idx);
        let mut rng = seed ^ 0x9E3779B97F4A7C15;

        let xor_shift = |s: &mut u64| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
        };

        let diff_model = difficulty_model_for(self.name());
        let tp_noise: f32 = (0.05 + config.time_pressure as f32 * 0.12)
            / diff_model.signal_multiplier(config.difficulty) as f32;
        let noise_degrade = config.effective_noise() as f32 * 0.30;

        // Create a spatial layout: 5 objects at 5 locations
        let n_objects = 5;
        let mut obj_hvs = Vec::with_capacity(n_objects);
        let mut loc_hvs = Vec::with_capacity(n_objects);

        for i in 0..n_objects {
            obj_hvs.push(BinaryHV::random(seed.wrapping_add(100 + i as u64 * 41)));
            loc_hvs.push(BinaryHV::random(seed.wrapping_add(500 + i as u64 * 67)));
        }

        // Encode perspective-0 scene: scene = Σ obj_i ⊕ loc_i (bundle of bindings).
        let scene_p0 = BinaryHV::bundle_safe(
            &obj_hvs
                .iter()
                .zip(loc_hvs.iter())
                .map(|(o, l)| o.bind(l))
                .collect::<Vec<_>>(),
        );

        // Test perspective rotations at 6 angles: 0, 60, 120, 180, 240, 300
        let n_angles = 6;
        let mut angle_data: Vec<(f64, f64)> = Vec::new(); // (angle_norm, accuracy)
        let mut rt_sum = 0.0f64;
        let mut rt_count = 0u32;

        for angle_idx in 0..n_angles {
            let angle_deg = angle_idx as f64 * 60.0;
            let angle_norm = angle_deg / 360.0;

            // Permutation shift for perspective rotation.
            let shift = ((angle_norm * BinaryHV::DIM as f64) as usize) % BinaryHV::DIM;

            // Build scene_k from perspective k: ALL locations are rotated by shift.
            // scene_k = Σ obj_i ⊕ permute(loc_i, k)
            // This is the correct construction: the scene as it looks from the new
            // perspective, where each object is still the same but its location HV
            // is rotated to match the new reference frame.
            let scene_k = if shift == 0 {
                scene_p0
            } else {
                BinaryHV::bundle_safe(
                    &obj_hvs
                        .iter()
                        .zip(loc_hvs.iter())
                        .map(|(o, l)| o.bind(&l.permute(shift)))
                        .collect::<Vec<_>>(),
                )
            };

            // Rotated location HVs (for comparison targets).
            let rotated_locs: Vec<BinaryHV> = loc_hvs
                .iter()
                .map(|l| if shift == 0 { *l } else { l.permute(shift) })
                .collect();

            let n_queries = 25;
            let mut correct = 0u32;

            for _query in 0..n_queries {
                xor_shift(&mut rng);
                let query_obj_idx = (rng as usize) % n_objects;

                // Query: what location is object q at from perspective k?
                // Unbind obj_q from scene_k: scene_k ⊕ obj_q ≈ permute(loc_q, k)
                // (approximate due to bundling noise from other n_objects-1 items)
                let recovered_loc = scene_k.bind(&obj_hvs[query_obj_idx]);

                // The correct answer is permute(loc_q, k).
                // Compare recovered_loc against all rotated location HVs.
                let mut best_sim = f32::MIN;
                let mut best_idx = 0usize;
                for (i, rloc) in rotated_locs.iter().enumerate() {
                    let sim = recovered_loc.similarity(rloc) * (1.0 - noise_degrade);
                    if sim > best_sim {
                        best_sim = sim;
                        best_idx = i;
                    }
                }

                // Add decision noise.
                xor_shift(&mut rng);
                let noise = ((rng % 10000) as f32 / 10000.0 - 0.5)
                    * tp_noise
                    * diff_model.temperature_multiplier(config.difficulty) as f32
                    * 0.3;

                // Correct if best match is the query object's rotated location
                // and noise doesn't flip the decision.
                if best_idx == query_obj_idx && (best_sim + noise > 0.0) {
                    correct += 1;
                }

                // RT: larger perspective shifts require more mental rotation.
                let base_rt = 3.0 + angle_norm * 4.0;
                let tp_speedup = config.time_pressure * 1.2;
                rt_sum += (base_rt - tp_speedup).max(1.0);
                rt_count += 1;
            }

            let accuracy = correct as f64 / n_queries as f64;
            angle_data.push((angle_norm, accuracy));
        }

        // Overall accuracy
        let n = angle_data.len() as f64;
        let perspective_accuracy: f64 = angle_data.iter().map(|d| d.1).sum::<f64>() / n;

        // Small angle accuracy (< 90 degrees: indices 0, 1)
        let small_angle_accuracy = if angle_data.len() >= 2 {
            (angle_data[0].1 + angle_data[1].1) / 2.0
        } else {
            angle_data[0].1
        };

        // Large angle accuracy (>= 120 degrees: indices 2, 3, 4, 5)
        let large_count = angle_data.len() - 2;
        let large_angle_accuracy: f64 =
            angle_data[2..].iter().map(|d| d.1).sum::<f64>() / large_count as f64;

        // Angular error slope: how much does accuracy decrease with angle?
        let mean_angle: f64 = angle_data.iter().map(|d| d.0).sum::<f64>() / n;
        let mean_acc: f64 = angle_data.iter().map(|d| d.1).sum::<f64>() / n;
        let cov: f64 = angle_data
            .iter()
            .map(|d| (d.0 - mean_angle) * (d.1 - mean_acc))
            .sum::<f64>();
        let var_angle: f64 = angle_data
            .iter()
            .map(|d| (d.0 - mean_angle).powi(2))
            .sum::<f64>();
        let raw_slope = if var_angle > 1e-10 {
            -(cov / var_angle)
        } else {
            0.0
        };
        let angular_error_slope = raw_slope.clamp(0.0, 1.0);

        let mean_rt = if rt_count > 0 {
            rt_sum / rt_count as f64
        } else {
            4.0
        };

        TrialResult {
            perspective_accuracy: perspective_accuracy.clamp(0.0, 1.0),
            angular_error_slope,
            small_angle_accuracy: small_angle_accuracy.clamp(0.0, 1.0),
            large_angle_accuracy: large_angle_accuracy.clamp(0.0, 1.0),
            mean_rt,
        }
    }
}

impl PsychBenchmark for PerspectiveTakingBenchmark {
    fn name(&self) -> &str {
        "Spatial::PerspectiveTaking"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Spatial Perspective Taking",
            citation: "Kozhevnikov & Hegarty (2001)",
            year: 2001,
            doi: Some("10.1037/0278-7393.27.1.235"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut accuracies = Vec::new();
        let mut slopes = Vec::new();
        let mut smalls = Vec::new();
        let mut larges = Vec::new();
        let mut rts = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            accuracies.push(r.perspective_accuracy);
            slopes.push(r.angular_error_slope);
            smalls.push(r.small_angle_accuracy);
            larges.push(r.large_angle_accuracy);
            rts.push(r.mean_rt);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "perspective_taking".to_string(),
                    correct: r.perspective_accuracy > 0.5,
                    rt_ticks: r.mean_rt,
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert(
            "perspective_accuracy",
            MetricValue::from_samples(&accuracies),
        );
        result.insert("angular_error_slope", MetricValue::from_samples(&slopes));
        result.insert("small_angle_accuracy", MetricValue::from_samples(&smalls));
        result.insert("large_angle_accuracy", MetricValue::from_samples(&larges));
        result.insert("rt_ticks", MetricValue::from_samples(&rts));

        result.conditions = 6; // number of angle levels
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
    fn test_perspective_taking_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = PerspectiveTakingBenchmark.run(&config);
        assert!(result.metrics.contains_key("perspective_accuracy"));
        assert!(result.metrics.contains_key("angular_error_slope"));
        assert!(result.metrics.contains_key("small_angle_accuracy"));
        assert!(result.metrics.contains_key("large_angle_accuracy"));
    }

    #[test]
    fn test_perspective_taking_finite_metrics() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = PerspectiveTakingBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_perspective_taking_metrics_bounded() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = PerspectiveTakingBenchmark.run(&config);
        let acc = result.metrics["perspective_accuracy"].mean;
        assert!(
            acc >= 0.0 && acc <= 1.0,
            "perspective_accuracy ({:.3}) out of bounds",
            acc
        );
        let slope = result.metrics["angular_error_slope"].mean;
        assert!(
            slope >= 0.0 && slope <= 1.0,
            "angular_error_slope ({:.3}) out of bounds",
            slope
        );
    }

    #[test]
    fn test_small_angle_easier_than_large() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 15,
            ..Default::default()
        };
        let result = PerspectiveTakingBenchmark.run(&config);
        let small = result.metrics["small_angle_accuracy"].mean;
        let large = result.metrics["large_angle_accuracy"].mean;
        assert!(
            small >= large - 0.05,
            "Small angle ({:.3}) should be >= large angle ({:.3})",
            small,
            large
        );
    }

    #[test]
    fn test_perspective_taking_provenance() {
        let prov = PerspectiveTakingBenchmark.provenance().unwrap();
        assert_eq!(prov.year, 2001);
        assert!(prov.citation.contains("Kozhevnikov"));
    }
}
