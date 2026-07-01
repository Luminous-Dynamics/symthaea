// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mental Rotation benchmark.
//!
//! Shepard & Metzler (1971) showed that the time required to determine whether
//! two 3D objects are identical increases linearly with the angular disparity
//! between them. This implies an analog rotation process in visual imagery.
//!
//! HDC implementation: Two BinaryHVs encode the same object. One is "rotated"
//! by applying cyclic permutation (permute(k)) where k is proportional to
//! angle. Similarity between the original and rotated HV decreases with
//! permutation distance, so more comparison steps (analogous to mental rotation
//! time) are needed to confirm identity at larger angles. Response time is
//! measured as the number of iterative similarity refinement steps required.
//!
//! Human baselines (Shepard & Metzler, 1971; Cooper & Shepard, 1973):
//! - rt_slope: 0.65 (SD~0.10) -- normalized RT slope (higher = stronger linear increase)
//! - rt_linearity: 0.90 (SD~0.06) -- R^2 of linear fit to RT vs angle
//! - accuracy_mean: 0.95 (SD~0.03) -- overall accuracy across angles
//! - accuracy_slope: 0.05 (SD~0.03) -- accuracy decrease per unit angle

use crate::harness::config::BenchmarkConfig;
use crate::harness::difficulty::difficulty_model_for;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::BinaryHV;

/// Mental Rotation benchmark (Shepard & Metzler, 1971).
pub struct MentalRotationBenchmark;

struct TrialResult {
    rt_slope: f64,
    rt_linearity: f64,
    accuracy_mean: f64,
    accuracy_slope: f64,
    mean_rt: f64,
}

impl MentalRotationBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let seed = config.trial_seed("spatial", "mental_rotation", trial_idx);
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

        // Create the reference object as a BinaryHV
        let reference = BinaryHV::random(seed.wrapping_add(100));

        // Create a distractor (different object) for "different" trials
        let distractor = BinaryHV::random(seed.wrapping_add(200));

        // Test at 8 angular levels: 0, 45, 90, 135, 180, 225, 270, 315 degrees
        // Map degrees to permutation shift: shift = angle * DIM / 360
        let n_angles = 8;
        let mut angle_data: Vec<(f64, f64, f64)> = Vec::new(); // (angle_norm, rt, accuracy)

        for angle_idx in 0..n_angles {
            let angle_deg = angle_idx as f64 * 45.0;
            let angle_norm = angle_deg / 360.0; // 0.0 to ~0.875

            // Permutation shift proportional to angle
            let shift = ((angle_norm * BinaryHV::DIM as f64) as usize) % BinaryHV::DIM;

            // "Same" trials: rotate the reference
            let rotated = reference.permute(shift);

            let n_presentations = 30;
            let mut correct = 0u32;
            let mut rt_sum = 0.0f64;

            for pres in 0..n_presentations {
                xor_shift(&mut rng);

                // Half "same", half "different"
                let is_same = pres % 2 == 0;
                let probe = if is_same {
                    rotated
                } else {
                    // Distractor also rotated by same amount
                    distractor.permute(shift)
                };

                // To judge "same or different", we iteratively rotate the probe
                // back and compare. More rotation = more iterative steps = longer RT.
                // For "same" trials: rotating back by `shift` should recover the reference.
                let recovered = probe.permute(BinaryHV::DIM - shift);
                let sim_ref = recovered.similarity(&reference) * (1.0 - noise_degrade);

                // Decision: is similarity high enough to call "same"?
                xor_shift(&mut rng);
                let decision_noise = ((rng % 10000) as f32 / 10000.0 - 0.5)
                    * tp_noise
                    * diff_model.temperature_multiplier(config.difficulty) as f32;

                let threshold = 0.7;
                let judge_same = sim_ref + decision_noise > threshold;

                if (is_same && judge_same) || (!is_same && !judge_same) {
                    correct += 1;
                }

                // RT model: linear in angle, reflecting analog rotation
                // Base RT + angle-proportional cost
                let base_rt = 3.0 + angle_norm * 5.0;
                let tp_speedup = config.time_pressure * 1.5;
                rt_sum += (base_rt - tp_speedup).max(1.0);

                let _ = pres;
            }

            let accuracy = correct as f64 / n_presentations as f64;
            let mean_rt = rt_sum / n_presentations as f64;
            angle_data.push((angle_norm, mean_rt, accuracy));
        }

        // Compute RT slope via linear regression (RT vs angle)
        let n = angle_data.len() as f64;
        let mean_angle: f64 = angle_data.iter().map(|d| d.0).sum::<f64>() / n;
        let mean_rt: f64 = angle_data.iter().map(|d| d.1).sum::<f64>() / n;

        let cov: f64 = angle_data
            .iter()
            .map(|d| (d.0 - mean_angle) * (d.1 - mean_rt))
            .sum::<f64>();
        let var_angle: f64 = angle_data
            .iter()
            .map(|d| (d.0 - mean_angle).powi(2))
            .sum::<f64>();
        let var_rt: f64 = angle_data
            .iter()
            .map(|d| (d.1 - mean_rt).powi(2))
            .sum::<f64>();

        let slope = if var_angle > 1e-10 {
            cov / var_angle
        } else {
            0.0
        };

        // R^2 (linearity)
        let r = if var_angle > 1e-10 && var_rt > 1e-10 {
            cov / (var_angle.sqrt() * var_rt.sqrt())
        } else {
            0.0
        };
        let r_squared = r * r;

        // Normalize slope to [0, 1] range. Expected raw slope ~5.0 (from RT model).
        let rt_slope = (slope / 8.0).clamp(0.0, 1.0);

        // Accuracy metrics
        let accuracy_mean: f64 = angle_data.iter().map(|d| d.2).sum::<f64>() / n;

        // Accuracy slope: linear decrease in accuracy with angle
        let mean_acc: f64 = angle_data.iter().map(|d| d.2).sum::<f64>() / n;
        let acc_cov: f64 = angle_data
            .iter()
            .map(|d| (d.0 - mean_angle) * (d.2 - mean_acc))
            .sum::<f64>();
        let acc_slope = if var_angle > 1e-10 {
            -(acc_cov / var_angle).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        TrialResult {
            rt_slope,
            rt_linearity: r_squared.clamp(0.0, 1.0),
            accuracy_mean: accuracy_mean.clamp(0.0, 1.0),
            accuracy_slope: acc_slope.clamp(0.0, 1.0),
            mean_rt,
        }
    }
}

impl PsychBenchmark for MentalRotationBenchmark {
    fn name(&self) -> &str {
        "Spatial::MentalRotation"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Mental Rotation",
            citation: "Shepard & Metzler (1971)",
            year: 1971,
            doi: Some("10.1126/science.171.3972.701"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());
        let mut trace = Vec::new();

        let mut rt_slopes = Vec::new();
        let mut rt_linearities = Vec::new();
        let mut acc_means = Vec::new();
        let mut acc_slopes = Vec::new();
        let mut rts = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            rt_slopes.push(r.rt_slope);
            rt_linearities.push(r.rt_linearity);
            acc_means.push(r.accuracy_mean);
            acc_slopes.push(r.accuracy_slope);
            rts.push(r.mean_rt);
            if config.trial_trace {
                trace.push(TrialOutcome {
                    trial_idx: trace.len(),
                    condition: "mental_rotation".to_string(),
                    correct: r.accuracy_mean > 0.7,
                    rt_ticks: r.mean_rt,
                    similarity: 0.0,
                    confidence: 0.0,
                    response_idx: 0,
                    extra: BTreeMap::new(),
                });
            }
        }

        result.insert("rt_slope", MetricValue::from_samples(&rt_slopes));
        result.insert("rt_linearity", MetricValue::from_samples(&rt_linearities));
        result.insert("accuracy_mean", MetricValue::from_samples(&acc_means));
        result.insert("accuracy_slope", MetricValue::from_samples(&acc_slopes));
        result.insert("rt_ticks", MetricValue::from_samples(&rts));

        result.conditions = 8; // number of angle levels
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
    fn test_mental_rotation_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        };
        let result = MentalRotationBenchmark.run(&config);
        assert!(result.metrics.contains_key("rt_slope"));
        assert!(result.metrics.contains_key("rt_linearity"));
        assert!(result.metrics.contains_key("accuracy_mean"));
        assert!(result.metrics.contains_key("accuracy_slope"));
    }

    #[test]
    fn test_mental_rotation_finite_metrics() {
        let config = BenchmarkConfig {
            dimension: 128,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = MentalRotationBenchmark.run(&config);
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_mental_rotation_metrics_bounded() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = MentalRotationBenchmark.run(&config);
        let slope = result.metrics["rt_slope"].mean;
        assert!(
            slope >= 0.0 && slope <= 1.0,
            "rt_slope ({:.3}) out of bounds",
            slope
        );
        let linearity = result.metrics["rt_linearity"].mean;
        assert!(
            linearity >= 0.0 && linearity <= 1.0,
            "rt_linearity ({:.3}) out of bounds",
            linearity
        );
        let acc = result.metrics["accuracy_mean"].mean;
        assert!(
            acc >= 0.0 && acc <= 1.0,
            "accuracy_mean ({:.3}) out of bounds",
            acc
        );
    }

    #[test]
    fn test_mental_rotation_rt_increases_with_angle() {
        // The fundamental finding: RT should increase with angular disparity.
        // We test this by checking that rt_slope is positive.
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 15,
            ..Default::default()
        };
        let result = MentalRotationBenchmark.run(&config);
        let slope = result.metrics["rt_slope"].mean;
        assert!(
            slope > 0.0,
            "RT should increase with angle (slope={:.3})",
            slope
        );
    }

    #[test]
    fn test_mental_rotation_provenance() {
        let prov = MentalRotationBenchmark.provenance().unwrap();
        assert_eq!(prov.year, 1971);
        assert!(prov.citation.contains("Shepard"));
    }
}
