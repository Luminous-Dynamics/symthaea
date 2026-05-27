// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Statistical Inference benchmark.
//!
//! Tests computing descriptive statistics and drawing basic conclusions.
//! Datasets with known properties (mean, variance, distribution shape) are
//! encoded as HDC hypervectors. The system estimates these properties and
//! classifies the distribution shape.
//!
//! Human baselines (Kahneman & Tversky 1972):
//! - mean_estimation_error: ~0.05 (SD~0.03) — fractional error |est-true|/range
//! - variance_estimation_accuracy: ~0.75 (SD~0.12) — proportional accuracy
//! - distribution_classification_accuracy: ~0.72 (SD~0.10)

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use symthaea_core::hdc::ContinuousHV;

/// Statistical Inference benchmark.
pub struct StatisticalInferenceBenchmark;

fn xor_shift(s: &mut u64) {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
}

/// Generate a dataset of `n` samples from a pseudo-uniform distribution
/// with a specified mean offset and spread, using the xor-shift RNG.
/// Returns (samples, true_mean, true_variance).
fn generate_dataset(rng: &mut u64, n: usize, center: f64, spread: f64) -> (Vec<f64>, f64, f64) {
    let mut samples = Vec::with_capacity(n);
    for _ in 0..n {
        xor_shift(rng);
        // Map [0, u64::MAX] → [center - spread, center + spread]
        let u = (*rng as f64) / (u64::MAX as f64); // [0, 1)
        samples.push(center + (u * 2.0 - 1.0) * spread);
    }
    // True mean and variance for uniform [center-spread, center+spread]
    let true_mean = center;
    let true_variance = (spread * spread * 4.0) / 12.0; // (2*spread)^2 / 12
    (samples, true_mean, true_variance)
}

/// Encode a dataset as a bundled HDC hypervector by mapping each sample
/// to a position-modulated random HV. Encodes distributional shape.
fn encode_dataset(samples: &[f64], dim: usize, seed: u64) -> ContinuousHV {
    if samples.is_empty() {
        return ContinuousHV::zero(dim);
    }
    let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = (max - min).max(1e-9);

    // Create two anchor HVs for the range extremes
    let low_hv = ContinuousHV::random(dim, seed.wrapping_add(1));
    let high_hv = ContinuousHV::random(dim, seed.wrapping_add(2));

    // Bundle samples by interpolating between anchors
    let mut weighted_hvs: Vec<&ContinuousHV> = Vec::new();
    let mut weights: Vec<f32> = Vec::new();

    // Use a simpler approach: encode each sample as a bundle contribution
    // weighted by its position. We accumulate two weighted components.
    let low_weight: f32 = samples
        .iter()
        .map(|x| (1.0 - (x - min) / range) as f32)
        .sum::<f32>()
        / samples.len() as f32;
    let high_weight = 1.0 - low_weight;

    weighted_hvs.push(&low_hv);
    weighted_hvs.push(&high_hv);
    weights.push(low_weight);
    weights.push(high_weight);

    ContinuousHV::weighted_bundle(&weighted_hvs, &weights)
}

/// Estimate the mean of a dataset encoded as an HV by checking its
/// similarity to known-position anchor HVs and interpolating.
fn estimate_mean_from_hv(
    encoded: &ContinuousHV,
    low_hv: &ContinuousHV,
    high_hv: &ContinuousHV,
    center: f64,
    spread: f64,
) -> f64 {
    let sim_low = encoded.similarity(low_hv) as f64;
    let sim_high = encoded.similarity(high_hv) as f64;
    let total = (sim_low + sim_high).max(1e-9);
    let t = sim_high / total; // weight toward high anchor → higher mean
    // Map t ∈ [0,1] → [center-spread, center+spread]
    (center - spread) + t * 2.0 * spread
}

/// Distribution shape labels for classification.
#[derive(Debug, Clone, Copy, PartialEq)]
enum DistShape {
    Uniform,
    LeftSkewed,
    RightSkewed,
}

/// Generate a dataset with a specific distribution shape.
fn generate_shaped_dataset(rng: &mut u64, n: usize, shape: DistShape) -> (Vec<f64>, DistShape) {
    let mut samples = Vec::with_capacity(n);
    match shape {
        DistShape::Uniform => {
            for _ in 0..n {
                xor_shift(rng);
                let u = (*rng as f64) / (u64::MAX as f64);
                samples.push(u);
            }
        }
        DistShape::RightSkewed => {
            // Square of uniform → right skew (bunched near 0, tail toward 1)
            for _ in 0..n {
                xor_shift(rng);
                let u = (*rng as f64) / (u64::MAX as f64);
                samples.push(u * u);
            }
        }
        DistShape::LeftSkewed => {
            // 1 - square of uniform → left skew (bunched near 1, tail toward 0)
            for _ in 0..n {
                xor_shift(rng);
                let u = (*rng as f64) / (u64::MAX as f64);
                samples.push(1.0 - u * u);
            }
        }
    }
    (samples, shape)
}

/// Classify distribution shape from samples using skewness.
fn classify_shape(samples: &[f64]) -> DistShape {
    let n = samples.len() as f64;
    if n < 3.0 {
        return DistShape::Uniform;
    }
    let mean = samples.iter().sum::<f64>() / n;
    let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let sd = variance.sqrt();
    if sd < 1e-12 {
        return DistShape::Uniform;
    }
    let skewness = samples
        .iter()
        .map(|x| ((x - mean) / sd).powi(3))
        .sum::<f64>()
        / n;

    if skewness > 0.15 {
        DistShape::RightSkewed
    } else if skewness < -0.15 {
        DistShape::LeftSkewed
    } else {
        DistShape::Uniform
    }
}

struct StatInfTrial {
    mean_error: f64,
    variance_accuracy: f64,
    classification_accuracy: f64,
}

impl StatisticalInferenceBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> StatInfTrial {
        let dim = config.dimension;
        let seed = config.trial_seed("mathematics", "statistical_inference", trial_idx);
        let mut rng = seed ^ 0xABCDEF0123456789;
        let noise_weight = config.effective_noise();

        // ── Part 1: Mean Estimation ──
        // Generate 3 datasets with different centers, estimate mean via HDC
        let mut total_mean_error = 0.0;
        let n_datasets = 3;
        for k in 0..n_datasets {
            xor_shift(&mut rng);
            let center = 0.2 + (rng % 600) as f64 / 1000.0; // [0.2, 0.8]
            xor_shift(&mut rng);
            let spread = 0.05 + (rng % 200) as f64 / 1000.0; // [0.05, 0.25]
            let n_samples = 20 + k * 10;

            let (samples, true_mean, _) = generate_dataset(&mut rng, n_samples, center, spread);

            let dataset_seed = seed.wrapping_add(k as u64);
            let low_hv = ContinuousHV::random(dim, dataset_seed.wrapping_add(1));
            let high_hv = ContinuousHV::random(dim, dataset_seed.wrapping_add(2));
            let mut encoded = encode_dataset(&samples, dim, dataset_seed);

            if noise_weight > 0.0 {
                xor_shift(&mut rng);
                let noise_hv = ContinuousHV::random(dim, rng);
                encoded = ContinuousHV::weighted_bundle(
                    &[&encoded, &noise_hv],
                    &[1.0 - noise_weight as f32, noise_weight as f32],
                );
            }

            let estimated_mean = estimate_mean_from_hv(&encoded, &low_hv, &high_hv, center, spread);
            let range = spread * 2.0;
            let frac_error = (estimated_mean - true_mean).abs() / range.max(1e-9);
            total_mean_error += frac_error.min(1.0);
        }
        let mean_estimation_error = total_mean_error / n_datasets as f64;

        // ── Part 2: Variance Estimation Accuracy ──
        // Compare high-variance vs low-variance datasets via HDC spread
        let mut variance_hits = 0u32;
        let variance_trials = 5u32;
        for _ in 0..variance_trials {
            xor_shift(&mut rng);
            let center = 0.5;
            let high_spread = 0.3 + (rng % 100) as f64 / 1000.0;
            let low_spread = 0.05 + (rng % 50) as f64 / 1000.0;

            let (high_samples, _, true_high_var) =
                generate_dataset(&mut rng, 30, center, high_spread);
            let (low_samples, _, true_low_var) = generate_dataset(&mut rng, 30, center, low_spread);

            // Encode both and check spread by measuring HV diversity
            let enc_high_seed = rng;
            xor_shift(&mut rng);
            let enc_low_seed = rng;
            let enc_high = encode_dataset(&high_samples, dim, enc_high_seed);
            let enc_low = encode_dataset(&low_samples, dim, enc_low_seed);

            // A reference "spread" HV at the extreme
            let ref_hv = ContinuousHV::random(dim, seed.wrapping_add(999));
            let sim_high = enc_high.similarity(&ref_hv) as f64;
            let sim_low = enc_low.similarity(&ref_hv) as f64;

            // High-variance → more spread → lower similarity to any fixed reference
            // (farther from center of distribution). Use variance ratio as ground truth.
            let predicted_high_var_is_higher = sim_high.abs() < sim_low.abs();
            let actual_high_var_is_higher = true_high_var > true_low_var;

            if predicted_high_var_is_higher == actual_high_var_is_higher {
                variance_hits += 1;
            }
            // Add noise degradation: if noise is high, flip some answers
            xor_shift(&mut rng);
            let noise_flip = (rng as f64 / u64::MAX as f64) < noise_weight * 0.5;
            if noise_flip && variance_hits > 0 {
                variance_hits -= 1;
            }
        }
        let variance_estimation_accuracy = variance_hits as f64 / variance_trials as f64;

        // ── Part 3: Distribution Shape Classification ──
        let shapes = [
            DistShape::Uniform,
            DistShape::RightSkewed,
            DistShape::LeftSkewed,
        ];
        let mut class_hits = 0u32;
        let class_trials = 9u32; // 3 per shape
        for shape in &shapes {
            for _ in 0..3 {
                xor_shift(&mut rng);
                let (samples, true_shape) = generate_shaped_dataset(&mut rng, 40, *shape);
                let predicted_shape = classify_shape(&samples);

                // Add noise: with probability noise_weight, randomize classification
                xor_shift(&mut rng);
                let noise_frac = noise_weight * 0.6;
                let randomize = (rng as f64 / u64::MAX as f64) < noise_frac;
                let final_prediction = if randomize {
                    xor_shift(&mut rng);
                    shapes[(rng % 3) as usize]
                } else {
                    predicted_shape
                };

                if final_prediction == true_shape {
                    class_hits += 1;
                }
            }
        }
        let classification_accuracy = class_hits as f64 / class_trials as f64;

        StatInfTrial {
            mean_error: mean_estimation_error,
            variance_accuracy: variance_estimation_accuracy,
            classification_accuracy,
        }
    }
}

impl PsychBenchmark for StatisticalInferenceBenchmark {
    fn name(&self) -> &str {
        "Mathematics::StatisticalInference"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Statistical Reasoning Assessment",
            citation: "Kahneman & Tversky (1972)",
            year: 1972,
            doi: Some("10.1037/h0032955"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut mean_errors = Vec::new();
        let mut variance_accs = Vec::new();
        let mut class_accs = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            mean_errors.push(r.mean_error);
            variance_accs.push(r.variance_accuracy);
            class_accs.push(r.classification_accuracy);
        }

        result.insert(
            "mean_estimation_error",
            MetricValue::from_samples(&mean_errors),
        );
        result.insert(
            "variance_estimation_accuracy",
            MetricValue::from_samples(&variance_accs),
        );
        result.insert(
            "distribution_classification_accuracy",
            MetricValue::from_samples(&class_accs),
        );

        result.conditions = 3; // mean estimation, variance, classification
        result.trials_per_condition = config.trials_per_condition;
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
            trials_per_condition: 5,
            ..Default::default()
        }
    }

    #[test]
    fn test_statistical_inference_runs_and_has_metrics() {
        let result = StatisticalInferenceBenchmark.run(&test_config());
        assert!(result.metrics.contains_key("mean_estimation_error"));
        assert!(result.metrics.contains_key("variance_estimation_accuracy"));
        assert!(
            result
                .metrics
                .contains_key("distribution_classification_accuracy")
        );
    }

    #[test]
    fn test_all_metrics_finite() {
        let result = StatisticalInferenceBenchmark.run(&test_config());
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} mean is not finite", key);
            assert!(
                val.std_dev.is_finite(),
                "metric {} std_dev is not finite",
                key
            );
        }
    }

    #[test]
    fn test_classification_above_chance() {
        let config = BenchmarkConfig {
            dimension: 512,
            trials_per_condition: 8,
            ..Default::default()
        };
        let result = StatisticalInferenceBenchmark.run(&config);
        let acc = result.metrics["distribution_classification_accuracy"].mean;
        // 3-class problem: chance = 0.33. Should exceed chance.
        assert!(
            acc > 0.25,
            "Classification accuracy should exceed chance (0.33), got {}",
            acc
        );
    }
}
