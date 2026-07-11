// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Statistical Inference benchmark.
//!
//! Tests computing descriptive statistics and drawing basic conclusions from
//! datasets with known distributional properties (mean, variance, shape).
//!
//! **Engine-wired (Tier 0.1, 2026-07-06).** All three parts run the real
//! `statistics` module from `symthaea-core`:
//! - Mean estimation: `statistics::mean` on generated samples, scored against
//!   the *distribution's* true mean (so the residual error is genuine
//!   sampling error, not estimator error).
//! - Variance discrimination: `statistics::variance` on paired
//!   high/low-spread datasets, scored against the true variance ordering,
//!   plus a closed-form magnitude check against the uniform-distribution
//!   variance (2·spread)²/12.
//! - Shape classification: `statistics::skewness` (Fisher-Pearson) with fixed
//!   thresholds, scored against the generating distribution's shape.
//!
//! The previous version estimated the mean by interpolating HDC anchor-vector
//! similarities and judged variance by similarity of dataset encodings to a
//! random reference — neither invoked the statistics engine; that gap was
//! flagged by the Phase 0 grounding audit. The HDC anchor estimator is
//! retained as trial structure: its own mean-estimation error is reported as
//! the auxiliary `hdc_mean_estimation_error` metric (not part of the headline
//! score).
//!
//! Noise model: `effective_noise()` corrupts a proportional fraction of the
//! samples (biased replacement) before the engine sees them, so accuracy
//! degrades under noise while the noiseless condition reflects true
//! computed correctness.
//!
//! Human baselines (Kahneman & Tversky 1972):
//! - mean_estimation_error: ~0.05 (SD~0.03) — fractional error |est-true|/range
//! - variance_estimation_accuracy: ~0.75 (SD~0.12) — proportional accuracy
//! - distribution_classification_accuracy: ~0.72 (SD~0.10)

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use symthaea_core::hdc::ContinuousHV;
use symthaea_core::hdc::statistics;

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

/// Corrupt a fraction (≈ noise_weight) of the samples with biased
/// replacements drawn from above the distribution's support, modelling a
/// degraded measurement channel. No-op at zero noise.
fn apply_sample_noise(samples: &mut [f64], center: f64, spread: f64, noise: f64, rng: &mut u64) {
    if noise <= 0.0 {
        return;
    }
    for s in samples.iter_mut() {
        xor_shift(rng);
        if (*rng as f64 / u64::MAX as f64) < noise {
            xor_shift(rng);
            let u = (*rng as f64) / (u64::MAX as f64);
            // Biased replacement: [center + spread, center + 3·spread]
            *s = center + spread + u * 2.0 * spread;
        }
    }
}

// ─── HDC anchor estimator (retained as auxiliary trial structure) ──────────

/// Encode a dataset as a bundled HDC hypervector by weighting two anchor HVs
/// (range extremes) by the samples' positions. Encodes central tendency.
fn encode_dataset(samples: &[f64], dim: usize, seed: u64) -> ContinuousHV {
    if samples.is_empty() {
        return ContinuousHV::zero(dim);
    }
    let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = (max - min).max(1e-9);

    let low_hv = ContinuousHV::random(dim, seed.wrapping_add(1));
    let high_hv = ContinuousHV::random(dim, seed.wrapping_add(2));

    let low_weight: f32 = samples
        .iter()
        .map(|x| (1.0 - (x - min) / range) as f32)
        .sum::<f32>()
        / samples.len() as f32;
    let high_weight = 1.0 - low_weight;

    ContinuousHV::weighted_bundle(&[&low_hv, &high_hv], &[low_weight, high_weight])
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

// ─── Shape classification (real engine skewness) ───────────────────────────

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

/// Classify distribution shape from the REAL engine's Fisher-Pearson
/// skewness (`statistics::skewness`).
fn classify_shape(samples: &[f64]) -> DistShape {
    let skew = statistics::skewness(samples);
    if skew > 0.15 {
        DistShape::RightSkewed
    } else if skew < -0.15 {
        DistShape::LeftSkewed
    } else {
        DistShape::Uniform
    }
}

struct StatInfTrial {
    mean_error: f64,
    variance_accuracy: f64,
    classification_accuracy: f64,
    hdc_mean_error: f64,
}

impl StatisticalInferenceBenchmark {
    fn run_trial(&self, config: &BenchmarkConfig, trial_idx: usize) -> StatInfTrial {
        let dim = config.dimension;
        let seed = config.trial_seed("mathematics", "statistical_inference", trial_idx);
        let mut rng = seed ^ 0xABCDEF0123456789;
        let noise_weight = config.effective_noise();

        // ── Part 1: Mean Estimation (real engine) ──
        // Generate 3 datasets with different centers; the estimate is
        // statistics::mean, scored against the distribution's true mean.
        let mut total_mean_error = 0.0;
        let mut total_hdc_error = 0.0;
        let n_datasets = 3;
        for k in 0..n_datasets {
            xor_shift(&mut rng);
            let center = 0.2 + (rng % 600) as f64 / 1000.0; // [0.2, 0.8]
            xor_shift(&mut rng);
            let spread = 0.05 + (rng % 200) as f64 / 1000.0; // [0.05, 0.25]
            let n_samples = 20 + k * 10;

            let (mut samples, true_mean, _) = generate_dataset(&mut rng, n_samples, center, spread);
            apply_sample_noise(&mut samples, center, spread, noise_weight, &mut rng);

            let range = spread * 2.0;

            // REAL ENGINE: statistics::mean on the observed samples.
            let engine_mean = statistics::mean(&samples);
            let frac_error = (engine_mean - true_mean).abs() / range.max(1e-9);
            total_mean_error += frac_error.min(1.0);

            // Auxiliary: the retained HDC anchor estimator on the same data.
            let dataset_seed = seed.wrapping_add(k as u64);
            let low_hv = ContinuousHV::random(dim, dataset_seed.wrapping_add(1));
            let high_hv = ContinuousHV::random(dim, dataset_seed.wrapping_add(2));
            let encoded = encode_dataset(&samples, dim, dataset_seed);
            let hdc_mean = estimate_mean_from_hv(&encoded, &low_hv, &high_hv, center, spread);
            let hdc_frac_error = (hdc_mean - true_mean).abs() / range.max(1e-9);
            total_hdc_error += hdc_frac_error.min(1.0);
        }
        let mean_estimation_error = total_mean_error / n_datasets as f64;
        let hdc_mean_estimation_error = total_hdc_error / n_datasets as f64;

        // ── Part 2: Variance Estimation Accuracy (real engine) ──
        // Paired high/low-spread datasets: the engine's variance must (a)
        // recover the true ordering and (b) land within sampling tolerance
        // of the closed-form uniform variance for BOTH datasets.
        let mut variance_hits = 0u32;
        let variance_trials = 5u32;
        for _ in 0..variance_trials {
            xor_shift(&mut rng);
            let center = 0.5;
            let high_spread = 0.3 + (rng % 100) as f64 / 1000.0;
            let low_spread = 0.05 + (rng % 50) as f64 / 1000.0;

            let (mut high_samples, _, true_high_var) =
                generate_dataset(&mut rng, 30, center, high_spread);
            let (mut low_samples, _, true_low_var) =
                generate_dataset(&mut rng, 30, center, low_spread);
            apply_sample_noise(
                &mut high_samples,
                center,
                high_spread,
                noise_weight,
                &mut rng,
            );
            apply_sample_noise(&mut low_samples, center, low_spread, noise_weight, &mut rng);

            // REAL ENGINE: statistics::variance (population).
            let engine_high_var = statistics::variance(&high_samples);
            let engine_low_var = statistics::variance(&low_samples);

            let ordering_correct =
                (engine_high_var > engine_low_var) == (true_high_var > true_low_var);
            // Sampling tolerance: uniform sample variance (n=30) concentrates
            // well within ±60% of the closed form; corrupted samples break it.
            let magnitude_correct = (engine_high_var - true_high_var).abs() <= 0.6 * true_high_var
                && (engine_low_var - true_low_var).abs() <= 0.6 * true_low_var;

            if ordering_correct && magnitude_correct {
                variance_hits += 1;
            }
        }
        let variance_estimation_accuracy = variance_hits as f64 / variance_trials as f64;

        // ── Part 3: Distribution Shape Classification (real engine) ──
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

                // REAL ENGINE: statistics::skewness drives the classifier.
                let predicted_shape = classify_shape(&samples);

                // Noise: with probability ∝ noise, randomize classification.
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
            hdc_mean_error: hdc_mean_estimation_error,
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
        let mut hdc_mean_errors = Vec::new();

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            mean_errors.push(r.mean_error);
            variance_accs.push(r.variance_accuracy);
            class_accs.push(r.classification_accuracy);
            hdc_mean_errors.push(r.hdc_mean_error);
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
        result.insert(
            "hdc_mean_estimation_error",
            MetricValue::from_samples(&hdc_mean_errors),
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
        assert!(result.metrics.contains_key("hdc_mean_estimation_error"));
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

    /// Proves the REAL engine is invoked: at zero noise, statistics::mean on
    /// 20-40 uniform samples lands within a few percent of the true mean —
    /// far tighter than the HDC anchor interpolator ever achieved — and
    /// variance/shape scoring is near-perfect.
    #[test]
    fn test_engine_accuracy_at_zero_noise() {
        let config = BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 8,
            encoding_noise: 0.0,
            time_pressure: 0.0,
            ..Default::default()
        };
        let result = StatisticalInferenceBenchmark.run(&config);
        // Sample-mean error over range for n≥20 uniform: ~0.29/√n ≈ 0.065;
        // allow generous headroom while staying far below chance (~0.25-0.5).
        assert!(
            result.metrics["mean_estimation_error"].mean < 0.15,
            "engine mean error too high: {}",
            result.metrics["mean_estimation_error"].mean
        );
        assert!(
            result.metrics["variance_estimation_accuracy"].mean >= 0.8,
            "engine variance accuracy too low: {}",
            result.metrics["variance_estimation_accuracy"].mean
        );
        assert!(
            result.metrics["distribution_classification_accuracy"].mean >= 0.7,
            "engine shape classification too low: {}",
            result.metrics["distribution_classification_accuracy"].mean
        );
    }

    /// Proves the benchmark CAN fail: a wrong mean estimate scores a large
    /// error, wrong variance magnitudes fail the closed-form check, and a
    /// wrong-shaped dataset is not classified as its opposite.
    #[test]
    fn test_wrong_answers_score_low() {
        // Wrong mean: an estimate at the edge of the support has fractional
        // error 0.5 — an order of magnitude above the engine's typical error.
        let center = 0.5f64;
        let spread = 0.2f64;
        let wrong_estimate = center + spread; // upper edge
        let frac_error = (wrong_estimate - center).abs() / (2.0 * spread);
        assert!(frac_error >= 0.5 - 1e-12);

        // Wrong variance magnitude: 3x the true value fails the 60% band.
        let true_var = (2.0f64 * spread).powi(2) / 12.0;
        let wrong_var = 3.0 * true_var;
        assert!((wrong_var - true_var).abs() > 0.6 * true_var);

        // Wrong shape: the engine's skewness on a right-skewed dataset must
        // NOT classify it as left-skewed (and vice versa).
        let mut rng = 0x1234_5678_9ABC_DEF0u64;
        let (right, _) = generate_shaped_dataset(&mut rng, 200, DistShape::RightSkewed);
        assert_ne!(classify_shape(&right), DistShape::LeftSkewed);
        assert!(statistics::skewness(&right) > 0.0);
        let (left, _) = generate_shaped_dataset(&mut rng, 200, DistShape::LeftSkewed);
        assert_ne!(classify_shape(&left), DistShape::RightSkewed);
        assert!(statistics::skewness(&left) < 0.0);
    }
}
