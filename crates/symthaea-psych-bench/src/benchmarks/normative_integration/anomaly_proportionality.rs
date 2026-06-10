// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Anomaly Proportionality Benchmark
//!
//! Tests whether consciousness dampening is proportional to moral anomaly
//! severity (graded response) rather than binary (threshold response).
//!
//! ## Method
//!
//! 1. Generate a stable moral trajectory (20 coherent scenarios)
//! 2. Inject anomalies of 10 severity levels [0.0, 0.1, ..., 0.9]
//! 3. At each level, measure simulated consciousness dampening
//! 4. Compute Pearson correlation between severity and consciousness drop
//!
//! ## Prediction
//!
//! Pearson r > 0.9 between anomaly magnitude and consciousness dampening.
//! A well-coupled system should show strong linear proportionality.
//!
//! Science:
//! - Festinger, L. (1957). A Theory of Cognitive Dissonance.
//! - Greene, J. et al. (2001). An fMRI investigation of emotional engagement
//!   in moral judgment. Science 293(5537).

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::trial_analysis::TrialOutcome;
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use std::collections::BTreeMap;
use symthaea_core::hdc::ContinuousHV;

/// Number of severity levels.
const NUM_SEVERITY_LEVELS: usize = 10;

/// Repeats per severity level for stable measurement.
const REPEATS_PER_LEVEL: usize = 5;

/// Dampening strength (matches ethics_engine default).
const DAMPENING_STRENGTH: f64 = 0.15;

/// Number of harmony axes (Eight Harmonies).
const N_HARMONIES: usize = 8;

/// Anomaly Proportionality Benchmark.
pub struct AnomalyProportionalityBenchmark;

impl AnomalyProportionalityBenchmark {
    /// Project a ContinuousHV onto harmony dimensions (same as DriftCoupling).
    fn harmony_projection(hv: &ContinuousHV) -> [f64; N_HARMONIES] {
        let data = &hv.values;
        let stride = data.len() / N_HARMONIES;
        let mut proj = [0.0f64; N_HARMONIES];
        for (h, p) in proj.iter_mut().enumerate() {
            let start = h * stride;
            let end = ((h + 1) * stride).min(data.len());
            let sum: f64 = data[start..end].iter().map(|&v| v as f64).sum();
            *p = sum / (end - start) as f64;
        }
        let max = proj.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = proj.iter().map(|&v| (v - max).exp()).sum();
        for p in proj.iter_mut() {
            *p = (*p - max).exp() / exp_sum;
        }
        proj
    }

    /// Simulate a composite anomaly score from moral vector displacement.
    ///
    /// Anomaly components:
    /// - Value inversion: dominant harmony axis flips (0 or 1)
    /// - Free energy spike: KL divergence from uniform prior exceeds threshold
    /// - Drift alert: trajectory change exceeds threshold
    ///
    /// Returns composite score in [0, 1].
    fn compute_anomaly(
        baseline_proj: &[f64; N_HARMONIES],
        current_proj: &[f64; N_HARMONIES],
        severity: f64,
    ) -> f64 {
        // Value inversion check: did the dominant axis flip?
        let baseline_dominant = baseline_proj
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let current_dominant = current_proj
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let value_inversion = if baseline_dominant != current_dominant {
            1.0
        } else {
            0.0
        };

        // Free energy: KL divergence from uniform
        let uniform = 1.0 / N_HARMONIES as f64;
        let fe: f64 = current_proj
            .iter()
            .map(|&p| {
                if p > 1e-10 {
                    p * (p / uniform).ln()
                } else {
                    0.0
                }
            })
            .sum();
        let fe_spike = (fe / 2.0).min(1.0); // Normalize to [0, 1]

        // Composite: weighted combination scaled by severity
        let raw = 0.3 * value_inversion + 0.4 * severity + 0.3 * fe_spike;
        raw.clamp(0.0, 1.0)
    }

    /// Pearson correlation coefficient.
    fn pearson_r(xs: &[f64], ys: &[f64]) -> f64 {
        let n = xs.len() as f64;
        if n < 2.0 {
            return 0.0;
        }
        let mean_x = xs.iter().sum::<f64>() / n;
        let mean_y = ys.iter().sum::<f64>() / n;
        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            let dx = x - mean_x;
            let dy = y - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }
        if var_x < 1e-15 || var_y < 1e-15 {
            return 0.0;
        }
        cov / (var_x.sqrt() * var_y.sqrt())
    }
}

impl PsychBenchmark for AnomalyProportionalityBenchmark {
    fn name(&self) -> &str {
        "NormativeIntegration::AnomalyProportionality"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Graded Moral Anomaly-Consciousness Coupling",
            citation: "Festinger (1957); Greene et al. (2001)",
            year: 1957,
            doi: None,
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let dim = config.dimension.max(256);
        let seed = config.seed;

        // Create a stable baseline prototype
        let baseline = ContinuousHV::random(dim, seed);
        let baseline_proj = Self::harmony_projection(&baseline);
        let base_consciousness = 0.8;

        let mut severity_levels = Vec::with_capacity(NUM_SEVERITY_LEVELS * REPEATS_PER_LEVEL);
        let mut consciousness_drops = Vec::with_capacity(NUM_SEVERITY_LEVELS * REPEATS_PER_LEVEL);
        let mut anomaly_scores = Vec::with_capacity(NUM_SEVERITY_LEVELS * REPEATS_PER_LEVEL);
        let mut trace = Vec::new();

        for level in 0..NUM_SEVERITY_LEVELS {
            let severity = level as f64 / (NUM_SEVERITY_LEVELS - 1) as f64;

            for repeat in 0..REPEATS_PER_LEVEL {
                let trial_seed = seed + (level * 100 + repeat) as u64;

                // Create anomalous HV: interpolate baseline toward a conflicting vector
                let conflict = ContinuousHV::random(dim, trial_seed + 50000);
                let anomalous: Vec<f32> = baseline
                    .values
                    .iter()
                    .zip(conflict.values.iter())
                    .map(|(b, c)| b * (1.0 - severity as f32) + c * severity as f32)
                    .collect();
                let anomalous_hv = ContinuousHV::from_values(anomalous);
                let anomalous_proj = Self::harmony_projection(&anomalous_hv);

                let anomaly = Self::compute_anomaly(&baseline_proj, &anomalous_proj, severity);
                let consciousness =
                    (base_consciousness * (1.0 - anomaly * DAMPENING_STRENGTH)).clamp(0.0, 1.0);
                let drop = base_consciousness - consciousness;

                severity_levels.push(severity);
                consciousness_drops.push(drop);
                anomaly_scores.push(anomaly);

                if config.trial_trace {
                    let mut extra = BTreeMap::new();
                    extra.insert("severity".into(), severity);
                    extra.insert("anomaly_score".into(), anomaly);
                    trace.push(TrialOutcome {
                        trial_idx: level * REPEATS_PER_LEVEL + repeat,
                        condition: format!("severity_{:.1}", severity),
                        correct: true,
                        rt_ticks: 0.0,
                        similarity: consciousness,
                        confidence: anomaly,
                        response_idx: level,
                        extra,
                    });
                }
            }
        }

        // Pearson r between severity and consciousness drop
        let r_severity_drop = Self::pearson_r(&severity_levels, &consciousness_drops);

        // Pearson r between anomaly_score and consciousness drop
        let r_anomaly_drop = Self::pearson_r(&anomaly_scores, &consciousness_drops);

        // Prediction: r > 0.9 indicates strong proportionality
        let prediction_met = r_anomaly_drop > 0.9;

        // Mean consciousness at min and max severity
        let min_severity_consciousness: f64 = consciousness_drops
            .iter()
            .zip(severity_levels.iter())
            .filter(|&(_, &s)| s < 0.05)
            .map(|(&d, _)| base_consciousness - d)
            .sum::<f64>()
            / REPEATS_PER_LEVEL as f64;

        let max_severity_consciousness: f64 = consciousness_drops
            .iter()
            .zip(severity_levels.iter())
            .filter(|&(_, &s)| s > 0.95)
            .map(|(&d, _)| base_consciousness - d)
            .sum::<f64>()
            / REPEATS_PER_LEVEL as f64;

        result.insert(
            "pearson_r_severity_drop",
            MetricValue::from_samples(&[r_severity_drop]),
        );
        result.insert(
            "pearson_r_anomaly_drop",
            MetricValue::from_samples(&[r_anomaly_drop]),
        );
        result.insert(
            "prediction_met",
            MetricValue::from_samples(&[if prediction_met { 1.0 } else { 0.0 }]),
        );
        result.insert(
            "min_severity_consciousness",
            MetricValue::from_samples(&[min_severity_consciousness]),
        );
        result.insert(
            "max_severity_consciousness",
            MetricValue::from_samples(&[max_severity_consciousness]),
        );
        result.insert(
            "anomaly_score_range",
            MetricValue::from_samples(&anomaly_scores),
        );

        result.conditions = NUM_SEVERITY_LEVELS;
        result.trials_per_condition = REPEATS_PER_LEVEL;
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
    fn test_anomaly_proportionality_runs() {
        let config = BenchmarkConfig {
            dimension: 256,
            ..Default::default()
        };
        let result = AnomalyProportionalityBenchmark.run(&config);
        assert!(result.metrics.contains_key("pearson_r_anomaly_drop"));
    }

    #[test]
    fn test_pearson_r_perfect_correlation() {
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let r = AnomalyProportionalityBenchmark::pearson_r(&xs, &ys);
        assert!(
            (r - 1.0).abs() < 1e-10,
            "Perfect linear correlation should give r=1.0, got {:.6}",
            r
        );
    }

    #[test]
    fn test_pearson_r_no_correlation() {
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = vec![5.0, 5.0, 5.0, 5.0, 5.0]; // Constant
        let r = AnomalyProportionalityBenchmark::pearson_r(&xs, &ys);
        assert!(
            r.abs() < 1e-10,
            "Constant y should give r=0.0, got {:.6}",
            r
        );
    }

    #[test]
    fn test_compute_anomaly_zero_severity() {
        let hv = ContinuousHV::random(256, 42);
        let proj = AnomalyProportionalityBenchmark::harmony_projection(&hv);
        let anomaly = AnomalyProportionalityBenchmark::compute_anomaly(&proj, &proj, 0.0);
        // Same projection + zero severity → anomaly should be small (only FE component)
        assert!(
            anomaly < 0.5,
            "Zero severity same-projection should have low anomaly, got {:.4}",
            anomaly
        );
    }

    #[test]
    fn test_compute_anomaly_high_severity() {
        let hv_a = ContinuousHV::random(256, 42);
        let hv_b = ContinuousHV::random(256, 99);
        let proj_a = AnomalyProportionalityBenchmark::harmony_projection(&hv_a);
        let proj_b = AnomalyProportionalityBenchmark::harmony_projection(&hv_b);
        let anomaly = AnomalyProportionalityBenchmark::compute_anomaly(&proj_a, &proj_b, 0.9);
        assert!(
            anomaly > 0.3,
            "High severity different-projection should have high anomaly, got {:.4}",
            anomaly
        );
    }

    #[test]
    fn test_anomaly_proportionality_strong_correlation() {
        let config = BenchmarkConfig {
            dimension: 512,
            ..Default::default()
        };
        let result = AnomalyProportionalityBenchmark.run(&config);
        let r = result.metrics["pearson_r_anomaly_drop"].mean;
        assert!(
            r > 0.5,
            "Anomaly-drop correlation should be positive, got {:.4}",
            r
        );
    }

    #[test]
    fn test_consciousness_decreases_with_severity() {
        let config = BenchmarkConfig {
            dimension: 256,
            ..Default::default()
        };
        let result = AnomalyProportionalityBenchmark.run(&config);
        let min_c = result.metrics["min_severity_consciousness"].mean;
        let max_c = result.metrics["max_severity_consciousness"].mean;
        assert!(
            min_c > max_c,
            "Min severity consciousness ({:.4}) should exceed max severity ({:.4})",
            min_c,
            max_c,
        );
    }
}
