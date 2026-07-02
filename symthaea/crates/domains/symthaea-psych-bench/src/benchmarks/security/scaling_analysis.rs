// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Scaling Analysis Benchmark
//!
//! Tests encrypted classification accuracy as task difficulty scales with
//! the number of classes. BinaryHV dimension is fixed at 16,384; scaling
//! is achieved by varying the class count (2, 4, 8, 16, 32, 64, 128).
//!
//! ## Protocol
//!
//! For each class count K:
//! 1. Create K class prototypes (bundles of 10 random word HVs each)
//! 2. Generate noisy test items (15% noise) for each class
//! 3. Classify under encryption with shared session mask
//! 4. Measure accuracy and timing overhead
//!
//! ## Mathematical Basis
//!
//! HDC classification accuracy degrades gracefully with class count due to
//! the concentration of measure in high-dimensional spaces. With D = 16,384
//! bits, random BinaryHVs are nearly orthogonal (expected similarity ≈ 0.5),
//! providing exponential capacity scaling: O(2^D) possible near-orthogonal vectors.
//!
//! ## Key Metrics
//!
//! - `accuracy_at_scale`: mean accuracy across all class-count levels
//! - `overhead_at_scale`: mean encryption overhead ratio across levels
//! - `accuracy_2_classes` through `accuracy_128_classes`: per-level accuracy
//!
//! ## References
//!
//! - Kanerva, P. (2009). Hyperdimensional computing: An introduction to computing
//!   in distributed representation with high-dimensional random vectors.
//! - Rahimi, A. et al. (2016). A robust and energy-efficient classifier using
//!   brain-inspired hyperdimensional computing. ISLPED.

use crate::harness::{
    BenchmarkConfig, BenchmarkProvenance, BenchmarkResult, MetricValue, PsychBenchmark,
};
use std::collections::BTreeMap;
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::hdc_fhe::EncryptedHV;

/// Class counts to test scaling behavior.
const CLASS_COUNTS: &[usize] = &[2, 4, 8, 16, 32, 64, 128];

/// Words per prototype.
const WORDS_PER_PROTO: usize = 10;

/// Noise level for test items.
const TEST_NOISE: f32 = 0.15;

/// Test items per class.
const ITEMS_PER_CLASS: usize = 10;

pub struct ScalingAnalysisBenchmark;

impl PsychBenchmark for ScalingAnalysisBenchmark {
    fn name(&self) -> &str {
        "Security::ScalingAnalysis"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "HDC Classification Scaling",
            citation: "Rahimi, A. et al. (2016). A robust and energy-efficient classifier using brain-inspired hyperdimensional computing. ISLPED.",
            year: 2016,
            doi: Some("10.1145/2934583.2934624"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let seed = config.seed;
        let trials = config.trials_per_condition.max(5);

        // Collect per-level accuracy across trials
        let mut level_accuracies: Vec<Vec<f64>> =
            vec![Vec::with_capacity(trials); CLASS_COUNTS.len()];
        let mut level_overheads: Vec<Vec<f64>> =
            vec![Vec::with_capacity(trials); CLASS_COUNTS.len()];

        for trial in 0..trials {
            let trial_seed = seed.wrapping_add(trial as u64 * 4219);

            for (level_idx, &num_classes) in CLASS_COUNTS.iter().enumerate() {
                let level_seed = trial_seed.wrapping_add(level_idx as u64 * 100_000);

                // Build class prototypes
                let prototypes: Vec<BinaryHV> = (0..num_classes)
                    .map(|c| {
                        let words: Vec<BinaryHV> = (0..WORDS_PER_PROTO)
                            .map(|w| {
                                BinaryHV::random(
                                    level_seed
                                        .wrapping_add(c as u64 * 500)
                                        .wrapping_add(w as u64),
                                )
                            })
                            .collect();
                        BinaryHV::bundle(&words)
                    })
                    .collect();

                // Session mask
                let mask = BinaryHV::random(level_seed.wrapping_add(50_000));

                // Encrypt prototypes
                let enc_protos: Vec<EncryptedHV> = prototypes
                    .iter()
                    .map(|p| EncryptedHV::encrypt(p, &mask))
                    .collect();

                // Generate test items and classify
                let mut enc_correct = 0usize;
                let total_items = num_classes * ITEMS_PER_CLASS;

                // Plaintext classification (timed for overhead comparison)
                let pt_start = std::time::Instant::now();
                for c in 0..num_classes {
                    for item in 0..ITEMS_PER_CLASS {
                        let noise_seed = level_seed
                            .wrapping_add(60_000)
                            .wrapping_add(c as u64 * 100)
                            .wrapping_add(item as u64);
                        let test_item = prototypes[c].add_noise(TEST_NOISE, noise_seed);

                        let _pt_pred = prototypes
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| {
                                test_item.similarity(a).total_cmp(&test_item.similarity(b))
                            })
                            .map(|(i, _)| i)
                            .unwrap_or(0);
                    }
                }
                let pt_elapsed = pt_start.elapsed();

                let enc_start = std::time::Instant::now();
                for (c, prototype) in prototypes.iter().enumerate() {
                    for item in 0..ITEMS_PER_CLASS {
                        let noise_seed = level_seed
                            .wrapping_add(60_000)
                            .wrapping_add(c as u64 * 100)
                            .wrapping_add(item as u64);
                        let test_item = prototype.add_noise(TEST_NOISE, noise_seed);
                        let enc_test = EncryptedHV::encrypt(&test_item, &mask);

                        let enc_pred = enc_protos
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| {
                                enc_test
                                    .encrypted_similarity(a)
                                    .total_cmp(&enc_test.encrypted_similarity(b))
                            })
                            .map(|(i, _)| i)
                            .unwrap_or(0);

                        if enc_pred == c {
                            enc_correct += 1;
                        }
                    }
                }
                let enc_elapsed = enc_start.elapsed();

                let enc_acc = enc_correct as f64 / total_items as f64;
                level_accuracies[level_idx].push(enc_acc);

                // Encrypted accuracy equals plaintext (XOR preserves distance),
                // so overhead is purely timing
                let pt_ns = pt_elapsed.as_nanos() as f64;
                let enc_ns = enc_elapsed.as_nanos() as f64;
                let overhead = if pt_ns > 0.0 { enc_ns / pt_ns } else { 1.0 };
                level_overheads[level_idx].push(overhead);
            }
        }

        let mut metrics = BTreeMap::new();

        // Per-level metrics
        let mut all_accs = Vec::new();
        let mut all_overheads = Vec::new();
        for (level_idx, &num_classes) in CLASS_COUNTS.iter().enumerate() {
            let key = format!("accuracy_{}_classes", num_classes);
            metrics.insert(key, MetricValue::from_samples(&level_accuracies[level_idx]));
            all_accs.extend_from_slice(&level_accuracies[level_idx]);
            all_overheads.extend_from_slice(&level_overheads[level_idx]);
        }

        // Aggregate metrics
        metrics.insert(
            "accuracy_at_scale".to_string(),
            MetricValue::from_samples(&all_accs),
        );
        metrics.insert(
            "overhead_at_scale".to_string(),
            MetricValue::from_samples(&all_overheads),
        );

        BenchmarkResult {
            benchmark: self.name().to_string(),
            config_label: config.label.clone(),
            metrics,
            elapsed_ms: start.elapsed().as_millis() as u64,
            conditions: CLASS_COUNTS.len(),
            trials_per_condition: trials,
            trial_trace: Vec::new(),
            notes: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaling_analysis_has_provenance() {
        assert!(ScalingAnalysisBenchmark.provenance().is_some());
    }

    #[test]
    fn test_scaling_analysis_high_accuracy() {
        let config = BenchmarkConfig {
            seed: 42,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = ScalingAnalysisBenchmark.run(&config);

        let acc = result.metrics.get("accuracy_at_scale").unwrap();
        assert!(
            acc.mean > 0.80,
            "Mean accuracy across scales should be high with D=16384. Got: {}",
            acc.mean
        );

        // 2-class should be very high
        let acc2 = result.metrics.get("accuracy_2_classes").unwrap();
        assert!(
            acc2.mean > 0.95,
            "2-class accuracy should be near-perfect. Got: {}",
            acc2.mean
        );
    }

    #[test]
    fn test_scaling_analysis_monotonic_degradation() {
        let config = BenchmarkConfig {
            seed: 42,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = ScalingAnalysisBenchmark.run(&config);

        // Accuracy should generally decrease (or stay flat) as classes increase
        let acc2 = result.metrics.get("accuracy_2_classes").unwrap().mean;
        let acc128 = result.metrics.get("accuracy_128_classes").unwrap().mean;
        assert!(
            acc2 >= acc128,
            "2-class accuracy ({}) should be >= 128-class accuracy ({})",
            acc2,
            acc128
        );
    }
}
