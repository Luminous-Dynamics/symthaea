// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Encrypted Classification Benchmark
//!
//! Demonstrates that a reused XOR mask preserves nearest-neighbor
//! classification—and therefore leaks the exact distance structure. The
//! ciphertext terminology is historical; this is not a privacy benchmark.
//!
//! ## Mathematical Basis
//!
//! XOR encryption with a shared session mask M is a distance-preserving bijection:
//!
//! ```text
//! sim(enc(A, M), enc(B, M)) = sim(A ⊗ M, B ⊗ M) = sim(A, B)
//! ```
//!
//! because Hamming distance is invariant under XOR with a constant:
//! d_H(A ⊗ M, B ⊗ M) = d_H(A, B).
//!
//! This means nearest-neighbor classification is **mathematically guaranteed**
//! to produce identical results on encrypted data — not approximately, but exactly.
//!
//! ## Benchmark Design
//!
//! 1. Create K class prototypes as word-bundle BinaryHVs (realistic NLP-style encoding)
//! 2. Generate N test items per class (noisy versions of prototypes)
//! 3. Classify in plaintext: argmax_k sim(test, prototype_k)
//! 4. Encrypt all vectors with shared session mask, classify on ciphertexts
//! 5. Measure: accuracy parity, similarity preservation, timing overhead, ciphertext hiding
//!
//! ## Key Metrics
//!
//! - `encrypted_accuracy`: Classification accuracy on encrypted data (should = plaintext)
//! - `accuracy_loss`: |encrypted_accuracy - plaintext_accuracy| (should = 0.0)
//! - `similarity_mae`: Mean absolute error of encrypted vs plaintext similarities (should ≈ 0.0)
//! - `encryption_overhead_ratio`: Time(encrypted) / Time(plaintext) (should ≈ 1.0)
//! - `ciphertext_hiding`: Mean similarity of ciphertext to plaintext (should ≈ 0.5)
//!
//! ## References
//!
//! - Shannon, C. (1949). Communication theory of secrecy systems.
//! - Imani et al. (2019). A framework for collaborative learning in secure HDC.
//! - Kanerva, P. (2009). Hyperdimensional computing: An introduction.

use crate::harness::{
    BenchmarkConfig, BenchmarkProvenance, BenchmarkResult, MetricValue, PsychBenchmark,
};
use std::collections::BTreeMap;
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::hdc_fhe::EncryptedHV;

/// Number of classes for the classification task.
const NUM_CLASSES: usize = 10;

/// Number of words per class prototype.
const WORDS_PER_PROTOTYPE: usize = 12;

/// Noise level for generating test items from prototypes.
const TEST_NOISE: f64 = 0.15;

/// Generate a deterministic word HV from a word seed.
fn word_hv(seed: u64) -> BinaryHV {
    BinaryHV::random(seed)
}

/// Build a class prototype as a bundle of word HVs.
fn build_prototype(class_id: usize, base_seed: u64) -> BinaryHV {
    let word_hvs: Vec<BinaryHV> = (0..WORDS_PER_PROTOTYPE)
        .map(|w| {
            let seed = base_seed
                .wrapping_add(class_id as u64 * 1000)
                .wrapping_add(w as u64);
            word_hv(seed)
        })
        .collect();
    BinaryHV::bundle(&word_hvs)
}

/// Classify a test item against prototypes, return (predicted_class, similarities).
fn classify_plaintext(test: &BinaryHV, prototypes: &[BinaryHV]) -> (usize, Vec<f32>) {
    let sims: Vec<f32> = prototypes.iter().map(|p| test.similarity(p)).collect();
    let predicted = sims
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    (predicted, sims)
}

/// Classify an encrypted test item against encrypted prototypes.
fn classify_encrypted(test: &EncryptedHV, prototypes: &[EncryptedHV]) -> (usize, Vec<f32>) {
    let sims: Vec<f32> = prototypes
        .iter()
        .map(|p| test.encrypted_similarity(p))
        .collect();
    let predicted = sims
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    (predicted, sims)
}

pub struct EncryptedClassificationBenchmark;

impl PsychBenchmark for EncryptedClassificationBenchmark {
    fn name(&self) -> &str {
        "InsecureAlgebraDemo::EncryptedClassification"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Shared-Mask Distance-Leakage Classification",
            citation: "Imani, M. et al. (2019). A framework for collaborative learning in secure high-dimensional space. IEEE HPCA.",
            year: 2019,
            doi: Some("10.1109/HPCA.2019.00036"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let seed = config.seed;
        let trials = config.trials_per_condition.max(10);

        let mut plaintext_accuracies = Vec::with_capacity(trials);
        let mut encrypted_accuracies = Vec::with_capacity(trials);
        let mut accuracy_losses = Vec::with_capacity(trials);
        let mut similarity_maes = Vec::with_capacity(trials);
        let mut overhead_ratios = Vec::with_capacity(trials);
        let mut hiding_values = Vec::with_capacity(trials);
        let mut plaintext_times_ns = Vec::with_capacity(trials);
        let mut encrypted_times_ns = Vec::with_capacity(trials);

        for trial in 0..trials {
            let trial_seed = seed.wrapping_add(trial as u64 * 7919);

            // Build class prototypes
            let prototypes: Vec<BinaryHV> = (0..NUM_CLASSES)
                .map(|c| build_prototype(c, trial_seed))
                .collect();

            // Generate test items: noisy versions of each prototype
            let items_per_class = 20;
            let mut test_items = Vec::with_capacity(NUM_CLASSES * items_per_class);
            let mut true_labels = Vec::with_capacity(NUM_CLASSES * items_per_class);

            for (class_id, prototype) in prototypes.iter().enumerate() {
                for item in 0..items_per_class {
                    let noise_seed = trial_seed
                        .wrapping_add(50_000)
                        .wrapping_add(class_id as u64 * 100)
                        .wrapping_add(item as u64);
                    let noisy = prototype.add_noise(TEST_NOISE as f32, noise_seed);
                    test_items.push(noisy);
                    true_labels.push(class_id);
                }
            }

            // --- Plaintext classification ---
            let pt_start = std::time::Instant::now();
            let mut pt_correct = 0usize;
            let mut all_pt_sims = Vec::new();

            for (i, test) in test_items.iter().enumerate() {
                let (predicted, sims) = classify_plaintext(test, &prototypes);
                if predicted == true_labels[i] {
                    pt_correct += 1;
                }
                all_pt_sims.push(sims);
            }
            let pt_elapsed = pt_start.elapsed();
            let pt_accuracy = pt_correct as f64 / test_items.len() as f64;

            // --- Encrypted classification ---
            // Generate session mask (fresh per trial, OTP requirement)
            let mask = BinaryHV::random(trial_seed.wrapping_add(99_999));

            // Encrypt prototypes
            let enc_prototypes: Vec<EncryptedHV> = prototypes
                .iter()
                .map(|p| EncryptedHV::encrypt(p, &mask))
                .collect();

            // Encrypt test items
            let enc_items: Vec<EncryptedHV> = test_items
                .iter()
                .map(|t| EncryptedHV::encrypt(t, &mask))
                .collect();

            let enc_start = std::time::Instant::now();
            let mut enc_correct = 0usize;
            let mut all_enc_sims = Vec::new();

            for (i, enc_test) in enc_items.iter().enumerate() {
                let (predicted, sims) = classify_encrypted(enc_test, &enc_prototypes);
                if predicted == true_labels[i] {
                    enc_correct += 1;
                }
                all_enc_sims.push(sims);
            }
            let enc_elapsed = enc_start.elapsed();
            let enc_accuracy = enc_correct as f64 / test_items.len() as f64;

            // --- Similarity preservation ---
            let mut total_sim_error = 0.0f64;
            let mut sim_count = 0usize;
            for (pt_sims, enc_sims) in all_pt_sims.iter().zip(all_enc_sims.iter()) {
                for (ps, es) in pt_sims.iter().zip(enc_sims.iter()) {
                    total_sim_error += (*ps as f64 - *es as f64).abs();
                    sim_count += 1;
                }
            }
            let sim_mae = if sim_count > 0 {
                total_sim_error / sim_count as f64
            } else {
                0.0
            };

            // --- Ciphertext hiding ---
            // Measure similarity between ciphertext and plaintext (should ≈ 0.5)
            let mut hiding_sum = 0.0f64;
            for (pt, enc) in prototypes.iter().zip(enc_prototypes.iter()) {
                hiding_sum += enc.ciphertext.similarity(pt) as f64;
            }
            let hiding = hiding_sum / NUM_CLASSES as f64;

            // --- Timing overhead ---
            let pt_ns = pt_elapsed.as_nanos() as f64;
            let enc_ns = enc_elapsed.as_nanos() as f64;
            let overhead = if pt_ns > 0.0 { enc_ns / pt_ns } else { 1.0 };

            plaintext_accuracies.push(pt_accuracy);
            encrypted_accuracies.push(enc_accuracy);
            accuracy_losses.push((pt_accuracy - enc_accuracy).abs());
            similarity_maes.push(sim_mae);
            overhead_ratios.push(overhead);
            hiding_values.push(hiding);
            plaintext_times_ns.push(pt_ns);
            encrypted_times_ns.push(enc_ns);
        }

        let mut metrics = BTreeMap::new();
        metrics.insert(
            "plaintext_accuracy".to_string(),
            MetricValue::from_samples(&plaintext_accuracies),
        );
        metrics.insert(
            "encrypted_accuracy".to_string(),
            MetricValue::from_samples(&encrypted_accuracies),
        );
        metrics.insert(
            "accuracy_loss".to_string(),
            MetricValue::from_samples(&accuracy_losses),
        );
        metrics.insert(
            "similarity_mae".to_string(),
            MetricValue::from_samples(&similarity_maes),
        );
        metrics.insert(
            "encryption_overhead_ratio".to_string(),
            MetricValue::from_samples(&overhead_ratios),
        );
        metrics.insert(
            "ciphertext_hiding".to_string(),
            MetricValue::from_samples(&hiding_values),
        );
        metrics.insert(
            "plaintext_time_ns".to_string(),
            MetricValue::from_samples(&plaintext_times_ns),
        );
        metrics.insert(
            "encrypted_time_ns".to_string(),
            MetricValue::from_samples(&encrypted_times_ns),
        );

        BenchmarkResult {
            benchmark: self.name().to_string(),
            config_label: config.label.clone(),
            metrics,
            elapsed_ms: start.elapsed().as_millis() as u64,
            conditions: 2, // plaintext vs encrypted
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
    fn test_encrypted_classification_zero_accuracy_loss() {
        let config = BenchmarkConfig {
            seed: 42,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = EncryptedClassificationBenchmark.run(&config);

        let loss = result.metrics.get("accuracy_loss").unwrap();
        assert!(
            loss.mean < 1e-10,
            "Accuracy loss must be exactly zero (XOR is distance-preserving). Got: {}",
            loss.mean
        );

        let sim_mae = result.metrics.get("similarity_mae").unwrap();
        assert!(
            sim_mae.mean < 1e-6,
            "Similarity MAE must be near-zero. Got: {}",
            sim_mae.mean
        );
    }

    #[test]
    fn test_encrypted_classification_ciphertext_hiding() {
        let config = BenchmarkConfig {
            seed: 42,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = EncryptedClassificationBenchmark.run(&config);

        let hiding = result.metrics.get("ciphertext_hiding").unwrap();
        assert!(
            (hiding.mean - 0.5).abs() < 0.05,
            "Ciphertext should be ~0.5 similar to plaintext (OTP hiding). Got: {}",
            hiding.mean
        );
    }

    #[test]
    fn test_encrypted_classification_accuracy_is_high() {
        let config = BenchmarkConfig {
            seed: 42,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = EncryptedClassificationBenchmark.run(&config);

        let pt_acc = result.metrics.get("plaintext_accuracy").unwrap();
        assert!(
            pt_acc.mean > 0.85,
            "Plaintext accuracy should be high with 15% noise. Got: {}",
            pt_acc.mean
        );

        let enc_acc = result.metrics.get("encrypted_accuracy").unwrap();
        assert_eq!(
            pt_acc.mean, enc_acc.mean,
            "Encrypted and plaintext accuracy must be identical"
        );
    }

    #[test]
    fn test_encrypted_classification_timing() {
        let config = BenchmarkConfig {
            seed: 42,
            trials_per_condition: 5,
            ..Default::default()
        };
        let result = EncryptedClassificationBenchmark.run(&config);

        let overhead = result.metrics.get("encryption_overhead_ratio").unwrap();
        assert!(
            overhead.mean < 3.0,
            "Encrypted classification should be near-plaintext speed. Got: {:.2}x",
            overhead.mean
        );
    }

    #[test]
    fn test_encrypted_classification_has_provenance() {
        assert!(EncryptedClassificationBenchmark.provenance().is_some());
    }

    #[test]
    fn test_print_security_metrics() {
        let config = BenchmarkConfig {
            seed: 42,
            trials_per_condition: 20,
            ..Default::default()
        };

        let result = EncryptedClassificationBenchmark.run(&config);

        println!("\n═══ Shared-Mask Classification Leakage Results ═══");
        for (key, val) in &result.metrics {
            println!(
                "  {:<30} mean={:.6}  sd={:.6}  CI=[{:.6}, {:.6}]",
                key, val.mean, val.std_dev, val.ci_lower, val.ci_upper
            );
        }
        println!("  Elapsed: {} ms", result.elapsed_ms);
        println!("═══════════════════════════════════════════════\n");

        // Also run collective aggregation
        let agg_result =
            super::super::collective_aggregation::CollectiveAggregationBenchmark.run(&config);

        println!("═══ Shared-Mask Aggregation Demo Results ═══");
        for (key, val) in &agg_result.metrics {
            println!(
                "  {:<30} mean={:.6}  sd={:.6}  CI=[{:.6}, {:.6}]",
                key, val.mean, val.std_dev, val.ci_lower, val.ci_upper
            );
        }
        println!("  Elapsed: {} ms", agg_result.elapsed_ms);
        println!("═══════════════════════════════════════════════\n");
    }
}
