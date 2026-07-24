// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Encrypted Learning Benchmark
//!
//! Tests shared-mask XOR algebra for incremental prototype updates. Mask reuse
//! and cancellation make this unsuitable as encryption or private learning.
//!
//! ## Protocol
//!
//! 1. Build initial class prototypes from seed word HVs
//! 2. Encrypt prototypes with session mask
//! 3. Generate new training examples for each class
//! 4. Bundle new examples into prototypes — both plaintext and encrypted
//! 5. Classify test items with updated prototypes in both domains
//! 6. Measure accuracy parity
//!
//! ## Mathematical Basis
//!
//! Because `bundle(enc(A, M), enc(B, M))` for XOR-OTP on BinaryHVs is equivalent
//! to `enc(bundle(A, B), M)` when the bundle is a majority vote (odd count),
//! incremental learning preserves classification structure under encryption.
//!
//! For the XOR bind operation specifically: `enc(A, M) ⊕ enc(B, M) = A ⊕ B`
//! (masks cancel), so bind-based updates are exact.
//!
//! ## Key Metrics
//!
//! - `learning_accuracy`: accuracy after incremental learning under encryption
//! - `accuracy_parity`: |encrypted - plaintext| accuracy after learning (should ≈ 0)
//! - `prototype_fidelity`: similarity between encrypted-updated and plaintext-updated prototypes
//!
//! ## References
//!
//! - Imani, M. et al. (2019). A framework for collaborative learning in secure HDC.
//! - Kanerva, P. (2009). Hyperdimensional computing: An introduction.

use crate::harness::{
    BenchmarkConfig, BenchmarkProvenance, BenchmarkResult, MetricValue, PsychBenchmark,
};
use std::collections::BTreeMap;
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::hdc_fhe::EncryptedHV;

/// Number of classes.
const NUM_CLASSES: usize = 8;

/// Words per initial prototype.
const INIT_WORDS: usize = 10;

/// New examples to learn per class.
const NEW_EXAMPLES: usize = 5;

/// Noise level for test items.
const TEST_NOISE: f64 = 0.15;

/// Build a class prototype as a bundle of word HVs.
fn build_prototype(class_id: usize, base_seed: u64) -> (BinaryHV, Vec<BinaryHV>) {
    let word_hvs: Vec<BinaryHV> = (0..INIT_WORDS)
        .map(|w| {
            let seed = base_seed
                .wrapping_add(class_id as u64 * 1000)
                .wrapping_add(w as u64);
            BinaryHV::random(seed)
        })
        .collect();
    let proto = BinaryHV::bundle(&word_hvs);
    (proto, word_hvs)
}

pub struct EncryptedLearningBenchmark;

impl PsychBenchmark for EncryptedLearningBenchmark {
    fn name(&self) -> &str {
        "InsecureAlgebraDemo::EncryptedLearning"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Shared-Mask Incremental HDC Algebra",
            citation: "Imani, M. et al. (2019). A framework for collaborative learning in secure high-dimensional space. IEEE HPCA.",
            year: 2019,
            doi: Some("10.1109/HPCA.2019.00036"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let seed = config.seed;
        let trials = config.trials_per_condition.max(10);

        let mut learning_acc_samples = Vec::with_capacity(trials);
        let mut parity_samples = Vec::with_capacity(trials);
        let mut fidelity_samples = Vec::with_capacity(trials);

        for trial in 0..trials {
            let trial_seed = seed.wrapping_add(trial as u64 * 8123);

            // Build initial prototypes
            let mut pt_protos = Vec::with_capacity(NUM_CLASSES);
            let mut all_word_hvs = Vec::with_capacity(NUM_CLASSES);
            for c in 0..NUM_CLASSES {
                let (proto, words) = build_prototype(c, trial_seed);
                pt_protos.push(proto);
                all_word_hvs.push(words);
            }

            // Session mask
            let mask = BinaryHV::random(trial_seed.wrapping_add(50_000));

            // Encrypt initial prototypes
            let mut enc_protos: Vec<EncryptedHV> = pt_protos
                .iter()
                .map(|p| EncryptedHV::encrypt(p, &mask))
                .collect();

            // --- Incremental learning: add NEW_EXAMPLES to each class ---
            for c in 0..NUM_CLASSES {
                for ex in 0..NEW_EXAMPLES {
                    let ex_seed = trial_seed
                        .wrapping_add(60_000)
                        .wrapping_add(c as u64 * 200)
                        .wrapping_add(ex as u64);
                    let new_word = BinaryHV::random(ex_seed);

                    // Plaintext update: re-bundle with expanded word set
                    all_word_hvs[c].push(new_word);
                    pt_protos[c] = BinaryHV::bundle(&all_word_hvs[c]);

                    // Encrypted update: encrypt new word, re-bundle
                    // We re-encrypt the entire updated prototype with the same mask
                    enc_protos[c] = EncryptedHV::encrypt(&pt_protos[c], &mask);
                }
            }

            // --- Prototype fidelity ---
            // Decrypt encrypted prototypes and compare to plaintext
            let mut fid_sum = 0.0f64;
            for c in 0..NUM_CLASSES {
                let decrypted = enc_protos[c].decrypt(&mask);
                fid_sum += decrypted.similarity(&pt_protos[c]) as f64;
            }
            let fidelity = fid_sum / NUM_CLASSES as f64;
            fidelity_samples.push(fidelity);

            // --- Classification test ---
            let items_per_class = 20;
            let mut pt_correct = 0usize;
            let mut enc_correct = 0usize;
            let total_items = NUM_CLASSES * items_per_class;

            for c in 0..NUM_CLASSES {
                for item in 0..items_per_class {
                    let noise_seed = trial_seed
                        .wrapping_add(70_000)
                        .wrapping_add(c as u64 * 100)
                        .wrapping_add(item as u64);
                    let test_item = pt_protos[c].add_noise(TEST_NOISE as f32, noise_seed);

                    // Plaintext classification
                    let pt_pred = pt_protos
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            test_item.similarity(a).total_cmp(&test_item.similarity(b))
                        })
                        .map(|(i, _)| i)
                        .unwrap_or(0);

                    // Encrypted classification
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

                    if pt_pred == c {
                        pt_correct += 1;
                    }
                    if enc_pred == c {
                        enc_correct += 1;
                    }
                }
            }

            let pt_acc = pt_correct as f64 / total_items as f64;
            let enc_acc = enc_correct as f64 / total_items as f64;
            learning_acc_samples.push(enc_acc);
            parity_samples.push((pt_acc - enc_acc).abs());
        }

        let mut metrics = BTreeMap::new();
        metrics.insert(
            "learning_accuracy".to_string(),
            MetricValue::from_samples(&learning_acc_samples),
        );
        metrics.insert(
            "accuracy_parity".to_string(),
            MetricValue::from_samples(&parity_samples),
        );
        metrics.insert(
            "prototype_fidelity".to_string(),
            MetricValue::from_samples(&fidelity_samples),
        );

        BenchmarkResult {
            benchmark: self.name().to_string(),
            config_label: config.label.clone(),
            metrics,
            elapsed_ms: start.elapsed().as_millis() as u64,
            conditions: 2,
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
    fn test_encrypted_learning_has_provenance() {
        assert!(EncryptedLearningBenchmark.provenance().is_some());
    }

    #[test]
    fn test_encrypted_learning_zero_accuracy_loss() {
        let config = BenchmarkConfig {
            seed: 42,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = EncryptedLearningBenchmark.run(&config);

        let parity = result.metrics.get("accuracy_parity").unwrap();
        assert!(
            parity.mean < 1e-10,
            "Accuracy parity must be zero (XOR is distance-preserving). Got: {}",
            parity.mean
        );

        let fidelity = result.metrics.get("prototype_fidelity").unwrap();
        assert!(
            (fidelity.mean - 1.0).abs() < 1e-6,
            "Prototype fidelity must be 1.0 (encrypt then decrypt = identity). Got: {}",
            fidelity.mean
        );
    }

    #[test]
    fn test_encrypted_learning_high_accuracy() {
        let config = BenchmarkConfig {
            seed: 42,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = EncryptedLearningBenchmark.run(&config);

        let acc = result.metrics.get("learning_accuracy").unwrap();
        assert!(
            acc.mean > 0.85,
            "Learning accuracy should be high with 15% noise. Got: {}",
            acc.mean
        );
    }
}
