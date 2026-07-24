// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Collective Aggregation Benchmark
//!
//! Measures shared-mask majority-bundling fidelity. This is an insecure algebra
//! demonstration: pairwise relationships leak and every threshold share
//! reveals the mask.
//!
//! ## Protocol
//!
//! ```text
//! 1. Coordinator generates a session mask and broken compatibility records
//! 2. Each of n peers encrypts their wisdom vector: enc(wisdom_i, mask)
//! 3. CollectiveWisdomPool bundles encrypted contributions
//! 4. one record suffices to recover the mask (the vulnerability under test)
//! 5. Decrypt the aggregate → collective wisdom
//! ```
//!
//! ## Mathematical Note
//!
//! Unlike bind (XOR), majority-vote bundling does NOT perfectly commute with
//! XOR encryption:
//!
//! ```text
//! bundle(enc(w_1, M), ..., enc(w_n, M)) ≈ enc(bundle(w_1, ..., w_n), M)
//! ```
//!
//! The approximation quality depends on n (number of contributors) and the
//! correlation structure of the wisdom vectors. With independent random HVs,
//! the CLT predicts improving accuracy as n grows.
//!
//! ## Key Metrics
//!
//! - `aggregation_fidelity`: Similarity between decrypted aggregate and plaintext bundle
//! - `classification_preservation`: Accuracy of classifying with aggregate vs plaintext bundle
//! - `broken_share_recovery_exact`: Whether the broken share payload recovers the mask
//! - `peer_similarity`: Similarity between individual contribution and aggregate
//!
//! ## References
//!
//! - Imani et al. (2019). A framework for collaborative learning in secure HDC.
//! - Shamir, A. (1979). How to share a secret. CACM.
//! - Kanerva, P. (2009). Hyperdimensional computing: An introduction.

use crate::harness::{
    BenchmarkConfig, BenchmarkProvenance, BenchmarkResult, MetricValue, PsychBenchmark,
};
use std::collections::BTreeMap;
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::hdc_crypto::HdcThresholdSharing;
use symthaea_core::hdc::hdc_fhe::{CollectiveWisdomPool, EncryptedHV};

/// Number of peers contributing to the collective.
const NUM_PEERS: usize = 7;

/// Historical `k` parameter; it is not a security threshold.
const THRESHOLD_K: usize = 3;

/// Number of classes for classification test.
const NUM_CLASSES: usize = 8;

/// Words per prototype for building class vectors.
const WORDS_PER_PROTO: usize = 10;

pub struct CollectiveAggregationBenchmark;

impl PsychBenchmark for CollectiveAggregationBenchmark {
    fn name(&self) -> &str {
        "InsecureAlgebraDemo::CollectiveAggregation"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Insecure Shared-Mask Aggregation Algebra",
            citation: "Shamir, A. (1979). How to share a secret. Communications of the ACM, 22(11), 612-613.",
            year: 1979,
            doi: Some("10.1145/359168.359176"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let seed = config.seed;
        let trials = config.trials_per_condition.max(10);

        let mut fidelity_samples = Vec::with_capacity(trials);
        let mut classification_samples = Vec::with_capacity(trials);
        let mut threshold_exact_samples = Vec::with_capacity(trials);
        let mut privacy_samples = Vec::with_capacity(trials);
        let mut aggregate_time_ns = Vec::with_capacity(trials);

        for trial in 0..trials {
            let trial_seed = seed.wrapping_add(trial as u64 * 6271);

            // Generate a collective mask and broken compatibility records.
            let mask = BinaryHV::random(trial_seed.wrapping_add(10_000));
            let shares = HdcThresholdSharing::split(
                &mask,
                THRESHOLD_K,
                NUM_PEERS,
                trial_seed.wrapping_add(20_000),
            );

            // Each peer builds a "wisdom vector" — a prototype for a randomly assigned class
            // Peers see overlapping but distinct subsets of training data
            let mut peer_wisdoms = Vec::with_capacity(NUM_PEERS);
            for peer in 0..NUM_PEERS {
                // Each peer bundles a few word HVs into their wisdom
                let word_hvs: Vec<BinaryHV> = (0..WORDS_PER_PROTO)
                    .map(|w| {
                        let ws = trial_seed
                            .wrapping_add(peer as u64 * 500)
                            .wrapping_add(w as u64 * 13);
                        BinaryHV::random(ws)
                    })
                    .collect();
                peer_wisdoms.push(BinaryHV::bundle(&word_hvs));
            }

            // --- Plaintext aggregate ---
            let plaintext_bundle = BinaryHV::bundle(&peer_wisdoms);

            // --- Encrypted aggregate via CollectiveWisdomPool ---
            let agg_start = std::time::Instant::now();

            let mut pool = CollectiveWisdomPool::new();
            for (i, wisdom) in peer_wisdoms.iter().enumerate() {
                let encrypted = EncryptedHV::encrypt(wisdom, &mask);
                pool.contribute(&format!("peer-{i}"), encrypted);
            }
            let encrypted_aggregate = pool.aggregate().expect("non-empty pool");

            // Recover mask from k-of-n shares
            let recovered_mask = HdcThresholdSharing::recover(&shares[..THRESHOLD_K]);
            let decrypted_aggregate = encrypted_aggregate.decrypt(&recovered_mask);

            let agg_elapsed = agg_start.elapsed();

            // --- Metric 1: Aggregation fidelity ---
            let fidelity = decrypted_aggregate.similarity(&plaintext_bundle) as f64;
            fidelity_samples.push(fidelity);

            // --- Metric 2: Threshold recovery exactness ---
            let threshold_exact = if recovered_mask == mask { 1.0 } else { 0.0 };
            threshold_exact_samples.push(threshold_exact);

            // --- Metric 3: Classification preservation ---
            // Build class prototypes and test items, classify with plaintext vs aggregate
            let class_protos: Vec<BinaryHV> = (0..NUM_CLASSES)
                .map(|c| {
                    let words: Vec<BinaryHV> = (0..8)
                        .map(|w| {
                            BinaryHV::random(
                                trial_seed
                                    .wrapping_add(80_000)
                                    .wrapping_add(c as u64 * 200)
                                    .wrapping_add(w as u64),
                            )
                        })
                        .collect();
                    BinaryHV::bundle(&words)
                })
                .collect();

            // Create test items and classify using aggregate-enriched prototypes
            // Test: does the aggregate encode the same "directions" as plaintext bundle?
            let mut pt_correct = 0usize;
            let mut agg_correct = 0usize;
            let test_count = 50;

            for t in 0..test_count {
                let class_id = t % NUM_CLASSES;
                let noise_seed = trial_seed.wrapping_add(90_000).wrapping_add(t as u64);
                let test_item = class_protos[class_id].add_noise(0.15, noise_seed);

                // Plaintext enriched: bind class proto with plaintext aggregate
                let pt_enriched: Vec<BinaryHV> = class_protos
                    .iter()
                    .map(|p| {
                        // Use aggregate as context: bundle(proto, aggregate) to test if
                        // aggregate quality affects downstream classification
                        BinaryHV::bundle(&[*p, plaintext_bundle])
                    })
                    .collect();

                let agg_enriched: Vec<BinaryHV> = class_protos
                    .iter()
                    .map(|p| BinaryHV::bundle(&[*p, decrypted_aggregate]))
                    .collect();

                // Classify
                let pt_pred = pt_enriched
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        test_item.similarity(a).total_cmp(&test_item.similarity(b))
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                let agg_pred = agg_enriched
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        test_item.similarity(a).total_cmp(&test_item.similarity(b))
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                if pt_pred == class_id {
                    pt_correct += 1;
                }
                if agg_pred == class_id {
                    agg_correct += 1;
                }
            }

            let classification_pres = agg_correct as f64 / (pt_correct.max(1)) as f64;
            classification_samples.push(classification_pres.min(1.0));

            // --- Metric 4: individual-to-aggregate similarity (not privacy) ---
            let mut privacy_sum = 0.0f64;
            for wisdom in &peer_wisdoms {
                // Similarity between individual wisdom and decrypted aggregate
                // With 7 peers, each contributes ~1/7, so similarity should be moderate
                let sim = wisdom.similarity(&decrypted_aggregate) as f64;
                privacy_sum += sim;
            }
            let avg_individual_sim = privacy_sum / NUM_PEERS as f64;
            privacy_samples.push(avg_individual_sim);

            aggregate_time_ns.push(agg_elapsed.as_nanos() as f64);
        }

        let mut metrics = BTreeMap::new();
        metrics.insert(
            "aggregation_fidelity".to_string(),
            MetricValue::from_samples(&fidelity_samples),
        );
        metrics.insert(
            "classification_preservation".to_string(),
            MetricValue::from_samples(&classification_samples),
        );
        metrics.insert(
            "broken_share_recovery_exact".to_string(),
            MetricValue::from_samples(&threshold_exact_samples),
        );
        metrics.insert(
            "peer_similarity".to_string(),
            MetricValue::from_samples(&privacy_samples),
        );
        metrics.insert(
            "aggregate_time_ns".to_string(),
            MetricValue::from_samples(&aggregate_time_ns),
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
    fn test_collective_aggregation_high_fidelity() {
        let config = BenchmarkConfig {
            seed: 42,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = CollectiveAggregationBenchmark.run(&config);

        let fidelity = result.metrics.get("aggregation_fidelity").unwrap();
        assert!(
            fidelity.mean > 0.80,
            "Aggregation fidelity should be high. Got: {:.3}",
            fidelity.mean
        );
    }

    #[test]
    fn test_collective_aggregation_threshold_exact() {
        let config = BenchmarkConfig {
            seed: 42,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = CollectiveAggregationBenchmark.run(&config);

        let exact = result.metrics.get("broken_share_recovery_exact").unwrap();
        assert_eq!(
            exact.mean, 1.0,
            "Threshold recovery must be bit-exact (XOR secret sharing)"
        );
    }

    #[test]
    fn test_collective_aggregation_has_provenance() {
        assert!(CollectiveAggregationBenchmark.provenance().is_some());
    }

    #[test]
    fn test_collective_aggregation_classification() {
        let config = BenchmarkConfig {
            seed: 42,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = CollectiveAggregationBenchmark.run(&config);

        let pres = result.metrics.get("classification_preservation").unwrap();
        assert!(
            pres.mean > 0.70,
            "Classification with aggregate should be close to plaintext. Got: {:.3}",
            pres.mean
        );
    }
}
