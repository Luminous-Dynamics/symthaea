// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Different-Mask Decorrelation Benchmark
//!
//! Measures decorrelation under different deterministic XOR masks. This is an
//! algebra benchmark, not evidence of privacy or an OTP security experiment.
//!
//! ## Protocol
//!
//! 1. Generate a plaintext vector
//! 2. Encrypt it with two independent masks
//! 3. Measure similarity between the two ciphertexts (should ≈ 0.5)
//! 4. Also measure: similarity between ciphertexts from different sessions
//!
//! ## Mathematical Basis
//!
//! For independent random masks M1, M2:
//! ```text
//! sim(enc(A, M1), enc(A, M2)) = sim(A ⊕ M1, A ⊕ M2) = sim(M1, M2) ≈ 0.5
//! ```
//!
//! This follows from XOR being a group operation: A ⊕ M1 and A ⊕ M2 differ
//! by M1 ⊕ M2, which is a random vector when M1 and M2 are independent.
//!
//! ## Key Metrics
//!
//! - `cross_session_leakage`: |sim - 0.5| averaged over trials (should ≈ 0.0)
//! - `cross_mask_similarity`: raw similarity between cross-mask ciphertexts (should ≈ 0.5)
//! - `same_mask_similarity`: similarity with same mask (should = 1.0, sanity check)
//!
//! ## References
//!
//! - Shannon, C. E. (1949). Communication theory of secrecy systems. Bell System Technical Journal.

use crate::harness::{
    BenchmarkConfig, BenchmarkProvenance, BenchmarkResult, MetricValue, PsychBenchmark,
};
use std::collections::BTreeMap;
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::hdc_fhe::EncryptedHV;

/// Number of distinct vectors to test per trial.
const VECTORS_PER_TRIAL: usize = 20;

pub struct CrossMaskPrivacyBenchmark;

impl PsychBenchmark for CrossMaskPrivacyBenchmark {
    fn name(&self) -> &str {
        "InsecureAlgebraDemo::CrossMaskPrivacy"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Different-Mask XOR Decorrelation",
            citation: "Shannon, C. E. (1949). Communication theory of secrecy systems. Bell System Technical Journal, 28(4), 656-715.",
            year: 1949,
            doi: Some("10.1002/j.1538-7305.1949.tb00928.x"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let seed = config.seed;
        let trials = config.trials_per_condition.max(10);

        let mut leakage_samples = Vec::with_capacity(trials);
        let mut cross_sim_samples = Vec::with_capacity(trials);
        let mut same_sim_samples = Vec::with_capacity(trials);

        for trial in 0..trials {
            let trial_seed = seed.wrapping_add(trial as u64 * 5741);

            // Two independent session masks
            let mask1 = BinaryHV::random(trial_seed.wrapping_add(10_000));
            let mask2 = BinaryHV::random(trial_seed.wrapping_add(20_000));

            let mut trial_leakage = 0.0f64;
            let mut trial_cross_sim = 0.0f64;
            let mut trial_same_sim = 0.0f64;

            for v in 0..VECTORS_PER_TRIAL {
                let plaintext = BinaryHV::random(trial_seed.wrapping_add(v as u64 * 37));

                let enc1 = EncryptedHV::encrypt(&plaintext, &mask1);
                let enc2 = EncryptedHV::encrypt(&plaintext, &mask2);

                // Cross-mask similarity (should ≈ 0.5)
                let cross_sim = enc1.encrypted_similarity(&enc2) as f64;
                trial_cross_sim += cross_sim;
                trial_leakage += (cross_sim - 0.5).abs();

                // Same-mask similarity (should = 1.0, sanity check)
                let enc1b = EncryptedHV::encrypt(&plaintext, &mask1);
                let same_sim = enc1.encrypted_similarity(&enc1b) as f64;
                trial_same_sim += same_sim;
            }

            leakage_samples.push(trial_leakage / VECTORS_PER_TRIAL as f64);
            cross_sim_samples.push(trial_cross_sim / VECTORS_PER_TRIAL as f64);
            same_sim_samples.push(trial_same_sim / VECTORS_PER_TRIAL as f64);
        }

        let mut metrics = BTreeMap::new();
        metrics.insert(
            "cross_session_leakage".to_string(),
            MetricValue::from_samples(&leakage_samples),
        );
        metrics.insert(
            "cross_mask_similarity".to_string(),
            MetricValue::from_samples(&cross_sim_samples),
        );
        metrics.insert(
            "same_mask_similarity".to_string(),
            MetricValue::from_samples(&same_sim_samples),
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
    fn test_cross_mask_privacy_has_provenance() {
        assert!(CrossMaskPrivacyBenchmark.provenance().is_some());
    }

    #[test]
    fn test_cross_mask_leakage_near_zero() {
        let config = BenchmarkConfig {
            seed: 42,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = CrossMaskPrivacyBenchmark.run(&config);

        let leakage = result.metrics.get("cross_session_leakage").unwrap();
        assert!(
            leakage.mean < 0.02,
            "Cross-session leakage should be near 0. Got: {}",
            leakage.mean
        );

        let cross_sim = result.metrics.get("cross_mask_similarity").unwrap();
        assert!(
            (cross_sim.mean - 0.5).abs() < 0.02,
            "Cross-mask similarity should be ~0.5. Got: {}",
            cross_sim.mean
        );
    }

    #[test]
    fn test_same_mask_similarity_exact() {
        let config = BenchmarkConfig {
            seed: 42,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = CrossMaskPrivacyBenchmark.run(&config);

        let same = result.metrics.get("same_mask_similarity").unwrap();
        assert!(
            (same.mean - 1.0).abs() < 1e-10,
            "Same-mask similarity must be exactly 1.0. Got: {}",
            same.mean
        );
    }
}
