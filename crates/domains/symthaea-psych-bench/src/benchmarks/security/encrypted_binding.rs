// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Encrypted Binding Benchmark
//!
//! Tests that the homomorphic XOR bind operation preserves role-filler structure
//! under encryption — decrypted homomorphic bind equals plaintext bind exactly.
//!
//! ## Protocol
//!
//! 1. Create role HVs (AGENT, ACTION, OBJECT) and filler HVs (ALICE, RUN, BALL)
//! 2. Bind role-filler pairs in plaintext: AGENT ⊕ ALICE, ACTION ⊕ RUN, etc.
//! 3. Encrypt all HVs with session mask, perform bind on encrypted domain
//! 4. Decrypt the encrypted bind results
//! 5. Compare: decrypted result should exactly equal plaintext bind
//!
//! ## Mathematical Basis
//!
//! For XOR bind under XOR-OTP encryption:
//! ```text
//! enc(A, M) ⊕ enc(B, M) = (A ⊕ M) ⊕ (B ⊕ M) = A ⊕ B
//! ```
//!
//! The masks cancel perfectly, so the encrypted bind of two ciphertexts
//! produces the **unencrypted** bind of the plaintexts. This means:
//! 1. Binding is exact under encryption (similarity = 1.0)
//! 2. The result is automatically decrypted (no mask removal needed)
//!
//! ## Key Metrics
//!
//! - `binding_preservation`: similarity between encrypted-bind result and plaintext bind
//! - `role_extraction_accuracy`: can we recover fillers from the encrypted composite?
//! - `structure_fidelity`: overall structural preservation across multiple bindings
//!
//! ## References
//!
//! - Plate, T. A. (2003). Holographic Reduced Representations. CSLI.
//! - Imani, M. et al. (2019). A framework for collaborative learning in secure HDC.

use crate::harness::{
    BenchmarkConfig, BenchmarkProvenance, BenchmarkResult, MetricValue, PsychBenchmark,
};
use std::collections::BTreeMap;
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::hdc_fhe::EncryptedHV;

/// Number of role-filler pairs per trial.
const NUM_PAIRS: usize = 6;

pub struct EncryptedBindingBenchmark;

impl PsychBenchmark for EncryptedBindingBenchmark {
    fn name(&self) -> &str {
        "Security::EncryptedBinding"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Homomorphic Binding Preservation",
            citation: "Plate, T. A. (2003). Holographic Reduced Representations. CSLI Publications.",
            year: 2003,
            doi: None,
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let seed = config.seed;
        let trials = config.trials_per_condition.max(10);

        let mut preservation_samples = Vec::with_capacity(trials);
        let mut extraction_samples = Vec::with_capacity(trials);
        let mut structure_samples = Vec::with_capacity(trials);

        for trial in 0..trials {
            let trial_seed = seed.wrapping_add(trial as u64 * 9311);

            // Session mask
            let mask = BinaryHV::random(trial_seed.wrapping_add(10_000));

            // Generate role and filler HVs
            let roles: Vec<BinaryHV> = (0..NUM_PAIRS)
                .map(|r| BinaryHV::random(trial_seed.wrapping_add(20_000).wrapping_add(r as u64)))
                .collect();
            let fillers: Vec<BinaryHV> = (0..NUM_PAIRS)
                .map(|f| BinaryHV::random(trial_seed.wrapping_add(30_000).wrapping_add(f as u64)))
                .collect();

            // --- Plaintext bindings ---
            let pt_bindings: Vec<BinaryHV> = roles
                .iter()
                .zip(fillers.iter())
                .map(|(r, f)| r.bind(f))
                .collect();

            // --- Encrypted bindings ---
            // enc(role, M) ⊕ enc(filler, M) = role ⊕ filler (masks cancel)
            let enc_bindings: Vec<BinaryHV> = roles
                .iter()
                .zip(fillers.iter())
                .map(|(r, f)| {
                    let enc_r = EncryptedHV::encrypt(r, &mask);
                    let enc_f = EncryptedHV::encrypt(f, &mask);
                    // XOR of two encrypted vectors: masks cancel
                    enc_r.ciphertext.bind(&enc_f.ciphertext)
                })
                .collect();

            // --- Metric 1: Binding preservation ---
            // enc(A,M) ⊕ enc(B,M) = A ⊕ B exactly
            let mut pres_sum = 0.0f64;
            for (pt, enc) in pt_bindings.iter().zip(enc_bindings.iter()) {
                pres_sum += pt.similarity(enc) as f64;
            }
            let preservation = pres_sum / NUM_PAIRS as f64;
            preservation_samples.push(preservation);

            // --- Metric 2: Role extraction accuracy ---
            // Given composite = bundle of bindings, extract filler by unbinding with role
            let pt_composite = BinaryHV::bundle(&pt_bindings);
            let enc_composite = BinaryHV::bundle(&enc_bindings);

            let mut extract_correct = 0usize;
            for (i, role) in roles.iter().enumerate() {
                // Unbind: composite ⊕ role should be closest to filler[i]
                let extracted = enc_composite.bind(role);
                let best_match = fillers
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        extracted.similarity(a).total_cmp(&extracted.similarity(b))
                    })
                    .map(|(idx, _)| idx)
                    .unwrap_or(usize::MAX);
                if best_match == i {
                    extract_correct += 1;
                }
            }
            extraction_samples.push(extract_correct as f64 / NUM_PAIRS as f64);

            // --- Metric 3: Structure fidelity ---
            // Encrypted composite should equal plaintext composite
            let struct_fid = pt_composite.similarity(&enc_composite) as f64;
            structure_samples.push(struct_fid);
        }

        let mut metrics = BTreeMap::new();
        metrics.insert(
            "binding_preservation".to_string(),
            MetricValue::from_samples(&preservation_samples),
        );
        metrics.insert(
            "role_extraction_accuracy".to_string(),
            MetricValue::from_samples(&extraction_samples),
        );
        metrics.insert(
            "structure_fidelity".to_string(),
            MetricValue::from_samples(&structure_samples),
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
    fn test_encrypted_binding_has_provenance() {
        assert!(EncryptedBindingBenchmark.provenance().is_some());
    }

    #[test]
    fn test_encrypted_binding_exact_preservation() {
        let config = BenchmarkConfig {
            seed: 42,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = EncryptedBindingBenchmark.run(&config);

        let pres = result.metrics.get("binding_preservation").unwrap();
        assert!(
            (pres.mean - 1.0).abs() < 1e-10,
            "XOR bind must be exactly preserved (masks cancel). Got: {}",
            pres.mean
        );

        let structure = result.metrics.get("structure_fidelity").unwrap();
        assert!(
            (structure.mean - 1.0).abs() < 1e-10,
            "Composite structure must be exactly preserved. Got: {}",
            structure.mean
        );
    }

    #[test]
    fn test_encrypted_binding_role_extraction() {
        let config = BenchmarkConfig {
            seed: 42,
            trials_per_condition: 10,
            ..Default::default()
        };
        let result = EncryptedBindingBenchmark.run(&config);

        let extraction = result.metrics.get("role_extraction_accuracy").unwrap();
        assert!(
            extraction.mean > 0.80,
            "Role extraction should work well with 6 pairs. Got: {}",
            extraction.mean
        );
    }
}
