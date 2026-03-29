#![allow(

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
    clippy::absurd_extreme_comparisons,
    clippy::manual_range_contains,
    clippy::needless_range_loop,
    clippy::manual_is_multiple_of
)]
//! K-Vector Zero-Knowledge Proofs
//!
//! This crate provides STARK-based zero-knowledge proofs for validating
//! K-Vector components are in the valid range [0, 1] without revealing
//! the actual values.
//!
//! # Overview
//!
//! K-Vectors are 8-dimensional trust metrics used in the Mycelix network:
//! - k_r: Reputation score
//! - k_a: Activity score
//! - k_i: Integrity score
//! - k_p: Performance score
//! - k_m: Membership duration score
//! - k_s: Stake weight score
//! - k_h: Historical consistency score
//! - k_topo: Network topology contribution score
//!
//! All components must be in the range [0, 1]. This crate allows proving
//! this constraint without revealing the actual values.
//!
//! # Technical Details
//!
//! The proof system uses:
//! - Winterfell STARK library for proof generation/verification
//! - 128-bit field elements (f128) for security
//! - Fixed-point scaling (4 decimal places) for f32 values
//! - Bit decomposition for range proofs
//!
//! # Example
//!
//! ```ignore
//! use kvector_zkp::{KVectorWitness, KVectorRangeProof};
//!
//! // Create a witness with K-Vector values
//! let witness = KVectorWitness {
//!     k_r: 0.8,
//!     k_a: 0.7,
//!     k_i: 0.9,
//!     k_p: 0.6,
//!     k_m: 0.5,
//!     k_s: 0.4,
//!     k_h: 0.85,
//!     k_topo: 0.75,
//! };
//!
//! // Generate a proof
//! let proof = KVectorRangeProof::prove(&witness)?;
//!
//! // Verify the proof
//! proof.verify()?;
//! ```

pub mod air;
pub mod error;
pub mod optimized_prover;
pub mod proof;
pub mod prover;
pub mod verifier;

// Re-exports for convenience
pub use air::{columns, KVectorPublicInputs, BITS_PER_VALUE, NUM_COMPONENTS, SCALE_FACTOR, TRACE_LENGTH};
pub use error::{ZkpError, ZkpResult};
pub use optimized_prover::{OptimizedKVectorProver, OptimizedProverBuilder, SecurityLevel};
pub use proof::{KVectorRangeProof, SerializableProof};
pub use prover::{KVectorProver, KVectorTrace, KVectorWitness};
pub use verifier::{
    verify_kvector_proof,
    verify_kvector_proof_production,
    verify_kvector_proof_secure,
    PRODUCTION_MIN_SECURITY_BITS,
};

/// Compute the trust score from K-Vector components
///
/// Formula: T = 0.25×k_r + 0.15×k_a + 0.20×k_i + 0.15×k_p + 0.05×k_m + 0.10×k_s + 0.05×k_h + 0.05×k_topo
pub fn compute_trust_score(witness: &KVectorWitness) -> f32 {
    0.25 * witness.k_r
        + 0.15 * witness.k_a
        + 0.20 * witness.k_i
        + 0.15 * witness.k_p
        + 0.05 * witness.k_m
        + 0.10 * witness.k_s
        + 0.05 * witness.k_h
        + 0.05 * witness.k_topo
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trust_score_calculation() {
        let witness = KVectorWitness {
            k_r: 1.0,
            k_a: 1.0,
            k_i: 1.0,
            k_p: 1.0,
            k_m: 1.0,
            k_s: 1.0,
            k_h: 1.0,
            k_topo: 1.0,
        };

        let score = compute_trust_score(&witness);
        assert!((score - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_trust_score_weighted() {
        let witness = KVectorWitness {
            k_r: 0.8,  // 0.25 * 0.8 = 0.20
            k_a: 0.6,  // 0.15 * 0.6 = 0.09
            k_i: 0.9,  // 0.20 * 0.9 = 0.18
            k_p: 0.7,  // 0.15 * 0.7 = 0.105
            k_m: 0.5,  // 0.05 * 0.5 = 0.025
            k_s: 0.4,  // 0.10 * 0.4 = 0.04
            k_h: 0.6,  // 0.05 * 0.6 = 0.03
            k_topo: 0.7, // 0.05 * 0.7 = 0.035
        };

        let score = compute_trust_score(&witness);
        let expected = 0.20 + 0.09 + 0.18 + 0.105 + 0.025 + 0.04 + 0.03 + 0.035;
        assert!((score - expected).abs() < 0.0001);
    }

    #[test]
    fn test_witness_from_array() {
        let values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let witness = KVectorWitness::from_array(values);

        assert!((witness.k_r - 0.1).abs() < 0.0001);
        assert!((witness.k_topo - 0.8).abs() < 0.0001);
    }

    #[test]
    fn test_witness_to_array() {
        let witness = KVectorWitness {
            k_r: 0.1,
            k_a: 0.2,
            k_i: 0.3,
            k_p: 0.4,
            k_m: 0.5,
            k_s: 0.6,
            k_h: 0.7,
            k_topo: 0.8,
        };

        let array = witness.to_array();
        assert_eq!(array.len(), 8);
        assert!((array[0] - 0.1).abs() < 0.0001);
        assert!((array[7] - 0.8).abs() < 0.0001);
    }
}
