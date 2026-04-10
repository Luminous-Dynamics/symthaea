// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Miden VM ZK-STARK Consciousness Proofs
//!
//! Zero-knowledge consciousness tier attestation using Miden VM.
//! Unlike Winterfell (which is NOT zero-knowledge), Miden's advice provider
//! ensures the secret Phi score is never revealed to the verifier.
//!
//! ## Architecture
//!
//! ```text
//! Client (native):
//!   1. phi_score → advice stack (secret, never revealed)
//!   2. threshold + commitment → operand stack (public)
//!   3. Miden prover executes program → proof bytes (~80-100KB)
//!
//! Verifier (WASM or native):
//!   1. Receives proof bytes + public inputs (threshold, commitment)
//!   2. miden_verifier::verify() → bool
//!   3. Phi score remains unknown to verifier
//! ```
//!
//! ## Feature Gate
//!
//! Enable with `backend-miden` feature in Cargo.toml.

use crate::consciousness::ConsciousnessTier;
use crate::error::ZkpError;
use sha2::{Digest, Sha256};

/// Result type for Miden ZKP operations.
pub type MidenResult<T> = Result<T, ZkpError>;

/// Maximum proof size for Miden STARK proofs (150KB).
/// Miden proofs are typically 80-136KB depending on program complexity.
pub const MAX_MIDEN_PROOF_SIZE: usize = 150_000;

/// Proof system identifier for serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MidenProofSystem {
    /// Miden VM ZK-STARK (actual zero-knowledge)
    MidenZkStark,
}

/// Result of generating a Miden consciousness proof.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MidenConsciousnessProof {
    /// STARK proof bytes (80-136KB typical)
    pub proof_bytes: Vec<u8>,
    /// SHA-256 commitment to the score: H(phi_score || domain_tag)
    pub score_commitment: [u8; 32],
    /// The tier being proven (public — verifier knows which tier)
    pub proven_tier: ConsciousnessTier,
    /// Proof system identifier
    pub proof_system: MidenProofSystem,
    /// Stack outputs from the Miden program (public)
    pub stack_outputs: Vec<u64>,
}

/// Compute the score commitment: SHA-256(phi_bytes || "MYCELIX:ConsciousnessTier:v2")
pub fn compute_score_commitment(phi_score: f64) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(phi_score.to_le_bytes());
    hasher.update(b"MYCELIX:ConsciousnessTier:v2");
    let result = hasher.finalize();
    let mut commitment = [0u8; 32];
    commitment.copy_from_slice(&result);
    commitment
}

/// Map a consciousness tier to its minimum Phi threshold.
///
/// These thresholds must match the values used in bridge-common's
/// `consciousness_profile.rs` tier computation.
pub fn tier_threshold(tier: &ConsciousnessTier) -> f64 {
    match tier {
        ConsciousnessTier::Observer => 0.0,
        ConsciousnessTier::Participant => 0.2,
        ConsciousnessTier::Citizen => 0.3,
        ConsciousnessTier::Steward => 0.4,
        ConsciousnessTier::Guardian => 0.6,
    }
}

/// Validate a Miden consciousness proof structurally.
///
/// This can run in WASM without the Miden prover dependency.
/// Checks proof size, commitment non-zero, and tier validity.
pub fn validate_miden_proof_structure(proof: &MidenConsciousnessProof) -> MidenResult<()> {
    if proof.proof_bytes.is_empty() {
        return Err(ZkpError::ProvingError("Empty proof bytes".into()));
    }
    if proof.proof_bytes.len() > MAX_MIDEN_PROOF_SIZE {
        return Err(ZkpError::ProvingError(format!(
            "Proof too large: {} > {}",
            proof.proof_bytes.len(),
            MAX_MIDEN_PROOF_SIZE
        )));
    }
    if proof.score_commitment == [0u8; 32] {
        return Err(ZkpError::ProvingError("Zero score commitment".into()));
    }
    if proof.proof_system != MidenProofSystem::MidenZkStark {
        return Err(ZkpError::ProvingError("Wrong proof system".into()));
    }
    Ok(())
}

// ============================================================================
// Tests (run without Miden dependency — structural validation only)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_commitment_deterministic() {
        let c1 = compute_score_commitment(0.75);
        let c2 = compute_score_commitment(0.75);
        assert_eq!(c1, c2, "Same score should produce same commitment");
    }

    #[test]
    fn test_score_commitment_different_scores() {
        let c1 = compute_score_commitment(0.5);
        let c2 = compute_score_commitment(0.6);
        assert_ne!(c1, c2, "Different scores should produce different commitments");
    }

    #[test]
    fn test_score_commitment_non_zero() {
        let c = compute_score_commitment(0.0);
        assert_ne!(c, [0u8; 32], "Even zero score should produce non-zero commitment");
    }

    #[test]
    fn test_tier_thresholds_monotonic() {
        let tiers = [
            ConsciousnessTier::Observer,
            ConsciousnessTier::Participant,
            ConsciousnessTier::Citizen,
            ConsciousnessTier::Steward,
            ConsciousnessTier::Guardian,
        ];
        for pair in tiers.windows(2) {
            assert!(
                tier_threshold(&pair[0]) < tier_threshold(&pair[1]),
                "{:?} threshold should be less than {:?}",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn test_validate_empty_proof_rejected() {
        let proof = MidenConsciousnessProof {
            proof_bytes: vec![],
            score_commitment: [0xAA; 32],
            proven_tier: ConsciousnessTier::Citizen,
            proof_system: MidenProofSystem::MidenZkStark,
            stack_outputs: vec![],
        };
        assert!(validate_miden_proof_structure(&proof).is_err());
    }

    #[test]
    fn test_validate_oversized_proof_rejected() {
        let proof = MidenConsciousnessProof {
            proof_bytes: vec![0u8; MAX_MIDEN_PROOF_SIZE + 1],
            score_commitment: [0xBB; 32],
            proven_tier: ConsciousnessTier::Steward,
            proof_system: MidenProofSystem::MidenZkStark,
            stack_outputs: vec![],
        };
        assert!(validate_miden_proof_structure(&proof).is_err());
    }

    #[test]
    fn test_validate_zero_commitment_rejected() {
        let proof = MidenConsciousnessProof {
            proof_bytes: vec![1, 2, 3],
            score_commitment: [0u8; 32],
            proven_tier: ConsciousnessTier::Guardian,
            proof_system: MidenProofSystem::MidenZkStark,
            stack_outputs: vec![],
        };
        assert!(validate_miden_proof_structure(&proof).is_err());
    }

    #[test]
    fn test_validate_valid_proof_accepted() {
        let proof = MidenConsciousnessProof {
            proof_bytes: vec![0u8; 80_000], // 80KB — typical Miden proof
            score_commitment: compute_score_commitment(0.75),
            proven_tier: ConsciousnessTier::Steward,
            proof_system: MidenProofSystem::MidenZkStark,
            stack_outputs: vec![1], // 1 = tier check passed
        };
        assert!(validate_miden_proof_structure(&proof).is_ok());
    }

    #[test]
    fn test_miden_proof_serde_roundtrip() {
        let proof = MidenConsciousnessProof {
            proof_bytes: vec![1, 2, 3, 4, 5],
            score_commitment: compute_score_commitment(0.5),
            proven_tier: ConsciousnessTier::Citizen,
            proof_system: MidenProofSystem::MidenZkStark,
            stack_outputs: vec![1],
        };
        let json = serde_json::to_string(&proof).unwrap();
        let back: MidenConsciousnessProof = serde_json::from_str(&json).unwrap();
        assert_eq!(proof.proof_bytes, back.proof_bytes);
        assert_eq!(proof.score_commitment, back.score_commitment);
        assert_eq!(proof.proven_tier, back.proven_tier);
    }

    #[test]
    fn test_proof_size_at_limit_accepted() {
        let proof = MidenConsciousnessProof {
            proof_bytes: vec![0u8; MAX_MIDEN_PROOF_SIZE], // exactly at limit
            score_commitment: [0xFF; 32],
            proven_tier: ConsciousnessTier::Guardian,
            proof_system: MidenProofSystem::MidenZkStark,
            stack_outputs: vec![],
        };
        assert!(validate_miden_proof_structure(&proof).is_ok());
    }
}
