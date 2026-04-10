// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cognitive cycle integrity proofs.
//!
//! Proves: "Symthaea's cognitive loop ran correctly from perception to output"
//! without revealing internal state (HDC vectors, CfC weights, moral reasoning).
//!
//! ## Chain of Commitments
//!
//! Rather than proving the entire ~70KB cycle in one circuit, we commit to
//! each phase independently and prove the chain:
//!
//! ```text
//! Phase 1: commit(perception) → hash₁
//! Phase 2: prove CfC step(hash₁) → hash₂  [~512 AND for 512 neurons]
//! Phase 3: prove Phi ≥ threshold(hash₂) → hash₃  [~32 AND, range proof]
//! Phase 4: prove verdict(hash₃) → hash₄  [~16 AND, deterministic]
//! Chain:   prove hash₁→hash₂→hash₃→hash₄  [~256 AND for hash chain]
//! ```
//!
//! Total: ~816 AND constraints for a full cycle integrity proof.
//!
//! ## Public Output
//!
//! The verifier sees:
//! - `consciousness_level: f64` (Phi)
//! - `moral_verdict: String` ("Allow", "Warn", or "Veto")
//! - `cycle_commitment: [u8; 32]` (SHA-256 chain of all phase commitments)
//!
//! The verifier does NOT see:
//! - HDC perception vectors (16,384D)
//! - CfC neuron states and weights
//! - Moral reasoning chain
//! - Internal prediction errors or quality metrics

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// A cycle integrity proof — proves the cognitive loop ran correctly.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CycleIntegrityProof {
    /// Commitment to Phase 1 (perception): hash of HDC vector + moral score
    pub perception_commitment: [u8; 32],
    /// Commitment to Phase 2 (dynamics): hash of CfC output + prediction
    pub dynamics_commitment: [u8; 32],
    /// Commitment to Phase 3 (feedback): hash of Phi + quality + verdict
    pub feedback_commitment: [u8; 32],
    /// Commitment to Phase 4 (output): hash of final CycleResult
    pub output_commitment: [u8; 32],
    /// Chain commitment: hash(h1 || h2 || h3 || h4) — binds all phases
    pub chain_commitment: [u8; 32],
    /// Public: consciousness level (Phi)
    pub consciousness_level: f64,
    /// Public: moral verdict
    pub moral_verdict: String,
    /// Public: cycle number
    pub cycle_number: u64,
    /// STARK proof bytes (optional — for full cryptographic verification)
    pub proof_bytes: Vec<u8>,
    /// Domain tag
    pub domain_tag: String,
}

/// Phase commitment input — what gets hashed for each phase.
#[derive(Clone, Debug)]
pub struct PhaseCommitmentInput {
    pub phase_name: String,
    pub data_bytes: Vec<u8>,
}

/// Compute a phase commitment: SHA-256(phase_name || data_bytes)
pub fn commit_phase(input: &PhaseCommitmentInput) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"ZTML:CycleIntegrity:");
    hasher.update(input.phase_name.as_bytes());
    hasher.update(b":");
    hasher.update(&input.data_bytes);
    let result = hasher.finalize();
    let mut commitment = [0u8; 32];
    commitment.copy_from_slice(&result);
    commitment
}

/// Compute the chain commitment from 4 phase commitments.
pub fn compute_chain_commitment(
    perception: &[u8; 32],
    dynamics: &[u8; 32],
    feedback: &[u8; 32],
    output: &[u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"ZTML:ChainCommitment:v1:");
    hasher.update(perception);
    hasher.update(dynamics);
    hasher.update(feedback);
    hasher.update(output);
    let result = hasher.finalize();
    let mut chain = [0u8; 32];
    chain.copy_from_slice(&result);
    chain
}

/// Generate a cycle integrity proof from phase data.
///
/// This computes commitments and chains them. The STARK proof
/// (proving the chain is valid) is generated separately via Binius.
pub fn generate_cycle_proof(
    perception_data: &[u8],
    dynamics_data: &[u8],
    feedback_data: &[u8],
    output_data: &[u8],
    consciousness_level: f64,
    moral_verdict: &str,
    cycle_number: u64,
) -> CycleIntegrityProof {
    let perception_commitment = commit_phase(&PhaseCommitmentInput {
        phase_name: "perception".to_string(),
        data_bytes: perception_data.to_vec(),
    });

    let dynamics_commitment = commit_phase(&PhaseCommitmentInput {
        phase_name: "dynamics".to_string(),
        data_bytes: dynamics_data.to_vec(),
    });

    let feedback_commitment = commit_phase(&PhaseCommitmentInput {
        phase_name: "feedback".to_string(),
        data_bytes: feedback_data.to_vec(),
    });

    let output_commitment = commit_phase(&PhaseCommitmentInput {
        phase_name: "output".to_string(),
        data_bytes: output_data.to_vec(),
    });

    let chain_commitment = compute_chain_commitment(
        &perception_commitment,
        &dynamics_commitment,
        &feedback_commitment,
        &output_commitment,
    );

    CycleIntegrityProof {
        perception_commitment,
        dynamics_commitment,
        feedback_commitment,
        output_commitment,
        chain_commitment,
        consciousness_level,
        moral_verdict: moral_verdict.to_string(),
        cycle_number,
        proof_bytes: vec![], // STARK proof generated separately
        domain_tag: "ZTML:Symthaea:CycleIntegrity:v1".to_string(),
    }
}

/// Verify that phase commitments chain correctly.
///
/// This is the structural verification (no STARK needed).
/// Full cryptographic verification uses the STARK proof bytes.
pub fn verify_chain(proof: &CycleIntegrityProof) -> bool {
    let expected_chain = compute_chain_commitment(
        &proof.perception_commitment,
        &proof.dynamics_commitment,
        &proof.feedback_commitment,
        &proof.output_commitment,
    );
    expected_chain == proof.chain_commitment
}

/// Verify consciousness level is in valid range.
pub fn verify_consciousness_range(proof: &CycleIntegrityProof) -> bool {
    proof.consciousness_level >= 0.0 && proof.consciousness_level <= 100.0
}

/// Verify moral verdict is one of the allowed values.
pub fn verify_moral_verdict(proof: &CycleIntegrityProof) -> bool {
    matches!(
        proof.moral_verdict.as_str(),
        "Allow" | "Warn" | "Caution" | "Veto"
    )
}

/// Full structural verification of a cycle integrity proof.
pub fn verify_cycle_proof(proof: &CycleIntegrityProof) -> Result<(), String> {
    if !verify_chain(proof) {
        return Err("Chain commitment mismatch".to_string());
    }
    if !verify_consciousness_range(proof) {
        return Err(format!(
            "Consciousness level out of range: {}",
            proof.consciousness_level
        ));
    }
    if !verify_moral_verdict(proof) {
        return Err(format!("Invalid moral verdict: {}", proof.moral_verdict));
    }
    if proof.domain_tag != "ZTML:Symthaea:CycleIntegrity:v1" {
        return Err("Wrong domain tag".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_proof() -> CycleIntegrityProof {
        generate_cycle_proof(
            b"perception: hdv_16384d + moral_score=0.7",
            b"dynamics: cfc_output_512d + prediction_error=0.15",
            b"feedback: phi=3.2 + quality=0.85 + verdict=Allow",
            b"output: wisdom_hv_16384bit + language=None",
            3.2,
            "Allow",
            42,
        )
    }

    #[test]
    fn test_generate_and_verify_cycle_proof() {
        let proof = sample_proof();
        assert!(verify_cycle_proof(&proof).is_ok());
        assert!(verify_chain(&proof));
        assert!(verify_consciousness_range(&proof));
        assert!(verify_moral_verdict(&proof));
    }

    #[test]
    fn test_chain_commitment_deterministic() {
        let p1 = sample_proof();
        let p2 = sample_proof();
        assert_eq!(
            p1.chain_commitment, p2.chain_commitment,
            "same inputs must produce same chain commitment"
        );
    }

    #[test]
    fn test_tampered_phase_breaks_chain() {
        let mut proof = sample_proof();
        // Tamper with dynamics commitment
        proof.dynamics_commitment[0] ^= 0xFF;
        assert!(!verify_chain(&proof), "tampered phase must break chain");
        assert!(verify_cycle_proof(&proof).is_err());
    }

    #[test]
    fn test_invalid_consciousness_rejected() {
        let mut proof = sample_proof();
        proof.consciousness_level = -1.0;
        assert!(verify_cycle_proof(&proof).is_err());

        proof.consciousness_level = 200.0;
        assert!(verify_cycle_proof(&proof).is_err());
    }

    #[test]
    fn test_invalid_verdict_rejected() {
        let mut proof = sample_proof();
        proof.moral_verdict = "Execute".to_string();
        assert!(verify_cycle_proof(&proof).is_err());
    }

    #[test]
    fn test_different_inputs_different_commitments() {
        let p1 = generate_cycle_proof(
            b"perception_A",
            b"dynamics_A",
            b"feedback_A",
            b"output_A",
            3.0,
            "Allow",
            1,
        );
        let p2 = generate_cycle_proof(
            b"perception_B",
            b"dynamics_B",
            b"feedback_B",
            b"output_B",
            4.0,
            "Warn",
            2,
        );
        assert_ne!(p1.chain_commitment, p2.chain_commitment);
        assert_ne!(p1.perception_commitment, p2.perception_commitment);
    }

    #[test]
    fn test_moral_verdicts_accepted() {
        for verdict in &["Allow", "Warn", "Caution", "Veto"] {
            let proof = generate_cycle_proof(b"p", b"d", b"f", b"o", 1.0, verdict, 1);
            assert!(
                verify_moral_verdict(&proof),
                "verdict '{}' should be accepted",
                verdict
            );
        }
    }

    #[test]
    fn test_serialization_roundtrip() {
        let proof = sample_proof();
        let json = serde_json::to_string(&proof).unwrap();
        let parsed: CycleIntegrityProof = serde_json::from_str(&json).unwrap();
        assert_eq!(proof.chain_commitment, parsed.chain_commitment);
        assert_eq!(proof.consciousness_level, parsed.consciousness_level);
        assert_eq!(proof.moral_verdict, parsed.moral_verdict);
    }
}
