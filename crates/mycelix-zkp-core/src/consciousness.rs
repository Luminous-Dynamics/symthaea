// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Consciousness tier range proofs.
//!
//! Proves: "my consciousness score ≥ threshold" without revealing the exact score.
//!
//! Used by 6+ Mycelix clusters for consciousness gating:
//! - Governance: voting (0.4), proposals (0.3), constitutional (0.6)
//! - Finance: payments (0.2), TEND matching (0.3)
//! - Commons, Personal, Attribution: basic gating (0.2)
//!
//! Currently ALL consciousness gating sends the score in plaintext.
//! This module replaces plaintext comparisons with real Winterfell STARK proofs.
//!
//! Domain tag: `ZTML:Consciousness:TierProof:v1`

use serde::{Deserialize, Serialize};
#[cfg(feature = "backend-winterfell")]
use sha2::{Digest, Sha256};

#[cfg(feature = "backend-winterfell")]
use crate::domain::tag_consciousness_tier;
#[cfg(feature = "backend-winterfell")]
use crate::error::{ZkpError, ZkpResult};

/// Well-known consciousness gate thresholds (from bridge-common).
pub mod thresholds {
    /// Basic participation (Finance pledge, Attribution)
    pub const BASIC: f64 = 0.2;
    /// Proposal submission (Governance propose, Finance TEND matching)
    pub const PROPOSAL: f64 = 0.3;
    /// Voting rights (Governance vote)
    pub const VOTING: f64 = 0.4;
    /// Constitutional authority (Governance constitutional amendment)
    pub const CONSTITUTIONAL: f64 = 0.6;
}

/// Consciousness tier names for human-readable output.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CivicTier {
    Observer,    // < 0.2
    Participant, // >= 0.2
    Citizen,     // >= 0.3
    Steward,     // >= 0.4
    Guardian,    // >= 0.6
}

impl CivicTier {
    /// Minimum Φ threshold for this tier.
    pub fn min_phi(&self) -> f64 {
        match self {
            CivicTier::Observer => 0.0,
            CivicTier::Participant => thresholds::BASIC,
            CivicTier::Citizen => thresholds::PROPOSAL,
            CivicTier::Steward => thresholds::VOTING,
            CivicTier::Guardian => thresholds::CONSTITUTIONAL,
        }
    }

    /// Convert Φ threshold to Q16.16 fixed-point scaled to [0, 10000].
    /// Φ is in [0.0, 1.0], multiply by 10000 for integer range proof.
    pub fn threshold_scaled(&self) -> u64 {
        (self.min_phi() * 10000.0) as u64
    }
}

/// Request to prove consciousness tier eligibility.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConsciousnessProofRequest {
    /// The actual consciousness score (private — never revealed).
    pub phi_score: f64,
    /// The tier to prove eligibility for (public).
    pub required_tier: CivicTier,
    /// Agent DID (for commitment binding).
    pub agent_did: String,
    /// Random nonce for replay prevention (32 bytes).
    /// Each proof request MUST use a unique nonce.
    pub nonce: [u8; 32],
    /// Unix epoch seconds when this proof was requested.
    /// Verifiers reject proofs outside the validity window.
    pub epoch_secs: u64,
}

/// Maximum age of a consciousness proof before it is considered stale (24 hours).
pub const PROOF_VALIDITY_SECS: u64 = 86_400;

/// Result of consciousness tier proof.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConsciousnessProofResult {
    /// Winterfell STARK proof bytes.
    pub proof_bytes: Vec<u8>,
    /// Commitment to the score (SHA-256, for replay prevention).
    /// Binds: agent_did + score + domain_tag + nonce + epoch_secs.
    pub score_commitment: [u8; 32],
    /// Which tier was proven.
    pub proven_tier: CivicTier,
    /// Domain tag.
    pub domain_tag: String,
    /// Whether the proof was generated (false if score < threshold).
    pub eligible: bool,
    /// Proof generation time in ms.
    pub prove_time_ms: f64,
    /// Nonce used in the commitment (must match the request).
    pub nonce: [u8; 32],
    /// Epoch seconds when the proof was generated.
    pub epoch_secs: u64,
}

/// Generate a REAL Winterfell STARK proof that `phi_score >= tier.min_phi()`.
///
/// Uses the shared range proof circuit from `circuits::range_proof`.
/// The score is scaled to [0, 10000] for integer range verification.
#[cfg(feature = "backend-winterfell")]
pub fn prove_consciousness_tier(
    request: &ConsciousnessProofRequest,
) -> ZkpResult<ConsciousnessProofResult> {
    use std::time::Instant;

    let domain_tag = tag_consciousness_tier();
    let threshold = request.required_tier.threshold_scaled();

    // Scale Φ from [0.0, 1.0] to [0, 10000]
    let score_scaled = (request.phi_score.clamp(0.0, 1.0) * 10000.0) as u64;

    // Check eligibility before expensive proof generation
    if score_scaled < threshold {
        return Ok(ConsciousnessProofResult {
            proof_bytes: vec![],
            score_commitment: [0; 32],
            proven_tier: request.required_tier.clone(),
            domain_tag: domain_tag.as_str().to_string(),
            eligible: false,
            prove_time_ms: 0.0,
            nonce: request.nonce,
            epoch_secs: request.epoch_secs,
        });
    }

    // Commitment to the score (for replay prevention).
    // Includes nonce + epoch_secs to ensure each proof is unique and time-bound.
    let score_commitment = {
        let mut h = Sha256::new();
        h.update(request.agent_did.as_bytes());
        h.update(score_scaled.to_le_bytes());
        h.update(domain_tag.as_bytes());
        h.update(&request.nonce);
        h.update(request.epoch_secs.to_le_bytes());
        let result = h.finalize();
        let mut c = [0u8; 32];
        c.copy_from_slice(&result);
        c
    };

    // Generate REAL Winterfell STARK range proof
    // Proves: score_scaled ∈ [threshold, 10000]
    let prove_start = Instant::now();
    let proof = crate::circuits::range_proof::prove_range(
        score_scaled,
        threshold,
        10000, // Max Φ = 1.0 = 10000 scaled
        score_commitment,
    )
    .map_err(|e| ZkpError::ProvingError(e))?;

    let prove_time = prove_start.elapsed();

    Ok(ConsciousnessProofResult {
        proof_bytes: proof.to_bytes(),
        score_commitment,
        proven_tier: request.required_tier.clone(),
        domain_tag: domain_tag.as_str().to_string(),
        eligible: true,
        prove_time_ms: prove_time.as_secs_f64() * 1000.0,
        nonce: request.nonce,
        epoch_secs: request.epoch_secs,
    })
}

/// Verify a consciousness tier proof using REAL Winterfell STARK verification.
///
/// Also checks freshness: rejects proofs older than `PROOF_VALIDITY_SECS`.
/// The `current_epoch_secs` parameter should be the current Unix timestamp.
#[cfg(feature = "backend-winterfell")]
pub fn verify_consciousness_tier(
    proof_bytes: &[u8],
    required_tier: &CivicTier,
    score_commitment: &[u8; 32],
    proof_epoch_secs: u64,
    current_epoch_secs: u64,
) -> ZkpResult<bool> {
    use winterfell::Proof;

    if proof_bytes.is_empty() {
        return Ok(false);
    }

    // Reject stale proofs (older than validity window)
    if current_epoch_secs > proof_epoch_secs + PROOF_VALIDITY_SECS {
        return Err(ZkpError::VerificationFailed(format!(
            "Proof expired: generated at {}, current {}, validity {}s",
            proof_epoch_secs, current_epoch_secs, PROOF_VALIDITY_SECS
        )));
    }

    // Reject proofs from the future (clock skew tolerance: 5 minutes)
    if proof_epoch_secs > current_epoch_secs + 300 {
        return Err(ZkpError::VerificationFailed(format!(
            "Proof from the future: generated at {}, current {}",
            proof_epoch_secs, current_epoch_secs
        )));
    }

    let proof = Proof::from_bytes(proof_bytes)
        .map_err(|e| ZkpError::InvalidProofFormat(format!("Winterfell: {:?}", e)))?;

    let threshold = required_tier.threshold_scaled();

    crate::circuits::range_proof::verify_range(proof, threshold, 10000, *score_commitment)
        .map(|_| true)
        .or_else(|_| {
            // Verification failed — proof is invalid
            Ok(false)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: current epoch for tests (2026-04-10 ~ 1,776,000,000).
    fn test_epoch() -> u64 {
        1_776_000_000
    }

    /// Helper: unique nonce for tests.
    fn test_nonce(seed: u8) -> [u8; 32] {
        let mut n = [0u8; 32];
        n[0] = seed;
        n[31] = seed.wrapping_mul(7);
        n
    }

    #[cfg(feature = "backend-winterfell")]
    #[test]
    fn test_prove_voting_tier() {
        let request = ConsciousnessProofRequest {
            phi_score: 0.55, // Above voting threshold (0.4)
            required_tier: CivicTier::Steward,
            agent_did: "did:mycelix:test_agent".to_string(),
            nonce: test_nonce(1),
            epoch_secs: test_epoch(),
        };

        let result = prove_consciousness_tier(&request).unwrap();
        assert!(result.eligible);
        assert!(!result.proof_bytes.is_empty());
        assert!(result.prove_time_ms > 0.0);
        assert_eq!(result.nonce, test_nonce(1));
        assert_eq!(result.epoch_secs, test_epoch());
        println!(
            "Voting tier proof: {:.1}ms, {} bytes",
            result.prove_time_ms,
            result.proof_bytes.len()
        );

        // Verify the proof (current time = same epoch)
        let valid = verify_consciousness_tier(
            &result.proof_bytes,
            &CivicTier::Steward,
            &result.score_commitment,
            result.epoch_secs,
            test_epoch(),
        )
        .unwrap();
        assert!(valid, "real STARK proof must verify");
    }

    #[cfg(feature = "backend-winterfell")]
    #[test]
    fn test_below_threshold_not_eligible() {
        let request = ConsciousnessProofRequest {
            phi_score: 0.15, // Below basic threshold (0.2)
            required_tier: CivicTier::Participant,
            agent_did: "did:mycelix:low_phi".to_string(),
            nonce: test_nonce(2),
            epoch_secs: test_epoch(),
        };

        let result = prove_consciousness_tier(&request).unwrap();
        assert!(!result.eligible);
        assert!(result.proof_bytes.is_empty());
    }

    #[cfg(feature = "backend-winterfell")]
    #[test]
    fn test_constitutional_tier() {
        let request = ConsciousnessProofRequest {
            phi_score: 0.75, // Above constitutional threshold (0.6)
            required_tier: CivicTier::Guardian,
            agent_did: "did:mycelix:guardian".to_string(),
            nonce: test_nonce(3),
            epoch_secs: test_epoch(),
        };

        let result = prove_consciousness_tier(&request).unwrap();
        assert!(result.eligible);

        let valid = verify_consciousness_tier(
            &result.proof_bytes,
            &CivicTier::Guardian,
            &result.score_commitment,
            result.epoch_secs,
            test_epoch(),
        )
        .unwrap();
        assert!(valid);
    }

    #[cfg(feature = "backend-winterfell")]
    #[test]
    fn test_wrong_tier_fails() {
        // Prove Participant (0.2) but verify against Guardian (0.6)
        let request = ConsciousnessProofRequest {
            phi_score: 0.25,
            required_tier: CivicTier::Participant,
            agent_did: "did:mycelix:participant".to_string(),
            nonce: test_nonce(4),
            epoch_secs: test_epoch(),
        };

        let result = prove_consciousness_tier(&request).unwrap();
        assert!(result.eligible); // Eligible for Participant

        // But verifying against Guardian should fail
        let valid = verify_consciousness_tier(
            &result.proof_bytes,
            &CivicTier::Guardian,
            &result.score_commitment,
            result.epoch_secs,
            test_epoch(),
        )
        .unwrap();
        assert!(!valid, "Participant proof must not verify as Guardian");
    }

    #[cfg(feature = "backend-winterfell")]
    #[test]
    fn test_different_nonces_produce_different_commitments() {
        let req1 = ConsciousnessProofRequest {
            phi_score: 0.55,
            required_tier: CivicTier::Steward,
            agent_did: "did:mycelix:same_agent".to_string(),
            nonce: test_nonce(10),
            epoch_secs: test_epoch(),
        };
        let req2 = ConsciousnessProofRequest {
            phi_score: 0.55,
            required_tier: CivicTier::Steward,
            agent_did: "did:mycelix:same_agent".to_string(),
            nonce: test_nonce(11), // Different nonce
            epoch_secs: test_epoch(),
        };

        let r1 = prove_consciousness_tier(&req1).unwrap();
        let r2 = prove_consciousness_tier(&req2).unwrap();

        assert_ne!(
            r1.score_commitment, r2.score_commitment,
            "Different nonces must produce different commitments (replay prevention)"
        );
    }

    #[cfg(feature = "backend-winterfell")]
    #[test]
    fn test_expired_proof_rejected() {
        let request = ConsciousnessProofRequest {
            phi_score: 0.55,
            required_tier: CivicTier::Steward,
            agent_did: "did:mycelix:test".to_string(),
            nonce: test_nonce(20),
            epoch_secs: test_epoch(),
        };

        let result = prove_consciousness_tier(&request).unwrap();
        assert!(result.eligible);

        // Verify with current time 25 hours later (beyond 24h validity)
        let stale_time = test_epoch() + PROOF_VALIDITY_SECS + 3600;
        let err = verify_consciousness_tier(
            &result.proof_bytes,
            &CivicTier::Steward,
            &result.score_commitment,
            result.epoch_secs,
            stale_time,
        );
        assert!(err.is_err(), "Expired proof must be rejected");
    }

    #[cfg(feature = "backend-winterfell")]
    #[test]
    fn test_future_proof_rejected() {
        let request = ConsciousnessProofRequest {
            phi_score: 0.55,
            required_tier: CivicTier::Steward,
            agent_did: "did:mycelix:test".to_string(),
            nonce: test_nonce(21),
            epoch_secs: test_epoch() + 600, // 10 minutes in the future
        };

        let result = prove_consciousness_tier(&request).unwrap();
        assert!(result.eligible);

        // Verify with "current" time that is 10 minutes before the proof epoch
        let err = verify_consciousness_tier(
            &result.proof_bytes,
            &CivicTier::Steward,
            &result.score_commitment,
            result.epoch_secs,
            test_epoch(), // 10 min before proof epoch
        );
        assert!(err.is_err(), "Proof from the future must be rejected");
    }

    #[test]
    fn test_tier_thresholds() {
        assert_eq!(CivicTier::Participant.threshold_scaled(), 2000);
        assert_eq!(CivicTier::Citizen.threshold_scaled(), 3000);
        assert_eq!(CivicTier::Steward.threshold_scaled(), 4000);
        assert_eq!(CivicTier::Guardian.threshold_scaled(), 6000);
    }
}
