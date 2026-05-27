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
}

/// Result of consciousness tier proof.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConsciousnessProofResult {
    /// Winterfell STARK proof bytes.
    pub proof_bytes: Vec<u8>,
    /// Commitment to the score (SHA-256, for replay prevention).
    pub score_commitment: [u8; 32],
    /// Which tier was proven.
    pub proven_tier: CivicTier,
    /// Domain tag.
    pub domain_tag: String,
    /// Whether the proof was generated (false if score < threshold).
    pub eligible: bool,
    /// Proof generation time in ms.
    pub prove_time_ms: f64,
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
        });
    }

    // Commitment to the score (for replay prevention)
    let score_commitment = {
        let mut h = Sha256::new();
        h.update(request.agent_did.as_bytes());
        h.update(score_scaled.to_le_bytes());
        h.update(domain_tag.as_bytes());
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
    })
}

/// Verify a consciousness tier proof using REAL Winterfell STARK verification.
#[cfg(feature = "backend-winterfell")]
pub fn verify_consciousness_tier(
    proof_bytes: &[u8],
    required_tier: &CivicTier,
    score_commitment: &[u8; 32],
) -> ZkpResult<bool> {
    use winterfell::Proof;

    if proof_bytes.is_empty() {
        return Ok(false);
    }

    let proof = Proof::from_bytes(proof_bytes)
        .map_err(|e| ZkpError::InvalidProofFormat(format!("Winterfell: {:?}", e)))?;

    let threshold = required_tier.threshold_scaled();

    crate::circuits::range_proof::verify_range(proof, threshold, 10000, *score_commitment)
        .map(|_| true)
        .or_else(|e| {
            // Verification failed — proof is invalid
            Ok(false)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "backend-winterfell")]
    #[test]
    fn test_prove_voting_tier() {
        let request = ConsciousnessProofRequest {
            phi_score: 0.55, // Above voting threshold (0.4)
            required_tier: CivicTier::Steward,
            agent_did: "did:mycelix:test_agent".to_string(),
        };

        let result = prove_consciousness_tier(&request).unwrap();
        assert!(result.eligible);
        assert!(!result.proof_bytes.is_empty());
        assert!(result.prove_time_ms > 0.0);
        println!(
            "Voting tier proof: {:.1}ms, {} bytes",
            result.prove_time_ms,
            result.proof_bytes.len()
        );

        // Verify the proof
        let valid = verify_consciousness_tier(
            &result.proof_bytes,
            &CivicTier::Steward,
            &result.score_commitment,
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
        };

        let result = prove_consciousness_tier(&request).unwrap();
        assert!(result.eligible);

        let valid = verify_consciousness_tier(
            &result.proof_bytes,
            &CivicTier::Guardian,
            &result.score_commitment,
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
        };

        let result = prove_consciousness_tier(&request).unwrap();
        assert!(result.eligible); // Eligible for Participant

        // But verifying against Guardian should fail
        let valid = verify_consciousness_tier(
            &result.proof_bytes,
            &CivicTier::Guardian,
            &result.score_commitment,
        )
        .unwrap();
        assert!(!valid, "Participant proof must not verify as Guardian");
    }

    #[test]
    fn test_tier_thresholds() {
        assert_eq!(CivicTier::Participant.threshold_scaled(), 2000);
        assert_eq!(CivicTier::Citizen.threshold_scaled(), 3000);
        assert_eq!(CivicTier::Steward.threshold_scaled(), 4000);
        assert_eq!(CivicTier::Guardian.threshold_scaled(), 6000);
    }
}
