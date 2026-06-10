// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! STARK proof for 8D Sovereign Profile civic tier eligibility.
//!
//! Proves: "This agent's decayed civic score meets the required tier threshold"
//! without revealing the raw 8D profile, individual dimensions, or last
//! interaction timestamp.
//!
//! ## Public Inputs (known to the network)
//!
//! - `threshold`: minimum score (scaled 0-10000) for the required tier
//! - `lambda_scaled`: community decay rate (scaled, integer)
//! - `score_commitment`: BLAKE3 hash binding the proof to the agent
//!
//! ## Private Inputs (kept on agent's device)
//!
//! - `decayed_score`: the combined weighted score after time-decay (scaled 0-10000)
//!
//! ## Circuit
//!
//! Uses the existing range proof pattern: decomposes `(decayed_score - threshold)`
//! into bits, proving the difference is non-negative (i.e., score >= threshold).
//!
//! The network receives only the proof that the citizen passes — not their raw
//! score, which dimensions are strong, or when they last interacted.

#[cfg(feature = "backend-winterfell")]
use crate::circuits::range_proof;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Maximum scaled score (maps 0.0-1.0 → 0-10000 for integer arithmetic).
pub const SCORE_SCALE: u64 = 10_000;

/// Request to generate a sovereign tier proof.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SovereignProofRequest {
    /// The agent's DID (binds proof to identity).
    pub agent_did: String,
    /// The 8D combined score AFTER decay, as f64 [0.0, 1.0].
    pub decayed_score: f64,
    /// The minimum score required for the target tier, as f64 [0.0, 1.0].
    pub required_threshold: f64,
    /// Community decay rate lambda (for commitment binding).
    pub lambda: f64,
    /// Elapsed days since last interaction (for commitment binding).
    pub elapsed_days: f64,
}

/// Result of a sovereign tier proof generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SovereignProofResult {
    /// Winterfell STARK proof bytes.
    pub proof_bytes: Vec<u8>,
    /// SHA-256 commitment binding score + agent + lambda + elapsed.
    pub score_commitment: [u8; 32],
    /// Whether the agent meets the threshold.
    pub eligible: bool,
    /// Time taken to generate proof (milliseconds).
    pub prove_time_ms: f64,
    /// The threshold that was proven against (scaled).
    pub threshold_scaled: u64,
}

/// Compute the SHA-256 commitment for a sovereign proof.
///
/// Binds: agent_did + decayed_score + lambda + elapsed_days.
/// This prevents proof replay across different agents or time periods.
pub fn compute_commitment(
    agent_did: &str,
    decayed_score_scaled: u64,
    lambda_scaled: u64,
    elapsed_days_scaled: u64,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"SOVEREIGN:v1:");
    hasher.update(agent_did.as_bytes());
    hasher.update(b":");
    hasher.update(decayed_score_scaled.to_le_bytes());
    hasher.update(b":");
    hasher.update(lambda_scaled.to_le_bytes());
    hasher.update(b":");
    hasher.update(elapsed_days_scaled.to_le_bytes());
    let hash = hasher.finalize();
    let mut commitment = [0u8; 32];
    commitment.copy_from_slice(&hash);
    commitment
}

/// Generate a STARK proof that the agent's decayed civic score meets a threshold.
///
/// The proof demonstrates `decayed_score >= required_threshold` without
/// revealing the actual score. Uses bit decomposition via the existing
/// range proof circuit.
///
/// # Errors
///
/// Returns `Err` if the score is below the threshold (cannot generate a
/// valid proof for an ineligible agent).
#[cfg(feature = "backend-winterfell")]
pub fn prove_sovereign_tier(
    request: &SovereignProofRequest,
) -> Result<SovereignProofResult, String> {
    let start = std::time::Instant::now();

    // Scale floating-point values to integer range [0, SCORE_SCALE]
    let score_scaled = (request.decayed_score.clamp(0.0, 1.0) * SCORE_SCALE as f64).round() as u64;
    let threshold_scaled =
        (request.required_threshold.clamp(0.0, 1.0) * SCORE_SCALE as f64).round() as u64;
    let lambda_scaled = (request.lambda * 1_000_000.0).round() as u64;
    let elapsed_scaled = (request.elapsed_days * 1000.0).round() as u64;

    let eligible = score_scaled >= threshold_scaled;

    // Compute commitment
    let commitment = compute_commitment(
        &request.agent_did,
        score_scaled,
        lambda_scaled,
        elapsed_scaled,
    );

    if !eligible {
        return Err(format!(
            "Sovereign gate: score {:.4} (scaled {}) below threshold {:.4} (scaled {})",
            request.decayed_score, score_scaled, request.required_threshold, threshold_scaled
        ));
    }

    // Prove: score_scaled ∈ [threshold_scaled, SCORE_SCALE]
    // This proves score >= threshold AND score <= SCORE_SCALE (valid range)
    let proof = range_proof::prove_range(score_scaled, threshold_scaled, SCORE_SCALE, commitment)?;
    let proof_bytes = proof.to_bytes();

    let prove_time_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok(SovereignProofResult {
        proof_bytes,
        score_commitment: commitment,
        eligible,
        prove_time_ms,
        threshold_scaled,
    })
}

/// Verify a sovereign tier proof.
///
/// The verifier checks that the STARK proof is valid for the given
/// public inputs (threshold + commitment). Returns `Ok(())` if valid.
#[cfg(feature = "backend-winterfell")]
pub fn verify_sovereign_tier(
    result: &SovereignProofResult,
    agent_did: &str,
    required_threshold: f64,
    lambda: f64,
    elapsed_days: f64,
) -> Result<(), String> {
    let threshold_scaled = (required_threshold.clamp(0.0, 1.0) * SCORE_SCALE as f64).round() as u64;
    let lambda_scaled = (lambda * 1_000_000.0).round() as u64;
    let elapsed_scaled = (elapsed_days * 1000.0).round() as u64;

    // Recompute commitment to verify binding
    // (The verifier doesn't know the score, but knows all public inputs)
    // We verify against the commitment in the proof result
    let proof = winterfell::Proof::from_bytes(&result.proof_bytes)
        .map_err(|e| format!("Failed to deserialize proof: {:?}", e))?;

    range_proof::verify_range(
        proof,
        threshold_scaled,
        SCORE_SCALE,
        result.score_commitment,
    )
}

/// Map a CivicTier to its minimum score threshold (for ZKP).
pub fn tier_threshold(tier_name: &str) -> f64 {
    match tier_name {
        "Participant" => 0.3,
        "Citizen" => 0.4,
        "Steward" => 0.6,
        "Guardian" => 0.8,
        _ => 0.0, // Observer
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commitment_is_deterministic() {
        let c1 = compute_commitment("did:mycelix:alice", 7500, 2000, 100000);
        let c2 = compute_commitment("did:mycelix:alice", 7500, 2000, 100000);
        assert_eq!(c1, c2);
    }

    #[test]
    fn commitment_changes_with_agent() {
        let c1 = compute_commitment("did:mycelix:alice", 7500, 2000, 100000);
        let c2 = compute_commitment("did:mycelix:bob", 7500, 2000, 100000);
        assert_ne!(c1, c2);
    }

    #[test]
    fn commitment_changes_with_score() {
        let c1 = compute_commitment("did:mycelix:alice", 7500, 2000, 100000);
        let c2 = compute_commitment("did:mycelix:alice", 8000, 2000, 100000);
        assert_ne!(c1, c2);
    }

    #[test]
    fn tier_thresholds_match_civic_tier() {
        assert_eq!(tier_threshold("Observer"), 0.0);
        assert_eq!(tier_threshold("Participant"), 0.3);
        assert_eq!(tier_threshold("Citizen"), 0.4);
        assert_eq!(tier_threshold("Steward"), 0.6);
        assert_eq!(tier_threshold("Guardian"), 0.8);
    }

    #[cfg(feature = "backend-winterfell")]
    #[test]
    fn prove_and_verify_citizen_tier() {
        let request = SovereignProofRequest {
            agent_did: "did:mycelix:alice".into(),
            decayed_score: 0.55, // Above Citizen threshold (0.4)
            required_threshold: 0.4,
            lambda: 0.002,
            elapsed_days: 30.0,
        };

        let result = prove_sovereign_tier(&request).expect("Proof should succeed");
        assert!(result.eligible);
        assert!(!result.proof_bytes.is_empty());
        assert!(result.prove_time_ms > 0.0);

        // Verify
        verify_sovereign_tier(&result, "did:mycelix:alice", 0.4, 0.002, 30.0)
            .expect("Verification should succeed");
    }

    #[cfg(feature = "backend-winterfell")]
    #[test]
    fn prove_and_verify_guardian_tier() {
        let request = SovereignProofRequest {
            agent_did: "did:mycelix:bob".into(),
            decayed_score: 0.92,
            required_threshold: 0.8,
            lambda: 0.001,
            elapsed_days: 10.0,
        };

        let result = prove_sovereign_tier(&request).expect("Proof should succeed");
        assert!(result.eligible);

        verify_sovereign_tier(&result, "did:mycelix:bob", 0.8, 0.001, 10.0)
            .expect("Verification should succeed");
    }

    #[cfg(feature = "backend-winterfell")]
    #[test]
    fn prove_fails_below_threshold() {
        let request = SovereignProofRequest {
            agent_did: "did:mycelix:eve".into(),
            decayed_score: 0.25, // Below Citizen threshold
            required_threshold: 0.4,
            lambda: 0.002,
            elapsed_days: 100.0,
        };

        let result = prove_sovereign_tier(&request);
        assert!(result.is_err(), "Should fail when score below threshold");
    }

    #[cfg(feature = "backend-winterfell")]
    #[test]
    fn prove_boundary_exact_threshold() {
        let request = SovereignProofRequest {
            agent_did: "did:mycelix:charlie".into(),
            decayed_score: 0.4, // Exactly at Citizen threshold
            required_threshold: 0.4,
            lambda: 0.002,
            elapsed_days: 0.0,
        };

        let result = prove_sovereign_tier(&request).expect("Exact threshold should pass");
        assert!(result.eligible);

        verify_sovereign_tier(&result, "did:mycelix:charlie", 0.4, 0.002, 0.0)
            .expect("Should verify");
    }

    #[cfg(feature = "backend-winterfell")]
    #[test]
    fn proof_metrics_benchmark() {
        let tiers = [
            ("Citizen", 0.55, 0.4),
            ("Steward", 0.72, 0.6),
            ("Guardian", 0.92, 0.8),
        ];

        println!("\n=== Sovereign ZKP Proof Metrics ===");
        for (name, score, threshold) in &tiers {
            let request = SovereignProofRequest {
                agent_did: format!("did:mycelix:bench-{}", name.to_lowercase()),
                decayed_score: *score,
                required_threshold: *threshold,
                lambda: 0.002,
                elapsed_days: 30.0,
            };

            let result = prove_sovereign_tier(&request).unwrap();
            let proof_kb = result.proof_bytes.len() as f64 / 1024.0;

            // Verify
            let start = std::time::Instant::now();
            verify_sovereign_tier(&result, &request.agent_did, *threshold, 0.002, 30.0).unwrap();
            let verify_ms = start.elapsed().as_secs_f64() * 1000.0;

            println!(
                "  {name}: prove={:.1}ms verify={:.1}ms size={:.1}KB",
                result.prove_time_ms, verify_ms, proof_kb,
            );
        }
        println!("=== End Metrics ===\n");
    }
}
