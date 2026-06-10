// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Zero-knowledge consciousness tier attestation.
//!
//! Replaces plaintext consciousness score transmission across clusters.
//! Instead of sending `consciousness_level: 0.55`, agents now submit
//! a STARK proof that proves `Φ ≥ threshold` without revealing Φ.
//!
//! ## Privacy Fix
//!
//! Before: `verify_citizen_tier()` sends raw Φ score to identity cluster
//! After: `verify_consciousness_attestation()` checks STARK proof bytes
//!
//! ## Usage
//!
//! Client-side (native Rust):
//! ```ignore
//! use mycelix_zkp_core::consciousness::prove_consciousness_tier;
//! let proof = prove_consciousness_tier(&request)?;
//! ```
//!
//! Zome-side (WASM):
//! ```ignore
//! use mycelix_bridge_common::consciousness_zkp::ConsciousnessAttestation;
//! // Validate proof structure (size, commitment format)
//! attestation.validate_structure()?;
//! // Off-chain verifier does the actual STARK verification
//! ```

use serde::{Deserialize, Serialize};

/// A consciousness tier attestation carrying ZKP proof bytes.
///
/// This replaces the plaintext `consciousness_level: f64` in cross-cluster calls.
/// The proof demonstrates `Φ ≥ threshold` without revealing Φ.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConsciousnessAttestation {
    /// Which tier this attestation proves (0=Observer, 1=Participant, 2=Citizen, 3=Steward, 4=Guardian)
    pub tier: u8,
    /// Winterfell STARK proof bytes (generated client-side via mycelix-zkp-core)
    pub proof_bytes: Vec<u8>,
    /// SHA-256 commitment to the score (agent_did + score + domain_tag)
    pub score_commitment: [u8; 32],
    /// When the attestation was generated (Unix timestamp)
    pub generated_at: u64,
    /// Optional: off-chain verifier's Ed25519 signature over the attestation
    pub verifier_signature: Option<Vec<u8>>,
    /// Optional: off-chain verifier's public key
    pub verifier_pubkey: Option<Vec<u8>>,
}

/// Maximum proof size (prevent DoS)
pub const MAX_CONSCIOUSNESS_PROOF_SIZE: usize = 50_000; // 50KB

/// Default attestation validity (24 hours in seconds)
pub const ATTESTATION_VALIDITY_SECS: u64 = 86_400;

impl ConsciousnessAttestation {
    /// Validate the structural integrity of the attestation.
    ///
    /// This runs in WASM (no winterfell dependency needed).
    /// Actual STARK verification happens off-chain.
    pub fn validate_structure(&self) -> Result<(), String> {
        if self.proof_bytes.is_empty() {
            return Err("Empty proof bytes".to_string());
        }
        if self.proof_bytes.len() > MAX_CONSCIOUSNESS_PROOF_SIZE {
            return Err(format!(
                "Proof too large: {} > {}",
                self.proof_bytes.len(),
                MAX_CONSCIOUSNESS_PROOF_SIZE
            ));
        }
        if self.score_commitment == [0u8; 32] {
            return Err("Zero commitment".to_string());
        }
        if self.tier > 4 {
            return Err(format!("Invalid tier: {} (max 4)", self.tier));
        }
        Ok(())
    }

    /// Check if this attestation has been verified by an off-chain verifier.
    pub fn is_verified(&self) -> bool {
        self.verifier_signature.is_some() && self.verifier_pubkey.is_some()
    }

    /// Check if this attestation has expired.
    pub fn is_expired(&self, current_time: u64) -> bool {
        current_time > self.generated_at + ATTESTATION_VALIDITY_SECS
    }

    /// Validate that this attestation is fresh: not expired, not future-dated.
    ///
    /// Used for replay prevention in cross-cluster attested calls (e.g. voting).
    /// `epoch` is the current Unix timestamp in seconds. Allows up to 60s of
    /// clock skew before rejecting future-dated attestations.
    pub fn validate_with_freshness(&self, epoch: u64) -> Result<(), String> {
        if self.is_expired(epoch) {
            return Err(format!(
                "attestation expired (generated_at={}, now={}, validity={}s)",
                self.generated_at, epoch, ATTESTATION_VALIDITY_SECS
            ));
        }
        if self.generated_at > epoch.saturating_add(60) {
            return Err(format!(
                "attestation dated in the future (generated_at={}, now={})",
                self.generated_at, epoch
            ));
        }
        Ok(())
    }

    /// Get the tier name.
    pub fn tier_name(&self) -> &'static str {
        match self.tier {
            0 => "Observer",
            1 => "Participant",
            2 => "Citizen",
            3 => "Steward",
            4 => "Guardian",
            _ => "Unknown",
        }
    }

    /// Minimum Φ threshold for this tier.
    pub fn min_threshold(&self) -> f64 {
        match self.tier {
            0 => 0.0,
            1 => 0.2,
            2 => 0.3,
            3 => 0.4,
            4 => 0.6,
            _ => 1.0,
        }
    }
}

/// Input for cross-cluster consciousness verification.
///
/// Replaces the old `{ consciousness_level: f64 }` input.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerifyConsciousnessInput {
    /// The attestation with STARK proof
    pub attestation: ConsciousnessAttestation,
    /// Agent DID (for commitment verification)
    pub agent_did: String,
    /// What action requires this tier (for audit)
    pub action_description: String,
}

/// Output from consciousness verification.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerifyConsciousnessOutput {
    /// Whether the attestation is structurally valid
    pub structure_valid: bool,
    /// Whether the off-chain STARK verification passed (None if not yet verified)
    pub stark_verified: Option<bool>,
    /// The tier that was proven
    pub tier: u8,
    /// Whether the attestation has expired
    pub expired: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_attestation() {
        let att = ConsciousnessAttestation {
            tier: 3,                       // Steward
            proof_bytes: vec![1, 2, 3, 4], // Minimal non-empty
            score_commitment: [0xAA; 32],
            generated_at: 1700000000,
            verifier_signature: None,
            verifier_pubkey: None,
        };
        assert!(att.validate_structure().is_ok());
        assert_eq!(att.tier_name(), "Steward");
        assert!((att.min_threshold() - 0.4).abs() < f64::EPSILON);
        assert!(!att.is_verified());
    }

    #[test]
    fn test_empty_proof_rejected() {
        let att = ConsciousnessAttestation {
            tier: 1,
            proof_bytes: vec![],
            score_commitment: [0xBB; 32],
            generated_at: 0,
            verifier_signature: None,
            verifier_pubkey: None,
        };
        assert!(att.validate_structure().is_err());
    }

    #[test]
    fn test_oversized_proof_rejected() {
        let att = ConsciousnessAttestation {
            tier: 1,
            proof_bytes: vec![0; MAX_CONSCIOUSNESS_PROOF_SIZE + 1],
            score_commitment: [0xCC; 32],
            generated_at: 0,
            verifier_signature: None,
            verifier_pubkey: None,
        };
        assert!(att.validate_structure().is_err());
    }

    #[test]
    fn test_expiration() {
        let att = ConsciousnessAttestation {
            tier: 2,
            proof_bytes: vec![1],
            score_commitment: [0xDD; 32],
            generated_at: 1000,
            verifier_signature: None,
            verifier_pubkey: None,
        };
        assert!(!att.is_expired(1000 + ATTESTATION_VALIDITY_SECS - 1));
        assert!(att.is_expired(1000 + ATTESTATION_VALIDITY_SECS + 1));
    }
}
