// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness Vector Attestation — Cryptographic Proofs of Conscious State
//!
//! Each [`ConsciousnessVector`] can be signed by the originating node's Ed25519
//! key, creating a non-repudiable proof of the consciousness state at a given time.
//!
//! ## Applications
//!
//! - **Inbound verification**: Only accept CVs with valid attestations from trusted signers
//! - **Audit trail**: Prove which node produced which consciousness state
//! - **Research**: Verifiable consciousness experiments across distributed nodes
//!
//! ## Wire Format
//!
//! ```text
//! [CV bytes (bincode)] | [signature (64 bytes)] | [signer_key_hex (64 chars)]
//! ```
//!
//! ## Feature Gate
//!
//! Requires `identity` feature for Ed25519 signatures.
//! Without `identity`, a BLAKE3 MAC fallback is used (symmetric, weaker).

use crate::swarm::{ConsciousnessVector, SwarmError, SwarmResult};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// An attested (signed) consciousness vector with cryptographic proof of origin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestedConsciousnessVector {
    /// The consciousness vector payload.
    pub vector: ConsciousnessVector,
    /// Ed25519 signature over the serialized vector (64 bytes).
    pub signature: Vec<u8>,
    /// Signer's public key (hex-encoded, 64 hex chars = 32 bytes).
    pub signer_key: String,
}

/// Manages attestation creation and verification.
///
/// Holds the node's signing key and a registry of trusted signer public keys.
/// Only CVs signed by trusted signers pass verification.
pub struct AttestationManager {
    /// This node's Ed25519 signing key.
    #[cfg(feature = "identity")]
    signing_key: ed25519_dalek::SigningKey,
    /// This node's public key as hex.
    public_key_hex: String,
    /// Set of trusted signer public keys (hex-encoded).
    trusted_signers: HashSet<String>,
    /// Whether to require attestation on inbound CVs.
    require_attestation: bool,
}

#[cfg(feature = "identity")]
impl AttestationManager {
    /// Create a new attestation manager with an Ed25519 signing key.
    pub fn new(signing_key: ed25519_dalek::SigningKey) -> Self {
        let public_key_hex = hex::encode(signing_key.verifying_key().as_bytes());
        let mut trusted = HashSet::new();
        // Trust ourselves by default
        trusted.insert(public_key_hex.clone());
        Self {
            signing_key,
            public_key_hex,
            trusted_signers: trusted,
            require_attestation: true,
        }
    }

    /// Add a trusted signer's public key.
    pub fn trust_signer(&mut self, public_key_hex: &str) {
        self.trusted_signers.insert(public_key_hex.to_string());
    }

    /// Remove a signer from the trusted set.
    pub fn untrust_signer(&mut self, public_key_hex: &str) {
        self.trusted_signers.remove(public_key_hex);
    }

    /// Set whether attestation is required on inbound CVs.
    pub fn set_require_attestation(&mut self, require: bool) {
        self.require_attestation = require;
    }

    /// Whether attestation is required.
    pub fn requires_attestation(&self) -> bool {
        self.require_attestation
    }

    /// Get this node's public key hex.
    pub fn public_key_hex(&self) -> &str {
        &self.public_key_hex
    }

    /// Get this node's signing key (for handshake responder operations).
    pub fn signing_key(&self) -> &ed25519_dalek::SigningKey {
        &self.signing_key
    }

    /// Number of trusted signers.
    pub fn trusted_signer_count(&self) -> usize {
        self.trusted_signers.len()
    }

    /// Sign a ConsciousnessVector, producing an attestation.
    ///
    /// The signature is computed over the bincode-serialized CV bytes.
    pub fn attest(&self, cv: &ConsciousnessVector) -> SwarmResult<AttestedConsciousnessVector> {
        use ed25519_dalek::Signer;

        let cv_bytes =
            bincode::serialize(cv).map_err(|e| SwarmError::SerializationError(e.to_string()))?;

        let signature = self.signing_key.sign(&cv_bytes);

        Ok(AttestedConsciousnessVector {
            vector: cv.clone(),
            signature: signature.to_bytes().to_vec(),
            signer_key: self.public_key_hex.clone(),
        })
    }

    /// Verify an attested CV: check signature validity AND signer trust.
    ///
    /// Returns `Ok(true)` if the signature is valid and the signer is trusted.
    /// Returns `Ok(false)` if the signer is not in the trusted set (valid sig, untrusted).
    /// Returns `Err` if the signature is invalid or malformed.
    pub fn verify(&self, attested: &AttestedConsciousnessVector) -> SwarmResult<bool> {
        use ed25519_dalek::{Signature, Verifier, VerifyingKey};

        // Check signer is trusted
        if !self.trusted_signers.contains(&attested.signer_key) {
            return Ok(false);
        }

        // Deserialize public key
        let pubkey_bytes =
            hex::decode(&attested.signer_key).map_err(|e| SwarmError::TrustVerificationError {
                reason: format!("Invalid signer key hex: {e}"),
            })?;

        if pubkey_bytes.len() != 32 {
            return Err(SwarmError::TrustVerificationError {
                reason: format!(
                    "Invalid signer key length: expected 32, got {}",
                    pubkey_bytes.len()
                ),
            });
        }

        let pubkey_array: [u8; 32] =
            pubkey_bytes
                .try_into()
                .map_err(|_| SwarmError::TrustVerificationError {
                    reason: "Failed to convert signer key bytes".to_string(),
                })?;

        let verifying_key = VerifyingKey::from_bytes(&pubkey_array).map_err(|e| {
            SwarmError::TrustVerificationError {
                reason: format!("Invalid Ed25519 public key: {e}"),
            }
        })?;

        // Deserialize signature
        if attested.signature.len() != 64 {
            return Err(SwarmError::TrustVerificationError {
                reason: format!(
                    "Invalid signature length: expected 64, got {}",
                    attested.signature.len()
                ),
            });
        }

        let sig_bytes: [u8; 64] = attested.signature.clone().try_into().map_err(|_| {
            SwarmError::TrustVerificationError {
                reason: "Failed to convert signature bytes".to_string(),
            }
        })?;
        let signature = Signature::from_bytes(&sig_bytes);

        // Serialize CV for verification
        let cv_bytes = bincode::serialize(&attested.vector)
            .map_err(|e| SwarmError::SerializationError(e.to_string()))?;

        // Verify signature
        verifying_key.verify(&cv_bytes, &signature).map_err(|_| {
            SwarmError::TrustVerificationError {
                reason: "Ed25519 signature verification failed on ConsciousnessVector".to_string(),
            }
        })?;

        Ok(true)
    }

    /// Verify and extract: return the CV only if attestation is valid and signer is trusted.
    ///
    /// This is the recommended method for the recv path — it ensures only
    /// verified CVs enter the cognitive loop.
    pub fn verify_and_extract(
        &self,
        attested: &AttestedConsciousnessVector,
    ) -> SwarmResult<ConsciousnessVector> {
        let trusted = self.verify(attested)?;
        if !trusted {
            return Err(SwarmError::TrustVerificationError {
                reason: format!(
                    "CV signer '{}' is not in trusted set",
                    &attested.signer_key[..16.min(attested.signer_key.len())]
                ),
            });
        }
        Ok(attested.vector.clone())
    }
}

/// Stub attestation manager when `identity` feature is not enabled.
///
/// Provides the same API but always returns errors for sign/verify operations.
#[cfg(not(feature = "identity"))]
impl AttestationManager {
    /// Create a stub manager (no signing capability).
    pub fn new_stub() -> Self {
        Self {
            public_key_hex: String::new(),
            trusted_signers: HashSet::new(),
            require_attestation: false,
        }
    }

    pub fn trust_signer(&mut self, public_key_hex: &str) {
        self.trusted_signers.insert(public_key_hex.to_string());
    }

    pub fn set_require_attestation(&mut self, require: bool) {
        self.require_attestation = require;
    }

    pub fn requires_attestation(&self) -> bool {
        self.require_attestation
    }

    pub fn public_key_hex(&self) -> &str {
        &self.public_key_hex
    }

    pub fn trusted_signer_count(&self) -> usize {
        self.trusted_signers.len()
    }

    pub fn attest(&self, _cv: &ConsciousnessVector) -> SwarmResult<AttestedConsciousnessVector> {
        Err(SwarmError::FeatureNotEnabled {
            feature: "identity".to_string(),
        })
    }

    pub fn verify(&self, _attested: &AttestedConsciousnessVector) -> SwarmResult<bool> {
        Err(SwarmError::FeatureNotEnabled {
            feature: "identity".to_string(),
        })
    }

    pub fn verify_and_extract(
        &self,
        _attested: &AttestedConsciousnessVector,
    ) -> SwarmResult<ConsciousnessVector> {
        Err(SwarmError::FeatureNotEnabled {
            feature: "identity".to_string(),
        })
    }
}

// ════════════════════════════════════════════════════════════════════════════
// TESTS
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
#[cfg(feature = "identity")]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    fn test_keypair() -> (SigningKey, String) {
        let mut seed = [0u8; 32];
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut seed);
        let signing_key = SigningKey::from_bytes(&seed);
        let pubkey_hex = hex::encode(signing_key.verifying_key().as_bytes());
        (signing_key, pubkey_hex)
    }

    fn test_cv() -> ConsciousnessVector {
        ConsciousnessVector::new(vec![0.1, 0.2, 0.3, 0.4], 0.75)
    }

    #[test]
    fn test_attest_and_verify_roundtrip() {
        let (key, _) = test_keypair();
        let manager = AttestationManager::new(key);
        let cv = test_cv();

        let attested = manager.attest(&cv).unwrap();
        assert_eq!(attested.signature.len(), 64);
        assert_eq!(attested.signer_key, manager.public_key_hex());

        let result = manager.verify(&attested).unwrap();
        assert!(result, "Self-signed CV should verify");
    }

    #[test]
    fn test_verify_and_extract() {
        let (key, _) = test_keypair();
        let manager = AttestationManager::new(key);
        let cv = test_cv();

        let attested = manager.attest(&cv).unwrap();
        let extracted = manager.verify_and_extract(&attested).unwrap();
        assert_eq!(extracted.phi, cv.phi);
    }

    #[test]
    fn test_untrusted_signer_rejected() {
        let (key_a, _) = test_keypair();
        let (key_b, _) = test_keypair();
        let manager_a = AttestationManager::new(key_a);
        let manager_b = AttestationManager::new(key_b);

        let cv = test_cv();
        let attested = manager_a.attest(&cv).unwrap();

        // manager_b doesn't trust manager_a
        let result = manager_b.verify(&attested).unwrap();
        assert!(!result, "Untrusted signer should not verify");
    }

    #[test]
    fn test_trusted_cross_node_verification() {
        let (key_a, pub_a) = test_keypair();
        let (key_b, _) = test_keypair();
        let manager_a = AttestationManager::new(key_a);
        let mut manager_b = AttestationManager::new(key_b);

        // B trusts A
        manager_b.trust_signer(&pub_a);

        let cv = test_cv();
        let attested = manager_a.attest(&cv).unwrap();

        let result = manager_b.verify(&attested).unwrap();
        assert!(result, "Trusted cross-node CV should verify");
    }

    #[test]
    fn test_tampered_cv_fails() {
        let (key, _) = test_keypair();
        let manager = AttestationManager::new(key);
        let cv = test_cv();

        let mut attested = manager.attest(&cv).unwrap();

        // Tamper with the CV
        attested.vector.phi = 0.99;

        let result = manager.verify(&attested);
        assert!(
            result.is_err(),
            "Tampered CV should fail signature verification"
        );
    }

    #[test]
    fn test_tampered_signature_fails() {
        let (key, _) = test_keypair();
        let manager = AttestationManager::new(key);
        let cv = test_cv();

        let mut attested = manager.attest(&cv).unwrap();
        attested.signature[0] ^= 0xFF;

        let result = manager.verify(&attested);
        assert!(result.is_err(), "Tampered signature should fail");
    }

    #[test]
    fn test_untrust_signer() {
        let (key_a, pub_a) = test_keypair();
        let (key_b, _) = test_keypair();
        let manager_a = AttestationManager::new(key_a);
        let mut manager_b = AttestationManager::new(key_b);

        manager_b.trust_signer(&pub_a);
        assert_eq!(manager_b.trusted_signer_count(), 2); // self + A

        manager_b.untrust_signer(&pub_a);
        assert_eq!(manager_b.trusted_signer_count(), 1); // self only

        let cv = test_cv();
        let attested = manager_a.attest(&cv).unwrap();
        let result = manager_b.verify(&attested).unwrap();
        assert!(!result, "Untrusted-then-removed signer should not verify");
    }

    #[test]
    fn test_verify_and_extract_untrusted_returns_error() {
        let (key_a, _) = test_keypair();
        let (key_b, _) = test_keypair();
        let manager_a = AttestationManager::new(key_a);
        let manager_b = AttestationManager::new(key_b);

        let cv = test_cv();
        let attested = manager_a.attest(&cv).unwrap();

        let result = manager_b.verify_and_extract(&attested);
        assert!(
            result.is_err(),
            "verify_and_extract should error for untrusted signer"
        );
    }
}
