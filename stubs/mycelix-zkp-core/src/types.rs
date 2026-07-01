// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Core types for the DASTARK proof system.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::domain::DomainTag;

/// Raw proof bytes from any backend.
pub type ProofBytes = Vec<u8>;

/// Result of proof generation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProofResult {
    /// The proof bytes (backend-specific format).
    pub proof: ProofBytes,
    /// Which backend produced this proof.
    pub backend: BackendId,
    /// Proving time in milliseconds.
    pub proving_time_ms: u64,
    /// Proof size in bytes.
    pub proof_size: usize,
}

/// Result of proof verification.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Whether the proof is valid.
    pub valid: bool,
    /// Verification time in milliseconds.
    pub verification_time_ms: u64,
    /// Human-readable message (empty on success).
    pub message: String,
}

impl VerificationResult {
    pub fn ok(verification_time_ms: u64) -> Self {
        Self {
            valid: true,
            verification_time_ms,
            message: String::new(),
        }
    }

    pub fn fail(message: impl Into<String>, verification_time_ms: u64) -> Self {
        Self {
            valid: false,
            verification_time_ms,
            message: message.into(),
        }
    }
}

/// Identifies which backend produced a proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendId {
    Risc0,
    Winterfell,
    Binius,
}

impl BackendId {
    pub fn as_str(&self) -> &'static str {
        match self {
            BackendId::Risc0 => "risc0",
            BackendId::Winterfell => "winterfell",
            BackendId::Binius => "binius",
        }
    }
}

/// Metadata attached to every proof for audit and replay protection.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProofMetadata {
    /// Domain separation tag.
    pub domain_tag: DomainTag,
    /// Protocol version.
    pub protocol_version: u32,
    /// Agent/client identity (SHA-256 of public key).
    pub client_id: [u8; 32],
    /// Timestamp (Unix seconds).
    pub timestamp: u64,
    /// Random nonce (32 bytes) for replay protection.
    pub nonce: [u8; 32],
    /// Which backend was used.
    pub backend: BackendId,
}

/// An authenticated proof: ZK proof + Dilithium5 PQ signature + metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuthenticatedProof {
    /// The ZK proof bytes (RISC0 receipt or Winterfell STARK).
    pub proof: ProofBytes,
    /// CRYSTALS-Dilithium5 signature over (domain_tag || metadata || proof_hash || joules).
    /// Empty if Dilithium is not enabled.
    pub signature: Vec<u8>,
    /// Proof metadata (domain, timestamp, nonce, client_id, backend).
    pub metadata: ProofMetadata,
    /// SHA-256 of any application-specific public inputs.
    pub public_inputs_hash: [u8; 32],
    /// Energy consumed during proof generation (Joules), verified by hardware.
    pub joules_consumed: f64,
}

impl AuthenticatedProof {
    /// Construct the message that was signed (for verification).
    ///
    /// Message = SHA-256(domain_tag || protocol_version || client_id ||
    ///                    timestamp || nonce || public_inputs_hash || proof_hash)
    pub fn construct_signed_message(&self) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(self.metadata.domain_tag.as_bytes());
        hasher.update(self.metadata.protocol_version.to_le_bytes());
        hasher.update(self.metadata.client_id);
        hasher.update(self.metadata.timestamp.to_le_bytes());
        hasher.update(self.metadata.nonce);
        hasher.update(self.public_inputs_hash);
        let proof_hash = Sha256::digest(&self.proof);
        hasher.update(proof_hash);
        hasher.finalize().to_vec()
    }

    /// Get total size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.proof.len()
            + self.signature.len()
            + 32 // client_id
            + 32 // nonce
            + 32 // public_inputs_hash
            + 8  // timestamp
            + 4  // protocol_version
            + self.metadata.domain_tag.as_bytes().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verification_result_ok() {
        let r = VerificationResult::ok(5);
        assert!(r.valid);
        assert!(r.message.is_empty());
    }

    #[test]
    fn test_backend_id_str() {
        assert_eq!(BackendId::Risc0.as_str(), "risc0");
        assert_eq!(BackendId::Winterfell.as_str(), "winterfell");
        assert_eq!(BackendId::Binius.as_str(), "binius");
    }

    #[test]
    fn test_authenticated_proof_signed_message_deterministic() {
        let meta = ProofMetadata {
            domain_tag: DomainTag::new("Test", "Unit", 1),
            protocol_version: 1,
            client_id: [0xAA; 32],
            timestamp: 1700000000,
            nonce: [0xBB; 32],
            backend: BackendId::Winterfell,
        };
        let proof = AuthenticatedProof {
            proof: vec![1, 2, 3, 4],
            signature: vec![],
            metadata: meta,
            public_inputs_hash: [0xCC; 32],
            joules_consumed: 0.0,
        };
        let msg1 = proof.construct_signed_message();
        let msg2 = proof.construct_signed_message();
        assert_eq!(msg1, msg2);
        assert_eq!(msg1.len(), 32);
    }
}
