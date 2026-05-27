// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! CRYSTALS-Dilithium5 post-quantum authentication.
//!
//! Ported from `mycelix-core/0TML/gen7-zkstark/bindings/src/dilithium.rs` (312 lines).
//! This version removes PyO3 dependencies and provides a pure Rust API.
//!
//! ## Security
//!
//! - NIST Level 5 (highest, comparable to AES-256)
//! - Public Key: 2,592 bytes
//! - Secret Key: 4,896 bytes (library actual, not 4,864 from spec)
//! - Signature: 4,595 bytes

use pqcrypto_dilithium::dilithium5;
use pqcrypto_traits::sign::{PublicKey as PQPublicKey, SecretKey as PQSecretKey, SignedMessage};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{ZkpError, ZkpResult};
use crate::types::AuthenticatedProof;

/// Dilithium5 exact sizes (empirically verified).
pub const PUBLIC_KEY_SIZE: usize = 2_592;
pub const SECRET_KEY_SIZE: usize = 4_896;
pub const SIGNATURE_SIZE: usize = 4_595;

/// A Dilithium5 keypair for proof authentication.
#[derive(Clone)]
pub struct DilithiumKeypair {
    public_key: Vec<u8>,
    secret_key: Vec<u8>,
    /// SHA-256(public_key) — used as client identity.
    client_id: [u8; 32],
}

impl DilithiumKeypair {
    /// Generate a new random keypair.
    pub fn generate() -> Self {
        let (pk, sk) = dilithium5::keypair();
        let public_key = pk.as_bytes().to_vec();
        let secret_key = sk.as_bytes().to_vec();

        debug_assert_eq!(public_key.len(), PUBLIC_KEY_SIZE);
        debug_assert_eq!(secret_key.len(), SECRET_KEY_SIZE);

        let client_id = Self::compute_client_id(&public_key);

        Self {
            public_key,
            secret_key,
            client_id,
        }
    }

    /// Load from existing key bytes.
    pub fn from_bytes(public_key: Vec<u8>, secret_key: Vec<u8>) -> ZkpResult<Self> {
        if public_key.len() != PUBLIC_KEY_SIZE {
            return Err(ZkpError::InvalidKeyMaterial(format!(
                "public key: expected {} bytes, got {}",
                PUBLIC_KEY_SIZE,
                public_key.len()
            )));
        }
        if secret_key.len() != SECRET_KEY_SIZE {
            return Err(ZkpError::InvalidKeyMaterial(format!(
                "secret key: expected {} bytes, got {}",
                SECRET_KEY_SIZE,
                secret_key.len()
            )));
        }

        let client_id = Self::compute_client_id(&public_key);

        Ok(Self {
            public_key,
            secret_key,
            client_id,
        })
    }

    /// Get the public key bytes.
    pub fn public_key(&self) -> &[u8] {
        &self.public_key
    }

    /// Get the client ID (SHA-256 of public key).
    pub fn client_id(&self) -> &[u8; 32] {
        &self.client_id
    }

    /// Sign a message with Dilithium5.
    pub fn sign(&self, message: &[u8]) -> ZkpResult<Vec<u8>> {
        let sk = dilithium5::SecretKey::from_bytes(&self.secret_key)
            .map_err(|e| ZkpError::InvalidKeyMaterial(format!("secret key: {:?}", e)))?;

        let signed = dilithium5::sign(message, &sk);
        Ok(signed.as_bytes().to_vec())
    }

    /// Sign an AuthenticatedProof in place (fills the signature field).
    pub fn sign_proof(&self, proof: &mut AuthenticatedProof) -> ZkpResult<()> {
        let message = proof.construct_signed_message();
        proof.signature = self.sign(&message)?;
        Ok(())
    }

    fn compute_client_id(public_key: &[u8]) -> [u8; 32] {
        let hash = Sha256::digest(public_key);
        let mut id = [0u8; 32];
        id.copy_from_slice(&hash);
        id
    }
}

/// Verify a Dilithium5 signature (standalone, no keypair needed).
///
/// This is the function called inside Holochain zome integrity validation.
pub fn verify_signature(message: &[u8], signature: &[u8], public_key: &[u8]) -> ZkpResult<bool> {
    if public_key.len() != PUBLIC_KEY_SIZE {
        return Err(ZkpError::InvalidKeyMaterial(format!(
            "public key: expected {} bytes, got {}",
            PUBLIC_KEY_SIZE,
            public_key.len()
        )));
    }

    let pk = dilithium5::PublicKey::from_bytes(public_key)
        .map_err(|e| ZkpError::InvalidKeyMaterial(format!("public key parse: {:?}", e)))?;

    let signed_msg =
        dilithium5::SignedMessage::from_bytes(signature).map_err(|_| ZkpError::SignatureInvalid)?;

    match dilithium5::open(&signed_msg, &pk) {
        Ok(verified_msg) => Ok(verified_msg == message),
        Err(_) => Ok(false),
    }
}

/// Verify an AuthenticatedProof's Dilithium signature.
///
/// Checks:
/// 1. client_id matches SHA-256(public_key)
/// 2. Signature is valid for the constructed message
///
/// Does NOT verify the ZK proof itself — that's backend-specific.
pub fn verify_authenticated_signature(
    proof: &AuthenticatedProof,
    client_public_key: &[u8],
) -> ZkpResult<bool> {
    // Check client_id matches public key
    let expected_id = Sha256::digest(client_public_key);
    if proof.metadata.client_id != expected_id.as_slice() {
        return Err(ZkpError::ClientIdMismatch);
    }

    // Verify signature
    let message = proof.construct_signed_message();
    verify_signature(&message, &proof.signature, client_public_key)
}

/// Serializable public key wrapper for storage on DHT.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DilithiumPublicKey {
    pub bytes: Vec<u8>,
    pub client_id: [u8; 32],
}

impl DilithiumPublicKey {
    pub fn from_keypair(kp: &DilithiumKeypair) -> Self {
        Self {
            bytes: kp.public_key.clone(),
            client_id: kp.client_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::DomainTag;
    use crate::types::{BackendId, ProofMetadata};

    #[test]
    fn test_keypair_generation() {
        let kp = DilithiumKeypair::generate();
        assert_eq!(kp.public_key().len(), PUBLIC_KEY_SIZE);
        assert_eq!(kp.client_id().len(), 32);
    }

    #[test]
    fn test_sign_verify_roundtrip() {
        let kp = DilithiumKeypair::generate();
        let message = b"test message for DASTARK";
        let signature = kp.sign(message).unwrap();

        let valid = verify_signature(message, &signature, kp.public_key()).unwrap();
        assert!(valid, "signature must verify with correct key");
    }

    #[test]
    fn test_wrong_key_fails() {
        let kp1 = DilithiumKeypair::generate();
        let kp2 = DilithiumKeypair::generate();
        let message = b"test message";
        let signature = kp1.sign(message).unwrap();

        let valid = verify_signature(message, &signature, kp2.public_key()).unwrap();
        assert!(!valid, "signature must fail with wrong key");
    }

    #[test]
    fn test_tampered_message_fails() {
        let kp = DilithiumKeypair::generate();
        let message = b"original message";
        let signature = kp.sign(message).unwrap();

        let valid = verify_signature(b"tampered message", &signature, kp.public_key()).unwrap();
        assert!(!valid, "signature must fail with tampered message");
    }

    #[test]
    fn test_sign_authenticated_proof() {
        let kp = DilithiumKeypair::generate();
        let meta = ProofMetadata {
            domain_tag: DomainTag::new("Test", "SignProof", 1),
            protocol_version: 1,
            client_id: *kp.client_id(),
            timestamp: 1700000000,
            nonce: [0x42; 32],
            backend: BackendId::Winterfell,
        };
        let mut proof = AuthenticatedProof {
            proof: vec![0xDE, 0xAD, 0xBE, 0xEF],
            signature: vec![],
            metadata: meta,
            public_inputs_hash: [0xCC; 32],
        };

        kp.sign_proof(&mut proof).unwrap();
        assert!(!proof.signature.is_empty());
        // pqcrypto sign() returns message + signature concatenated
        // Signature size = message_len + SIGNATURE_SIZE
        assert!(
            proof.signature.len() > SIGNATURE_SIZE,
            "signature bytes ({}) must be > SIGNATURE_SIZE ({})",
            proof.signature.len(),
            SIGNATURE_SIZE
        );

        let valid = verify_authenticated_signature(&proof, kp.public_key()).unwrap();
        assert!(valid, "authenticated proof signature must verify");
    }

    #[test]
    fn test_client_id_mismatch() {
        let kp = DilithiumKeypair::generate();
        let meta = ProofMetadata {
            domain_tag: DomainTag::new("Test", "Mismatch", 1),
            protocol_version: 1,
            client_id: [0xFF; 32], // Wrong client ID
            timestamp: 1700000000,
            nonce: [0x42; 32],
            backend: BackendId::Risc0,
        };
        let proof = AuthenticatedProof {
            proof: vec![1, 2, 3],
            signature: vec![],
            metadata: meta,
            public_inputs_hash: [0; 32],
        };

        let result = verify_authenticated_signature(&proof, kp.public_key());
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ZkpError::ClientIdMismatch));
    }

    #[test]
    fn test_invalid_key_size() {
        let result = DilithiumKeypair::from_bytes(vec![0; 100], vec![0; 100]);
        assert!(result.is_err());
    }

    #[test]
    fn test_public_key_serialization() {
        let kp = DilithiumKeypair::generate();
        let pk = DilithiumPublicKey::from_keypair(&kp);
        let json = serde_json::to_string(&pk).unwrap();
        let parsed: DilithiumPublicKey = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.client_id, pk.client_id);
        assert_eq!(parsed.bytes.len(), PUBLIC_KEY_SIZE);
    }
}
