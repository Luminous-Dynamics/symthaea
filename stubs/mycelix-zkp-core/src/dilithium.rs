// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! CRYSTALS-Dilithium5 post-quantum authentication.
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
    /// SHA-256(public_key) -- used as client identity.
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

        // Validate both encodings and prove the keys belong together before
        // constructing an object that can sign under one key and identify as another.
        let public_key_obj = dilithium5::PublicKey::from_bytes(&public_key)
            .map_err(|e| ZkpError::InvalidKeyMaterial(format!("public key parse: {:?}", e)))?;
        let secret_key_obj = dilithium5::SecretKey::from_bytes(&secret_key)
            .map_err(|e| ZkpError::InvalidKeyMaterial(format!("secret key parse: {:?}", e)))?;

        // PQCrypto's byte parsers validate representation shape, not key-pair
        // consistency. A sign/verify challenge is therefore required here.
        // Keep this challenge fixed and domain-specific; it is never exposed.
        const KEYPAIR_CHECK: &[u8] =
            b"mycelix-zkp-core Dilithium5 keypair consistency check v1";
        let signed_check = dilithium5::sign(KEYPAIR_CHECK, &secret_key_obj);
        match dilithium5::open(&signed_check, &public_key_obj) {
            Ok(message) if message == KEYPAIR_CHECK => {}
            _ => {
                return Err(ZkpError::InvalidKeyMaterial(
                    "public and secret keys do not form a valid pair".to_string(),
                ));
            }
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
/// Does NOT verify the ZK proof itself -- that's backend-specific.
pub fn verify_authenticated_signature(
    proof: &AuthenticatedProof,
    client_public_key: &[u8],
) -> ZkpResult<bool> {
    let expected_id = Sha256::digest(client_public_key);
    if proof.metadata.client_id != expected_id.as_slice() {
        return Err(ZkpError::ClientIdMismatch);
    }

    let message = proof.construct_signed_message();
    verify_signature(&message, &proof.signature, client_public_key)
}

/// Serializable public key wrapper for storage on DHT.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DilithiumPublicKey {
    pub bytes: Vec<u8>,
}

impl DilithiumPublicKey {
    pub fn from_keypair(kp: &DilithiumKeypair) -> Self {
        Self {
            bytes: kp.public_key().to_vec(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keypair_generate_and_sign_verify() {
        let kp = DilithiumKeypair::generate();
        let message = b"hello dastark";
        let sig = kp.sign(message).unwrap();
        assert!(verify_signature(message, &sig, kp.public_key()).unwrap());
    }

    #[test]
    fn test_from_bytes_accepts_matching_keypair() {
        let kp = DilithiumKeypair::generate();
        let restored =
            DilithiumKeypair::from_bytes(kp.public_key().to_vec(), kp.secret_key.clone());
        assert!(restored.is_ok(), "matching generated keypair must load");
    }

    #[test]
    fn test_from_bytes_rejects_mismatched_keypair() {
        let public_owner = DilithiumKeypair::generate();
        let secret_owner = DilithiumKeypair::generate();
        let result = DilithiumKeypair::from_bytes(
            public_owner.public_key().to_vec(),
            secret_owner.secret_key.clone(),
        );
        assert!(
            result.is_err(),
            "a valid-length public key and unrelated secret key must not be accepted"
        );
    }

    #[test]
    fn test_from_bytes_rejects_invalid_encoded_key_material() {
        let result = DilithiumKeypair::from_bytes(
            vec![0u8; PUBLIC_KEY_SIZE],
            vec![0u8; SECRET_KEY_SIZE],
        );
        assert!(result.is_err(), "non-key byte strings must fail closed");
    }

    #[test]
    fn test_verify_rejects_tampered_message() {
        let kp = DilithiumKeypair::generate();
        let sig = kp.sign(b"original").unwrap();
        assert!(!verify_signature(b"tampered", &sig, kp.public_key()).unwrap());
    }

    #[test]
    fn test_verify_rejects_mutated_signed_message() {
        let kp = DilithiumKeypair::generate();
        let message = b"red-team mutation test";
        let sig = kp.sign(message).unwrap();

        for index in [0, sig.len() / 2, sig.len() - 1] {
            let mut mutated = sig.clone();
            mutated[index] ^= 0x01;
            assert!(
                !matches!(verify_signature(message, &mutated, kp.public_key()), Ok(true)),
                "mutation at byte {index} was accepted"
            );
        }
    }

    #[test]
    fn test_verify_rejects_truncated_signed_message() {
        let kp = DilithiumKeypair::generate();
        let message = b"truncation test";
        let sig = kp.sign(message).unwrap();
        let truncated = &sig[..sig.len() - 1];
        assert!(!matches!(
            verify_signature(message, truncated, kp.public_key()),
            Ok(true)
        ));
    }

    #[test]
    fn test_verify_rejects_another_keypair_public_key() {
        let signer = DilithiumKeypair::generate();
        let other = DilithiumKeypair::generate();
        let message = b"key substitution test";
        let sig = signer.sign(message).unwrap();
        assert!(!matches!(
            verify_signature(message, &sig, other.public_key()),
            Ok(true)
        ));
    }

    #[test]
    fn test_verify_rejects_wrong_public_key_length() {
        let err = verify_signature(b"message", b"signature", &[0u8; PUBLIC_KEY_SIZE - 1]);
        assert!(matches!(err, Err(ZkpError::InvalidKeyMaterial(_))));
    }

    #[test]
    fn test_client_id_is_deterministic_for_loaded_valid_keypair() {
        let kp = DilithiumKeypair::generate();
        let loaded_a =
            DilithiumKeypair::from_bytes(kp.public_key().to_vec(), kp.secret_key.clone()).unwrap();
        let loaded_b =
            DilithiumKeypair::from_bytes(kp.public_key().to_vec(), kp.secret_key.clone()).unwrap();
        assert_eq!(loaded_a.client_id(), loaded_b.client_id());
    }
}
