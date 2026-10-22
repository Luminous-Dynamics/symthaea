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
use pqcrypto_traits::sign::{
    DetachedSignature as PQDetachedSignature, PublicKey as PQPublicKey,
    SecretKey as PQSecretKey, SignedMessage,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{ZkpError, ZkpResult};
use crate::types::AuthenticatedProof;

/// Dilithium5 encoded public-key size (bytes).
pub const PUBLIC_KEY_SIZE: usize = 2_592;
/// Dilithium5 encoded secret-key size (bytes).
pub const SECRET_KEY_SIZE: usize = 4_896;
/// Detached Dilithium5 signature size (bytes). The legacy `sign` method
/// returns an attached SignedMessage containing both the signature and message.
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

    /// Sign a message and return only the detached signature bytes.
    ///
    /// New protocol formats should prefer detached signatures when the message
    /// is already carried separately. The legacy `sign` method intentionally
    /// retains PQCrypto's attached SignedMessage format for compatibility.
    pub fn sign_detached(&self, message: &[u8]) -> ZkpResult<Vec<u8>> {
        let sk = dilithium5::SecretKey::from_bytes(&self.secret_key)
            .map_err(|e| ZkpError::InvalidKeyMaterial(format!("secret key: {:?}", e)))?;
        let signature = dilithium5::detached_sign(message, &sk);
        Ok(signature.as_bytes().to_vec())
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

/// Verify a legacy PQCrypto attached SignedMessage.
///
/// The `signature` argument must contain the encoded message plus its
/// signature, as returned by `DilithiumKeypair::sign`. For new protocols,
/// prefer `verify_detached_signature`.
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

/// Verify a detached Dilithium5 signature against a separately supplied message.
///
/// Returns `Ok(false)` for a cryptographically invalid signature and a
/// structured error for malformed key/signature encodings.
pub fn verify_detached_signature(
    message: &[u8],
    signature: &[u8],
    public_key: &[u8],
) -> ZkpResult<bool> {
    if public_key.len() != PUBLIC_KEY_SIZE {
        return Err(ZkpError::InvalidKeyMaterial(format!(
            "public key: expected {} bytes, got {}",
            PUBLIC_KEY_SIZE,
            public_key.len()
        )));
    }
    if signature.len() != SIGNATURE_SIZE {
        return Err(ZkpError::SignatureInvalid);
    }

    let pk = dilithium5::PublicKey::from_bytes(public_key)
        .map_err(|e| ZkpError::InvalidKeyMaterial(format!("public key parse: {:?}", e)))?;
    let sig = dilithium5::DetachedSignature::from_bytes(signature)
        .map_err(|_| ZkpError::SignatureInvalid)?;

    match dilithium5::verify_detached_signature(&sig, message, &pk) {
        Ok(()) => Ok(true),
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
        let valid = verify_signature(message, &sig, kp.public_key()).unwrap();
        assert!(valid);
    }

    #[test]
    fn test_detached_signature_roundtrip_and_exact_size() {
        let kp = DilithiumKeypair::generate();
        let message = b"detached signature test";
        let signature = kp.sign_detached(message).unwrap();

        assert_eq!(signature.len(), SIGNATURE_SIZE);
        assert!(verify_detached_signature(message, &signature, kp.public_key()).unwrap());
    }

    #[test]
    fn test_detached_signature_rejects_mutated_message() {
        let kp = DilithiumKeypair::generate();
        let signature = kp.sign_detached(b"original statement").unwrap();
        assert!(
            !verify_detached_signature(b"changed statement", &signature, kp.public_key())
                .unwrap()
        );
    }

    #[test]
    fn test_detached_signature_rejects_mutated_signature() {
        let kp = DilithiumKeypair::generate();
        let message = b"mutation test";
        let signature = kp.sign_detached(message).unwrap();

        for index in [0, signature.len() / 2, signature.len() - 1] {
            let mut mutated = signature.clone();
            mutated[index] ^= 0x01;
            assert!(
                !matches!(
                    verify_detached_signature(message, &mutated, kp.public_key()),
                    Ok(true)
                ),
                "mutated detached signature at byte {index} was accepted"
            );
        }
    }

    #[test]
    fn test_detached_signature_rejects_truncation_and_wrong_key() {
        let signer = DilithiumKeypair::generate();
        let other = DilithiumKeypair::generate();
        let message = b"key and length negative controls";
        let signature = signer.sign_detached(message).unwrap();

        assert!(matches!(
            verify_detached_signature(
                message,
                &signature[..signature.len() - 1],
                signer.public_key()
            ),
            Err(ZkpError::SignatureInvalid)
        ));
        assert!(!verify_detached_signature(message, &signature, other.public_key()).unwrap());
    }

    #[test]
    fn test_detached_signature_rejects_wrong_public_key_length() {
        assert!(matches!(
            verify_detached_signature(
                b"message",
                &[0; SIGNATURE_SIZE],
                &[0; PUBLIC_KEY_SIZE - 1]
            ),
            Err(ZkpError::InvalidKeyMaterial(_))
        ));
    }

    #[test]
    fn test_verify_rejects_tampered_message() {
        let kp = DilithiumKeypair::generate();
        let sig = kp.sign(b"original").unwrap();
        let valid = verify_signature(b"tampered", &sig, kp.public_key()).unwrap();
        assert!(!valid);
    }

    #[test]
    fn test_client_id_deterministic() {
        let generated = DilithiumKeypair::generate();
        let kp1 = DilithiumKeypair::from_bytes(
            generated.public_key().to_vec(),
            generated.secret_key.clone(),
        )
        .unwrap();
        let kp2 = DilithiumKeypair::from_bytes(
            generated.public_key().to_vec(),
            generated.secret_key.clone(),
        )
        .unwrap();
        assert_eq!(kp1.client_id(), kp2.client_id());
    }
}
