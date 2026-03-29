// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cryptographic operations for Mycelix supply chain.
//!
//! This crate provides signing, verification, and future SD-JWT/BBS+ support.

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CryptoError {
    #[error("Signature verification failed")]
    SignatureVerificationFailed,
    #[error("Invalid key format: {0}")]
    InvalidKeyFormat(String),
    #[error("Signing error: {0}")]
    SigningError(String),
}

/// Ed25519 keypair for signing VCs
pub struct KeyPair {
    signing_key: SigningKey,
}

impl KeyPair {
    /// Generate a new random keypair
    pub fn generate() -> Self {
        let signing_key = SigningKey::generate(&mut OsRng);
        Self { signing_key }
    }

    /// Create from a seed (32 bytes)
    pub fn from_seed(seed: &[u8; 32]) -> Self {
        let signing_key = SigningKey::from_bytes(seed);
        Self { signing_key }
    }

    /// Get the public key (verifying key)
    pub fn public_key(&self) -> VerifyingKey {
        self.signing_key.verifying_key()
    }

    /// Get the DID for this keypair
    /// Format: did:key:<multibase-encoded-public-key>
    pub fn did(&self) -> String {
        let public_key_bytes = self.public_key().to_bytes();
        // Simple base58 encoding (in production, use proper multicodec/multibase)
        format!("did:key:{}", hex::encode(public_key_bytes))
    }

    /// Get the secret key bytes (seed)
    pub fn to_bytes(&self) -> [u8; 32] {
        self.signing_key.to_bytes()
    }

    /// Sign a message
    pub fn sign(&self, message: &[u8]) -> Signature {
        self.signing_key.sign(message)
    }

    /// Create a JWT signature for a VC
    pub fn sign_jwt(&self, header: &str, payload: &str) -> Result<String, CryptoError> {
        let message = format!("{}.{}", header, payload);
        let signature = self.sign(message.as_bytes());

        // Encode signature as base64url
        let sig_b64 = base64_url::encode(&signature.to_bytes());

        Ok(format!("{}.{}", message, sig_b64))
    }
}

/// Verify a signature
pub fn verify_signature(
    public_key: &VerifyingKey,
    message: &[u8],
    signature: &Signature,
) -> Result<(), CryptoError> {
    public_key
        .verify(message, signature)
        .map_err(|_| CryptoError::SignatureVerificationFailed)
}

/// Compute SHA-256 hash
pub fn hash_sha256(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    format!("{:x}", result)
}

/// JWT header for VC signatures
#[derive(Debug, Serialize, Deserialize)]
pub struct JwtHeader {
    pub alg: String,
    pub typ: String,
}

impl Default for JwtHeader {
    fn default() -> Self {
        Self {
            alg: "EdDSA".to_string(),
            typ: "JWT".to_string(),
        }
    }
}

/// Create a signed VC JWT
pub fn create_vc_jwt(keypair: &KeyPair, vc: &serde_json::Value) -> Result<String, CryptoError> {
    let header = JwtHeader::default();
    let header_json = serde_json::to_string(&header)
        .map_err(|e| CryptoError::SigningError(e.to_string()))?;
    let header_b64 = base64_url::encode(header_json.as_bytes());

    let payload_json =
        serde_json::to_string(vc).map_err(|e| CryptoError::SigningError(e.to_string()))?;
    let payload_b64 = base64_url::encode(payload_json.as_bytes());

    keypair.sign_jwt(&header_b64, &payload_b64)
}

// Helper module for base64url encoding
mod base64_url {
    use base64::Engine;

    pub fn encode(data: &[u8]) -> String {
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(data)
    }

    #[allow(dead_code)] // Public API for base64url decoding
    pub fn decode(data: &str) -> Result<Vec<u8>, base64::DecodeError> {
        base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keypair_generation() {
        let keypair = KeyPair::generate();
        let public_key = keypair.public_key();
        let did = keypair.did();

        assert!(did.starts_with("did:key:"));
        assert_eq!(did.len(), 72); // did:key: prefix + 64 hex chars
    }

    #[test]
    fn test_keypair_from_seed() {
        let seed = [42u8; 32];
        let keypair1 = KeyPair::from_seed(&seed);
        let keypair2 = KeyPair::from_seed(&seed);

        // Same seed should produce same DID
        assert_eq!(keypair1.did(), keypair2.did());
    }

    #[test]
    fn test_keypair_different_seeds() {
        let seed1 = [1u8; 32];
        let seed2 = [2u8; 32];
        let keypair1 = KeyPair::from_seed(&seed1);
        let keypair2 = KeyPair::from_seed(&seed2);

        // Different seeds should produce different DIDs
        assert_ne!(keypair1.did(), keypair2.did());
    }

    #[test]
    fn test_sign_and_verify() {
        let keypair = KeyPair::generate();
        let message = b"test message";

        let signature = keypair.sign(message);
        let result = verify_signature(&keypair.public_key(), message, &signature);

        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_wrong_message_fails() {
        let keypair = KeyPair::generate();
        let message = b"original message";
        let wrong_message = b"tampered message";

        let signature = keypair.sign(message);
        let result = verify_signature(&keypair.public_key(), wrong_message, &signature);

        assert!(result.is_err());
    }

    #[test]
    fn test_verify_wrong_key_fails() {
        let keypair1 = KeyPair::generate();
        let keypair2 = KeyPair::generate();
        let message = b"test message";

        let signature = keypair1.sign(message);
        let result = verify_signature(&keypair2.public_key(), message, &signature);

        assert!(result.is_err());
    }

    #[test]
    fn test_hash_deterministic() {
        let data = b"test data";
        let hash1 = hash_sha256(data);
        let hash2 = hash_sha256(data);

        assert_eq!(hash1, hash2);
        assert_eq!(hash1.len(), 64); // SHA-256 produces 64 hex characters
    }

    #[test]
    fn test_hash_different_inputs() {
        let data1 = b"test data 1";
        let data2 = b"test data 2";
        let hash1 = hash_sha256(data1);
        let hash2 = hash_sha256(data2);

        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_create_vc_jwt() {
        let keypair = KeyPair::generate();
        let vc = serde_json::json!({
            "@context": ["https://www.w3.org/2018/credentials/v1"],
            "type": ["VerifiableCredential"],
            "issuer": keypair.did(),
            "credentialSubject": {
                "test": "data"
            }
        });

        let jwt = create_vc_jwt(&keypair, &vc).unwrap();

        // JWT should have 3 parts separated by dots
        let parts: Vec<&str> = jwt.split('.').collect();
        assert_eq!(parts.len(), 3);

        // Each part should be base64url encoded
        assert!(!parts[0].is_empty());
        assert!(!parts[1].is_empty());
        assert!(!parts[2].is_empty());
    }

    #[test]
    fn test_jwt_header() {
        let header = JwtHeader::default();
        assert_eq!(header.alg, "EdDSA");
        assert_eq!(header.typ, "JWT");
    }

    #[test]
    fn test_base64url_encode_decode() {
        let data = b"hello world";
        let encoded = base64_url::encode(data);
        let decoded = base64_url::decode(&encoded).unwrap();

        assert_eq!(data, decoded.as_slice());
        // base64url should not contain = padding
        assert!(!encoded.contains('='));
    }
}
