// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! ML-DSA-65 post-quantum digital signatures for DKG attestation.
//!
//! Provides sign/verify operations using FIPS 204 ML-DSA-65 (Dilithium3),
//! compatible with Holochain WASM zomes (pure Rust, no FFI).
//!
//! # Example
//!
//! ```ignore
//! use feldman_dkg::pq_sig::{generate_signing_keypair, sign, verify};
//!
//! let kp = generate_signing_keypair();
//! let message = b"governance decision hash";
//! let signature = sign(&kp.signing_key, message);
//! assert!(verify(&kp.verifying_key_bytes(), message, &signature).is_ok());
//! ```

use ml_dsa::signature::{Keypair, SignatureEncoding, Signer, Verifier};
use ml_dsa::{EncodedVerifyingKey, KeyGen, MlDsa65, SigningKey, VerifyingKey};

/// An ML-DSA-65 keypair for post-quantum digital signatures.
pub struct MlDsaKeyPair {
    /// The signing (private) key.
    pub signing_key: SigningKey<MlDsa65>,
    /// The verifying (public) key.
    pub verifying_key: VerifyingKey<MlDsa65>,
}

impl MlDsaKeyPair {
    /// Serialize the verifying (public) key to bytes for distribution.
    ///
    /// Returns 1,952 bytes for ML-DSA-65.
    pub fn verifying_key_bytes(&self) -> Vec<u8> {
        self.verifying_key.encode().to_vec()
    }
}

/// Generate a new ML-DSA-65 keypair using OS randomness.
///
/// # Panics
///
/// Panics if the OS CSPRNG is unavailable.
pub fn generate_signing_keypair() -> MlDsaKeyPair {
    let mut seed = ml_dsa::B32::default();
    getrandom::fill(&mut seed).expect("OS CSPRNG unavailable");
    let kp = MlDsa65::from_seed(&seed);
    let verifying_key = kp.verifying_key();
    MlDsaKeyPair {
        signing_key: kp,
        verifying_key,
    }
}

/// Sign a message with an ML-DSA-65 signing key.
///
/// Returns a 3,309-byte signature.
pub fn sign(signing_key: &SigningKey<MlDsa65>, message: &[u8]) -> Vec<u8> {
    let sig = signing_key.sign(message);
    sig.to_bytes().to_vec()
}

/// Verify an ML-DSA-65 signature against a verifying key.
///
/// # Arguments
///
/// * `verifying_key_bytes` - The signer's public key (1,952 bytes)
/// * `message` - The original signed message
/// * `signature_bytes` - The signature to verify (3,309 bytes)
pub fn verify(
    verifying_key_bytes: &[u8],
    message: &[u8],
    signature_bytes: &[u8],
) -> Result<(), String> {
    let vk_enc = EncodedVerifyingKey::<MlDsa65>::try_from(verifying_key_bytes).map_err(|_| {
        format!(
            "Invalid verifying key length: expected 1952 bytes, got {}",
            verifying_key_bytes.len()
        )
    })?;
    let vk = VerifyingKey::<MlDsa65>::decode(&vk_enc);

    let sig = ml_dsa::Signature::<MlDsa65>::try_from(signature_bytes).map_err(|_| {
        format!(
            "Invalid signature: expected 3309 bytes, got {}",
            signature_bytes.len()
        )
    })?;

    vk.verify(message, &sig)
        .map_err(|e| format!("ML-DSA-65 signature verification failed: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keypair_generation() {
        let kp = generate_signing_keypair();
        let vk_bytes = kp.verifying_key_bytes();
        // ML-DSA-65 verifying key is 1,952 bytes
        assert_eq!(vk_bytes.len(), 1952);
    }

    #[test]
    fn test_sign_verify_roundtrip() {
        let kp = generate_signing_keypair();
        let message = b"governance proposal MIP-42 tally hash";
        let signature = sign(&kp.signing_key, message);
        assert_eq!(signature.len(), 3309);
        assert!(verify(&kp.verifying_key_bytes(), message, &signature).is_ok());
    }

    #[test]
    fn test_wrong_key_rejected() {
        let kp1 = generate_signing_keypair();
        let kp2 = generate_signing_keypair();
        let message = b"governance proposal";
        let signature = sign(&kp1.signing_key, message);
        let result = verify(&kp2.verifying_key_bytes(), message, &signature);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("verification failed"));
    }

    #[test]
    fn test_wrong_message_rejected() {
        let kp = generate_signing_keypair();
        let signature = sign(&kp.signing_key, b"correct message");
        let result = verify(&kp.verifying_key_bytes(), b"wrong message", &signature);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("verification failed"));
    }

    #[test]
    fn test_tampered_signature_rejected() {
        let kp = generate_signing_keypair();
        let message = b"governance proposal";
        let mut signature = sign(&kp.signing_key, message);
        signature[0] ^= 0xFF;
        let result = verify(&kp.verifying_key_bytes(), message, &signature);
        assert!(result.is_err());
    }

    #[test]
    fn test_verifying_key_roundtrip() {
        let kp = generate_signing_keypair();
        let vk_bytes = kp.verifying_key_bytes();
        let message = b"test message for key roundtrip";
        let signature = sign(&kp.signing_key, message);
        // Verify using the serialized/deserialized key bytes
        assert!(verify(&vk_bytes, message, &signature).is_ok());
    }

    #[test]
    fn test_empty_message_signing() {
        let kp = generate_signing_keypair();
        let signature = sign(&kp.signing_key, b"");
        assert_eq!(signature.len(), 3309);
        assert!(verify(&kp.verifying_key_bytes(), b"", &signature).is_ok());
        // Empty-message sig must not verify against non-empty message
        assert!(verify(&kp.verifying_key_bytes(), b"x", &signature).is_err());
    }

    #[test]
    fn test_large_message_signing() {
        let kp = generate_signing_keypair();
        let large_msg = vec![0xABu8; 100_000]; // 100 KB
        let signature = sign(&kp.signing_key, &large_msg);
        assert_eq!(signature.len(), 3309);
        assert!(verify(&kp.verifying_key_bytes(), &large_msg, &signature).is_ok());
    }

    #[test]
    fn test_tampered_verifying_key_rejected() {
        let kp = generate_signing_keypair();
        let message = b"governance proposal";
        let signature = sign(&kp.signing_key, message);

        let mut vk_bytes = kp.verifying_key_bytes();
        vk_bytes[0] ^= 0xFF; // flip one byte
        let result = verify(&vk_bytes, message, &signature);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_key_lengths_rejected() {
        let kp = generate_signing_keypair();
        let message = b"test";
        let signature = sign(&kp.signing_key, message);

        // Too-short verifying key
        let result = verify(&[0u8; 100], message, &signature);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("1952"));

        // Too-short signature
        let result = verify(&kp.verifying_key_bytes(), message, &[0u8; 100]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("3309"));
    }
}
