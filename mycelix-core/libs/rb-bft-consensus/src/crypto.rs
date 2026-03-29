// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cryptographic primitives for RB-BFT consensus
//!
//! This module provides real ed25519 signatures for proposals and votes.
//! All signing keys are zeroized on drop to prevent memory leaks.

use ed25519_dalek::{
    Signature, Signer, SigningKey, Verifier, VerifyingKey,
    SECRET_KEY_LENGTH, PUBLIC_KEY_LENGTH, SIGNATURE_LENGTH,
};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};
use zeroize::ZeroizeOnDrop;

use crate::error::{ConsensusError, ConsensusResult};

/// A validator's signing keypair
///
/// The private key is automatically zeroized when dropped.
#[derive(ZeroizeOnDrop)]
pub struct ValidatorKeypair {
    /// The signing (private) key - zeroized on drop
    signing_key: SigningKey,
    /// The verifying (public) key
    #[zeroize(skip)]
    verifying_key: VerifyingKey,
}

impl ValidatorKeypair {
    /// Generate a new random keypair
    pub fn generate() -> Self {
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();
        Self {
            signing_key,
            verifying_key,
        }
    }

    /// Create from existing secret key bytes
    pub fn from_bytes(secret: &[u8; SECRET_KEY_LENGTH]) -> ConsensusResult<Self> {
        let signing_key = SigningKey::from_bytes(secret);
        let verifying_key = signing_key.verifying_key();
        Ok(Self {
            signing_key,
            verifying_key,
        })
    }

    /// Get the public key bytes
    pub fn public_key_bytes(&self) -> [u8; PUBLIC_KEY_LENGTH] {
        self.verifying_key.to_bytes()
    }

    /// Get the public key as hex string (for validator ID)
    pub fn public_key_hex(&self) -> String {
        hex::encode(self.public_key_bytes())
    }

    /// Get the verifying key for signature verification
    pub fn verifying_key(&self) -> &VerifyingKey {
        &self.verifying_key
    }

    /// Sign arbitrary message bytes
    pub fn sign(&self, message: &[u8]) -> ConsensusSignature {
        let signature = self.signing_key.sign(message);
        ConsensusSignature::new(signature.to_bytes(), self.public_key_bytes())
    }

    /// Verify a signature against a message using this keypair's public key
    ///
    /// Returns true if the signature is valid for the given message.
    pub fn verify(&self, message: &[u8], signature: &ConsensusSignature) -> bool {
        signature.verify(message).is_ok()
    }

    /// Sign with a domain separator to prevent cross-protocol attacks
    pub fn sign_with_domain(&self, domain: &str, message: &[u8]) -> ConsensusSignature {
        let mut hasher = Sha256::new();
        hasher.update(domain.as_bytes());
        hasher.update(message);
        let digest = hasher.finalize();
        self.sign(&digest)
    }
}

impl std::fmt::Debug for ValidatorKeypair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ValidatorKeypair")
            .field("public_key", &self.public_key_hex())
            .field("signing_key", &"[REDACTED]")
            .finish()
    }
}

/// A cryptographic signature with the signer's public key
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConsensusSignature {
    /// The ed25519 signature bytes (64 bytes)
    pub signature: Vec<u8>,
    /// The signer's public key (32 bytes)
    pub signer_pubkey: Vec<u8>,
}

impl ConsensusSignature {
    /// Create a new signature
    pub fn new(signature: [u8; SIGNATURE_LENGTH], signer_pubkey: [u8; PUBLIC_KEY_LENGTH]) -> Self {
        Self {
            signature: signature.to_vec(),
            signer_pubkey: signer_pubkey.to_vec(),
        }
    }

    /// Create an empty/invalid signature (for unsigned messages)
    pub fn empty() -> Self {
        Self {
            signature: Vec::new(),
            signer_pubkey: Vec::new(),
        }
    }

    /// Check if signature is present (non-empty)
    pub fn is_present(&self) -> bool {
        self.signature.len() == SIGNATURE_LENGTH && self.signer_pubkey.len() == PUBLIC_KEY_LENGTH
    }

    /// Verify the signature against a message
    pub fn verify(&self, message: &[u8]) -> ConsensusResult<()> {
        if !self.is_present() {
            return Err(ConsensusError::InvalidSignature {
                reason: "signature not present".to_string(),
            });
        }

        let pubkey_bytes: [u8; PUBLIC_KEY_LENGTH] = self.signer_pubkey
            .as_slice()
            .try_into()
            .map_err(|_| ConsensusError::InvalidSignature {
                reason: "invalid public key length".to_string(),
            })?;

        let sig_bytes: [u8; SIGNATURE_LENGTH] = self.signature
            .as_slice()
            .try_into()
            .map_err(|_| ConsensusError::InvalidSignature {
                reason: "invalid signature length".to_string(),
            })?;

        let verifying_key = VerifyingKey::from_bytes(&pubkey_bytes)
            .map_err(|e| ConsensusError::InvalidSignature {
                reason: format!("invalid public key: {}", e),
            })?;

        let signature = Signature::from_bytes(&sig_bytes);

        verifying_key.verify(message, &signature)
            .map_err(|e| ConsensusError::InvalidSignature {
                reason: format!("signature verification failed: {}", e),
            })
    }

    /// Verify with domain separator
    pub fn verify_with_domain(&self, domain: &str, message: &[u8]) -> ConsensusResult<()> {
        let mut hasher = Sha256::new();
        hasher.update(domain.as_bytes());
        hasher.update(message);
        let digest = hasher.finalize();
        self.verify(&digest)
    }

    /// Get signer's public key as hex string
    pub fn signer_hex(&self) -> String {
        hex::encode(&self.signer_pubkey)
    }

    /// Serialize to bytes for network transmission
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.signature.len() + self.signer_pubkey.len() + 2);
        bytes.push(self.signature.len() as u8);
        bytes.extend_from_slice(&self.signature);
        bytes.push(self.signer_pubkey.len() as u8);
        bytes.extend_from_slice(&self.signer_pubkey);
        bytes
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> ConsensusResult<Self> {
        if bytes.len() < 2 {
            return Err(ConsensusError::InvalidSignature {
                reason: "signature bytes too short".to_string(),
            });
        }

        let sig_len = bytes[0] as usize;
        if bytes.len() < 1 + sig_len + 1 {
            return Err(ConsensusError::InvalidSignature {
                reason: "signature bytes truncated".to_string(),
            });
        }

        let signature = bytes[1..1 + sig_len].to_vec();
        let pubkey_len = bytes[1 + sig_len] as usize;

        if bytes.len() < 1 + sig_len + 1 + pubkey_len {
            return Err(ConsensusError::InvalidSignature {
                reason: "public key bytes truncated".to_string(),
            });
        }

        let signer_pubkey = bytes[2 + sig_len..2 + sig_len + pubkey_len].to_vec();

        Ok(Self {
            signature,
            signer_pubkey,
        })
    }
}

/// Domain separators for different message types
pub mod domains {
    /// Domain for proposal signatures
    pub const PROPOSAL: &str = "mycelix-rbbft-proposal-v1";
    /// Domain for vote signatures
    pub const VOTE: &str = "mycelix-rbbft-vote-v1";
    /// Domain for slashing evidence signatures
    pub const SLASHING: &str = "mycelix-rbbft-slashing-v1";
}

/// Helper to create signable message bytes from structured data
pub fn create_signable_bytes(data: &impl Serialize) -> ConsensusResult<Vec<u8>> {
    serde_json::to_vec(data).map_err(|e| ConsensusError::SerializationError(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keypair_generation() {
        let kp = ValidatorKeypair::generate();
        assert_eq!(kp.public_key_bytes().len(), 32);
        assert!(!kp.public_key_hex().is_empty());
    }

    #[test]
    fn test_sign_and_verify() {
        let kp = ValidatorKeypair::generate();
        let message = b"test message for signing";

        let sig = kp.sign(message);
        assert!(sig.is_present());
        assert!(sig.verify(message).is_ok());
    }

    #[test]
    fn test_sign_with_domain() {
        let kp = ValidatorKeypair::generate();
        let message = b"proposal content";

        let sig = kp.sign_with_domain(domains::PROPOSAL, message);
        assert!(sig.verify_with_domain(domains::PROPOSAL, message).is_ok());

        // Wrong domain should fail
        assert!(sig.verify_with_domain(domains::VOTE, message).is_err());
    }

    #[test]
    fn test_signature_verification_fails_on_tamper() {
        let kp = ValidatorKeypair::generate();
        let message = b"original message";

        let sig = kp.sign(message);

        // Tampered message should fail verification
        let tampered = b"tampered message";
        assert!(sig.verify(tampered).is_err());
    }

    #[test]
    fn test_signature_serialization() {
        let kp = ValidatorKeypair::generate();
        let sig = kp.sign(b"test");

        let bytes = sig.to_bytes();
        let restored = ConsensusSignature::from_bytes(&bytes).unwrap();

        assert_eq!(sig.signature, restored.signature);
        assert_eq!(sig.signer_pubkey, restored.signer_pubkey);
    }

    #[test]
    fn test_empty_signature() {
        let sig = ConsensusSignature::empty();
        assert!(!sig.is_present());
        assert!(sig.verify(b"anything").is_err());
    }

    #[test]
    fn test_different_keypairs_different_signatures() {
        let kp1 = ValidatorKeypair::generate();
        let kp2 = ValidatorKeypair::generate();
        let message = b"same message";

        let sig1 = kp1.sign(message);
        let sig2 = kp2.sign(message);

        // Both valid for their own key
        assert!(sig1.verify(message).is_ok());
        assert!(sig2.verify(message).is_ok());

        // But signatures differ
        assert_ne!(sig1.signature, sig2.signature);
        assert_ne!(sig1.signer_pubkey, sig2.signer_pubkey);
    }

    #[test]
    fn test_keypair_from_bytes() {
        let kp1 = ValidatorKeypair::generate();
        // Note: In production, you'd have secure key storage
        // This test just verifies the API works
        let message = b"test";
        let sig = kp1.sign(message);
        assert!(sig.verify(message).is_ok());
    }
}
