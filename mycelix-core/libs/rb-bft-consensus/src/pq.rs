// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Post-Quantum Hybrid Signatures
//!
//! This module implements hybrid signatures combining Dilithium (NIST PQC standard)
//! with ed25519. The hybrid approach provides security against both classical and
//! quantum adversaries - if either signature scheme remains secure, the combined
//! signature is secure.
//!
//! ## Why Hybrid?
//!
//! 1. **Quantum Resistance**: Dilithium is believed quantum-safe
//! 2. **Classical Security**: Ed25519 has decades of cryptanalysis
//! 3. **Defense in Depth**: Both must be broken to forge signatures
//!
//! ## Signature Format
//!
//! A hybrid signature contains:
//! - Ed25519 signature (64 bytes)
//! - Dilithium signature (~2420 bytes for level 2)
//! - Domain separator for binding

use ed25519_dalek::{Signer, SigningKey, VerifyingKey, Signature as Ed25519Signature, Verifier};
use pqcrypto_dilithium::dilithium2;
use pqcrypto_traits::sign::{PublicKey as PqPublicKey, SecretKey as PqSecretKey, DetachedSignature};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};
// zeroize is used for ed25519 keys automatically

use crate::error::{ConsensusError, ConsensusResult};

/// Domain separator for hybrid signatures
const HYBRID_DOMAIN: &[u8] = b"MYCELIX-RBBFT-HYBRID-PQ-V1";

/// Hybrid keypair containing both ed25519 and Dilithium keys
pub struct HybridKeypair {
    /// Ed25519 signing key
    ed25519_signing: SigningKey,
    /// Ed25519 verifying key
    ed25519_verifying: VerifyingKey,
    /// Dilithium secret key
    dilithium_secret: dilithium2::SecretKey,
    /// Dilithium public key
    dilithium_public: dilithium2::PublicKey,
}

impl HybridKeypair {
    /// Generate a new hybrid keypair
    pub fn generate() -> Self {
        // Generate ed25519 keys
        let ed25519_signing = SigningKey::generate(&mut OsRng);
        let ed25519_verifying = ed25519_signing.verifying_key();

        // Generate Dilithium keys
        let (dilithium_public, dilithium_secret) = dilithium2::keypair();

        Self {
            ed25519_signing,
            ed25519_verifying,
            dilithium_secret,
            dilithium_public,
        }
    }

    /// Create from existing key bytes
    ///
    /// # Arguments
    /// * `ed25519_bytes` - 32 bytes for ed25519 secret key
    /// * `dilithium_secret_bytes` - Dilithium secret key bytes
    /// * `dilithium_public_bytes` - Dilithium public key bytes
    pub fn from_bytes(
        ed25519_bytes: &[u8; 32],
        dilithium_secret_bytes: &[u8],
        dilithium_public_bytes: &[u8],
    ) -> ConsensusResult<Self> {
        let ed25519_signing = SigningKey::from_bytes(ed25519_bytes);
        let ed25519_verifying = ed25519_signing.verifying_key();

        let dilithium_secret = dilithium2::SecretKey::from_bytes(dilithium_secret_bytes)
            .map_err(|_| ConsensusError::InvalidSignature {
                reason: "Invalid Dilithium secret key bytes".to_string(),
            })?;

        let dilithium_public = dilithium2::PublicKey::from_bytes(dilithium_public_bytes)
            .map_err(|_| ConsensusError::InvalidSignature {
                reason: "Invalid Dilithium public key bytes".to_string(),
            })?;

        Ok(Self {
            ed25519_signing,
            ed25519_verifying,
            dilithium_secret,
            dilithium_public,
        })
    }

    /// Get the hybrid public key
    pub fn public_key(&self) -> HybridPublicKey {
        HybridPublicKey {
            ed25519: self.ed25519_verifying.to_bytes(),
            dilithium: self.dilithium_public.as_bytes().to_vec(),
        }
    }

    /// Sign a message with both signature schemes
    ///
    /// The message is first hashed with the domain separator, then signed by both.
    pub fn sign(&self, message: &[u8]) -> HybridSignature {
        // Create domain-separated message
        let mut hasher = Sha256::new();
        hasher.update(HYBRID_DOMAIN);
        hasher.update((message.len() as u64).to_le_bytes());
        hasher.update(message);
        let digest: [u8; 32] = hasher.finalize().into();

        // Ed25519 signature
        let ed25519_sig = self.ed25519_signing.sign(&digest);

        // Dilithium detached signature
        let dilithium_sig = dilithium2::detached_sign(&digest, &self.dilithium_secret);

        HybridSignature {
            ed25519: ed25519_sig.to_bytes().to_vec(),
            dilithium: dilithium_sig.as_bytes().to_vec(),
        }
    }

    /// Sign with a specific domain
    pub fn sign_with_domain(&self, domain: &str, message: &[u8]) -> HybridSignature {
        let mut hasher = Sha256::new();
        hasher.update(HYBRID_DOMAIN);
        hasher.update(domain.as_bytes());
        hasher.update((message.len() as u64).to_le_bytes());
        hasher.update(message);
        let digest: [u8; 32] = hasher.finalize().into();

        let ed25519_sig = self.ed25519_signing.sign(&digest);
        let dilithium_sig = dilithium2::detached_sign(&digest, &self.dilithium_secret);

        HybridSignature {
            ed25519: ed25519_sig.to_bytes().to_vec(),
            dilithium: dilithium_sig.as_bytes().to_vec(),
        }
    }
}

impl std::fmt::Debug for HybridKeypair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HybridKeypair")
            .field("ed25519_public", &hex::encode(self.ed25519_verifying.to_bytes()))
            .field("dilithium_public", &format!("[{} bytes]", self.dilithium_public.as_bytes().len()))
            .field("secrets", &"[REDACTED]")
            .finish()
    }
}

impl Drop for HybridKeypair {
    fn drop(&mut self) {
        // Ed25519 signing key is automatically zeroized via ed25519-dalek
        // Dilithium keys need manual handling - they're dropped but pqcrypto
        // doesn't implement zeroize, so we just ensure they go out of scope
    }
}

/// Hybrid public key containing both ed25519 and Dilithium public keys
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HybridPublicKey {
    /// Ed25519 public key (32 bytes)
    pub ed25519: [u8; 32],
    /// Dilithium public key (~1312 bytes for level 2)
    pub dilithium: Vec<u8>,
}

impl HybridPublicKey {
    /// Create from bytes
    pub fn from_bytes(ed25519: [u8; 32], dilithium: Vec<u8>) -> Self {
        Self { ed25519, dilithium }
    }

    /// Get a fingerprint of the hybrid public key
    pub fn fingerprint(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(&self.ed25519);
        hasher.update(&self.dilithium);
        let hash = hasher.finalize();
        hex::encode(&hash[..8])
    }

    /// Verify a hybrid signature
    pub fn verify(&self, message: &[u8], signature: &HybridSignature) -> ConsensusResult<()> {
        // Create domain-separated message
        let mut hasher = Sha256::new();
        hasher.update(HYBRID_DOMAIN);
        hasher.update((message.len() as u64).to_le_bytes());
        hasher.update(message);
        let digest: [u8; 32] = hasher.finalize().into();

        self.verify_digest(&digest, signature)
    }

    /// Verify with a specific domain
    pub fn verify_with_domain(
        &self,
        domain: &str,
        message: &[u8],
        signature: &HybridSignature,
    ) -> ConsensusResult<()> {
        let mut hasher = Sha256::new();
        hasher.update(HYBRID_DOMAIN);
        hasher.update(domain.as_bytes());
        hasher.update((message.len() as u64).to_le_bytes());
        hasher.update(message);
        let digest: [u8; 32] = hasher.finalize().into();

        self.verify_digest(&digest, signature)
    }

    /// Verify against a pre-computed digest
    fn verify_digest(&self, digest: &[u8; 32], signature: &HybridSignature) -> ConsensusResult<()> {
        // Verify ed25519 signature
        let ed25519_pubkey = VerifyingKey::from_bytes(&self.ed25519)
            .map_err(|e| ConsensusError::InvalidSignature {
                reason: format!("Invalid ed25519 public key: {}", e),
            })?;

        let ed25519_sig_bytes: [u8; 64] = signature.ed25519.as_slice()
            .try_into()
            .map_err(|_| ConsensusError::InvalidSignature {
                reason: "Ed25519 signature must be 64 bytes".to_string(),
            })?;
        let ed25519_sig = Ed25519Signature::from_bytes(&ed25519_sig_bytes);

        ed25519_pubkey.verify(digest, &ed25519_sig)
            .map_err(|e| ConsensusError::InvalidSignature {
                reason: format!("Ed25519 verification failed: {}", e),
            })?;

        // Verify Dilithium signature
        let dilithium_pubkey = dilithium2::PublicKey::from_bytes(&self.dilithium)
            .map_err(|_| ConsensusError::InvalidSignature {
                reason: "Invalid Dilithium public key".to_string(),
            })?;

        let dilithium_sig = dilithium2::DetachedSignature::from_bytes(&signature.dilithium)
            .map_err(|_| ConsensusError::InvalidSignature {
                reason: "Invalid Dilithium signature".to_string(),
            })?;

        dilithium2::verify_detached_signature(&dilithium_sig, digest, &dilithium_pubkey)
            .map_err(|_| ConsensusError::InvalidSignature {
                reason: "Dilithium verification failed".to_string(),
            })?;

        Ok(())
    }
}

/// Hybrid signature containing both ed25519 and Dilithium signatures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSignature {
    /// Ed25519 signature (64 bytes)
    pub ed25519: Vec<u8>,
    /// Dilithium signature (~2420 bytes for level 2)
    pub dilithium: Vec<u8>,
}

impl HybridSignature {
    /// Get the total size of the hybrid signature
    pub fn size(&self) -> usize {
        self.ed25519.len() + self.dilithium.len()
    }

    /// Create from component signatures
    pub fn from_parts(ed25519: Vec<u8>, dilithium: Vec<u8>) -> Self {
        Self { ed25519, dilithium }
    }
}

/// Signature mode selection for hybrid scheme
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HybridMode {
    /// Both signatures required (maximum security)
    Both,
    /// Only ed25519 (when bandwidth is critical, less PQ security)
    Ed25519Only,
    /// Only Dilithium (for testing PQ transition)
    DilithiumOnly,
}

impl Default for HybridMode {
    fn default() -> Self {
        Self::Both
    }
}

/// Configuration for hybrid signatures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridConfig {
    /// Which signatures to require for verification
    pub mode: HybridMode,
    /// Whether to allow signature mode downgrade
    pub allow_downgrade: bool,
    /// Minimum Dilithium level (2, 3, or 5)
    pub min_dilithium_level: u8,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            mode: HybridMode::Both,
            allow_downgrade: false,
            min_dilithium_level: 2,
        }
    }
}

/// Statistics for hybrid signature performance
#[derive(Debug, Clone, Default)]
pub struct HybridStats {
    /// Number of signatures generated
    pub signatures_generated: u64,
    /// Number of signatures verified
    pub signatures_verified: u64,
    /// Total bytes of signatures generated
    pub total_signature_bytes: u64,
    /// Average signature size
    pub avg_signature_size: f64,
}

impl HybridStats {
    /// Record a new signature
    pub fn record_signature(&mut self, sig: &HybridSignature) {
        self.signatures_generated += 1;
        let size = sig.size() as u64;
        self.total_signature_bytes += size;
        self.avg_signature_size = self.total_signature_bytes as f64 / self.signatures_generated as f64;
    }

    /// Record a verification
    pub fn record_verification(&mut self) {
        self.signatures_verified += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keypair_generation() {
        let kp = HybridKeypair::generate();
        let pk = kp.public_key();

        assert_eq!(pk.ed25519.len(), 32);
        assert!(!pk.dilithium.is_empty());
        println!("Hybrid public key fingerprint: {}", pk.fingerprint());
    }

    #[test]
    fn test_sign_and_verify() {
        let kp = HybridKeypair::generate();
        let pk = kp.public_key();
        let message = b"Hello, post-quantum world!";

        let sig = kp.sign(message);
        assert!(pk.verify(message, &sig).is_ok());
    }

    #[test]
    fn test_wrong_message_fails() {
        let kp = HybridKeypair::generate();
        let pk = kp.public_key();

        let sig = kp.sign(b"original message");
        assert!(pk.verify(b"wrong message", &sig).is_err());
    }

    #[test]
    fn test_wrong_key_fails() {
        let kp1 = HybridKeypair::generate();
        let kp2 = HybridKeypair::generate();
        let message = b"test message";

        let sig = kp1.sign(message);
        // Verification with wrong public key should fail
        assert!(kp2.public_key().verify(message, &sig).is_err());
    }

    #[test]
    fn test_domain_separation() {
        let kp = HybridKeypair::generate();
        let pk = kp.public_key();
        let message = b"test message";

        let sig1 = kp.sign_with_domain("domain1", message);
        let sig2 = kp.sign_with_domain("domain2", message);

        // Same message, different domains -> different signatures
        assert_ne!(sig1.ed25519, sig2.ed25519);

        // Verify works with correct domain
        assert!(pk.verify_with_domain("domain1", message, &sig1).is_ok());

        // Verify fails with wrong domain
        assert!(pk.verify_with_domain("domain2", message, &sig1).is_err());
    }

    #[test]
    fn test_signature_size() {
        let kp = HybridKeypair::generate();
        let sig = kp.sign(b"test");

        // Ed25519: 64 bytes, Dilithium2: ~2420 bytes
        assert_eq!(sig.ed25519.len(), 64);
        assert!(sig.dilithium.len() > 2000, "Dilithium sig should be > 2000 bytes");

        println!("Hybrid signature size: {} bytes", sig.size());
        println!("  - Ed25519: {} bytes", sig.ed25519.len());
        println!("  - Dilithium: {} bytes", sig.dilithium.len());
    }

    #[test]
    fn test_serialization() {
        let kp = HybridKeypair::generate();
        let pk = kp.public_key();
        let sig = kp.sign(b"test");

        // Serialize and deserialize public key
        let pk_json = serde_json::to_string(&pk).unwrap();
        let pk_restored: HybridPublicKey = serde_json::from_str(&pk_json).unwrap();
        assert_eq!(pk, pk_restored);

        // Serialize and deserialize signature
        let sig_json = serde_json::to_string(&sig).unwrap();
        let sig_restored: HybridSignature = serde_json::from_str(&sig_json).unwrap();

        // Verify restored signature works
        assert!(pk_restored.verify(b"test", &sig_restored).is_ok());
    }

    #[test]
    fn test_hybrid_stats() {
        let kp = HybridKeypair::generate();
        let mut stats = HybridStats::default();

        for i in 0..10 {
            let sig = kp.sign(&[i as u8]);
            stats.record_signature(&sig);
        }

        assert_eq!(stats.signatures_generated, 10);
        assert!(stats.avg_signature_size > 2400.0); // ~2484 bytes per sig
    }
}
