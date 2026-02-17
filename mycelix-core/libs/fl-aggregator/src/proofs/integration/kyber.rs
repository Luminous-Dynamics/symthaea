//! Kyber Key Encapsulation Mechanism (KEM)
//!
//! Post-quantum secure key exchange for encrypted proof transport using
//! CRYSTALS-Kyber (NIST FIPS 203).
//!
//! ## Features
//!
//! - Post-quantum secure key encapsulation
//! - Hybrid encryption (Kyber + AES-256-GCM)
//! - Proof envelope encryption
//!
//! ## Usage
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::integration::kyber::{KyberKeyPair, encrypt_envelope, decrypt_envelope};
//!
//! // Generate keypair
//! let keypair = KyberKeyPair::generate();
//!
//! // Encrypt a proof envelope
//! let encrypted = encrypt_envelope(&envelope, &keypair.public_key)?;
//!
//! // Decrypt
//! let decrypted = decrypt_envelope(&encrypted, &keypair.secret_key)?;
//! ```

#[cfg(feature = "proofs-kyber")]
use pqcrypto_kyber::kyber1024;
#[cfg(feature = "proofs-kyber")]
use pqcrypto_traits::kem::{PublicKey, SecretKey, SharedSecret, Ciphertext};

use crate::proofs::{ProofError, ProofResult};
use super::serialization::ProofEnvelope;
use serde::{Deserialize, Serialize};

/// Kyber key pair for post-quantum key encapsulation
#[cfg(feature = "proofs-kyber")]
pub struct KyberKeyPair {
    /// Public key for encapsulation (1568 bytes for Kyber1024)
    pub public_key: kyber1024::PublicKey,
    /// Secret key for decapsulation (3168 bytes for Kyber1024)
    pub secret_key: kyber1024::SecretKey,
}

#[cfg(feature = "proofs-kyber")]
impl KyberKeyPair {
    /// Generate a new Kyber1024 keypair
    pub fn generate() -> Self {
        let (public_key, secret_key) = kyber1024::keypair();
        Self { public_key, secret_key }
    }

    /// Get the public key bytes
    pub fn public_key_bytes(&self) -> Vec<u8> {
        self.public_key.as_bytes().to_vec()
    }

    /// Get the secret key bytes
    pub fn secret_key_bytes(&self) -> Vec<u8> {
        self.secret_key.as_bytes().to_vec()
    }

    /// Create keypair from bytes
    pub fn from_bytes(public_key: &[u8], secret_key: &[u8]) -> ProofResult<Self> {
        let public_key = kyber1024::PublicKey::from_bytes(public_key)
            .map_err(|_| ProofError::InvalidPublicInputs("Invalid Kyber public key".to_string()))?;
        let secret_key = kyber1024::SecretKey::from_bytes(secret_key)
            .map_err(|_| ProofError::InvalidPublicInputs("Invalid Kyber secret key".to_string()))?;
        Ok(Self { public_key, secret_key })
    }
}

/// Encrypted proof envelope with Kyber KEM
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncryptedProofEnvelope {
    /// Kyber ciphertext (encapsulated shared secret)
    pub kem_ciphertext: Vec<u8>,

    /// AES-GCM nonce (12 bytes)
    pub nonce: [u8; 12],

    /// Encrypted envelope data
    pub ciphertext: Vec<u8>,

    /// AES-GCM authentication tag (16 bytes)
    pub auth_tag: [u8; 16],
}

/// Encrypt a proof envelope using Kyber KEM + AES-256-GCM
#[cfg(feature = "proofs-kyber")]
pub fn encrypt_envelope(
    envelope: &ProofEnvelope,
    recipient_public_key: &kyber1024::PublicKey,
) -> ProofResult<EncryptedProofEnvelope> {
    use aes_gcm::{Aes256Gcm, KeyInit, aead::Aead};
    use aes_gcm::aead::generic_array::GenericArray;
    use rand::RngCore;

    // Encapsulate shared secret
    let (shared_secret, kem_ciphertext) = kyber1024::encapsulate(recipient_public_key);

    // Derive AES key from shared secret (first 32 bytes)
    let shared_bytes = shared_secret.as_bytes();
    let aes_key = GenericArray::from_slice(&shared_bytes[..32]);

    // Generate random nonce
    let mut nonce_bytes = [0u8; 12];
    rand::rngs::OsRng.fill_bytes(&mut nonce_bytes);
    let nonce = GenericArray::from_slice(&nonce_bytes);

    // Serialize envelope
    let plaintext = serde_json::to_vec(envelope)
        .map_err(|e| ProofError::SerializationError(e.to_string()))?;

    // Encrypt with AES-256-GCM
    let cipher = Aes256Gcm::new(aes_key);
    let ciphertext_with_tag = cipher.encrypt(nonce, plaintext.as_slice())
        .map_err(|e| ProofError::SerializationError(format!("Encryption failed: {:?}", e)))?;

    // Split ciphertext and auth tag
    let tag_start = ciphertext_with_tag.len() - 16;
    let ciphertext = ciphertext_with_tag[..tag_start].to_vec();
    let mut auth_tag = [0u8; 16];
    auth_tag.copy_from_slice(&ciphertext_with_tag[tag_start..]);

    Ok(EncryptedProofEnvelope {
        kem_ciphertext: kem_ciphertext.as_bytes().to_vec(),
        nonce: nonce_bytes,
        ciphertext,
        auth_tag,
    })
}

/// Decrypt an encrypted proof envelope using Kyber KEM
#[cfg(feature = "proofs-kyber")]
pub fn decrypt_envelope(
    encrypted: &EncryptedProofEnvelope,
    secret_key: &kyber1024::SecretKey,
) -> ProofResult<ProofEnvelope> {
    use aes_gcm::{Aes256Gcm, KeyInit, aead::Aead};
    use aes_gcm::aead::generic_array::GenericArray;

    // Decapsulate shared secret
    let kem_ciphertext = kyber1024::Ciphertext::from_bytes(&encrypted.kem_ciphertext)
        .map_err(|_| ProofError::VerificationFailed("Invalid KEM ciphertext".to_string()))?;

    let shared_secret = kyber1024::decapsulate(&kem_ciphertext, secret_key);

    // Derive AES key
    let shared_bytes = shared_secret.as_bytes();
    let aes_key = GenericArray::from_slice(&shared_bytes[..32]);
    let nonce = GenericArray::from_slice(&encrypted.nonce);

    // Reconstruct ciphertext with tag
    let mut ciphertext_with_tag = encrypted.ciphertext.clone();
    ciphertext_with_tag.extend_from_slice(&encrypted.auth_tag);

    // Decrypt
    let cipher = Aes256Gcm::new(aes_key);
    let plaintext = cipher.decrypt(nonce, ciphertext_with_tag.as_slice())
        .map_err(|_| ProofError::VerificationFailed("Decryption failed - invalid ciphertext or tampered data".to_string()))?;

    // Deserialize envelope
    let envelope: ProofEnvelope = serde_json::from_slice(&plaintext)
        .map_err(|e| ProofError::SerializationError(e.to_string()))?;

    Ok(envelope)
}

/// Kyber security level information
#[derive(Clone, Debug)]
pub struct KyberSecurityInfo {
    /// Kyber variant name
    pub variant: &'static str,
    /// NIST security level (1-5)
    pub nist_level: u8,
    /// Public key size in bytes
    pub public_key_size: usize,
    /// Secret key size in bytes
    pub secret_key_size: usize,
    /// Ciphertext size in bytes
    pub ciphertext_size: usize,
    /// Shared secret size in bytes
    pub shared_secret_size: usize,
}

/// Get Kyber1024 security information
pub fn kyber1024_info() -> KyberSecurityInfo {
    KyberSecurityInfo {
        variant: "Kyber1024",
        nist_level: 5,
        public_key_size: 1568,
        secret_key_size: 3168,
        ciphertext_size: 1568,
        shared_secret_size: 32,
    }
}

#[cfg(all(test, feature = "proofs-kyber"))]
mod tests {
    use super::*;
    use crate::proofs::{RangeProof, ProofConfig, SecurityLevel};

    fn test_config() -> ProofConfig {
        ProofConfig {
            security_level: SecurityLevel::Standard96,
            parallel: false,
            max_proof_size: 0,
        }
    }

    fn make_envelope() -> ProofEnvelope {
        let proof = RangeProof::generate(50, 0, 100, test_config()).unwrap();
        ProofEnvelope::from_range_proof(&proof).unwrap()
    }

    #[test]
    fn test_kyber_keypair_generation() {
        let keypair = KyberKeyPair::generate();

        // Kyber1024 key sizes
        assert_eq!(keypair.public_key_bytes().len(), 1568);
        assert_eq!(keypair.secret_key_bytes().len(), 3168);
    }

    #[test]
    fn test_encrypt_decrypt_envelope() {
        let keypair = KyberKeyPair::generate();
        let envelope = make_envelope();

        // Encrypt
        let encrypted = encrypt_envelope(&envelope, &keypair.public_key).unwrap();

        // Decrypt
        let decrypted = decrypt_envelope(&encrypted, &keypair.secret_key).unwrap();

        // Verify
        assert_eq!(decrypted.proof_type, envelope.proof_type);
        assert_eq!(decrypted.checksum, envelope.checksum);
        assert_eq!(decrypted.proof_bytes, envelope.proof_bytes);
    }

    #[test]
    fn test_tampered_ciphertext_fails() {
        let keypair = KyberKeyPair::generate();
        let envelope = make_envelope();

        let mut encrypted = encrypt_envelope(&envelope, &keypair.public_key).unwrap();

        // Tamper with ciphertext
        if !encrypted.ciphertext.is_empty() {
            encrypted.ciphertext[0] ^= 0xFF;
        }

        // Should fail to decrypt
        let result = decrypt_envelope(&encrypted, &keypair.secret_key);
        assert!(result.is_err());
    }

    #[test]
    fn test_wrong_key_fails() {
        let keypair1 = KyberKeyPair::generate();
        let keypair2 = KyberKeyPair::generate();
        let envelope = make_envelope();

        // Encrypt with keypair1
        let encrypted = encrypt_envelope(&envelope, &keypair1.public_key).unwrap();

        // Try to decrypt with keypair2
        let result = decrypt_envelope(&encrypted, &keypair2.secret_key);
        assert!(result.is_err());
    }

    #[test]
    fn test_security_info() {
        let info = kyber1024_info();
        assert_eq!(info.nist_level, 5);
        assert_eq!(info.public_key_size, 1568);
    }
}
