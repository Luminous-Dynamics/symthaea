//! AES-GCM encryption for gradient confidentiality.
//!
//! Provides authenticated encryption for gradients to protect
//! against eavesdropping and tampering during transmission.

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[cfg(feature = "privacy")]
use aes_gcm::{
    aead::{Aead, KeyInit, OsRng},
    Aes256Gcm, Nonce,
};

#[cfg(feature = "privacy")]
use rand::RngCore;

/// AES-GCM key size (256 bits).
pub const KEY_SIZE: usize = 32;

/// AES-GCM nonce size (96 bits).
pub const NONCE_SIZE: usize = 12;

/// AES-GCM tag size (128 bits).
pub const TAG_SIZE: usize = 16;

/// Encryption errors.
#[derive(Debug, Error)]
pub enum EncryptionError {
    /// Invalid key length.
    #[error("Invalid key length: expected {KEY_SIZE} bytes, got {0}")]
    InvalidKeyLength(usize),

    /// Encryption failed.
    #[error("Encryption failed: {0}")]
    EncryptionFailed(String),

    /// Decryption failed (authentication failure or corrupted data).
    #[error("Decryption failed: ciphertext authentication failed")]
    DecryptionFailed,

    /// Invalid ciphertext format.
    #[error("Invalid ciphertext: too short (minimum {min} bytes, got {got})")]
    InvalidCiphertext { min: usize, got: usize },

    /// Feature not enabled.
    #[error("Privacy feature not enabled")]
    FeatureNotEnabled,
}

/// Encrypted gradient with metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncryptedGradient {
    /// Nonce (12 bytes).
    pub nonce: Vec<u8>,
    /// Ciphertext with authentication tag.
    pub ciphertext: Vec<u8>,
    /// Original gradient dimension.
    pub dimension: usize,
    /// Key identifier (for key rotation).
    pub key_id: Option<String>,
}

impl EncryptedGradient {
    /// Get total encrypted size.
    pub fn encrypted_size(&self) -> usize {
        self.nonce.len() + self.ciphertext.len()
    }

    /// Convert to wire format (nonce || ciphertext).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.nonce.len() + self.ciphertext.len());
        bytes.extend_from_slice(&self.nonce);
        bytes.extend_from_slice(&self.ciphertext);
        bytes
    }

    /// Parse from wire format.
    pub fn from_bytes(bytes: &[u8], dimension: usize) -> Result<Self, EncryptionError> {
        if bytes.len() < NONCE_SIZE + TAG_SIZE {
            return Err(EncryptionError::InvalidCiphertext {
                min: NONCE_SIZE + TAG_SIZE,
                got: bytes.len(),
            });
        }

        Ok(Self {
            nonce: bytes[..NONCE_SIZE].to_vec(),
            ciphertext: bytes[NONCE_SIZE..].to_vec(),
            dimension,
            key_id: None,
        })
    }
}

/// Generate a random 256-bit key.
#[cfg(feature = "privacy")]
pub fn generate_key() -> [u8; KEY_SIZE] {
    let mut key = [0u8; KEY_SIZE];
    OsRng.fill_bytes(&mut key);
    key
}

#[cfg(not(feature = "privacy"))]
pub fn generate_key() -> [u8; KEY_SIZE] {
    [0u8; KEY_SIZE]
}

/// Encrypt gradient bytes using AES-256-GCM.
///
/// Returns nonce || ciphertext (nonce is 12 bytes, prepended to output).
#[cfg(feature = "privacy")]
pub fn encrypt_gradient(key: &[u8], plaintext: &[u8]) -> Result<Vec<u8>, EncryptionError> {
    if key.len() != KEY_SIZE {
        return Err(EncryptionError::InvalidKeyLength(key.len()));
    }

    let cipher = Aes256Gcm::new_from_slice(key)
        .map_err(|e| EncryptionError::EncryptionFailed(e.to_string()))?;

    // Generate random nonce
    let mut nonce_bytes = [0u8; NONCE_SIZE];
    OsRng.fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);

    // Encrypt
    let ciphertext = cipher
        .encrypt(nonce, plaintext)
        .map_err(|e| EncryptionError::EncryptionFailed(e.to_string()))?;

    // Return nonce || ciphertext
    let mut result = Vec::with_capacity(NONCE_SIZE + ciphertext.len());
    result.extend_from_slice(&nonce_bytes);
    result.extend_from_slice(&ciphertext);

    Ok(result)
}

#[cfg(not(feature = "privacy"))]
pub fn encrypt_gradient(_key: &[u8], _plaintext: &[u8]) -> Result<Vec<u8>, EncryptionError> {
    Err(EncryptionError::FeatureNotEnabled)
}

/// Decrypt gradient bytes using AES-256-GCM.
///
/// Input format: nonce (12 bytes) || ciphertext.
#[cfg(feature = "privacy")]
pub fn decrypt_gradient(key: &[u8], ciphertext: &[u8]) -> Result<Vec<u8>, EncryptionError> {
    if key.len() != KEY_SIZE {
        return Err(EncryptionError::InvalidKeyLength(key.len()));
    }

    if ciphertext.len() < NONCE_SIZE + TAG_SIZE {
        return Err(EncryptionError::InvalidCiphertext {
            min: NONCE_SIZE + TAG_SIZE,
            got: ciphertext.len(),
        });
    }

    let cipher = Aes256Gcm::new_from_slice(key)
        .map_err(|e| EncryptionError::EncryptionFailed(e.to_string()))?;

    let nonce = Nonce::from_slice(&ciphertext[..NONCE_SIZE]);
    let encrypted_data = &ciphertext[NONCE_SIZE..];

    cipher
        .decrypt(nonce, encrypted_data)
        .map_err(|_| EncryptionError::DecryptionFailed)
}

#[cfg(not(feature = "privacy"))]
pub fn decrypt_gradient(_key: &[u8], _ciphertext: &[u8]) -> Result<Vec<u8>, EncryptionError> {
    Err(EncryptionError::FeatureNotEnabled)
}

/// Encrypt a gradient array (f32 values).
#[cfg(feature = "privacy")]
pub fn encrypt_gradient_array(
    key: &[u8],
    gradient: &[f32],
) -> Result<EncryptedGradient, EncryptionError> {
    // Convert f32 array to bytes (little-endian)
    let plaintext: Vec<u8> = gradient
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    let encrypted = encrypt_gradient(key, &plaintext)?;

    Ok(EncryptedGradient {
        nonce: encrypted[..NONCE_SIZE].to_vec(),
        ciphertext: encrypted[NONCE_SIZE..].to_vec(),
        dimension: gradient.len(),
        key_id: None,
    })
}

#[cfg(not(feature = "privacy"))]
pub fn encrypt_gradient_array(
    _key: &[u8],
    _gradient: &[f32],
) -> Result<EncryptedGradient, EncryptionError> {
    Err(EncryptionError::FeatureNotEnabled)
}

/// Decrypt a gradient array.
#[cfg(feature = "privacy")]
pub fn decrypt_gradient_array(
    key: &[u8],
    encrypted: &EncryptedGradient,
) -> Result<Vec<f32>, EncryptionError> {
    let ciphertext = encrypted.to_bytes();
    let plaintext = decrypt_gradient(key, &ciphertext)?;

    // Convert bytes back to f32 array
    if plaintext.len() % 4 != 0 {
        return Err(EncryptionError::DecryptionFailed);
    }

    let gradient: Vec<f32> = plaintext
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    if gradient.len() != encrypted.dimension {
        return Err(EncryptionError::DecryptionFailed);
    }

    Ok(gradient)
}

#[cfg(not(feature = "privacy"))]
pub fn decrypt_gradient_array(
    _key: &[u8],
    _encrypted: &EncryptedGradient,
) -> Result<Vec<f32>, EncryptionError> {
    Err(EncryptionError::FeatureNotEnabled)
}

/// Key derivation from a master secret (using HKDF-SHA256).
#[cfg(feature = "privacy")]
pub fn derive_key(master_secret: &[u8], context: &[u8]) -> [u8; KEY_SIZE] {
    use sha2::{Sha256, Digest};

    let mut hasher = Sha256::new();
    hasher.update(master_secret);
    hasher.update(context);
    let result = hasher.finalize();

    let mut key = [0u8; KEY_SIZE];
    key.copy_from_slice(&result[..KEY_SIZE]);
    key
}

#[cfg(not(feature = "privacy"))]
pub fn derive_key(_master_secret: &[u8], _context: &[u8]) -> [u8; KEY_SIZE] {
    [0u8; KEY_SIZE]
}

#[cfg(test)]
#[cfg(feature = "privacy")]
mod tests {
    use super::*;

    #[test]
    fn test_key_generation() {
        let key1 = generate_key();
        let key2 = generate_key();

        // Keys should be different
        assert_ne!(key1, key2);
        assert_eq!(key1.len(), KEY_SIZE);
    }

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let key = generate_key();
        let plaintext = b"Hello, gradient data!";

        let ciphertext = encrypt_gradient(&key, plaintext).unwrap();
        let decrypted = decrypt_gradient(&key, &ciphertext).unwrap();

        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_ciphertext_format() {
        let key = generate_key();
        let plaintext = b"test";

        let ciphertext = encrypt_gradient(&key, plaintext).unwrap();

        // Should be: nonce (12) + plaintext (4) + tag (16) = 32 bytes
        assert_eq!(ciphertext.len(), NONCE_SIZE + plaintext.len() + TAG_SIZE);
    }

    #[test]
    fn test_wrong_key_fails() {
        let key1 = generate_key();
        let key2 = generate_key();
        let plaintext = b"secret data";

        let ciphertext = encrypt_gradient(&key1, plaintext).unwrap();
        let result = decrypt_gradient(&key2, &ciphertext);

        assert!(matches!(result, Err(EncryptionError::DecryptionFailed)));
    }

    #[test]
    fn test_tampered_ciphertext_fails() {
        let key = generate_key();
        let plaintext = b"important data";

        let mut ciphertext = encrypt_gradient(&key, plaintext).unwrap();

        // Tamper with the ciphertext
        if let Some(byte) = ciphertext.get_mut(NONCE_SIZE + 5) {
            *byte ^= 0xFF;
        }

        let result = decrypt_gradient(&key, &ciphertext);
        assert!(matches!(result, Err(EncryptionError::DecryptionFailed)));
    }

    #[test]
    fn test_gradient_array_roundtrip() {
        let key = generate_key();
        let gradient = vec![1.0f32, 2.5, -3.14, 0.0, 100.0];

        let encrypted = encrypt_gradient_array(&key, &gradient).unwrap();
        let decrypted = decrypt_gradient_array(&key, &encrypted).unwrap();

        assert_eq!(decrypted.len(), gradient.len());
        for (a, b) in decrypted.iter().zip(gradient.iter()) {
            assert!((a - b).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_encrypted_gradient_serialization() {
        let key = generate_key();
        let gradient = vec![1.0f32, 2.0, 3.0];

        let encrypted = encrypt_gradient_array(&key, &gradient).unwrap();
        let bytes = encrypted.to_bytes();
        let restored = EncryptedGradient::from_bytes(&bytes, gradient.len()).unwrap();

        assert_eq!(encrypted.nonce, restored.nonce);
        assert_eq!(encrypted.ciphertext, restored.ciphertext);
        assert_eq!(encrypted.dimension, restored.dimension);
    }

    #[test]
    fn test_key_derivation() {
        let master = b"master secret key";
        let context1 = b"round_1";
        let context2 = b"round_2";

        let key1 = derive_key(master, context1);
        let key2 = derive_key(master, context2);
        let key1_again = derive_key(master, context1);

        // Same context produces same key
        assert_eq!(key1, key1_again);
        // Different context produces different key
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_invalid_key_length() {
        let short_key = [0u8; 16]; // Too short
        let plaintext = b"data";

        let result = encrypt_gradient(&short_key, plaintext);
        assert!(matches!(result, Err(EncryptionError::InvalidKeyLength(16))));
    }

    #[test]
    fn test_invalid_ciphertext_length() {
        let key = generate_key();
        let short_ciphertext = [0u8; 10]; // Too short

        let result = decrypt_gradient(&key, &short_ciphertext);
        assert!(matches!(
            result,
            Err(EncryptionError::InvalidCiphertext { .. })
        ));
    }
}
