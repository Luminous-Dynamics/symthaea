//! Symmetric encryption using XChaCha20-Poly1305 AEAD.
//!
//! Combined with ML-KEM key encapsulation, this provides a complete
//! post-quantum encryption workflow:
//! 1. ML-KEM encapsulate → shared_secret
//! 2. XChaCha20-Poly1305 encrypt with shared_secret → ciphertext
//! 3. Store EncryptedEnvelope on DHT

use crate::algorithm::AlgorithmId;
use crate::envelope::EncryptedEnvelope;
use crate::error::CryptoError;
use crate::traits::Encryptor;

use chacha20poly1305::aead::AeadCore;
use chacha20poly1305::aead::{Aead, KeyInit, OsRng};
use chacha20poly1305::{XChaCha20Poly1305, XNonce};
use zeroize::Zeroizing;

/// XChaCha20-Poly1305 encryptor with a pre-shared key.
pub struct XChaCha20Encryptor {
    /// 32-byte symmetric key (typically derived from KEM shared secret). Zeroized on drop.
    key: Zeroizing<Vec<u8>>,
    /// KEM algorithm used to derive the key (for EncryptedEnvelope metadata)
    kem_algorithm: AlgorithmId,
    /// KEM ciphertext to include in envelope
    encapsulated_key: Vec<u8>,
}

impl XChaCha20Encryptor {
    /// Create from a KEM-derived shared secret.
    ///
    /// The `encapsulated_key` is the KEM ciphertext that the recipient needs
    /// to recover the shared secret. The `shared_secret` must be at least 32 bytes.
    pub fn from_kem(
        shared_secret: &[u8],
        encapsulated_key: Vec<u8>,
        kem_algorithm: AlgorithmId,
    ) -> Result<Self, CryptoError> {
        if shared_secret.len() < 32 {
            return Err(CryptoError::Validation(format!(
                "Shared secret too short: {} bytes, need 32",
                shared_secret.len()
            )));
        }
        Ok(Self {
            key: Zeroizing::new(shared_secret[..32].to_vec()),
            kem_algorithm,
            encapsulated_key,
        })
    }

    /// Create from a raw 32-byte symmetric key (for self-encryption without KEM).
    pub fn from_raw_key(key: &[u8; 32]) -> Self {
        Self {
            key: Zeroizing::new(key.to_vec()),
            kem_algorithm: AlgorithmId::XChaCha20Poly1305,
            encapsulated_key: Vec::new(),
        }
    }
}

impl Encryptor for XChaCha20Encryptor {
    fn encrypt(
        &self,
        plaintext: &[u8],
        _associated_data: &[u8],
        recipient_key_id: &str,
    ) -> Result<EncryptedEnvelope, CryptoError> {
        let key_array: [u8; 32] = self
            .key
            .as_slice()
            .try_into()
            .map_err(|_| CryptoError::Validation("Key must be exactly 32 bytes".into()))?;

        let cipher = XChaCha20Poly1305::new(&key_array.into());
        let nonce = XChaCha20Poly1305::generate_nonce(&mut OsRng);

        let ciphertext = cipher
            .encrypt(&nonce, plaintext)
            .map_err(|e| CryptoError::Validation(format!("Encryption failed: {}", e)))?;

        Ok(EncryptedEnvelope {
            kem_algorithm: self.kem_algorithm,
            encapsulated_key: self.encapsulated_key.clone(),
            nonce: nonce.to_vec(),
            ciphertext,
            recipient_key_id: recipient_key_id.to_string(),
        })
    }

    fn decrypt(
        &self,
        envelope: &EncryptedEnvelope,
        _associated_data: &[u8],
    ) -> Result<Vec<u8>, CryptoError> {
        let key_array: [u8; 32] = self
            .key
            .as_slice()
            .try_into()
            .map_err(|_| CryptoError::Validation("Key must be exactly 32 bytes".into()))?;

        let cipher = XChaCha20Poly1305::new(&key_array.into());

        let nonce_array: [u8; 24] = envelope
            .nonce
            .as_slice()
            .try_into()
            .map_err(|_| CryptoError::Validation("Nonce must be 24 bytes".into()))?;
        let nonce = XNonce::from(nonce_array);

        cipher
            .decrypt(&nonce, envelope.ciphertext.as_slice())
            .map_err(|e| CryptoError::Validation(format!("Decryption failed: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pqc::ml_kem::MlKem768KeyPair;
    use crate::traits::KeyEncapsulator;

    #[test]
    fn encrypt_decrypt_raw_key() {
        let key = [0x42u8; 32];
        let encryptor = XChaCha20Encryptor::from_raw_key(&key);

        let plaintext = b"sensitive MFA state data";
        let envelope = encryptor.encrypt(plaintext, b"", "self").unwrap();

        let decrypted = encryptor.decrypt(&envelope, b"").unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn encrypt_decrypt_with_kem() {
        let recipient = MlKem768KeyPair::generate();
        let sender_kem = MlKem768KeyPair::generate();

        let recipient_pk = recipient.public_key();
        let (encapsulated_key, shared_secret) = sender_kem.encapsulate(&recipient_pk).unwrap();

        // Sender encrypts
        let encryptor = XChaCha20Encryptor::from_kem(
            &shared_secret,
            encapsulated_key.clone(),
            AlgorithmId::MlKem768,
        )
        .unwrap();

        let plaintext = b"encrypted credential claims";
        let envelope = encryptor
            .encrypt(plaintext, b"", "did:mycelix:abc#kem-1")
            .unwrap();

        // Recipient decapsulates and decrypts
        let recipient_shared_secret = recipient.decapsulate(&encapsulated_key).unwrap();
        let decryptor = XChaCha20Encryptor::from_kem(
            &recipient_shared_secret,
            Vec::new(),
            AlgorithmId::MlKem768,
        )
        .unwrap();

        let decrypted = decryptor.decrypt(&envelope, b"").unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn wrong_key_fails() {
        let encryptor = XChaCha20Encryptor::from_raw_key(&[0x42; 32]);
        let envelope = encryptor.encrypt(b"secret", b"", "self").unwrap();

        let wrong_decryptor = XChaCha20Encryptor::from_raw_key(&[0x43; 32]);
        assert!(wrong_decryptor.decrypt(&envelope, b"").is_err());
    }
}
