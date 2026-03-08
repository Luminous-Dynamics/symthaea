//! ML-KEM-768 post-quantum key encapsulation for encrypted DKG deals.
//!
//! Provides ready-to-use encrypt/decrypt callbacks compatible with
//! [`EncryptedDeal::from_deal`] and [`EncryptedDeal::decrypt_share`].
//!
//! # Security
//!
//! This module uses AES-256-GCM authenticated encryption with the KEM shared
//! secret as the symmetric key (derived via SHA-256). The 12-byte nonce is
//! generated from OS randomness and stored alongside the ciphertext. AES-GCM
//! provides both confidentiality and integrity — any tampering with the
//! ciphertext, nonce, or auth tag will be detected during decryption.
//!
//! # Example
//!
//! ```ignore
//! use feldman_dkg::pq_kem::{generate_keypair, ml_kem_encrypt_fn, ml_kem_decrypt_fn};
//! use feldman_dkg::{Dealer, ParticipantId, EncryptedDeal};
//!
//! // Each participant generates a keypair
//! let kp = generate_keypair();
//!
//! // Dealer creates an encrypted deal
//! let dealer = Dealer::new(ParticipantId(1), 2, 3, &mut rng).unwrap();
//! let deal = dealer.generate_deal();
//!
//! // Encrypt using recipient's encapsulation key
//! let ek_bytes = kp.encapsulation_key_bytes();
//! let encrypted = EncryptedDeal::from_deal(&deal, ml_kem_encrypt_fn(&ek_bytes)).unwrap();
//!
//! // Decrypt using recipient's decapsulation key
//! let share = encrypted.decrypt_share(1, ml_kem_decrypt_fn(&kp.dk)).unwrap();
//! ```

use aes_gcm::{Aes256Gcm, Key, Nonce, aead::{Aead, KeyInit}};
use ml_kem::{
    kem::{Decapsulate, KeyExport},
    MlKem768,
    DecapsulationKey768, EncapsulationKey768,
    B32,
};
use sha2::{Digest, Sha256};

use crate::encrypted_deal::EncryptResult;

/// A ML-KEM-768 keypair for post-quantum key encapsulation.
pub struct MlKemKeyPair {
    /// The decapsulation (private) key.
    pub dk: DecapsulationKey768,
    /// The encapsulation (public) key.
    pub ek: EncapsulationKey768,
}

impl MlKemKeyPair {
    /// Serialize the encapsulation (public) key to bytes for distribution.
    pub fn encapsulation_key_bytes(&self) -> Vec<u8> {
        self.ek.to_bytes().to_vec()
    }
}

/// Generate a new ML-KEM-768 keypair using OS randomness.
///
/// # Panics
///
/// Panics if the OS CSPRNG is unavailable (should not happen on supported platforms).
pub fn generate_keypair() -> MlKemKeyPair {
    let mut seed = [0u8; 64];
    getrandom::fill(&mut seed).expect("OS CSPRNG unavailable");
    let dk = DecapsulationKey768::from_seed(ml_kem::Seed::from(seed));
    let ek = dk.encapsulation_key().clone();
    MlKemKeyPair { dk, ek }
}

/// Derive a 32-byte AES-256 key from the KEM shared secret via SHA-256.
#[allow(deprecated)] // generic-array 0.x from_slice; upstream dep
fn derive_aes_key(shared_secret: &[u8]) -> Key<Aes256Gcm> {
    let hash = Sha256::digest(shared_secret);
    *Key::<Aes256Gcm>::from_slice(&hash)
}

/// Create an encryption callback compatible with [`EncryptedDeal::from_deal`].
///
/// The returned closure:
/// 1. Parses `recipient_ek` as an ML-KEM-768 encapsulation key
/// 2. Encapsulates a shared secret using OS randomness
/// 3. Derives a 32-byte AES key via SHA-256 from the shared secret
/// 4. Generates a random 12-byte nonce
/// 5. Encrypts the plaintext with AES-256-GCM (ciphertext includes 16-byte auth tag)
///
/// The `EncryptResult` stores:
/// - `encapsulated_key`: the KEM ciphertext (needed for decapsulation)
/// - `nonce`: 12-byte AES-GCM nonce
/// - `ciphertext`: AES-256-GCM encrypted data with appended 16-byte auth tag
///
/// # Arguments
///
/// * `recipient_ek` - The recipient's ML-KEM-768 encapsulation key bytes
pub fn ml_kem_encrypt_fn(
    recipient_ek: &[u8],
) -> impl FnMut(u32, &[u8]) -> Result<EncryptResult, String> {
    let ek_bytes = recipient_ek.to_vec();
    move |_recipient, plaintext| {
        // Parse encapsulation key
        let ek_array = ml_kem::kem::Key::<EncapsulationKey768>::try_from(ek_bytes.as_slice())
            .map_err(|_| "Invalid encapsulation key length".to_string())?;
        let ek = EncapsulationKey768::new(&ek_array)
            .map_err(|_| "Failed to parse encapsulation key".to_string())?;

        // Generate 32 random bytes for deterministic encapsulation
        let mut m = [0u8; 32];
        getrandom::fill(&mut m).map_err(|e| format!("RNG failure: {e}"))?;
        let m = B32::from(m);

        // Encapsulate: produces (ciphertext, shared_secret)
        let (ct, shared_secret) = ek.encapsulate_deterministic(&m);

        // Derive AES-256 key from shared secret
        let aes_key = derive_aes_key(shared_secret.as_slice());
        let cipher = Aes256Gcm::new(&aes_key);

        // Generate random 12-byte nonce
        let mut nonce_bytes = [0u8; 12];
        getrandom::fill(&mut nonce_bytes).map_err(|e| format!("RNG failure: {e}"))?;
        #[allow(deprecated)] // generic-array 0.x
        let nonce = Nonce::from_slice(&nonce_bytes);

        // Encrypt with AES-256-GCM (ciphertext includes 16-byte auth tag)
        let encrypted = cipher
            .encrypt(nonce, plaintext)
            .map_err(|e| format!("AES-GCM encryption failed: {e}"))?;

        Ok(EncryptResult {
            encapsulated_key: <_ as AsRef<[u8]>>::as_ref(&ct).to_vec(),
            nonce: nonce_bytes.to_vec(),
            ciphertext: encrypted,
        })
    }
}

/// Create a decryption callback compatible with [`EncryptedDeal::decrypt_share`].
///
/// The returned closure:
/// 1. Decapsulates the KEM ciphertext to recover the shared secret
/// 2. Derives the same AES-256 key via SHA-256
/// 3. Decrypts and authenticates the AES-256-GCM ciphertext
///
/// Returns an error if the ciphertext has been tampered with (authentication failure).
///
/// # Arguments
///
/// * `dk` - The recipient's ML-KEM-768 decapsulation (private) key
pub fn ml_kem_decrypt_fn(
    dk: &DecapsulationKey768,
) -> impl FnMut(&[u8], &[u8], &[u8]) -> Result<Vec<u8>, String> {
    let dk = dk.clone();
    move |encapsulated_key, nonce_bytes, ciphertext| {
        // Parse KEM ciphertext
        let ct = ml_kem::Ciphertext::<MlKem768>::try_from(encapsulated_key)
            .map_err(|_| "Invalid KEM ciphertext length".to_string())?;

        // Decapsulate to recover shared secret
        let shared_secret = dk.decapsulate(&ct);

        // Derive AES-256 key from shared secret
        let aes_key = derive_aes_key(shared_secret.as_slice());
        let cipher = Aes256Gcm::new(&aes_key);

        // Parse nonce (must be 12 bytes)
        #[allow(deprecated)] // generic-array 0.x
        let nonce = Nonce::from_slice(nonce_bytes);

        // Decrypt and authenticate with AES-256-GCM
        let plaintext = cipher
            .decrypt(nonce, ciphertext)
            .map_err(|_| "AES-GCM decryption failed: authentication tag mismatch (tampered or wrong key)".to_string())?;

        Ok(plaintext)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dealer::Dealer;
    use crate::encrypted_deal::EncryptedDeal;
    use crate::participant::ParticipantId;
    use rand::rngs::OsRng;

    #[test]
    fn test_keypair_generation() {
        let kp = generate_keypair();
        let ek_bytes = kp.encapsulation_key_bytes();
        // ML-KEM-768 encapsulation key is 1184 bytes
        assert_eq!(ek_bytes.len(), 1184);
    }

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let kp = generate_keypair();
        let ek_bytes = kp.encapsulation_key_bytes();

        let plaintext = b"hello post-quantum world!!!12345"; // 31 bytes
        let mut encrypt = ml_kem_encrypt_fn(&ek_bytes);
        let result = encrypt(1, plaintext).unwrap();

        // Ciphertext should differ from plaintext (and be 16 bytes longer due to auth tag)
        assert_ne!(result.ciphertext, plaintext.to_vec());
        assert_eq!(result.ciphertext.len(), plaintext.len() + 16);
        // KEM ciphertext should be 1088 bytes for ML-KEM-768
        assert_eq!(result.encapsulated_key.len(), 1088);
        // Nonce should be 12 bytes
        assert_eq!(result.nonce.len(), 12);

        let mut decrypt = ml_kem_decrypt_fn(&kp.dk);
        let recovered = decrypt(
            &result.encapsulated_key,
            &result.nonce,
            &result.ciphertext,
        )
        .unwrap();

        assert_eq!(recovered, plaintext.to_vec());
    }

    #[test]
    fn test_wrong_key_produces_auth_error() {
        let kp1 = generate_keypair();
        let kp2 = generate_keypair();
        let ek_bytes = kp1.encapsulation_key_bytes();

        let plaintext = [0xABu8; 32];
        let mut encrypt = ml_kem_encrypt_fn(&ek_bytes);
        let result = encrypt(1, &plaintext).unwrap();

        // Decrypt with the wrong key — AES-GCM should reject (auth tag mismatch)
        let mut decrypt_wrong = ml_kem_decrypt_fn(&kp2.dk);
        let recovered = decrypt_wrong(
            &result.encapsulated_key,
            &result.nonce,
            &result.ciphertext,
        );

        assert!(recovered.is_err());
        assert!(recovered.unwrap_err().contains("authentication tag mismatch"));
    }

    #[test]
    fn test_tampered_ciphertext_rejected() {
        let kp = generate_keypair();
        let ek_bytes = kp.encapsulation_key_bytes();

        let plaintext = b"sensitive DKG share data here!!!"; // 32 bytes
        let mut encrypt = ml_kem_encrypt_fn(&ek_bytes);
        let mut result = encrypt(1, plaintext).unwrap();

        // Tamper with one byte of the ciphertext
        result.ciphertext[0] ^= 0xFF;

        // Decryption should fail due to authentication tag mismatch
        let mut decrypt = ml_kem_decrypt_fn(&kp.dk);
        let recovered = decrypt(
            &result.encapsulated_key,
            &result.nonce,
            &result.ciphertext,
        );

        assert!(recovered.is_err());
        assert!(recovered.unwrap_err().contains("authentication tag mismatch"));
    }

    #[test]
    fn test_encrypted_deal_integration() {
        // Generate per-recipient keypairs (participants 1, 2, 3)
        let keypairs: Vec<MlKemKeyPair> = (0..3).map(|_| generate_keypair()).collect();

        let dealer = Dealer::new(ParticipantId(1), 2, 3, &mut OsRng).unwrap();
        let deal = dealer.generate_deal();

        // In a real protocol, each recipient would have their own keypair.
        // For this test, we encrypt all shares to recipient 1's key (since
        // the encrypt_fn receives the same ek for all recipients).
        // We test per-recipient by encrypting+decrypting for each.
        for (idx, kp) in keypairs.iter().enumerate() {
            let recipient_id = (idx + 1) as u32;
            let ek_bytes = kp.encapsulation_key_bytes();

            // Encrypt the deal using this recipient's encapsulation key
            let encrypted =
                EncryptedDeal::from_deal(&deal, ml_kem_encrypt_fn(&ek_bytes)).unwrap();

            // Decrypt the share for this recipient
            let share = encrypted
                .decrypt_share(recipient_id, ml_kem_decrypt_fn(&kp.dk))
                .unwrap();

            // Verify the decrypted share matches the original
            let original = deal.get_share(recipient_id).unwrap();
            assert_eq!(share.value, original.value);
            assert_eq!(share.index, recipient_id);
        }
    }

    #[test]
    fn test_invalid_encapsulation_key() {
        let bad_ek = vec![0u8; 100]; // wrong size
        let mut encrypt = ml_kem_encrypt_fn(&bad_ek);
        let result = encrypt(1, &[0u8; 32]);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_kem_ciphertext() {
        let kp = generate_keypair();
        let bad_ct = vec![0u8; 100]; // wrong size
        let mut decrypt = ml_kem_decrypt_fn(&kp.dk);
        let result = decrypt(&bad_ct, &[], &[0u8; 32]);
        assert!(result.is_err());
    }
}
