//! Encrypted deal support for post-quantum secure share distribution
//!
//! Shares are encrypted per-recipient using a callback-based interface,
//! allowing integration with any KEM (ML-KEM-768, ECIES, etc.) without
//! coupling the DKG library to a specific crypto backend.

use serde::{Deserialize, Serialize};

use crate::commitment::CommitmentSet;
use crate::dealer::Deal;
use crate::error::{DkgError, DkgResult};
use crate::participant::ParticipantId;
use crate::share::Share;

/// An encrypted share payload for a single recipient
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncryptedSharePayload {
    /// The recipient's participant index
    pub recipient: u32,
    /// KEM encapsulated key (ciphertext from encapsulate())
    pub encapsulated_key: Vec<u8>,
    /// Nonce used for AEAD encryption
    pub nonce: Vec<u8>,
    /// AEAD-encrypted share value (the scalar bytes)
    pub ciphertext: Vec<u8>,
}

/// An encrypted deal: commitments are public, shares are encrypted per-recipient
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncryptedDeal {
    /// The dealer's ID
    pub dealer: ParticipantId,
    /// Commitments to the polynomial coefficients (public)
    pub commitments: CommitmentSet,
    /// Encrypted shares, one per recipient
    pub encrypted_shares: Vec<EncryptedSharePayload>,
}

/// Result of encrypting a share: the three components needed for the envelope
pub struct EncryptResult {
    /// KEM encapsulated key
    pub encapsulated_key: Vec<u8>,
    /// AEAD nonce
    pub nonce: Vec<u8>,
    /// AEAD ciphertext
    pub ciphertext: Vec<u8>,
}

impl EncryptedDeal {
    /// Create an encrypted deal from a plaintext deal using a per-recipient encryption callback.
    ///
    /// The `encrypt_fn` is called once for each share with:
    /// - `recipient`: the participant index
    /// - `plaintext`: the serialized share value (32 bytes, big-endian scalar)
    ///
    /// It must return an `EncryptResult` containing the KEM ciphertext, nonce, and AEAD ciphertext.
    /// In production, this would call ML-KEM-768 encapsulate + XChaCha20-Poly1305 encrypt.
    pub fn from_deal<F>(deal: &Deal, mut encrypt_fn: F) -> DkgResult<Self>
    where
        F: FnMut(u32, &[u8]) -> Result<EncryptResult, String>,
    {
        let mut encrypted_shares = Vec::with_capacity(deal.shares.len());

        for share in &deal.shares {
            let plaintext = share.value.to_bytes();
            let result = encrypt_fn(share.index, &plaintext)
                .map_err(|e| DkgError::EncryptionError(e))?;

            encrypted_shares.push(EncryptedSharePayload {
                recipient: share.index,
                encapsulated_key: result.encapsulated_key,
                nonce: result.nonce,
                ciphertext: result.ciphertext,
            });
        }

        Ok(Self {
            dealer: deal.dealer,
            commitments: deal.commitments.clone(),
            encrypted_shares,
        })
    }

    /// Decrypt a single share from this encrypted deal using a decryption callback.
    ///
    /// The `decrypt_fn` is called with:
    /// - `encapsulated_key`: the KEM ciphertext
    /// - `nonce`: the AEAD nonce
    /// - `ciphertext`: the AEAD ciphertext
    ///
    /// It must return the decrypted plaintext (32-byte scalar).
    pub fn decrypt_share<F>(
        &self,
        recipient: u32,
        mut decrypt_fn: F,
    ) -> DkgResult<Share>
    where
        F: FnMut(&[u8], &[u8], &[u8]) -> Result<Vec<u8>, String>,
    {
        let payload = self
            .encrypted_shares
            .iter()
            .find(|p| p.recipient == recipient)
            .ok_or(DkgError::ParticipantNotFound(recipient))?;

        let plaintext = decrypt_fn(
            &payload.encapsulated_key,
            &payload.nonce,
            &payload.ciphertext,
        )
        .map_err(|e| DkgError::DecryptionError(e))?;

        let arr: [u8; 32] = plaintext.try_into().map_err(|_| {
            DkgError::DecryptionError(format!("Expected 32 bytes, got different length"))
        })?;
        let value = crate::scalar::Scalar::from_bytes(&arr)?;

        // Verify the decrypted share against commitments
        if !self.commitments.verify_share(recipient, &value) {
            return Err(DkgError::ShareVerificationFailed {
                participant: recipient,
                dealer: self.dealer.0,
            });
        }

        Ok(Share::new(recipient, self.dealer.0, value))
    }

    /// Get the number of encrypted shares
    pub fn share_count(&self) -> usize {
        self.encrypted_shares.len()
    }

    /// Check if a share exists for the given recipient
    pub fn has_share_for(&self, recipient: u32) -> bool {
        self.encrypted_shares.iter().any(|p| p.recipient == recipient)
    }

    /// Convert back to a plaintext Deal using a decryption callback.
    /// Useful for the ceremony coordinator to verify all shares.
    pub fn to_deal<F>(&self, mut decrypt_fn: F) -> DkgResult<Deal>
    where
        F: FnMut(&[u8], &[u8], &[u8]) -> Result<Vec<u8>, String>,
    {
        let mut shares = Vec::with_capacity(self.encrypted_shares.len());

        for payload in &self.encrypted_shares {
            let plaintext = decrypt_fn(
                &payload.encapsulated_key,
                &payload.nonce,
                &payload.ciphertext,
            )
            .map_err(|e| DkgError::DecryptionError(e))?;

            let arr: [u8; 32] = plaintext.try_into().map_err(|_| {
            DkgError::DecryptionError(format!("Expected 32 bytes, got different length"))
        })?;
        let value = crate::scalar::Scalar::from_bytes(&arr)?;
            shares.push(Share::new(payload.recipient, self.dealer.0, value));
        }

        Ok(Deal {
            dealer: self.dealer,
            commitments: self.commitments.clone(),
            shares,
        })
    }

    /// Serialize to bytes for transport
    pub fn to_bytes(&self) -> DkgResult<Vec<u8>> {
        serde_json::to_vec(self)
            .map_err(|e| DkgError::SerializationError(e.to_string()))
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> DkgResult<Self> {
        serde_json::from_slice(bytes)
            .map_err(|e| DkgError::SerializationError(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dealer::Dealer;
    use rand::rngs::OsRng;

    /// Simple XOR "encryption" for testing (NOT secure — test only)
    fn test_encrypt(recipient: u32, plaintext: &[u8]) -> Result<EncryptResult, String> {
        let key = vec![recipient as u8; 32];
        let nonce = vec![0u8; 24];
        let ciphertext: Vec<u8> = plaintext.iter().zip(key.iter().cycle()).map(|(a, b)| a ^ b).collect();
        Ok(EncryptResult {
            encapsulated_key: key,
            nonce,
            ciphertext,
        })
    }

    /// Corresponding test decryption
    fn test_decrypt(encapsulated_key: &[u8], _nonce: &[u8], ciphertext: &[u8]) -> Result<Vec<u8>, String> {
        let plaintext: Vec<u8> = ciphertext
            .iter()
            .zip(encapsulated_key.iter().cycle())
            .map(|(a, b)| a ^ b)
            .collect();
        Ok(plaintext)
    }

    #[test]
    fn test_encrypted_deal_roundtrip() {
        let dealer = Dealer::new(ParticipantId(1), 2, 3, &mut OsRng).unwrap();
        let deal = dealer.generate_deal();

        // Encrypt
        let encrypted = EncryptedDeal::from_deal(&deal, test_encrypt).unwrap();
        assert_eq!(encrypted.share_count(), 3);
        assert!(encrypted.has_share_for(1));
        assert!(encrypted.has_share_for(2));
        assert!(encrypted.has_share_for(3));

        // Decrypt each share and verify
        for i in 1..=3u32 {
            let share = encrypted.decrypt_share(i, test_decrypt).unwrap();
            assert_eq!(share.index, i);
            assert_eq!(share.dealer, 1);
            // Verify against original
            let original = deal.get_share(i).unwrap();
            assert_eq!(share.value, original.value);
        }
    }

    #[test]
    fn test_encrypted_deal_to_deal() {
        let dealer = Dealer::new(ParticipantId(1), 2, 3, &mut OsRng).unwrap();
        let deal = dealer.generate_deal();

        let encrypted = EncryptedDeal::from_deal(&deal, test_encrypt).unwrap();
        let recovered = encrypted.to_deal(test_decrypt).unwrap();

        assert_eq!(recovered.dealer, deal.dealer);
        assert_eq!(recovered.share_count(), deal.share_count());

        for i in 1..=3u32 {
            let orig = deal.get_share(i).unwrap();
            let recov = recovered.get_share(i).unwrap();
            assert_eq!(orig.value, recov.value);
        }
    }

    #[test]
    fn test_encrypted_deal_wrong_recipient() {
        let dealer = Dealer::new(ParticipantId(1), 2, 3, &mut OsRng).unwrap();
        let deal = dealer.generate_deal();

        let encrypted = EncryptedDeal::from_deal(&deal, test_encrypt).unwrap();

        // Non-existent recipient
        let result = encrypted.decrypt_share(99, test_decrypt);
        assert!(matches!(result, Err(DkgError::ParticipantNotFound(99))));
    }

    #[test]
    fn test_encrypted_deal_tampered_ciphertext() {
        let dealer = Dealer::new(ParticipantId(1), 2, 3, &mut OsRng).unwrap();
        let deal = dealer.generate_deal();

        let mut encrypted = EncryptedDeal::from_deal(&deal, test_encrypt).unwrap();

        // Tamper with ciphertext of share for participant 1
        if let Some(payload) = encrypted.encrypted_shares.iter_mut().find(|p| p.recipient == 1) {
            payload.ciphertext[0] ^= 0xFF;
        }

        // Decryption should succeed but share verification should fail
        let result = encrypted.decrypt_share(1, test_decrypt);
        assert!(matches!(result, Err(DkgError::ShareVerificationFailed { .. })));
    }

    #[test]
    fn test_encrypted_deal_serialization() {
        let dealer = Dealer::new(ParticipantId(1), 2, 3, &mut OsRng).unwrap();
        let deal = dealer.generate_deal();

        let encrypted = EncryptedDeal::from_deal(&deal, test_encrypt).unwrap();

        let bytes = encrypted.to_bytes().unwrap();
        let recovered = EncryptedDeal::from_bytes(&bytes).unwrap();

        assert_eq!(recovered.dealer, encrypted.dealer);
        assert_eq!(recovered.share_count(), encrypted.share_count());
    }

    #[test]
    fn test_encrypt_callback_error_propagates() {
        let dealer = Dealer::new(ParticipantId(1), 2, 3, &mut OsRng).unwrap();
        let deal = dealer.generate_deal();

        let result = EncryptedDeal::from_deal(&deal, |_, _| {
            Err("KEM failure".to_string())
        });
        assert!(matches!(result, Err(DkgError::EncryptionError(_))));
    }

    #[test]
    fn test_decrypt_callback_error_propagates() {
        let dealer = Dealer::new(ParticipantId(1), 2, 3, &mut OsRng).unwrap();
        let deal = dealer.generate_deal();

        let encrypted = EncryptedDeal::from_deal(&deal, test_encrypt).unwrap();

        let result = encrypted.decrypt_share(1, |_, _, _| {
            Err("Decapsulation failure".to_string())
        });
        assert!(matches!(result, Err(DkgError::DecryptionError(_))));
    }

    #[test]
    fn test_encrypted_deal_preserves_commitments() {
        let dealer = Dealer::new(ParticipantId(1), 3, 5, &mut OsRng).unwrap();
        let deal = dealer.generate_deal();

        let encrypted = EncryptedDeal::from_deal(&deal, test_encrypt).unwrap();

        // Commitments are public and should be identical
        assert_eq!(encrypted.commitments.len(), deal.commitments.len());
        for (orig, enc) in deal.commitments.commitments().iter().zip(encrypted.commitments.commitments().iter()) {
            assert_eq!(orig, enc);
        }
    }

    #[test]
    fn test_encrypted_deal_in_ceremony() {
        // Full DKG ceremony using encrypted deals
        use crate::ceremony::{DkgCeremony, DkgConfig};

        let config = DkgConfig::new(2, 3).unwrap();
        let mut ceremony = DkgCeremony::new(config, 0);

        for i in 1..=3 {
            ceremony.add_participant(ParticipantId(i), 0).unwrap();
        }

        // Each participant creates encrypted deal, then decrypts to submit to ceremony
        for i in 1..=3u32 {
            let dealer = Dealer::new(ParticipantId(i), 2, 3, &mut OsRng).unwrap();
            let deal = dealer.generate_deal();
            let encrypted = EncryptedDeal::from_deal(&deal, test_encrypt).unwrap();

            // In real protocol, only the ceremony coordinator or each recipient
            // would decrypt their own share. Here we decrypt all for testing.
            let recovered_deal = encrypted.to_deal(test_decrypt).unwrap();
            ceremony.submit_deal(ParticipantId(i), recovered_deal, 0).unwrap();
        }

        let result = ceremony.finalize().unwrap();
        assert_eq!(result.qualified_count, 3);
    }
}
