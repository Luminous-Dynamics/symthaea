//! ML-KEM (Kyber) key encapsulation using pqcrypto-kyber.
//!
//! Used for post-quantum key agreement: the sender encapsulates a shared secret
//! to the recipient's public key, which the recipient decapsulates with their
//! private key. The shared secret is then used as input to an AEAD cipher.

use crate::algorithm::AlgorithmId;
use crate::envelope::TaggedPublicKey;
use crate::error::CryptoError;
use crate::traits::KeyEncapsulator;

use pqcrypto_kyber::kyber768;
use pqcrypto_traits::kem::{Ciphertext, PublicKey, SharedSecret};

/// ML-KEM-768 key pair for key encapsulation.
pub struct MlKem768KeyPair {
    public_key: kyber768::PublicKey,
    secret_key: kyber768::SecretKey,
}

impl MlKem768KeyPair {
    /// Generate a new ML-KEM-768 keypair.
    pub fn generate() -> Self {
        let (pk, sk) = kyber768::keypair();
        Self {
            public_key: pk,
            secret_key: sk,
        }
    }

    /// Get the public key as a TaggedPublicKey.
    pub fn public_key(&self) -> TaggedPublicKey {
        TaggedPublicKey::new(AlgorithmId::MlKem768, self.public_key.as_bytes().to_vec())
            .expect("MlKem768 public key size is constant per FIPS 203")
    }
}

impl KeyEncapsulator for MlKem768KeyPair {
    fn algorithm(&self) -> AlgorithmId {
        AlgorithmId::MlKem768
    }

    fn encapsulate(&self, public_key: &TaggedPublicKey) -> Result<(Vec<u8>, Vec<u8>), CryptoError> {
        if public_key.algorithm != AlgorithmId::MlKem768 {
            return Err(CryptoError::Validation(format!(
                "Expected ML-KEM-768 key, got {:?}",
                public_key.algorithm
            )));
        }

        let pk = kyber768::PublicKey::from_bytes(&public_key.key_bytes)
            .map_err(|_| CryptoError::Validation("Invalid ML-KEM-768 public key".into()))?;

        let (shared_secret, ciphertext) = kyber768::encapsulate(&pk);

        Ok((
            ciphertext.as_bytes().to_vec(),
            shared_secret.as_bytes().to_vec(),
        ))
    }

    fn decapsulate(&self, ciphertext: &[u8]) -> Result<Vec<u8>, CryptoError> {
        let ct = kyber768::Ciphertext::from_bytes(ciphertext)
            .map_err(|_| CryptoError::Validation("Invalid ML-KEM-768 ciphertext".into()))?;

        let shared_secret = kyber768::decapsulate(&ct, &self.secret_key);
        Ok(shared_secret.as_bytes().to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kem_encapsulate_decapsulate() {
        let recipient = MlKem768KeyPair::generate();
        let sender = MlKem768KeyPair::generate();

        let recipient_pk = recipient.public_key();
        let (ciphertext, sender_shared_secret) = sender.encapsulate(&recipient_pk).unwrap();
        let recipient_shared_secret = recipient.decapsulate(&ciphertext).unwrap();

        assert_eq!(sender_shared_secret, recipient_shared_secret);
        assert!(!sender_shared_secret.is_empty());
    }
}
