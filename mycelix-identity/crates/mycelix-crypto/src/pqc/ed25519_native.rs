//! Ed25519 native signer/verifier using ed25519-dalek.

use crate::algorithm::AlgorithmId;
use crate::envelope::{TaggedPublicKey, TaggedSignature};
use crate::error::CryptoError;
use crate::traits::{Signer, Verifier};

use ed25519_dalek::Signer as DalekSigner;
use ed25519_dalek::Verifier as DalekVerifier;
use ed25519_dalek::{Signature, SigningKey, VerifyingKey};
use zeroize::Zeroizing;

/// Ed25519 signer backed by ed25519-dalek.
pub struct Ed25519Signer {
    signing_key: SigningKey,
}

impl Ed25519Signer {
    /// Create from raw 32-byte secret key.
    pub fn from_bytes(secret: &[u8; 32]) -> Self {
        Self {
            signing_key: SigningKey::from_bytes(secret),
        }
    }

    /// Generate a new random keypair.
    pub fn generate() -> Self {
        let mut csprng = rand::rngs::OsRng;
        Self {
            signing_key: SigningKey::generate(&mut csprng),
        }
    }

    /// Raw secret key bytes (32 bytes). Zeroized on drop.
    pub fn secret_key_bytes(&self) -> Zeroizing<Vec<u8>> {
        Zeroizing::new(self.signing_key.to_bytes().to_vec())
    }
}

impl Signer for Ed25519Signer {
    fn algorithm(&self) -> AlgorithmId {
        AlgorithmId::Ed25519
    }

    fn sign(&self, message: &[u8]) -> Result<TaggedSignature, CryptoError> {
        let sig = self.signing_key.sign(message);
        TaggedSignature::new(AlgorithmId::Ed25519, sig.to_bytes().to_vec())
    }

    fn public_key(&self) -> TaggedPublicKey {
        let vk = self.signing_key.verifying_key();
        TaggedPublicKey::new(AlgorithmId::Ed25519, vk.to_bytes().to_vec())
            .expect("Ed25519 public key is always 32 bytes")
    }
}

/// Ed25519 verifier backed by ed25519-dalek.
pub struct Ed25519Verifier;

impl Verifier for Ed25519Verifier {
    fn algorithm(&self) -> AlgorithmId {
        AlgorithmId::Ed25519
    }

    fn verify(
        &self,
        public_key: &TaggedPublicKey,
        message: &[u8],
        signature: &TaggedSignature,
    ) -> Result<bool, CryptoError> {
        if public_key.algorithm != AlgorithmId::Ed25519 {
            return Err(CryptoError::Validation(format!(
                "Expected Ed25519 public key, got {:?}",
                public_key.algorithm
            )));
        }

        let vk_bytes: [u8; 32] = public_key.key_bytes.as_slice().try_into().map_err(|_| {
            CryptoError::InvalidKeyLength {
                algorithm: "Ed25519VerificationKey2020",
                expected: 32,
                actual: public_key.key_bytes.len(),
            }
        })?;

        let vk = VerifyingKey::from_bytes(&vk_bytes)
            .map_err(|e| CryptoError::Validation(format!("Invalid Ed25519 public key: {}", e)))?;

        let sig_bytes: [u8; 64] = signature
            .ed25519_component()
            .ok_or_else(|| CryptoError::Validation("No Ed25519 signature component".into()))?
            .try_into()
            .map_err(|_| CryptoError::InvalidSignatureLength {
                algorithm: "Ed25519VerificationKey2020",
                expected: 64,
                actual: signature.signature_bytes.len(),
            })?;

        let sig = Signature::from_bytes(&sig_bytes);
        Ok(vk.verify(message, &sig).is_ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_and_verify() {
        let signer = Ed25519Signer::generate();
        let msg = b"hello mycelix";
        let sig = signer.sign(msg).unwrap();
        let pk = signer.public_key();

        let verifier = Ed25519Verifier;
        assert!(verifier.verify(&pk, msg, &sig).unwrap());
    }

    #[test]
    fn wrong_message_fails() {
        let signer = Ed25519Signer::generate();
        let sig = signer.sign(b"correct").unwrap();
        let pk = signer.public_key();

        let verifier = Ed25519Verifier;
        assert!(!verifier.verify(&pk, b"wrong", &sig).unwrap());
    }
}
