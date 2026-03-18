//! SLH-DSA (SPHINCS+) signer/verifier using pqcrypto-sphincsplus.
//!
//! Hash-based stateless signatures — no lattice assumptions, purely hash-based security.
//! Larger signatures (~7.8KB) but maximally conservative post-quantum assumption.

use crate::algorithm::AlgorithmId;
use crate::envelope::{TaggedPublicKey, TaggedSignature};
use crate::error::CryptoError;
use crate::traits::{Signer, Verifier};

use pqcrypto_sphincsplus::sphincssha2128ssimple as sphincs_sha2;
use pqcrypto_traits::sign::{DetachedSignature, PublicKey, SecretKey};
use zeroize::Zeroizing;

/// SLH-DSA-SHA2-128s signer (SPHINCS+-SHA2-128s-simple).
pub struct SlhDsaSha2128sSigner {
    public_key: sphincs_sha2::PublicKey,
    secret_key: sphincs_sha2::SecretKey,
}

impl SlhDsaSha2128sSigner {
    /// Generate a new keypair.
    pub fn generate() -> Self {
        let (pk, sk) = sphincs_sha2::keypair();
        Self {
            public_key: pk,
            secret_key: sk,
        }
    }

    /// Create from raw key bytes.
    pub fn from_bytes(
        public_key_bytes: &[u8],
        secret_key_bytes: &[u8],
    ) -> Result<Self, CryptoError> {
        let pk = sphincs_sha2::PublicKey::from_bytes(public_key_bytes)
            .map_err(|_| CryptoError::Validation("Invalid SLH-DSA-SHA2-128s public key".into()))?;
        let sk = sphincs_sha2::SecretKey::from_bytes(secret_key_bytes)
            .map_err(|_| CryptoError::Validation("Invalid SLH-DSA-SHA2-128s secret key".into()))?;
        Ok(Self {
            public_key: pk,
            secret_key: sk,
        })
    }

    /// Raw secret key bytes. Zeroized on drop.
    pub fn secret_key_bytes(&self) -> Zeroizing<Vec<u8>> {
        Zeroizing::new(self.secret_key.as_bytes().to_vec())
    }

    /// Raw public key bytes.
    pub fn public_key_bytes(&self) -> Vec<u8> {
        self.public_key.as_bytes().to_vec()
    }
}

impl Signer for SlhDsaSha2128sSigner {
    fn algorithm(&self) -> AlgorithmId {
        AlgorithmId::SlhDsaSha2_128s
    }

    fn sign(&self, message: &[u8]) -> Result<TaggedSignature, CryptoError> {
        let sig = sphincs_sha2::detached_sign(message, &self.secret_key);
        TaggedSignature::new(AlgorithmId::SlhDsaSha2_128s, sig.as_bytes().to_vec())
    }

    fn public_key(&self) -> TaggedPublicKey {
        TaggedPublicKey::new(
            AlgorithmId::SlhDsaSha2_128s,
            self.public_key.as_bytes().to_vec(),
        )
        .expect("SlhDsaSha2128s public key size is constant")
    }
}

/// SLH-DSA-SHA2-128s verifier.
pub struct SlhDsaSha2128sVerifier;

impl Verifier for SlhDsaSha2128sVerifier {
    fn algorithm(&self) -> AlgorithmId {
        AlgorithmId::SlhDsaSha2_128s
    }

    fn verify(
        &self,
        public_key: &TaggedPublicKey,
        message: &[u8],
        signature: &TaggedSignature,
    ) -> Result<bool, CryptoError> {
        if public_key.algorithm != AlgorithmId::SlhDsaSha2_128s {
            return Err(CryptoError::Validation(format!(
                "Expected SLH-DSA-SHA2-128s key, got {:?}",
                public_key.algorithm
            )));
        }

        let pk = sphincs_sha2::PublicKey::from_bytes(&public_key.key_bytes)
            .map_err(|_| CryptoError::Validation("Invalid SPHINCS+ public key".into()))?;

        let sig_bytes = signature
            .pqc_component()
            .ok_or_else(|| CryptoError::Validation("No PQC signature component".into()))?;

        let sig = sphincs_sha2::DetachedSignature::from_bytes(sig_bytes)
            .map_err(|_| CryptoError::Validation("Invalid SPHINCS+ signature".into()))?;

        Ok(sphincs_sha2::verify_detached_signature(&sig, message, &pk).is_ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphincs_sign_verify() {
        let signer = SlhDsaSha2128sSigner::generate();
        let msg = b"hash-based quantum resistance";
        let sig = signer.sign(msg).unwrap();
        let pk = signer.public_key();

        let verifier = SlhDsaSha2128sVerifier;
        assert!(verifier.verify(&pk, msg, &sig).unwrap());
    }

    #[test]
    fn sphincs_wrong_message() {
        let signer = SlhDsaSha2128sSigner::generate();
        let sig = signer.sign(b"correct").unwrap();
        let pk = signer.public_key();

        let verifier = SlhDsaSha2128sVerifier;
        assert!(!verifier.verify(&pk, b"wrong", &sig).unwrap());
    }
}
