//! Hybrid Ed25519 + ML-DSA-65 signer/verifier.
//!
//! Produces concatenated signatures: `ed25519_sig (64 bytes) || mldsa65_sig (3293 bytes)`.
//! During the quantum transition period, hybrid signatures provide:
//! - Classical security via Ed25519 (verifiable on-chain by Holochain HDK)
//! - Post-quantum security via ML-DSA-65 (verifiable off-chain)

use crate::algorithm::AlgorithmId;
use crate::envelope::{TaggedPublicKey, TaggedSignature};
use crate::error::CryptoError;
use crate::traits::{Signer, Verifier};
use crate::pqc::ed25519_native::{Ed25519Signer, Ed25519Verifier};
use crate::pqc::dilithium::{MlDsa65Signer, MlDsa65Verifier};
use zeroize::Zeroizing;

/// Hybrid signer that produces dual Ed25519 + ML-DSA-65 signatures.
pub struct HybridSigner {
    ed25519: Ed25519Signer,
    mldsa65: MlDsa65Signer,
}

impl HybridSigner {
    /// Generate new hybrid keypair.
    pub fn generate() -> Self {
        Self {
            ed25519: Ed25519Signer::generate(),
            mldsa65: MlDsa65Signer::generate(),
        }
    }

    /// Get the Ed25519 public key component.
    pub fn ed25519_public_key(&self) -> TaggedPublicKey {
        self.ed25519.public_key()
    }

    /// Get the ML-DSA-65 public key component.
    pub fn mldsa65_public_key(&self) -> TaggedPublicKey {
        self.mldsa65.public_key()
    }

    /// Concatenated secret key bytes: Ed25519 (32) || ML-DSA-65 secret key. Zeroized on drop.
    pub fn secret_key_bytes(&self) -> Zeroizing<Vec<u8>> {
        let mut buf = Vec::from(self.ed25519.secret_key_bytes().as_slice());
        buf.extend(self.mldsa65.secret_key_bytes().as_slice());
        Zeroizing::new(buf)
    }

    /// Create from concatenated secret key bytes + public key bytes.
    ///
    /// `secret_key_bytes`: Ed25519 secret (32) || ML-DSA-65 secret key
    /// `public_key_bytes`: Ed25519 public (32) || ML-DSA-65 public key (1952)
    pub fn from_bytes(
        public_key_bytes: &[u8],
        secret_key_bytes: &[u8],
    ) -> Result<Self, CryptoError> {
        if public_key_bytes.len() != 32 + 1952 {
            return Err(CryptoError::InvalidKeyLength {
                algorithm: "HybridEd25519MlDsa65",
                expected: 32 + 1952,
                actual: public_key_bytes.len(),
            });
        }
        let ed_sk: [u8; 32] = secret_key_bytes[..32]
            .try_into()
            .map_err(|_| CryptoError::Validation("Hybrid secret key too short for Ed25519 component".into()))?;
        let ed = Ed25519Signer::from_bytes(&ed_sk);
        let mldsa = MlDsa65Signer::from_bytes(&public_key_bytes[32..], &secret_key_bytes[32..])?;
        Ok(Self { ed25519: ed, mldsa65: mldsa })
    }
}

impl Signer for HybridSigner {
    fn algorithm(&self) -> AlgorithmId {
        AlgorithmId::HybridEd25519MlDsa65
    }

    fn sign(&self, message: &[u8]) -> Result<TaggedSignature, CryptoError> {
        let ed_sig = self.ed25519.sign(message)?;
        let pqc_sig = self.mldsa65.sign(message)?;

        let mut combined = Vec::with_capacity(64 + pqc_sig.signature_bytes.len());
        combined.extend_from_slice(&ed_sig.signature_bytes);
        combined.extend_from_slice(&pqc_sig.signature_bytes);

        TaggedSignature::new(AlgorithmId::HybridEd25519MlDsa65, combined)
    }

    fn public_key(&self) -> TaggedPublicKey {
        let ed_pk = self.ed25519.public_key();
        let pqc_pk = self.mldsa65.public_key();

        let mut combined = Vec::with_capacity(32 + pqc_pk.key_bytes.len());
        combined.extend_from_slice(&ed_pk.key_bytes);
        combined.extend_from_slice(&pqc_pk.key_bytes);

        TaggedPublicKey::new(AlgorithmId::HybridEd25519MlDsa65, combined).unwrap()
    }
}

/// Hybrid verifier that checks BOTH Ed25519 and ML-DSA-65 components.
pub struct HybridVerifier;

impl Verifier for HybridVerifier {
    fn algorithm(&self) -> AlgorithmId {
        AlgorithmId::HybridEd25519MlDsa65
    }

    fn verify(
        &self,
        public_key: &TaggedPublicKey,
        message: &[u8],
        signature: &TaggedSignature,
    ) -> Result<bool, CryptoError> {
        if public_key.algorithm != AlgorithmId::HybridEd25519MlDsa65 {
            return Err(CryptoError::Validation(format!(
                "Expected hybrid key, got {:?}",
                public_key.algorithm
            )));
        }

        // Split combined public key: Ed25519 (32) || ML-DSA-65 (1952)
        if public_key.key_bytes.len() != 32 + 1952 {
            return Err(CryptoError::InvalidKeyLength {
                algorithm: "HybridEd25519MlDsa65VerificationKey2024",
                expected: 32 + 1952,
                actual: public_key.key_bytes.len(),
            });
        }

        let ed_pk = TaggedPublicKey::new(
            AlgorithmId::Ed25519,
            public_key.key_bytes[..32].to_vec(),
        )?;
        let pqc_pk = TaggedPublicKey::new(
            AlgorithmId::MlDsa65,
            public_key.key_bytes[32..].to_vec(),
        )?;

        // Split signature: Ed25519 (64) || ML-DSA-65 (rest)
        let ed_component = signature
            .ed25519_component()
            .ok_or_else(|| CryptoError::Validation("No Ed25519 component in hybrid sig".into()))?;
        let pqc_component = signature
            .pqc_component()
            .ok_or_else(|| CryptoError::Validation("No PQC component in hybrid sig".into()))?;

        let ed_sig = TaggedSignature::new(AlgorithmId::Ed25519, ed_component.to_vec())?;
        let pqc_sig = TaggedSignature::new(AlgorithmId::MlDsa65, pqc_component.to_vec())?;

        // BOTH must pass
        let ed_ok = Ed25519Verifier.verify(&ed_pk, message, &ed_sig)?;
        let pqc_ok = MlDsa65Verifier.verify(&pqc_pk, message, &pqc_sig)?;

        Ok(ed_ok && pqc_ok)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_sign_verify() {
        let signer = HybridSigner::generate();
        let msg = b"hybrid quantum-resistant message";
        let sig = signer.sign(msg).unwrap();
        let pk = signer.public_key();

        assert_eq!(pk.algorithm, AlgorithmId::HybridEd25519MlDsa65);
        assert_eq!(pk.key_bytes.len(), 32 + 1952);
        assert!(sig.ed25519_component().is_some());
        assert!(sig.pqc_component().is_some());

        let verifier = HybridVerifier;
        assert!(verifier.verify(&pk, msg, &sig).unwrap());
    }

    #[test]
    fn hybrid_wrong_message() {
        let signer = HybridSigner::generate();
        let sig = signer.sign(b"correct").unwrap();
        let pk = signer.public_key();

        let verifier = HybridVerifier;
        assert!(!verifier.verify(&pk, b"wrong", &sig).unwrap());
    }
}
