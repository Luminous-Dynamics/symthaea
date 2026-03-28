// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! ML-DSA (Dilithium) signer/verifier using pqcrypto-dilithium.
//!
//! Supports ML-DSA-65 (Dilithium3, NIST Level 3) and ML-DSA-87 (Dilithium5, NIST Level 5).

use crate::algorithm::AlgorithmId;
use crate::envelope::{TaggedPublicKey, TaggedSignature};
use crate::error::CryptoError;
use crate::traits::{Signer, Verifier};

use pqcrypto_dilithium::dilithium3;
use pqcrypto_dilithium::dilithium5;
use pqcrypto_traits::sign::{DetachedSignature, PublicKey, SecretKey};
use zeroize::Zeroizing;

// ==================== ML-DSA-65 (Dilithium3) ====================

/// ML-DSA-65 signer (NIST Level 3).
pub struct MlDsa65Signer {
    public_key: dilithium3::PublicKey,
    secret_key: dilithium3::SecretKey,
}

impl MlDsa65Signer {
    /// Generate a new ML-DSA-65 keypair.
    pub fn generate() -> Self {
        let (pk, sk) = dilithium3::keypair();
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
        let pk = dilithium3::PublicKey::from_bytes(public_key_bytes)
            .map_err(|_| CryptoError::Validation("Invalid ML-DSA-65 public key".into()))?;
        let sk = dilithium3::SecretKey::from_bytes(secret_key_bytes)
            .map_err(|_| CryptoError::Validation("Invalid ML-DSA-65 secret key".into()))?;
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

impl Signer for MlDsa65Signer {
    fn algorithm(&self) -> AlgorithmId {
        AlgorithmId::MlDsa65
    }

    fn sign(&self, message: &[u8]) -> Result<TaggedSignature, CryptoError> {
        let sig = dilithium3::detached_sign(message, &self.secret_key);
        TaggedSignature::new(AlgorithmId::MlDsa65, sig.as_bytes().to_vec())
    }

    fn public_key(&self) -> TaggedPublicKey {
        TaggedPublicKey::new(AlgorithmId::MlDsa65, self.public_key.as_bytes().to_vec())
            .expect("MlDsa65 public key size is constant per FIPS 204")
    }
}

/// ML-DSA-65 verifier.
pub struct MlDsa65Verifier;

impl Verifier for MlDsa65Verifier {
    fn algorithm(&self) -> AlgorithmId {
        AlgorithmId::MlDsa65
    }

    fn verify(
        &self,
        public_key: &TaggedPublicKey,
        message: &[u8],
        signature: &TaggedSignature,
    ) -> Result<bool, CryptoError> {
        if public_key.algorithm != AlgorithmId::MlDsa65 {
            return Err(CryptoError::Validation(format!(
                "Expected ML-DSA-65 key, got {:?}",
                public_key.algorithm
            )));
        }

        let pk = dilithium3::PublicKey::from_bytes(&public_key.key_bytes)
            .map_err(|_| CryptoError::Validation("Invalid ML-DSA-65 public key bytes".into()))?;

        let sig_bytes = signature
            .pqc_component()
            .ok_or_else(|| CryptoError::Validation("No PQC signature component".into()))?;

        let sig = dilithium3::DetachedSignature::from_bytes(sig_bytes)
            .map_err(|_| CryptoError::Validation("Invalid ML-DSA-65 signature bytes".into()))?;

        Ok(dilithium3::verify_detached_signature(&sig, message, &pk).is_ok())
    }
}

// ==================== ML-DSA-87 (Dilithium5) ====================

/// ML-DSA-87 signer (NIST Level 5).
pub struct MlDsa87Signer {
    public_key: dilithium5::PublicKey,
    secret_key: dilithium5::SecretKey,
}

impl MlDsa87Signer {
    /// Generate a new ML-DSA-87 keypair.
    pub fn generate() -> Self {
        let (pk, sk) = dilithium5::keypair();
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
        let pk = dilithium5::PublicKey::from_bytes(public_key_bytes)
            .map_err(|_| CryptoError::Validation("Invalid ML-DSA-87 public key".into()))?;
        let sk = dilithium5::SecretKey::from_bytes(secret_key_bytes)
            .map_err(|_| CryptoError::Validation("Invalid ML-DSA-87 secret key".into()))?;
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

impl Signer for MlDsa87Signer {
    fn algorithm(&self) -> AlgorithmId {
        AlgorithmId::MlDsa87
    }

    fn sign(&self, message: &[u8]) -> Result<TaggedSignature, CryptoError> {
        let sig = dilithium5::detached_sign(message, &self.secret_key);
        TaggedSignature::new(AlgorithmId::MlDsa87, sig.as_bytes().to_vec())
    }

    fn public_key(&self) -> TaggedPublicKey {
        TaggedPublicKey::new(AlgorithmId::MlDsa87, self.public_key.as_bytes().to_vec())
            .expect("MlDsa87 public key size is constant per FIPS 204")
    }
}

/// ML-DSA-87 verifier.
pub struct MlDsa87Verifier;

impl Verifier for MlDsa87Verifier {
    fn algorithm(&self) -> AlgorithmId {
        AlgorithmId::MlDsa87
    }

    fn verify(
        &self,
        public_key: &TaggedPublicKey,
        message: &[u8],
        signature: &TaggedSignature,
    ) -> Result<bool, CryptoError> {
        if public_key.algorithm != AlgorithmId::MlDsa87 {
            return Err(CryptoError::Validation(format!(
                "Expected ML-DSA-87 key, got {:?}",
                public_key.algorithm
            )));
        }

        let pk = dilithium5::PublicKey::from_bytes(&public_key.key_bytes)
            .map_err(|_| CryptoError::Validation("Invalid ML-DSA-87 public key bytes".into()))?;

        let sig_bytes = signature
            .pqc_component()
            .ok_or_else(|| CryptoError::Validation("No PQC signature component".into()))?;

        let sig = dilithium5::DetachedSignature::from_bytes(sig_bytes)
            .map_err(|_| CryptoError::Validation("Invalid ML-DSA-87 signature bytes".into()))?;

        Ok(dilithium5::verify_detached_signature(&sig, message, &pk).is_ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mldsa65_sign_verify() {
        let signer = MlDsa65Signer::generate();
        let msg = b"post-quantum mycelix";
        let sig = signer.sign(msg).unwrap();
        let pk = signer.public_key();

        let verifier = MlDsa65Verifier;
        assert!(verifier.verify(&pk, msg, &sig).unwrap());
    }

    #[test]
    fn mldsa65_wrong_message() {
        let signer = MlDsa65Signer::generate();
        let sig = signer.sign(b"correct").unwrap();
        let pk = signer.public_key();

        let verifier = MlDsa65Verifier;
        assert!(!verifier.verify(&pk, b"wrong", &sig).unwrap());
    }

    #[test]
    fn mldsa87_sign_verify() {
        let signer = MlDsa87Signer::generate();
        let msg = b"highest security";
        let sig = signer.sign(msg).unwrap();
        let pk = signer.public_key();

        let verifier = MlDsa87Verifier;
        assert!(verifier.verify(&pk, msg, &sig).unwrap());
    }
}
