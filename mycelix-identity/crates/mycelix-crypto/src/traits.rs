// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cryptographic trait abstractions for modular PQE.
//!
//! These traits are only available with the `native` feature since they require
//! actual cryptographic operations backed by real libraries.

use crate::algorithm::AlgorithmId;
use crate::envelope::{EncryptedEnvelope, TaggedPublicKey, TaggedSignature};
use crate::error::CryptoError;

/// A key pair that can produce signatures.
pub trait Signer {
    /// Which algorithm this signer uses.
    fn algorithm(&self) -> AlgorithmId;

    /// Sign arbitrary message bytes.
    fn sign(&self, message: &[u8]) -> Result<TaggedSignature, CryptoError>;

    /// Get the public key for this signer.
    fn public_key(&self) -> TaggedPublicKey;
}

/// Verifies signatures against public keys.
pub trait Verifier {
    /// Which algorithm this verifier handles.
    fn algorithm(&self) -> AlgorithmId;

    /// Verify a signature over a message using the given public key.
    fn verify(
        &self,
        public_key: &TaggedPublicKey,
        message: &[u8],
        signature: &TaggedSignature,
    ) -> Result<bool, CryptoError>;
}

/// Key encapsulation mechanism (for PQE key exchange).
pub trait KeyEncapsulator {
    /// Which KEM algorithm.
    fn algorithm(&self) -> AlgorithmId;

    /// Encapsulate: given recipient's public key, produce (kem_ciphertext, shared_secret).
    fn encapsulate(&self, public_key: &TaggedPublicKey) -> Result<(Vec<u8>, Vec<u8>), CryptoError>;

    /// Decapsulate: recover shared_secret from KEM ciphertext using private key.
    fn decapsulate(&self, ciphertext: &[u8]) -> Result<Vec<u8>, CryptoError>;
}

/// Authenticated encryption with associated data (AEAD).
pub trait Encryptor {
    /// Encrypt plaintext, returning an EncryptedEnvelope.
    fn encrypt(
        &self,
        plaintext: &[u8],
        associated_data: &[u8],
        recipient_key_id: &str,
    ) -> Result<EncryptedEnvelope, CryptoError>;

    /// Decrypt an EncryptedEnvelope, returning plaintext.
    fn decrypt(
        &self,
        envelope: &EncryptedEnvelope,
        associated_data: &[u8],
    ) -> Result<Vec<u8>, CryptoError>;
}
