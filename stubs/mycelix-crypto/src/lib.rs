//! mycelix-crypto stub
//!
//! Minimal stub providing the types and traits that symthaea imports
//! from the full mycelix-crypto crate. This allows symthaea to compile
//! without the full mycelix-identity dependency.

pub mod pqc;
pub mod traits;

use std::fmt;

/// Algorithm identifier for tagged public keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AlgorithmId {
    Ed25519,
    MlDsa65,
    MlKem768,
    SlhDsaShake128f,
}

/// A public key tagged with its algorithm identifier.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TaggedPublicKey {
    pub algorithm: AlgorithmId,
    pub key_bytes: Vec<u8>,
}

/// Crypto error type.
#[derive(Debug, Clone)]
pub struct CryptoError(pub String);

impl fmt::Display for CryptoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CryptoError: {}", self.0)
    }
}

impl std::error::Error for CryptoError {}

impl TaggedPublicKey {
    pub fn new(algorithm: AlgorithmId, key_bytes: Vec<u8>) -> Result<Self, CryptoError> {
        if key_bytes.is_empty() {
            return Err(CryptoError("empty key bytes".into()));
        }
        Ok(Self {
            algorithm,
            key_bytes,
        })
    }
}
