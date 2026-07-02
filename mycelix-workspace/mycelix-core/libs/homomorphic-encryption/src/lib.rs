// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix Homomorphic Encryption Library
//!
//! Pure Rust implementation of homomorphic encryption schemes for
//! privacy-preserving federated learning. Supports:
//!
//! - **Paillier**: Additively homomorphic, ideal for gradient aggregation
//! - **ElGamal**: Multiplicatively homomorphic variant
//! - **Encrypted Vectors**: High-level API for FL gradient operations
//!
//! # Example
//!
//! ```rust
//! use mycelix_homomorphic::{PaillierKeyPair};
//!
//! // Generate keys (using smaller key for example speed)
//! let keypair = PaillierKeyPair::generate(1024).unwrap();
//!
//! // Encrypt values
//! let a = keypair.public_key().encrypt(100i64);
//! let b = keypair.public_key().encrypt(50i64);
//!
//! // Add encrypted values (homomorphic addition)
//! let c = a.add(&b);
//!
//! // Decrypt result
//! let result = keypair.decrypt_i64(&c).unwrap();
//! assert_eq!(result, 150);
//! ```

#[cfg(test)]
mod tests;

pub mod paillier;
pub mod encrypted_vector;
pub mod encoding;
pub mod utils;

pub use paillier::{Paillier, PaillierKeyPair, PaillierPublicKey, PaillierCiphertext};
pub use encrypted_vector::{EncryptedVector, EncryptedGradient};
pub use encoding::{FixedPointEncoder, EncodingParams};

use thiserror::Error;

/// Errors that can occur in homomorphic encryption operations
#[derive(Error, Debug)]
pub enum HeError {
    #[error("Key generation failed: {0}")]
    KeyGenError(String),

    #[error("Encryption failed: {0}")]
    EncryptionError(String),

    #[error("Decryption failed: {0}")]
    DecryptionError(String),

    #[error("Invalid parameters: {0}")]
    ParameterError(String),

    #[error("Encoding error: {0}")]
    EncodingError(String),

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Overflow in homomorphic operation")]
    Overflow,

    #[error("Serialization error: {0}")]
    SerializationError(String),
}

/// Result type for HE operations
pub type HeResult<T> = Result<T, HeError>;

/// Version of the library
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Supported key sizes in bits
pub const SUPPORTED_KEY_SIZES: &[usize] = &[1024, 2048, 3072, 4096];

/// Recommended key size for security
pub const RECOMMENDED_KEY_SIZE: usize = 2048;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert_eq!(VERSION, "0.1.0");
    }

    #[test]
    fn test_supported_key_sizes() {
        assert!(SUPPORTED_KEY_SIZES.contains(&2048));
        assert!(SUPPORTED_KEY_SIZES.contains(&4096));
    }
}
