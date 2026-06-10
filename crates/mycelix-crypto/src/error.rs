// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Crypto error types for mycelix-crypto

use core::fmt;

/// Errors from cryptographic operations and type validation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CryptoError {
    /// Key bytes have wrong length for the algorithm
    InvalidKeyLength {
        algorithm: &'static str,
        expected: usize,
        actual: usize,
    },
    /// Signature bytes have wrong length for the algorithm
    InvalidSignatureLength {
        algorithm: &'static str,
        expected: usize,
        actual: usize,
    },
    /// Algorithm is not recognized or not supported in this build
    UnsupportedAlgorithm(u16),
    /// Multibase string is malformed
    InvalidMultibase(String),
    /// Multicodec prefix does not match the declared algorithm
    MulticodecMismatch {
        expected_prefix: [u8; 2],
        actual_prefix: [u8; 2],
    },
    /// Base58 decoding failed
    Base58Decode(String),
    /// Generic validation error
    Validation(String),
}

impl fmt::Display for CryptoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CryptoError::InvalidKeyLength {
                algorithm,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "Invalid key length for {}: expected {}, got {}",
                    algorithm, expected, actual
                )
            }
            CryptoError::InvalidSignatureLength {
                algorithm,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "Invalid signature length for {}: expected {}, got {}",
                    algorithm, expected, actual
                )
            }
            CryptoError::UnsupportedAlgorithm(id) => {
                write!(f, "Unsupported algorithm ID: {:#06x}", id)
            }
            CryptoError::InvalidMultibase(msg) => {
                write!(f, "Invalid multibase encoding: {}", msg)
            }
            CryptoError::MulticodecMismatch {
                expected_prefix,
                actual_prefix,
            } => {
                write!(
                    f,
                    "Multicodec prefix mismatch: expected [{:#04x}, {:#04x}], got [{:#04x}, {:#04x}]",
                    expected_prefix[0], expected_prefix[1],
                    actual_prefix[0], actual_prefix[1],
                )
            }
            CryptoError::Base58Decode(msg) => {
                write!(f, "Base58 decode error: {}", msg)
            }
            CryptoError::Validation(msg) => {
                write!(f, "Validation error: {}", msg)
            }
        }
    }
}

// Manual std::error::Error impl that works in no_std (thiserror would work too,
// but we keep it simple since this is a leaf crate).
#[cfg(feature = "native")]
impl std::error::Error for CryptoError {}
