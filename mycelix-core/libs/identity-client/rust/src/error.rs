// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Error types for Mycelix Identity Client

use thiserror::Error;

/// Identity client errors
#[derive(Error, Debug)]
pub enum IdentityError {
    /// Invalid DID format
    #[error("Invalid DID format: {0}")]
    InvalidDid(String),

    /// DID not found
    #[error("DID not found: {0}")]
    DidNotFound(String),

    /// Identity hApp not available
    #[error("Identity hApp not available")]
    IdentityHappUnavailable,

    /// Credential verification failed
    #[error("Credential verification failed: {0}")]
    CredentialVerificationFailed(String),

    /// Credential revoked
    #[error("Credential has been revoked: {0}")]
    CredentialRevoked(String),

    /// Credential expired
    #[error("Credential has expired: {0}")]
    CredentialExpired(String),

    /// Insufficient assurance level
    #[error("Insufficient assurance level: required {required:?}, got {actual:?}")]
    InsufficientAssuranceLevel {
        required: crate::AssuranceLevel,
        actual: crate::AssuranceLevel,
    },

    /// Zome call failed
    #[error("Zome call failed: {0}")]
    ZomeCallFailed(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Connection error
    #[error("Connection error: {0}")]
    ConnectionError(String),

    /// Cache error
    #[error("Cache error: {0}")]
    CacheError(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Unknown error
    #[error("Unknown error: {0}")]
    Unknown(String),
}

impl From<serde_json::Error> for IdentityError {
    fn from(err: serde_json::Error) -> Self {
        IdentityError::SerializationError(err.to_string())
    }
}

/// Result type for identity operations
pub type IdentityResult<T> = Result<T, IdentityError>;
