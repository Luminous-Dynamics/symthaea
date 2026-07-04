// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Error types for the ZKP core library.

use thiserror::Error;

/// Result type alias for ZKP operations.
pub type ZkpResult<T> = Result<T, ZkpError>;

/// Errors that can occur during ZKP operations.
#[derive(Debug, Error)]
pub enum ZkpError {
    #[error("proof verification failed: {0}")]
    VerificationFailed(String),

    #[error("invalid proof format: {0}")]
    InvalidProofFormat(String),

    #[error("backend unavailable: {0}")]
    BackendUnavailable(String),

    #[error("invalid key material: {0}")]
    InvalidKeyMaterial(String),

    #[error("signature verification failed")]
    SignatureInvalid,

    #[error("domain tag mismatch: expected {expected}, got {actual}")]
    DomainTagMismatch { expected: String, actual: String },

    #[error("proof expired: {0}")]
    ProofExpired(String),

    #[error("replay detected: {0}")]
    ReplayDetected(String),

    #[error("client identity mismatch")]
    ClientIdMismatch,

    #[error("proof size exceeds limit: {size} > {limit}")]
    ProofTooLarge { size: usize, limit: usize },

    #[error("security level insufficient: need {required}, have {actual}")]
    InsufficientSecurity { required: String, actual: String },

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("proving error: {0}")]
    ProvingError(String),
}
