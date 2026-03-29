// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Error types for K-Vector ZKP

use thiserror::Error;

/// Errors that can occur during ZKP operations
#[derive(Debug, Error)]
pub enum ZkpError {
    /// Invalid K-Vector value (out of range)
    #[error("K-Vector component out of range [0, 1]: {component} = {value}")]
    ValueOutOfRange {
        component: &'static str,
        value: f32,
    },

    /// Degenerate input that would produce trivial constraints
    #[error("Degenerate K-Vector: {0}")]
    DegenerateInput(String),

    /// Proof generation failed
    #[error("Failed to generate proof: {0}")]
    ProofGenerationFailed(String),

    /// Proof verification failed
    #[error("Proof verification failed: {0}")]
    VerificationFailed(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Invalid proof format
    #[error("Invalid proof format: {0}")]
    InvalidProof(String),
}

/// Result type for ZKP operations
pub type ZkpResult<T> = Result<T, ZkpError>;
