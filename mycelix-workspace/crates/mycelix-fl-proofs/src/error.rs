// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Ported from fl-aggregator/src/proofs/error.rs
//! Proof error types for zkSTARK proof generation and verification.

use thiserror::Error;

/// Errors that can occur during proof operations.
#[derive(Debug, Error)]
pub enum ProofError {
    #[error("Proof generation failed: {0}")]
    GenerationFailed(String),

    #[error("Proof verification failed: {0}")]
    VerificationFailed(String),

    #[error("Invalid witness: {0}")]
    InvalidWitness(String),

    #[error("Invalid public inputs: {0}")]
    InvalidPublicInputs(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),

    #[error("Trace construction error: {0}")]
    TraceError(String),

    #[error("Value out of field range: {0}")]
    FieldOverflow(String),

    #[error("Invalid proof format: {0}")]
    InvalidProofFormat(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

/// Result type for proof operations.
pub type ProofResult<T> = Result<T, ProofError>;

impl From<std::io::Error> for ProofError {
    fn from(err: std::io::Error) -> Self {
        ProofError::SerializationError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = ProofError::GenerationFailed("timeout".into());
        assert!(err.to_string().contains("timeout"));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "missing file");
        let proof_err: ProofError = io_err.into();
        assert!(matches!(proof_err, ProofError::SerializationError(_)));
    }
}
