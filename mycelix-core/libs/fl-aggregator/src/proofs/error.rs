//! Proof Error Types
//!
//! Error types for zkSTARK proof generation and verification.

use thiserror::Error;

/// Errors that can occur during proof operations
#[derive(Debug, Error)]
pub enum ProofError {
    /// Proof generation failed
    #[error("Proof generation failed: {0}")]
    GenerationFailed(String),

    /// Proof verification failed
    #[error("Proof verification failed: {0}")]
    VerificationFailed(String),

    /// Invalid witness data
    #[error("Invalid witness: {0}")]
    InvalidWitness(String),

    /// Invalid public inputs
    #[error("Invalid public inputs: {0}")]
    InvalidPublicInputs(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Constraint violation during trace construction
    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),

    /// Trace construction error
    #[error("Trace construction error: {0}")]
    TraceError(String),

    /// Backend-specific error
    #[error("Backend error: {0}")]
    BackendError(String),

    /// Value out of range for field representation
    #[error("Value out of field range: {0}")]
    FieldOverflow(String),

    /// Invalid proof format
    #[error("Invalid proof format: {0}")]
    InvalidProofFormat(String),

    /// GPU acceleration error
    #[error("GPU error: {0}")]
    GpuError(String),

    /// Invalid input parameters
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// HSM operation error
    #[error("HSM error: {0}")]
    HsmError(String),

    /// Cross-chain anchoring error
    #[error("Anchoring error: {0}")]
    AnchoringError(String),

    /// Timeout error
    #[error("Operation timed out: {0}")]
    Timeout(String),
}

/// Result type for proof operations
pub type ProofResult<T> = Result<T, ProofError>;

impl From<std::io::Error> for ProofError {
    fn from(err: std::io::Error) -> Self {
        ProofError::SerializationError(err.to_string())
    }
}
