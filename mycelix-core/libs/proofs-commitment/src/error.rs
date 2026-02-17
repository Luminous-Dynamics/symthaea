//! Error types for commitment operations

use thiserror::Error;

/// Errors that can occur during commitment operations
#[derive(Error, Debug, Clone)]
pub enum CommitmentError {
    /// Value is out of the expected range
    #[error("value out of range: {value} not in [{min}, {max}]")]
    ValueOutOfRange {
        value: f64,
        min: f64,
        max: f64,
    },

    /// Value is not finite (NaN or Infinity)
    #[error("value is not finite: {0}")]
    NonFiniteValue(String),

    /// Commitment verification failed
    #[error("commitment verification failed: {0}")]
    VerificationFailed(String),

    /// Invalid input length
    #[error("invalid input length: expected {expected}, got {actual}")]
    InvalidLength {
        expected: usize,
        actual: usize,
    },

    /// Scaling overflow
    #[error("scaling overflow: value {value} with scale {scale} exceeds maximum")]
    ScalingOverflow {
        value: f64,
        scale: u64,
    },
}

/// Result type for commitment operations
pub type CommitmentResult<T> = Result<T, CommitmentError>;
