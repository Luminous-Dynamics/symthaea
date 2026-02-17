//! Error types for FL aggregator operations.

use thiserror::Error;

/// Result type alias for aggregator operations.
pub type Result<T> = std::result::Result<T, AggregatorError>;

/// Errors that can occur during aggregation.
#[derive(Debug, Error)]
pub enum AggregatorError {
    /// Not enough gradients for the selected defense algorithm.
    #[error("Insufficient gradients: have {have}, need {need} for {defense} defense")]
    InsufficientGradients {
        have: usize,
        need: usize,
        defense: String,
    },

    /// Gradient dimensions don't match.
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// Memory limit exceeded.
    #[error("Memory limit exceeded: {used} bytes > {limit} bytes")]
    MemoryLimitExceeded { used: usize, limit: usize },

    /// Invalid node (not registered).
    #[error("Invalid node: {0}")]
    InvalidNode(String),

    /// Duplicate submission in same round.
    #[error("Duplicate submission from node {node} in round {round}")]
    DuplicateSubmission { node: String, round: u64 },

    /// No gradients submitted for round.
    #[error("No gradients submitted for round {0}")]
    NoGradients(u64),

    /// Round not complete.
    #[error("Round {round} not complete: {submitted}/{expected} nodes")]
    RoundNotComplete {
        round: u64,
        submitted: usize,
        expected: usize,
    },

    /// Invalid defense configuration.
    #[error("Invalid defense config: {0}")]
    InvalidConfig(String),

    /// Signature verification failed.
    #[error("Signature verification failed for node {0}")]
    InvalidSignature(String),

    /// Compression error.
    #[error("Compression error: {0}")]
    Compression(String),

    /// Network error.
    #[error("Network error: {0}")]
    Network(String),

    /// Internal error.
    #[error("Internal error: {0}")]
    Internal(String),
}
