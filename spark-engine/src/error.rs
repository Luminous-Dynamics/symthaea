//! # Error Types

use thiserror::Error;

/// Spark Engine error types.
#[derive(Error, Debug)]
pub enum SparkError {
    #[error("Physics calculation error: {0}")]
    Physics(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Design infeasible: {0}")]
    DesignInfeasible(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Result type for Spark Engine operations.
pub type Result<T> = std::result::Result<T, SparkError>;
