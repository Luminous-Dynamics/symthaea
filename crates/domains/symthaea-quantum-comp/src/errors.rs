//! Error types for the crate.

use core::fmt;

/// Result alias used by `symthaea-quantum-comp`.
pub type Result<T> = core::result::Result<T, QuantumCompError>;

/// Errors produced by research probes and substrate adapters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantumCompError {
    /// A hypervector dimension was zero or incompatible.
    InvalidDimension,
    /// Two structures had incompatible lengths or dimensions.
    DimensionMismatch {
        /// Expected dimension or element count.
        expected: usize,
        /// Actual dimension or element count.
        actual: usize,
    },
    /// A probability, threshold, or noise parameter was outside its valid range.
    InvalidProbability,
    /// A benchmark configuration was invalid.
    InvalidConfig(&'static str),
}

impl fmt::Display for QuantumCompError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimension => write!(f, "invalid zero dimension"),
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
            Self::InvalidProbability => write!(f, "invalid probability or threshold"),
            Self::InvalidConfig(msg) => write!(f, "invalid benchmark config: {msg}"),
        }
    }
}

impl std::error::Error for QuantumCompError {}
