use crate::hypergradient::HyperGradient;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Description of a model's architecture (framework-agnostic).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArchitectureInfo {
    pub name: String,
    pub parameter_count: u64,
}

/// Minimal representation of a model update.
///
/// For now this is just a flattened gradient vector; in the future
/// this can hold richer framework-specific metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelUpdate {
    pub gradient: Vec<f32>,
}

/// Errors that can occur when adapting model updates.
#[derive(Debug, Error)]
pub enum AdapterError {
    #[error("gradient is empty")]
    EmptyGradient,

    #[error("internal error: {0}")]
    Internal(String),
}

/// Generic interface for adapting model updates to HyperFeel.
pub trait ModelAdapter: Send + Sync {
    /// Convert a model update into a hypergradient.
    fn to_hypergradient(&self, update: &ModelUpdate) -> Result<HyperGradient, AdapterError>;

    /// Apply a hypergradient back onto the model.
    fn apply_hypergradient(&mut self, hg: &HyperGradient) -> Result<(), AdapterError>;

    /// Report high-level model architecture information.
    fn architecture(&self) -> ArchitectureInfo;
}

