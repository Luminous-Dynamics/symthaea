//! Mycelix HyperFeel core.
//!
//! This crate provides a small, Rust-first core for
//! hypervector-based gradient encoding and aggregation.
//!
//! It is intentionally minimal and backend-agnostic:
//! the internal `Hypervector` representation can later
//! be swapped for Symthaea's HV16 type without changing
//! the public API.

pub mod hypervector;
pub mod hypergradient;
pub mod model_adapter;
#[cfg(feature = "symthaea-hv")]
pub mod symthaea_backend;

pub use crate::hypergradient::HyperGradient;
pub use crate::hypervector::Hypervector;
pub use crate::model_adapter::{ModelAdapter, ModelUpdate, ArchitectureInfo};
