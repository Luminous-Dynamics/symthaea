//! # Closed-form Continuous-time (CfC) Neural Networks
//!
//! Implementation of CfC networks with closed-form solutions for continuous-time
//! neural dynamics. The core equation is:
//!
//! ```text
//! h(t) = h_inf + (h_0 - h_inf) * exp(-dt/tau)
//! ```
//!
//! ## Modules
//!
//! - [`types`] - Configuration types, activation functions, and utility functions
//! - [`cell`] - Single CfC cell with forward/backward passes
//! - [`gradients`] - Gradient accumulators and Adam optimizer state
//! - [`network`] - Multi-layer CfC network with BPTT and SPSA training
//! - [`phi_gated`] - Phi-gated attention utilities for IIT integration

mod types;
mod cell;
mod gradients;
mod network;
mod phi_gated;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

// Re-export all public API from submodules

// Types, config, and utilities
pub use types::{
    CfCConfig, ActivationType, OnlineLearningConfig,
    OnlineLearningStats, NetworkOnlineLearningStats,
    DynamicsDiagnostic,
};

// MIN_TAU, fast_sigmoid, sigmoid, mse_loss are pub(crate) in types.rs
// and accessed by cell.rs/network.rs via `use super::types::*`.

// Cell
pub use cell::CfCCell;

// Gradients and optimizer state
pub use gradients::{CfCGradients, CfCCellCache, AdamState, OutputAdamState};

// Network
pub use network::{CfCNetworkConfig, CfCNetwork};

// Phi-gated attention
pub use phi_gated::{PhiGatedConfig, compute_phi_attention_weights};
// weighted_array_bundle is pub(crate) in phi_gated.rs and accessed
// by network.rs via `use super::phi_gated::weighted_array_bundle`.
