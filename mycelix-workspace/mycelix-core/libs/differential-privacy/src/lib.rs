// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix Differential Privacy Library
//!
//! Provides differential privacy primitives for federated learning:
//! - Gaussian mechanism (for (ε,δ)-DP)
//! - Laplace mechanism (for ε-DP)
//! - Privacy budget tracking (Moments Accountant)
//! - Gradient clipping utilities
//!
//! # Example
//!
//! ```rust
//! use mycelix_differential_privacy::{GaussianMechanism, PrivacyBudget, Mechanism};
//!
//! // Create a privacy budget
//! let mut budget = PrivacyBudget::new(1.0, 1e-5); // ε=1.0, δ=1e-5
//!
//! // Create Gaussian mechanism
//! let mechanism = GaussianMechanism::new(1.0, 1.0, 1e-5).unwrap();
//!
//! // Add noise to gradients
//! let gradients = vec![0.5, -0.3, 0.8];
//! let noisy_gradients = mechanism.apply(&gradients);
//!
//! // Track privacy expenditure
//! budget.consume(1.0, 1e-6).unwrap();
//! ```

pub mod mechanisms;
pub mod budget;
pub mod clipping;
pub mod composition;

#[cfg(feature = "python")]
pub mod python;

pub use mechanisms::{Mechanism, GaussianMechanism, LaplaceMechanism};
pub use budget::PrivacyBudget;
pub use clipping::{GradientClipper, ClippingMethod};
pub use composition::{AdvancedComposition, MomentsAccountant};

use thiserror::Error;

/// Errors that can occur in differential privacy operations
#[derive(Error, Debug)]
pub enum DpError {
    #[error("Invalid epsilon: {0} (must be positive)")]
    InvalidEpsilon(f64),

    #[error("Invalid delta: {0} (must be in [0, 1))")]
    InvalidDelta(f64),

    #[error("Invalid sensitivity: {0} (must be positive)")]
    InvalidSensitivity(f64),

    #[error("Privacy budget exhausted: requested ε={requested}, remaining={remaining}")]
    BudgetExhausted { requested: f64, remaining: f64 },

    #[error("Invalid clipping norm: {0} (must be positive)")]
    InvalidClipNorm(f64),

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

/// Result type for DP operations
pub type DpResult<T> = Result<T, DpError>;

/// Version of the library
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_version() {
        assert_eq!(VERSION, "0.1.0");
    }
}
