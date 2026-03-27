// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Aggregation defense algorithms for Byzantine-robust federated learning.
//!
//! All stateless defenses implement the [`Defense`] trait, which provides a
//! uniform `aggregate()` interface.
//!
//! ## Implemented algorithms
//!
//! | Algorithm | Module | BFT Guarantee |
//! |-----------|--------|---------------|
//! | FedAvg | [`fedavg`] | None (baseline) |
//! | Trimmed Mean | [`trimmed_mean`] | f < (1-2β)n |
//! | Coordinate Median | [`coordinate_median`] | f < n/2 |
//! | RFA (Geometric Median) | [`rfa`] | Provably robust |
//!
//! ## PoGQ (stateful)
//!
//! PoGQ v4.1 Enhanced and Adaptive PoGQ are **stateful** defenses that
//! maintain per-node history across rounds. They live in their own module
//! [`crate::pogq`] rather than here, because they do not implement the
//! stateless [`Defense`] trait.

pub mod coordinate_median;
pub mod fedavg;
pub mod rfa;
pub mod trimmed_mean;

pub mod boba;
pub mod cbf;
pub mod foolsgold;
pub mod fltrust;
pub mod krum;
pub mod sybil_weights;

pub use coordinate_median::CoordinateMedian;
pub use fedavg::FedAvg;
pub use rfa::Rfa;
pub use trimmed_mean::TrimmedMean;

use crate::error::FlError;
use crate::types::{AggregationResult, DefenseConfig, Gradient};

/// Trait for all aggregation/defense algorithms.
pub trait Defense: Send + Sync {
    /// Aggregate gradients, returning the aggregated result.
    fn aggregate(
        &self,
        gradients: &[Gradient],
        config: &DefenseConfig,
    ) -> Result<AggregationResult, FlError>;

    /// Algorithm name.
    fn name(&self) -> &str;
}

/// Validate that gradients are non-empty and consistently dimensioned.
///
/// Returns the dimensionality on success.
pub(crate) fn validate_gradients(gradients: &[Gradient]) -> Result<usize, FlError> {
    if gradients.is_empty() {
        return Err(FlError::InsufficientGradients { got: 0, need: 1 });
    }
    let dim = gradients[0].values.len();
    if dim == 0 {
        return Err(FlError::EmptyGradient);
    }
    for g in &gradients[1..] {
        if g.values.len() != dim {
            return Err(FlError::DimensionMismatch {
                expected: dim,
                got: g.values.len(),
            });
        }
    }
    Ok(dim)
}
