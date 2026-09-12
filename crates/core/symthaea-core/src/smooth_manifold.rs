// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Optional smooth-manifold refinement for planner-neutral state spaces.
//!
//! Not every [`StateSpace`](crate::state_space::StateSpace) is smooth. This
//! module therefore adds tangent-space calculus as an explicit refinement over
//! [`MetricSpace`](crate::state_space::MetricSpace), preserving discrete and
//! hybrid spaces as first-class state spaces without forcing differentiability.

use crate::state_space::{MetricSpace, StateSpaceError, StateValidity};
use thiserror::Error;

/// Strength of the local shortest-geodesic relationship between two valid states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeodesicRelation {
    /// The selected profile establishes a unique local shortest geodesic.
    Unique,
    /// More than one shortest geodesic is admissible under the profile.
    NonUnique,
    /// Numerical conditioning prevents a reliable uniqueness determination.
    NumericallyDegenerate,
}

/// Result of validating a tangent vector at an exact base state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TangentValidity {
    /// The tangent satisfies the manifold's declared tangent-space constraints.
    Valid,
    /// The tangent's base state is not explicitly valid.
    BaseStateInvalid(StateValidity),
    /// The tangent representation has an incompatible dimension.
    DimensionMismatch { expected: usize, actual: usize },
    /// A tangent coordinate is NaN or infinite.
    NonFinite { coordinate: usize },
    /// A manifold-specific tangent constraint is violated.
    ConstraintViolation { reason: String },
    /// Validation could not establish validity or invalidity.
    Unknown { reason: String },
}

impl TangentValidity {
    /// Returns true only for an explicitly validated tangent vector.
    pub fn is_valid(&self) -> bool {
        matches!(self, Self::Valid)
    }
}

/// Fail-closed errors for smooth-manifold operations.
#[derive(Debug, Error, PartialEq)]
pub enum ManifoldError {
    /// A lower-level state-space precondition or calculation failed.
    #[error(transparent)]
    StateSpace(#[from] StateSpaceError),
    /// A tangent vector is invalid or could not be established as valid.
    #[error("{role} tangent is not valid: {validity:?}")]
    InvalidTangent {
        role: &'static str,
        validity: TangentValidity,
    },
    /// The requested operation requires a unique geodesic, but the cut locus does not provide one.
    #[error("operation requires a unique geodesic, but the selected endpoints are non-unique")]
    NonUniqueGeodesic,
    /// Numerical conditioning prevents a reliable manifold operation.
    #[error("{operation} is numerically degenerate: {reason}")]
    NumericallyDegenerate {
        operation: &'static str,
        reason: String,
    },
    /// A manifold profile is internally invalid.
    #[error("invalid manifold profile: {reason}")]
    InvalidProfile { reason: String },
    /// A manifold operation produced a non-finite result.
    #[error("{operation} produced a non-finite result")]
    NonFiniteComputation { operation: &'static str },
}

/// Smooth-manifold capabilities layered over a metric state space.
///
/// Implementations must preserve the attachment of each tangent vector to its
/// exact base point. Tangents may not be silently reused at another point
/// without transport or an explicit reinterpretation step.
pub trait SmoothManifold: MetricSpace {
    /// Tangent-vector representation for this manifold.
    type Tangent: Clone;

    /// Validate a tangent vector at an exact base state.
    fn validate_tangent(
        &self,
        point: &Self::State,
        tangent: &Self::Tangent,
    ) -> TangentValidity;

    /// Classify the local shortest-geodesic relation between two valid states.
    fn geodesic_relation(
        &self,
        from: &Self::State,
        to: &Self::State,
    ) -> Result<GeodesicRelation, ManifoldError>;

    /// Exponential map from one tangent space back onto the manifold.
    fn exp_map(
        &self,
        point: &Self::State,
        tangent: &Self::Tangent,
    ) -> Result<Self::State, ManifoldError>;

    /// Logarithmic map from a manifold point into a tangent space.
    ///
    /// Implementations should fail closed at a non-unique cut locus rather than
    /// pretending a selected tangent is intrinsically unique.
    fn log_map(
        &self,
        from: &Self::State,
        to: &Self::State,
    ) -> Result<Self::Tangent, ManifoldError>;

    /// Riemannian inner product of two tangents at the same base state.
    fn inner_product(
        &self,
        point: &Self::State,
        a: &Self::Tangent,
        b: &Self::Tangent,
    ) -> Result<f64, ManifoldError>;

    /// Parallel transport a tangent along a qualified geodesic.
    fn parallel_transport(
        &self,
        from: &Self::State,
        to: &Self::State,
        tangent: &Self::Tangent,
    ) -> Result<Self::Tangent, ManifoldError>;

    /// Convert tangent validation into a fail-closed operation precondition.
    fn require_valid_tangent(
        &self,
        role: &'static str,
        point: &Self::State,
        tangent: &Self::Tangent,
    ) -> Result<(), ManifoldError> {
        let validity = self.validate_tangent(point, tangent);
        if validity.is_valid() {
            Ok(())
        } else {
            Err(ManifoldError::InvalidTangent { role, validity })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_tangent_validity_never_becomes_valid() {
        assert!(!TangentValidity::Unknown {
            reason: "not evaluated".to_string(),
        }
        .is_valid());
    }
}
