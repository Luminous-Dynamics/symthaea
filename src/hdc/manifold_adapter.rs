// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Strict state-space/manifold adapter over Symthaea's existing HDC hypersphere math.
//!
//! The wrapped [`HypersphereOps`] implementation remains the mathematical engine.
//! This module adds claim-bearing validation around it: exact dimensionality,
//! unit-norm states, base-point-bound tangent validity, cut-locus handling, and
//! deterministic profile identity. It does not reinterpret HDC similarity as a
//! physical metric or grant physical-world planning authority.

use std::f64::consts::PI;

use super::geometric_ops::HypersphereOps;
use symthaea_core::smooth_manifold::{
    GeodesicRelation, ManifoldError, SmoothManifold, TangentValidity,
};
use symthaea_core::state_space::{
    MetricSpace, StateSpace, StateSpaceError, StateSpaceProfile, StateValidity,
};

/// Strict adapter for the unit hypersphere `S^(d-1)` embedded in `R^d`.
#[derive(Clone, Debug)]
pub struct HdcHypersphereSpace {
    ambient_dimension: usize,
    tolerance: f64,
    profile: StateSpaceProfile,
}

impl HdcHypersphereSpace {
    /// Construct a strict unit-hypersphere space.
    ///
    /// `ambient_dimension` must be at least 2 so the resulting `S^(d-1)` is a
    /// smooth positive-dimensional sphere. `tolerance` must be finite and in
    /// `(0, 1)` and is bound into the exact state-space profile identity.
    pub fn new(ambient_dimension: usize, tolerance: f64) -> Result<Self, ManifoldError> {
        if ambient_dimension < 2 {
            return Err(ManifoldError::InvalidProfile {
                reason: "hypersphere ambient dimension must be at least 2".to_string(),
            });
        }
        if !tolerance.is_finite() || tolerance <= 0.0 || tolerance >= 1.0 {
            return Err(ManifoldError::InvalidProfile {
                reason: format!(
                    "hypersphere tolerance must be finite and in (0, 1), got {tolerance}"
                ),
            });
        }

        let mut parameters = Vec::with_capacity(16);
        parameters.extend_from_slice(&(ambient_dimension as u64).to_le_bytes());
        parameters.extend_from_slice(&tolerance.to_bits().to_le_bytes());

        Ok(Self {
            ambient_dimension,
            tolerance,
            profile: StateSpaceProfile::new(
                "hdc-unit-hypersphere",
                Some(ambient_dimension - 1),
                "great-circle-hypersphere-v1",
                parameters,
                Vec::new(),
            ),
        })
    }

    /// Ambient Euclidean dimension `d` for this `S^(d-1)`.
    pub fn ambient_dimension(&self) -> usize {
        self.ambient_dimension
    }

    /// Numerical validation tolerance bound into the profile identity.
    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    fn norm(&self, values: &[f64]) -> Option<f64> {
        let mut norm = 0.0_f64;
        for &value in values {
            norm = norm.hypot(value);
            if !norm.is_finite() {
                return None;
            }
        }
        Some(norm)
    }

    fn dot(&self, a: &[f64], b: &[f64]) -> Option<f64> {
        if a.len() != b.len() {
            return None;
        }
        let mut sum = 0.0_f64;
        for (&left, &right) in a.iter().zip(b) {
            let product = left * right;
            if !product.is_finite() {
                return None;
            }
            sum += product;
            if !sum.is_finite() {
                return None;
            }
        }
        Some(sum)
    }

    fn antipodal_residual(&self, from: &[f64], to: &[f64]) -> Option<f64> {
        if from.len() != to.len() {
            return None;
        }
        let mut residual = 0.0_f64;
        for (&a, &b) in from.iter().zip(to) {
            residual = residual.hypot(a + b);
            if !residual.is_finite() {
                return None;
            }
        }
        Some(residual)
    }

    fn deterministic_antipodal_interpolation(&self, from: &[f64], t: f64) -> Vec<f64> {
        // Pick the coordinate least aligned with `from`, then Gram-Schmidt it
        // into the tangent space. This is a deterministic convention for an
        // intrinsically non-unique geodesic, not a uniqueness claim.
        let axis = from
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
            .map(|(index, _)| index)
            .unwrap_or(0);

        let projection = from[axis];
        let mut tangent = Vec::with_capacity(from.len());
        for (index, &coordinate) in from.iter().enumerate() {
            let basis = if index == axis { 1.0 } else { 0.0 };
            tangent.push(basis - projection * coordinate);
        }

        let tangent_norm = self.norm(&tangent).unwrap_or(0.0);
        if tangent_norm > 0.0 {
            for value in &mut tangent {
                *value /= tangent_norm;
            }
        }

        let angle = PI * t;
        from.iter()
            .zip(tangent.iter())
            .map(|(&point, &direction)| angle.cos() * point + angle.sin() * direction)
            .collect()
    }
}

impl StateSpace for HdcHypersphereSpace {
    type State = Vec<f64>;

    fn profile(&self) -> &StateSpaceProfile {
        &self.profile
    }

    fn validate_state(&self, state: &Self::State) -> StateValidity {
        if state.len() != self.ambient_dimension {
            return StateValidity::DimensionMismatch {
                expected: self.ambient_dimension,
                actual: state.len(),
            };
        }
        for (coordinate, value) in state.iter().enumerate() {
            if !value.is_finite() {
                return StateValidity::NonFinite { coordinate };
            }
        }

        let Some(norm) = self.norm(state) else {
            return StateValidity::Unknown {
                reason: "unit-norm validation overflowed".to_string(),
            };
        };
        let deviation = (norm - 1.0).abs();
        if deviation > self.tolerance {
            return StateValidity::ConstraintViolation {
                reason: format!(
                    "state is not unit norm under profile tolerance: norm={norm}, deviation={deviation}, tolerance={}",
                    self.tolerance
                ),
            };
        }

        StateValidity::Valid
    }

    fn interpolate(
        &self,
        from: &Self::State,
        to: &Self::State,
        t: f64,
    ) -> Result<Self::State, StateSpaceError> {
        if !t.is_finite() || !(0.0..=1.0).contains(&t) {
            return Err(StateSpaceError::InvalidInterpolationParameter { t });
        }
        self.require_valid("from", from)?;
        self.require_valid("to", to)?;

        if t <= 0.0 {
            return Ok(from.clone());
        }
        if t >= 1.0 {
            return Ok(to.clone());
        }

        let residual = self.antipodal_residual(from, to).ok_or(
            StateSpaceError::NonFiniteComputation {
                operation: "hypersphere antipodal residual",
            },
        )?;

        // Exact-within-profile antipodes have infinitely many shortest
        // geodesics. StateSpace interpolation may still use one deterministic
        // convention, but near-cut-locus cases outside that equivalence band are
        // rejected rather than delegated to numerically fragile SLERP.
        let result = if residual <= self.tolerance {
            self.deterministic_antipodal_interpolation(from, t)
        } else if residual <= 10.0 * self.tolerance {
            return Err(StateSpaceError::InvalidState {
                role: "interpolation endpoint pair",
                validity: StateValidity::Unknown {
                    reason: "endpoint pair is numerically degenerate near the antipodal cut locus"
                        .to_string(),
                },
            });
        } else {
            HypersphereOps::slerp(from, to, t)
        };

        let validity = self.validate_state(&result);
        if !validity.is_valid() {
            return Err(StateSpaceError::InvalidState {
                role: "interpolated",
                validity,
            });
        }
        Ok(result)
    }
}

impl MetricSpace for HdcHypersphereSpace {
    fn distance(
        &self,
        a: &Self::State,
        b: &Self::State,
    ) -> Result<f64, StateSpaceError> {
        self.require_valid("a", a)?;
        self.require_valid("b", b)?;
        let distance = HypersphereOps::geodesic_distance(a, b);
        if !distance.is_finite() {
            return Err(StateSpaceError::NonFiniteComputation {
                operation: "hypersphere geodesic distance",
            });
        }
        Ok(distance)
    }
}

impl SmoothManifold for HdcHypersphereSpace {
    type Tangent = Vec<f64>;

    fn validate_tangent(
        &self,
        point: &Self::State,
        tangent: &Self::Tangent,
    ) -> TangentValidity {
        let point_validity = self.validate_state(point);
        if !point_validity.is_valid() {
            return TangentValidity::BaseStateInvalid(point_validity);
        }
        if tangent.len() != self.ambient_dimension {
            return TangentValidity::DimensionMismatch {
                expected: self.ambient_dimension,
                actual: tangent.len(),
            };
        }
        for (coordinate, value) in tangent.iter().enumerate() {
            if !value.is_finite() {
                return TangentValidity::NonFinite { coordinate };
            }
        }

        let Some(tangent_norm) = self.norm(tangent) else {
            return TangentValidity::Unknown {
                reason: "tangent norm overflowed".to_string(),
            };
        };
        let Some(inner) = self.dot(point, tangent) else {
            return TangentValidity::Unknown {
                reason: "tangent orthogonality calculation overflowed".to_string(),
            };
        };
        let allowed = self.tolerance * tangent_norm.max(1.0);
        if inner.abs() > allowed {
            return TangentValidity::ConstraintViolation {
                reason: format!(
                    "tangent is not orthogonal to base point: |dot|={}, allowed={allowed}",
                    inner.abs()
                ),
            };
        }

        TangentValidity::Valid
    }

    fn geodesic_relation(
        &self,
        from: &Self::State,
        to: &Self::State,
    ) -> Result<GeodesicRelation, ManifoldError> {
        self.require_valid("from", from)?;
        self.require_valid("to", to)?;
        let residual = self.antipodal_residual(from, to).ok_or_else(|| {
            ManifoldError::NumericallyDegenerate {
                operation: "hypersphere cut-locus classification",
                reason: "antipodal residual is non-finite".to_string(),
            }
        })?;

        if residual <= self.tolerance {
            Ok(GeodesicRelation::NonUnique)
        } else if residual <= 10.0 * self.tolerance {
            Ok(GeodesicRelation::NumericallyDegenerate)
        } else {
            Ok(GeodesicRelation::Unique)
        }
    }

    fn exp_map(
        &self,
        point: &Self::State,
        tangent: &Self::Tangent,
    ) -> Result<Self::State, ManifoldError> {
        self.require_valid("point", point)?;
        self.require_valid_tangent("exp_map", point, tangent)?;
        let result = HypersphereOps::exp_map(point, tangent);
        let validity = self.validate_state(&result);
        if !validity.is_valid() {
            return Err(ManifoldError::StateSpace(StateSpaceError::InvalidState {
                role: "exp_map result",
                validity,
            }));
        }
        Ok(result)
    }

    fn log_map(
        &self,
        from: &Self::State,
        to: &Self::State,
    ) -> Result<Self::Tangent, ManifoldError> {
        match self.geodesic_relation(from, to)? {
            GeodesicRelation::Unique => {}
            GeodesicRelation::NonUnique => return Err(ManifoldError::NonUniqueGeodesic),
            GeodesicRelation::NumericallyDegenerate => {
                return Err(ManifoldError::NumericallyDegenerate {
                    operation: "hypersphere log map",
                    reason: "endpoint is too close to the antipodal cut locus".to_string(),
                });
            }
        }

        let tangent = HypersphereOps::log_map(from, to);
        self.require_valid_tangent("log_map result", from, &tangent)?;
        Ok(tangent)
    }

    fn inner_product(
        &self,
        point: &Self::State,
        a: &Self::Tangent,
        b: &Self::Tangent,
    ) -> Result<f64, ManifoldError> {
        self.require_valid("point", point)?;
        self.require_valid_tangent("a", point, a)?;
        self.require_valid_tangent("b", point, b)?;
        self.dot(a, b).ok_or(ManifoldError::NonFiniteComputation {
            operation: "hypersphere tangent inner product",
        })
    }

    fn parallel_transport(
        &self,
        from: &Self::State,
        to: &Self::State,
        tangent: &Self::Tangent,
    ) -> Result<Self::Tangent, ManifoldError> {
        self.require_valid_tangent("source", from, tangent)?;
        match self.geodesic_relation(from, to)? {
            GeodesicRelation::Unique => {}
            GeodesicRelation::NonUnique => return Err(ManifoldError::NonUniqueGeodesic),
            GeodesicRelation::NumericallyDegenerate => {
                return Err(ManifoldError::NumericallyDegenerate {
                    operation: "hypersphere parallel transport",
                    reason: "endpoint is too close to the antipodal cut locus".to_string(),
                });
            }
        }

        let transported = HypersphereOps::parallel_transport(from, to, tangent);
        self.require_valid_tangent("transport result", to, &transported)?;
        Ok(transported)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    fn assert_vector_close(actual: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());
        for (&a, &e) in actual.iter().zip(expected) {
            assert_close(a, e, tolerance);
        }
    }

    #[test]
    fn quarter_circle_distance_reuses_existing_hypersphere_math() {
        let space = HdcHypersphereSpace::new(3, 1e-10).unwrap();
        let x = vec![1.0, 0.0, 0.0];
        let y = vec![0.0, 1.0, 0.0];
        assert_close(space.distance(&x, &y).unwrap(), PI / 2.0, 1e-12);
    }

    #[test]
    fn strict_adapter_rejects_non_unit_and_non_finite_states() {
        let space = HdcHypersphereSpace::new(3, 1e-10).unwrap();
        assert!(matches!(
            space.validate_state(&vec![2.0, 0.0, 0.0]),
            StateValidity::ConstraintViolation { .. }
        ));
        assert_eq!(
            space.validate_state(&vec![1.0, f64::NAN, 0.0]),
            StateValidity::NonFinite { coordinate: 1 }
        );
    }

    #[test]
    fn slerp_midpoint_and_endpoints_are_valid() {
        let space = HdcHypersphereSpace::new(3, 1e-10).unwrap();
        let x = vec![1.0, 0.0, 0.0];
        let y = vec![0.0, 1.0, 0.0];
        assert_eq!(space.interpolate(&x, &y, 0.0).unwrap(), x);
        assert_eq!(space.interpolate(&x, &y, 1.0).unwrap(), y);
        let midpoint = space.interpolate(&x, &y, 0.5).unwrap();
        let half = 0.5_f64.sqrt();
        assert_vector_close(&midpoint, &[half, half, 0.0], 1e-12);
        assert!(space.validate_state(&midpoint).is_valid());
    }

    #[test]
    fn log_exp_round_trip_holds_away_from_cut_locus() {
        let space = HdcHypersphereSpace::new(3, 1e-10).unwrap();
        let x = vec![1.0, 0.0, 0.0];
        let y = vec![0.0, 1.0, 0.0];
        let tangent = space.log_map(&x, &y).unwrap();
        let recovered = space.exp_map(&x, &tangent).unwrap();
        assert_vector_close(&recovered, &y, 1e-10);
    }

    #[test]
    fn tangent_must_be_orthogonal_to_exact_base_point() {
        let space = HdcHypersphereSpace::new(3, 1e-10).unwrap();
        let x = vec![1.0, 0.0, 0.0];
        assert!(space
            .validate_tangent(&x, &vec![0.0, 2.0, 0.0])
            .is_valid());
        assert!(matches!(
            space.validate_tangent(&x, &vec![1.0, 0.0, 0.0]),
            TangentValidity::ConstraintViolation { .. }
        ));
    }

    #[test]
    fn parallel_transport_preserves_transverse_tangent_fixture() {
        let space = HdcHypersphereSpace::new(3, 1e-9).unwrap();
        let x = vec![1.0, 0.0, 0.0];
        let y = vec![0.0, 1.0, 0.0];
        let tangent = vec![0.0, 0.0, 1.0];
        let transported = space.parallel_transport(&x, &y, &tangent).unwrap();
        assert_vector_close(&transported, &tangent, 1e-9);
        assert!(space.validate_tangent(&y, &transported).is_valid());
    }

    #[test]
    fn antipodal_cut_locus_is_explicitly_non_unique() {
        let space = HdcHypersphereSpace::new(3, 1e-10).unwrap();
        let x = vec![1.0, 0.0, 0.0];
        let minus_x = vec![-1.0, 0.0, 0.0];
        assert_eq!(
            space.geodesic_relation(&x, &minus_x).unwrap(),
            GeodesicRelation::NonUnique
        );
        assert_eq!(
            space.log_map(&x, &minus_x),
            Err(ManifoldError::NonUniqueGeodesic)
        );
    }

    #[test]
    fn antipodal_interpolation_uses_deterministic_convention_without_claiming_uniqueness() {
        let space = HdcHypersphereSpace::new(3, 1e-10).unwrap();
        let x = vec![1.0, 0.0, 0.0];
        let minus_x = vec![-1.0, 0.0, 0.0];
        let midpoint = space.interpolate(&x, &minus_x, 0.5).unwrap();
        assert!(space.validate_state(&midpoint).is_valid());
        assert_close(space.distance(&x, &midpoint).unwrap(), PI / 2.0, 1e-10);
    }

    #[test]
    fn near_antipodal_interpolation_fails_closed_in_degenerate_band() {
        let tolerance = 1e-6;
        let space = HdcHypersphereSpace::new(3, tolerance).unwrap();
        let x = vec![1.0, 0.0, 0.0];
        let epsilon = 5e-6_f64;
        let near_minus_x = vec![-epsilon.cos(), epsilon.sin(), 0.0];
        assert_eq!(
            space.geodesic_relation(&x, &near_minus_x).unwrap(),
            GeodesicRelation::NumericallyDegenerate
        );
        assert!(matches!(
            space.interpolate(&x, &near_minus_x, 0.5),
            Err(StateSpaceError::InvalidState {
                validity: StateValidity::Unknown { .. },
                ..
            })
        ));
    }

    #[test]
    fn tolerance_is_bound_into_profile_identity() {
        let strict = HdcHypersphereSpace::new(3, 1e-10).unwrap();
        let loose = HdcHypersphereSpace::new(3, 1e-8).unwrap();
        assert_ne!(strict.profile().identity(), loose.profile().identity());
    }

    #[test]
    fn invalid_profile_is_rejected() {
        assert!(matches!(
            HdcHypersphereSpace::new(1, 1e-10),
            Err(ManifoldError::InvalidProfile { .. })
        ));
        assert!(matches!(
            HdcHypersphereSpace::new(3, f64::NAN),
            Err(ManifoldError::InvalidProfile { .. })
        ));
    }
}
