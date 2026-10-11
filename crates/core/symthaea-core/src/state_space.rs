// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Planner-neutral state-space primitives.
//!
//! This module intentionally separates mathematical state-space feasibility from
//! physical-world claims. A valid state or interpolated trajectory is not, by
//! itself, evidence of collision clearance, dynamic realizability, safety, or
//! actuation authority.

use blake3::Hasher;
use thiserror::Error;

/// Result of validating a state against one exact state-space profile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StateValidity {
    /// The state satisfies every validation rule evaluated by the space.
    Valid,
    /// A coordinate lies outside an explicit bound.
    OutOfBounds { coordinate: usize },
    /// A coordinate is NaN or infinite.
    NonFinite { coordinate: usize },
    /// The state has a different dimension than the space expects.
    DimensionMismatch { expected: usize, actual: usize },
    /// A declared constraint is violated.
    ConstraintViolation { reason: String },
    /// Validation could not establish either validity or invalidity.
    Unknown { reason: String },
    /// A component of a product state is invalid or unknown.
    ComponentInvalid {
        component: usize,
        validity: Box<StateValidity>,
    },
}

impl StateValidity {
    /// Returns true only for an explicitly validated state.
    pub fn is_valid(&self) -> bool {
        matches!(self, Self::Valid)
    }
}

/// Fail-closed errors produced by state-space operations.
#[derive(Debug, Error, PartialEq)]
pub enum StateSpaceError {
    /// Interpolation is defined only for finite `t` in the closed unit interval.
    #[error("interpolation parameter must be finite and in [0, 1], got {t}")]
    InvalidInterpolationParameter { t: f64 },
    /// An input state was invalid or could not be established as valid.
    #[error("{role} state is not valid: {validity:?}")]
    InvalidState {
        role: &'static str,
        validity: StateValidity,
    },
    /// A derived numeric result overflowed or became non-finite.
    #[error("{operation} produced a non-finite result")]
    NonFiniteComputation { operation: &'static str },
    /// Product-space metric weights must be finite and strictly positive.
    #[error("metric weight for component {component} must be finite and > 0, got {weight}")]
    InvalidMetricWeight { component: usize, weight: f64 },
}

/// Identity-bearing description of one exact state-space/metric profile.
///
/// `identity()` is deterministic over the fields below and recursively binds
/// component profiles. The profile deliberately contains opaque parameter
/// bytes so future spaces can bind exact numerical policies without widening
/// this common type for every new manifold.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateSpaceProfile {
    kind: String,
    intrinsic_dimension: Option<usize>,
    metric_profile: String,
    parameters: Vec<u8>,
    components: Vec<StateSpaceProfile>,
}

impl StateSpaceProfile {
    /// Construct an exact state-space profile for a core or downstream space.
    ///
    /// `kind` and `metric_profile` should be stable/versioned identifiers.
    /// `parameters` are opaque canonical bytes for any numerical or policy
    /// choices that affect semantics, while `components` preserve ordered
    /// identities for compound spaces.
    pub fn new(
        kind: impl Into<String>,
        intrinsic_dimension: Option<usize>,
        metric_profile: impl Into<String>,
        parameters: Vec<u8>,
        components: Vec<StateSpaceProfile>,
    ) -> Self {
        Self {
            kind: kind.into(),
            intrinsic_dimension,
            metric_profile: metric_profile.into(),
            parameters,
            components,
        }
    }

    /// Stable kind identifier, such as `euclidean` or `product`.
    pub fn kind(&self) -> &str {
        &self.kind
    }

    /// Intrinsic dimension when that concept is meaningful for the space.
    pub fn intrinsic_dimension(&self) -> Option<usize> {
        self.intrinsic_dimension
    }

    /// Metric/interpolation policy identifier bound by this profile.
    pub fn metric_profile(&self) -> &str {
        &self.metric_profile
    }

    /// Ordered component profiles for compound spaces.
    pub fn components(&self) -> &[StateSpaceProfile] {
        &self.components
    }

    /// Deterministic BLAKE3 identity for this exact profile.
    pub fn identity(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(b"symthaea-state-space-profile-v1\0");
        self.update_hash(&mut hasher);
        *hasher.finalize().as_bytes()
    }

    fn update_hash(&self, hasher: &mut Hasher) {
        update_len_prefixed(hasher, self.kind.as_bytes());
        match self.intrinsic_dimension {
            Some(dimension) => {
                hasher.update(&[1]);
                hasher.update(&(dimension as u64).to_le_bytes());
            }
            None => {
                hasher.update(&[0]);
            }
        }
        update_len_prefixed(hasher, self.metric_profile.as_bytes());
        update_len_prefixed(hasher, &self.parameters);
        hasher.update(&(self.components.len() as u64).to_le_bytes());
        for component in &self.components {
            component.update_hash(hasher);
        }
    }
}

fn update_len_prefixed(hasher: &mut Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

/// A planner-neutral mathematical state space.
///
/// Implementations must not silently promote `Unknown` validation to `Valid`.
pub trait StateSpace {
    /// Concrete state representation used by this space.
    type State: Clone;

    /// Exact identity-bearing profile for this state space.
    fn profile(&self) -> &StateSpaceProfile;

    /// Intrinsic dimension when meaningful; discrete/hybrid spaces may return `None`.
    fn intrinsic_dimension(&self) -> Option<usize> {
        self.profile().intrinsic_dimension()
    }

    /// Validate one state under this exact state-space profile.
    fn validate_state(&self, state: &Self::State) -> StateValidity;

    /// Interpolate between two states for `t` in `[0, 1]`.
    fn interpolate(
        &self,
        from: &Self::State,
        to: &Self::State,
        t: f64,
    ) -> Result<Self::State, StateSpaceError>;

    /// Convert explicit state validity into a fail-closed operation precondition.
    fn require_valid(
        &self,
        role: &'static str,
        state: &Self::State,
    ) -> Result<(), StateSpaceError> {
        let validity = self.validate_state(state);
        if validity.is_valid() {
            Ok(())
        } else {
            Err(StateSpaceError::InvalidState { role, validity })
        }
    }
}

/// A state space with a scalar metric distance.
pub trait MetricSpace: StateSpace {
    /// Metric distance between two explicitly valid states.
    fn distance(
        &self,
        a: &Self::State,
        b: &Self::State,
    ) -> Result<f64, StateSpaceError>;
}

/// Euclidean `R^n` with the ordinary L2 metric.
#[derive(Clone, Debug)]
pub struct EuclideanSpace {
    dimension: usize,
    profile: StateSpaceProfile,
}

impl EuclideanSpace {
    /// Construct `R^dimension`.
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            profile: StateSpaceProfile::new(
                "euclidean",
                Some(dimension),
                "euclidean-l2-v1",
                Vec::new(),
                Vec::new(),
            ),
        }
    }

    /// Coordinate dimension of this Euclidean space.
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

impl StateSpace for EuclideanSpace {
    type State = Vec<f64>;

    fn profile(&self) -> &StateSpaceProfile {
        &self.profile
    }

    fn validate_state(&self, state: &Self::State) -> StateValidity {
        if state.len() != self.dimension {
            return StateValidity::DimensionMismatch {
                expected: self.dimension,
                actual: state.len(),
            };
        }
        for (coordinate, value) in state.iter().enumerate() {
            if !value.is_finite() {
                return StateValidity::NonFinite { coordinate };
            }
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

        let mut result = Vec::with_capacity(self.dimension);
        for (&a, &b) in from.iter().zip(to) {
            let value = (1.0 - t) * a + t * b;
            if !value.is_finite() {
                return Err(StateSpaceError::NonFiniteComputation {
                    operation: "euclidean interpolation",
                });
            }
            result.push(value);
        }
        Ok(result)
    }
}

impl MetricSpace for EuclideanSpace {
    fn distance(
        &self,
        a: &Self::State,
        b: &Self::State,
    ) -> Result<f64, StateSpaceError> {
        self.require_valid("a", a)?;
        self.require_valid("b", b)?;

        let mut distance = 0.0_f64;
        for (&left, &right) in a.iter().zip(b) {
            let delta = left - right;
            if !delta.is_finite() {
                return Err(StateSpaceError::NonFiniteComputation {
                    operation: "euclidean distance delta",
                });
            }
            distance = distance.hypot(delta);
            if !distance.is_finite() {
                return Err(StateSpaceError::NonFiniteComputation {
                    operation: "euclidean distance",
                });
            }
        }
        Ok(distance)
    }
}

/// Ordered product of two state spaces with an explicit weighted L2 product metric.
///
/// Tuple state representation preserves component semantics and ordering instead
/// of flattening heterogeneous components into an unlabeled numeric vector.
#[derive(Clone, Debug)]
pub struct ProductSpace<A, B> {
    first: A,
    second: B,
    first_weight: f64,
    second_weight: f64,
    profile: StateSpaceProfile,
}

impl<A, B> ProductSpace<A, B>
where
    A: StateSpace,
    B: StateSpace,
{
    /// Construct a product space with strictly positive finite component weights.
    pub fn new(
        first: A,
        second: B,
        first_weight: f64,
        second_weight: f64,
    ) -> Result<Self, StateSpaceError> {
        validate_weight(0, first_weight)?;
        validate_weight(1, second_weight)?;

        let intrinsic_dimension = match (first.intrinsic_dimension(), second.intrinsic_dimension()) {
            (Some(a), Some(b)) => a.checked_add(b),
            _ => None,
        };

        let mut parameters = Vec::with_capacity(16);
        parameters.extend_from_slice(&first_weight.to_bits().to_le_bytes());
        parameters.extend_from_slice(&second_weight.to_bits().to_le_bytes());
        let profile = StateSpaceProfile::new(
            "product",
            intrinsic_dimension,
            "weighted-l2-product-v1",
            parameters,
            vec![first.profile().clone(), second.profile().clone()],
        );

        Ok(Self {
            first,
            second,
            first_weight,
            second_weight,
            profile,
        })
    }

    /// First component space.
    pub fn first(&self) -> &A {
        &self.first
    }

    /// Second component space.
    pub fn second(&self) -> &B {
        &self.second
    }

    /// Positive metric weight of the first component.
    pub fn first_weight(&self) -> f64 {
        self.first_weight
    }

    /// Positive metric weight of the second component.
    pub fn second_weight(&self) -> f64 {
        self.second_weight
    }
}

fn validate_weight(component: usize, weight: f64) -> Result<(), StateSpaceError> {
    if weight.is_finite() && weight > 0.0 {
        Ok(())
    } else {
        Err(StateSpaceError::InvalidMetricWeight { component, weight })
    }
}

impl<A, B> StateSpace for ProductSpace<A, B>
where
    A: StateSpace,
    B: StateSpace,
{
    type State = (A::State, B::State);

    fn profile(&self) -> &StateSpaceProfile {
        &self.profile
    }

    fn validate_state(&self, state: &Self::State) -> StateValidity {
        let first = self.first.validate_state(&state.0);
        if !first.is_valid() {
            return StateValidity::ComponentInvalid {
                component: 0,
                validity: Box::new(first),
            };
        }

        let second = self.second.validate_state(&state.1);
        if !second.is_valid() {
            return StateValidity::ComponentInvalid {
                component: 1,
                validity: Box::new(second),
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
        self.require_valid("from", from)?;
        self.require_valid("to", to)?;
        Ok((
            self.first.interpolate(&from.0, &to.0, t)?,
            self.second.interpolate(&from.1, &to.1, t)?,
        ))
    }
}

impl<A, B> MetricSpace for ProductSpace<A, B>
where
    A: MetricSpace,
    B: MetricSpace,
{
    fn distance(
        &self,
        a: &Self::State,
        b: &Self::State,
    ) -> Result<f64, StateSpaceError> {
        self.require_valid("a", a)?;
        self.require_valid("b", b)?;
        let first = self.first.distance(&a.0, &b.0)? * self.first_weight.sqrt();
        let second = self.second.distance(&a.1, &b.1)? * self.second_weight.sqrt();
        if !first.is_finite() || !second.is_finite() {
            return Err(StateSpaceError::NonFiniteComputation {
                operation: "product metric scaling",
            });
        }
        let distance = first.hypot(second);
        if !distance.is_finite() {
            return Err(StateSpaceError::NonFiniteComputation {
                operation: "product distance",
            });
        }
        Ok(distance)
    }
}

/// Neutral waypoint trajectory.
///
/// This type intentionally carries no `collision_free`, `dynamically_reachable`,
/// or `safe` flag. Those stronger claims require independent downstream evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct Trajectory<S> {
    states: Vec<S>,
    interpolation_profile: String,
    time_parameterization: Option<String>,
}

impl<S> Trajectory<S> {
    /// Construct a waypoint trajectory under an explicit interpolation profile.
    pub fn new(states: Vec<S>, interpolation_profile: impl Into<String>) -> Self {
        Self {
            states,
            interpolation_profile: interpolation_profile.into(),
            time_parameterization: None,
        }
    }

    /// Attach a named time-parameterization profile without claiming dynamics validity.
    pub fn with_time_parameterization(mut self, profile: impl Into<String>) -> Self {
        self.time_parameterization = Some(profile.into());
        self
    }

    /// Ordered waypoint states.
    pub fn states(&self) -> &[S] {
        &self.states
    }

    /// Named interpolation policy for segments between waypoints.
    pub fn interpolation_profile(&self) -> &str {
        &self.interpolation_profile
    }

    /// Optional named time-parameterization profile.
    pub fn time_parameterization(&self) -> Option<&str> {
        self.time_parameterization.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= 1e-12 * scale,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn euclidean_distance_matches_closed_form_across_dimensions() {
        for dimension in [1, 3, 4, 16] {
            let space = EuclideanSpace::new(dimension);
            let origin = vec![0.0; dimension];
            let point = vec![1.0; dimension];
            assert_close(
                space.distance(&origin, &point).unwrap(),
                (dimension as f64).sqrt(),
            );
        }
    }

    #[test]
    fn euclidean_interpolation_preserves_endpoints_and_midpoint() {
        let space = EuclideanSpace::new(3);
        let from = vec![1.0, -2.0, 4.0];
        let to = vec![5.0, 6.0, -2.0];
        assert_eq!(space.interpolate(&from, &to, 0.0).unwrap(), from);
        assert_eq!(space.interpolate(&from, &to, 1.0).unwrap(), to);
        assert_eq!(
            space.interpolate(&from, &to, 0.5).unwrap(),
            vec![3.0, 2.0, 1.0]
        );
    }

    #[test]
    fn euclidean_rejects_dimension_mismatch_and_non_finite_state() {
        let space = EuclideanSpace::new(3);
        assert_eq!(
            space.validate_state(&vec![0.0, 1.0]),
            StateValidity::DimensionMismatch {
                expected: 3,
                actual: 2
            }
        );
        assert_eq!(
            space.validate_state(&vec![0.0, f64::NAN, 1.0]),
            StateValidity::NonFinite { coordinate: 1 }
        );
        assert_eq!(
            space.validate_state(&vec![0.0, f64::INFINITY, 1.0]),
            StateValidity::NonFinite { coordinate: 1 }
        );
    }

    #[test]
    fn invalid_interpolation_parameter_fails_closed() {
        let space = EuclideanSpace::new(1);
        let a = vec![0.0];
        let b = vec![1.0];
        for t in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
            assert!(matches!(
                space.interpolate(&a, &b, t),
                Err(StateSpaceError::InvalidInterpolationParameter { .. })
            ));
        }
    }

    #[test]
    fn euclidean_metric_satisfies_deterministic_triangle_fixture() {
        let space = EuclideanSpace::new(4);
        let a = vec![0.0, 0.0, 0.0, 0.0];
        let b = vec![1.0, -2.0, 3.0, -4.0];
        let c = vec![-2.0, 1.0, 4.0, 2.0];
        let ab = space.distance(&a, &b).unwrap();
        let bc = space.distance(&b, &c).unwrap();
        let ac = space.distance(&a, &c).unwrap();
        assert!(ab >= 0.0);
        assert_close(ab, space.distance(&b, &a).unwrap());
        assert!(ac <= ab + bc + 1e-12);
    }

    #[test]
    fn product_space_preserves_component_order_and_weighted_metric() {
        let space = ProductSpace::new(
            EuclideanSpace::new(2),
            EuclideanSpace::new(1),
            4.0,
            9.0,
        )
        .unwrap();
        let a = (vec![0.0, 0.0], vec![0.0]);
        let b = (vec![3.0, 4.0], vec![2.0]);
        assert_close(space.distance(&a, &b).unwrap(), 136.0_f64.sqrt());
        assert_eq!(space.intrinsic_dimension(), Some(3));
        assert_eq!(space.profile().components().len(), 2);
        assert_eq!(
            space.profile().components()[0].intrinsic_dimension(),
            Some(2)
        );
        assert_eq!(
            space.profile().components()[1].intrinsic_dimension(),
            Some(1)
        );
    }

    #[test]
    fn product_profile_identity_binds_weight_and_component_order() {
        let a = ProductSpace::new(
            EuclideanSpace::new(2),
            EuclideanSpace::new(3),
            1.0,
            2.0,
        )
        .unwrap();
        let changed_weight = ProductSpace::new(
            EuclideanSpace::new(2),
            EuclideanSpace::new(3),
            1.0,
            3.0,
        )
        .unwrap();
        let swapped = ProductSpace::new(
            EuclideanSpace::new(3),
            EuclideanSpace::new(2),
            1.0,
            2.0,
        )
        .unwrap();
        assert_ne!(a.profile().identity(), changed_weight.profile().identity());
        assert_ne!(a.profile().identity(), swapped.profile().identity());
    }

    #[test]
    fn public_profile_constructor_binds_extension_parameters() {
        let a = StateSpaceProfile::new(
            "test-extension",
            Some(2),
            "test-metric-v1",
            vec![1, 2, 3],
            Vec::new(),
        );
        let b = StateSpaceProfile::new(
            "test-extension",
            Some(2),
            "test-metric-v1",
            vec![1, 2, 4],
            Vec::new(),
        );
        assert_ne!(a.identity(), b.identity());
    }

    #[test]
    fn invalid_product_weight_is_rejected() {
        for bad_weight in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                ProductSpace::new(
                    EuclideanSpace::new(1),
                    EuclideanSpace::new(1),
                    bad_weight,
                    1.0,
                ),
                Err(StateSpaceError::InvalidMetricWeight { component: 0, .. })
            ));
        }
    }

    #[test]
    fn unknown_validity_never_becomes_valid() {
        assert!(!StateValidity::Unknown {
            reason: "not evaluated".to_string()
        }
        .is_valid());
    }

    #[test]
    fn trajectory_is_only_a_waypoint_container() {
        let trajectory = Trajectory::new(vec![vec![0.0], vec![1.0]], "euclidean-linear-v1")
            .with_time_parameterization("external-clock-v1");
        assert_eq!(trajectory.states().len(), 2);
        assert_eq!(trajectory.interpolation_profile(), "euclidean-linear-v1");
        assert_eq!(
            trajectory.time_parameterization(),
            Some("external-clock-v1")
        );
    }
}
