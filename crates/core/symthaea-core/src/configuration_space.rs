// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Lie-group and articulated configuration-space primitives.
//!
//! These types model mathematical configuration geometry only. They do not
//! establish collision clearance, inverse-kinematics feasibility, dynamics,
//! safety, or actuation authority.

use std::collections::HashSet;
use std::f64::consts::{PI, TAU};
use std::num::FpCategory;

use thiserror::Error;

use crate::state_space::{
    MetricSpace, StateSpace, StateSpaceError, StateSpaceProfile, StateValidity,
};

/// Default near-identity dot-product tolerance used by [`So3Space::new`] and
/// [`Se3Space::new`].
///
/// The value is not ambient hidden state: it is encoded into the resulting
/// [`StateSpaceProfile`] identity. Use the explicit numerical-policy constructors
/// to select another value.
pub const DEFAULT_SO3_NEAR_IDENTITY_DOT_TOLERANCE: f64 = 1e-12;

/// Group operations layered over a metric state space.
///
/// This trait intentionally does not claim every Lie group carries one unique
/// task-independent metric. Each implementation binds its metric policy in its
/// [`StateSpaceProfile`].
pub trait LieGroup: MetricSpace {
    /// Group identity.
    fn identity(&self) -> Self::State;

    /// Group composition `left * right`.
    fn compose(
        &self,
        left: &Self::State,
        right: &Self::State,
    ) -> Result<Self::State, StateSpaceError>;

    /// Group inverse.
    fn inverse(&self, state: &Self::State) -> Result<Self::State, StateSpaceError>;
}

/// Construction errors for configuration-space profiles.
#[derive(Debug, Error, PartialEq)]
pub enum ConfigurationSpaceError {
    /// Numerical tolerance must satisfy the named policy.
    #[error("invalid numerical tolerance {name}: {value}")]
    InvalidTolerance { name: &'static str, value: f64 },
    /// A metric weight must be finite and strictly positive.
    #[error("metric weight {name} must be finite and > 0, got {value}")]
    InvalidMetricWeight { name: String, value: f64 },
    /// A bounded dynamic coordinate must have finite strict bounds `min < max`.
    #[error("invalid dynamic bounds for {name}: [{min}, {max}]")]
    InvalidBounds { name: String, min: f64, max: f64 },
    /// Joint names must be non-empty and unique.
    #[error("invalid joint name: {reason}")]
    InvalidJointName { reason: String },
    /// A joint configuration space requires at least one declared factor.
    #[error("joint configuration space must contain at least one joint")]
    EmptyJointSpace,
    /// An underlying state-space construction failed.
    #[error(transparent)]
    StateSpace(#[from] StateSpaceError),
}

fn validate_positive(name: &'static str, value: f64) -> Result<(), ConfigurationSpaceError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(ConfigurationSpaceError::InvalidTolerance { name, value })
    }
}

fn wrap_angle(angle: f64) -> f64 {
    // Adding +0.0 canonicalizes an IEEE-754 negative zero without a floating
    // equality branch. The ordinary representative is [-pi, pi).
    ((angle + PI).rem_euclid(TAU) - PI) + 0.0
}

/// The circle `S^1`, represented by a finite angle in radians.
#[derive(Clone, Debug)]
pub struct CircleSpace {
    profile: StateSpaceProfile,
}

impl CircleSpace {
    /// Construct the shortest-angular-distance circle profile.
    pub fn new() -> Self {
        Self {
            profile: StateSpaceProfile::new(
                "circle-s1",
                Some(1),
                "shortest-angular-distance-v1",
                Vec::new(),
                Vec::new(),
            ),
        }
    }

    /// Canonical representative in `[-pi, pi)`.
    pub fn canonical_angle(angle: f64) -> Option<f64> {
        angle.is_finite().then(|| wrap_angle(angle))
    }
}

impl Default for CircleSpace {
    fn default() -> Self {
        Self::new()
    }
}

impl StateSpace for CircleSpace {
    type State = f64;

    fn profile(&self) -> &StateSpaceProfile {
        &self.profile
    }

    fn validate_state(&self, state: &Self::State) -> StateValidity {
        if state.is_finite() {
            StateValidity::Valid
        } else {
            StateValidity::NonFinite { coordinate: 0 }
        }
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
            return Ok(*from);
        }
        if t >= 1.0 {
            return Ok(*to);
        }
        let delta = wrap_angle(*to - *from);
        Ok(wrap_angle(*from + t * delta))
    }
}

impl MetricSpace for CircleSpace {
    fn distance(
        &self,
        a: &Self::State,
        b: &Self::State,
    ) -> Result<f64, StateSpaceError> {
        self.require_valid("a", a)?;
        self.require_valid("b", b)?;
        Ok(wrap_angle(*b - *a).abs())
    }
}

impl LieGroup for CircleSpace {
    fn identity(&self) -> Self::State {
        0.0
    }

    fn compose(
        &self,
        left: &Self::State,
        right: &Self::State,
    ) -> Result<Self::State, StateSpaceError> {
        self.require_valid("left", left)?;
        self.require_valid("right", right)?;
        Ok(wrap_angle(*left + *right))
    }

    fn inverse(&self, state: &Self::State) -> Result<Self::State, StateSpaceError> {
        self.require_valid("state", state)?;
        Ok(wrap_angle(-*state))
    }
}

fn quaternion_norm(q: &[f64; 4]) -> f64 {
    q.iter().fold(0.0_f64, |norm, value| norm.hypot(*value))
}

fn normalized_quaternion(q: &[f64; 4]) -> Option<[f64; 4]> {
    let norm = quaternion_norm(q);
    if !norm.is_finite() || norm <= 0.0 {
        return None;
    }
    Some([q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm])
}

fn quaternion_dot(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

fn quaternion_multiply(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    let [aw, ax, ay, az] = *a;
    let [bw, bx, by, bz] = *b;
    [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]
}

fn quaternion_conjugate(q: &[f64; 4]) -> [f64; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

fn quaternion_rotate(q: &[f64; 4], vector: &[f64; 3]) -> [f64; 3] {
    let pure = [0.0, vector[0], vector[1], vector[2]];
    let rotated = quaternion_multiply(&quaternion_multiply(q, &pure), &quaternion_conjugate(q));
    [rotated[1], rotated[2], rotated[3]]
}

fn representative_is_negative(q: &[f64; 4]) -> bool {
    for value in q {
        if matches!(value.classify(), FpCategory::Zero) {
            continue;
        }
        return value.is_sign_negative();
    }
    false
}

/// `SO(3)` represented by a w-first unit quaternion `[w, x, y, z]`.
///
/// `q` and `-q` are treated as the same physical orientation for metric and
/// interpolation semantics.
#[derive(Clone, Debug)]
pub struct So3Space {
    unit_norm_tolerance: f64,
    near_identity_dot_tolerance: f64,
    profile: StateSpaceProfile,
}

impl So3Space {
    /// Construct an `SO(3)` profile using the documented default interpolation policy.
    pub fn new(unit_norm_tolerance: f64) -> Result<Self, ConfigurationSpaceError> {
        Self::with_numerical_policy(
            unit_norm_tolerance,
            DEFAULT_SO3_NEAR_IDENTITY_DOT_TOLERANCE,
        )
    }

    /// Construct an `SO(3)` profile with all numerical thresholds explicit.
    pub fn with_numerical_policy(
        unit_norm_tolerance: f64,
        near_identity_dot_tolerance: f64,
    ) -> Result<Self, ConfigurationSpaceError> {
        validate_positive("so3_unit_norm_tolerance", unit_norm_tolerance)?;
        validate_positive(
            "so3_near_identity_dot_tolerance",
            near_identity_dot_tolerance,
        )?;
        if unit_norm_tolerance >= 1.0 {
            return Err(ConfigurationSpaceError::InvalidTolerance {
                name: "so3_unit_norm_tolerance",
                value: unit_norm_tolerance,
            });
        }
        if near_identity_dot_tolerance >= 1.0 {
            return Err(ConfigurationSpaceError::InvalidTolerance {
                name: "so3_near_identity_dot_tolerance",
                value: near_identity_dot_tolerance,
            });
        }

        let mut parameters = Vec::with_capacity(16);
        parameters.extend_from_slice(&unit_norm_tolerance.to_bits().to_le_bytes());
        parameters.extend_from_slice(&near_identity_dot_tolerance.to_bits().to_le_bytes());
        Ok(Self {
            unit_norm_tolerance,
            near_identity_dot_tolerance,
            profile: StateSpaceProfile::new(
                "so3-unit-quaternion-wxyz",
                Some(3),
                "so3-shortest-angle-v2",
                parameters,
                Vec::new(),
            ),
        })
    }

    /// Unit-norm validation tolerance bound into the profile identity.
    pub fn tolerance(&self) -> f64 {
        self.unit_norm_tolerance
    }

    /// Near-identity SLERP fallback threshold bound into the profile identity.
    pub fn near_identity_dot_tolerance(&self) -> f64 {
        self.near_identity_dot_tolerance
    }

    /// Return one deterministic representative of the `q ~ -q` equivalence class.
    pub fn canonicalize(&self, state: &[f64; 4]) -> Result<[f64; 4], StateSpaceError> {
        self.require_valid("state", state)?;
        let mut q = normalized_quaternion(state).ok_or(StateSpaceError::NonFiniteComputation {
            operation: "SO(3) canonicalization",
        })?;
        if representative_is_negative(&q) {
            for value in &mut q {
                *value = -*value;
            }
        }
        for value in &mut q {
            if matches!(value.classify(), FpCategory::Zero) {
                *value = 0.0;
            }
        }
        Ok(q)
    }

    fn normalized_valid(
        &self,
        role: &'static str,
        q: &[f64; 4],
    ) -> Result<[f64; 4], StateSpaceError> {
        self.require_valid(role, q)?;
        normalized_quaternion(q).ok_or(StateSpaceError::NonFiniteComputation {
            operation: "SO(3) normalization",
        })
    }
}

impl StateSpace for So3Space {
    type State = [f64; 4];

    fn profile(&self) -> &StateSpaceProfile {
        &self.profile
    }

    fn validate_state(&self, state: &Self::State) -> StateValidity {
        for (coordinate, value) in state.iter().enumerate() {
            if !value.is_finite() {
                return StateValidity::NonFinite { coordinate };
            }
        }
        let norm = quaternion_norm(state);
        if !norm.is_finite() {
            return StateValidity::Unknown {
                reason: "quaternion norm overflowed".to_string(),
            };
        }
        let deviation = (norm - 1.0).abs();
        if deviation > self.unit_norm_tolerance {
            return StateValidity::ConstraintViolation {
                reason: format!(
                    "quaternion is not unit norm: norm={norm}, deviation={deviation}, tolerance={}",
                    self.unit_norm_tolerance
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
        if t <= 0.0 {
            self.require_valid("from", from)?;
            return Ok(*from);
        }
        if t >= 1.0 {
            self.require_valid("to", to)?;
            return Ok(*to);
        }

        let a = self.normalized_valid("from", from)?;
        let mut b = self.normalized_valid("to", to)?;
        let mut dot = quaternion_dot(&a, &b);
        if dot < 0.0 {
            for value in &mut b {
                *value = -*value;
            }
            dot = -dot;
        }
        dot = dot.clamp(0.0, 1.0);

        let result = if 1.0 - dot <= self.near_identity_dot_tolerance {
            let mixed = [
                (1.0 - t) * a[0] + t * b[0],
                (1.0 - t) * a[1] + t * b[1],
                (1.0 - t) * a[2] + t * b[2],
                (1.0 - t) * a[3] + t * b[3],
            ];
            normalized_quaternion(&mixed).ok_or(StateSpaceError::NonFiniteComputation {
                operation: "SO(3) near-identity interpolation",
            })?
        } else {
            let theta = dot.acos();
            let sin_theta = theta.sin();
            if !sin_theta.is_finite() || sin_theta <= 0.0 {
                return Err(StateSpaceError::NonFiniteComputation {
                    operation: "SO(3) interpolation denominator",
                });
            }
            let left = ((1.0 - t) * theta).sin() / sin_theta;
            let right = (t * theta).sin() / sin_theta;
            let mixed = [
                left * a[0] + right * b[0],
                left * a[1] + right * b[1],
                left * a[2] + right * b[2],
                left * a[3] + right * b[3],
            ];
            normalized_quaternion(&mixed).ok_or(StateSpaceError::NonFiniteComputation {
                operation: "SO(3) interpolation",
            })?
        };

        Ok(result)
    }
}

impl MetricSpace for So3Space {
    fn distance(
        &self,
        a: &Self::State,
        b: &Self::State,
    ) -> Result<f64, StateSpaceError> {
        let a = self.normalized_valid("a", a)?;
        let b = self.normalized_valid("b", b)?;
        let dot = quaternion_dot(&a, &b).abs().clamp(0.0, 1.0);
        let angle = 2.0 * dot.acos();
        if angle.is_finite() {
            Ok(angle)
        } else {
            Err(StateSpaceError::NonFiniteComputation {
                operation: "SO(3) angular distance",
            })
        }
    }
}

impl LieGroup for So3Space {
    fn identity(&self) -> Self::State {
        [1.0, 0.0, 0.0, 0.0]
    }

    fn compose(
        &self,
        left: &Self::State,
        right: &Self::State,
    ) -> Result<Self::State, StateSpaceError> {
        let left = self.normalized_valid("left", left)?;
        let right = self.normalized_valid("right", right)?;
        normalized_quaternion(&quaternion_multiply(&left, &right)).ok_or(
            StateSpaceError::NonFiniteComputation {
                operation: "SO(3) composition",
            },
        )
    }

    fn inverse(&self, state: &Self::State) -> Result<Self::State, StateSpaceError> {
        let state = self.normalized_valid("state", state)?;
        Ok(quaternion_conjugate(&state))
    }
}

/// A rigid 3D pose using world translation and a w-first unit quaternion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RigidPose3 {
    /// Translation in the selected coordinate frame.
    pub translation: [f64; 3],
    /// Orientation `[w, x, y, z]`.
    pub rotation: [f64; 4],
}

/// `SE(3)` with an explicit task/profile-dependent weighted product metric.
///
/// No universal conversion between metres and radians is implied. Both metric
/// weights are explicit and bound into the profile identity.
#[derive(Clone, Debug)]
pub struct Se3Space {
    rotation: So3Space,
    translation_weight: f64,
    rotation_weight: f64,
    profile: StateSpaceProfile,
}

impl Se3Space {
    /// Construct a rigid-pose space using the documented default `SO(3)` interpolation policy.
    pub fn new(
        translation_weight: f64,
        rotation_weight: f64,
        quaternion_tolerance: f64,
    ) -> Result<Self, ConfigurationSpaceError> {
        Self::with_numerical_policy(
            translation_weight,
            rotation_weight,
            quaternion_tolerance,
            DEFAULT_SO3_NEAR_IDENTITY_DOT_TOLERANCE,
        )
    }

    /// Construct a rigid-pose space with every metric and numerical policy explicit.
    pub fn with_numerical_policy(
        translation_weight: f64,
        rotation_weight: f64,
        quaternion_tolerance: f64,
        near_identity_dot_tolerance: f64,
    ) -> Result<Self, ConfigurationSpaceError> {
        for (name, value) in [
            ("translation", translation_weight),
            ("rotation", rotation_weight),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(ConfigurationSpaceError::InvalidMetricWeight {
                    name: name.to_string(),
                    value,
                });
            }
        }
        let rotation = So3Space::with_numerical_policy(
            quaternion_tolerance,
            near_identity_dot_tolerance,
        )?;
        let mut parameters = Vec::with_capacity(16);
        parameters.extend_from_slice(&translation_weight.to_bits().to_le_bytes());
        parameters.extend_from_slice(&rotation_weight.to_bits().to_le_bytes());
        let profile = StateSpaceProfile::new(
            "se3-rigid-pose-wxyz",
            Some(6),
            "weighted-r3-so3-product-v2",
            parameters,
            vec![rotation.profile().clone()],
        );
        Ok(Self {
            rotation,
            translation_weight,
            rotation_weight,
            profile,
        })
    }

    /// Translation metric weight.
    pub fn translation_weight(&self) -> f64 {
        self.translation_weight
    }

    /// Rotation metric weight.
    pub fn rotation_weight(&self) -> f64 {
        self.rotation_weight
    }

    /// `SO(3)` factor used by this pose space.
    pub fn rotation_space(&self) -> &So3Space {
        &self.rotation
    }
}

impl StateSpace for Se3Space {
    type State = RigidPose3;

    fn profile(&self) -> &StateSpaceProfile {
        &self.profile
    }

    fn validate_state(&self, state: &Self::State) -> StateValidity {
        for (coordinate, value) in state.translation.iter().enumerate() {
            if !value.is_finite() {
                return StateValidity::NonFinite { coordinate };
            }
        }
        let rotation = self.rotation.validate_state(&state.rotation);
        if !rotation.is_valid() {
            return StateValidity::ComponentInvalid {
                component: 1,
                validity: Box::new(rotation),
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
            return Ok(*from);
        }
        if t >= 1.0 {
            return Ok(*to);
        }
        let mut translation = [0.0; 3];
        for index in 0..3 {
            translation[index] =
                (1.0 - t) * from.translation[index] + t * to.translation[index];
            if !translation[index].is_finite() {
                return Err(StateSpaceError::NonFiniteComputation {
                    operation: "SE(3) translation interpolation",
                });
            }
        }
        let rotation = self.rotation.interpolate(&from.rotation, &to.rotation, t)?;
        Ok(RigidPose3 {
            translation,
            rotation,
        })
    }
}

impl MetricSpace for Se3Space {
    fn distance(
        &self,
        a: &Self::State,
        b: &Self::State,
    ) -> Result<f64, StateSpaceError> {
        self.require_valid("a", a)?;
        self.require_valid("b", b)?;
        let mut translation = 0.0_f64;
        for index in 0..3 {
            translation = translation.hypot(a.translation[index] - b.translation[index]);
        }
        let rotation = self.rotation.distance(&a.rotation, &b.rotation)?;
        let scaled_translation = self.translation_weight.sqrt() * translation;
        let scaled_rotation = self.rotation_weight.sqrt() * rotation;
        let result = scaled_translation.hypot(scaled_rotation);
        if result.is_finite() {
            Ok(result)
        } else {
            Err(StateSpaceError::NonFiniteComputation {
                operation: "SE(3) weighted distance",
            })
        }
    }
}

impl LieGroup for Se3Space {
    fn identity(&self) -> Self::State {
        RigidPose3 {
            translation: [0.0; 3],
            rotation: self.rotation.identity(),
        }
    }

    fn compose(
        &self,
        left: &Self::State,
        right: &Self::State,
    ) -> Result<Self::State, StateSpaceError> {
        self.require_valid("left", left)?;
        self.require_valid("right", right)?;
        let left_rotation = self
            .rotation
            .normalized_valid("left rotation", &left.rotation)?;
        let rotated_translation = quaternion_rotate(&left_rotation, &right.translation);
        let translation = [
            left.translation[0] + rotated_translation[0],
            left.translation[1] + rotated_translation[1],
            left.translation[2] + rotated_translation[2],
        ];
        if translation.iter().any(|value| !value.is_finite()) {
            return Err(StateSpaceError::NonFiniteComputation {
                operation: "SE(3) composition translation",
            });
        }
        let rotation = self.rotation.compose(&left.rotation, &right.rotation)?;
        Ok(RigidPose3 {
            translation,
            rotation,
        })
    }

    fn inverse(&self, state: &Self::State) -> Result<Self::State, StateSpaceError> {
        self.require_valid("state", state)?;
        let rotation = self.rotation.inverse(&state.rotation)?;
        let negative = [
            -state.translation[0],
            -state.translation[1],
            -state.translation[2],
        ];
        let translation = quaternion_rotate(&rotation, &negative);
        if translation.iter().any(|value| !value.is_finite()) {
            return Err(StateSpaceError::NonFiniteComputation {
                operation: "SE(3) inverse translation",
            });
        }
        Ok(RigidPose3 {
            translation,
            rotation,
        })
    }
}

/// Topology/constraint semantics for one ordered articulated joint coordinate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JointKind {
    /// Continuous revolute coordinate with `S^1` topology.
    ContinuousRevolute,
    /// Bounded revolute coordinate in radians, with no wrap across the limits.
    BoundedRevolute { min: f64, max: f64 },
    /// Bounded prismatic coordinate in the model's declared length unit.
    Prismatic { min: f64, max: f64 },
    /// Fixed coordinate retained for semantic alignment but contributing zero intrinsic dimension.
    Fixed { value: f64 },
}

/// One ordered joint factor in a configuration space.
#[derive(Clone, Debug, PartialEq)]
pub struct JointSpec {
    /// Stable semantic joint name.
    pub name: String,
    /// Topology and bounds.
    pub kind: JointKind,
    /// Positive metric weight for non-fixed factors. Ignored for fixed factors.
    pub weight: f64,
}

impl JointSpec {
    /// Construct a continuous revolute joint.
    pub fn continuous_revolute(name: impl Into<String>, weight: f64) -> Self {
        Self {
            name: name.into(),
            kind: JointKind::ContinuousRevolute,
            weight,
        }
    }

    /// Construct a bounded revolute joint.
    pub fn bounded_revolute(
        name: impl Into<String>,
        min: f64,
        max: f64,
        weight: f64,
    ) -> Self {
        Self {
            name: name.into(),
            kind: JointKind::BoundedRevolute { min, max },
            weight,
        }
    }

    /// Construct a bounded prismatic joint.
    pub fn prismatic(name: impl Into<String>, min: f64, max: f64, weight: f64) -> Self {
        Self {
            name: name.into(),
            kind: JointKind::Prismatic { min, max },
            weight,
        }
    }

    /// Construct a fixed semantic coordinate.
    pub fn fixed(name: impl Into<String>, value: f64) -> Self {
        Self {
            name: name.into(),
            kind: JointKind::Fixed { value },
            weight: 0.0,
        }
    }
}

/// Dynamic ordered articulated configuration space.
#[derive(Clone, Debug)]
pub struct JointConfigurationSpace {
    specs: Vec<JointSpec>,
    fixed_tolerance: f64,
    profile: StateSpaceProfile,
}

impl JointConfigurationSpace {
    /// Construct a configuration space from exact ordered joint semantics.
    pub fn new(
        specs: Vec<JointSpec>,
        fixed_tolerance: f64,
    ) -> Result<Self, ConfigurationSpaceError> {
        if specs.is_empty() {
            return Err(ConfigurationSpaceError::EmptyJointSpace);
        }
        validate_positive("fixed_tolerance", fixed_tolerance)?;

        let mut names = HashSet::with_capacity(specs.len());
        let mut parameters = Vec::new();
        parameters.extend_from_slice(&fixed_tolerance.to_bits().to_le_bytes());
        parameters.extend_from_slice(&(specs.len() as u64).to_le_bytes());
        let mut intrinsic_dimension = 0usize;

        for spec in &specs {
            if spec.name.trim().is_empty() {
                return Err(ConfigurationSpaceError::InvalidJointName {
                    reason: "joint name must not be empty".to_string(),
                });
            }
            if !names.insert(spec.name.clone()) {
                return Err(ConfigurationSpaceError::InvalidJointName {
                    reason: format!("duplicate joint name {}", spec.name),
                });
            }

            let name_bytes = spec.name.as_bytes();
            parameters.extend_from_slice(&(name_bytes.len() as u64).to_le_bytes());
            parameters.extend_from_slice(name_bytes);

            match spec.kind {
                JointKind::ContinuousRevolute => {
                    intrinsic_dimension += 1;
                    parameters.push(0);
                    validate_joint_weight(spec)?;
                    parameters.extend_from_slice(&spec.weight.to_bits().to_le_bytes());
                }
                JointKind::BoundedRevolute { min, max } => {
                    intrinsic_dimension += 1;
                    parameters.push(1);
                    validate_dynamic_bounds(&spec.name, min, max)?;
                    validate_joint_weight(spec)?;
                    parameters.extend_from_slice(&min.to_bits().to_le_bytes());
                    parameters.extend_from_slice(&max.to_bits().to_le_bytes());
                    parameters.extend_from_slice(&spec.weight.to_bits().to_le_bytes());
                }
                JointKind::Prismatic { min, max } => {
                    intrinsic_dimension += 1;
                    parameters.push(2);
                    validate_dynamic_bounds(&spec.name, min, max)?;
                    validate_joint_weight(spec)?;
                    parameters.extend_from_slice(&min.to_bits().to_le_bytes());
                    parameters.extend_from_slice(&max.to_bits().to_le_bytes());
                    parameters.extend_from_slice(&spec.weight.to_bits().to_le_bytes());
                }
                JointKind::Fixed { value } => {
                    parameters.push(3);
                    if !value.is_finite() {
                        return Err(ConfigurationSpaceError::InvalidBounds {
                            name: spec.name.clone(),
                            min: value,
                            max: value,
                        });
                    }
                    // Fixed coordinates have no metric contribution, so their
                    // arbitrary public `weight` representation is deliberately
                    // omitted from semantic profile identity.
                    parameters.extend_from_slice(&value.to_bits().to_le_bytes());
                }
            }
        }

        Ok(Self {
            specs,
            fixed_tolerance,
            profile: StateSpaceProfile::new(
                "joint-configuration-space",
                Some(intrinsic_dimension),
                "weighted-joint-product-v2",
                parameters,
                Vec::new(),
            ),
        })
    }

    /// Ordered exact joint semantics.
    pub fn specs(&self) -> &[JointSpec] {
        &self.specs
    }

    /// Number of stored coordinates, including fixed semantic coordinates.
    pub fn coordinate_count(&self) -> usize {
        self.specs.len()
    }

    /// Fixed-coordinate comparison tolerance.
    pub fn fixed_tolerance(&self) -> f64 {
        self.fixed_tolerance
    }
}

fn validate_joint_weight(spec: &JointSpec) -> Result<(), ConfigurationSpaceError> {
    if spec.weight.is_finite() && spec.weight > 0.0 {
        Ok(())
    } else {
        Err(ConfigurationSpaceError::InvalidMetricWeight {
            name: spec.name.clone(),
            value: spec.weight,
        })
    }
}

fn validate_dynamic_bounds(
    name: &str,
    min: f64,
    max: f64,
) -> Result<(), ConfigurationSpaceError> {
    if min.is_finite() && max.is_finite() && min < max {
        Ok(())
    } else {
        Err(ConfigurationSpaceError::InvalidBounds {
            name: name.to_string(),
            min,
            max,
        })
    }
}

impl StateSpace for JointConfigurationSpace {
    type State = Vec<f64>;

    fn profile(&self) -> &StateSpaceProfile {
        &self.profile
    }

    fn validate_state(&self, state: &Self::State) -> StateValidity {
        if state.len() != self.specs.len() {
            return StateValidity::DimensionMismatch {
                expected: self.specs.len(),
                actual: state.len(),
            };
        }
        for (index, (value, spec)) in state.iter().zip(&self.specs).enumerate() {
            if !value.is_finite() {
                return StateValidity::NonFinite { coordinate: index };
            }
            match spec.kind {
                JointKind::ContinuousRevolute => {}
                JointKind::BoundedRevolute { min, max } | JointKind::Prismatic { min, max } => {
                    if *value < min || *value > max {
                        return StateValidity::OutOfBounds { coordinate: index };
                    }
                }
                JointKind::Fixed { value: expected } => {
                    if (*value - expected).abs() > self.fixed_tolerance {
                        return StateValidity::ConstraintViolation {
                            reason: format!(
                                "fixed joint {} expected {expected}, got {value}",
                                spec.name
                            ),
                        };
                    }
                }
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

        let mut result = Vec::with_capacity(self.specs.len());
        for ((left, right), spec) in from.iter().zip(to).zip(&self.specs) {
            let value = match spec.kind {
                JointKind::ContinuousRevolute => {
                    wrap_angle(*left + t * wrap_angle(*right - *left))
                }
                JointKind::BoundedRevolute { .. } | JointKind::Prismatic { .. } => {
                    (1.0 - t) * *left + t * *right
                }
                JointKind::Fixed { value } => value,
            };
            if !value.is_finite() {
                return Err(StateSpaceError::NonFiniteComputation {
                    operation: "joint configuration interpolation",
                });
            }
            result.push(value);
        }
        Ok(result)
    }
}

impl MetricSpace for JointConfigurationSpace {
    fn distance(
        &self,
        a: &Self::State,
        b: &Self::State,
    ) -> Result<f64, StateSpaceError> {
        self.require_valid("a", a)?;
        self.require_valid("b", b)?;
        let mut distance = 0.0_f64;
        for ((left, right), spec) in a.iter().zip(b).zip(&self.specs) {
            let contribution = match spec.kind {
                JointKind::ContinuousRevolute => {
                    spec.weight.sqrt() * wrap_angle(*right - *left)
                }
                JointKind::BoundedRevolute { .. } | JointKind::Prismatic { .. } => {
                    spec.weight.sqrt() * (*right - *left)
                }
                JointKind::Fixed { .. } => 0.0,
            };
            distance = distance.hypot(contribution);
            if !distance.is_finite() {
                return Err(StateSpaceError::NonFiniteComputation {
                    operation: "joint configuration distance",
                });
            }
        }
        Ok(distance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn circle_wraparound_uses_short_path() {
        let space = CircleSpace::new();
        let from = 179.0_f64.to_radians();
        let to = (-179.0_f64).to_radians();
        close(
            space.distance(&from, &to).unwrap(),
            2.0_f64.to_radians(),
            1e-12,
        );
        let midpoint = space.interpolate(&from, &to, 0.5).unwrap();
        close(midpoint.abs(), PI, 1e-12);
        assert!(!CircleSpace::canonical_angle(-0.0).unwrap().is_sign_negative());
    }

    #[test]
    fn so3_double_cover_and_signed_zero_canonicalize() {
        let space = So3Space::new(1e-10).unwrap();
        let q = [0.0, 1.0, 0.0, 0.0];
        let minus_q = [-0.0, -1.0, -0.0, -0.0];
        close(space.distance(&q, &minus_q).unwrap(), 0.0, 1e-12);
        assert_eq!(
            space.canonicalize(&q).unwrap(),
            space.canonicalize(&minus_q).unwrap()
        );
    }

    #[test]
    fn so3_profile_binds_interpolation_policy() {
        let a = So3Space::with_numerical_policy(1e-10, 1e-12).unwrap();
        let b = So3Space::with_numerical_policy(1e-10, 1e-9).unwrap();
        assert_ne!(a.profile().identity(), b.profile().identity());
        assert_eq!(a.near_identity_dot_tolerance(), 1e-12);
    }

    #[test]
    fn so3_group_identity_inverse_and_interpolation_hold() {
        let space = So3Space::new(1e-10).unwrap();
        let q = [(PI / 8.0).cos(), 0.0, 0.0, (PI / 8.0).sin()];
        let identity = space.identity();
        close(
            space
                .distance(&space.compose(&q, &space.inverse(&q).unwrap()).unwrap(), &identity)
                .unwrap(),
            0.0,
            1e-12,
        );
        let same_orientation = [-1.0, 0.0, 0.0, 0.0];
        let midpoint = space
            .interpolate(&identity, &same_orientation, 0.5)
            .unwrap();
        close(space.distance(&identity, &midpoint).unwrap(), 0.0, 1e-12);
    }

    #[test]
    fn se3_profile_binds_metric_and_numerical_policy() {
        let a = Se3Space::with_numerical_policy(1.0, 1.0, 1e-10, 1e-12).unwrap();
        let changed_metric =
            Se3Space::with_numerical_policy(4.0, 1.0, 1e-10, 1e-12).unwrap();
        let changed_numerics =
            Se3Space::with_numerical_policy(1.0, 1.0, 1e-10, 1e-9).unwrap();
        assert_ne!(a.profile().identity(), changed_metric.profile().identity());
        assert_ne!(a.profile().identity(), changed_numerics.profile().identity());

        let translated = RigidPose3 {
            translation: [3.0, 4.0, 0.0],
            rotation: [1.0, 0.0, 0.0, 0.0],
        };
        close(a.distance(&a.identity(), &translated).unwrap(), 5.0, 1e-12);
    }

    #[test]
    fn se3_group_inverse_cancels_pose() {
        let space = Se3Space::new(1.0, 1.0, 1e-10).unwrap();
        let angle = PI / 3.0;
        let pose = RigidPose3 {
            translation: [1.0, -2.0, 0.5],
            rotation: [
                (angle / 2.0).cos(),
                0.0,
                0.0,
                (angle / 2.0).sin(),
            ],
        };
        let inverse = space.inverse(&pose).unwrap();
        let composed = space.compose(&pose, &inverse).unwrap();
        close(
            space.distance(&composed, &space.identity()).unwrap(),
            0.0,
            1e-10,
        );
    }

    #[test]
    fn joint_space_preserves_topology_bounds_and_order() {
        let space = JointConfigurationSpace::new(
            vec![
                JointSpec::continuous_revolute("yaw", 1.0),
                JointSpec::bounded_revolute("elbow", -1.0, 1.0, 4.0),
                JointSpec::prismatic("slide", 0.0, 2.0, 9.0),
                JointSpec::fixed("fixture", 0.25),
            ],
            1e-12,
        )
        .unwrap();
        assert_eq!(space.intrinsic_dimension(), Some(3));
        assert_eq!(space.coordinate_count(), 4);

        let a = vec![179.0_f64.to_radians(), 0.0, 0.0, 0.25];
        let b = vec![(-179.0_f64).to_radians(), 0.5, 1.0, 0.25];
        let expected = ((2.0_f64.to_radians()).powi(2)
            + 4.0 * 0.5_f64.powi(2)
            + 9.0)
            .sqrt();
        close(space.distance(&a, &b).unwrap(), expected, 1e-12);
        assert_eq!(space.specs()[0].name, "yaw");
        assert_eq!(space.specs()[1].name, "elbow");
    }

    #[test]
    fn dynamic_bounds_are_fail_closed() {
        let bounded = JointConfigurationSpace::new(
            vec![JointSpec::bounded_revolute("joint", -1.0, 1.0, 1.0)],
            1e-12,
        )
        .unwrap();
        assert_eq!(
            bounded.validate_state(&vec![1.1]),
            StateValidity::OutOfBounds { coordinate: 0 }
        );

        let degenerate = JointConfigurationSpace::new(
            vec![JointSpec::bounded_revolute("locked", 0.25, 0.25, 1.0)],
            1e-12,
        );
        assert!(matches!(
            degenerate,
            Err(ConfigurationSpaceError::InvalidBounds { .. })
        ));
    }

    #[test]
    fn fixed_joint_weight_is_operationally_irrelevant() {
        let canonical = JointSpec::fixed("fixture", 0.25);
        let mut altered = canonical.clone();
        altered.weight = f64::INFINITY;
        let a = JointConfigurationSpace::new(vec![canonical], 1e-12).unwrap();
        let b = JointConfigurationSpace::new(vec![altered], 1e-12).unwrap();
        assert_eq!(a.profile().identity(), b.profile().identity());
        close(
            b.distance(&vec![0.25], &vec![0.25]).unwrap(),
            0.0,
            0.0,
        );
    }

    #[test]
    fn joint_profile_identity_binds_order_and_limits() {
        let a = JointConfigurationSpace::new(
            vec![
                JointSpec::bounded_revolute("a", -1.0, 1.0, 1.0),
                JointSpec::bounded_revolute("b", -2.0, 2.0, 1.0),
            ],
            1e-12,
        )
        .unwrap();
        let swapped = JointConfigurationSpace::new(
            vec![
                JointSpec::bounded_revolute("b", -2.0, 2.0, 1.0),
                JointSpec::bounded_revolute("a", -1.0, 1.0, 1.0),
            ],
            1e-12,
        )
        .unwrap();
        let changed_limit = JointConfigurationSpace::new(
            vec![
                JointSpec::bounded_revolute("a", -1.0, 1.1, 1.0),
                JointSpec::bounded_revolute("b", -2.0, 2.0, 1.0),
            ],
            1e-12,
        )
        .unwrap();
        assert_ne!(a.profile().identity(), swapped.profile().identity());
        assert_ne!(a.profile().identity(), changed_limit.profile().identity());
    }
}
