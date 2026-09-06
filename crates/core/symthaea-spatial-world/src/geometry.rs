// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Metric geometry and bounded pose uncertainty.

use std::num::NonZeroU64;

use serde::{Deserialize, Serialize};

use crate::SpatialValidationError;

/// Stable non-zero namespace for source-local reference-frame identities.
///
/// Frames from independent sensors, simulators, robots, or map providers may
/// legitimately reuse the same local identifier. The namespace prevents those
/// unrelated coordinate systems from collapsing merely because their local IDs
/// match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ReferenceFrameNamespaceId(NonZeroU64);

impl ReferenceFrameNamespaceId {
    /// Construct a non-zero reference-frame namespace.
    pub fn new(value: u64) -> Result<Self, SpatialValidationError> {
        NonZeroU64::new(value)
            .map(Self)
            .ok_or(SpatialValidationError::ZeroId {
                kind: "reference-frame-namespace",
            })
    }

    /// Return the numeric namespace identity.
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// Source- and generation-qualified stable identity for a spatial reference frame.
///
/// `generation` must advance whenever the semantic definition of the frame changes
/// (for example after recalibration, map reset, relocalization-root replacement,
/// or reuse of the same source-local frame ID). A changed generation is a changed
/// frame identity; continuity requires an explicit transform rather than equality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceFrameId {
    namespace: ReferenceFrameNamespaceId,
    local_id: NonZeroU64,
    generation: NonZeroU64,
}

impl ReferenceFrameId {
    /// Construct a source- and generation-qualified reference-frame identity.
    pub fn new(
        namespace: ReferenceFrameNamespaceId,
        local_id: u64,
        generation: u64,
    ) -> Result<Self, SpatialValidationError> {
        let local_id = NonZeroU64::new(local_id).ok_or(SpatialValidationError::ZeroId {
            kind: "reference-frame-local",
        })?;
        let generation = NonZeroU64::new(generation).ok_or(SpatialValidationError::ZeroId {
            kind: "reference-frame-generation",
        })?;
        Ok(Self {
            namespace,
            local_id,
            generation,
        })
    }

    /// Namespace that owns the source-local frame identity.
    pub const fn namespace(self) -> ReferenceFrameNamespaceId {
        self.namespace
    }

    /// Source-local frame identifier.
    pub const fn local_id(self) -> u64 {
        self.local_id.get()
    }

    /// Semantic generation of this frame definition.
    pub const fn generation(self) -> u64 {
        self.generation.get()
    }
}

/// Finite point in a metric Cartesian frame, expressed in metres.
///
/// A point denotes a position. It is intentionally distinct from [`MetricVector3`],
/// which denotes a free displacement/translation and is the type used by `Pose3`.
/// Signed floating-point zero is canonicalized to `+0.0` so semantically identical
/// metric points cannot acquire different persisted/content-addressed identities.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "[f64; 3]", into = "[f64; 3]")]
pub struct MetricPoint3 {
    x_m: f64,
    y_m: f64,
    z_m: f64,
}

impl MetricPoint3 {
    /// Construct a point after rejecting NaN/infinite coordinates and
    /// canonicalizing signed zero.
    pub fn new(x_m: f64, y_m: f64, z_m: f64) -> Result<Self, SpatialValidationError> {
        validate_finite("x_m", x_m)?;
        validate_finite("y_m", y_m)?;
        validate_finite("z_m", z_m)?;
        Ok(Self {
            x_m: canonicalize_zero(x_m),
            y_m: canonicalize_zero(y_m),
            z_m: canonicalize_zero(z_m),
        })
    }

    /// X coordinate in metres.
    pub const fn x_m(self) -> f64 {
        self.x_m
    }

    /// Y coordinate in metres.
    pub const fn y_m(self) -> f64 {
        self.y_m
    }

    /// Z coordinate in metres.
    pub const fn z_m(self) -> f64 {
        self.z_m
    }

    /// Return `[x, y, z]` in metres.
    pub const fn as_array(self) -> [f64; 3] {
        [self.x_m, self.y_m, self.z_m]
    }
}

impl TryFrom<[f64; 3]> for MetricPoint3 {
    type Error = SpatialValidationError;

    fn try_from(value: [f64; 3]) -> Result<Self, Self::Error> {
        Self::new(value[0], value[1], value[2])
    }
}

impl From<MetricPoint3> for [f64; 3] {
    fn from(value: MetricPoint3) -> Self {
        value.as_array()
    }
}

/// Finite free displacement/translation vector in a metric Cartesian frame.
///
/// This is deliberately not interchangeable with [`MetricPoint3`]. A rigid
/// transform translation is a vector from the reference-frame origin to the
/// local-frame origin, even though both points and vectors have three metric
/// components on the wire. Signed floating-point zero is canonicalized to
/// `+0.0` for stable persistence/content identity.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "[f64; 3]", into = "[f64; 3]")]
pub struct MetricVector3 {
    x_m: f64,
    y_m: f64,
    z_m: f64,
}

impl MetricVector3 {
    /// Construct a finite displacement vector in metres and canonicalize signed zero.
    pub fn new(x_m: f64, y_m: f64, z_m: f64) -> Result<Self, SpatialValidationError> {
        validate_finite("vector_x_m", x_m)?;
        validate_finite("vector_y_m", y_m)?;
        validate_finite("vector_z_m", z_m)?;
        Ok(Self {
            x_m: canonicalize_zero(x_m),
            y_m: canonicalize_zero(y_m),
            z_m: canonicalize_zero(z_m),
        })
    }

    /// X displacement in metres.
    pub const fn x_m(self) -> f64 {
        self.x_m
    }

    /// Y displacement in metres.
    pub const fn y_m(self) -> f64 {
        self.y_m
    }

    /// Z displacement in metres.
    pub const fn z_m(self) -> f64 {
        self.z_m
    }

    /// Return `[x, y, z]` displacement components in metres.
    pub const fn as_array(self) -> [f64; 3] {
        [self.x_m, self.y_m, self.z_m]
    }
}

impl TryFrom<[f64; 3]> for MetricVector3 {
    type Error = SpatialValidationError;

    fn try_from(value: [f64; 3]) -> Result<Self, Self::Error> {
        Self::new(value[0], value[1], value[2])
    }
}

impl From<MetricVector3> for [f64; 3] {
    fn from(value: MetricVector3) -> Self {
        value.as_array()
    }
}

/// Canonical normalized quaternion representing 3D orientation.
///
/// `q` and `-q` encode the same physical rotation. Construction normalizes the
/// quaternion with a scale-safe algorithm and chooses a deterministic sign so
/// equivalent rotations serialize identically within ordinary floating-point
/// tolerance.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "[f64; 4]", into = "[f64; 4]")]
pub struct UnitQuaternion {
    w: f64,
    x: f64,
    y: f64,
    z: f64,
}

impl UnitQuaternion {
    /// Normalize and canonicalize a finite quaternion.
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Result<Self, SpatialValidationError> {
        validate_finite("quaternion.w", w)?;
        validate_finite("quaternion.x", x)?;
        validate_finite("quaternion.y", y)?;
        validate_finite("quaternion.z", z)?;

        let raw = [w, x, y, z];
        let scale = raw.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);
        if scale == 0.0 {
            return Err(SpatialValidationError::DegenerateQuaternion);
        }

        // Scale before squaring so finite f64 components near f64::MAX remain
        // normalizable rather than overflowing the naive sum-of-squares norm.
        let scaled = raw.map(|component| component / scale);
        let scaled_norm = scaled
            .iter()
            .map(|component| component * component)
            .sum::<f64>()
            .sqrt();

        // Preserve the historical ~1e-12 minimum quaternion norm without ever
        // multiplying `scale * scaled_norm`, which itself could overflow.
        if scale <= 1e-12 / scaled_norm {
            return Err(SpatialValidationError::DegenerateQuaternion);
        }

        let inv_norm = scale.recip() / scaled_norm;
        let mut q = raw.map(|component| component * inv_norm);

        // Choose one representative from the q/-q equivalence class. If the
        // scalar component is exactly zero, fall through lexicographically.
        if should_flip_quaternion(q) {
            for component in &mut q {
                *component = -*component;
            }
        }
        for component in &mut q {
            *component = canonicalize_zero(*component);
        }

        Ok(Self {
            w: q[0],
            x: q[1],
            y: q[2],
            z: q[3],
        })
    }

    /// Identity rotation.
    pub const fn identity() -> Self {
        Self {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Return `[w, x, y, z]`.
    pub const fn as_array(self) -> [f64; 4] {
        [self.w, self.x, self.y, self.z]
    }
}

impl TryFrom<[f64; 4]> for UnitQuaternion {
    type Error = SpatialValidationError;

    fn try_from(value: [f64; 4]) -> Result<Self, Self::Error> {
        Self::new(value[0], value[1], value[2], value[3])
    }
}

impl From<UnitQuaternion> for [f64; 4] {
    fn from(value: UnitQuaternion) -> Self {
        value.as_array()
    }
}

/// Rigid 3D pose consisting of a metric translation vector and orientation.
///
/// Per the crate convention this is `T_reference_from_local`: `translation` is
/// the vector from the reference-frame origin to the local-frame origin,
/// expressed in reference-frame axes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pose3 {
    translation: MetricVector3,
    rotation: UnitQuaternion,
}

impl Pose3 {
    /// Construct a pose from already validated metric components.
    pub const fn new(translation: MetricVector3, rotation: UnitQuaternion) -> Self {
        Self {
            translation,
            rotation,
        }
    }

    /// Translation vector in metres, expressed in reference-frame axes.
    pub const fn translation(self) -> MetricVector3 {
        self.translation
    }

    /// Unit-quaternion orientation.
    pub const fn rotation(self) -> UnitQuaternion {
        self.rotation
    }
}

/// Independent one-sigma uncertainty for six local SE(3) perturbation components.
///
/// Translation sigmas are `[x, y, z]` in metres. Rotation sigmas are the
/// components of a small-angle tangent vector `[rx, ry, rz]` in radians, defined
/// as a **left-multiplicative** perturbation expressed in the pose reference-frame
/// axes: `R_perturbed = Exp(delta_theta^) * R`.
///
/// This avoids ambiguous Euler roll/pitch/yaw conventions and singularities.
/// V1 still models only diagonal uncertainty; a future fusion tranche may add a
/// fully qualified SE(3) covariance with positive-semidefinite validation.
/// Signed zero is canonicalized to `+0.0` so exact-zero uncertainty has one wire
/// representation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "[f64; 6]", into = "[f64; 6]")]
pub struct PoseUncertainty {
    translation_sigma_m: [f64; 3],
    rotation_tangent_sigma_rad: [f64; 3],
}

impl PoseUncertainty {
    /// Construct diagonal SE(3) uncertainty from translation and tangent-rotation sigmas.
    pub fn new(
        translation_sigma_m: [f64; 3],
        rotation_tangent_sigma_rad: [f64; 3],
    ) -> Result<Self, SpatialValidationError> {
        for (index, value) in translation_sigma_m.iter().copied().enumerate() {
            validate_sigma(translation_sigma_name(index), value)?;
        }
        for (index, value) in rotation_tangent_sigma_rad.iter().copied().enumerate() {
            validate_sigma(rotation_tangent_sigma_name(index), value)?;
        }
        Ok(Self {
            translation_sigma_m: translation_sigma_m.map(canonicalize_zero),
            rotation_tangent_sigma_rad: rotation_tangent_sigma_rad.map(canonicalize_zero),
        })
    }

    /// Translation standard deviations `[x, y, z]` in metres.
    pub const fn translation_sigma_m(self) -> [f64; 3] {
        self.translation_sigma_m
    }

    /// Small-angle tangent-space standard deviations `[rx, ry, rz]` in radians.
    pub const fn rotation_tangent_sigma_rad(self) -> [f64; 3] {
        self.rotation_tangent_sigma_rad
    }

    /// Return translation then SO(3)-tangent standard deviations.
    pub const fn as_array(self) -> [f64; 6] {
        [
            self.translation_sigma_m[0],
            self.translation_sigma_m[1],
            self.translation_sigma_m[2],
            self.rotation_tangent_sigma_rad[0],
            self.rotation_tangent_sigma_rad[1],
            self.rotation_tangent_sigma_rad[2],
        ]
    }
}

impl TryFrom<[f64; 6]> for PoseUncertainty {
    type Error = SpatialValidationError;

    fn try_from(value: [f64; 6]) -> Result<Self, Self::Error> {
        Self::new(
            [value[0], value[1], value[2]],
            [value[3], value[4], value[5]],
        )
    }
}

impl From<PoseUncertainty> for [f64; 6] {
    fn from(value: PoseUncertainty) -> Self {
        value.as_array()
    }
}

/// Pose plus explicitly represented measurement/belief uncertainty.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PoseEstimate {
    pose: Pose3,
    uncertainty: PoseUncertainty,
}

impl PoseEstimate {
    /// Construct a pose estimate from validated components.
    pub const fn new(pose: Pose3, uncertainty: PoseUncertainty) -> Self {
        Self { pose, uncertainty }
    }

    /// Estimated pose.
    pub const fn pose(self) -> Pose3 {
        self.pose
    }

    /// Associated one-sigma pose uncertainty.
    pub const fn uncertainty(self) -> PoseUncertainty {
        self.uncertainty
    }
}

fn validate_finite(field: &'static str, value: f64) -> Result<(), SpatialValidationError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(SpatialValidationError::NonFinite { field, value })
    }
}

fn validate_sigma(field: &'static str, value: f64) -> Result<(), SpatialValidationError> {
    validate_finite(field, value)?;
    if value < 0.0 {
        Err(SpatialValidationError::NegativeUncertainty { field, value })
    } else {
        Ok(())
    }
}

fn canonicalize_zero(value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        value
    }
}

const fn translation_sigma_name(index: usize) -> &'static str {
    match index {
        0 => "translation_sigma_x_m",
        1 => "translation_sigma_y_m",
        _ => "translation_sigma_z_m",
    }
}

const fn rotation_tangent_sigma_name(index: usize) -> &'static str {
    match index {
        0 => "rotation_tangent_sigma_x_rad",
        1 => "rotation_tangent_sigma_y_rad",
        _ => "rotation_tangent_sigma_z_rad",
    }
}

fn should_flip_quaternion(q: [f64; 4]) -> bool {
    for component in q {
        if component > 0.0 {
            return false;
        }
        if component < 0.0 {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn namespace(value: u64) -> ReferenceFrameNamespaceId {
        ReferenceFrameNamespaceId::new(value).unwrap()
    }

    #[test]
    fn frame_identity_is_source_and_generation_qualified() {
        let a = ReferenceFrameId::new(namespace(1), 7, 1).unwrap();
        let other_source = ReferenceFrameId::new(namespace(2), 7, 1).unwrap();
        let next_generation = ReferenceFrameId::new(namespace(1), 7, 2).unwrap();
        assert_ne!(a, other_source);
        assert_ne!(a, next_generation);
        assert_eq!(a.local_id(), other_source.local_id());
    }

    #[test]
    fn zero_frame_namespace_local_id_and_generation_are_rejected() {
        assert!(ReferenceFrameNamespaceId::new(0).is_err());
        assert!(ReferenceFrameId::new(namespace(1), 0, 1).is_err());
        assert!(ReferenceFrameId::new(namespace(1), 1, 0).is_err());
    }

    #[test]
    fn metric_point_and_vector_reject_non_finite_values() {
        assert!(MetricPoint3::new(f64::NAN, 0.0, 0.0).is_err());
        assert!(MetricPoint3::new(0.0, f64::INFINITY, 0.0).is_err());
        assert!(MetricVector3::new(0.0, 0.0, f64::NEG_INFINITY).is_err());
    }

    #[test]
    fn metric_geometry_and_uncertainty_canonicalize_signed_zero() {
        let point = MetricPoint3::new(-0.0, 0.0, -0.0).unwrap();
        for value in point.as_array() {
            assert_eq!(value.to_bits(), 0.0_f64.to_bits());
        }

        let vector = MetricVector3::new(-0.0, 0.0, -0.0).unwrap();
        for value in vector.as_array() {
            assert_eq!(value.to_bits(), 0.0_f64.to_bits());
        }

        let uncertainty = PoseUncertainty::new([-0.0, 0.0, -0.0], [0.0, -0.0, 0.0]).unwrap();
        for value in uncertainty.as_array() {
            assert_eq!(value.to_bits(), 0.0_f64.to_bits());
        }

        let point_json = serde_json::to_string(&point).unwrap();
        let vector_json = serde_json::to_string(&vector).unwrap();
        let uncertainty_json = serde_json::to_string(&uncertainty).unwrap();
        assert!(!point_json.contains("-0.0"));
        assert!(!vector_json.contains("-0.0"));
        assert!(!uncertainty_json.contains("-0.0"));
    }

    #[test]
    fn pose_translation_is_a_vector_not_a_point() {
        let translation = MetricVector3::new(1.0, -2.0, 3.0).unwrap();
        let pose = Pose3::new(translation, UnitQuaternion::identity());
        assert_eq!(pose.translation(), translation);
    }

    #[test]
    fn quaternion_rejects_degenerate_and_non_finite_inputs() {
        assert!(UnitQuaternion::new(0.0, 0.0, 0.0, 0.0).is_err());
        assert!(UnitQuaternion::new(f64::NAN, 0.0, 0.0, 1.0).is_err());
    }

    #[test]
    fn quaternion_normalizes_and_canonicalizes_sign() {
        let a = UnitQuaternion::new(2.0, 0.0, 0.0, 0.0).unwrap();
        let b = UnitQuaternion::new(-2.0, 0.0, 0.0, 0.0).unwrap();
        assert_eq!(a, UnitQuaternion::identity());
        assert_eq!(a, b);
    }

    #[test]
    fn quaternion_normalization_is_safe_for_extreme_finite_components() {
        let q = UnitQuaternion::new(f64::MAX, f64::MAX, 0.0, 0.0).unwrap();
        assert!(q.as_array().iter().all(|component| component.is_finite()));
        let norm_sq = q
            .as_array()
            .iter()
            .map(|component| component * component)
            .sum::<f64>();
        assert!((norm_sq - 1.0).abs() < 1e-12);
    }

    #[test]
    fn uncertainty_rejects_negative_or_non_finite_sigma() {
        assert!(PoseUncertainty::new([-0.1, 0.0, 0.0], [0.0; 3]).is_err());
        assert!(PoseUncertainty::new([0.0; 3], [0.0, f64::NAN, 0.0]).is_err());
    }

    #[test]
    fn serde_try_from_contracts_reject_invalid_geometry() {
        let invalid_point = serde_json::from_str::<MetricPoint3>("[0.0,1e999,0.0]");
        assert!(invalid_point.is_err());
        let invalid_vector = serde_json::from_str::<MetricVector3>("[0.0,1e999,0.0]");
        assert!(invalid_vector.is_err());
        let invalid_quaternion = serde_json::from_str::<UnitQuaternion>("[0.0,0.0,0.0,0.0]");
        assert!(invalid_quaternion.is_err());
        let invalid_uncertainty =
            serde_json::from_str::<PoseUncertainty>("[-1.0,0.0,0.0,0.0,0.0,0.0]");
        assert!(invalid_uncertainty.is_err());
    }

    #[test]
    fn nested_pose_structs_reject_unknown_fields() {
        let pose = r#"{
            "translation":[1.0,2.0,3.0],
            "rotation":[1.0,0.0,0.0,0.0],
            "unexpected":true
        }"#;
        assert!(serde_json::from_str::<Pose3>(pose).is_err());

        let estimate = r#"{
            "pose":{"translation":[1.0,2.0,3.0],"rotation":[1.0,0.0,0.0,0.0]},
            "uncertainty":[0.0,0.0,0.0,0.0,0.0,0.0],
            "unexpected":true
        }"#;
        assert!(serde_json::from_str::<PoseEstimate>(estimate).is_err());
    }
}
