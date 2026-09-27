// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic C0 robot-design to fabrication-geometry translation.
//!
//! This bridge projects exact design intent into the fabrication kernel's
//! millimetre/f32 CSG and triangle-mesh representations. It does not grant
//! manufacturability, fabrication, structural, safety, or physical authority.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;
use symthaea_fabrication_kernel::csg::{CSGNode, Transform3D};
use symthaea_fabrication_kernel::mesh::{
    TessellationPolicy, TriangleMesh, resolve_to_mesh_with_policy,
};
use symthaea_fabrication_kernel::validate::{ValidationReport, validate_mesh};
use symthaea_robot_design::c0_joint_link_coupon::{
    C0CompiledCouponV1, C0CouponError, C0GeometryIntentId, C0SectionOrientationV1,
    validate_strict_c0_subject,
};
use symthaea_robot_design::exact_parameters::ExactDesignLengthUmV1;
use symthaea_robot_design::{ContentDigest, RobotDesignError, RobotDesignId};

pub const C0_CSG_PROJECTION_SCHEMA_ID: &str =
    "symthaea.robot-fabrication.c0-csg-projection.v1";
pub const C0_MESH_ARTIFACT_SCHEMA_ID: &str = "symthaea.robot-fabrication.c0-mesh-artifact.v1";
pub const C0_MESH_POLICY_SCHEMA_ID: &str = "symthaea.robot-fabrication.c0-mesh-policy.v1";
pub const C0_TRANSLATION_SCHEMA_ID: &str = "symthaea.robot-fabrication.c0-translation.v1";
pub const C0_TRANSLATION_SCHEMA_VERSION: u32 = 1;

/// Numerical adapter ceiling only; this is not a manufacturing tolerance.
pub const MAX_F32_LENGTH_ERROR_UM: f64 = 0.1;
/// Geometry cross-check tolerance only; this is not a physical tolerance.
pub const MAX_AABB_ERROR_MM: f64 = 0.001;
pub const MAX_CENTER_ERROR_MM: f64 = 0.001;
pub const MAX_VOLUME_ABS_ERROR_MM3: f64 = 0.01;
pub const MAX_VOLUME_REL_ERROR: f64 = 1.0e-5;
const MAX_EXACT_F64_INTEGER_UM: u64 = 1_u64 << 53;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum C0ProjectionAxisV1 {
    LongitudinalLength,
    SectionWidth,
    SectionHeight,
}

impl C0ProjectionAxisV1 {
    const fn tag(self) -> u8 {
        match self {
            Self::LongitudinalLength => 0,
            Self::SectionWidth => 1,
            Self::SectionHeight => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct C0AxisProjectionV1 {
    pub semantic_axis: C0ProjectionAxisV1,
    pub source_um: u64,
    pub emitted_mm: f32,
    pub emitted_f32_bits: u32,
    pub represented_um: f64,
    pub absolute_error_um: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct C0CsgProjectionId(ContentDigest);

impl C0CsgProjectionId {
    pub const fn digest(self) -> ContentDigest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

impl fmt::Display for C0CsgProjectionId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct C0MeshArtifactId(ContentDigest);

impl C0MeshArtifactId {
    pub const fn digest(self) -> ContentDigest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

impl fmt::Display for C0MeshArtifactId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct C0MeshPolicyId(ContentDigest);

impl C0MeshPolicyId {
    pub const fn digest(self) -> ContentDigest {
        self.0
    }

    pub fn to_hex(self) -> String {
        self.0.to_hex()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct C0AabbSummaryV1 {
    pub min_mm: [f32; 3],
    pub max_mm: [f32; 3],
    pub dimensions_mm: [f64; 3],
    pub center_mm: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct C0MeshValidationSummaryV1 {
    pub triangle_count: usize,
    pub vertex_count: usize,
    pub is_valid: bool,
    pub normal_count_matches: bool,
    pub is_watertight: bool,
    pub boundary_edges: usize,
    pub non_manifold_edges: usize,
    pub connected_components: usize,
    pub duplicate_triangle_count: usize,
    pub degenerate_triangle_count: usize,
    pub inconsistent_normal_count: usize,
    pub out_of_bounds_index_count: usize,
    pub non_finite_vertex_count: usize,
    pub non_finite_normal_count: usize,
    pub self_intersection_count: usize,
    pub self_intersection_scan_complete: bool,
    pub signed_volume_mm3: f32,
}

impl C0MeshValidationSummaryV1 {
    pub fn satisfies_c0_closed_solid_gate(&self) -> bool {
        self.is_valid
            && self.normal_count_matches
            && self.is_watertight
            && self.boundary_edges == 0
            && self.non_manifold_edges == 0
            && self.connected_components == 1
            && self.duplicate_triangle_count == 0
            && self.degenerate_triangle_count == 0
            && self.inconsistent_normal_count == 0
            && self.out_of_bounds_index_count == 0
            && self.non_finite_vertex_count == 0
            && self.non_finite_normal_count == 0
            && self.self_intersection_scan_complete
            && self.self_intersection_count == 0
            && self.signed_volume_mm3 > 0.0
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct C0GeometryTranslationReceiptV1 {
    pub schema_id: String,
    pub schema_version: u32,
    pub robot_design_id: RobotDesignId,
    pub geometry_intent_id: C0GeometryIntentId,
    pub csg_projection_id: C0CsgProjectionId,
    pub mesh_policy_id: C0MeshPolicyId,
    pub mesh_artifact_id: C0MeshArtifactId,
    pub orientation: C0SectionOrientationV1,
    pub x_axis: C0AxisProjectionV1,
    pub y_axis: C0AxisProjectionV1,
    pub z_axis: C0AxisProjectionV1,
    pub exact_design_volume_um3: u128,
    pub emitted_csg_volume_mm3: f64,
    pub mesh_signed_volume_mm3: f64,
    pub aabb: C0AabbSummaryV1,
    pub validation: C0MeshValidationSummaryV1,
}

/// Execution artifacts are intentionally separate from the serializable receipt.
#[derive(Debug, Clone)]
pub struct C0FabricationProjectionV1 {
    pub csg: CSGNode,
    pub mesh: TriangleMesh,
    pub receipt: C0GeometryTranslationReceiptV1,
}

pub fn translate_c0_coupon_to_fabrication_geometry(
    compiled: &C0CompiledCouponV1,
) -> Result<C0FabricationProjectionV1, C0BridgeError> {
    verify_compiled_subject(compiled)?;

    let geometry = compiled.geometry_intent;
    let length = project_length(
        C0ProjectionAxisV1::LongitudinalLength,
        geometry.link_length,
    )?;
    let width = project_length(C0ProjectionAxisV1::SectionWidth, geometry.section_width)?;
    let height = project_length(C0ProjectionAxisV1::SectionHeight, geometry.section_height)?;

    let (x_axis, y_axis, z_axis) = match geometry.orientation {
        C0SectionOrientationV1::LongitudinalXWidthYHeightZ => (length, width, height),
        C0SectionOrientationV1::LongitudinalXWidthZHeightY => (length, height, width),
    };

    let transform = Transform3D {
        scale: [x_axis.emitted_mm, y_axis.emitted_mm, z_axis.emitted_mm],
        rotate: [0.0, 0.0, 0.0],
        translate: [0.0, 0.0, 0.0],
    };
    let csg = CSGNode::cube().with_transform(transform.clone());
    let csg_projection_id = csg_projection_id(geometry.orientation, &transform);

    let mesh_policy = frozen_c0_mesh_policy();
    let mesh_policy_id = mesh_policy_id(mesh_policy);
    let mesh = resolve_to_mesh_with_policy(&csg, mesh_policy);
    let mesh_artifact_id = mesh_artifact_id(&mesh);
    let validation_report = validate_mesh(&mesh);
    let validation = summarize_validation(&validation_report);
    if !validation.satisfies_c0_closed_solid_gate() {
        return Err(C0BridgeError::MeshValidationFailed(validation));
    }

    let aabb = compute_aabb(&mesh)?;
    let expected_dimensions = [
        f64::from(x_axis.emitted_mm),
        f64::from(y_axis.emitted_mm),
        f64::from(z_axis.emitted_mm),
    ];
    for axis in 0..3 {
        if (aabb.dimensions_mm[axis] - expected_dimensions[axis]).abs() > MAX_AABB_ERROR_MM {
            return Err(C0BridgeError::AabbDimensionMismatch {
                axis,
                expected_mm: expected_dimensions[axis],
                actual_mm: aabb.dimensions_mm[axis],
            });
        }
        if aabb.center_mm[axis].abs() > MAX_CENTER_ERROR_MM {
            return Err(C0BridgeError::AabbCenterMismatch {
                axis,
                actual_mm: aabb.center_mm[axis],
            });
        }
    }

    let exact_design_volume_um3 = checked_volume_um3(
        geometry.link_length,
        geometry.section_width,
        geometry.section_height,
    )?;
    let emitted_csg_volume_mm3 = expected_dimensions.iter().product::<f64>();
    let mesh_signed_volume_mm3 = f64::from(validation.signed_volume_mm3);
    let volume_tolerance = MAX_VOLUME_ABS_ERROR_MM3.max(
        emitted_csg_volume_mm3.abs() * MAX_VOLUME_REL_ERROR,
    );
    if (mesh_signed_volume_mm3 - emitted_csg_volume_mm3).abs() > volume_tolerance {
        return Err(C0BridgeError::MeshVolumeMismatch {
            expected_mm3: emitted_csg_volume_mm3,
            actual_mm3: mesh_signed_volume_mm3,
            tolerance_mm3: volume_tolerance,
        });
    }

    Ok(C0FabricationProjectionV1 {
        csg,
        mesh,
        receipt: C0GeometryTranslationReceiptV1 {
            schema_id: C0_TRANSLATION_SCHEMA_ID.to_string(),
            schema_version: C0_TRANSLATION_SCHEMA_VERSION,
            robot_design_id: compiled.receipt.robot_design_id,
            geometry_intent_id: compiled.receipt.geometry_intent_id,
            csg_projection_id,
            mesh_policy_id,
            mesh_artifact_id,
            orientation: geometry.orientation,
            x_axis,
            y_axis,
            z_axis,
            exact_design_volume_um3,
            emitted_csg_volume_mm3,
            mesh_signed_volume_mm3,
            aabb,
            validation,
        },
    })
}

fn verify_compiled_subject(compiled: &C0CompiledCouponV1) -> Result<(), C0BridgeError> {
    validate_strict_c0_subject(&compiled.subject)?;
    let design_id = compiled.subject.design_id()?;
    if design_id != compiled.receipt.robot_design_id {
        return Err(C0BridgeError::RobotDesignIdentityMismatch);
    }
    let geometry_intent_id = compiled.geometry_intent.geometry_intent_id()?;
    if geometry_intent_id != compiled.receipt.geometry_intent_id {
        return Err(C0BridgeError::GeometryIntentIdentityMismatch);
    }
    if compiled.subject.geometry.len() != 1
        || compiled.subject.geometry[0].artifact_digest != geometry_intent_id.digest()
    {
        return Err(C0BridgeError::GeometryReferenceMismatch);
    }
    if compiled.subject.parameter_set != compiled.receipt.parameter_set_id.digest() {
        return Err(C0BridgeError::ParameterSetReferenceMismatch);
    }
    Ok(())
}

fn project_length(
    semantic_axis: C0ProjectionAxisV1,
    value: ExactDesignLengthUmV1,
) -> Result<C0AxisProjectionV1, C0BridgeError> {
    let source_um = value.as_um();
    if source_um == 0 {
        return Err(C0BridgeError::NonPositiveLength);
    }
    if source_um > MAX_EXACT_F64_INTEGER_UM {
        return Err(C0BridgeError::LengthTooLargeForExactDiagnostic(source_um));
    }
    let exact_mm = source_um as f64 / 1_000.0;
    let emitted_mm = exact_mm as f32;
    if !emitted_mm.is_finite() || emitted_mm <= 0.0 {
        return Err(C0BridgeError::InvalidF32Projection(source_um));
    }
    let represented_um = f64::from(emitted_mm) * 1_000.0;
    let absolute_error_um = (represented_um - source_um as f64).abs();
    if absolute_error_um > MAX_F32_LENGTH_ERROR_UM {
        return Err(C0BridgeError::F32ProjectionErrorTooLarge {
            source_um,
            error_um: absolute_error_um,
            max_error_um: MAX_F32_LENGTH_ERROR_UM,
        });
    }
    Ok(C0AxisProjectionV1 {
        semantic_axis,
        source_um,
        emitted_mm,
        emitted_f32_bits: emitted_mm.to_bits(),
        represented_um,
        absolute_error_um,
    })
}

fn frozen_c0_mesh_policy() -> TessellationPolicy {
    TessellationPolicy {
        max_chord_error_mm: 0.01,
        min_segments: 12,
        max_segments: 256,
    }
}

fn csg_projection_id(
    orientation: C0SectionOrientationV1,
    transform: &Transform3D,
) -> C0CsgProjectionId {
    let mut out = Vec::new();
    put_str(&mut out, C0_CSG_PROJECTION_SCHEMA_ID);
    put_u32(&mut out, C0_TRANSLATION_SCHEMA_VERSION);
    put_u8(&mut out, 0); // Primitive::Cube
    put_u8(&mut out, orientation_tag(orientation));
    for value in transform.scale {
        put_u32(&mut out, value.to_bits());
    }
    for value in transform.rotate {
        put_u32(&mut out, value.to_bits());
    }
    for value in transform.translate {
        put_u32(&mut out, value.to_bits());
    }
    C0CsgProjectionId(hash_bytes(out))
}

fn mesh_policy_id(policy: TessellationPolicy) -> C0MeshPolicyId {
    let mut out = Vec::new();
    put_str(&mut out, C0_MESH_POLICY_SCHEMA_ID);
    put_u32(&mut out, C0_TRANSLATION_SCHEMA_VERSION);
    put_u32(&mut out, policy.max_chord_error_mm.to_bits());
    put_u64(&mut out, policy.min_segments as u64);
    put_u64(&mut out, policy.max_segments as u64);
    C0MeshPolicyId(hash_bytes(out))
}

fn mesh_artifact_id(mesh: &TriangleMesh) -> C0MeshArtifactId {
    let mut out = Vec::new();
    put_str(&mut out, C0_MESH_ARTIFACT_SCHEMA_ID);
    put_u32(&mut out, C0_TRANSLATION_SCHEMA_VERSION);
    put_len(&mut out, mesh.vertices.len());
    for vertex in &mesh.vertices {
        for value in vertex {
            put_u32(&mut out, value.to_bits());
        }
    }
    put_len(&mut out, mesh.normals.len());
    for normal in &mesh.normals {
        for value in normal {
            put_u32(&mut out, value.to_bits());
        }
    }
    put_len(&mut out, mesh.indices.len());
    for triangle in &mesh.indices {
        for index in triangle {
            put_u32(&mut out, *index);
        }
    }
    C0MeshArtifactId(hash_bytes(out))
}

fn summarize_validation(report: &ValidationReport) -> C0MeshValidationSummaryV1 {
    C0MeshValidationSummaryV1 {
        triangle_count: report.triangle_count,
        vertex_count: report.vertex_count,
        is_valid: report.is_valid(),
        normal_count_matches: report.normal_count_matches,
        is_watertight: report.is_watertight,
        boundary_edges: report.boundary_edges,
        non_manifold_edges: report.non_manifold_edges,
        connected_components: report.connected_components,
        duplicate_triangle_count: report.duplicate_triangles.len(),
        degenerate_triangle_count: report.degenerate_triangles.len(),
        inconsistent_normal_count: report.inconsistent_normals.len(),
        out_of_bounds_index_count: report.out_of_bounds_indices.len(),
        non_finite_vertex_count: report.non_finite_vertices.len(),
        non_finite_normal_count: report.non_finite_normals.len(),
        self_intersection_count: report.self_intersections.len(),
        self_intersection_scan_complete: report.self_intersection_scan_complete,
        signed_volume_mm3: report.signed_volume,
    }
}

fn compute_aabb(mesh: &TriangleMesh) -> Result<C0AabbSummaryV1, C0BridgeError> {
    let first = *mesh.vertices.first().ok_or(C0BridgeError::EmptyMesh)?;
    if !first.iter().all(|value| value.is_finite()) {
        return Err(C0BridgeError::NonFiniteMeshCoordinate);
    }
    let mut min = first;
    let mut max = first;
    for vertex in &mesh.vertices[1..] {
        if !vertex.iter().all(|value| value.is_finite()) {
            return Err(C0BridgeError::NonFiniteMeshCoordinate);
        }
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex[axis]);
            max[axis] = max[axis].max(vertex[axis]);
        }
    }
    let mut dimensions = [0.0_f64; 3];
    let mut center = [0.0_f64; 3];
    for axis in 0..3 {
        dimensions[axis] = f64::from(max[axis]) - f64::from(min[axis]);
        center[axis] = (f64::from(max[axis]) + f64::from(min[axis])) / 2.0;
    }
    Ok(C0AabbSummaryV1 {
        min_mm: min,
        max_mm: max,
        dimensions_mm: dimensions,
        center_mm: center,
    })
}

fn checked_volume_um3(
    length: ExactDesignLengthUmV1,
    width: ExactDesignLengthUmV1,
    height: ExactDesignLengthUmV1,
) -> Result<u128, C0BridgeError> {
    u128::from(length.as_um())
        .checked_mul(u128::from(width.as_um()))
        .and_then(|value| value.checked_mul(u128::from(height.as_um())))
        .ok_or(C0BridgeError::ExactVolumeOverflow)
}

fn orientation_tag(orientation: C0SectionOrientationV1) -> u8 {
    match orientation {
        C0SectionOrientationV1::LongitudinalXWidthYHeightZ => 0,
        C0SectionOrientationV1::LongitudinalXWidthZHeightY => 1,
    }
}

fn hash_bytes(bytes: Vec<u8>) -> ContentDigest {
    let digest: [u8; 32] = Sha256::digest(bytes).into();
    ContentDigest::from_bytes(digest)
}

fn put_u8(out: &mut Vec<u8>, value: u8) {
    out.push(value);
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn put_len(out: &mut Vec<u8>, len: usize) {
    out.extend_from_slice(&(len as u64).to_be_bytes());
}

fn put_str(out: &mut Vec<u8>, value: &str) {
    put_len(out, value.len());
    out.extend_from_slice(value.as_bytes());
}

#[derive(Debug, Clone, PartialEq)]
pub enum C0BridgeError {
    C0Design(C0CouponError),
    RobotDesign(RobotDesignError),
    RobotDesignIdentityMismatch,
    GeometryIntentIdentityMismatch,
    GeometryReferenceMismatch,
    ParameterSetReferenceMismatch,
    NonPositiveLength,
    LengthTooLargeForExactDiagnostic(u64),
    InvalidF32Projection(u64),
    F32ProjectionErrorTooLarge {
        source_um: u64,
        error_um: f64,
        max_error_um: f64,
    },
    MeshValidationFailed(C0MeshValidationSummaryV1),
    EmptyMesh,
    NonFiniteMeshCoordinate,
    AabbDimensionMismatch {
        axis: usize,
        expected_mm: f64,
        actual_mm: f64,
    },
    AabbCenterMismatch {
        axis: usize,
        actual_mm: f64,
    },
    MeshVolumeMismatch {
        expected_mm3: f64,
        actual_mm3: f64,
        tolerance_mm3: f64,
    },
    ExactVolumeOverflow,
}

impl fmt::Display for C0BridgeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::C0Design(error) => write!(formatter, "C0 design error: {error}"),
            Self::RobotDesign(error) => write!(formatter, "robot-design error: {error}"),
            Self::RobotDesignIdentityMismatch => formatter.write_str("C0 robot-design identity mismatch"),
            Self::GeometryIntentIdentityMismatch => formatter.write_str("C0 geometry-intent identity mismatch"),
            Self::GeometryReferenceMismatch => formatter.write_str("C0 subject geometry reference does not bind the exact geometry intent"),
            Self::ParameterSetReferenceMismatch => formatter.write_str("C0 subject parameter-set reference does not match compile receipt"),
            Self::NonPositiveLength => formatter.write_str("C0 projected length must be strictly positive"),
            Self::LengthTooLargeForExactDiagnostic(value) => write!(formatter, "C0 source length {value} um exceeds the exact f64 diagnostic envelope"),
            Self::InvalidF32Projection(value) => write!(formatter, "C0 source length {value} um cannot be represented as a positive finite f32 millimetre value"),
            Self::F32ProjectionErrorTooLarge { source_um, error_um, max_error_um } => write!(formatter, "C0 f32 projection error for {source_um} um is {error_um} um, exceeding {max_error_um} um"),
            Self::MeshValidationFailed(_) => formatter.write_str("C0 mesh failed the bounded closed-solid translation gate"),
            Self::EmptyMesh => formatter.write_str("C0 translation produced an empty mesh"),
            Self::NonFiniteMeshCoordinate => formatter.write_str("C0 translation produced a non-finite mesh coordinate"),
            Self::AabbDimensionMismatch { axis, expected_mm, actual_mm } => write!(formatter, "C0 mesh AABB dimension mismatch on axis {axis}: expected {expected_mm} mm, got {actual_mm} mm"),
            Self::AabbCenterMismatch { axis, actual_mm } => write!(formatter, "C0 mesh AABB center moved from the origin on axis {axis}: {actual_mm} mm"),
            Self::MeshVolumeMismatch { expected_mm3, actual_mm3, tolerance_mm3 } => write!(formatter, "C0 mesh volume mismatch: expected {expected_mm3} mm^3, got {actual_mm3} mm^3, tolerance {tolerance_mm3} mm^3"),
            Self::ExactVolumeOverflow => formatter.write_str("C0 exact design-volume arithmetic overflow"),
        }
    }
}

impl std::error::Error for C0BridgeError {}

impl From<C0CouponError> for C0BridgeError {
    fn from(value: C0CouponError) -> Self {
        Self::C0Design(value)
    }
}

impl From<RobotDesignError> for C0BridgeError {
    fn from(value: RobotDesignError) -> Self {
        Self::RobotDesign(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_robot_design::c0_joint_link_coupon::{
        C0CouponSearchDomainV1, C0JointLinkCouponTemplateV1, compile_c0_joint_link_coupon,
    };
    use symthaea_robot_design::exact_parameters::{
        DesignParameterId, ExactDesignParameterV1, ExactDesignParameterSetV1,
        ExactLengthDomainKindV1, ExactLengthDomainV1,
    };

    fn digest(seed: u8) -> ContentDigest {
        ContentDigest::from_bytes([seed; 32])
    }

    fn parameter(id: &str, value_um: u64) -> ExactDesignParameterV1 {
        ExactDesignParameterV1::length(
            DesignParameterId::new(id).unwrap(),
            ExactDesignLengthUmV1::from_um(value_um),
        )
    }

    fn domain(id: &str, values: &[u64]) -> ExactLengthDomainV1 {
        ExactLengthDomainV1::new(
            DesignParameterId::new(id).unwrap(),
            true,
            values.len() as u32,
            ExactLengthDomainKindV1::Explicit {
                values: values
                    .iter()
                    .copied()
                    .map(ExactDesignLengthUmV1::from_um)
                    .collect(),
            },
        )
    }

    fn compiled(orientation: C0SectionOrientationV1) -> C0CompiledCouponV1 {
        let template = C0JointLinkCouponTemplateV1::new(
            ExactDesignLengthUmV1::from_um(300_000),
            orientation,
            digest(1),
            digest(2),
            digest(3),
        );
        let parameters = ExactDesignParameterSetV1::new(vec![
            parameter("link_length", 300_000),
            parameter("section_width", 20_000),
            parameter("section_height", 6_000),
        ]);
        let search = C0CouponSearchDomainV1::new(
            domain("section_width", &[16_000, 18_000, 20_000, 22_000, 24_000]),
            domain("section_height", &[4_800, 5_400, 6_000, 6_600, 7_200]),
        );
        compile_c0_joint_link_coupon(&template, &parameters, &search).unwrap()
    }

    #[test]
    fn reference_coupon_translates_to_expected_centered_box() {
        let result = translate_c0_coupon_to_fabrication_geometry(&compiled(
            C0SectionOrientationV1::LongitudinalXWidthYHeightZ,
        ))
        .unwrap();
        assert_eq!(result.receipt.x_axis.emitted_mm, 300.0);
        assert_eq!(result.receipt.y_axis.emitted_mm, 20.0);
        assert_eq!(result.receipt.z_axis.emitted_mm, 6.0);
        assert_eq!(result.receipt.aabb.dimensions_mm, [300.0, 20.0, 6.0]);
        assert_eq!(result.receipt.aabb.center_mm, [0.0, 0.0, 0.0]);
        assert_eq!(result.receipt.exact_design_volume_um3, 36_000_000_000_000_u128);
        assert!((result.receipt.mesh_signed_volume_mm3 - 36_000.0).abs() <= 0.01);
        assert!(result.receipt.validation.satisfies_c0_closed_solid_gate());
    }

    #[test]
    fn alternate_orientation_swaps_execution_axes_not_design_semantics() {
        let result = translate_c0_coupon_to_fabrication_geometry(&compiled(
            C0SectionOrientationV1::LongitudinalXWidthZHeightY,
        ))
        .unwrap();
        assert_eq!(result.receipt.x_axis.semantic_axis, C0ProjectionAxisV1::LongitudinalLength);
        assert_eq!(result.receipt.y_axis.semantic_axis, C0ProjectionAxisV1::SectionHeight);
        assert_eq!(result.receipt.z_axis.semantic_axis, C0ProjectionAxisV1::SectionWidth);
        assert_eq!(result.receipt.aabb.dimensions_mm, [300.0, 6.0, 20.0]);
    }

    #[test]
    fn repeated_projection_has_stable_execution_artifact_ids() {
        let source = compiled(C0SectionOrientationV1::LongitudinalXWidthYHeightZ);
        let first = translate_c0_coupon_to_fabrication_geometry(&source).unwrap();
        let second = translate_c0_coupon_to_fabrication_geometry(&source).unwrap();
        assert_eq!(first.receipt.csg_projection_id, second.receipt.csg_projection_id);
        assert_eq!(first.receipt.mesh_policy_id, second.receipt.mesh_policy_id);
        assert_eq!(first.receipt.mesh_artifact_id, second.receipt.mesh_artifact_id);
    }

    #[test]
    fn one_micrometre_projection_records_bounded_rounding_without_identity_feedback() {
        let projection = project_length(
            C0ProjectionAxisV1::SectionWidth,
            ExactDesignLengthUmV1::from_um(20_001),
        )
        .unwrap();
        assert_eq!(projection.source_um, 20_001);
        assert!(projection.absolute_error_um <= MAX_F32_LENGTH_ERROR_UM);
    }

    #[test]
    fn tampered_robot_design_identity_is_rejected() {
        let mut source = compiled(C0SectionOrientationV1::LongitudinalXWidthYHeightZ);
        source.subject.requirement_snapshot = digest(9);
        assert_eq!(
            translate_c0_coupon_to_fabrication_geometry(&source).unwrap_err(),
            C0BridgeError::RobotDesignIdentityMismatch
        );
    }

    #[test]
    fn tampered_geometry_reference_is_rejected() {
        let mut source = compiled(C0SectionOrientationV1::LongitudinalXWidthYHeightZ);
        source.subject.geometry[0].artifact_digest = digest(9);
        assert!(matches!(
            translate_c0_coupon_to_fabrication_geometry(&source),
            Err(C0BridgeError::RobotDesignIdentityMismatch | C0BridgeError::GeometryReferenceMismatch)
        ));
    }
}
