// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic, domain-separated fabrication fingerprints.
//!
//! These 256-bit fingerprints are designed for reproducibility, cache keys, and
//! accidental-mismatch detection. They are intentionally **not** cryptographic
//! signatures or tamper-evident evidence.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::mesh::TriangleMesh;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Stable non-cryptographic 256-bit fingerprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StableFingerprint(pub [u64; 4]);

impl StableFingerprint {
    pub fn to_hex(self) -> String {
        format!(
            "{:016x}{:016x}{:016x}{:016x}",
            self.0[0], self.0[1], self.0[2], self.0[3]
        )
    }
}

impl fmt::Display for StableFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

/// Canonical geometry fingerprinting policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeometryFingerprintPolicy {
    /// Position quantization in millimetres.
    pub quantization_mm: f32,
    /// Hard triangle bound before canonical sorting.
    pub max_triangles: usize,
}

impl Default for GeometryFingerprintPolicy {
    fn default() -> Self {
        Self {
            quantization_mm: 1.0e-6,
            max_triangles: 2_000_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GeometryFingerprintError {
    InvalidPolicy(&'static str),
    TriangleBudgetExceeded { actual: usize, maximum: usize },
    InvalidTriangle { triangle: usize },
    CoordinateOutOfRange { vertex: usize },
}

/// Fingerprint mesh geometry independent of vertex-table and triangle ordering.
///
/// Cyclic triangle rotation is normalized while winding is retained, so a
/// globally reversed shell produces a different identity.
pub fn fingerprint_mesh_geometry(
    mesh: &TriangleMesh,
    policy: GeometryFingerprintPolicy,
) -> Result<StableFingerprint, GeometryFingerprintError> {
    validate_policy(policy)?;
    if mesh.indices.len() > policy.max_triangles {
        return Err(GeometryFingerprintError::TriangleBudgetExceeded {
            actual: mesh.indices.len(),
            maximum: policy.max_triangles,
        });
    }

    let mut triangles = Vec::<[[i64; 3]; 3]>::with_capacity(mesh.indices.len());
    for (triangle_index, triangle) in mesh.indices.iter().enumerate() {
        let indices = [
            triangle[0] as usize,
            triangle[1] as usize,
            triangle[2] as usize,
        ];
        if indices.iter().any(|index| *index >= mesh.vertices.len()) {
            return Err(GeometryFingerprintError::InvalidTriangle {
                triangle: triangle_index,
            });
        }
        let a = quantize(
            mesh.vertices[indices[0]],
            policy.quantization_mm,
            indices[0],
        )?;
        let b = quantize(
            mesh.vertices[indices[1]],
            policy.quantization_mm,
            indices[1],
        )?;
        let c = quantize(
            mesh.vertices[indices[2]],
            policy.quantization_mm,
            indices[2],
        )?;
        let rotations = [[a, b, c], [b, c, a], [c, a, b]];
        triangles.push(*rotations.iter().min().expect("three rotations"));
    }
    triangles.sort_unstable();

    let mut builder = FingerprintBuilder::new(b"symthaea.fabrication.mesh-geometry.v1");
    builder.write_f32(policy.quantization_mm);
    builder.write_u64(triangles.len() as u64);
    for triangle in triangles {
        for vertex in triangle {
            for coordinate in vertex {
                builder.write_i64(coordinate);
            }
        }
    }
    Ok(builder.finish())
}

fn validate_policy(policy: GeometryFingerprintPolicy) -> Result<(), GeometryFingerprintError> {
    if !policy.quantization_mm.is_finite() || policy.quantization_mm <= 0.0 {
        return Err(GeometryFingerprintError::InvalidPolicy("quantization_mm"));
    }
    if policy.max_triangles == 0 {
        return Err(GeometryFingerprintError::InvalidPolicy("max_triangles"));
    }
    Ok(())
}

fn quantize(
    vertex: [f32; 3],
    quantization: f32,
    vertex_index: usize,
) -> Result<[i64; 3], GeometryFingerprintError> {
    let mut result = [0i64; 3];
    for axis in 0..3 {
        let scaled = (vertex[axis] as f64 / quantization as f64).round();
        if !scaled.is_finite() || scaled < i64::MIN as f64 || scaled > i64::MAX as f64 {
            return Err(GeometryFingerprintError::CoordinateOutOfRange {
                vertex: vertex_index,
            });
        }
        result[axis] = scaled as i64;
    }
    Ok(result)
}

pub(crate) struct FingerprintBuilder {
    state: [u64; 4],
}

impl FingerprintBuilder {
    pub(crate) fn new(domain: &[u8]) -> Self {
        let mut builder = Self {
            state: [
                0xcbf2_9ce4_8422_2325,
                0x8422_2325_cbf2_9ce4,
                0x9e37_79b1_85eb_ca87,
                0xd6e8_feb8_6659_fd93,
            ],
        };
        builder.write_u64(domain.len() as u64);
        builder.write_bytes(domain);
        builder
    }

    pub(crate) fn write_bytes(&mut self, bytes: &[u8]) {
        const PRIMES: [u64; 4] = [
            0x0000_0100_0000_01b3,
            0x0000_0100_0000_01e7,
            0x9e37_79b1_85eb_ca87,
            0xc2b2_ae3d_27d4_eb4f,
        ];
        for byte in bytes {
            for lane in 0..4 {
                self.state[lane] ^= (*byte as u64).wrapping_add((lane as u64) << 8);
                self.state[lane] = self.state[lane].wrapping_mul(PRIMES[lane]);
                self.state[lane] ^= self.state[lane] >> (29 + lane as u32);
            }
        }
    }

    pub(crate) fn write_bool(&mut self, value: bool) {
        self.write_bytes(&[u8::from(value)]);
    }

    pub(crate) fn write_u16(&mut self, value: u16) {
        self.write_bytes(&value.to_le_bytes());
    }

    pub(crate) fn write_u64(&mut self, value: u64) {
        self.write_bytes(&value.to_le_bytes());
    }

    pub(crate) fn write_i64(&mut self, value: i64) {
        self.write_bytes(&value.to_le_bytes());
    }

    pub(crate) fn write_f32(&mut self, value: f32) {
        self.write_bytes(&value.to_bits().to_le_bytes());
    }

    pub(crate) fn write_f64(&mut self, value: f64) {
        self.write_bytes(&value.to_bits().to_le_bytes());
    }

    pub(crate) fn write_str(&mut self, value: &str) {
        self.write_u64(value.len() as u64);
        self.write_bytes(value.as_bytes());
    }

    pub(crate) fn write_fingerprint(&mut self, fingerprint: StableFingerprint) {
        for word in fingerprint.0 {
            self.write_u64(word);
        }
    }

    pub(crate) fn finish(mut self) -> StableFingerprint {
        for lane in 0..4 {
            self.state[lane] ^= self.state[lane] >> 33;
            self.state[lane] = self.state[lane].wrapping_mul(0xff51_afd7_ed55_8ccd);
            self.state[lane] ^= self.state[lane] >> 33;
            self.state[lane] = self.state[lane].wrapping_mul(0xc4ce_b9fe_1a85_ec53);
            self.state[lane] ^= self.state[lane] >> 33;
        }
        StableFingerprint(self.state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csg::CSGNode;
    use crate::mesh::resolve_to_mesh;

    #[test]
    fn geometry_identity_ignores_storage_order() {
        let mesh = resolve_to_mesh(&CSGNode::cube());
        let expected =
            fingerprint_mesh_geometry(&mesh, GeometryFingerprintPolicy::default()).unwrap();

        let mut reordered = mesh.clone();
        reordered.vertices.reverse();
        reordered.normals.reverse();
        let last = reordered.vertices.len() - 1;
        for triangle in &mut reordered.indices {
            for index in triangle {
                *index = (last - *index as usize) as u32;
            }
        }
        reordered.indices.reverse();
        let actual =
            fingerprint_mesh_geometry(&reordered, GeometryFingerprintPolicy::default()).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn geometry_identity_retains_winding() {
        let mesh = resolve_to_mesh(&CSGNode::cube());
        let expected =
            fingerprint_mesh_geometry(&mesh, GeometryFingerprintPolicy::default()).unwrap();
        let mut reversed = mesh;
        for triangle in &mut reversed.indices {
            triangle.swap(1, 2);
        }
        let actual =
            fingerprint_mesh_geometry(&reversed, GeometryFingerprintPolicy::default()).unwrap();
        assert_ne!(actual, expected);
    }

    #[test]
    fn hex_encoding_is_fixed_width() {
        let fingerprint = FingerprintBuilder::new(b"test").finish();
        assert_eq!(fingerprint.to_hex().len(), 64);
    }
}

use crate::infill::InfillPattern;
use crate::machine::{MachineProfile, ValidatedGCode};
use crate::manufacturability::{MinimumFeaturePolicy, MinimumFeatureReport};
use crate::process::{FabricationProcessPolicy, ProcessPreparationReport, ProcessViolation};
use crate::qualification::ManufacturingReadyMesh;
use crate::slicer::{SliceConfig, SliceLayer};
use crate::toolpath::{GCodeCommand, GCodeProgram, ToolpathConfig};

/// Fingerprint retained process-preparation policy.
pub fn fingerprint_process_policy(policy: &FabricationProcessPolicy) -> StableFingerprint {
    let mut builder = FingerprintBuilder::new(b"symthaea.fabrication.process-policy.v1");
    builder.write_f32(policy.build_plate_z_mm);
    builder.write_f32(policy.placement_tolerance_mm);
    builder.write_bool(policy.require_single_component);
    builder.write_bool(policy.allow_sacrificial_supports);
    let support = policy.support;
    builder.write_f32(support.build_plate_z_mm);
    builder.write_f32(support.max_overhang_from_vertical_degrees);
    builder.write_f32(support.contact_tolerance_mm);
    builder.write_f32(support.interface_gap_mm);
    builder.write_f32(support.column_width_mm);
    builder.write_f32(support.column_pitch_mm);
    builder.write_u64(support.max_columns as u64);
    builder.finish()
}

pub fn fingerprint_minimum_feature_policy(policy: &MinimumFeaturePolicy) -> StableFingerprint {
    let mut builder = FingerprintBuilder::new(b"symthaea.fabrication.minimum-feature-policy.v1");
    builder.write_f32(policy.minimum_wall_thickness_mm);
    builder.write_f32(policy.ray_origin_epsilon_mm);
    builder.write_f32(policy.intersection_epsilon);
    builder.write_u64(policy.max_ray_triangle_tests as u64);
    builder.finish()
}

/// Fingerprint process-preparation evidence, including the synthesized support plan.
pub fn fingerprint_process_report(report: &ProcessPreparationReport) -> StableFingerprint {
    let mut builder = FingerprintBuilder::new(b"symthaea.fabrication.process-report.v1");
    builder.write_f32(report.minimum_z_mm);
    builder.write_f32(report.maximum_z_mm);
    builder.write_u64(report.support_plan.overhang_triangles.len() as u64);
    for triangle in &report.support_plan.overhang_triangles {
        builder.write_u64(*triangle as u64);
    }
    builder.write_f32(report.support_plan.unsupported_surface_area_mm2);
    builder.write_u64(report.support_plan.columns.len() as u64);
    for column in &report.support_plan.columns {
        builder.write_f32(column.center_xy_mm[0]);
        builder.write_f32(column.center_xy_mm[1]);
        builder.write_f32(column.bottom_z_mm);
        builder.write_f32(column.top_z_mm);
        builder.write_f32(column.width_mm);
    }
    builder.write_bool(report.support_plan.truncated);
    builder.write_u64(report.violations.len() as u64);
    for violation in &report.violations {
        match violation {
            ProcessViolation::NonFinitePolicy(field) => {
                builder.write_u16(1);
                builder.write_str(field);
            }
            ProcessViolation::GeometryBelowBuildPlate { vertex, z_mm } => {
                builder.write_u16(2);
                builder.write_u64(*vertex as u64);
                builder.write_f32(*z_mm);
            }
            ProcessViolation::MultipleComponents { count } => {
                builder.write_u16(3);
                builder.write_u64(*count as u64);
            }
            ProcessViolation::SupportRequiredButDisabled => builder.write_u16(4),
            ProcessViolation::SupportPlanTruncated => builder.write_u16(5),
        }
    }
    builder.finish()
}

/// Fingerprint bounded minimum-feature evidence.
pub fn fingerprint_minimum_feature_report(report: &MinimumFeatureReport) -> StableFingerprint {
    let mut builder = FingerprintBuilder::new(b"symthaea.fabrication.minimum-feature-report.v1");
    builder.write_u64(report.rays_cast as u64);
    builder.write_u64(report.ray_triangle_tests as u64);
    builder.write_bool(report.minimum_observed_thickness_mm.is_some());
    if let Some(thickness) = report.minimum_observed_thickness_mm {
        builder.write_f32(thickness);
    }
    builder.write_u64(report.thin_source_triangles.len() as u64);
    for triangle in &report.thin_source_triangles {
        builder.write_u64(*triangle as u64);
    }
    builder.write_u64(report.unresolved_source_triangles.len() as u64);
    for triangle in &report.unresolved_source_triangles {
        builder.write_u64(*triangle as u64);
    }
    builder.write_bool(report.truncated);
    builder.finish()
}

/// Fingerprint the exact ordered slice-layer artifact.
pub fn fingerprint_slice_layers(layers: &[SliceLayer]) -> StableFingerprint {
    let mut builder = FingerprintBuilder::new(b"symthaea.fabrication.slice-layers.v1");
    builder.write_u64(layers.len() as u64);
    for layer in layers {
        builder.write_f32(layer.z);
        builder.write_u64(layer.outer_contours.len() as u64);
        for contour in &layer.outer_contours {
            builder.write_u64(contour.points.len() as u64);
            for point in &contour.points {
                builder.write_f32(point.x);
                builder.write_f32(point.y);
            }
        }
        builder.write_u64(layer.inner_contours.len() as u64);
        for contour in &layer.inner_contours {
            builder.write_u64(contour.points.len() as u64);
            for point in &contour.points {
                builder.write_f32(point.x);
                builder.write_f32(point.y);
            }
        }
        builder.write_u64(layer.infill_lines.len() as u64);
        for segment in &layer.infill_lines {
            builder.write_f32(segment.start.x);
            builder.write_f32(segment.start.y);
            builder.write_f32(segment.end.x);
            builder.write_f32(segment.end.y);
        }
    }
    builder.finish()
}

pub fn fingerprint_slice_config(config: &SliceConfig) -> StableFingerprint {
    let mut builder = FingerprintBuilder::new(b"symthaea.fabrication.slice-config.v1");
    builder.write_f32(config.nozzle_diameter);
    builder.write_f32(config.tolerance);
    builder.write_f32(config.layer_height);
    match &config.infill {
        None => builder.write_bool(false),
        Some(infill) => {
            builder.write_bool(true);
            builder.write_u16(match infill.pattern {
                InfillPattern::Rectilinear => 1,
                InfillPattern::Grid => 2,
                InfillPattern::Honeycomb => 3,
            });
            builder.write_f32(infill.density);
            builder.write_f32(infill.angle_degrees);
        }
    }
    builder.finish()
}

pub fn fingerprint_toolpath_config(config: &ToolpathConfig) -> StableFingerprint {
    let mut builder = FingerprintBuilder::new(b"symthaea.fabrication.toolpath-config.v1");
    builder.write_f32(config.print_speed_mm_s);
    builder.write_f32(config.travel_speed_mm_s);
    builder.write_f32(config.retract_distance_mm);
    builder.write_f32(config.retract_speed_mm_s);
    builder.write_f32(config.extrusion_width_mm);
    builder.write_f32(config.filament_diameter_mm);
    builder.write_u16(config.bed_temp_c);
    builder.write_u16(config.nozzle_temp_c);
    builder.finish()
}

pub fn fingerprint_machine_profile(profile: &MachineProfile) -> StableFingerprint {
    let mut builder = FingerprintBuilder::new(b"symthaea.fabrication.machine-profile.v1");
    builder.write_str(&profile.name);
    for value in profile.build_min_mm {
        builder.write_f32(value);
    }
    for value in profile.build_max_mm {
        builder.write_f32(value);
    }
    builder.write_f32(profile.max_feed_rate_mm_min);
    builder.write_u16(profile.max_nozzle_temp_c);
    builder.write_u16(profile.max_bed_temp_c);
    builder.write_f32(profile.max_retraction_mm);
    builder.write_bool(profile.require_homing);
    builder.finish()
}

pub fn fingerprint_gcode_program(program: &GCodeProgram) -> StableFingerprint {
    let mut builder = FingerprintBuilder::new(b"symthaea.fabrication.gcode-program.v1");
    builder.write_u64(program.commands.len() as u64);
    for command in &program.commands {
        match command {
            GCodeCommand::G0 { x, y, z, f } => {
                builder.write_u16(1);
                write_optional_f32(&mut builder, *x);
                write_optional_f32(&mut builder, *y);
                write_optional_f32(&mut builder, *z);
                write_optional_f32(&mut builder, *f);
            }
            GCodeCommand::G1 { x, y, z, e, f } => {
                builder.write_u16(2);
                write_optional_f32(&mut builder, *x);
                write_optional_f32(&mut builder, *y);
                write_optional_f32(&mut builder, *z);
                write_optional_f32(&mut builder, *e);
                write_optional_f32(&mut builder, *f);
            }
            GCodeCommand::G28 => builder.write_u16(3),
            GCodeCommand::M104 { s } => {
                builder.write_u16(4);
                builder.write_u16(*s);
            }
            GCodeCommand::M109 { s } => {
                builder.write_u16(5);
                builder.write_u16(*s);
            }
            GCodeCommand::M140 { s } => {
                builder.write_u16(6);
                builder.write_u16(*s);
            }
            GCodeCommand::M190 { s } => {
                builder.write_u16(7);
                builder.write_u16(*s);
            }
            GCodeCommand::Comment(text) => {
                builder.write_u16(8);
                builder.write_str(text);
            }
        }
    }
    builder.write_f64(program.total_extrusion_mm);
    builder.finish()
}

fn write_optional_f32(builder: &mut FingerprintBuilder, value: Option<f32>) {
    builder.write_bool(value.is_some());
    if let Some(value) = value {
        builder.write_f32(value);
    }
}

/// Deterministic identity chain for one qualified fabrication job.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FabricationManifest {
    pub schema_version: String,
    pub geometry: StableFingerprint,
    pub process_policy: StableFingerprint,
    pub process_evidence: StableFingerprint,
    pub minimum_feature_policy: StableFingerprint,
    pub minimum_feature_evidence: StableFingerprint,
    pub slice_config: StableFingerprint,
    pub slice_layers: StableFingerprint,
    pub toolpath_config: StableFingerprint,
    pub machine_profile: StableFingerprint,
    pub gcode_program: StableFingerprint,
    pub pipeline: StableFingerprint,
    pub layer_count: usize,
    pub command_count: usize,
    pub total_extrusion_mm: f64,
}

/// Serialize a manifest in its schema-defined field order.
///
/// `FabricationManifest` is a struct rather than a map, so serde_json preserves
/// a deterministic field order. Changing fields requires a schema-version bump.
pub fn canonical_manifest_bytes(
    manifest: &FabricationManifest,
) -> Result<Vec<u8>, serde_json::Error> {
    serde_json::to_vec(manifest)
}

/// Cryptographically digest the canonical manifest representation.
pub fn digest_fabrication_manifest(
    manifest: &FabricationManifest,
) -> Result<Sha256Digest, serde_json::Error> {
    let canonical = canonical_manifest_bytes(manifest)?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.manifest-digest.v1\0");
    hasher.update(&canonical);
    Ok(hasher.finalize())
}

#[derive(Debug, Clone, PartialEq)]
pub enum FabricationManifestError {
    Geometry(GeometryFingerprintError),
    MachineProfileMismatch {
        validated: MachineProfile,
        supplied: MachineProfile,
    },
}

/// Build a deterministic manifest from a fully qualified and machine-validated job.
pub fn build_fabrication_manifest(
    geometry: &ManufacturingReadyMesh,
    slice_config: &SliceConfig,
    toolpath_config: &ToolpathConfig,
    machine_profile: &MachineProfile,
    layers: &[SliceLayer],
    gcode: &ValidatedGCode,
) -> Result<FabricationManifest, FabricationManifestError> {
    if gcode.profile() != machine_profile {
        return Err(FabricationManifestError::MachineProfileMismatch {
            validated: gcode.profile().clone(),
            supplied: machine_profile.clone(),
        });
    }
    let geometry_fingerprint =
        fingerprint_mesh_geometry(geometry.mesh(), GeometryFingerprintPolicy::default())
            .map_err(FabricationManifestError::Geometry)?;
    let process_policy = fingerprint_process_policy(geometry.process().policy());
    let process_evidence = fingerprint_process_report(geometry.process().report());
    let minimum_feature_policy =
        fingerprint_minimum_feature_policy(geometry.minimum_feature_policy());
    let minimum_feature_evidence =
        fingerprint_minimum_feature_report(geometry.minimum_feature_report());
    let slice_config_fingerprint = fingerprint_slice_config(slice_config);
    let slice_layers = fingerprint_slice_layers(layers);
    let toolpath_config_fingerprint = fingerprint_toolpath_config(toolpath_config);
    let machine_profile_fingerprint = fingerprint_machine_profile(machine_profile);
    let gcode_program = fingerprint_gcode_program(gcode.program());

    let mut pipeline = FingerprintBuilder::new(b"symthaea.fabrication.pipeline-manifest.v1");
    for fingerprint in [
        geometry_fingerprint,
        process_policy,
        process_evidence,
        minimum_feature_policy,
        minimum_feature_evidence,
        slice_config_fingerprint,
        slice_layers,
        toolpath_config_fingerprint,
        machine_profile_fingerprint,
        gcode_program,
    ] {
        pipeline.write_fingerprint(fingerprint);
    }
    pipeline.write_u64(layers.len() as u64);
    pipeline.write_u64(gcode.program().commands.len() as u64);
    pipeline.write_f64(gcode.program().total_extrusion_mm);

    Ok(FabricationManifest {
        schema_version: "symthaea.fabrication.manifest.v1".into(),
        geometry: geometry_fingerprint,
        process_policy,
        process_evidence,
        minimum_feature_policy,
        minimum_feature_evidence,
        slice_config: slice_config_fingerprint,
        slice_layers,
        toolpath_config: toolpath_config_fingerprint,
        machine_profile: machine_profile_fingerprint,
        gcode_program,
        pipeline: pipeline.finish(),
        layer_count: layers.len(),
        command_count: gcode.program().commands.len(),
        total_extrusion_mm: gcode.program().total_extrusion_mm,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifestMismatch {
    SchemaVersion,
    Geometry,
    ProcessPolicy,
    ProcessEvidence,
    MinimumFeaturePolicy,
    MinimumFeatureEvidence,
    SliceConfig,
    SliceLayers,
    ToolpathConfig,
    MachineProfile,
    GCodeProgram,
    Pipeline,
    LayerCount,
    CommandCount,
    TotalExtrusion,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestVerificationReport {
    pub mismatches: Vec<ManifestMismatch>,
}

impl ManifestVerificationReport {
    pub fn matches(&self) -> bool {
        self.mismatches.is_empty()
    }
}

/// Recompute and compare every manifest identity from live pipeline values.
pub fn verify_fabrication_manifest(
    manifest: &FabricationManifest,
    geometry: &ManufacturingReadyMesh,
    slice_config: &SliceConfig,
    toolpath_config: &ToolpathConfig,
    machine_profile: &MachineProfile,
    layers: &[SliceLayer],
    gcode: &ValidatedGCode,
) -> Result<ManifestVerificationReport, FabricationManifestError> {
    let actual = build_fabrication_manifest(
        geometry,
        slice_config,
        toolpath_config,
        machine_profile,
        layers,
        gcode,
    )?;
    let mut mismatches = Vec::new();
    if manifest.schema_version != actual.schema_version {
        mismatches.push(ManifestMismatch::SchemaVersion);
    }
    for (different, mismatch) in [
        (
            manifest.geometry != actual.geometry,
            ManifestMismatch::Geometry,
        ),
        (
            manifest.process_policy != actual.process_policy,
            ManifestMismatch::ProcessPolicy,
        ),
        (
            manifest.process_evidence != actual.process_evidence,
            ManifestMismatch::ProcessEvidence,
        ),
        (
            manifest.minimum_feature_policy != actual.minimum_feature_policy,
            ManifestMismatch::MinimumFeaturePolicy,
        ),
        (
            manifest.minimum_feature_evidence != actual.minimum_feature_evidence,
            ManifestMismatch::MinimumFeatureEvidence,
        ),
        (
            manifest.slice_config != actual.slice_config,
            ManifestMismatch::SliceConfig,
        ),
        (
            manifest.slice_layers != actual.slice_layers,
            ManifestMismatch::SliceLayers,
        ),
        (
            manifest.toolpath_config != actual.toolpath_config,
            ManifestMismatch::ToolpathConfig,
        ),
        (
            manifest.machine_profile != actual.machine_profile,
            ManifestMismatch::MachineProfile,
        ),
        (
            manifest.gcode_program != actual.gcode_program,
            ManifestMismatch::GCodeProgram,
        ),
        (
            manifest.pipeline != actual.pipeline,
            ManifestMismatch::Pipeline,
        ),
        (
            manifest.layer_count != actual.layer_count,
            ManifestMismatch::LayerCount,
        ),
        (
            manifest.command_count != actual.command_count,
            ManifestMismatch::CommandCount,
        ),
        (
            manifest.total_extrusion_mm.to_bits() != actual.total_extrusion_mm.to_bits(),
            ManifestMismatch::TotalExtrusion,
        ),
    ] {
        if different {
            mismatches.push(mismatch);
        }
    }
    Ok(ManifestVerificationReport { mismatches })
}

#[cfg(test)]
mod manifest_tests {
    use super::*;
    #[test]
    fn canonical_manifest_digest_detects_tampering() {
        let fingerprint = StableFingerprint([1, 2, 3, 4]);
        let manifest = FabricationManifest {
            schema_version: "symthaea.fabrication.manifest.v1".into(),
            geometry: fingerprint,
            process_policy: fingerprint,
            process_evidence: fingerprint,
            minimum_feature_policy: fingerprint,
            minimum_feature_evidence: fingerprint,
            slice_config: fingerprint,
            slice_layers: fingerprint,
            toolpath_config: fingerprint,
            machine_profile: fingerprint,
            gcode_program: fingerprint,
            pipeline: fingerprint,
            layer_count: 1,
            command_count: 2,
            total_extrusion_mm: 3.0,
        };
        let first = digest_fabrication_manifest(&manifest).unwrap();
        let second = digest_fabrication_manifest(&manifest).unwrap();
        assert_eq!(first, second);

        let mut changed = manifest;
        changed.command_count += 1;
        assert_ne!(digest_fabrication_manifest(&changed).unwrap(), first);
    }

    use crate::csg::{CSGNode, Transform3D};
    use crate::machine::ValidatedGCode;
    use crate::manufacturability::MinimumFeaturePolicy;
    use crate::mesh::resolve_to_mesh;
    use crate::process::FabricationProcessPolicy;
    use crate::qualification::ManufacturingReadyMesh;
    use crate::slicer::{SliceConfig, slice_manufacturing_ready};
    use crate::toolpath::{ToolpathConfig, try_generate_gcode};

    fn qualified_job() -> (
        ManufacturingReadyMesh,
        SliceConfig,
        ToolpathConfig,
        MachineProfile,
        Vec<SliceLayer>,
        ValidatedGCode,
    ) {
        let mesh = resolve_to_mesh(&CSGNode::cube().with_transform(Transform3D {
            translate: [10.0, 10.0, 0.5],
            ..Default::default()
        }));
        let ready = ManufacturingReadyMesh::try_new(
            mesh,
            FabricationProcessPolicy::default(),
            MinimumFeaturePolicy::default(),
        )
        .unwrap();
        let slice_config = SliceConfig::default();
        let toolpath_config = ToolpathConfig::default();
        let machine = MachineProfile::default();
        let layers = slice_manufacturing_ready(&ready, &slice_config).unwrap();
        let program = try_generate_gcode(&layers, &slice_config, &toolpath_config).unwrap();
        let validated = ValidatedGCode::try_new(program, &machine).unwrap();
        (
            ready,
            slice_config,
            toolpath_config,
            machine,
            layers,
            validated,
        )
    }

    #[test]
    fn manifest_round_trip_verifies() {
        let (ready, slice, toolpath, machine, layers, gcode) = qualified_job();
        let manifest =
            build_fabrication_manifest(&ready, &slice, &toolpath, &machine, &layers, &gcode)
                .unwrap();
        let verification = verify_fabrication_manifest(
            &manifest, &ready, &slice, &toolpath, &machine, &layers, &gcode,
        )
        .unwrap();
        assert!(verification.matches());
        assert_eq!(manifest.pipeline.to_hex().len(), 64);
    }

    #[test]
    fn manifest_detects_changed_slice_policy() {
        let (ready, slice, toolpath, machine, layers, gcode) = qualified_job();
        let manifest =
            build_fabrication_manifest(&ready, &slice, &toolpath, &machine, &layers, &gcode)
                .unwrap();
        let mut changed_slice = slice.clone();
        changed_slice.layer_height = 0.1;
        let verification = verify_fabrication_manifest(
            &manifest,
            &ready,
            &changed_slice,
            &toolpath,
            &machine,
            &layers,
            &gcode,
        )
        .unwrap();
        assert!(
            verification
                .mismatches
                .contains(&ManifestMismatch::SliceConfig)
        );
        assert!(
            verification
                .mismatches
                .contains(&ManifestMismatch::Pipeline)
        );
    }
}
