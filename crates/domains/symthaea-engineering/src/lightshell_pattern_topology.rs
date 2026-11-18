// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Lightshell robot-panel to apparel-pattern topology adapter.
//!
//! Consumes source-native pattern/artifact identities and the Robot Garment
//! Compiler's panel IDs. Generic pattern, DXF/Knitout, 2D/3D correspondence
//! and machine semantics remain owned by APPAREL-INTEROP (#11750).
//! This module validates robot-specific panel/seam/articulation/service
//! topology. It does not author patterns, solve cloth, run machines or
//! qualify physical fit/manufacture.

use serde::Deserialize;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PatternTopologyManifest {
    pub schema_version: String,
    pub topology_id: String,
    pub state: String,
    pub compiler_binding: CompilerBinding,
    pub interop_binding: InteropBinding,
    pub source_artifacts: Vec<SourceArtifact>,
    pub compiler_panels: Vec<CompilerPanelRef>,
    pub compiler_keepouts: Vec<CompilerKeepoutRef>,
    pub panels: Vec<TopologyPanel>,
    pub boundary_edges: Vec<BoundaryEdge>,
    pub connections: Vec<PanelConnection>,
    pub articulation_features: Vec<ArticulationFeature>,
    pub service_features: Vec<ServiceFeature>,
    pub semantic_obligations: Vec<SemanticObligation>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompilerBinding {
    pub compiler_issue_ref: String,
    pub compiler_revision: String,
    pub compilation_id: String,
    pub robot_id: String,
    pub robot_article_revision: Option<String>,
    pub geometry_state: String,
    pub geometry_digest: Option<String>,
    pub geometry_units: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InteropBinding {
    pub interop_issue_ref: String,
    pub interop_contract_revision: String,
    pub sidecar_ref: String,
    pub sidecar_digest: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceArtifact {
    pub artifact_id: String,
    pub source_format: String,
    pub format_version: String,
    pub artifact_ref: String,
    pub artifact_digest: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompilerPanelRef {
    pub panel_id: String,
    pub zone_id: String,
    pub coordinate_frame_ref: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompilerKeepoutRef {
    pub keepout_id: String,
    pub kind: String,
    pub affected_zone_ids: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TopologyPanel {
    pub panel_id: String,
    pub zone_id: String,
    pub compiler_panel_id: String,
    pub source_artifact_id: String,
    pub source_piece_id: String,
    pub pattern_revision_ref: String,
    pub pattern_piece_digest: String,
    pub surface_3d_ref: String,
    pub correspondence_ref: String,
    pub correspondence_state: String,
    pub coordinate_frame_ref: String,
    pub units: String,
    pub grain_direction: String,
    pub material_orientation_ref: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundaryEdge {
    pub edge_id: String,
    pub panel_id: String,
    pub role: String,
    pub curve_ref: String,
    pub length_mm: f64,
    pub length_tolerance_mm: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PanelConnection {
    pub connection_id: String,
    pub connection_kind: String,
    pub edge_refs: Vec<String>,
    pub construction_route_ref: String,
    pub path_ref: String,
    pub max_length_mismatch_mm: f64,
    pub construction_revision_ref: String,
    pub interface_design_ref: Option<String>,
    pub retention_evidence_ref: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArticulationFeature {
    pub feature_id: String,
    pub feature_kind: String,
    pub zone_id: String,
    pub joint_ref: String,
    pub motion_envelope_ref: String,
    pub affected_panel_ids: Vec<String>,
    pub feature_region_ref: String,
    pub test_plan_ref: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceFeature {
    pub feature_id: String,
    pub panel_id: String,
    pub keepout_ref: String,
    pub opening_edge_refs: Vec<String>,
    pub access_mode: String,
    pub access_path_ref: String,
    pub service_test_plan_ref: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticObligation {
    pub semantic_id: String,
    pub preservation_state: String,
    pub source_location_ref: Option<String>,
    pub sidecar_location_ref: Option<String>,
    pub evidence_ref: Option<String>,
    pub external_owner_ref: Option<String>,
    pub declared_out_of_scope: bool,
    pub scope_disposition_ref: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    pub code: &'static str,
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub findings: Vec<Finding>,
    pub panel_count: usize,
    pub boundary_edge_count: usize,
    pub connection_count: usize,
    pub articulation_feature_count: usize,
    pub service_feature_count: usize,
    pub semantic_obligation_count: usize,
    pub semantic_coverage_complete: bool,
    pub topology_preflight_admitted: bool,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.findings.is_empty()
    }

    /// A topology preflight is a software consistency result over exact
    /// source/robot references; it is not physical garment qualification.
    pub fn can_preflight(&self) -> bool {
        self.is_valid() && self.topology_preflight_admitted
    }
}

pub fn parse_and_validate(input: &str) -> Result<ValidationReport, serde_json::Error> {
    let manifest: PatternTopologyManifest = serde_json::from_str(input)?;
    Ok(manifest.validate())
}

impl PatternTopologyManifest {
    pub fn validate(&self) -> ValidationReport {
        let mut findings = Vec::new();
        let mut artifacts: HashMap<&str, &SourceArtifact> = HashMap::new();
        let mut compiler_panels: HashMap<&str, &CompilerPanelRef> = HashMap::new();
        let mut keepouts: HashMap<&str, &CompilerKeepoutRef> = HashMap::new();
        let mut panels: HashMap<&str, &TopologyPanel> = HashMap::new();
        let mut edges: HashMap<&str, &BoundaryEdge> = HashMap::new();
        let mut connections = HashSet::new();
        let mut used_edges: HashMap<&str, &str> = HashMap::new();
        let mut source_piece_ids = HashSet::new();
        let mut feature_ids = HashSet::new();
        let mut semantic_ids = HashSet::new();
        let mut semantic_coverage_complete = true;

        if self.schema_version != "0.1.0" {
            push(&mut findings, "unsupported_schema_version", "schema_version",
                "Only Lightshell pattern-topology schema version 0.1.0 is admitted.");
        }
        if is_blank(&self.topology_id) || is_blank(&self.compiler_binding.compilation_id)
            || is_blank(&self.compiler_binding.robot_id)
        {
            push(&mut findings, "missing_topology_identity", "topology_id/compiler_binding",
                "Topology, compilation and robot identities must be non-empty.");
        }
        if !matches!(self.state.as_str(), "proposed" | "source_bound" | "topology_preflighted") {
            push(&mut findings, "invalid_topology_state", "state",
                "Topology state must be proposed, source_bound or topology_preflighted.");
        }
        if !matches!(self.compiler_binding.geometry_state.as_str(), "synthetic_fixture" | "cad_revision_bound") {
            push(&mut findings, "invalid_geometry_state", "compiler_binding.geometry_state",
                "Compiler geometry must be synthetic_fixture or cad_revision_bound.");
        }
        if self.compiler_binding.geometry_units != "mm" {
            push(&mut findings, "unsupported_geometry_units", "compiler_binding.geometry_units",
                "The P0 topology adapter accepts millimetres only; implicit unit conversion is prohibited.");
        }
        if self.compiler_binding.geometry_state == "cad_revision_bound" {
            if is_blank_option(&self.compiler_binding.robot_article_revision)
                || !self.compiler_binding.geometry_digest.as_deref().is_some_and(is_sha256_digest)
                || is_blank(&self.compiler_binding.compiler_revision)
            {
                push(&mut findings, "cad_geometry_not_revision_bound", "compiler_binding",
                    "CAD-bound geometry requires exact robot article revision, compiler revision and SHA-256 geometry digest.");
            }
        }
        if is_blank(&self.compiler_binding.compiler_issue_ref)
            || is_blank(&self.interop_binding.interop_issue_ref)
            || is_blank(&self.interop_binding.interop_contract_revision)
            || is_blank(&self.interop_binding.sidecar_ref)
            || !is_sha256_digest(&self.interop_binding.sidecar_digest)
        {
            push(&mut findings, "interop_or_compiler_binding_incomplete", "compiler_binding/interop_binding",
                "Compiler and APPAREL-INTEROP sidecar bindings must include exact source/contract identities and valid digests.");
        }

        for (index, artifact) in self.source_artifacts.iter().enumerate() {
            let path = format!("source_artifacts[{index}]");
            if is_blank(&artifact.artifact_id) || is_blank(&artifact.format_version) || is_blank(&artifact.artifact_ref) {
                push(&mut findings, "source_artifact_identity_missing", &path,
                    "Source artifacts require ID, format version and source reference.");
            }
            if !matches!(artifact.source_format.as_str(), "dxf" | "obj" | "usd" | "knitout" | "native_pattern") {
                push(&mut findings, "unsupported_source_format", format!("{path}.source_format"),
                    "Unsupported source-format profile; add it through APPAREL-INTEROP review.");
            }
            if !is_sha256_digest(&artifact.artifact_digest) {
                push(&mut findings, "invalid_source_artifact_digest", format!("{path}.artifact_digest"),
                    "Source artifact digest must use sha256: followed by 64 hexadecimal characters.");
            }
            if artifact.source_format == "knitout" && artifact.format_version != "0.6" {
                push(&mut findings, "unsupported_knitout_version", format!("{path}.format_version"),
                    "This adapter currently recognizes Knitout v0.6 only; target-machine execution capability remains external.");
            }
            if artifacts.insert(artifact.artifact_id.as_str(), artifact).is_some() {
                push(&mut findings, "duplicate_source_artifact_id", format!("{path}.artifact_id"),
                    "Source artifact IDs must be unique.");
            }
        }

        for (index, compiler_panel) in self.compiler_panels.iter().enumerate() {
            let path = format!("compiler_panels[{index}]");
            if is_blank(&compiler_panel.panel_id) || is_blank(&compiler_panel.zone_id) || is_blank(&compiler_panel.coordinate_frame_ref) {
                push(&mut findings, "compiler_panel_identity_missing", &path,
                    "Compiler panel, zone and coordinate-frame identities must be non-empty.");
            }
            if compiler_panels.insert(compiler_panel.panel_id.as_str(), compiler_panel).is_some() {
                push(&mut findings, "duplicate_compiler_panel_id", format!("{path}.panel_id"),
                    "Compiler panel IDs must be unique.");
            }
        }

        for (index, keepout) in self.compiler_keepouts.iter().enumerate() {
            let path = format!("compiler_keepouts[{index}]");
            if is_blank(&keepout.keepout_id) || keepout.affected_zone_ids.is_empty() {
                push(&mut findings, "compiler_keepout_identity_missing", &path,
                    "Compiler keep-outs require an identity and at least one affected zone.");
            }
            if !matches!(keepout.kind.as_str(), "service_access" | "emergency_access" | "swept_motion"
                | "cable_route" | "sensor_aperture" | "thermal_vent" | "pinch_shear" | "charging_interface")
            {
                push(&mut findings, "unknown_compiler_keepout_kind", format!("{path}.kind"),
                    "Compiler keep-out kind is not recognized by the topology adapter.");
            }
            if keepouts.insert(keepout.keepout_id.as_str(), keepout).is_some() {
                push(&mut findings, "duplicate_compiler_keepout_id", format!("{path}.keepout_id"),
                    "Compiler keep-out IDs must be unique.");
            }
        }

        for (index, panel) in self.panels.iter().enumerate() {
            let path = format!("panels[{index}]");
            if is_blank(&panel.panel_id) || is_blank(&panel.source_piece_id) || is_blank(&panel.zone_id) {
                push(&mut findings, "pattern_panel_identity_missing", &path,
                    "Pattern panel, source piece and zone identities must be non-empty.");
            }
            if panels.insert(panel.panel_id.as_str(), panel).is_some() {
                push(&mut findings, "duplicate_pattern_panel_id", format!("{path}.panel_id"),
                    "Pattern panel IDs must be unique.");
            }
            if !source_piece_ids.insert((panel.source_artifact_id.as_str(), panel.source_piece_id.as_str())) {
                push(&mut findings, "duplicate_source_piece_identity", format!("{path}.source_piece_id"),
                    "A source artifact/piece identity must map to only one robot panel in this P0 topology.");
            }
            if !artifacts.contains_key(panel.source_artifact_id.as_str()) {
                push(&mut findings, "panel_unknown_source_artifact", format!("{path}.source_artifact_id"),
                    "Pattern panel references an undeclared source artifact.");
            }
            let Some(compiler_panel) = compiler_panels.get(panel.compiler_panel_id.as_str()).copied() else {
                push(&mut findings, "panel_unknown_compiler_panel", format!("{path}.compiler_panel_id"),
                    "Pattern panel must resolve to an exact candidate panel from Robot Garment Compiler.");
                continue;
            };
            if compiler_panel.panel_id != panel.panel_id {
                push(&mut findings, "panel_compiler_identity_mismatch", format!("{path}.compiler_panel_id"),
                    "P0 requires the topology panel ID to match the compiler candidate panel ID exactly.");
            }
            if compiler_panel.zone_id != panel.zone_id {
                push(&mut findings, "panel_compiler_zone_mismatch", format!("{path}.zone_id"),
                    "Pattern and compiler panel must bind the same functional zone.");
            }
            if compiler_panel.coordinate_frame_ref != panel.coordinate_frame_ref {
                push(&mut findings, "panel_compiler_frame_mismatch", format!("{path}.coordinate_frame_ref"),
                    "Pattern and compiler panel must bind the same declared coordinate frame.");
            }
            if panel.units != self.compiler_binding.geometry_units {
                push(&mut findings, "panel_unit_mismatch", format!("{path}.units"),
                    "Pattern and robot geometry units must match exactly.");
            }
            if is_blank(&panel.pattern_revision_ref) || !is_sha256_digest(&panel.pattern_piece_digest)
                || is_blank(&panel.surface_3d_ref) || is_blank(&panel.correspondence_ref)
                || is_blank(&panel.material_orientation_ref)
            {
                push(&mut findings, "pattern_panel_source_binding_incomplete", &path,
                    "Each panel requires exact pattern revision, piece digest, 3D surface, 2D/3D correspondence and material orientation.");
            }
            if !matches!(panel.correspondence_state.as_str(), "verified" | "declared" | "unknown" | "lossy") {
                push(&mut findings, "invalid_correspondence_state", format!("{path}.correspondence_state"),
                    "Correspondence state must be verified, declared, unknown or lossy.");
            }
            if panel.correspondence_state == "unknown" || panel.correspondence_state == "lossy" {
                semantic_coverage_complete = false;
                if self.state != "proposed" {
                    push(&mut findings, "panel_correspondence_not_preserved", format!("{path}.correspondence_state"),
                        "Source-bound topology cannot contain unknown or lossy 2D/3D correspondence.");
                }
            }
            if self.state == "topology_preflighted" && panel.correspondence_state != "verified" {
                semantic_coverage_complete = false;
                push(&mut findings, "topology_preflight_requires_verified_correspondence", format!("{path}.correspondence_state"),
                    "Topology preflight requires verified, not merely declared, 2D/3D panel correspondence.");
            }
            if !matches!(panel.grain_direction.as_str(), "warp" | "weft" | "course" | "wale" | "bias" | "none" | "unknown") {
                push(&mut findings, "invalid_grain_direction", format!("{path}.grain_direction"),
                    "Unknown grain/material direction vocabulary.");
            }
            if panel.grain_direction == "unknown" {
                semantic_coverage_complete = false;
                if self.state != "proposed" {
                    push(&mut findings, "grain_direction_unknown", format!("{path}.grain_direction"),
                        "Source-bound topology requires an explicit grain/course/wale/bias/none declaration.");
                }
            }
        }

        for (index, edge) in self.boundary_edges.iter().enumerate() {
            let path = format!("boundary_edges[{index}]");
            if is_blank(&edge.edge_id) || is_blank(&edge.curve_ref) {
                push(&mut findings, "boundary_edge_identity_missing", &path,
                    "Boundary edges require globally unique identity and source curve reference.");
            }
            if !panels.contains_key(edge.panel_id.as_str()) {
                push(&mut findings, "boundary_edge_unknown_panel", format!("{path}.panel_id"),
                    "Boundary edge references an undeclared topology panel.");
            }
            if !matches!(edge.role.as_str(), "seam" | "attachment" | "service_opening" | "articulation_boundary" | "free_boundary") {
                push(&mut findings, "unknown_boundary_edge_role", format!("{path}.role"),
                    "Unknown boundary-edge role.");
            }
            if !edge.length_mm.is_finite() || edge.length_mm <= 0.0
                || !edge.length_tolerance_mm.is_finite() || edge.length_tolerance_mm < 0.0
            {
                push(&mut findings, "invalid_boundary_edge_length", &path,
                    "Boundary edge length must be positive finite mm and tolerance non-negative finite mm.");
            }
            if edges.insert(edge.edge_id.as_str(), edge).is_some() {
                push(&mut findings, "duplicate_boundary_edge_id", format!("{path}.edge_id"),
                    "Boundary-edge IDs must be globally unique.");
            }
        }

        for (index, connection) in self.connections.iter().enumerate() {
            let path = format!("connections[{index}]");
            if is_blank(&connection.connection_id) {
                push(&mut findings, "connection_identity_missing", &path,
                    "Connection ID must be non-empty.");
            }
            if !connections.insert(connection.connection_id.as_str()) {
                push(&mut findings, "duplicate_connection_id", format!("{path}.connection_id"),
                    "Connection IDs must be unique.");
            }
            if !matches!(connection.connection_kind.as_str(), "sewn_seam" | "knit_integrated" | "bonded_seam" | "removable_attachment") {
                push(&mut findings, "unknown_connection_kind", format!("{path}.connection_kind"),
                    "Unknown panel connection construction.");
            }
            if connection.edge_refs.len() != 2 {
                push(&mut findings, "connection_requires_two_edges", format!("{path}.edge_refs"),
                    "P0 connections require exactly two boundary edges.");
                continue;
            }
            if is_blank(&connection.construction_route_ref) || is_blank(&connection.path_ref)
                || is_blank(&connection.construction_revision_ref)
            {
                push(&mut findings, "connection_route_binding_missing", &path,
                    "Every connection requires exact construction route, path and construction revision references.");
            }
            if !connection.max_length_mismatch_mm.is_finite() || connection.max_length_mismatch_mm < 0.0 {
                push(&mut findings, "invalid_connection_length_tolerance", format!("{path}.max_length_mismatch_mm"),
                    "Maximum connection edge-length mismatch must be finite and non-negative.");
            }
            let edge_a = edges.get(connection.edge_refs[0].as_str()).copied();
            let edge_b = edges.get(connection.edge_refs[1].as_str()).copied();
            let (Some(edge_a), Some(edge_b)) = (edge_a, edge_b) else {
                push(&mut findings, "connection_unknown_edge", format!("{path}.edge_refs"),
                    "Connection must reference two existing boundary edges.");
                continue;
            };
            if edge_a.edge_id == edge_b.edge_id {
                push(&mut findings, "connection_self_edge", format!("{path}.edge_refs"),
                    "A connection cannot join an edge to itself.");
            }
            if edge_a.panel_id == edge_b.panel_id {
                push(&mut findings, "connection_same_panel", format!("{path}.edge_refs"),
                    "P0 panel connections must join two distinct panels; darts/internal self-seams are not yet represented.");
            }
            for (edge, expected_role) in [
                (edge_a, if connection.connection_kind == "removable_attachment" { "attachment" } else { "seam" }),
                (edge_b, if connection.connection_kind == "removable_attachment" { "attachment" } else { "seam" }),
            ] {
                if edge.role != expected_role && edge.role != "articulation_boundary" {
                    push(&mut findings, "connection_edge_role_mismatch", format!("{path}.edge_refs"),
                        "Connection edge role is incompatible with the selected seam/attachment construction.");
                }
            }
            for edge in [edge_a, edge_b] {
                if let Some(previous) = used_edges.insert(edge.edge_id.as_str(), connection.connection_id.as_str()) {
                    push(&mut findings, "boundary_edge_reused", format!("{path}.edge_refs"),
                        format!("Boundary edge is already consumed by connection {previous}."));
                }
            }
            let mismatch = (edge_a.length_mm - edge_b.length_mm).abs();
            let allowed = connection.max_length_mismatch_mm
                + edge_a.length_tolerance_mm
                + edge_b.length_tolerance_mm;
            if mismatch > allowed {
                push(&mut findings, "connection_edge_length_mismatch", format!("{path}.edge_refs"),
                    "Connected pattern edges exceed the declared mismatch plus source tolerances; no implicit ease is applied.");
            }
            if connection.connection_kind == "removable_attachment"
                && is_blank_option(&connection.interface_design_ref)
            {
                push(&mut findings, "removable_interface_design_missing", format!("{path}.interface_design_ref"),
                    "A removable attachment requires an explicit interface design reference.");
            }
            if self.state != "proposed" && connection.connection_kind == "removable_attachment"
                && is_blank_option(&connection.retention_evidence_ref)
            {
                push(&mut findings, "removable_interface_retention_missing", format!("{path}.retention_evidence_ref"),
                    "Source-bound removable interfaces must at least bind the planned/available retention evidence record; it is not a pass result.");
            }
        }

        for edge in self.boundary_edges.iter() {
            if matches!(edge.role.as_str(), "seam" | "attachment")
                && !used_edges.contains_key(edge.edge_id.as_str())
            {
                push(&mut findings, "unconnected_required_boundary_edge", format!("boundary_edges.{}", edge.edge_id),
                    "Seam and attachment edges must be consumed by exactly one connection.");
            }
        }

        for (index, feature) in self.articulation_features.iter().enumerate() {
            let path = format!("articulation_features[{index}]");
            if is_blank(&feature.feature_id) || !feature_ids.insert(feature.feature_id.as_str()) {
                push(&mut findings, "duplicate_or_blank_feature_id", format!("{path}.feature_id"),
                    "Articulation/service feature IDs must be non-empty and globally unique.");
            }
            if !matches!(feature.feature_kind.as_str(), "gaiter" | "gusset" | "fold" | "pleat" | "slit" | "segmented_overlap") {
                push(&mut findings, "unknown_articulation_feature_kind", format!("{path}.feature_kind"),
                    "Unknown articulation feature kind.");
            }
            if is_blank(&feature.zone_id) || is_blank(&feature.joint_ref)
                || is_blank(&feature.motion_envelope_ref) || is_blank(&feature.feature_region_ref)
            {
                push(&mut findings, "articulation_binding_missing", &path,
                    "Articulation feature must bind zone, robot joint, owner-supplied motion envelope and exact feature region.");
            }
            if feature.affected_panel_ids.len() < 2 {
                push(&mut findings, "articulation_requires_multiple_panels", format!("{path}.affected_panel_ids"),
                    "P0 articulation features must bind at least two affected pattern panels.");
            }
            for panel_id in &feature.affected_panel_ids {
                if !panels.contains_key(panel_id.as_str()) {
                    push(&mut findings, "articulation_unknown_panel", format!("{path}.affected_panel_ids"),
                        "Articulation feature references an undeclared panel.");
                }
            }
        }

        for (index, feature) in self.service_features.iter().enumerate() {
            let path = format!("service_features[{index}]");
            if is_blank(&feature.feature_id) || !feature_ids.insert(feature.feature_id.as_str()) {
                push(&mut findings, "duplicate_or_blank_feature_id", format!("{path}.feature_id"),
                    "Articulation/service feature IDs must be non-empty and globally unique.");
            }
            let Some(panel) = panels.get(feature.panel_id.as_str()).copied() else {
                push(&mut findings, "service_unknown_panel", format!("{path}.panel_id"),
                    "Service feature references an undeclared panel.");
                continue;
            };
            let Some(keepout) = keepouts.get(feature.keepout_ref.as_str()).copied() else {
                push(&mut findings, "service_unknown_keepout", format!("{path}.keepout_ref"),
                    "Service feature must reference an exact compiler keep-out.");
                continue;
            };
            if !matches!(keepout.kind.as_str(), "service_access" | "emergency_access")
                || !keepout.affected_zone_ids.iter().any(|zone| zone == &panel.zone_id)
            {
                push(&mut findings, "service_keepout_scope_mismatch", format!("{path}.keepout_ref"),
                    "Service opening must map to a service/emergency-access keep-out affecting the same panel zone.");
            }
            if feature.opening_edge_refs.is_empty() || is_blank(&feature.access_path_ref) {
                push(&mut findings, "service_access_path_missing", &path,
                    "Service feature requires at least one opening edge and an explicit access path.");
            }
            if !matches!(feature.access_mode.as_str(), "removable_panel" | "zip_opening" | "roll_back" | "tool_access") {
                push(&mut findings, "unknown_service_access_mode", format!("{path}.access_mode"),
                    "Unknown service-access mode.");
            }
            for edge_ref in &feature.opening_edge_refs {
                match edges.get(edge_ref.as_str()) {
                    Some(edge) if edge.role == "service_opening" && edge.panel_id == feature.panel_id => {}
                    _ => push(&mut findings, "service_opening_edge_mismatch", format!("{path}.opening_edge_refs"),
                        "Service opening edge must exist, belong to the declared panel and use the service_opening role."),
                }
            }
        }

        let required_semantics = [
            "pattern_piece_identity",
            "pattern_geometry_2d",
            "seam_topology",
            "correspondence_2d_3d",
            "coordinate_frame_units",
            "grain_direction",
            "material_orientation",
            "knit_operation_semantics",
            "machine_capability",
            "assembly_operation",
        ];
        for required in required_semantics {
            if !self.semantic_obligations.iter().any(|item| item.semantic_id == required) {
                semantic_coverage_complete = false;
                if self.state != "proposed" {
                    push(&mut findings, "required_semantic_obligation_missing", "semantic_obligations",
                        format!("Required semantic obligation {required} is absent."));
                }
            }
        }
        for (index, obligation) in self.semantic_obligations.iter().enumerate() {
            let path = format!("semantic_obligations[{index}]");
            if is_blank(&obligation.semantic_id) || !semantic_ids.insert(obligation.semantic_id.as_str()) {
                push(&mut findings, "duplicate_or_blank_semantic_id", format!("{path}.semantic_id"),
                    "Semantic obligation IDs must be non-empty and unique.");
            }
            if !required_semantics.contains(&obligation.semantic_id.as_str()) {
                push(&mut findings, "unknown_semantic_obligation", format!("{path}.semantic_id"),
                    "Unknown semantic obligation; extend the owner-reviewed APPAREL-INTEROP profile.");
            }
            if !matches!(obligation.preservation_state.as_str(), "native_preserved" | "sidecar_preserved"
                | "external_owner_ref" | "unsupported" | "lossy" | "unknown")
            {
                push(&mut findings, "invalid_semantic_preservation_state", format!("{path}.preservation_state"),
                    "Unknown semantic preservation state.");
            }
            let has_source = !is_blank_option(&obligation.source_location_ref);
            match obligation.preservation_state.as_str() {
                "native_preserved" => {
                    if !has_source || is_blank_option(&obligation.evidence_ref) {
                        push(&mut findings, "native_semantic_evidence_missing", &path,
                            "Native preservation requires source location and evidence reference.");
                    }
                }
                "sidecar_preserved" => {
                    if !has_source || is_blank_option(&obligation.sidecar_location_ref)
                        || is_blank_option(&obligation.evidence_ref)
                    {
                        push(&mut findings, "sidecar_semantic_evidence_missing", &path,
                            "Sidecar preservation requires source location, sidecar location and evidence reference.");
                    }
                }
                "external_owner_ref" => {
                    if is_blank_option(&obligation.external_owner_ref) {
                        push(&mut findings, "external_semantic_owner_missing", &path,
                            "Externally owned semantics require an exact owner reference.");
                    }
                }
                "unsupported" | "lossy" | "unknown" => {
                    if !obligation.declared_out_of_scope {
                        semantic_coverage_complete = false;
                        if self.state != "proposed" {
                            push(&mut findings, "semantic_preservation_blocked", format!("{path}.preservation_state"),
                                "Unsupported, lossy or unknown required semantics block source-bound topology.");
                        }
                    }
                }
                _ => {}
            }
            if obligation.declared_out_of_scope && is_blank_option(&obligation.scope_disposition_ref) {
                push(&mut findings, "out_of_scope_semantic_without_disposition", format!("{path}.scope_disposition_ref"),
                    "An out-of-scope semantic obligation must bind an explicit engineering scope disposition.");
            }
        }

        if self.state != "proposed" && self.source_artifacts.is_empty() {
            push(&mut findings, "source_bound_without_artifacts", "source_artifacts",
                "Source-bound topology requires at least one exact native source artifact.");
        }
        if self.state != "proposed" && (self.panels.is_empty() || self.boundary_edges.is_empty()) {
            push(&mut findings, "source_bound_topology_empty", "panels/boundary_edges",
                "Source-bound topology requires panels and boundary edges.");
        }
        if self.state == "topology_preflighted" && self.compiler_binding.geometry_state != "cad_revision_bound" {
            push(&mut findings, "preflight_requires_cad_geometry", "compiler_binding.geometry_state",
                "Topology preflight cannot be promoted from synthetic compiler geometry.");
        }

        let topology_preflight_admitted = self.state == "topology_preflighted"
            && self.compiler_binding.geometry_state == "cad_revision_bound"
            && findings.is_empty()
            && semantic_coverage_complete;

        ValidationReport {
            findings,
            panel_count: self.panels.len(),
            boundary_edge_count: self.boundary_edges.len(),
            connection_count: self.connections.len(),
            articulation_feature_count: self.articulation_features.len(),
            service_feature_count: self.service_features.len(),
            semantic_obligation_count: self.semantic_obligations.len(),
            semantic_coverage_complete,
            topology_preflight_admitted,
        }
    }
}

fn push(findings: &mut Vec<Finding>, code: &'static str, path: impl Into<String>, message: impl Into<String>) {
    findings.push(Finding { code, path: path.into(), message: message.into() });
}

fn is_blank(value: &str) -> bool {
    value.trim().is_empty()
}

fn is_blank_option(value: &Option<String>) -> bool {
    value.as_deref().is_none_or(is_blank)
}

fn is_sha256_digest(value: &str) -> bool {
    value.strip_prefix("sha256:")
        .is_some_and(|hex| hex.len() == 64 && hex.bytes().all(|byte| byte.is_ascii_hexdigit()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn base_manifest() -> Value {
        json!({
            "schema_version":"0.1.0",
            "topology_id":"LIGHTSHELL-TOPOLOGY-SYNTHETIC-001",
            "state":"proposed",
            "compiler_binding":{
                "compiler_issue_ref":"https://github.com/Luminous-Dynamics/symthaea/issues/12710",
                "compiler_revision":"b506162d14998177604610f6907bcea42a821df3",
                "compilation_id":"ROBOT-GARMENT-COMPILER-SYNTHETIC-ELBOW-001",
                "robot_id":"synthetic-elbow-robot",
                "robot_article_revision":null,
                "geometry_state":"synthetic_fixture",
                "geometry_digest":null,
                "geometry_units":"mm"
            },
            "interop_binding":{
                "interop_issue_ref":"https://github.com/Luminous-Dynamics/symthaea/issues/11750",
                "interop_contract_revision":"APPAREL-INTEROP-001-P0",
                "sidecar_ref":"SYNTHETIC-PATTERN-SIDECAR-001",
                "sidecar_digest":format!("sha256:{}", "a".repeat(64))
            },
            "source_artifacts":[
                {"artifact_id":"PATTERN-DXF-001","source_format":"dxf","format_version":"legacy-dxf-profile","artifact_ref":"SYNTHETIC-DXF-ARTIFACT","artifact_digest":format!("sha256:{}", "b".repeat(64))},
                {"artifact_id":"KNITOUT-001","source_format":"knitout","format_version":"0.6","artifact_ref":"SYNTHETIC-KNITOUT-ARTIFACT","artifact_digest":format!("sha256:{}", "c".repeat(64))}
            ],
            "compiler_panels":[
                {"panel_id":"upper_arm_shell","zone_id":"upper_arm","coordinate_frame_ref":"robot_root"},
                {"panel_id":"elbow_gusset","zone_id":"elbow","coordinate_frame_ref":"robot_root"},
                {"panel_id":"forearm_shell","zone_id":"forearm","coordinate_frame_ref":"robot_root"}
            ],
            "compiler_keepouts":[
                {"keepout_id":"service_connector_access","kind":"service_access","affected_zone_ids":["forearm"]}
            ],
            "panels":[
                {"panel_id":"upper_arm_shell","zone_id":"upper_arm","compiler_panel_id":"upper_arm_shell","source_artifact_id":"PATTERN-DXF-001","source_piece_id":"UPPER-ARM-01","pattern_revision_ref":"PATTERN-REV-A","pattern_piece_digest":format!("sha256:{}", "d".repeat(64)),"surface_3d_ref":"SURFACE-UPPER-ARM-01","correspondence_ref":"MAP-UPPER-ARM-01","correspondence_state":"declared","coordinate_frame_ref":"robot_root","units":"mm","grain_direction":"warp","material_orientation_ref":"MATERIAL-ORIENTATION-UPPER"},
                {"panel_id":"elbow_gusset","zone_id":"elbow","compiler_panel_id":"elbow_gusset","source_artifact_id":"PATTERN-DXF-001","source_piece_id":"ELBOW-GUSSET-01","pattern_revision_ref":"PATTERN-REV-A","pattern_piece_digest":format!("sha256:{}", "e".repeat(64)),"surface_3d_ref":"SURFACE-ELBOW-01","correspondence_ref":"MAP-ELBOW-01","correspondence_state":"declared","coordinate_frame_ref":"robot_root","units":"mm","grain_direction":"bias","material_orientation_ref":"MATERIAL-ORIENTATION-ELBOW"},
                {"panel_id":"forearm_shell","zone_id":"forearm","compiler_panel_id":"forearm_shell","source_artifact_id":"PATTERN-DXF-001","source_piece_id":"FOREARM-01","pattern_revision_ref":"PATTERN-REV-A","pattern_piece_digest":format!("sha256:{}", "f".repeat(64)),"surface_3d_ref":"SURFACE-FOREARM-01","correspondence_ref":"MAP-FOREARM-01","correspondence_state":"declared","coordinate_frame_ref":"robot_root","units":"mm","grain_direction":"warp","material_orientation_ref":"MATERIAL-ORIENTATION-FOREARM"}
            ],
            "boundary_edges":[
                {"edge_id":"upper-elbow-seam","panel_id":"upper_arm_shell","role":"seam","curve_ref":"CURVE-UPPER-ELBOW","length_mm":80.0,"length_tolerance_mm":0.5},
                {"edge_id":"gusset-upper-seam","panel_id":"elbow_gusset","role":"seam","curve_ref":"CURVE-GUSSET-UPPER","length_mm":80.8,"length_tolerance_mm":0.5},
                {"edge_id":"gusset-forearm-attach","panel_id":"elbow_gusset","role":"attachment","curve_ref":"CURVE-GUSSET-FOREARM","length_mm":75.0,"length_tolerance_mm":0.5},
                {"edge_id":"forearm-gusset-attach","panel_id":"forearm_shell","role":"attachment","curve_ref":"CURVE-FOREARM-GUSSET","length_mm":75.5,"length_tolerance_mm":0.5},
                {"edge_id":"forearm-service-opening","panel_id":"forearm_shell","role":"service_opening","curve_ref":"CURVE-SERVICE-OPENING","length_mm":35.0,"length_tolerance_mm":0.5}
            ],
            "connections":[
                {"connection_id":"SEAM-UPPER-GUSSET","connection_kind":"sewn_seam","edge_refs":["upper-elbow-seam","gusset-upper-seam"],"construction_route_ref":"SEW-ROUTE-01","path_ref":"SEAM-PATH-UPPER-GUSSET","max_length_mismatch_mm":1.0,"construction_revision_ref":"SEAM-REV-A","interface_design_ref":null,"retention_evidence_ref":null},
                {"connection_id":"ATTACH-GUSSET-FOREARM","connection_kind":"removable_attachment","edge_refs":["gusset-forearm-attach","forearm-gusset-attach"],"construction_route_ref":"ATTACH-ROUTE-01","path_ref":"ATTACH-PATH-GUSSET-FOREARM","max_length_mismatch_mm":1.0,"construction_revision_ref":"ATTACH-REV-A","interface_design_ref":"REMOVABLE-INTERFACE-DESIGN-01","retention_evidence_ref":null}
            ],
            "articulation_features":[
                {"feature_id":"ELBOW-ARTICULATION-001","feature_kind":"gusset","zone_id":"elbow","joint_ref":"ROBOT-JOINT-ELBOW-REV-A","motion_envelope_ref":"JOINT-SWEEP-ENVELOPE-REV-A","affected_panel_ids":["upper_arm_shell","elbow_gusset","forearm_shell"],"feature_region_ref":"ELBOW-FEATURE-REGION-01","test_plan_ref":null}
            ],
            "service_features":[
                {"feature_id":"FOREARM-SERVICE-001","panel_id":"forearm_shell","keepout_ref":"service_connector_access","opening_edge_refs":["forearm-service-opening"],"access_mode":"removable_panel","access_path_ref":"SERVICE-ACCESS-PATH-01","service_test_plan_ref":null}
            ],
            "semantic_obligations":[
                {"semantic_id":"pattern_piece_identity","preservation_state":"native_preserved","source_location_ref":"DXF-PIECE-IDS","sidecar_location_ref":null,"evidence_ref":"PIECE-ID-ROUNDTRIP-01","external_owner_ref":null,"declared_out_of_scope":false,"scope_disposition_ref":null},
                {"semantic_id":"pattern_geometry_2d","preservation_state":"native_preserved","source_location_ref":"DXF-PIECE-GEOMETRY","sidecar_location_ref":null,"evidence_ref":"GEOMETRY-ROUNDTRIP-01","external_owner_ref":null,"declared_out_of_scope":false,"scope_disposition_ref":null},
                {"semantic_id":"seam_topology","preservation_state":"sidecar_preserved","source_location_ref":"PATTERN-SOURCE","sidecar_location_ref":"SIDECAR-SEAMS","evidence_ref":"SEAM-TOPOLOGY-ROUNDTRIP-01","external_owner_ref":null,"declared_out_of_scope":false,"scope_disposition_ref":null},
                {"semantic_id":"correspondence_2d_3d","preservation_state":"sidecar_preserved","source_location_ref":"DXF-TO-SURFACE","sidecar_location_ref":"SIDECAR-MAPPINGS","evidence_ref":"CORRESPONDENCE-ROUNDTRIP-01","external_owner_ref":null,"declared_out_of_scope":false,"scope_disposition_ref":null},
                {"semantic_id":"coordinate_frame_units","preservation_state":"native_preserved","source_location_ref":"PATTERN-UNITS","sidecar_location_ref":null,"evidence_ref":"UNITS-FRAME-ROUNDTRIP-01","external_owner_ref":null,"declared_out_of_scope":false,"scope_disposition_ref":null},
                {"semantic_id":"grain_direction","preservation_state":"native_preserved","source_location_ref":"PATTERN-GRAIN","sidecar_location_ref":null,"evidence_ref":"GRAIN-ROUNDTRIP-01","external_owner_ref":null,"declared_out_of_scope":false,"scope_disposition_ref":null},
                {"semantic_id":"material_orientation","preservation_state":"sidecar_preserved","source_location_ref":"MATERIAL-INTENT","sidecar_location_ref":"SIDECAR-ORIENTATION","evidence_ref":"ORIENTATION-ROUNDTRIP-01","external_owner_ref":null,"declared_out_of_scope":false,"scope_disposition_ref":null},
                {"semantic_id":"knit_operation_semantics","preservation_state":"sidecar_preserved","source_location_ref":"KNITOUT-OPS","sidecar_location_ref":"SIDECAR-KNIT-SEMANTICS","evidence_ref":"KNITOUT-SEMANTICS-ROUNDTRIP-01","external_owner_ref":null,"declared_out_of_scope":false,"scope_disposition_ref":null},
                {"semantic_id":"machine_capability","preservation_state":"external_owner_ref","source_location_ref":null,"sidecar_location_ref":null,"evidence_ref":null,"external_owner_ref":"MFG-PROC-5686 / Symtropy textile process capability","declared_out_of_scope":false,"scope_disposition_ref":null},
                {"semantic_id":"assembly_operation","preservation_state":"sidecar_preserved","source_location_ref":"ASSEMBLY-ROUTE","sidecar_location_ref":"SIDECAR-ASSEMBLY","evidence_ref":"ASSEMBLY-ROUNDTRIP-01","external_owner_ref":null,"declared_out_of_scope":false,"scope_disposition_ref":null}
            ]
        })
    }

    #[test]
    fn synthetic_topology_fixture_is_structurally_valid_but_not_preflighted() {
        let report = parse_and_validate(&base_manifest().to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
        assert_eq!(report.panel_count, 3);
        assert_eq!(report.connection_count, 2);
        assert_eq!(report.articulation_feature_count, 1);
        assert_eq!(report.service_feature_count, 1);
        assert!(report.semantic_coverage_complete);
        assert!(!report.can_preflight());
    }

    #[test]
    fn dangling_seam_edge_is_rejected() {
        let mut manifest = base_manifest();
        manifest["connections"][0]["edge_refs"][1] = json!("MISSING-EDGE");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "connection_unknown_edge"));
    }

    #[test]
    fn seam_edges_must_belong_to_distinct_panels() {
        let mut manifest = base_manifest();
        manifest["boundary_edges"][1]["panel_id"] = json!("upper_arm_shell");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "connection_same_panel"));
    }

    #[test]
    fn a_boundary_edge_cannot_be_consumed_twice() {
        let mut manifest = base_manifest();
        let duplicate = manifest["connections"][0].clone();
        manifest["connections"].as_array_mut().unwrap().push(duplicate);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "boundary_edge_reused"));
    }

    #[test]
    fn seam_edge_lengths_must_fit_declared_mismatch_tolerance() {
        let mut manifest = base_manifest();
        manifest["boundary_edges"][1]["length_mm"] = json!(90.0);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "connection_edge_length_mismatch"));
    }

    #[test]
    fn every_seam_and_attachment_edge_must_be_connected_once() {
        let mut manifest = base_manifest();
        manifest["connections"].as_array_mut().unwrap().pop();
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "unconnected_required_boundary_edge"));
    }

    #[test]
    fn panel_must_resolve_to_compiler_panel_in_same_zone_and_frame() {
        let mut manifest = base_manifest();
        manifest["panels"][0]["zone_id"] = json!("wrong_zone");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "panel_compiler_zone_mismatch"));
    }

    #[test]
    fn source_pattern_piece_and_artifact_digests_are_mandatory() {
        let mut manifest = base_manifest();
        manifest["panels"][0]["pattern_piece_digest"] = json!("bad-digest");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "pattern_panel_source_binding_incomplete"));
    }

    #[test]
    fn lossy_correspondence_blocks_source_bound_topology() {
        let mut manifest = base_manifest();
        manifest["state"] = json!("source_bound");
        manifest["panels"][0]["correspondence_state"] = json!("lossy");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "panel_correspondence_not_preserved"));
    }

    #[test]
    fn unknown_semantic_obligation_blocks_source_bound_topology() {
        let mut manifest = base_manifest();
        manifest["state"] = json!("source_bound");
        manifest["semantic_obligations"][2]["preservation_state"] = json!("unknown");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "semantic_preservation_blocked"));
    }

    #[test]
    fn missing_required_semantic_obligation_blocks_source_bound_topology() {
        let mut manifest = base_manifest();
        manifest["state"] = json!("source_bound");
        manifest["semantic_obligations"].as_array_mut().unwrap().retain(|item| item["semantic_id"] != "seam_topology");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "required_semantic_obligation_missing"));
    }

    #[test]
    fn service_feature_must_reference_actual_service_keepout_and_opening_edge() {
        let mut manifest = base_manifest();
        manifest["service_features"][0]["keepout_ref"] = json!("swept_joint");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "service_keepout_scope_mismatch")
            || report.findings.iter().any(|f| f.code == "service_unknown_keepout"));
    }

    #[test]
    fn articulation_feature_requires_robot_joint_and_motion_envelope() {
        let mut manifest = base_manifest();
        manifest["articulation_features"][0]["motion_envelope_ref"] = json!("");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "articulation_binding_missing"));
    }

    #[test]
    fn removable_attachment_requires_interface_design() {
        let mut manifest = base_manifest();
        manifest["connections"][1]["interface_design_ref"] = json!(null);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "removable_interface_design_missing"));
    }

    #[test]
    fn synthetic_geometry_cannot_be_promoted_to_topology_preflight() {
        let mut manifest = base_manifest();
        manifest["state"] = json!("topology_preflighted");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "preflight_requires_cad_geometry"));
        assert!(!report.can_preflight());
    }

    #[test]
    fn knitout_machine_capability_remains_external_owner_evidence() {
        let mut manifest = base_manifest();
        manifest["semantic_obligations"][8]["preservation_state"] = json!("unsupported");
        manifest["semantic_obligations"][8]["external_owner_ref"] = json!(null);
        manifest["state"] = json!("source_bound");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "semantic_preservation_blocked")
            || report.findings.iter().any(|f| f.code == "external_semantic_owner_missing"));
    }

    #[test]
    fn unsupported_knitout_version_is_not_assumed_compatible() {
        let mut manifest = base_manifest();
        manifest["source_artifacts"][1]["format_version"] = json!("0.7");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "unsupported_knitout_version"));
    }

    #[test]
    fn one_pattern_piece_cannot_silently_map_to_two_robot_panels() {
        let mut manifest = base_manifest();
        manifest["panels"][1]["source_piece_id"] = json!("UPPER-ARM-01");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "duplicate_source_piece_identity"));
    }

    #[test]
    fn duplicate_boundary_edge_ids_are_rejected() {
        let mut manifest = base_manifest();
        manifest["boundary_edges"][1]["edge_id"] = json!("upper-elbow-seam");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "duplicate_boundary_edge_id"));
    }

    #[test]
    fn missing_sidecar_digest_is_rejected() {
        let mut manifest = base_manifest();
        manifest["interop_binding"]["sidecar_digest"] = json!("unknown");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "interop_or_compiler_binding_incomplete"));
    }

    #[test]
    fn source_bound_topology_requires_pattern_artifacts_and_edge_graph() {
        let mut manifest = base_manifest();
        manifest["state"] = json!("source_bound");
        manifest["source_artifacts"] = json!([]);
        manifest["boundary_edges"] = json!([]);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "source_bound_without_artifacts"));
        assert!(report.findings.iter().any(|f| f.code == "source_bound_topology_empty"));
    }

    #[test]
    fn topology_preflight_requires_verified_2d_3d_correspondence() {
        let mut manifest = base_manifest();
        manifest["state"] = json!("topology_preflighted");
        manifest["compiler_binding"]["geometry_state"] = json!("cad_revision_bound");
        manifest["compiler_binding"]["robot_article_revision"] = json!("ROBOT-REV-A");
        manifest["compiler_binding"]["geometry_digest"] = json!(format!("sha256:{}", "1".repeat(64)));
        manifest["panels"][0]["correspondence_state"] = json!("declared");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "topology_preflight_requires_verified_correspondence"));
        assert!(!report.can_preflight());
    }

    #[test]
    fn explicitly_out_of_scope_semantics_need_disposition_but_do_not_fake_lossless_support() {
        let mut manifest = base_manifest();
        manifest["state"] = json!("source_bound");
        manifest["semantic_obligations"][7]["preservation_state"] = json!("unsupported");
        manifest["semantic_obligations"][7]["declared_out_of_scope"] = json!(true);
        manifest["semantic_obligations"][7]["scope_disposition_ref"] = json!("NOT-A-KNIT-PRODUCT-SCOPE-01");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
        assert!(report.semantic_coverage_complete);
    }

    #[test]
    fn out_of_scope_semantic_requires_explicit_disposition() {
        let mut manifest = base_manifest();
        manifest["semantic_obligations"][7]["declared_out_of_scope"] = json!(true);
        manifest["semantic_obligations"][7]["scope_disposition_ref"] = json!(null);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "out_of_scope_semantic_without_disposition"));
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let mut manifest = base_manifest();
        manifest["invented_seam_strength_n"] = json!(900);
        assert!(parse_and_validate(&manifest.to_string()).is_err());
    }
}
