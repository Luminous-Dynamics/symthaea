// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Robot-panel to apparel-pattern/seam-topology binding for Atelier Lightshell.
//!
//! This is a domain adapter over Robot Garment Compiler and APPAREL-INTEROP
//! owner-native records. It does not define a pattern format, parse DXF/Knitout,
//! generate sewing patterns, qualify a machine, or authorize manufacturing.

use serde::Deserialize;
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PanelTopologyManifest {
    pub schema_version: String,
    pub topology_id: String,
    pub state: String,
    pub compiler: CompilerBinding,
    pub assembly_groups: Vec<AssemblyGroup>,
    pub panels: Vec<PanelPatternBinding>,
    pub protected_interfaces: Vec<ProtectedInterface>,
    pub seam_edges: Vec<SeamEdgeBinding>,
    pub interoperability: Option<InteropBinding>,
    pub machine_preflight: Option<MachinePreflightBinding>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompilerBinding {
    pub compiler_manifest_ref: String,
    pub compilation_id: String,
    pub robot_id: String,
    pub article_revision: Option<String>,
    pub geometry_digest: Option<String>,
    pub zone_requirement_ref: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssemblyGroup {
    pub assembly_group_id: String,
    pub require_connected: bool,
    pub assembly_intent_ref: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PanelPatternBinding {
    pub panel_id: String,
    pub compiler_panel_ref: String,
    pub zone_id: String,
    pub assembly_group_id: String,
    pub pattern_piece_ref: Option<String>,
    pub pattern_revision_ref: Option<String>,
    pub source_pattern_ref: Option<String>,
    pub source_pattern_digest: Option<String>,
    pub correspondence_ref: Option<String>,
    pub coordinate_frame_ref: Option<String>,
    pub units: String,
    pub handedness: String,
    pub material_orientation_ref: Option<String>,
    pub orientation_kind: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProtectedInterface {
    pub interface_ref: String,
    pub zone_id: String,
    pub interface_kind: String,
    pub source_keepout_ref: String,
    pub seam_allowed: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeamEdgeBinding {
    pub seam_edge_ref: String,
    pub from_panel_id: String,
    pub from_pattern_edge_ref: String,
    pub to_panel_id: String,
    pub to_pattern_edge_ref: String,
    pub seam_profile_ref: String,
    pub seam_construction_ref: String,
    pub transition_ref: Option<String>,
    pub crossed_protected_interface_refs: Vec<String>,
    pub protected_interface_review_ref: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InteropBinding {
    pub projection_ref: String,
    pub format_profile_ref: String,
    pub toolchain_revision_ref: String,
    pub source_digest: String,
    pub semantic_loss_state: String,
    pub semantic_loss_report_ref: Option<String>,
    pub unsupported_field_refs: Vec<String>,
    pub round_trip_receipt_ref: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MachinePreflightBinding {
    pub machine_profile_ref: String,
    pub backend_ref: String,
    pub program_revision_ref: String,
    pub program_digest: String,
    pub preflight_state: String,
    pub preflight_evidence_ref: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    pub code: &'static str,
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PanelDisposition {
    pub panel_id: String,
    pub pattern_bound: bool,
    pub in_declared_assembly_group: bool,
}

#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub findings: Vec<Finding>,
    pub panel_dispositions: Vec<PanelDisposition>,
    pub panel_count: usize,
    pub seam_edge_count: usize,
    pub assembly_group_count: usize,
    pub pattern_bindings_complete: bool,
    pub interoperability_reviewed: bool,
    pub machine_preflight_admitted: bool,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.findings.is_empty()
    }

    /// This means owner-native pattern/seam references are complete and
    /// internally consistent. It is not a generated or physically fitted pattern.
    pub fn can_review_interop(&self) -> bool {
        self.is_valid() && self.pattern_bindings_complete
    }

    /// Interop review is a semantic-preservation disposition, not machine execution.
    pub fn can_admit_machine_preflight(&self) -> bool {
        self.is_valid() && self.interoperability_reviewed
    }

    /// Preflight admission is not authorization to execute a machine program.
    pub fn is_machine_preflight_admitted(&self) -> bool {
        self.is_valid() && self.machine_preflight_admitted
    }
}

pub fn parse_and_validate(input: &str) -> Result<ValidationReport, serde_json::Error> {
    let manifest: PanelTopologyManifest = serde_json::from_str(input)?;
    Ok(manifest.validate())
}

impl PanelTopologyManifest {
    pub fn validate(&self) -> ValidationReport {
        let mut findings = Vec::new();
        let mut groups: HashMap<&str, &AssemblyGroup> = HashMap::new();
        let mut panels: HashMap<&str, &PanelPatternBinding> = HashMap::new();
        let mut interfaces: HashMap<&str, &ProtectedInterface> = HashMap::new();
        let mut seam_ids = HashSet::new();
        let mut pattern_piece_refs = HashSet::new();
        let mut used_pattern_edges = HashSet::new();
        let mut pattern_bindings_complete = true;

        if self.schema_version != "0.1.0" {
            push(&mut findings, "unsupported_schema_version", "schema_version",
                "Only Lightshell panel-topology schema version 0.1.0 is admitted.");
        }
        if is_blank(&self.topology_id) {
            push(&mut findings, "missing_topology_id", "topology_id",
                "Panel-topology identity must be non-empty.");
        }
        if !matches!(self.state.as_str(), "proposed" | "pattern_bound" | "interop_reviewed" | "machine_preflight_admitted") {
            push(&mut findings, "invalid_topology_state", "state",
                "State must be proposed, pattern_bound, interop_reviewed or machine_preflight_admitted.");
        }

        for (field, value) in [
            ("compiler.compiler_manifest_ref", Some(self.compiler.compiler_manifest_ref.as_str())),
            ("compiler.compilation_id", Some(self.compiler.compilation_id.as_str())),
            ("compiler.robot_id", Some(self.compiler.robot_id.as_str())),
        ] {
            if value.is_none_or(is_blank) {
                push(&mut findings, "missing_compiler_binding", field,
                    "The topology must bind an exact upstream compiler manifest, compilation and robot.");
            }
        }
        if self.compiler.geometry_digest.as_deref().is_some_and(|value| !is_sha256_digest(value)) {
            push(&mut findings, "invalid_compiler_geometry_digest", "compiler.geometry_digest",
                "Compiler geometry digest must use sha256: followed by 64 hexadecimal characters.");
        }

        if self.assembly_groups.is_empty() {
            push(&mut findings, "no_assembly_groups", "assembly_groups",
                "At least one explicit garment assembly group is required.");
        }
        for (index, group) in self.assembly_groups.iter().enumerate() {
            let path = format!("assembly_groups[{index}]");
            if is_blank(&group.assembly_group_id) {
                push(&mut findings, "blank_assembly_group_id", format!("{path}.assembly_group_id"),
                    "Assembly-group identity must be non-empty.");
            }
            if groups.insert(group.assembly_group_id.as_str(), group).is_some() {
                push(&mut findings, "duplicate_assembly_group_id", format!("{path}.assembly_group_id"),
                    "Assembly-group IDs must be unique.");
            }
            if group.require_connected && is_blank_option(&group.assembly_intent_ref) {
                push(&mut findings, "connected_group_missing_intent", format!("{path}.assembly_intent_ref"),
                    "A group requiring connected assembly must bind its owner-native assembly intent.");
            }
        }

        for (index, panel) in self.panels.iter().enumerate() {
            let path = format!("panels[{index}]");
            if is_blank(&panel.panel_id) || is_blank(&panel.compiler_panel_ref) || is_blank(&panel.zone_id) {
                push(&mut findings, "missing_panel_identity", &path,
                    "Panel, compiler-panel and zone identities must be non-empty.");
            }
            if panels.insert(panel.panel_id.as_str(), panel).is_some() {
                push(&mut findings, "duplicate_panel_id", format!("{path}.panel_id"),
                    "Panel IDs must be unique.");
            }
            if !groups.contains_key(panel.assembly_group_id.as_str()) {
                push(&mut findings, "panel_unknown_assembly_group", format!("{path}.assembly_group_id"),
                    "Panel references an assembly group absent from this topology.");
            }
            if panel.units != "mm" {
                push(&mut findings, "panel_units_not_supported", format!("{path}.units"),
                    "P0 accepts millimetres only; implicit conversion is prohibited.");
            }
            if !matches!(panel.handedness.as_str(), "right_handed" | "left_handed") {
                push(&mut findings, "invalid_panel_handedness", format!("{path}.handedness"),
                    "Panel coordinate handedness must be explicit.");
            }
            if !matches!(panel.orientation_kind.as_str(), "grain_direction" | "stretch_axis" | "principal_material_axis") {
                push(&mut findings, "invalid_material_orientation_kind", format!("{path}.orientation_kind"),
                    "Material orientation must identify grain, stretch or a declared principal material axis.");
            }

            if !is_complete_panel_pattern_binding(panel) {
                pattern_bindings_complete = false;
                if self.state != "proposed" {
                    push(&mut findings, "panel_pattern_binding_incomplete", &path,
                        "Every panel must bind exact owner-native pattern piece/revision, source digest, 2D/3D correspondence, frame and material orientation.");
                }
            } else {
                let piece_ref = panel.pattern_piece_ref.as_deref().unwrap();
                if !pattern_piece_refs.insert(piece_ref) {
                    push(&mut findings, "duplicate_pattern_piece_binding", format!("{path}.pattern_piece_ref"),
                        "A pattern piece may bind only one robot panel in this P0 one-to-one adapter.");
                }
                if panel.source_pattern_digest.as_deref().is_some_and(|value| !is_sha256_digest(value)) {
                    push(&mut findings, "invalid_source_pattern_digest", format!("{path}.source_pattern_digest"),
                        "Source pattern digest must use sha256: followed by 64 hexadecimal characters.");
                }
            }
        }
        if self.panels.is_empty() {
            pattern_bindings_complete = false;
            push(&mut findings, "no_panels", "panels",
                "At least one compiler candidate panel must be declared.");
        }

        for (index, interface) in self.protected_interfaces.iter().enumerate() {
            let path = format!("protected_interfaces[{index}]");
            if is_blank(&interface.interface_ref) || is_blank(&interface.source_keepout_ref) {
                push(&mut findings, "protected_interface_missing_identity", &path,
                    "Protected-interface and source keep-out references must be non-empty.");
            }
            if interfaces.insert(interface.interface_ref.as_str(), interface).is_some() {
                push(&mut findings, "duplicate_protected_interface", format!("{path}.interface_ref"),
                    "Protected-interface IDs must be unique.");
            }
            if !is_known_interface_kind(&interface.interface_kind) {
                push(&mut findings, "unknown_protected_interface_kind", format!("{path}.interface_kind"),
                    "Unknown protected interface kind.");
            }
        }

        for (index, seam) in self.seam_edges.iter().enumerate() {
            let path = format!("seam_edges[{index}]");
            if is_blank(&seam.seam_edge_ref) || is_blank(&seam.seam_profile_ref) || is_blank(&seam.seam_construction_ref) {
                push(&mut findings, "seam_missing_owner_identity", &path,
                    "Seam edge, seam profile and seam construction must bind owner-native references.");
            }
            if !seam_ids.insert(seam.seam_edge_ref.as_str()) {
                push(&mut findings, "duplicate_seam_edge_ref", format!("{path}.seam_edge_ref"),
                    "Seam-edge references must be unique.");
            }
            let from = panels.get(seam.from_panel_id.as_str()).copied();
            let to = panels.get(seam.to_panel_id.as_str()).copied();
            if from.is_none() {
                push(&mut findings, "seam_unknown_from_panel", format!("{path}.from_panel_id"),
                    "Seam source panel is absent from the panel bindings.");
            }
            if to.is_none() {
                push(&mut findings, "seam_unknown_to_panel", format!("{path}.to_panel_id"),
                    "Seam destination panel is absent from the panel bindings.");
            }
            if seam.from_panel_id == seam.to_panel_id {
                push(&mut findings, "seam_self_link", &path,
                    "P0 seam edges must connect two distinct panel identities.");
            }
            if let (Some(from), Some(to)) = (from, to) {
                if from.assembly_group_id != to.assembly_group_id {
                    push(&mut findings, "seam_crosses_assembly_groups", &path,
                        "A seam edge cannot connect panels in different assembly groups.");
                }
                if from.zone_id != to.zone_id && is_blank_option(&seam.transition_ref) {
                    push(&mut findings, "cross_zone_seam_missing_transition", format!("{path}.transition_ref"),
                        "A seam between different functional zones must bind an exact #12754 transition candidate.");
                }
                if from.zone_id == to.zone_id && seam.transition_ref.is_some() {
                    push(&mut findings, "same_zone_seam_has_unexpected_transition", format!("{path}.transition_ref"),
                        "A same-zone seam must not claim a cross-zone transition identity.");
                }
            }

            for (field, value) in [
                ("from_pattern_edge_ref", Some(seam.from_pattern_edge_ref.as_str())),
                ("to_pattern_edge_ref", Some(seam.to_pattern_edge_ref.as_str())),
            ] {
                if value.is_none_or(is_blank) {
                    push(&mut findings, "seam_missing_pattern_edge", format!("{path}.{field}"),
                        "Each seam endpoint must bind an exact owner-native pattern edge.");
                }
            }
            if from.is_some() && !is_blank(&seam.from_pattern_edge_ref) {
                if !used_pattern_edges.insert((seam.from_panel_id.as_str(), seam.from_pattern_edge_ref.as_str())) {
                    push(&mut findings, "pattern_edge_reused", format!("{path}.from_pattern_edge_ref"),
                        "A pattern edge may participate in only one seam in this P0 topology.");
                }
            }
            if to.is_some() && !is_blank(&seam.to_pattern_edge_ref) {
                if !used_pattern_edges.insert((seam.to_panel_id.as_str(), seam.to_pattern_edge_ref.as_str())) {
                    push(&mut findings, "pattern_edge_reused", format!("{path}.to_pattern_edge_ref"),
                        "A pattern edge may participate in only one seam in this P0 topology.");
                }
            }

            if !seam.crossed_protected_interface_refs.is_empty() && is_blank_option(&seam.protected_interface_review_ref) {
                push(&mut findings, "seam_protected_interface_review_missing", format!("{path}.protected_interface_review_ref"),
                    "A seam reporting protected-interface intersections must bind its geometry review evidence.");
            }
            for (interface_index, interface_ref) in seam.crossed_protected_interface_refs.iter().enumerate() {
                match interfaces.get(interface_ref.as_str()).copied() {
                    Some(interface) if !interface.seam_allowed => {
                        push(&mut findings, "seam_crosses_prohibited_interface",
                            format!("{path}.crossed_protected_interface_refs[{interface_index}]"),
                            "Seam crosses a protected interface declared as no-seam/no-cover.");
                    }
                    Some(_) => {}
                    None => push(&mut findings, "seam_unknown_protected_interface",
                        format!("{path}.crossed_protected_interface_refs[{interface_index}]"),
                        "Seam references a protected interface absent from this topology."),
                }
            }
        }

        validate_assembly_connectivity(&self.assembly_groups, &self.panels, &self.seam_edges, &mut findings);

        let interoperability_reviewed = self.interoperability.as_ref().is_some_and(|interop| {
            validate_interop(interop, self.state.as_str(), &mut findings)
        });
        if matches!(self.state.as_str(), "interop_reviewed" | "machine_preflight_admitted") {
            if self.interoperability.is_none() {
                push(&mut findings, "interop_binding_missing", "interoperability",
                    "Interop-reviewed states require exact APPAREL-INTEROP projection and toolchain references.");
            }
            if !pattern_bindings_complete {
                push(&mut findings, "interop_without_complete_pattern_bindings", "panels",
                    "Interop review requires complete panel-to-pattern and 2D/3D correspondence bindings.");
            }
        }

        let machine_preflight_admitted = self.machine_preflight.as_ref().is_some_and(|preflight| {
            validate_machine_preflight(preflight, self.state.as_str(), &mut findings)
        });
        if self.state == "machine_preflight_admitted" {
            if !interoperability_reviewed {
                push(&mut findings, "machine_preflight_without_interop_review", "interoperability",
                    "Machine preflight admission requires a reviewed interop projection with no unknown semantic loss.");
            }
            if self.machine_preflight.is_none() {
                push(&mut findings, "machine_preflight_binding_missing", "machine_preflight",
                    "Machine-preflight-admitted state requires exact machine/backend/program evidence.");
            }
        }

        let panel_dispositions = self.panels.iter().map(|panel| {
            PanelDisposition {
                panel_id: panel.panel_id.clone(),
                pattern_bound: is_complete_panel_pattern_binding(panel),
                in_declared_assembly_group: groups.contains_key(panel.assembly_group_id.as_str()),
            }
        }).collect();

        ValidationReport {
            findings,
            panel_dispositions,
            panel_count: self.panels.len(),
            seam_edge_count: self.seam_edges.len(),
            assembly_group_count: self.assembly_groups.len(),
            pattern_bindings_complete,
            interoperability_reviewed,
            machine_preflight_admitted,
        }
    }
}

fn is_complete_panel_pattern_binding(panel: &PanelPatternBinding) -> bool {
    [
        panel.pattern_piece_ref.as_deref(),
        panel.pattern_revision_ref.as_deref(),
        panel.source_pattern_ref.as_deref(),
        panel.source_pattern_digest.as_deref(),
        panel.correspondence_ref.as_deref(),
        panel.coordinate_frame_ref.as_deref(),
        panel.material_orientation_ref.as_deref(),
    ].into_iter().all(|value| value.is_some_and(|item| !is_blank(item)))
        && matches!(panel.handedness.as_str(), "right_handed" | "left_handed")
        && panel.units == "mm"
        && matches!(panel.orientation_kind.as_str(), "grain_direction" | "stretch_axis" | "principal_material_axis")
}

fn is_known_interface_kind(value: &str) -> bool {
    matches!(value,
        "joint_sweep" | "cable_route" | "sensor_aperture" | "thermal_vent" |
        "service_access" | "charging_interface" | "emergency_access" |
        "pinch_shear" | "protected_surface")
}

fn validate_assembly_connectivity(
    groups: &[AssemblyGroup],
    panels: &[PanelPatternBinding],
    seams: &[SeamEdgeBinding],
    findings: &mut Vec<Finding>,
) {
    for (group_index, group) in groups.iter().enumerate() {
        if !group.require_connected {
            continue;
        }
        let members: Vec<&str> = panels.iter()
            .filter(|panel| panel.assembly_group_id == group.assembly_group_id)
            .map(|panel| panel.panel_id.as_str())
            .collect();
        if members.len() <= 1 {
            continue;
        }
        let member_set: HashSet<&str> = members.iter().copied().collect();
        let mut adjacency: HashMap<&str, Vec<&str>> = HashMap::new();
        for panel in &members {
            adjacency.entry(panel).or_default();
        }
        for seam in seams {
            if member_set.contains(seam.from_panel_id.as_str())
                && member_set.contains(seam.to_panel_id.as_str())
            {
                adjacency.entry(seam.from_panel_id.as_str()).or_default().push(seam.to_panel_id.as_str());
                adjacency.entry(seam.to_panel_id.as_str()).or_default().push(seam.from_panel_id.as_str());
            }
        }

        let Some(start) = members.first().copied() else { continue; };
        let mut visited = HashSet::new();
        let mut queue = VecDeque::from([start]);
        while let Some(panel_id) = queue.pop_front() {
            if !visited.insert(panel_id) {
                continue;
            }
            if let Some(neighbors) = adjacency.get(panel_id) {
                for neighbor in neighbors {
                    if !visited.contains(neighbor) {
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        if visited.len() != member_set.len() {
            push(findings, "assembly_group_disconnected", format!("assembly_groups[{group_index}].assembly_group_id"),
                "Assembly group requires connectivity but its seam graph has disconnected panel components.");
        }
    }
}

fn validate_interop(interop: &InteropBinding, state: &str, findings: &mut Vec<Finding>) -> bool {
    for (field, value) in [
        ("projection_ref", Some(interop.projection_ref.as_str())),
        ("format_profile_ref", Some(interop.format_profile_ref.as_str())),
        ("toolchain_revision_ref", Some(interop.toolchain_revision_ref.as_str())),
        ("source_digest", Some(interop.source_digest.as_str())),
    ] {
        if value.is_none_or(is_blank) {
            push(findings, "interop_missing_owner_reference", format!("interoperability.{field}"),
                "Interop binding requires exact APPAREL-INTEROP projection, format profile, toolchain revision and source digest.");
        }
    }
    if !is_sha256_digest(&interop.source_digest) {
        push(findings, "invalid_interop_source_digest", "interoperability.source_digest",
            "Interop source digest must use sha256: followed by 64 hexadecimal characters.");
    }
    match interop.semantic_loss_state.as_str() {
        "none" => {
            if !interop.unsupported_field_refs.is_empty() || interop.semantic_loss_report_ref.is_some() {
                push(findings, "loss_free_interop_has_loss_claims", "interoperability",
                    "A loss-free interop projection cannot list unsupported fields or a semantic-loss report.");
            }
        }
        "explicitly_declared" => {
            if is_blank_option(&interop.semantic_loss_report_ref) || interop.unsupported_field_refs.is_empty() {
                push(findings, "declared_semantic_loss_incomplete", "interoperability",
                    "Explicit semantic loss requires both a report reference and the list of unsupported owner-native fields.");
            }
        }
        "unknown" => push(findings, "unknown_semantic_loss_blocks_interop", "interoperability.semantic_loss_state",
            "Unknown semantic loss blocks interop review and machine preflight."),
        "rejected" => push(findings, "interop_semantic_loss_rejected", "interoperability.semantic_loss_state",
            "Rejected semantic loss cannot be promoted to interop review."),
        _ => push(findings, "invalid_semantic_loss_state", "interoperability.semantic_loss_state",
            "Semantic-loss state must be none, explicitly_declared, unknown or rejected."),
    }
    let has_required_refs = !is_blank(&interop.projection_ref)
        && !is_blank(&interop.format_profile_ref)
        && !is_blank(&interop.toolchain_revision_ref)
        && is_sha256_digest(&interop.source_digest);
    let loss_disposition_admitted = interop.semantic_loss_state == "none"
        || interop.semantic_loss_state == "explicitly_declared";
    let reviewed = has_required_refs && loss_disposition_admitted;
    if state == "machine_preflight_admitted" && !reviewed {
        push(findings, "machine_preflight_interop_not_admitted", "interoperability",
            "Machine preflight cannot proceed until semantic loss has an admitted disposition.");
    }
    reviewed
}

fn validate_machine_preflight(
    preflight: &MachinePreflightBinding,
    state: &str,
    findings: &mut Vec<Finding>,
) -> bool {
    for (field, value) in [
        ("machine_profile_ref", Some(preflight.machine_profile_ref.as_str())),
        ("backend_ref", Some(preflight.backend_ref.as_str())),
        ("program_revision_ref", Some(preflight.program_revision_ref.as_str())),
        ("program_digest", Some(preflight.program_digest.as_str())),
    ] {
        if value.is_none_or(is_blank) {
            push(findings, "machine_preflight_missing_identity", format!("machine_preflight.{field}"),
                "Machine preflight requires exact machine profile, backend, program revision and digest.");
        }
    }
    if !is_sha256_digest(&preflight.program_digest) {
        push(findings, "invalid_machine_program_digest", "machine_preflight.program_digest",
            "Machine program digest must use sha256: followed by 64 hexadecimal characters.");
    }
    match preflight.preflight_state.as_str() {
        "pending" => {
            if state == "machine_preflight_admitted" {
                push(findings, "machine_preflight_still_pending", "machine_preflight.preflight_state",
                    "Machine-preflight-admitted state requires a passed owner-native preflight result.");
            }
            false
        }
        "passed" => {
            if is_blank_option(&preflight.preflight_evidence_ref) {
                push(findings, "machine_preflight_pass_missing_evidence", "machine_preflight.preflight_evidence_ref",
                    "A passed machine preflight requires its owner-native evidence reference.");
                false
            } else {
                true
            }
        }
        "failed" => {
            push(findings, "machine_preflight_failed", "machine_preflight.preflight_state",
                "A failed machine preflight blocks preflight admission.");
            false
        }
        _ => {
            push(findings, "invalid_machine_preflight_state", "machine_preflight.preflight_state",
                "Machine preflight state must be pending, passed or failed.");
            false
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
    let Some(hex) = value.strip_prefix("sha256:") else { return false; };
    hex.len() == 64 && hex.bytes().all(|byte| byte.is_ascii_hexdigit())
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
            "compiler":{
                "compiler_manifest_ref":"COMPILER-MANIFEST-12719",
                "compilation_id":"ROBOT-GARMENT-COMPILER-SYNTHETIC-ELBOW-001",
                "robot_id":"SYNTHETIC-ELBOW-001",
                "article_revision":null,
                "geometry_digest":null,
                "zone_requirement_ref":null
            },
            "assembly_groups":[{
                "assembly_group_id":"forearm_elbow_assembly",
                "require_connected":true,
                "assembly_intent_ref":"ASSEMBLY-INTENT-SYNTHETIC-001"
            }],
            "panels":[
                {
                    "panel_id":"forearm_textile_panel_A",
                    "compiler_panel_ref":"forearm_textile_panel_A",
                    "zone_id":"forearm_shell",
                    "assembly_group_id":"forearm_elbow_assembly",
                    "pattern_piece_ref":null,
                    "pattern_revision_ref":null,
                    "source_pattern_ref":null,
                    "source_pattern_digest":null,
                    "correspondence_ref":null,
                    "coordinate_frame_ref":null,
                    "units":"mm",
                    "handedness":"right_handed",
                    "material_orientation_ref":null,
                    "orientation_kind":"grain_direction"
                },
                {
                    "panel_id":"elbow_gaiter_panel_B",
                    "compiler_panel_ref":"elbow_gaiter_panel_B",
                    "zone_id":"elbow_articulation",
                    "assembly_group_id":"forearm_elbow_assembly",
                    "pattern_piece_ref":null,
                    "pattern_revision_ref":null,
                    "source_pattern_ref":null,
                    "source_pattern_digest":null,
                    "correspondence_ref":null,
                    "coordinate_frame_ref":null,
                    "units":"mm",
                    "handedness":"right_handed",
                    "material_orientation_ref":null,
                    "orientation_kind":"stretch_axis"
                }
            ],
            "protected_interfaces":[{
                "interface_ref":"ELBOW_SWEEP_INTERFACE",
                "zone_id":"elbow_articulation",
                "interface_kind":"joint_sweep",
                "source_keepout_ref":"KEEP_OUT-ELBOW-SWEEP",
                "seam_allowed":false
            }],
            "seam_edges":[{
                "seam_edge_ref":"SEAM-FOREARM-ELBOW-01",
                "from_panel_id":"forearm_textile_panel_A",
                "from_pattern_edge_ref":"PATTERN-EDGE-A1",
                "to_panel_id":"elbow_gaiter_panel_B",
                "to_pattern_edge_ref":"PATTERN-EDGE-B1",
                "seam_profile_ref":"APPAREL-INTEROP-SEAM-PROFILE-01",
                "seam_construction_ref":"LIGHTSHELL-SEAM-CONSTRUCTION-01",
                "transition_ref":"LIGHTSHELL-TRANSITION-IMPACT-TO-GAITer-01",
                "crossed_protected_interface_refs":[],
                "protected_interface_review_ref":"SEAM-KEEP-OUT-REVIEW-01"
            }],
            "interoperability":null,
            "machine_preflight":null
        })
    }

    fn bind_patterns(manifest: &mut Value) {
        for (index, panel) in manifest["panels"].as_array_mut().unwrap().iter_mut().enumerate() {
            let suffix = if index == 0 { "A" } else { "B" };
            panel["pattern_piece_ref"] = json!(format!("APPAREL-PATTERN-PIECE-{suffix}"));
            panel["pattern_revision_ref"] = json!("PATTERN-REV-A");
            panel["source_pattern_ref"] = json!("APPAREL-SOURCE-PATTERN-01");
            panel["source_pattern_digest"] = json!(format!("sha256:{}", "a".repeat(64)));
            panel["correspondence_ref"] = json!(format!("PATTERN-3D-CORRESPONDENCE-{suffix}"));
            panel["coordinate_frame_ref"] = json!("robot_root");
            panel["material_orientation_ref"] = json!(format!("MATERIAL-ORIENTATION-{suffix}"));
        }
    }

    fn add_interop(manifest: &mut Value, loss_state: &str) {
        manifest["interoperability"] = json!({
            "projection_ref":"APPAREL-INTEROP-PROJECTION-11750",
            "format_profile_ref":"APPAREL-INTEROP-DXF-KNITOUT-PROFILE-01",
            "toolchain_revision_ref":"APPAREL-INTEROP-TOOLCHAIN-REV-A",
            "source_digest":format!("sha256:{}", "b".repeat(64)),
            "semantic_loss_state":loss_state,
            "semantic_loss_report_ref":if loss_state == "explicitly_declared" { Some("SEMANTIC-LOSS-REPORT-01") } else { None },
            "unsupported_field_refs":if loss_state == "explicitly_declared" { vec!["unsupported:decorative-trim"] } else { vec![] },
            "round_trip_receipt_ref":if loss_state == "none" { Some("ROUND-TRIP-RECEIPT-01") } else { None }
        });
    }

    fn add_preflight(manifest: &mut Value, state: &str) {
        manifest["machine_preflight"] = json!({
            "machine_profile_ref":"MACHINE-PROFILE-01",
            "backend_ref":"KNITOUT-BACKEND-REV-A",
            "program_revision_ref":"PROGRAM-REV-A",
            "program_digest":format!("sha256:{}", "c".repeat(64)),
            "preflight_state":state,
            "preflight_evidence_ref":if state == "passed" { Some("MACHINE-PREFLIGHT-EVIDENCE-01") } else { None }
        });
    }

    #[test]
    fn proposed_panel_topology_is_admitted_without_pattern_or_machine_claims() {
        let report = parse_and_validate(&base_manifest().to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
        assert_eq!(report.panel_count, 2);
        assert_eq!(report.seam_edge_count, 1);
        assert!(!report.pattern_bindings_complete);
        assert!(!report.interoperability_reviewed);
        assert!(!report.machine_preflight_admitted);
    }

    #[test]
    fn pattern_bound_requires_exact_piece_revision_digest_and_correspondence() {
        let mut manifest = base_manifest();
        manifest["state"] = json!("pattern_bound");
        bind_patterns(&mut manifest);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
        assert!(report.pattern_bindings_complete);
    }

    #[test]
    fn duplicate_pattern_piece_binding_is_rejected() {
        let mut manifest = base_manifest();
        manifest["state"] = json!("pattern_bound");
        bind_patterns(&mut manifest);
        manifest["panels"][1]["pattern_piece_ref"] = manifest["panels"][0]["pattern_piece_ref"].clone();
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|item| item.code == "duplicate_pattern_piece_binding"));
    }

    #[test]
    fn cross_zone_seam_requires_transition_identity() {
        let mut manifest = base_manifest();
        manifest["seam_edges"][0]["transition_ref"] = json!(null);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|item| item.code == "cross_zone_seam_missing_transition"));
    }

    #[test]
    fn same_zone_seam_cannot_claim_cross_zone_transition() {
        let mut manifest = base_manifest();
        manifest["panels"][1]["zone_id"] = json!("forearm_shell");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|item| item.code == "same_zone_seam_has_unexpected_transition"));
    }

    #[test]
    fn seam_through_no_seam_protected_interface_is_rejected() {
        let mut manifest = base_manifest();
        manifest["seam_edges"][0]["crossed_protected_interface_refs"] = json!(["ELBOW_SWEEP_INTERFACE"]);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|item| item.code == "seam_crosses_prohibited_interface"));
    }

    #[test]
    fn seam_unknown_protected_interface_is_rejected() {
        let mut manifest = base_manifest();
        manifest["seam_edges"][0]["crossed_protected_interface_refs"] = json!(["UNKNOWN-INTERFACE"]);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|item| item.code == "seam_unknown_protected_interface"));
    }

    #[test]
    fn connected_assembly_group_requires_connected_seam_graph() {
        let mut manifest = base_manifest();
        manifest["seam_edges"] = json!([]);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|item| item.code == "assembly_group_disconnected"));
    }

    #[test]
    fn disconnected_assembly_is_allowed_when_group_declares_modularity() {
        let mut manifest = base_manifest();
        manifest["seam_edges"] = json!([]);
        manifest["assembly_groups"][0]["require_connected"] = json!(false);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
    }

    #[test]
    fn interop_review_requires_loss_free_or_explicitly_reported_semantic_loss() {
        let mut manifest = base_manifest();
        manifest["state"] = json!("interop_reviewed");
        bind_patterns(&mut manifest);
        add_interop(&mut manifest, "unknown");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|item| item.code == "unknown_semantic_loss_blocks_interop"));
        assert!(!report.interoperability_reviewed);
    }

    #[test]
    fn explicitly_declared_semantic_loss_requires_report_and_unsupported_fields() {
        let mut manifest = base_manifest();
        manifest["state"] = json!("interop_reviewed");
        bind_patterns(&mut manifest);
        add_interop(&mut manifest, "explicitly_declared");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
        assert!(report.interoperability_reviewed);
    }

    #[test]
    fn explicit_loss_without_report_is_rejected() {
        let mut manifest = base_manifest();
        manifest["state"] = json!("interop_reviewed");
        bind_patterns(&mut manifest);
        add_interop(&mut manifest, "explicitly_declared");
        manifest["interoperability"]["semantic_loss_report_ref"] = json!(null);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|item| item.code == "declared_semantic_loss_incomplete"));
    }

    #[test]
    fn machine_preflight_requires_interop_review_and_exact_program_identity() {
        let mut manifest = base_manifest();
        manifest["state"] = json!("machine_preflight_admitted");
        bind_patterns(&mut manifest);
        add_interop(&mut manifest, "none");
        add_preflight(&mut manifest, "passed");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
        assert!(report.machine_preflight_admitted);
    }

    #[test]
    fn pending_machine_preflight_cannot_be_admitted() {
        let mut manifest = base_manifest();
        manifest["state"] = json!("machine_preflight_admitted");
        bind_patterns(&mut manifest);
        add_interop(&mut manifest, "none");
        add_preflight(&mut manifest, "pending");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|item| item.code == "machine_preflight_still_pending"));
        assert!(!report.machine_preflight_admitted);
    }

    #[test]
    fn machine_preflight_does_not_imply_physical_execution() {
        let mut manifest = base_manifest();
        manifest["state"] = json!("machine_preflight_admitted");
        bind_patterns(&mut manifest);
        add_interop(&mut manifest, "none");
        add_preflight(&mut manifest, "passed");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.is_machine_preflight_admitted());
        // The output type intentionally exposes no machine execution or article qualification flag.
        assert!(report.is_valid());
    }

    #[test]
    fn stale_source_pattern_digest_is_rejected() {
        let mut manifest = base_manifest();
        manifest["state"] = json!("pattern_bound");
        bind_patterns(&mut manifest);
        manifest["panels"][0]["source_pattern_digest"] = json!("sha256:short");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|item| item.code == "invalid_source_pattern_digest"));
    }

    #[test]
    fn reused_pattern_edge_is_rejected() {
        let mut manifest = base_manifest();
        manifest["seam_edges"].push(json!({
            "seam_edge_ref":"SEAM-SECOND",
            "from_panel_id":"forearm_textile_panel_A",
            "from_pattern_edge_ref":"PATTERN-EDGE-A1",
            "to_panel_id":"elbow_gaiter_panel_B",
            "to_pattern_edge_ref":"PATTERN-EDGE-B2",
            "seam_profile_ref":"APPAREL-INTEROP-SEAM-PROFILE-01",
            "seam_construction_ref":"LIGHTSHELL-SEAM-CONSTRUCTION-01",
            "transition_ref":"LIGHTSHELL-TRANSITION-IMPACT-TO-GAITer-01",
            "crossed_protected_interface_refs":[],
            "protected_interface_review_ref":"SEAM-KEEP-OUT-REVIEW-02"
        }));
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|item| item.code == "pattern_edge_reused"));
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let mut manifest = base_manifest();
        manifest["autonomous_machine_execute"] = json!(true);
        assert!(parse_and_validate(&manifest.to_string()).is_err());
    }
}
