// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Lightshell topology-bound manufacturing route admission.
//!
//! This adapter binds robot-specific garment topology to operation nodes and
//! exact MFG-PROC capability references. It does not implement a generic
//! process registry, scheduler, machine program, production executor, or
//! manufacturing authorization.

use serde::Deserialize;
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManufacturingRouteManifest {
    pub schema_version: String,
    pub route_id: String,
    pub state: String,
    pub candidate_id: String,
    pub platform_id: String,
    pub article_revision: Option<String>,
    pub geometry_digest: Option<String>,
    pub topology_ref: String,
    pub topology_revision: String,
    pub topology_digest: String,
    pub topology_state: String,
    pub material_lots: Vec<MaterialLot>,
    pub topology_panels: Vec<String>,
    pub topology_connections: Vec<TopologyConnection>,
    pub articulation_features: Vec<String>,
    pub service_features: Vec<String>,
    pub operations: Vec<ManufacturingOperation>,
    pub evidence_policy: EvidencePolicy,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialLot {
    pub lot_id: String,
    pub material_ref: String,
    pub supplier_lot_ref: Option<String>,
    pub receipt_evidence_ref: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TopologyConnection {
    pub connection_id: String,
    pub connection_kind: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManufacturingOperation {
    pub operation_id: String,
    pub operation_index: u32,
    pub operation_kind: String,
    pub dependencies: Vec<String>,
    pub topology_refs: Vec<TopologyReference>,
    pub material_lot_refs: Vec<String>,
    pub input_component_refs: Vec<String>,
    pub output_component_refs: Vec<String>,
    pub process_binding: ProcessBinding,
    pub inspection_gate: Option<InspectionGate>,
    pub rework_of_component_ref: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TopologyReference {
    pub entity_kind: String,
    pub entity_ref: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessBinding {
    pub owner_ref: String,
    pub process_profile_ref: Option<String>,
    pub capability_state: String,
    pub machine_ref: Option<String>,
    pub tool_ref: Option<String>,
    pub process_revision_ref: Option<String>,
    pub program_revision_ref: Option<String>,
    pub capability_evidence_ref: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InspectionGate {
    pub gate_id: String,
    pub method_selection_ref: Option<String>,
    pub fixture_ref: Option<String>,
    pub fixture_revision_ref: Option<String>,
    pub acceptance_rule_ref: Option<String>,
    pub result_state: String,
    pub result_evidence_ref: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidencePolicy {
    pub capability_is_external_owner_authority: bool,
    pub every_topology_entity_has_operation_disposition: bool,
    pub output_component_has_one_producer: bool,
    pub dependencies_must_be_acyclic: bool,
    pub inspection_is_evidence_gated: bool,
    pub rework_preserves_failed_lineage: bool,
    pub no_execution_or_release_authority: bool,
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
    pub operation_count: usize,
    pub material_lot_count: usize,
    pub topology_entity_count: usize,
    pub accounted_topology_entity_count: usize,
    pub topological_order: Vec<String>,
    pub route_reviewed: bool,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.findings.is_empty()
    }

    /// Route review does not authorize manufacturing or equipment execution.
    pub fn can_enter_route_review(&self) -> bool {
        self.is_valid() && self.route_reviewed
    }
}

pub fn parse_and_validate(input: &str) -> Result<ValidationReport, serde_json::Error> {
    let manifest: ManufacturingRouteManifest = serde_json::from_str(input)?;
    Ok(manifest.validate())
}

impl ManufacturingRouteManifest {
    pub fn validate(&self) -> ValidationReport {
        let mut findings = Vec::new();
        let mut lots: HashMap<&str, &MaterialLot> = HashMap::new();
        let mut topology_connections: HashMap<&str, &TopologyConnection> = HashMap::new();
        let mut operations: HashMap<&str, &ManufacturingOperation> = HashMap::new();
        let mut output_owner: HashMap<&str, &str> = HashMap::new();
        let mut accounted_panels = HashSet::new();
        let mut accounted_connections = HashSet::new();
        let mut accounted_articulation = HashSet::new();
        let mut accounted_service = HashSet::new();
        let mut operation_indices = HashSet::new();

        if self.schema_version != "0.1.0" {
            push(&mut findings, "unsupported_schema_version", "schema_version",
                "Only Lightshell manufacturing-route schema version 0.1.0 is admitted.");
        }
        if is_blank(&self.route_id) || is_blank(&self.candidate_id) || is_blank(&self.platform_id) {
            push(&mut findings, "missing_route_identity", "route_id/candidate_id/platform_id",
                "Route, garment candidate and robot platform identities must be non-empty.");
        }
        if !matches!(self.state.as_str(), "proposed" | "process_review" | "route_reviewed") {
            push(&mut findings, "invalid_route_state", "state",
                "Route state must be proposed, process_review or route_reviewed; there is no executable/released P0 state.");
        }
        if is_blank(&self.topology_ref) || is_blank(&self.topology_revision) || !is_sha256_digest(&self.topology_digest) {
            push(&mut findings, "topology_binding_incomplete", "topology_ref/topology_revision/topology_digest",
                "Route must bind exact topology identity, revision and SHA-256 digest.");
        }
        if !matches!(self.topology_state.as_str(), "proposed" | "source_bound" | "topology_preflighted") {
            push(&mut findings, "invalid_topology_state", "topology_state",
                "Unknown topology state.");
        }
        if self.state == "route_reviewed" {
            if self.topology_state != "topology_preflighted" {
                push(&mut findings, "route_review_requires_topology_preflight", "topology_state",
                    "A route cannot be reviewed until its topology has passed CAD-bound topology preflight.");
            }
            if is_blank_option(&self.article_revision) || !self.geometry_digest.as_deref().is_some_and(is_sha256_digest) {
                push(&mut findings, "route_review_requires_robot_article_binding", "article_revision/geometry_digest",
                    "Route review requires exact robot article revision and geometry digest.");
            }
        }
        if self.operations.is_empty() {
            push(&mut findings, "route_without_operations", "operations",
                "A manufacturing route requires at least one operation node.");
        }
        if self.topology_panels.is_empty() {
            push(&mut findings, "route_without_topology_panels", "topology_panels",
                "A manufacturing route must declare the panels in its topology scope.");
        }

        let panel_ids: HashSet<&str> = self.topology_panels.iter().map(String::as_str).collect();
        if panel_ids.len() != self.topology_panels.len() {
            push(&mut findings, "duplicate_topology_panel_id", "topology_panels",
                "Topology panel IDs must be unique.");
        }
        let articulation_ids: HashSet<&str> = self.articulation_features.iter().map(String::as_str).collect();
        if articulation_ids.len() != self.articulation_features.len() {
            push(&mut findings, "duplicate_articulation_feature_id", "articulation_features",
                "Articulation feature IDs must be unique.");
        }
        let service_ids: HashSet<&str> = self.service_features.iter().map(String::as_str).collect();
        if service_ids.len() != self.service_features.len() {
            push(&mut findings, "duplicate_service_feature_id", "service_features",
                "Service feature IDs must be unique.");
        }

        for (index, lot) in self.material_lots.iter().enumerate() {
            let path = format!("material_lots[{index}]");
            if is_blank(&lot.lot_id) || is_blank(&lot.material_ref) {
                push(&mut findings, "material_lot_identity_missing", &path,
                    "Material lot and material identity must be non-empty.");
            }
            if lots.insert(lot.lot_id.as_str(), lot).is_some() {
                push(&mut findings, "duplicate_material_lot", format!("{path}.lot_id"),
                    "Material lot IDs must be unique within the route.");
            }
            if self.state == "route_reviewed"
                && (is_blank_option(&lot.supplier_lot_ref) || is_blank_option(&lot.receipt_evidence_ref))
            {
                push(&mut findings, "reviewed_route_lot_provenance_missing", &path,
                    "A route-reviewed material lot requires supplier-lot identity and receipt evidence.");
            }
        }

        for (index, connection) in self.topology_connections.iter().enumerate() {
            let path = format!("topology_connections[{index}]");
            if is_blank(&connection.connection_id) {
                push(&mut findings, "topology_connection_identity_missing", &path,
                    "Topology connection identity must be non-empty.");
            }
            if !matches!(connection.connection_kind.as_str(), "sewn_seam" | "knit_integrated" | "bonded_seam" | "removable_attachment") {
                push(&mut findings, "unknown_topology_connection_kind", format!("{path}.connection_kind"),
                    "Unknown topology connection kind.");
            }
            if topology_connections.insert(connection.connection_id.as_str(), connection).is_some() {
                push(&mut findings, "duplicate_topology_connection", format!("{path}.connection_id"),
                    "Topology connection IDs must be unique.");
            }
        }

        for (index, operation) in self.operations.iter().enumerate() {
            let path = format!("operations[{index}]");
            if is_blank(&operation.operation_id) {
                push(&mut findings, "operation_identity_missing", format!("{path}.operation_id"),
                    "Operation identity must be non-empty.");
            }
            if operations.insert(operation.operation_id.as_str(), operation).is_some() {
                push(&mut findings, "duplicate_operation_id", format!("{path}.operation_id"),
                    "Operation IDs must be unique.");
            }
            if !operation_indices.insert(operation.operation_index) {
                push(&mut findings, "duplicate_operation_index", format!("{path}.operation_index"),
                    "Operation index values must be unique and deterministic.");
            }
            if !is_known_operation(&operation.operation_kind) {
                push(&mut findings, "unknown_operation_kind", format!("{path}.operation_kind"),
                    "Operation kind is not in the Lightshell P0 operation vocabulary.");
            }
            if is_blank(&operation.process_binding.owner_ref) {
                push(&mut findings, "process_owner_missing", format!("{path}.process_binding.owner_ref"),
                    "Every operation must identify its generic process-capability owner.");
            }
            if !matches!(operation.process_binding.capability_state.as_str(), "unknown" | "pending" | "qualified" | "not_capable") {
                push(&mut findings, "invalid_operation_capability_state", format!("{path}.process_binding.capability_state"),
                    "Operation capability state must be unknown, pending, qualified or not_capable.");
            }
            if self.state != "proposed" && is_blank_option(&operation.process_binding.process_profile_ref) {
                push(&mut findings, "process_review_profile_missing", format!("{path}.process_binding.process_profile_ref"),
                    "Process-review operations must bind a generic MFG-PROC process-profile candidate.");
            }
            if self.state == "route_reviewed" {
                if operation.process_binding.capability_state != "qualified" {
                    push(&mut findings, "route_review_unqualified_operation", format!("{path}.process_binding.capability_state"),
                        "Every operation must have qualified process capability before route review.");
                }
                for (field, value) in [
                    ("machine_ref", operation.process_binding.machine_ref.as_deref()),
                    ("tool_ref", operation.process_binding.tool_ref.as_deref()),
                    ("process_revision_ref", operation.process_binding.process_revision_ref.as_deref()),
                    ("program_revision_ref", operation.process_binding.program_revision_ref.as_deref()),
                    ("capability_evidence_ref", operation.process_binding.capability_evidence_ref.as_deref()),
                    ("process_profile_ref", operation.process_binding.process_profile_ref.as_deref()),
                ] {
                    if value.is_none_or(is_blank) {
                        push(&mut findings, "route_review_process_binding_incomplete", format!("{path}.process_binding.{field}"),
                            "Qualified route review requires exact process profile, machine, tool, process/program revisions and capability evidence.");
                    }
                }
            }
            if operation.output_component_refs.is_empty() {
                push(&mut findings, "operation_without_outputs", format!("{path}.output_component_refs"),
                    "Every manufacturing operation must produce at least one component/article revision.");
            }
            if operation.output_component_refs.iter().any(|v| is_blank(v)) {
                push(&mut findings, "blank_output_component", format!("{path}.output_component_refs"),
                    "Output component IDs must be non-empty.");
            }
            for output in &operation.output_component_refs {
                if let Some(previous) = output_owner.insert(output.as_str(), operation.operation_id.as_str()) {
                    push(&mut findings, "duplicate_component_output_owner", format!("{path}.output_component_refs"),
                        format!("Component output {output} is already produced by operation {previous}; revisions must be distinct."));
                }
            }
            for dependency in &operation.dependencies {
                if dependency == &operation.operation_id {
                    push(&mut findings, "self_dependency", format!("{path}.dependencies"),
                        "An operation cannot depend on itself.");
                }
            }
            for lot_ref in &operation.material_lot_refs {
                if !lots.contains_key(lot_ref.as_str()) {
                    push(&mut findings, "operation_unknown_material_lot", format!("{path}.material_lot_refs"),
                        format!("Operation references unknown material lot {lot_ref}."));
                }
            }
            for input in &operation.input_component_refs {
                if let Some(producer) = output_owner.get(input.as_str()) {
                    if *producer == operation.operation_id {
                        push(&mut findings, "operation_consumes_own_output", format!("{path}.input_component_refs"),
                            "An operation cannot consume its own output.");
                    }
                }
            }
            for topology_ref in &operation.topology_refs {
                if is_blank(&topology_ref.entity_ref) {
                    push(&mut findings, "blank_topology_reference", format!("{path}.topology_refs"),
                        "Topology reference identity must be non-empty.");
                    continue;
                }
                match topology_ref.entity_kind.as_str() {
                    "panel" => {
                        if !panel_ids.contains(topology_ref.entity_ref.as_str()) {
                            push(&mut findings, "operation_unknown_topology_panel", format!("{path}.topology_refs"),
                                "Operation references a panel outside the bound topology.");
                        } else {
                            accounted_panels.insert(topology_ref.entity_ref.as_str());
                        }
                    }
                    "connection" => {
                        if !topology_connections.contains_key(topology_ref.entity_ref.as_str()) {
                            push(&mut findings, "operation_unknown_topology_connection", format!("{path}.topology_refs"),
                                "Operation references a connection outside the bound topology.");
                        } else {
                            accounted_connections.insert(topology_ref.entity_ref.as_str());
                            let connection = topology_connections[topology_ref.entity_ref.as_str()];
                            if !operation_matches_connection(&operation.operation_kind, &connection.connection_kind) {
                                push(&mut findings, "operation_connection_kind_mismatch", format!("{path}.topology_refs"),
                                    "Operation kind does not match the topology connection construction.");
                            }
                        }
                    }
                    "articulation_feature" => {
                        if !articulation_ids.contains(topology_ref.entity_ref.as_str()) {
                            push(&mut findings, "operation_unknown_articulation_feature", format!("{path}.topology_refs"),
                                "Operation references an articulation feature outside the bound topology.");
                        } else {
                            accounted_articulation.insert(topology_ref.entity_ref.as_str());
                        }
                    }
                    "service_feature" => {
                        if !service_ids.contains(topology_ref.entity_ref.as_str()) {
                            push(&mut findings, "operation_unknown_service_feature", format!("{path}.topology_refs"),
                                "Operation references a service feature outside the bound topology.");
                        } else {
                            accounted_service.insert(topology_ref.entity_ref.as_str());
                        }
                    }
                    "garment" => {
                        if topology_ref.entity_ref != self.candidate_id {
                            push(&mut findings, "operation_wrong_garment_reference", format!("{path}.topology_refs"),
                                "Garment-level operation must bind this exact candidate ID.");
                        }
                    }
                    _ => push(&mut findings, "unknown_topology_entity_kind", format!("{path}.topology_refs"),
                        "Unknown topology entity kind."),
                }
            }
            validate_operation_shape(operation, &path, &mut findings);
            validate_inspection_gate(operation, &path, self.state.as_str(), &mut findings);
        }

        for (index, operation) in self.operations.iter().enumerate() {
            let path = format!("operations[{index}]");
            for input in &operation.input_component_refs {
                match output_owner.get(input.as_str()) {
                    Some(producer) if *producer != operation.operation_id => {
                        if !operation.dependencies.iter().any(|dependency| dependency == producer) {
                            push(&mut findings, "component_input_missing_dependency", format!("{path}.input_component_refs"),
                                format!("Input component {input} is produced by {producer}, which must be an explicit predecessor dependency."));
                        }
                    }
                    Some(_) => {}
                    None => push(&mut findings, "component_input_without_producer", format!("{path}.input_component_refs"),
                        format!("Input component {input} has no operation producer in this route; material lots must be declared separately.")),
                }
            }
            for dependency in &operation.dependencies {
                if !operations.contains_key(dependency.as_str()) {
                    push(&mut findings, "unknown_operation_dependency", format!("{path}.dependencies"),
                        format!("Dependency {dependency} is not declared in this route."));
                }
            }
        }

        let topological_order = deterministic_topological_order(&self.operations, &mut findings);
        for panel in &self.topology_panels {
            if !accounted_panels.contains(panel.as_str()) {
                push(&mut findings, "topology_panel_without_operation", "topology_panels",
                    format!("Topology panel {panel} has no manufacturing operation disposition."));
            }
        }
        for connection in &self.topology_connections {
            if !accounted_connections.contains(connection.connection_id.as_str()) {
                push(&mut findings, "topology_connection_without_operation", "topology_connections",
                    format!("Topology connection {} has no matching join operation.", connection.connection_id));
            }
        }
        for feature in &self.articulation_features {
            if !accounted_articulation.contains(feature.as_str()) {
                push(&mut findings, "articulation_feature_without_operation", "articulation_features",
                    format!("Articulation feature {feature} has no operation disposition."));
            }
        }
        for feature in &self.service_features {
            if !accounted_service.contains(feature.as_str()) {
                push(&mut findings, "service_feature_without_operation", "service_features",
                    format!("Service feature {feature} has no operation disposition."));
            }
        }

        if !self.evidence_policy.capability_is_external_owner_authority
            || !self.evidence_policy.every_topology_entity_has_operation_disposition
            || !self.evidence_policy.output_component_has_one_producer
            || !self.evidence_policy.dependencies_must_be_acyclic
            || !self.evidence_policy.inspection_is_evidence_gated
            || !self.evidence_policy.rework_preserves_failed_lineage
            || !self.evidence_policy.no_execution_or_release_authority
        {
            push(&mut findings, "unsafe_route_policy", "evidence_policy",
                "Process authority, topology coverage, lineage, acyclicity, inspection and no-execution boundaries are mandatory.");
        }

        let topology_entity_count = panel_ids.len() + topology_connections.len() + articulation_ids.len() + service_ids.len();
        let accounted_topology_entity_count = accounted_panels.len() + accounted_connections.len()
            + accounted_articulation.len() + accounted_service.len();

        ValidationReport {
            findings,
            operation_count: self.operations.len(),
            material_lot_count: self.material_lots.len(),
            topology_entity_count,
            accounted_topology_entity_count,
            topological_order,
            route_reviewed: self.state == "route_reviewed",
        }
    }
}

fn validate_operation_shape(
    operation: &ManufacturingOperation,
    path: &str,
    findings: &mut Vec<Finding>,
) {
    let kinds: HashSet<&str> = operation.topology_refs.iter().map(|item| item.entity_kind.as_str()).collect();
    match operation.operation_kind.as_str() {
        "material_receipt" => {
            if operation.material_lot_refs.is_empty() {
                push(findings, "material_receipt_without_lot", format!("{path}.material_lot_refs"),
                    "Material receipt operation must reference at least one declared material lot.");
            }
        }
        "textile_forming" | "panel_cut" => {
            if !kinds.contains("panel") {
                push(findings, "panel_operation_without_panel", format!("{path}.topology_refs"),
                    "Textile forming or cutting must bind at least one topology panel.");
            }
        }
        "edge_finish" => {
            if !kinds.contains("panel") {
                push(findings, "edge_finish_without_panel", format!("{path}.topology_refs"),
                    "Edge finishing must bind at least one topology panel.");
            }
        }
        "seam_join" | "knit_integrate" | "bond_join" | "removable_attach" => {
            if !kinds.contains("connection") {
                push(findings, "join_operation_without_connection", format!("{path}.topology_refs"),
                    "A joining operation must bind at least one topology connection.");
            }
        }
        "sensor_install" => {
            if !kinds.contains("panel") {
                push(findings, "sensor_install_without_panel", format!("{path}.topology_refs"),
                    "Sensor installation must bind its host panel.");
            }
        }
        "clean_finish" | "in_process_inspect" => {
            if operation.topology_refs.is_empty() {
                push(findings, "operation_without_topology_scope", format!("{path}.topology_refs"),
                    "Finishing and in-process inspection must bind their topology scope.");
            }
        }
        "final_inspect" | "pack_release" => {
            if !operation.topology_refs.iter().any(|item| item.entity_kind == "garment") {
                push(findings, "garment_operation_without_candidate", format!("{path}.topology_refs"),
                    "Final inspection and pack/release planning must bind the exact garment candidate.");
            }
        }
        "rework" => {
            if is_blank_option(&operation.rework_of_component_ref) {
                push(findings, "rework_lineage_missing", format!("{path}.rework_of_component_ref"),
                    "Rework must point to the exact failed/predecessor component without overwriting it.");
            }
            if operation.input_component_refs.is_empty() {
                push(findings, "rework_without_input", format!("{path}.input_component_refs"),
                    "Rework must consume the prior component revision as an explicit input.");
            }
        }
        _ => {}
    }
}

fn validate_inspection_gate(
    operation: &ManufacturingOperation,
    path: &str,
    route_state: &str,
    findings: &mut Vec<Finding>,
) {
    let inspection_kind = matches!(operation.operation_kind.as_str(), "in_process_inspect" | "final_inspect");
    if inspection_kind && operation.inspection_gate.is_none() {
        push(findings, "inspection_operation_without_gate", format!("{path}.inspection_gate"),
            "Inspection operations require an explicit evidence-gated inspection record.");
        return;
    }
    if let Some(gate) = operation.inspection_gate.as_ref() {
        if !inspection_kind {
            push(findings, "gate_attached_to_noninspection", format!("{path}.inspection_gate"),
                "P0 inspection gates may only attach to in_process_inspect or final_inspect operations.");
        }
        if is_blank(&gate.gate_id) {
            push(findings, "inspection_gate_identity_missing", format!("{path}.inspection_gate.gate_id"),
                "Inspection gate identity must be non-empty.");
        }
        if !matches!(gate.result_state.as_str(), "not_run" | "pass" | "fail" | "inconclusive") {
            push(findings, "invalid_inspection_result_state", format!("{path}.inspection_gate.result_state"),
                "Inspection result state must be not_run, pass, fail or inconclusive.");
        }
        if gate.result_state == "pass" || gate.result_state == "fail" || gate.result_state == "inconclusive" {
            for (field, value) in [
                ("method_selection_ref", gate.method_selection_ref.as_deref()),
                ("fixture_ref", gate.fixture_ref.as_deref()),
                ("fixture_revision_ref", gate.fixture_revision_ref.as_deref()),
                ("acceptance_rule_ref", gate.acceptance_rule_ref.as_deref()),
                ("result_evidence_ref", gate.result_evidence_ref.as_deref()),
            ] {
                if value.is_none_or(is_blank) {
                    push(findings, "inspection_result_provenance_missing", format!("{path}.inspection_gate.{field}"),
                        "A recorded inspection result must bind method, fixture revision, acceptance rule and result evidence.");
                }
            }
        }
        if route_state == "route_reviewed" {
            for (field, value) in [
                ("method_selection_ref", gate.method_selection_ref.as_deref()),
                ("fixture_ref", gate.fixture_ref.as_deref()),
                ("fixture_revision_ref", gate.fixture_revision_ref.as_deref()),
                ("acceptance_rule_ref", gate.acceptance_rule_ref.as_deref()),
            ] {
                if value.is_none_or(is_blank) {
                    push(findings, "route_review_inspection_plan_incomplete", format!("{path}.inspection_gate.{field}"),
                        "Route review requires planned inspection method, fixture revision and acceptance rule.");
                }
            }
            if gate.result_state == "fail" || gate.result_state == "inconclusive" {
                push(findings, "route_review_inspection_not_admitted", format!("{path}.inspection_gate.result_state"),
                    "A failed or inconclusive inspection blocks route review.");
            }
        }
    }
}

fn operation_matches_connection(operation_kind: &str, connection_kind: &str) -> bool {
    matches!(
        (operation_kind, connection_kind),
        ("seam_join", "sewn_seam")
            | ("knit_integrate", "knit_integrated")
            | ("bond_join", "bonded_seam")
            | ("removable_attach", "removable_attachment")
    )
}

fn deterministic_topological_order(
    operations: &[ManufacturingOperation],
    findings: &mut Vec<Finding>,
) -> Vec<String> {
    let mut by_id: HashMap<&str, &ManufacturingOperation> = HashMap::new();
    for operation in operations {
        by_id.insert(operation.operation_id.as_str(), operation);
    }
    let mut indegree: HashMap<&str, usize> = HashMap::new();
    let mut children: HashMap<&str, Vec<&str>> = HashMap::new();
    for operation in operations {
        indegree.entry(operation.operation_id.as_str()).or_insert(0);
        for dependency in &operation.dependencies {
            if by_id.contains_key(dependency.as_str()) {
                *indegree.entry(operation.operation_id.as_str()).or_insert(0) += 1;
                children.entry(dependency.as_str()).or_default().push(operation.operation_id.as_str());
            }
        }
    }
    let mut ready: Vec<(&str, u32)> = indegree.iter()
        .filter_map(|(id, degree)| {
            (*degree == 0).then(|| (*id, by_id.get(id).map(|op| op.operation_index).unwrap_or(u32::MAX)))
        })
        .collect();
    ready.sort_by(|a, b| (a.1, a.0).cmp(&(b.1, b.0)));
    let mut queue: VecDeque<&str> = ready.into_iter().map(|(id, _)| id).collect();
    let mut order = Vec::new();
    while let Some(id) = queue.pop_front() {
        order.push(id.to_string());
        if let Some(next) = children.get(id) {
            for child in next {
                if let Some(degree) = indegree.get_mut(child) {
                    *degree -= 1;
                    if *degree == 0 {
                        let mut pending: Vec<(&str, u32)> = queue.drain(..)
                            .map(|queued| (queued, by_id.get(queued).map(|op| op.operation_index).unwrap_or(u32::MAX)))
                            .collect();
                        pending.push((*child, by_id.get(child).map(|op| op.operation_index).unwrap_or(u32::MAX)));
                        pending.sort_by(|a, b| (a.1, a.0).cmp(&(b.1, b.0)));
                        queue = pending.into_iter().map(|(queued, _)| queued).collect();
                    }
                }
            }
        }
    }
    if order.len() != operations.len() {
        push(findings, "operation_dependency_cycle", "operations.dependencies",
            "Manufacturing operation graph contains a dependency cycle.");
    }
    order
}

fn is_known_operation(value: &str) -> bool {
    matches!(value,
        "material_receipt" | "textile_forming" | "panel_cut" | "edge_finish" |
        "seam_join" | "knit_integrate" | "bond_join" | "removable_attach" |
        "sensor_install" | "clean_finish" | "in_process_inspect" |
        "final_inspect" | "rework" | "pack_release")
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

    fn operation(
        id: &str,
        index: u32,
        kind: &str,
        dependencies: Vec<&str>,
        topology_refs: Vec<Value>,
        inputs: Vec<&str>,
        outputs: Vec<&str>,
    ) -> Value {
        json!({
            "operation_id":id,
            "operation_index":index,
            "operation_kind":kind,
            "dependencies":dependencies,
            "topology_refs":topology_refs,
            "material_lot_refs":["LOT-001"],
            "input_component_refs":inputs,
            "output_component_refs":outputs,
            "process_binding":{
                "owner_ref":"MFG-PROC-5686",
                "process_profile_ref":null,
                "capability_state":"unknown",
                "machine_ref":null,
                "tool_ref":null,
                "process_revision_ref":null,
                "program_revision_ref":null,
                "capability_evidence_ref":null
            },
            "inspection_gate":null,
            "rework_of_component_ref":null
        })
    }

    fn base_manifest() -> Value {
        let panel = |id: &str| json!({"entity_kind":"panel","entity_ref":id});
        let conn = |id: &str| json!({"entity_kind":"connection","entity_ref":id});
        let art = |id: &str| json!({"entity_kind":"articulation_feature","entity_ref":id});
        let service = |id: &str| json!({"entity_kind":"service_feature","entity_ref":id});
        json!({
            "schema_version":"0.1.0",
            "route_id":"LIGHTSHELL-ROUTE-SYNTHETIC-001",
            "state":"proposed",
            "candidate_id":"ELBOW-GARMENT-CANDIDATE-001",
            "platform_id":"SYNTHETIC-ROBOT-001",
            "article_revision":null,
            "geometry_digest":null,
            "topology_ref":"LIGHTSHELL-TOPOLOGY-SYNTHETIC-001",
            "topology_revision":"TOPOLOGY-REV-A",
            "topology_digest":format!("sha256:{}", "a".repeat(64)),
            "topology_state":"proposed",
            "material_lots":[
                {"lot_id":"LOT-001","material_ref":"MATERIAL-SPACER-001","supplier_lot_ref":null,"receipt_evidence_ref":null}
            ],
            "topology_panels":["upper_arm_shell","elbow_gusset","forearm_shell"],
            "topology_connections":[
                {"connection_id":"SEAM-UPPER-GUSSET","connection_kind":"sewn_seam"},
                {"connection_id":"ATTACH-GUSSET-FOREARM","connection_kind":"removable_attachment"}
            ],
            "articulation_features":["ELBOW-ARTICULATION-001"],
            "service_features":["FOREARM-SERVICE-001"],
            "operations":[
                operation("OP-RECEIVE",0,"material_receipt",vec![],vec![],vec![],vec!["COMP-MATERIAL-001"]),
                operation("OP-FORM-UPPER",1,"textile_forming",vec!["OP-RECEIVE"],vec![panel("upper_arm_shell")],vec!["COMP-MATERIAL-001"],vec!["COMP-UPPER-001"]),
                operation("OP-CUT-GUSSET",2,"panel_cut",vec!["OP-RECEIVE"],vec![panel("elbow_gusset")],vec!["COMP-MATERIAL-001"],vec!["COMP-GUSSET-001"]),
                operation("OP-CUT-FOREARM",3,"panel_cut",vec!["OP-RECEIVE"],vec![panel("forearm_shell")],vec!["COMP-MATERIAL-001"],vec!["COMP-FOREARM-001"]),
                operation("OP-SEW",4,"seam_join",vec!["OP-FORM-UPPER","OP-CUT-GUSSET"],vec![conn("SEAM-UPPER-GUSSET"),art("ELBOW-ARTICULATION-001")],vec!["COMP-UPPER-001","COMP-GUSSET-001"],vec!["COMP-UPPER-GUSSET-001"]),
                operation("OP-ATTACH",5,"removable_attach",vec!["OP-SEW","OP-CUT-FOREARM"],vec![conn("ATTACH-GUSSET-FOREARM"),service("FOREARM-SERVICE-001")],vec!["COMP-UPPER-GUSSET-001","COMP-FOREARM-001"],vec!["COMP-ASSEMBLY-001"])
            ],
            "evidence_policy":{
                "capability_is_external_owner_authority":true,
                "every_topology_entity_has_operation_disposition":true,
                "output_component_has_one_producer":true,
                "dependencies_must_be_acyclic":true,
                "inspection_is_evidence_gated":true,
                "rework_preserves_failed_lineage":true,
                "no_execution_or_release_authority":true
            }
        })
    }

    #[test]
    fn synthetic_topology_operation_graph_is_valid_but_not_route_reviewed() {
        let report = parse_and_validate(&base_manifest().to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
        assert_eq!(report.operation_count, 6);
        assert_eq!(report.topology_entity_count, 7);
        assert_eq!(report.accounted_topology_entity_count, 7);
        assert_eq!(report.topological_order.len(), 6);
        assert!(!report.can_enter_route_review());
    }

    #[test]
    fn dependency_cycle_is_rejected() {
        let mut manifest = base_manifest();
        manifest["operations"][0]["dependencies"] = json!(["OP-ATTACH"]);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "operation_dependency_cycle"));
    }

    #[test]
    fn dangling_dependency_is_rejected() {
        let mut manifest = base_manifest();
        manifest["operations"][1]["dependencies"] = json!(["MISSING-OP"]);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "unknown_operation_dependency"));
    }

    #[test]
    fn duplicate_component_output_owner_is_rejected() {
        let mut manifest = base_manifest();
        manifest["operations"][2]["output_component_refs"] = json!(["COMP-UPPER-001"]);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "duplicate_component_output_owner"));
    }

    #[test]
    fn component_input_requires_a_producer() {
        let mut manifest = base_manifest();
        manifest["operations"][1]["input_component_refs"] = json!(["MYSTERY-COMPONENT"]);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "component_input_without_producer"));
    }

    #[test]
    fn component_producer_must_be_an_explicit_predecessor() {
        let mut manifest = base_manifest();
        manifest["operations"][4]["dependencies"] = json!(["OP-CUT-GUSSET"]);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "component_input_missing_dependency"));
    }

    #[test]
    fn operation_connection_kind_must_match_join_process() {
        let mut manifest = base_manifest();
        manifest["operations"][4]["operation_kind"] = json!("removable_attach");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "operation_connection_kind_mismatch"));
    }

    #[test]
    fn every_topology_panel_requires_a_manufacturing_disposition() {
        let mut manifest = base_manifest();
        manifest["operations"][3]["topology_refs"] = json!([]);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "topology_panel_without_operation"));
    }

    #[test]
    fn every_topology_connection_requires_matching_join_operation() {
        let mut manifest = base_manifest();
        manifest["operations"][5]["topology_refs"] = json!([]);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "topology_connection_without_operation"));
    }

    #[test]
    fn route_review_requires_cad_bound_topology_and_robot_identity() {
        let mut manifest = base_manifest();
        manifest["state"] = json!("route_reviewed");
        manifest["topology_state"] = json!("source_bound");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "route_review_requires_topology_preflight"));
        assert!(report.findings.iter().any(|f| f.code == "route_review_requires_robot_article_binding"));
    }

    #[test]
    fn route_review_rejects_pending_process_capability() {
        let mut manifest = base_manifest();
        manifest["state"] = json!("route_reviewed");
        manifest["topology_state"] = json!("topology_preflighted");
        manifest["article_revision"] = json!("ROBOT-REV-A");
        manifest["geometry_digest"] = json!(format!("sha256:{}", "b".repeat(64)));
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "route_review_unqualified_operation"));
    }

    #[test]
    fn route_review_requires_inspection_method_fixture_and_acceptance_rule() {
        let mut manifest = base_manifest();
        manifest["operations"].push(operation(
            "OP-FINAL-INSPECT",6,"final_inspect",vec!["OP-ATTACH"],
            vec![json!({"entity_kind":"garment","entity_ref":"ELBOW-GARMENT-CANDIDATE-001"})],
            vec!["COMP-ASSEMBLY-001"],vec!["COMP-INSPECTED-001"]
        ));
        manifest["operations"][6]["inspection_gate"] = json!({
            "gate_id":"FINAL-GATE-01","method_selection_ref":null,"fixture_ref":null,"fixture_revision_ref":null,
            "acceptance_rule_ref":null,"result_state":"not_run","result_evidence_ref":null
        });
        manifest["state"] = json!("route_reviewed");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "route_review_inspection_plan_incomplete"));
    }

    #[test]
    fn fail_or_inconclusive_inspection_blocks_route_review() {
        let mut manifest = base_manifest();
        manifest["operations"].push(operation(
            "OP-FINAL-INSPECT",6,"final_inspect",vec!["OP-ATTACH"],
            vec![json!({"entity_kind":"garment","entity_ref":"ELBOW-GARMENT-CANDIDATE-001"})],
            vec!["COMP-ASSEMBLY-001"],vec!["COMP-INSPECTED-001"]
        ));
        manifest["operations"][6]["inspection_gate"] = json!({
            "gate_id":"FINAL-GATE-01","method_selection_ref":"METHOD-01","fixture_ref":"FIXTURE-01",
            "fixture_revision_ref":"FIXTURE-REV-A","acceptance_rule_ref":"ACCEPT-01",
            "result_state":"fail","result_evidence_ref":"FAILED-INSPECTION-01"
        });
        manifest["state"] = json!("route_reviewed");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "route_review_inspection_not_admitted"));
    }

    #[test]
    fn rework_requires_prior_component_and_preserves_new_lineage() {
        let mut manifest = base_manifest();
        manifest["operations"].push(operation(
            "OP-REWORK",6,"rework",vec!["OP-ATTACH"],
            vec![json!({"entity_kind":"panel","entity_ref":"forearm_shell"})],
            vec!["COMP-ASSEMBLY-001"],vec!["COMP-ASSEMBLY-002"]
        ));
        manifest["operations"][6]["rework_of_component_ref"] = json!("COMP-ASSEMBLY-001");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
    }

    #[test]
    fn rework_cannot_overwrite_predecessor_component_identity() {
        let mut manifest = base_manifest();
        manifest["operations"].push(operation(
            "OP-REWORK",6,"rework",vec!["OP-ATTACH"],
            vec![json!({"entity_kind":"panel","entity_ref":"forearm_shell"})],
            vec!["COMP-ASSEMBLY-001"],vec!["COMP-ASSEMBLY-001"]
        ));
        manifest["operations"][6]["rework_of_component_ref"] = json!("COMP-ASSEMBLY-001");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "duplicate_component_output_owner"));
    }

    #[test]
    fn route_review_requires_qualified_machine_process_and_tool_binding() {
        let mut manifest = base_manifest();
        manifest["state"] = json!("route_reviewed");
        manifest["topology_state"] = json!("topology_preflighted");
        manifest["article_revision"] = json!("ROBOT-REV-A");
        manifest["geometry_digest"] = json!(format!("sha256:{}", "b".repeat(64)));
        for op in manifest["operations"].as_array_mut().unwrap() {
            op["process_binding"]["process_profile_ref"] = json!("MFG-PROC-PROFILE-01");
            op["process_binding"]["capability_state"] = json!("qualified");
            op["process_binding"]["machine_ref"] = json!("MACHINE-01");
            op["process_binding"]["tool_ref"] = json!("TOOL-01");
            op["process_binding"]["process_revision_ref"] = json!("PROCESS-REV-A");
            op["process_binding"]["program_revision_ref"] = json!("PROGRAM-REV-A");
            op["process_binding"]["capability_evidence_ref"] = json!("CAPABILITY-RECEIPT-01");
        }
        manifest["material_lots"][0]["supplier_lot_ref"] = json!("SUPPLIER-LOT-01");
        manifest["material_lots"][0]["receipt_evidence_ref"] = json!("RECEIPT-01");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.is_valid(), "{:?}", report.findings);
        assert!(report.can_enter_route_review());
    }

    #[test]
    fn operation_kind_vocabulary_rejects_arbitrary_machine_commands() {
        let mut manifest = base_manifest();
        manifest["operations"][0]["operation_kind"] = json!("run_arbitrary_machine_code");
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "unknown_operation_kind"));
    }

    #[test]
    fn material_lot_must_be_declared_before_use() {
        let mut manifest = base_manifest();
        manifest["operations"][1]["material_lot_refs"] = json!(["LOT-MISSING"]);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "operation_unknown_material_lot"));
    }

    #[test]
    fn duplicate_operation_indices_are_rejected() {
        let mut manifest = base_manifest();
        manifest["operations"][1]["operation_index"] = json!(0);
        let report = parse_and_validate(&manifest.to_string()).unwrap();
        assert!(report.findings.iter().any(|f| f.code == "duplicate_operation_index"));
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let mut manifest = base_manifest();
        manifest["invented_release_authorization"] = json!(true);
        assert!(parse_and_validate(&manifest.to_string()).is_err());
    }
}
