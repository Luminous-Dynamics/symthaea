// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Synthetic-only structural qualification for HABITAT-VERTICAL-001.
//!
//! This validator checks fixture integrity and internal consistency. It does
//! not schedule production, qualify materials or suppliers, validate a factory,
//! or establish physical, human, safety, economic, or environmental claims.

use std::collections::{HashMap, HashSet};

pub mod accounting;
pub mod scenario_campaign;
pub mod resilience_options;
pub mod assumption_falsification;
pub mod event_replay;
pub mod material_ledger;
pub mod measurement_gate;
pub mod audit_bundle;
pub mod recall_drill;
pub mod quantity_genealogy;

use serde_json::Value;
use symthaea_graph_theory::Graph;

/// Exact synthetic fixture pinned by this crate.
pub const EMBEDDED_FIXTURE: &str =
    include_str!("../../../../docs/research/fixtures/habitat-vertical-v0.1.json");

/// One deterministic structural-validation finding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationIssue {
    /// Stable machine-readable finding code.
    pub code: &'static str,
    /// JSON-style location within the input.
    pub path: String,
    /// Human-readable explanation.
    pub message: String,
}

/// Report for the fixture's structural integrity only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationReport {
    /// Findings sorted by code, path, and message.
    pub issues: Vec<ValidationIssue>,
}

impl ValidationReport {
    /// Whether no structural findings were produced.
    ///
    /// A true result is only a structural fixture pass. It is not product,
    /// supplier, factory, production, or safety qualification.
    pub fn is_valid(&self) -> bool {
        self.issues.is_empty()
    }
}

/// Parse and structurally validate a JSON fixture.
pub fn validate_fixture_str(input: &str) -> Result<ValidationReport, serde_json::Error> {
    let value: Value = serde_json::from_str(input)?;
    Ok(validate_fixture(&value))
}

/// Structurally validate a HABITAT-VERTICAL-001 synthetic fixture.
///
/// The routine fails closed on missing or contradictory critical structure.
/// It reports all findings it can safely determine, then sorts them so the
/// result is deterministic regardless of hash-map iteration order.
pub fn validate_fixture(root: &Value) -> ValidationReport {
    let mut issues = Vec::new();

    expect_string(root, "schema_version", "0.1.0", &mut issues);
    expect_string(root, "program_id", "HABITAT-VERTICAL-001", &mut issues);
    expect_string(
        root,
        "fixture_kind",
        "synthetic_protocol_fixture",
        &mut issues,
    );
    expect_string(root, "evidence_status", "synthetic_only", &mut issues);

    if !root
        .get("claim_ceiling")
        .and_then(Value::as_str)
        .is_some_and(|value| value.contains("No physical"))
    {
        push(
            &mut issues,
            "CLAIM_CEILING_MISSING",
            "claim_ceiling",
            "synthetic fixture must state that physical/product claims are not established",
        );
    }

    validate_synthetic_claim_boundary(root, &mut issues);
    validate_unique_ids(root, "existing_owners", "id", &mut issues);
    validate_unique_ids(root, "synthetic_suppliers", "supplier_id", &mut issues);
    validate_unique_ids(root, "synthetic_material_lots", "lot_id", &mut issues);
    validate_unique_ids(root, "synthetic_resources", "resource_id", &mut issues);
    validate_unique_ids(root, "synthetic_process_graph", "operation_id", &mut issues);
    validate_unique_ids(root, "proposed_targets", "id", &mut issues);
    validate_unique_ids(root, "synthetic_cost_scenarios", "scenario_id", &mut issues);
    validate_unique_ids(root, "synthetic_material_balance_cases", "case_id", &mut issues);
    validate_unique_ids(root, "metric_definitions", "id", &mut issues);
    validate_unique_ids(root, "disruptions", "id", &mut issues);
    validate_unique_ids(root, "negative_controls", "case_id", &mut issues);
    validate_unique_ids(root, "hypothetical_factory.zones", "zone_id", &mut issues);
    validate_unique_ids(root, "traceability_nodes", "node_id", &mut issues);
    validate_unique_ids(root, "traceability_edges", "edge_id", &mut issues);
    validate_unique_ids(root, "expected_containment_queries", "source_lot_id", &mut issues);

    validate_supplier_lot_references(root, &mut issues);
    validate_resources_and_process_graph(root, &mut issues);
    validate_traceability_graph(root, &mut issues);
    validate_final_inspection_gate(root, &mut issues);
    validate_negative_controls(root, &mut issues);
    validate_disruptions(root, &mut issues);
    validate_required_target_ranges(root, &mut issues);
    validate_cost_scenarios(root, &mut issues);
    validate_material_balance_cases(root, &mut issues);
    validate_verification_status(root, &mut issues);

    issues.sort_by(|left, right| {
        left.code
            .cmp(right.code)
            .then_with(|| left.path.cmp(&right.path))
            .then_with(|| left.message.cmp(&right.message))
    });

    ValidationReport { issues }
}

fn push(
    issues: &mut Vec<ValidationIssue>,
    code: &'static str,
    path: impl Into<String>,
    message: impl Into<String>,
) {
    issues.push(ValidationIssue {
        code,
        path: path.into(),
        message: message.into(),
    });
}

fn expect_string(
    root: &Value,
    field: &str,
    expected: &str,
    issues: &mut Vec<ValidationIssue>,
) {
    match root.get(field).and_then(Value::as_str) {
        Some(actual) if actual == expected => {}
        Some(actual) => push(
            issues,
            "UNEXPECTED_FIELD_VALUE",
            field,
            format!("expected {expected:?}, got {actual:?}"),
        ),
        None => push(
            issues,
            "MISSING_REQUIRED_STRING",
            field,
            format!("expected required string {expected:?}"),
        ),
    }
}

fn path_for_item(list: &str, index: usize, field: &str) -> String {
    format!("{list}[{index}].{field}")
}

fn id_set(root: &Value, field: &str, id_field: &str) -> HashSet<String> {
    let Some(items) = root.get(field).and_then(Value::as_array) else {
        return HashSet::new();
    };
    items
        .iter()
        .filter_map(|item| item.get(id_field).and_then(Value::as_str))
        .filter(|id| !id.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn validate_unique_ids(
    root: &Value,
    field: &str,
    id_field: &str,
    issues: &mut Vec<ValidationIssue>,
) {
    let Some(items) = get_nested_array(root, field, issues) else {
        return;
    };
    let mut seen = HashSet::new();
    for (index, item) in items.iter().enumerate() {
        let path = path_for_item(field, index, id_field);
        match item.get(id_field).and_then(Value::as_str) {
            Some(id) if !id.trim().is_empty() => {
                if !seen.insert(id.to_owned()) {
                    push(
                        issues,
                        "DUPLICATE_ID",
                        path,
                        format!("duplicate identifier {id:?}"),
                    );
                }
            }
            _ => push(
                issues,
                "MISSING_ID",
                path,
                "required identifier must be a non-empty string",
            ),
        }
    }
}

fn get_nested_array<'a>(
    root: &'a Value,
    dotted_field: &str,
    issues: &mut Vec<ValidationIssue>,
) -> Option<&'a Vec<Value>> {
    let mut current = root;
    for segment in dotted_field.split('.') {
        current = match current.get(segment) {
            Some(value) => value,
            None => {
                push(
                    issues,
                    "MISSING_REQUIRED_ARRAY",
                    dotted_field,
                    "required nested array path does not exist",
                );
                return None;
            }
        };
    }
    match current.as_array() {
        Some(values) => Some(values),
        None => {
            push(
                issues,
                "MISSING_REQUIRED_ARRAY",
                dotted_field,
                "required nested field must be a JSON array",
            );
            None
        }
    }
}

fn string_list(item: &Value, field: &str) -> Option<Vec<String>> {
    item.get(field)?
        .as_array()?
        .iter()
        .map(|value| value.as_str().map(ToOwned::to_owned))
        .collect()
}

/// Return all directed descendants of a lot in stable lexical order.
///
/// The result is a synthetic graph query, not proof that a physical operation
/// occurred or that an affected unit has been measured or quarantined.
pub fn traceability_descendants(
    root: &Value,
    source_lot_id: &str,
) -> Result<Vec<String>, String> {
    let nodes = root
        .get("traceability_nodes")
        .and_then(Value::as_array)
        .ok_or_else(|| "traceability_nodes must be an array".to_owned())?;
    let edges = root
        .get("traceability_edges")
        .and_then(Value::as_array)
        .ok_or_else(|| "traceability_edges must be an array".to_owned())?;

    let mut index_by_id: HashMap<String, usize> = HashMap::new();
    let mut node_ids = Vec::with_capacity(nodes.len());
    for (index, node) in nodes.iter().enumerate() {
        let id = node
            .get("node_id")
            .and_then(Value::as_str)
            .filter(|id| !id.trim().is_empty())
            .ok_or_else(|| format!("traceability node {index} has no valid node_id"))?;
        if index_by_id.insert(id.to_owned(), index).is_some() {
            return Err(format!("duplicate traceability node id {id:?}"));
        }
        node_ids.push(id.to_owned());
    }

    let mut graph = Graph::new(nodes.len(), true);
    for (index, edge) in edges.iter().enumerate() {
        let from_id = edge
            .get("from_id")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("traceability edge {index} has no from_id"))?;
        let to_id = edge
            .get("to_id")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("traceability edge {index} has no to_id"))?;
        if edge
            .get("relation")
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
        {
            return Err(format!("traceability edge {index} has no relation"));
        }
        let from = index_by_id
            .get(from_id)
            .copied()
            .ok_or_else(|| format!("traceability edge {index} has unknown source {from_id:?}"))?;
        let to = index_by_id
            .get(to_id)
            .copied()
            .ok_or_else(|| format!("traceability edge {index} has unknown target {to_id:?}"))?;
        graph.add_edge(from, to);
    }

    if graph.has_directed_cycle() {
        return Err("traceability graph contains a directed cycle".to_owned());
    }
    let source = index_by_id
        .get(source_lot_id)
        .copied()
        .ok_or_else(|| format!("source lot {source_lot_id:?} is not a traceability node"))?;

    let mut descendants: Vec<String> = graph
        .bfs(source)
        .into_iter()
        .filter(|&index| index != source)
        .map(|index| node_ids[index].clone())
        .collect();
    descendants.sort_unstable();
    Ok(descendants)
}

fn validate_traceability_graph(root: &Value, issues: &mut Vec<ValidationIssue>) {
    let Some(nodes) = root.get("traceability_nodes").and_then(Value::as_array) else {
        push(
            issues,
            "TRACEABILITY_NODES_MISSING",
            "traceability_nodes",
            "an explicit traceability node array is required",
        );
        return;
    };
    let Some(edges) = root.get("traceability_edges").and_then(Value::as_array) else {
        push(
            issues,
            "TRACEABILITY_EDGES_MISSING",
            "traceability_edges",
            "an explicit traceability edge array is required",
        );
        return;
    };
    if nodes.is_empty() {
        push(
            issues,
            "TRACEABILITY_NODES_EMPTY",
            "traceability_nodes",
            "traceability graph must contain nodes",
        );
        return;
    }

    let mut node_kind_by_id: HashMap<&str, &str> = HashMap::new();
    for node in nodes {
        if let (Some(id), Some(kind)) = (
            node.get("node_id").and_then(Value::as_str),
            node.get("node_kind").and_then(Value::as_str),
        ) {
            node_kind_by_id.insert(id, kind);
        } else {
            push(
                issues,
                "TRACEABILITY_NODE_INCOMPLETE",
                "traceability_nodes",
                "each node requires string node_id and node_kind",
            );
        }
    }

    if let Some(lots) = root.get("synthetic_material_lots").and_then(Value::as_array) {
        for lot in lots {
            let Some(id) = lot.get("lot_id").and_then(Value::as_str) else {
                continue;
            };
            if node_kind_by_id.get(id).copied() != Some("material_lot") {
                push(
                    issues,
                    "LOT_MISSING_TRACEABILITY_NODE",
                    format!("synthetic_material_lots[{id}].lot_id"),
                    "every declared material lot must have a material_lot node",
                );
            }
        }
    }
    if let Some(operations) = root.get("synthetic_process_graph").and_then(Value::as_array) {
        for operation in operations {
            let Some(id) = operation.get("operation_id").and_then(Value::as_str) else {
                continue;
            };
            if node_kind_by_id.get(id).copied() != Some("process_operation") {
                push(
                    issues,
                    "OPERATION_MISSING_TRACEABILITY_NODE",
                    format!("synthetic_process_graph[{id}].operation_id"),
                    "every declared process operation must have a process_operation node",
                );
            }
        }
    }

    let node_ids: HashSet<&str> = node_kind_by_id.keys().copied().collect();
    for (index, edge) in edges.iter().enumerate() {
        for field in ["from_id", "to_id"] {
            match edge.get(field).and_then(Value::as_str) {
                Some(id) if node_ids.contains(id) => {}
                Some(id) => push(
                    issues,
                    "TRACEABILITY_EDGE_UNKNOWN_NODE",
                    format!("traceability_edges[{index}].{field}"),
                    format!("edge references undeclared node {id:?}"),
                ),
                None => push(
                    issues,
                    "TRACEABILITY_EDGE_ENDPOINT_MISSING",
                    format!("traceability_edges[{index}].{field}"),
                    "edge endpoint must be a string node_id",
                ),
            }
        }
        if edge
            .get("relation")
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
        {
            push(
                issues,
                "TRACEABILITY_EDGE_RELATION_MISSING",
                format!("traceability_edges[{index}].relation"),
                "each traceability edge must declare a relation",
            );
        }
    }

    let known_lots = id_set(root, "synthetic_material_lots", "lot_id");
    let Some(queries) = root
        .get("expected_containment_queries")
        .and_then(Value::as_array)
    else {
        push(
            issues,
            "CONTAINMENT_QUERIES_MISSING",
            "expected_containment_queries",
            "known-answer lot-containment queries are required",
        );
        return;
    };
    if queries.is_empty() {
        push(
            issues,
            "CONTAINMENT_QUERIES_EMPTY",
            "expected_containment_queries",
            "at least one known-answer containment query is required",
        );
        return;
    }

    // Validate the whole graph once before comparing its known-answer queries.
    let probe = nodes[0]
        .get("node_id")
        .and_then(Value::as_str)
        .unwrap_or("");
    if let Err(error) = traceability_descendants(root, probe) {
        push(
            issues,
            "TRACEABILITY_GRAPH_INVALID",
            "traceability_edges",
            error,
        );
        return;
    }

    for (index, query) in queries.iter().enumerate() {
        let prefix = format!("expected_containment_queries[{index}]");
        let Some(source_lot_id) = query.get("source_lot_id").and_then(Value::as_str) else {
            push(
                issues,
                "CONTAINMENT_SOURCE_MISSING",
                format!("{prefix}.source_lot_id"),
                "query source_lot_id must be a string",
            );
            continue;
        };
        if !known_lots.contains(source_lot_id) {
            push(
                issues,
                "CONTAINMENT_SOURCE_UNKNOWN",
                format!("{prefix}.source_lot_id"),
                format!("source lot {source_lot_id:?} is not declared"),
            );
            continue;
        }
        if node_kind_by_id.get(source_lot_id).copied() != Some("material_lot") {
            push(
                issues,
                "CONTAINMENT_SOURCE_NOT_LOT",
                format!("{prefix}.source_lot_id"),
                "containment query must begin at a material_lot node",
            );
            continue;
        }

        let Some(expected_values) = query.get("expected_descendants").and_then(Value::as_array)
        else {
            push(
                issues,
                "CONTAINMENT_EXPECTED_DESCENDANTS_MISSING",
                format!("{prefix}.expected_descendants"),
                "expected_descendants must be an array of node ids",
            );
            continue;
        };
        let mut expected = Vec::with_capacity(expected_values.len());
        let mut valid_expected = true;
        let mut seen = HashSet::new();
        for (desc_index, value) in expected_values.iter().enumerate() {
            match value.as_str() {
                Some(id) if node_ids.contains(id) => {
                    if id == source_lot_id || !seen.insert(id.to_owned()) {
                        push(
                            issues,
                            "CONTAINMENT_EXPECTED_SET_INVALID",
                            format!("{prefix}.expected_descendants[{desc_index}]"),
                            "expected set must exclude its source and contain unique node ids",
                        );
                        valid_expected = false;
                    }
                    expected.push(id.to_owned());
                }
                Some(id) => {
                    push(
                        issues,
                        "CONTAINMENT_EXPECTED_NODE_UNKNOWN",
                        format!("{prefix}.expected_descendants[{desc_index}]"),
                        format!("expected descendant {id:?} is not declared"),
                    );
                    valid_expected = false;
                }
                None => {
                    push(
                        issues,
                        "CONTAINMENT_EXPECTED_NODE_INVALID",
                        format!("{prefix}.expected_descendants[{desc_index}]"),
                        "expected descendant must be a string node_id",
                    );
                    valid_expected = false;
                }
            }
        }
        if !valid_expected {
            continue;
        }
        expected.sort_unstable();

        match traceability_descendants(root, source_lot_id) {
            Ok(actual) if actual == expected => {}
            Ok(actual) => push(
                issues,
                "CONTAINMENT_KNOWN_ANSWER_MISMATCH",
                format!("{prefix}.expected_descendants"),
                format!("expected {} descendants, computed {}", expected.len(), actual.len()),
            ),
            Err(error) => push(
                issues,
                "CONTAINMENT_QUERY_NOT_EVALUABLE",
                format!("{prefix}.source_lot_id"),
                error,
            ),
        }
    }
}

fn validate_synthetic_claim_boundary(root: &Value, issues: &mut Vec<ValidationIssue>) {
    let Some(design) = root.get("design_intent") else {
        push(
            issues,
            "DESIGN_INTENT_MISSING",
            "design_intent",
            "synthetic claim boundary requires a design-intent object",
        );
        return;
    };

    for field in [
        "real_materials_selected",
        "physical_fixture_exists",
        "factory_or_supplier_verified",
    ] {
        match design.get(field).and_then(Value::as_bool) {
            Some(false) => {}
            Some(true) => push(
                issues,
                "SYNTHETIC_CLAIM_CONTRADICTION",
                format!("design_intent.{field}"),
                "must remain false in a synthetic-only fixture",
            ),
            None => push(
                issues,
                "SYNTHETIC_CLAIM_BOUNDARY_MISSING",
                format!("design_intent.{field}"),
                "explicit false value is required for synthetic-only evidence",
            ),
        }
    }

    match root
        .pointer("/hypothetical_factory/status")
        .and_then(Value::as_str)
    {
        Some("concept_only") => {}
        _ => push(
            issues,
            "FACTORY_STATUS_NOT_SYNTHETIC",
            "hypothetical_factory.status",
            "factory must remain concept_only in this fixture",
        ),
    }
}

fn validate_supplier_lot_references(root: &Value, issues: &mut Vec<ValidationIssue>) {
    let suppliers = id_set(root, "synthetic_suppliers", "supplier_id");
    let lots = id_set(root, "synthetic_material_lots", "lot_id");

    if let Some(items) = root
        .get("synthetic_material_lots")
        .and_then(Value::as_array)
    {
        for (index, lot) in items.iter().enumerate() {
            let path = path_for_item("synthetic_material_lots", index, "supplier_id");
            match lot.get("supplier_id").and_then(Value::as_str) {
                Some(id) if suppliers.contains(id) => {}
                Some(id) => push(
                    issues,
                    "UNKNOWN_SUPPLIER_REFERENCE",
                    path,
                    format!("supplier {id:?} is not declared"),
                ),
                None => push(
                    issues,
                    "MISSING_SUPPLIER_REFERENCE",
                    path,
                    "material lot must reference a supplier",
                ),
            }
        }
    }

    if let Some(items) = root
        .get("synthetic_process_graph")
        .and_then(Value::as_array)
    {
        for (index, operation) in items.iter().enumerate() {
            if let Some(input_lots) = operation
                .get("input_lots")
                .and_then(Value::as_array)
            {
                for (lot_index, lot_ref) in input_lots.iter().enumerate() {
                    let path = format!(
                        "synthetic_process_graph[{index}].input_lots[{lot_index}]"
                    );
                    match lot_ref.as_str() {
                        Some(id) if lots.contains(id) => {}
                        Some(id) => push(
                            issues,
                            "UNKNOWN_MATERIAL_LOT_REFERENCE",
                            path,
                            format!("material lot {id:?} is not declared"),
                        ),
                        None => push(
                            issues,
                            "INVALID_MATERIAL_LOT_REFERENCE",
                            path,
                            "input lot reference must be a string",
                        ),
                    }
                }
            } else {
                push(
                    issues,
                    "MISSING_INPUT_LOT_ARRAY",
                    format!("synthetic_process_graph[{index}].input_lots"),
                    "input_lots must be an explicit array; use [] if the operation consumes no new lot directly",
                );
            }
        }
    }
}

fn validate_resources_and_process_graph(root: &Value, issues: &mut Vec<ValidationIssue>) {
    let Some(resources) = root
        .get("synthetic_resources")
        .and_then(Value::as_array)
    else {
        push(
            issues,
            "MISSING_RESOURCES",
            "synthetic_resources",
            "resource array is required",
        );
        return;
    };
    let Some(operations) = root
        .get("synthetic_process_graph")
        .and_then(Value::as_array)
    else {
        push(
            issues,
            "MISSING_PROCESS_GRAPH",
            "synthetic_process_graph",
            "process graph array is required",
        );
        return;
    };

    let mut resource_by_id: HashMap<&str, &Value> = HashMap::new();
    for resource in resources {
        if let Some(id) = resource.get("resource_id").and_then(Value::as_str) {
            resource_by_id.insert(id, resource);
        }
        if resource
            .get("capability_profile")
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
        {
            push(
                issues,
                "RESOURCE_PROFILE_MISSING",
                "synthetic_resources",
                "each resource requires a non-empty capability_profile",
            );
        }
        if resource.get("qualification").and_then(Value::as_str) != Some("synthetic_only") {
            push(
                issues,
                "RESOURCE_QUALIFICATION_NOT_SYNTHETIC",
                "synthetic_resources",
                "fixture resources must declare synthetic_only qualification",
            );
        }
    }

    let mut operation_index_by_id: HashMap<String, usize> = HashMap::new();
    for (index, operation) in operations.iter().enumerate() {
        if let Some(id) = operation.get("operation_id").and_then(Value::as_str) {
            operation_index_by_id.insert(id.to_owned(), index);
        }
    }

    let mut graph = Graph::new(operations.len(), true);

    for (index, operation) in operations.iter().enumerate() {
        let prefix = format!("synthetic_process_graph[{index}]");

        match operation.get("duration_seconds").and_then(Value::as_u64) {
            Some(duration) if duration > 0 => {}
            _ => push(
                issues,
                "INVALID_OPERATION_DURATION",
                format!("{prefix}.duration_seconds"),
                "duration_seconds must be a positive integer",
            ),
        }

        if operation
            .get("output_state")
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
        {
            push(
                issues,
                "OUTPUT_STATE_MISSING",
                format!("{prefix}.output_state"),
                "each operation must declare its expected output-state label",
            );
        }

        let resource_id = operation.get("required_resource").and_then(Value::as_str);
        let resource = resource_id.and_then(|id| resource_by_id.get(id).copied());
        match (resource_id, resource) {
            (Some(_), Some(resource)) => {
                if resource.get("available").and_then(Value::as_bool) != Some(true) {
                    push(
                        issues,
                        "RESOURCE_UNAVAILABLE",
                        format!("{prefix}.required_resource"),
                        "synthetic operation references a resource not marked available",
                    );
                }
                let qualified_skills = string_list(resource, "qualified_skills");
                let required_skills = string_list(operation, "required_skills");
                match (qualified_skills, required_skills) {
                    (Some(available), Some(required)) => {
                        for skill in required {
                            if !available.contains(&skill) {
                                push(
                                    issues,
                                    "RESOURCE_SKILL_MISMATCH",
                                    format!("{prefix}.required_skills"),
                                    format!("resource {:?} does not declare skill {:?}", resource_id.unwrap_or("<missing>"), skill),
                                );
                            }
                        }
                    }
                    _ => push(
                        issues,
                        "SKILL_LIST_MISSING",
                        format!("{prefix}.required_skills"),
                        "operation and referenced resource must provide string arrays of skills",
                    ),
                }
            }
            (Some(id), None) => push(
                issues,
                "UNKNOWN_RESOURCE_REFERENCE",
                format!("{prefix}.required_resource"),
                format!("resource {id:?} is not declared"),
            ),
            (None, _) => push(
                issues,
                "MISSING_RESOURCE_REFERENCE",
                format!("{prefix}.required_resource"),
                "every process operation must name a resource",
            ),
        }

        match string_list(operation, "predecessors") {
            Some(predecessors) => {
                for predecessor in predecessors {
                    match operation_index_by_id.get(&predecessor) {
                        Some(&predecessor_index) => {
                            graph.add_edge(predecessor_index, index);
                        }
                        None => push(
                            issues,
                            "UNKNOWN_PREDECESSOR",
                            format!("{prefix}.predecessors"),
                            format!("predecessor {predecessor:?} is not declared"),
                        ),
                    }
                }
            }
            None => push(
                issues,
                "PREDECESSOR_LIST_MISSING",
                format!("{prefix}.predecessors"),
                "predecessors must be an explicit array; use [] for source operations",
            ),
        }

        if operation.get("process").and_then(Value::as_str) == Some("final_inspection")
            && operation.get("required_hold_point").and_then(Value::as_bool) != Some(true)
        {
            push(
                issues,
                "FINAL_INSPECTION_NOT_HELD",
                format!("{prefix}.required_hold_point"),
                "final inspection must have an explicit required hold point",
            );
        }
    }

    if graph.has_directed_cycle() {
        push(
            issues,
            "PROCESS_GRAPH_CYCLE",
            "synthetic_process_graph",
            "process predecessor graph must be a directed acyclic graph",
        );
    }
}

fn validate_final_inspection_gate(root: &Value, issues: &mut Vec<ValidationIssue>) {
    let Some(operations) = root
        .get("synthetic_process_graph")
        .and_then(Value::as_array)
    else {
        return;
    };

    let final_inspections: Vec<_> = operations
        .iter()
        .filter(|op| op.get("process").and_then(Value::as_str) == Some("final_inspection"))
        .collect();

    if final_inspections.len() != 1 {
        push(
            issues,
            "FINAL_INSPECTION_COUNT",
            "synthetic_process_graph",
            format!(
                "expected exactly one final_inspection operation, found {}",
                final_inspections.len()
            ),
        );
    }

    for (index, operation) in operations.iter().enumerate() {
        if operation.get("process").and_then(Value::as_str) != Some("final_inspection") {
            continue;
        }
        let Some(skills) = string_list(operation, "required_skills") else {
            push(
                issues,
                "FINAL_INSPECTION_SKILLS_MISSING",
                format!("synthetic_process_graph[{index}].required_skills"),
                "final inspection requires an explicit skill list",
            );
            continue;
        };
        for required in ["inspection", "measurement"] {
            if !skills.iter().any(|skill| skill == required) {
                push(
                    issues,
                    "FINAL_INSPECTION_SKILL_MISSING",
                    format!("synthetic_process_graph[{index}].required_skills"),
                    format!("final inspection must require {required:?}"),
                );
            }
        }
    }
}

fn validate_negative_controls(root: &Value, issues: &mut Vec<ValidationIssue>) {
    let Some(controls) = root.get("negative_controls").and_then(Value::as_array) else {
        return;
    };
    if controls.is_empty() {
        push(
            issues,
            "NEGATIVE_CONTROLS_EMPTY",
            "negative_controls",
            "at least one adversarial negative control is required",
        );
        return;
    }
    for (index, control) in controls.iter().enumerate() {
        for field in ["scenario", "expected_disposition", "invariant"] {
            if control
                .get(field)
                .and_then(Value::as_str)
                .is_none_or(str::is_empty)
            {
                push(
                    issues,
                    "NEGATIVE_CONTROL_INCOMPLETE",
                    format!("negative_controls[{index}].{field}"),
                    "each negative control must specify scenario, expected disposition and invariant",
                );
            }
        }
    }
}

fn validate_disruptions(root: &Value, issues: &mut Vec<ValidationIssue>) {
    let Some(disruptions) = root.get("disruptions").and_then(Value::as_array) else {
        return;
    };
    if disruptions.is_empty() {
        push(
            issues,
            "DISRUPTION_SET_EMPTY",
            "disruptions",
            "at least one supply-chain or factory disruption scenario is required",
        );
    }
    for (index, disruption) in disruptions.iter().enumerate() {
        if disruption
            .get("event")
            .and_then(Value::as_str)
            .is_none_or(str::is_empty)
        {
            push(
                issues,
                "DISRUPTION_EVENT_MISSING",
                format!("disruptions[{index}].event"),
                "disruption must declare an event",
            );
        }
        if !disruption
            .get("required_outputs")
            .and_then(Value::as_array)
            .is_some_and(|outputs| !outputs.is_empty() && outputs.iter().all(Value::is_string))
        {
            push(
                issues,
                "DISRUPTION_OUTPUTS_MISSING",
                format!("disruptions[{index}].required_outputs"),
                "disruption must declare non-empty required_outputs string array",
            );
        }
    }
}

fn validate_required_target_ranges(root: &Value, issues: &mut Vec<ValidationIssue>) {
    let Some(targets) = root.get("proposed_targets").and_then(Value::as_array) else {
        return;
    };

    for (index, target) in targets.iter().enumerate() {
        let path = format!("proposed_targets[{index}]");
        let id = target.get("id").and_then(Value::as_str).unwrap_or("");
        if target.get("status").and_then(Value::as_str).is_none_or(str::is_empty) {
            push(
                issues,
                "TARGET_STATUS_MISSING",
                format!("{path}.status"),
                "proposed target must state its evidence/status level",
            );
        }
        if target
            .get("target_value")
            .and_then(Value::as_f64)
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            push(
                issues,
                "TARGET_VALUE_INVALID",
                format!("{path}.target_value"),
                "numeric target values must be finite and non-negative",
            );
        }
        if id == "COST-01" {
            match target.get("target_value").and_then(Value::as_f64) {
                Some(value) if value > 0.0 && value < 1.0 => {}
                _ => push(
                    issues,
                    "COST_TARGET_INVALID",
                    format!("{path}.target_value"),
                    "delivered-cost target must be a ratio strictly between 0 and 1",
                ),
            }
        }
    }
}

fn validate_cost_scenarios(root: &Value, issues: &mut Vec<ValidationIssue>) {
    let Some(scenarios) = root
        .get("synthetic_cost_scenarios")
        .and_then(Value::as_array)
    else {
        push(
            issues,
            "COST_SCENARIOS_MISSING",
            "synthetic_cost_scenarios",
            "an explicit synthetic cost-scenario array is required",
        );
        return;
    };
    if scenarios.len() < 2 {
        push(
            issues,
            "COST_SCENARIOS_INSUFFICIENT",
            "synthetic_cost_scenarios",
            "at least a reference and candidate cost scenario are required",
        );
    }
    for scenario in scenarios {
        if let Some(id) = scenario.get("scenario_id").and_then(Value::as_str) {
            if let Err(error) = accounting::calculate_cost_scenario(root, id) {
                push(
                    issues,
                    "COST_SCENARIO_INVALID",
                    format!("synthetic_cost_scenarios[{id}]"),
                    error,
                );
            }
        }
    }
    if let Err(error) = accounting::calculate_cost_comparison(root) {
        push(
            issues,
            "COST_COMPARISON_INVALID",
            "synthetic_cost_comparison",
            error,
        );
    }
}

fn validate_material_balance_cases(root: &Value, issues: &mut Vec<ValidationIssue>) {
    let Some(cases) = root
        .get("synthetic_material_balance_cases")
        .and_then(Value::as_array)
    else {
        push(
            issues,
            "MATERIAL_BALANCE_CASES_MISSING",
            "synthetic_material_balance_cases",
            "an explicit material-balance known-answer array is required",
        );
        return;
    };
    if cases.is_empty() {
        push(
            issues,
            "MATERIAL_BALANCE_CASES_EMPTY",
            "synthetic_material_balance_cases",
            "at least one material-balance known-answer case is required",
        );
        return;
    }
    for case in cases {
        if let Some(id) = case.get("case_id").and_then(Value::as_str) {
            if let Err(error) = accounting::calculate_material_balance(root, id) {
                push(
                    issues,
                    "MATERIAL_BALANCE_INVALID",
                    format!("synthetic_material_balance_cases[{id}]"),
                    error,
                );
            }
        }
    }
}

fn validate_verification_status(root: &Value, issues: &mut Vec<ValidationIssue>) {
    let Some(status) = root.get("verification_status") else {
        push(
            issues,
            "VERIFICATION_STATUS_MISSING",
            "verification_status",
            "explicit verification status object is required",
        );
        return;
    };

    for (field, expected) in [
        ("physical_measurements", "none"),
        ("facility_commissioning", "not_started"),
        ("human_evaluation", "none"),
        ("independent_replication", "none"),
        ("process_plan_execution", "not_run"),
        ("independent_qualifier", "implemented_not_executed"),
    ] {
        if status.get(field).and_then(Value::as_str) != Some(expected) {
            push(
                issues,
                "VERIFICATION_STATUS_CONTRADICTION",
                format!("verification_status.{field}"),
                format!("synthetic-only fixture must retain status {expected:?}"),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Value {
        serde_json::from_str(EMBEDDED_FIXTURE).expect("embedded fixture must parse")
    }

    fn has_code(report: &ValidationReport, expected: &str) -> bool {
        report.issues.iter().any(|issue| issue.code == expected)
    }

    #[test]
    fn embedded_fixture_passes_structural_checks_only() {
        let report = validate_fixture(&fixture());
        assert!(
            report.is_valid(),
            "unexpected findings: {:#?}",
            report.issues
        );
    }

    #[test]
    fn unknown_predecessor_is_blocked() {
        let mut value = fixture();
        value["synthetic_process_graph"][0]["predecessors"] =
            serde_json::json!(["OP-DOES-NOT-EXIST"]);
        assert!(has_code(&validate_fixture(&value), "UNKNOWN_PREDECESSOR"));
    }

    #[test]
    fn dependency_cycle_is_blocked() {
        let mut value = fixture();
        value["synthetic_process_graph"][0]["predecessors"] =
            serde_json::json!(["OP-70-PACK"]);
        assert!(has_code(&validate_fixture(&value), "PROCESS_GRAPH_CYCLE"));
    }

    #[test]
    fn duplicate_operation_identity_is_blocked() {
        let mut value = fixture();
        value["synthetic_process_graph"][1]["operation_id"] =
            serde_json::json!("OP-10-RECEIVE");
        assert!(has_code(&validate_fixture(&value), "DUPLICATE_ID"));
    }

    #[test]
    fn unknown_resource_is_blocked() {
        let mut value = fixture();
        value["synthetic_process_graph"][2]["required_resource"] =
            serde_json::json!("SYN-NO-SUCH-RESOURCE");
        assert!(has_code(&validate_fixture(&value), "UNKNOWN_RESOURCE_REFERENCE"));
    }

    #[test]
    fn missing_resource_skill_is_blocked() {
        let mut value = fixture();
        value["synthetic_process_graph"][2]["required_skills"] =
            serde_json::json!(["textile_cutting", "unqualified_special_process"]);
        assert!(has_code(&validate_fixture(&value), "RESOURCE_SKILL_MISMATCH"));
    }

    #[test]
    fn unknown_material_lot_is_blocked() {
        let mut value = fixture();
        value["synthetic_process_graph"][2]["input_lots"] =
            serde_json::json!(["SYN-LOT-NOT-DECLARED"]);
        assert!(has_code(
            &validate_fixture(&value),
            "UNKNOWN_MATERIAL_LOT_REFERENCE"
        ));
    }

    #[test]
    fn unknown_supplier_on_lot_is_blocked() {
        let mut value = fixture();
        value["synthetic_material_lots"][0]["supplier_id"] =
            serde_json::json!("SYN-SUP-NOT-DECLARED");
        assert!(has_code(&validate_fixture(&value), "UNKNOWN_SUPPLIER_REFERENCE"));
    }

    #[test]
    fn contradictory_physical_claim_is_blocked() {
        let mut value = fixture();
        value["design_intent"]["physical_fixture_exists"] = serde_json::json!(true);
        assert!(has_code(
            &validate_fixture(&value),
            "SYNTHETIC_CLAIM_CONTRADICTION"
        ));
    }

    #[test]
    fn final_inspection_without_hold_point_is_blocked() {
        let mut value = fixture();
        value["synthetic_process_graph"][5]["required_hold_point"] =
            serde_json::json!(false);
        assert!(has_code(&validate_fixture(&value), "FINAL_INSPECTION_NOT_HELD"));
    }

    #[test]
    fn missing_final_inspection_skill_is_blocked() {
        let mut value = fixture();
        value["synthetic_process_graph"][5]["required_skills"] =
            serde_json::json!(["packaging"]);
        let report = validate_fixture(&value);
        assert!(has_code(&report, "FINAL_INSPECTION_SKILL_MISSING"));
        assert!(has_code(&report, "RESOURCE_SKILL_MISMATCH"));
    }

    #[test]
    fn invalid_cost_ratio_is_blocked() {
        let mut value = fixture();
        value["proposed_targets"][0]["target_value"] = serde_json::json!(1.2);
        assert!(has_code(&validate_fixture(&value), "COST_TARGET_INVALID"));
    }

    #[test]
    fn physical_measurements_cannot_be_claimed_by_synthetic_fixture() {
        let mut value = fixture();
        value["verification_status"]["physical_measurements"] =
            serde_json::json!("measured");
        assert!(has_code(
            &validate_fixture(&value),
            "VERIFICATION_STATUS_CONTRADICTION"
        ));
    }

    #[test]
    fn traceability_containment_known_answers_match() {
        let value = fixture();
        let report = validate_fixture(&value);
        assert!(report.is_valid(), "unexpected findings: {:#?}", report.issues);

        let descendants = traceability_descendants(&value, "SYN-LOT-FABRIC-001").unwrap();
        assert!(descendants.contains(&"SYN-UNIT-PACKED-A".to_owned()));
        assert!(descendants.contains(&"SYN-UNIT-PACKED-B".to_owned()));
        assert!(!descendants.contains(&"SYN-UNIT-PACKED-CONTROL".to_owned()));
    }

    #[test]
    fn unrelated_lot_does_not_recall_household_product_units() {
        let value = fixture();
        let descendants =
            traceability_descendants(&value, "SYN-LOT-UNRELATED-099").unwrap();
        assert_eq!(
            descendants,
            vec![
                "SYN-UNRELATED-ARTICLE-CONTROL".to_owned(),
                "SYN-UNRELATED-ASSEMBLY-CONTROL".to_owned(),
                "SYN-UNRELATED-PACKED-CONTROL".to_owned(),
            ]
        );
    }

    #[test]
    fn unknown_traceability_endpoint_is_blocked() {
        let mut value = fixture();
        value["traceability_edges"][0]["to_id"] =
            serde_json::json!("SYN-UNDECLARED-NODE");
        assert!(has_code(
            &validate_fixture(&value),
            "TRACEABILITY_EDGE_UNKNOWN_NODE"
        ));
    }

    #[test]
    fn traceability_cycle_is_blocked() {
        let mut value = fixture();
        value["traceability_edges"].as_array_mut().unwrap().push(serde_json::json!({
            "edge_id": "TR-CYCLE-CONTROL",
            "from_id": "SYN-UNIT-PACKED-A",
            "to_id": "SYN-LOT-FABRIC-001",
            "relation": "invalid_cycle_control"
        }));
        assert!(has_code(
            &validate_fixture(&value),
            "TRACEABILITY_GRAPH_INVALID"
        ));
    }

    #[test]
    fn omitted_expected_containment_descendant_is_blocked() {
        let mut value = fixture();
        value["expected_containment_queries"][0]["expected_descendants"]
            .as_array_mut()
            .unwrap()
            .pop();
        assert!(has_code(
            &validate_fixture(&value),
            "CONTAINMENT_KNOWN_ANSWER_MISMATCH"
        ));
    }

    #[test]
    fn synthetic_cost_scenarios_match_frozen_totals() {
        let value = fixture();
        let reference = accounting::calculate_cost_scenario(
            &value,
            "COST-REF-ARITHMETIC-01",
        )
        .unwrap();
        let candidate = accounting::calculate_cost_scenario(
            &value,
            "COST-CANDIDATE-ARITHMETIC-01",
        )
        .unwrap();
        assert_eq!(reference.total_cost_minor_units, 250_000);
        assert_eq!(candidate.total_cost_minor_units, 210_000);
        assert_eq!(reference.accepted_units, 10);
        assert_eq!(candidate.accepted_units, 10);
    }

    #[test]
    fn synthetic_cost_comparison_is_exact_and_does_not_meet_80_percent_target() {
        let value = fixture();
        let report = accounting::calculate_cost_comparison(&value).unwrap();
        assert_eq!(report.ratio_basis_points_floor, 8_400);
        assert_eq!(report.target_ratio_basis_points, 8_000);
        assert!(!report.target_met);
    }

    #[test]
    fn missing_cost_category_fails_closed() {
        let mut value = fixture();
        value["synthetic_cost_scenarios"][1]["cost_components"]
            .as_array_mut()
            .unwrap()
            .pop();
        assert!(has_code(&validate_fixture(&value), "COST_SCENARIO_INVALID"));
        assert!(accounting::calculate_cost_scenario(
            &value,
            "COST-CANDIDATE-ARITHMETIC-01"
        ).is_err());
    }

    #[test]
    fn zero_accepted_units_is_rejected() {
        let mut value = fixture();
        value["synthetic_cost_scenarios"][0]["accepted_units"] = serde_json::json!(0);
        assert!(has_code(&validate_fixture(&value), "COST_SCENARIO_INVALID"));
    }

    #[test]
    fn mixed_currency_cost_comparison_is_rejected() {
        let mut value = fixture();
        value["synthetic_cost_scenarios"][1]["currency"] =
            serde_json::json!("ANOTHER_SYNTHETIC_CURRENCY");
        assert!(has_code(&validate_fixture(&value), "COST_COMPARISON_INVALID"));
    }

    #[test]
    fn cost_sum_overflow_is_rejected() {
        let mut value = fixture();
        value["synthetic_cost_scenarios"][1]["cost_components"][0]["amount_minor_units"] =
            serde_json::json!(u64::MAX);
        assert!(accounting::calculate_cost_scenario(
            &value,
            "COST-CANDIDATE-ARITHMETIC-01"
        ).is_err());
    }

    #[test]
    fn modified_cost_known_answer_is_rejected() {
        let mut value = fixture();
        value["synthetic_cost_scenarios"][1]["expected_total_minor_units"] =
            serde_json::json!(200_000);
        assert!(has_code(&validate_fixture(&value), "COST_SCENARIO_INVALID"));
    }

    #[test]
    fn material_balance_known_answer_closes_exactly() {
        let value = fixture();
        let report = accounting::calculate_material_balance(
            &value,
            "MASS-BALANCE-ARITHMETIC-01",
        )
        .unwrap();
        assert_eq!(report.input_mass_mg, 10_000_000);
        assert_eq!(report.accounted_mass_mg, 10_000_000);
        assert_eq!(report.unaccounted_mass_mg, 0);
        assert_eq!(report.closure_basis_points_floor, 10_000);
        assert_eq!(report.accepted_yield_basis_points_floor, 8_500);
    }

    #[test]
    fn missing_material_mass_is_blocked() {
        let mut value = fixture();
        value["synthetic_material_balance_cases"][0]["scrap_mass_mg"] =
            serde_json::json!(500_000);
        assert!(has_code(&validate_fixture(&value), "MATERIAL_BALANCE_INVALID"));
    }

    #[test]
    fn reported_mass_residual_must_match_computed_residual() {
        let mut value = fixture();
        value["synthetic_material_balance_cases"][0]["reported_unaccounted_mass_mg"] =
            serde_json::json!(100);
        assert!(has_code(&validate_fixture(&value), "MATERIAL_BALANCE_INVALID"));
    }

    #[test]
    fn double_counted_material_mass_is_blocked() {
        let mut value = fixture();
        value["synthetic_material_balance_cases"][0]["accepted_output_mass_mg"] =
            serde_json::json!(9_000_000);
        assert!(has_code(&validate_fixture(&value), "MATERIAL_BALANCE_INVALID"));
    }

    #[test]
    fn zero_input_mass_is_blocked() {
        let mut value = fixture();
        value["synthetic_material_balance_cases"][0]["input_mass_mg"] =
            serde_json::json!(0);
        assert!(has_code(&validate_fixture(&value), "MATERIAL_BALANCE_INVALID"));
    }

    #[test]
    fn report_order_is_deterministic() {
        let mut value = fixture();
        value["synthetic_process_graph"][0]["predecessors"] =
            serde_json::json!(["UNKNOWN-A", "UNKNOWN-B"]);
        let first = validate_fixture(&value);
        let second = validate_fixture(&value);
        assert_eq!(first, second);
    }
}
