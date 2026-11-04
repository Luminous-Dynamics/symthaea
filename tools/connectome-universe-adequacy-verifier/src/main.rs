#![forbid(unsafe_code)]

//! Independent semantic verifier for query-scoped structural-universe adequacy.
//!
//! Inputs are a report plus two evaluator-side structure manifests. The verifier
//! hashes the exact sidecar bytes, recomputes canonical directed-graph identities,
//! and independently recomputes all adequacy counters and dispositions.
//!
//! This tool does not prove that the reference challenge set is biologically
//! exhaustive, authenticate a freeze signature, or establish causal truth.

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::process;

const REPORT_VERSION: &str = "0.1.0";
const SIDECAR_VERSION: &str = "0.1.0";
const CANONICALIZER_PROTOCOL: &str = "CONNECTOME-STRUCTURAL-GRAPH-CANONICAL-V1";
const EVALUATOR_PROTOCOL_SHA256: &str =
    "eb780dac0d14b867ef8b0b42773c911e8e01890cca3ec4709b870e0989f8b085";
const RULES_PROTOCOL_SHA256: &str =
    "ee9caeef8c104c10a170de7eb33aba058c8d33c443d7140b9ebac028aa818b11";
const CANONICALIZER_PROTOCOL_SHA256: &str =
    "604a03409b21c5c7af1ec41c0bf8466d6b999103f95a429c5d9add6083e3e05c";
const REPORT_SCHEMA_SHA256: &str =
    "7640424b4afff287f9d11d1d4a6a8474f533b1b1a90face915c56bd972f1afbc";
const STRUCTURE_SCHEMA_SHA256: &str =
    "16f2b9eac1f4a0145289d6100f8c902e5b470ece45a623d032e874ce5bd50476";

#[derive(Debug, Clone)]
struct CandidateWorld {
    id: String,
    family: String,
    state: String,
    structural_sha256: String,
}

#[derive(Debug, Clone)]
struct ReferenceWorld {
    id: String,
    family: String,
    compatibility: String,
    state: Option<String>,
    structural_sha256: String,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct FamilyCounts {
    compatible_reference_count: u64,
    incompatible_reference_count: u64,
    indeterminate_reference_count: u64,
    compatible_estimable_reference_count: u64,
    compatible_query_not_estimable_reference_count: u64,
    structurally_included_count: u64,
    query_flipping_reference_count: u64,
    omitted_query_flipping_count: u64,
    false_invariance_count: u64,
    false_ambiguity_count: u64,
    candidate_not_estimable_count: u64,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Summary {
    candidate_world_count: u64,
    reference_world_count: u64,
    compatible_reference_count: u64,
    incompatible_reference_count: u64,
    indeterminate_reference_count: u64,
    compatible_estimable_reference_count: u64,
    compatible_query_not_estimable_reference_count: u64,
    structurally_included_count: u64,
    query_state_covered_count: u64,
    query_flipping_reference_count: u64,
    omitted_query_flipping_count: u64,
    false_invariance_count: u64,
    false_ambiguity_count: u64,
    candidate_duplicate_structure_group_count: u64,
    reference_duplicate_structure_group_count: u64,
    candidate_not_estimable_count: u64,
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn object<'a>(value: &'a Value, context: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{context}: expected JSON object"))
}

fn array<'a>(value: &'a Value, context: &str) -> Result<&'a Vec<Value>, String> {
    value
        .as_array()
        .ok_or_else(|| format!("{context}: expected JSON array"))
}

fn string_field(value: &Value, key: &str, context: &str) -> Result<String, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| format!("{context}.{key}: expected non-empty string"))
}

fn digest_field(value: &Value, key: &str, context: &str) -> Result<String, String> {
    let digest = string_field(value, key, context)?;
    if !valid_sha256(&digest) {
        return Err(format!("{context}.{key}: expected lowercase SHA-256 hex"));
    }
    Ok(digest)
}

fn bool_field(value: &Value, key: &str, context: &str) -> Result<bool, String> {
    value
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("{context}.{key}: expected boolean"))
}

fn u64_field(value: &Value, key: &str, context: &str) -> Result<u64, String> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{context}.{key}: expected non-negative integer"))
}

fn require_keys(
    map: &Map<String, Value>,
    expected: &[&str],
    context: &str,
) -> Result<(), String> {
    if map.len() != expected.len() || expected.iter().any(|key| !map.contains_key(*key)) {
        return Err(format!(
            "{context}: expected exactly these keys: {}",
            expected.join(", ")
        ));
    }
    Ok(())
}

fn determinate_state(value: &Value, context: &str) -> Result<Option<String>, String> {
    if value.is_null() {
        return Ok(None);
    }
    let state = value
        .as_str()
        .ok_or_else(|| format!("{context}: expected decision string or null"))?;
    match state {
        "BELOW_THRESHOLD" | "AT_OR_ABOVE_THRESHOLD" => Ok(Some(state.to_owned())),
        "NOT_ESTIMABLE" => Ok(None),
        _ => Err(format!("{context}: unknown decision state {state}")),
    }
}

/// Canonicalizes a directed graph while preserving node/edge identities and attributes.
///
/// Input array ordering is ignored. JSON object keys use serde_json's default ordered-map
/// representation. This is a versioned internal encoding, not a claim of RFC 8785 compliance.
fn canonical_structure_sha256(structure: &Value, context: &str) -> Result<String, String> {
    let graph = object(structure, context)?;
    require_keys(graph, &["nodes", "edges"], context)?;

    let node_values = array(
        graph.get("nodes").ok_or_else(|| format!("{context}.nodes missing"))?,
        &format!("{context}.nodes"),
    )?;
    let edge_values = array(
        graph.get("edges").ok_or_else(|| format!("{context}.edges missing"))?,
        &format!("{context}.edges"),
    )?;

    let mut node_ids = BTreeSet::new();
    let mut normalized_nodes: Vec<(String, Value)> = Vec::with_capacity(node_values.len());

    for (index, node) in node_values.iter().enumerate() {
        let node_context = format!("{context}.nodes[{index}]");
        let node_object = object(node, &node_context)?;
        require_keys(node_object, &["node_id", "attributes"], &node_context)?;
        let node_id = string_field(node, "node_id", &node_context)?;
        if !node_ids.insert(node_id.clone()) {
            return Err(format!("{node_context}: duplicate node_id {node_id}"));
        }
        let attributes = node
            .get("attributes")
            .filter(|attributes| attributes.is_object())
            .ok_or_else(|| format!("{node_context}.attributes: expected object"))?;
        normalized_nodes.push((
            node_id.clone(),
            json!({
                "node_id": node_id,
                "attributes": attributes.clone()
            }),
        ));
    }

    let mut edge_ids = BTreeSet::new();
    let mut normalized_edges: Vec<(String, Value)> = Vec::with_capacity(edge_values.len());

    for (index, edge) in edge_values.iter().enumerate() {
        let edge_context = format!("{context}.edges[{index}]");
        let edge_object = object(edge, &edge_context)?;
        require_keys(
            edge_object,
            &["edge_id", "source", "target", "attributes"],
            &edge_context,
        )?;
        let edge_id = string_field(edge, "edge_id", &edge_context)?;
        let source = string_field(edge, "source", &edge_context)?;
        let target = string_field(edge, "target", &edge_context)?;
        if !edge_ids.insert(edge_id.clone()) {
            return Err(format!("{edge_context}: duplicate edge_id {edge_id}"));
        }
        if !node_ids.contains(&source) || !node_ids.contains(&target) {
            return Err(format!(
                "{edge_context}: edge endpoint is absent from node set ({source} -> {target})"
            ));
        }
        let attributes = edge
            .get("attributes")
            .filter(|attributes| attributes.is_object())
            .ok_or_else(|| format!("{edge_context}.attributes: expected object"))?;
        normalized_edges.push((
            edge_id.clone(),
            json!({
                "edge_id": edge_id,
                "source": source,
                "target": target,
                "attributes": attributes.clone()
            }),
        ));
    }

    normalized_nodes.sort_by(|left, right| left.0.cmp(&right.0));
    normalized_edges.sort_by(|left, right| left.0.cmp(&right.0));

    let canonical = json!({
        "nodes": normalized_nodes.into_iter().map(|(_, value)| value).collect::<Vec<_>>(),
        "edges": normalized_edges.into_iter().map(|(_, value)| value).collect::<Vec<_>>()
    });
    let canonical_bytes = serde_json::to_vec(&canonical)
        .map_err(|error| format!("{context}: canonical serialization failed: {error}"))?;

    let mut digest_input = CANONICALIZER_PROTOCOL.as_bytes().to_vec();
    digest_input.push(0);
    digest_input.extend_from_slice(&canonical_bytes);
    Ok(sha256_hex(&digest_input))
}

fn parse_structure_sidecar(
    bytes: &[u8],
    context: &str,
) -> Result<BTreeMap<String, String>, String> {
    let root: Value = serde_json::from_slice(bytes)
        .map_err(|error| format!("{context}: invalid JSON: {error}"))?;
    let root_object = object(&root, context)?;
    require_keys(root_object, &["schema_version", "worlds"], context)?;
    if string_field(&root, "schema_version", context)? != SIDECAR_VERSION {
        return Err(format!("{context}: unsupported schema_version"));
    }

    let worlds = array(
        root.get("worlds").ok_or_else(|| format!("{context}.worlds missing"))?,
        &format!("{context}.worlds"),
    )?;
    let mut result = BTreeMap::new();

    for (index, world) in worlds.iter().enumerate() {
        let world_context = format!("{context}.worlds[{index}]");
        let world_object = object(world, &world_context)?;
        require_keys(world_object, &["world_id", "structure"], &world_context)?;
        let world_id = string_field(world, "world_id", &world_context)?;
        let structure = world
            .get("structure")
            .ok_or_else(|| format!("{world_context}.structure missing"))?;
        let digest = canonical_structure_sha256(structure, &format!("{world_context}.structure"))?;
        if result.insert(world_id.clone(), digest).is_some() {
            return Err(format!("{context}: duplicate world_id {world_id}"));
        }
    }
    Ok(result)
}

fn duplicate_group_count(values: impl Iterator<Item = String>) -> u64 {
    let mut counts = BTreeMap::<String, u64>::new();
    for value in values {
        *counts.entry(value).or_default() += 1;
    }
    counts.values().filter(|count| **count > 1).count() as u64
}

fn set_summary_field(summary: &Summary, key: &str, value: &Value) -> Result<(), String> {
    let context = format!("summary.{key}");
    let expected = match key {
        "candidate_world_count" => summary.candidate_world_count,
        "reference_world_count" => summary.reference_world_count,
        "compatible_reference_count" => summary.compatible_reference_count,
        "incompatible_reference_count" => summary.incompatible_reference_count,
        "indeterminate_reference_count" => summary.indeterminate_reference_count,
        "compatible_estimable_reference_count" => summary.compatible_estimable_reference_count,
        "compatible_query_not_estimable_reference_count" => {
            summary.compatible_query_not_estimable_reference_count
        }
        "structurally_included_count" => summary.structurally_included_count,
        "query_state_covered_count" => summary.query_state_covered_count,
        "query_flipping_reference_count" => summary.query_flipping_reference_count,
        "omitted_query_flipping_count" => summary.omitted_query_flipping_count,
        "false_invariance_count" => summary.false_invariance_count,
        "false_ambiguity_count" => summary.false_ambiguity_count,
        "candidate_duplicate_structure_group_count" => {
            summary.candidate_duplicate_structure_group_count
        }
        "reference_duplicate_structure_group_count" => {
            summary.reference_duplicate_structure_group_count
        }
        "candidate_not_estimable_count" => summary.candidate_not_estimable_count,
        _ => return Err(format!("unknown summary metric {key}")),
    };
    let actual = u64_field(value, key, "summary")?;
    if actual != expected {
        return Err(format!("{context}: reported {actual}, independently recomputed {expected}"));
    }
    Ok(())
}

fn validate_family_summaries(
    report: &Value,
    required_families: &BTreeSet<String>,
    recomputed: &BTreeMap<String, FamilyCounts>,
) -> Result<(), String> {
    let values = array(
        report
            .get("family_summaries")
            .ok_or_else(|| "family_summaries missing".to_owned())?,
        "family_summaries",
    )?;
    if values.len() != required_families.len() {
        return Err("family_summaries must contain exactly one row per required family".to_owned());
    }

    let mut seen = BTreeSet::new();
    for value in values {
        require_keys(
            object(value, "family_summaries[]")?,
            &[
                "error_family",
                "compatible_reference_count",
                "incompatible_reference_count",
                "indeterminate_reference_count",
                "compatible_estimable_reference_count",
                "compatible_query_not_estimable_reference_count",
                "structurally_included_count",
                "query_flipping_reference_count",
                "omitted_query_flipping_count",
                "false_invariance_count",
                "false_ambiguity_count",
                "candidate_not_estimable_count",
            ],
            "family_summaries[]",
        )?;
        let family = string_field(value, "error_family", "family_summaries[]")?;
        if !seen.insert(family.clone()) {
            return Err(format!("duplicate family summary {family}"));
        }
        if !required_families.contains(&family) {
            return Err(format!("unexpected family summary {family}"));
        }
        let expected = recomputed
            .get(&family)
            .ok_or_else(|| format!("no recomputed counters for family {family}"))?;

        let metrics = [
            ("compatible_reference_count", expected.compatible_reference_count),
            ("incompatible_reference_count", expected.incompatible_reference_count),
            ("indeterminate_reference_count", expected.indeterminate_reference_count),
            ("compatible_estimable_reference_count", expected.compatible_estimable_reference_count),
            ("compatible_query_not_estimable_reference_count", expected.compatible_query_not_estimable_reference_count),
            ("structurally_included_count", expected.structurally_included_count),
            ("query_flipping_reference_count", expected.query_flipping_reference_count),
            ("omitted_query_flipping_count", expected.omitted_query_flipping_count),
            ("false_invariance_count", expected.false_invariance_count),
            ("false_ambiguity_count", expected.false_ambiguity_count),
            ("candidate_not_estimable_count", expected.candidate_not_estimable_count),
        ];
        for (key, expected_value) in metrics {
            let actual = u64_field(value, key, "family_summaries[]")?;
            if actual != expected_value {
                return Err(format!(
                    "family_summaries[{family}].{key}: reported {actual}, recomputed {expected_value}"
                ));
            }
        }
        if expected.compatible_estimable_reference_count
            + expected.compatible_query_not_estimable_reference_count
            != expected.compatible_reference_count
            || expected.structurally_included_count > expected.compatible_reference_count
            || expected.query_flipping_reference_count > expected.compatible_estimable_reference_count
            || expected.omitted_query_flipping_count > expected.query_flipping_reference_count
            || expected.false_invariance_count > expected.compatible_estimable_reference_count
        {
            return Err(format!("family_summaries[{family}] violates an eligible-denominator invariant"));
        }
    }

    if &seen != required_families {
        return Err("family_summaries do not exactly cover required_error_families".to_owned());
    }
    Ok(())
}

fn validate_report(
    report_bytes: &[u8],
    candidate_sidecar_bytes: &[u8],
    reference_sidecar_bytes: &[u8],
) -> Result<Value, String> {
    let report: Value = serde_json::from_slice(report_bytes)
        .map_err(|error| format!("report JSON invalid: {error}"))?;
    let report_object = object(&report, "report")?;
    require_keys(
        report_object,
        &[
            "schema_version",
            "evaluation_id",
            "query",
            "candidate_universe",
            "reference_challenge",
            "candidate_worlds",
            "reference_worlds",
            "summary",
            "family_summaries",
            "dispositions",
            "provenance",
        ],
        "report",
    )?;
    if string_field(&report, "schema_version", "report")? != REPORT_VERSION {
        return Err("unsupported report schema_version".to_owned());
    }
    let evaluation_id = string_field(&report, "evaluation_id", "report")?;

    let provenance = report
        .get("provenance")
        .ok_or_else(|| "provenance missing".to_owned())?;
    require_keys(
        object(provenance, "provenance")?,
        &[
            "evaluator_protocol_sha256",
            "rules_sha256",
            "truth_separation_verified",
            "canonicalizer_sha256",
            "report_schema_sha256",
            "structure_schema_sha256",
            "candidate_structure_manifest_sha256",
            "reference_structure_manifest_sha256",
        ],
        "provenance",
    )?;
    let evaluator_protocol_sha256 =
        digest_field(provenance, "evaluator_protocol_sha256", "provenance")?;
    let rules_sha256 = digest_field(provenance, "rules_sha256", "provenance")?;
    let canonicalizer_sha256 =
        digest_field(provenance, "canonicalizer_sha256", "provenance")?;
    if evaluator_protocol_sha256 != EVALUATOR_PROTOCOL_SHA256 {
        return Err("provenance.evaluator_protocol_sha256 does not match this verifier protocol".to_owned());
    }
    if rules_sha256 != RULES_PROTOCOL_SHA256 {
        return Err("provenance.rules_sha256 does not match the pinned adequacy rules protocol".to_owned());
    }
    if canonicalizer_sha256 != CANONICALIZER_PROTOCOL_SHA256 {
        return Err("provenance.canonicalizer_sha256 does not match the pinned graph canonicalizer".to_owned());
    }
    let report_schema_sha256 =
        digest_field(provenance, "report_schema_sha256", "provenance")?;
    let structure_schema_sha256 =
        digest_field(provenance, "structure_schema_sha256", "provenance")?;
    if report_schema_sha256 != REPORT_SCHEMA_SHA256 {
        return Err("provenance.report_schema_sha256 does not match the pinned report schema".to_owned());
    }
    if structure_schema_sha256 != STRUCTURE_SCHEMA_SHA256 {
        return Err("provenance.structure_schema_sha256 does not match the pinned structure schema".to_owned());
    }

    let candidate_manifest = digest_field(
        provenance,
        "candidate_structure_manifest_sha256",
        "provenance",
    )?;
    let reference_manifest = digest_field(
        provenance,
        "reference_structure_manifest_sha256",
        "provenance",
    )?;
    if sha256_hex(candidate_sidecar_bytes) != candidate_manifest {
        return Err("candidate structure sidecar raw-byte SHA-256 mismatch".to_owned());
    }
    if sha256_hex(reference_sidecar_bytes) != reference_manifest {
        return Err("reference structure sidecar raw-byte SHA-256 mismatch".to_owned());
    }

    let candidate_structures =
        parse_structure_sidecar(candidate_sidecar_bytes, "candidate_structures")?;
    let reference_structures =
        parse_structure_sidecar(reference_sidecar_bytes, "reference_structures")?;

    let query = report.get("query").ok_or_else(|| "query missing".to_owned())?;
    require_keys(
        object(query, "query")?,
        &[
            "query_id",
            "definition_sha256",
            "decision_rule_sha256",
            "metric",
            "unit",
            "baseline_decision",
        ],
        "query",
    )?;
    string_field(query, "query_id", "query")?;
    string_field(query, "metric", "query")?;
    string_field(query, "unit", "query")?;
    let baseline_decision = string_field(query, "baseline_decision", "query")?;
    if !matches!(
        baseline_decision.as_str(),
        "BELOW_THRESHOLD" | "AT_OR_ABOVE_THRESHOLD"
    ) {
        return Err("query.baseline_decision must be a determinate decision".to_owned());
    }
    digest_field(query, "definition_sha256", "query")?;
    digest_field(query, "decision_rule_sha256", "query")?;

    let reference_challenge = report
        .get("reference_challenge")
        .ok_or_else(|| "reference_challenge missing".to_owned())?;
    require_keys(
        object(reference_challenge, "reference_challenge")?,
        &[
            "challenge_id",
            "manifest_sha256",
            "generator_sha256",
            "freeze_receipt_sha256",
            "oracle_boundary",
            "required_error_families",
        ],
        "reference_challenge",
    )?;
    string_field(reference_challenge, "challenge_id", "reference_challenge")?;
    digest_field(reference_challenge, "manifest_sha256", "reference_challenge")?;
    digest_field(reference_challenge, "generator_sha256", "reference_challenge")?;
    let oracle_boundary = string_field(
        reference_challenge,
        "oracle_boundary",
        "reference_challenge",
    )?;
    if !matches!(
        oracle_boundary.as_str(),
        "EVALUATOR_ONLY_VERIFIED" | "NOT_VERIFIED" | "NOT_AUDITABLE"
    ) {
        return Err("unknown reference oracle_boundary".to_owned());
    }
    let freeze_receipt_present = match reference_challenge.get("freeze_receipt_sha256") {
        Some(Value::Null) => false,
        Some(Value::String(value)) if valid_sha256(value) => true,
        _ => return Err("reference_challenge.freeze_receipt_sha256 must be null or lowercase SHA-256 hex".to_owned()),
    };
    let truth_separation_verified =
        bool_field(provenance, "truth_separation_verified", "provenance")?;

    let required_values = array(
        reference_challenge
            .get("required_error_families")
            .ok_or_else(|| "reference_challenge.required_error_families missing".to_owned())?,
        "required_error_families",
    )?;
    if required_values.is_empty() {
        return Err("required_error_families must be non-empty".to_owned());
    }
    let mut required_families = BTreeSet::new();
    for value in required_values {
        let family = value
            .as_str()
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| "required_error_families entries must be non-empty strings".to_owned())?;
        if !required_families.insert(family.to_owned()) {
            return Err(format!("duplicate required error family {family}"));
        }
    }

    let candidate_values = array(
        report
            .get("candidate_worlds")
            .ok_or_else(|| "candidate_worlds missing".to_owned())?,
        "candidate_worlds",
    )?;
    let reference_values = array(
        report
            .get("reference_worlds")
            .ok_or_else(|| "reference_worlds missing".to_owned())?,
        "reference_worlds",
    )?;
    if candidate_values.is_empty() || reference_values.is_empty() {
        return Err("candidate_worlds and reference_worlds must both be non-empty".to_owned());
    }

    let candidate_universe = report
        .get("candidate_universe")
        .ok_or_else(|| "candidate_universe missing".to_owned())?;
    require_keys(
        object(candidate_universe, "candidate_universe")?,
        &[
            "universe_id",
            "manifest_sha256",
            "generator_sha256",
            "closure_disposition",
            "world_ids",
        ],
        "candidate_universe",
    )?;
    string_field(candidate_universe, "universe_id", "candidate_universe")?;
    let closure_disposition = string_field(
        candidate_universe,
        "closure_disposition",
        "candidate_universe",
    )?;
    if !matches!(
        closure_disposition.as_str(),
        "CLOSED_VERIFIED" | "NOT_CLOSED" | "NOT_AUDITABLE"
    ) {
        return Err("unknown candidate_universe.closure_disposition".to_owned());
    }
    digest_field(candidate_universe, "manifest_sha256", "candidate_universe")?;
    digest_field(candidate_universe, "generator_sha256", "candidate_universe")?;

    let universe_ids = array(
        candidate_universe
            .get("world_ids")
            .ok_or_else(|| "candidate_universe.world_ids missing".to_owned())?,
        "candidate_universe.world_ids",
    )?;
    let mut declared_ids = BTreeSet::new();
    for value in universe_ids {
        let id = value
            .as_str()
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| "candidate_universe.world_ids contains invalid ID".to_owned())?;
        if !declared_ids.insert(id.to_owned()) {
            return Err(format!("duplicate declared candidate world ID {id}"));
        }
    }

    let mut candidates = Vec::new();
    let mut candidate_ids = BTreeSet::new();
    for value in candidate_values {
        require_keys(
            object(value, "candidate_worlds[]")?,
            &["world_id", "query_state", "canonical_structural_sha256", "error_family"],
            "candidate_worlds[]",
        )?;
        let id = string_field(value, "world_id", "candidate_worlds[]")?;
        if !candidate_ids.insert(id.clone()) {
            return Err(format!("duplicate candidate world ID {id}"));
        }
        let family = string_field(value, "error_family", "candidate_worlds[]")?;
        if !required_families.contains(&family) {
            return Err(format!("candidate world {id} uses undeclared error family {family}"));
        }
        let state = string_field(value, "query_state", "candidate_worlds[]")?;
        if !matches!(
            state.as_str(),
            "BELOW_THRESHOLD" | "AT_OR_ABOVE_THRESHOLD" | "NOT_ESTIMABLE"
        ) {
            return Err(format!("candidate world {id} has invalid query_state"));
        }
        let structural_sha256 = digest_field(
            value,
            "canonical_structural_sha256",
            &format!("candidate_worlds[{id}]"),
        )?;
        let computed = candidate_structures
            .get(&id)
            .ok_or_else(|| format!("candidate structure sidecar is missing world {id}"))?;
        if computed != &structural_sha256 {
            return Err(format!(
                "candidate world {id}: canonical structural digest mismatch"
            ));
        }
        candidates.push(CandidateWorld {
            id,
            family,
            state,
            structural_sha256,
        });
    }
    if candidate_ids != declared_ids {
        return Err("candidate_universe.world_ids does not exactly match candidate_worlds".to_owned());
    }
    if candidate_ids != candidate_structures.keys().cloned().collect() {
        return Err("candidate structure sidecar IDs do not exactly match candidate_worlds".to_owned());
    }

    let mut references = Vec::new();
    let mut reference_ids = BTreeSet::new();
    for value in reference_values {
        require_keys(
            object(value, "reference_worlds[]")?,
            &["world_id", "error_family", "evidence_compatibility", "query_state", "canonical_structural_sha256"],
            "reference_worlds[]",
        )?;
        let id = string_field(value, "world_id", "reference_worlds[]")?;
        if !reference_ids.insert(id.clone()) {
            return Err(format!("duplicate reference world ID {id}"));
        }
        let family = string_field(value, "error_family", "reference_worlds[]")?;
        if !required_families.contains(&family) {
            return Err(format!("reference world {id} uses undeclared error family {family}"));
        }
        let compatibility =
            string_field(value, "evidence_compatibility", "reference_worlds[]")?;
        if !matches!(
            compatibility.as_str(),
            "COMPATIBLE" | "INCOMPATIBLE" | "INDETERMINATE"
        ) {
            return Err(format!("reference world {id} has invalid evidence_compatibility"));
        }
        let state = determinate_state(
            value
                .get("query_state")
                .ok_or_else(|| format!("reference world {id} missing query_state"))?,
            &format!("reference world {id}.query_state"),
        )?;
        let structural_sha256 = digest_field(
            value,
            "canonical_structural_sha256",
            &format!("reference_worlds[{id}]"),
        )?;
        let computed = reference_structures
            .get(&id)
            .ok_or_else(|| format!("reference structure sidecar is missing world {id}"))?;
        if computed != &structural_sha256 {
            return Err(format!(
                "reference world {id}: canonical structural digest mismatch"
            ));
        }
        references.push(ReferenceWorld {
            id,
            family,
            compatibility,
            state,
            structural_sha256,
        });
    }
    if reference_ids != reference_structures.keys().cloned().collect() {
        return Err("reference structure sidecar IDs do not exactly match reference_worlds".to_owned());
    }

    // Same canonical graph + same frozen query implementation must yield the same
    // determinate query state regardless of world ID or which manifest contains it.
    let mut state_by_structure = BTreeMap::<String, String>::new();
    for (structural_sha256, state) in candidates
        .iter()
        .map(|world| (&world.structural_sha256, Some(world.state.clone())))
        .chain(
            references
                .iter()
                .map(|world| (&world.structural_sha256, world.state.clone())),
        )
    {
        if let Some(state) = state.filter(|state| state.as_str() != "NOT_ESTIMABLE") {
            if let Some(previous) =
                state_by_structure.insert(structural_sha256.clone(), state.clone())
            {
                if previous != state {
                    return Err(format!(
                        "canonical structure {structural_sha256} has inconsistent query states ({previous} vs {state})"
                    ));
                }
            }
        }
    }

    let candidate_hashes: BTreeSet<String> = candidates
        .iter()
        .map(|world| world.structural_sha256.clone())
        .collect();
    let candidate_states: BTreeSet<String> = candidates
        .iter()
        .filter(|world| world.state != "NOT_ESTIMABLE")
        .map(|world| world.state.clone())
        .collect();
    let all_candidates_estimable = candidates
        .iter()
        .all(|world| world.state != "NOT_ESTIMABLE");
    let sole_candidate_state = if all_candidates_estimable && candidate_states.len() == 1 {
        candidate_states.iter().next().cloned()
    } else {
        None
    };

    let summary_object = report.get("summary").ok_or_else(|| "summary missing".to_owned())?;
    require_keys(
        object(summary_object, "summary")?,
        &[
            "candidate_world_count",
            "reference_world_count",
            "compatible_reference_count",
            "incompatible_reference_count",
            "indeterminate_reference_count",
            "compatible_estimable_reference_count",
            "compatible_query_not_estimable_reference_count",
            "structurally_included_count",
            "query_state_covered_count",
            "query_flipping_reference_count",
            "omitted_query_flipping_count",
            "false_invariance_count",
            "false_ambiguity_count",
            "candidate_duplicate_structure_group_count",
            "reference_duplicate_structure_group_count",
            "candidate_not_estimable_count",
        ],
        "summary",
    )?;

    let mut summary = Summary {
        candidate_world_count: candidates.len() as u64,
        reference_world_count: references.len() as u64,
        candidate_duplicate_structure_group_count: duplicate_group_count(
            candidates.iter().map(|world| world.structural_sha256.clone()),
        ),
        reference_duplicate_structure_group_count: duplicate_group_count(
            references.iter().map(|world| world.structural_sha256.clone()),
        ),
        candidate_not_estimable_count: candidates
            .iter()
            .filter(|world| world.state == "NOT_ESTIMABLE")
            .count() as u64,
        ..Summary::default()
    };

    let reference_states: BTreeSet<String> = references
        .iter()
        .filter(|world| world.compatibility == "COMPATIBLE")
        .filter_map(|world| world.state.clone())
        .collect();

    let mut family_counts = BTreeMap::<String, FamilyCounts>::new();
    for family in &required_families {
        family_counts.insert(family.clone(), FamilyCounts::default());
    }

    for world in &candidates {
        if world.state == "NOT_ESTIMABLE" {
            family_counts
                .get_mut(&world.family)
                .expect("family checked")
                .candidate_not_estimable_count += 1;
        }
    }

    for world in &references {
        let family = family_counts
            .get_mut(&world.family)
            .expect("reference family was validated");

        match world.compatibility.as_str() {
            "INCOMPATIBLE" => {
                summary.incompatible_reference_count += 1;
                family.incompatible_reference_count += 1;
                continue;
            }
            "INDETERMINATE" => {
                summary.indeterminate_reference_count += 1;
                family.indeterminate_reference_count += 1;
                continue;
            }
            "COMPATIBLE" => {
                summary.compatible_reference_count += 1;
                family.compatible_reference_count += 1;
            }
            _ => return Err(format!("reference world {} has invalid compatibility", world.id)),
        }

        // Structural inclusion is independent of whether the downstream query
        // can be evaluated for this otherwise evidence-compatible graph.
        if candidate_hashes.contains(&world.structural_sha256) {
            summary.structurally_included_count += 1;
            family.structurally_included_count += 1;
        }

        let Some(state) = world.state.as_ref() else {
            summary.compatible_query_not_estimable_reference_count += 1;
            family.compatible_query_not_estimable_reference_count += 1;
            continue;
        };

        summary.compatible_estimable_reference_count += 1;
        family.compatible_estimable_reference_count += 1;

        if candidate_states.contains(state) {
            summary.query_state_covered_count += 1;
        }

        if state != &baseline_decision {
            summary.query_flipping_reference_count += 1;
            family.query_flipping_reference_count += 1;
            if !candidate_hashes.contains(&world.structural_sha256) {
                summary.omitted_query_flipping_count += 1;
                family.omitted_query_flipping_count += 1;
            }
        }

        if let Some(candidate_state) = sole_candidate_state.as_ref() {
            if state != candidate_state {
                summary.false_invariance_count += 1;
                family.false_invariance_count += 1;
            }
        }
    }

    if !reference_states.is_empty() {
        for world in &candidates {
            if world.state != "NOT_ESTIMABLE" && !reference_states.contains(&world.state) {
                summary.false_ambiguity_count += 1;
                family_counts
                    .get_mut(&world.family)
                    .expect("family checked")
                    .false_ambiguity_count += 1;
            }
        }
    }

    let family_keys = [
        "compatible_reference_count",
        "incompatible_reference_count",
        "indeterminate_reference_count",
        "compatible_estimable_reference_count",
        "compatible_query_not_estimable_reference_count",
        "structurally_included_count",
        "query_flipping_reference_count",
        "omitted_query_flipping_count",
        "false_invariance_count",
        "false_ambiguity_count",
        "candidate_not_estimable_count",
    ];
    validate_family_summaries(&report, &required_families, &family_counts)?;

    for key in [
        "candidate_world_count",
        "reference_world_count",
        "compatible_reference_count",
        "incompatible_reference_count",
        "indeterminate_reference_count",
        "compatible_estimable_reference_count",
        "compatible_query_not_estimable_reference_count",
        "structurally_included_count",
        "query_state_covered_count",
        "query_flipping_reference_count",
        "omitted_query_flipping_count",
        "false_invariance_count",
        "false_ambiguity_count",
        "candidate_duplicate_structure_group_count",
        "reference_duplicate_structure_group_count",
        "candidate_not_estimable_count",
    ] {
        set_summary_field(
            &summary,
            key,
            report
                .get("summary")
                .and_then(|value| value.get(key))
                .ok_or_else(|| format!("summary.{key} missing"))?,
        )?;
    }

    for key in family_keys {
        let family_sum: u64 = family_counts
            .values()
            .map(|counts| match key {
                "compatible_reference_count" => counts.compatible_reference_count,
                "incompatible_reference_count" => counts.incompatible_reference_count,
                "indeterminate_reference_count" => counts.indeterminate_reference_count,
                "compatible_estimable_reference_count" => counts.compatible_estimable_reference_count,
                "compatible_query_not_estimable_reference_count" => counts.compatible_query_not_estimable_reference_count,
                "structurally_included_count" => counts.structurally_included_count,
                "query_flipping_reference_count" => counts.query_flipping_reference_count,
                "omitted_query_flipping_count" => counts.omitted_query_flipping_count,
                "false_invariance_count" => counts.false_invariance_count,
                "false_ambiguity_count" => counts.false_ambiguity_count,
                "candidate_not_estimable_count" => counts.candidate_not_estimable_count,
                _ => 0,
            })
            .sum();
        let aggregate = match key {
            "compatible_reference_count" => summary.compatible_reference_count,
            "incompatible_reference_count" => summary.incompatible_reference_count,
            "indeterminate_reference_count" => summary.indeterminate_reference_count,
            "compatible_estimable_reference_count" => summary.compatible_estimable_reference_count,
            "compatible_query_not_estimable_reference_count" => summary.compatible_query_not_estimable_reference_count,
            "structurally_included_count" => summary.structurally_included_count,
            "query_flipping_reference_count" => summary.query_flipping_reference_count,
            "omitted_query_flipping_count" => summary.omitted_query_flipping_count,
            "false_invariance_count" => summary.false_invariance_count,
            "false_ambiguity_count" => summary.false_ambiguity_count,
            "candidate_not_estimable_count" => summary.candidate_not_estimable_count,
            _ => 0,
        };
        if family_sum != aggregate {
            return Err(format!(
                "family summaries do not reconcile for {key}: {family_sum} != {aggregate}"
            ));
        }
    }

    if summary.compatible_reference_count
        + summary.incompatible_reference_count
        + summary.indeterminate_reference_count
        != summary.reference_world_count
    {
        return Err("reference compatibility counts do not reconcile to reference_world_count".to_owned());
    }
    if summary.compatible_estimable_reference_count
        + summary.compatible_query_not_estimable_reference_count
        != summary.compatible_reference_count
    {
        return Err("compatible query-estimability counts do not reconcile to compatible_reference_count".to_owned());
    }
    let candidate_estimable_denominator = summary
        .candidate_world_count
        .checked_sub(summary.candidate_not_estimable_count)
        .ok_or_else(|| "candidate_not_estimable_count exceeds candidate_world_count".to_owned())?;
    if summary.structurally_included_count > summary.compatible_reference_count
        || summary.query_state_covered_count > summary.compatible_estimable_reference_count
        || summary.query_flipping_reference_count > summary.compatible_estimable_reference_count
        || summary.omitted_query_flipping_count > summary.query_flipping_reference_count
        || summary.false_invariance_count > summary.compatible_estimable_reference_count
        || summary.false_ambiguity_count > candidate_estimable_denominator
        || summary.candidate_duplicate_structure_group_count > summary.candidate_world_count / 2
        || summary.reference_duplicate_structure_group_count > summary.reference_world_count / 2
    {
        return Err("one or more adequacy numerators exceed their eligible denominators".to_owned());
    }

    let dispositions = report
        .get("dispositions")
        .ok_or_else(|| "dispositions missing".to_owned())?;
    require_keys(
        object(dispositions, "dispositions")?,
        &["enumeration_closure", "universe_adequacy"],
        "dispositions",
    )?;
    let reported_closure = string_field(dispositions, "enumeration_closure", "dispositions")?;
    let reported_adequacy = string_field(dispositions, "universe_adequacy", "dispositions")?;
    if reported_closure != closure_disposition {
        return Err("dispositions.enumeration_closure disagrees with candidate_universe.closure_disposition".to_owned());
    }

    let has_counterexample = summary.omitted_query_flipping_count > 0
        || summary.false_invariance_count > 0
        || summary.false_ambiguity_count > 0;
    let every_required_family_supported = family_counts
        .values()
        .all(|counts| counts.compatible_estimable_reference_count > 0);
    // This crate does not yet consume an authenticated #10204 freeze-auth receipt or
    // independently replay the #10137 Cartesian-closure receipt. Until those adapters
    // are wired in, it must never promote a success claim based on report booleans.
    let authenticated_external_receipts_available = false;
    let can_support = authenticated_external_receipts_available
        && closure_disposition == "CLOSED_VERIFIED"
        && oracle_boundary == "EVALUATOR_ONLY_VERIFIED"
        && freeze_receipt_present
        && truth_separation_verified
        && summary.indeterminate_reference_count == 0
        && summary.compatible_query_not_estimable_reference_count == 0
        && summary.candidate_not_estimable_count == 0
        && every_required_family_supported
        && !candidate_states.is_empty()
        && !reference_states.is_empty();

    let recomputed_adequacy = if has_counterexample {
        "ADEQUACY_FAILED"
    } else if can_support {
        "ADEQUACY_SUPPORTED_WITHIN_TESTED_CHALLENGES"
    } else {
        "ADEQUACY_NOT_ESTIMABLE"
    };
    if reported_adequacy != recomputed_adequacy {
        return Err(format!(
            "dispositions.universe_adequacy reported {reported_adequacy}, recomputed {recomputed_adequacy}"
        ));
    }

    Ok(json!({
        "schema_version": REPORT_VERSION,
        "evaluation_id": evaluation_id,
        "source_report_sha256": sha256_hex(report_bytes),
        "status": "COUNTERS_RECOMPUTED",
        "trusted_promotion": false,
        "claim_ceiling": "supplied reference challenge only; external worldset closure and pre-outcome freeze authentication are not verified by this tool",
        "canonicalizer_protocol": CANONICALIZER_PROTOCOL,
        "evaluator_protocol_sha256": evaluator_protocol_sha256,
        "rules_sha256": rules_sha256,
        "canonicalizer_sha256": canonicalizer_sha256,
        "report_schema_sha256": report_schema_sha256,
        "structure_schema_sha256": structure_schema_sha256,
        "candidate_structure_manifest_sha256": candidate_manifest,
        "reference_structure_manifest_sha256": reference_manifest,
        "dispositions": {
            "claimed_enumeration_closure": closure_disposition,
            "worldset_closure_validation": "NOT_PERFORMED_BY_THIS_TOOL",
            "universe_adequacy_against_supplied_challenge": recomputed_adequacy
        },
        "recomputed_summary": {
            "candidate_world_count": summary.candidate_world_count,
            "reference_world_count": summary.reference_world_count,
            "compatible_reference_count": summary.compatible_reference_count,
            "incompatible_reference_count": summary.incompatible_reference_count,
            "indeterminate_reference_count": summary.indeterminate_reference_count,
            "structurally_included_count": summary.structurally_included_count,
            "query_state_covered_count": summary.query_state_covered_count,
            "query_flipping_reference_count": summary.query_flipping_reference_count,
            "omitted_query_flipping_count": summary.omitted_query_flipping_count,
            "false_invariance_count": summary.false_invariance_count,
            "compatible_estimable_reference_count": summary.compatible_estimable_reference_count,
            "compatible_query_not_estimable_reference_count": summary.compatible_query_not_estimable_reference_count,
            "false_ambiguity_count": summary.false_ambiguity_count,
            "candidate_duplicate_structure_group_count": summary.candidate_duplicate_structure_group_count,
            "reference_duplicate_structure_group_count": summary.reference_duplicate_structure_group_count,
            "candidate_not_estimable_count": summary.candidate_not_estimable_count
        }
    }))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!(
            "usage: {} <report.json> <candidate-structures.json> <reference-structures.json>",
            args.first().map(String::as_str).unwrap_or("connectome-universe-adequacy-verifier")
        );
        process::exit(64);
    }

    let report = fs::read(&args[1]).unwrap_or_else(|error| {
        eprintln!("cannot read report: {error}");
        process::exit(66);
    });
    let candidate = fs::read(&args[2]).unwrap_or_else(|error| {
        eprintln!("cannot read candidate structure manifest: {error}");
        process::exit(66);
    });
    let reference = fs::read(&args[3]).unwrap_or_else(|error| {
        eprintln!("cannot read reference structure manifest: {error}");
        process::exit(66);
    });

    match validate_report(&report, &candidate, &reference) {
        Ok(result) => {
            println!("{}", serde_json::to_string(&result).expect("JSON output serialization"));
        }
        Err(error) => {
            eprintln!("REJECTED: {error}");
            process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OMITTED_REPORT: &str = include_str!(
        "../../../research/connectomics/fixtures/connectome-universe-adequacy-omitted-flip-known-answer-v0.1.json"
    );
    const OMITTED_CANDIDATE: &str = include_str!(
        "../../../research/connectomics/fixtures/connectome-universe-adequacy-omitted-candidate-structures-v0.1.json"
    );
    const OMITTED_REFERENCE: &str = include_str!(
        "../../../research/connectomics/fixtures/connectome-universe-adequacy-omitted-reference-structures-v0.1.json"
    );

    const FALSE_AMBIGUITY_REPORT: &str = include_str!(
        "../../../research/connectomics/fixtures/connectome-universe-adequacy-false-ambiguity-known-answer-v0.1.json"
    );
    const FALSE_AMBIGUITY_CANDIDATE: &str = include_str!(
        "../../../research/connectomics/fixtures/connectome-universe-adequacy-false-ambiguity-candidate-structures-v0.1.json"
    );
    const FALSE_AMBIGUITY_REFERENCE: &str = include_str!(
        "../../../research/connectomics/fixtures/connectome-universe-adequacy-false-ambiguity-reference-structures-v0.1.json"
    );


    const STATUS_AXIS_REPORT: &str = include_str!(
        "../../../research/connectomics/fixtures/connectome-universe-adequacy-status-axis-known-answer-v0.1.json"
    );
    const STATUS_AXIS_CANDIDATE: &str = include_str!(
        "../../../research/connectomics/fixtures/connectome-universe-adequacy-status-axis-candidate-structures-v0.1.json"
    );
    const STATUS_AXIS_REFERENCE: &str = include_str!(
        "../../../research/connectomics/fixtures/connectome-universe-adequacy-status-axis-reference-structures-v0.1.json"
    );

    fn evaluate(report: &str, candidate: &str, reference: &str) -> Result<Value, String> {
        validate_report(report.as_bytes(), candidate.as_bytes(), reference.as_bytes())
    }

    #[test]
    fn detects_closed_but_inadequate_omitted_query_flip() {
        let result = evaluate(OMITTED_REPORT, OMITTED_CANDIDATE, OMITTED_REFERENCE)
            .expect("fixture should be internally consistent");
        assert_eq!(
            result["dispositions"]["claimed_enumeration_closure"],
            "CLOSED_VERIFIED"
        );
        assert_eq!(
            result["dispositions"]["worldset_closure_validation"],
            "NOT_PERFORMED_BY_THIS_TOOL"
        );
        assert_eq!(
            result["dispositions"]["universe_adequacy_against_supplied_challenge"],
            "ADEQUACY_FAILED"
        );
        assert_eq!(result["recomputed_summary"]["omitted_query_flipping_count"], 1);
        assert_eq!(result["recomputed_summary"]["false_invariance_count"], 1);
    }

    #[test]
    fn detects_closed_but_inadequate_false_ambiguity() {
        let result = evaluate(
            FALSE_AMBIGUITY_REPORT,
            FALSE_AMBIGUITY_CANDIDATE,
            FALSE_AMBIGUITY_REFERENCE,
        )
        .expect("fixture should be internally consistent");
        assert_eq!(
            result["dispositions"]["claimed_enumeration_closure"],
            "CLOSED_VERIFIED"
        );
        assert_eq!(
            result["dispositions"]["worldset_closure_validation"],
            "NOT_PERFORMED_BY_THIS_TOOL"
        );
        assert_eq!(
            result["dispositions"]["universe_adequacy_against_supplied_challenge"],
            "ADEQUACY_FAILED"
        );
        assert_eq!(result["recomputed_summary"]["false_ambiguity_count"], 1);
        assert_eq!(result["recomputed_summary"]["structurally_included_count"], 1);
        assert_eq!(result["recomputed_summary"]["query_state_covered_count"], 2);
    }

    #[test]
    fn rejects_mutated_candidate_sidecar_bytes() {
        let mut candidate = OMITTED_CANDIDATE.as_bytes().to_vec();
        candidate.push(b' ');
        assert!(validate_report(
            OMITTED_REPORT.as_bytes(),
            &candidate,
            OMITTED_REFERENCE.as_bytes()
        )
        .is_err());
    }

    #[test]
    fn rejects_report_structural_digest_mismatch() {
        let mut report: Value = serde_json::from_str(OMITTED_REPORT).expect("valid fixture");
        report["candidate_worlds"][0]["canonical_structural_sha256"] =
            Value::String("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned());
        let report_bytes = serde_json::to_vec(&report).expect("serializable report");
        assert!(validate_report(
            &report_bytes,
            OMITTED_CANDIDATE.as_bytes(),
            OMITTED_REFERENCE.as_bytes()
        )
        .is_err());
    }

    #[test]
    fn canonicalization_is_invariant_to_node_and_edge_order() {
        let mut candidate: Value =
            serde_json::from_str(OMITTED_CANDIDATE).expect("valid candidate sidecar");
        let structure = &mut candidate["worlds"][0]["structure"];
        structure["nodes"].as_array_mut().expect("nodes").reverse();
        structure["edges"].as_array_mut().expect("edges").reverse();
        let candidate_bytes = serde_json::to_vec(&candidate).expect("serialized sidecar");

        let mut report: Value = serde_json::from_str(OMITTED_REPORT).expect("valid report");
        report["provenance"]["candidate_structure_manifest_sha256"] =
            Value::String(sha256_hex(&candidate_bytes));
        let report_bytes = serde_json::to_vec(&report).expect("serialized report");

        assert!(validate_report(
            &report_bytes,
            &candidate_bytes,
            OMITTED_REFERENCE.as_bytes()
        )
        .is_ok());
    }

    #[test]
    fn rejects_inconsistent_query_state_for_same_canonical_graph() {
        let mut report: Value = serde_json::from_str(OMITTED_REPORT).expect("valid report");
        report["reference_worlds"][0]["query_state"] =
            Value::String("AT_OR_ABOVE_THRESHOLD".to_owned());
        let report_bytes = serde_json::to_vec(&report).expect("serialized report");
        assert!(validate_report(
            &report_bytes,
            OMITTED_CANDIDATE.as_bytes(),
            OMITTED_REFERENCE.as_bytes()
        )
        .is_err());
    }
    #[test]
    fn rejects_unpinned_rules_protocol_digest() {
        let mut report: Value = serde_json::from_str(OMITTED_REPORT).expect("valid fixture");
        report["provenance"]["rules_sha256"] =
            Value::String("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned());
        let report_bytes = serde_json::to_vec(&report).expect("serialized report");
        assert!(validate_report(
            &report_bytes,
            OMITTED_CANDIDATE.as_bytes(),
            OMITTED_REFERENCE.as_bytes()
        )
        .is_err());
    }

    #[test]
    fn keeps_evidence_compatibility_separate_from_query_estimability() {
        let result = evaluate(STATUS_AXIS_REPORT, STATUS_AXIS_CANDIDATE, STATUS_AXIS_REFERENCE)
            .expect("status-axis fixture should be internally consistent");
        let counts = &result["recomputed_summary"];
        assert_eq!(counts["compatible_reference_count"], 1);
        assert_eq!(counts["incompatible_reference_count"], 1);
        assert_eq!(counts["indeterminate_reference_count"], 1);
        assert_eq!(counts["compatible_estimable_reference_count"], 0);
        assert_eq!(counts["compatible_query_not_estimable_reference_count"], 1);
        assert_eq!(counts["structurally_included_count"], 1);
        assert_eq!(
            result["dispositions"]["universe_adequacy_against_supplied_challenge"],
            "ADEQUACY_NOT_ESTIMABLE"
        );
    }
    #[test]
    fn withholds_adequacy_promotion_without_authenticated_external_receipts() {
        let mut report: Value =
            serde_json::from_str(FALSE_AMBIGUITY_REPORT).expect("valid report fixture");
        report["candidate_worlds"][1]["query_state"] =
            Value::String("BELOW_THRESHOLD".to_owned());
        report["summary"]["false_ambiguity_count"] = Value::from(0);
        for family in report["family_summaries"]
            .as_array_mut()
            .expect("family summaries")
        {
            family["false_ambiguity_count"] = Value::from(0);
        }
        report["dispositions"]["universe_adequacy"] =
            Value::String("ADEQUACY_NOT_ESTIMABLE".to_owned());

        let report_bytes = serde_json::to_vec(&report).expect("serialized report");
        let result = validate_report(
            &report_bytes,
            FALSE_AMBIGUITY_CANDIDATE.as_bytes(),
            FALSE_AMBIGUITY_REFERENCE.as_bytes(),
        )
        .expect("stable, internally consistent supplied challenge should recompute");
        assert_eq!(result["status"], "COUNTERS_RECOMPUTED");
        assert_eq!(result["trusted_promotion"], false);
        assert_eq!(
            result["dispositions"]["worldset_closure_validation"],
            "NOT_PERFORMED_BY_THIS_TOOL"
        );
        assert_eq!(
            result["dispositions"]["universe_adequacy_against_supplied_challenge"],
            "ADEQUACY_NOT_ESTIMABLE"
        );
    }

    #[test]
    fn rejects_unknown_report_fields_and_malformed_freeze_receipts() {
        let mut report: Value = serde_json::from_str(OMITTED_REPORT).expect("valid fixture");
        report["unregistered_extension"] = Value::Bool(true);
        let bytes = serde_json::to_vec(&report).expect("serialized report");
        assert!(validate_report(
            &bytes,
            OMITTED_CANDIDATE.as_bytes(),
            OMITTED_REFERENCE.as_bytes()
        )
        .is_err());

        let mut report: Value = serde_json::from_str(OMITTED_REPORT).expect("valid fixture");
        report["reference_challenge"]["freeze_receipt_sha256"] =
            Value::String("not-a-sha256".to_owned());
        let bytes = serde_json::to_vec(&report).expect("serialized report");
        assert!(validate_report(
            &bytes,
            OMITTED_CANDIDATE.as_bytes(),
            OMITTED_REFERENCE.as_bytes()
        )
        .is_err());
    }

    #[test]
    fn rejects_report_schema_protocol_digest_mismatch() {
        let mut report: Value = serde_json::from_str(OMITTED_REPORT).expect("valid fixture");
        report["provenance"]["report_schema_sha256"] =
            Value::String("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_owned());
        let report_bytes = serde_json::to_vec(&report).expect("serialized report");
        assert!(validate_report(
            &report_bytes,
            OMITTED_CANDIDATE.as_bytes(),
            OMITTED_REFERENCE.as_bytes()
        )
        .is_err());
    }


    #[test]
    fn candidate_abstention_suppresses_false_invariance_and_forces_not_estimable() {
        let mut report: Value =
            serde_json::from_str(FALSE_AMBIGUITY_REPORT).expect("valid report fixture");
        report["candidate_worlds"][1]["query_state"] =
            Value::String("NOT_ESTIMABLE".to_owned());
        report["summary"]["candidate_not_estimable_count"] = Value::from(1);
        report["summary"]["false_ambiguity_count"] = Value::from(0);
        report["family_summaries"][1]["candidate_not_estimable_count"] = Value::from(1);
        report["family_summaries"][1]["false_ambiguity_count"] = Value::from(0);
        report["dispositions"]["universe_adequacy"] =
            Value::String("ADEQUACY_NOT_ESTIMABLE".to_owned());

        let report_bytes = serde_json::to_vec(&report).expect("serialized report");
        let result = validate_report(
            &report_bytes,
            FALSE_AMBIGUITY_CANDIDATE.as_bytes(),
            FALSE_AMBIGUITY_REFERENCE.as_bytes(),
        )
        .expect("abstention case should recompute");
        assert_eq!(result["recomputed_summary"]["candidate_not_estimable_count"], 1);
        assert_eq!(result["recomputed_summary"]["false_ambiguity_count"], 0);
        assert_eq!(result["recomputed_summary"]["false_invariance_count"], 0);
        assert_eq!(
            result["dispositions"]["universe_adequacy_against_supplied_challenge"],
            "ADEQUACY_NOT_ESTIMABLE"
        );
    }

    #[test]
    fn canonicalizer_is_sensitive_to_edge_direction_and_attributes() {
        let mut sidecar: Value =
            serde_json::from_str(OMITTED_CANDIDATE).expect("valid candidate sidecar");
        let structure = sidecar["worlds"][0]["structure"].clone();
        let original =
            canonical_structure_sha256(&structure, "test.graph").expect("valid graph");

        sidecar["worlds"][0]["structure"]["edges"][0]["source"] =
            Value::String("n2".to_owned());
        sidecar["worlds"][0]["structure"]["edges"][0]["target"] =
            Value::String("n1".to_owned());
        let reversed =
            canonical_structure_sha256(&sidecar["worlds"][0]["structure"], "test.graph")
                .expect("reversed edge is valid");
        assert_ne!(original, reversed);

        sidecar["worlds"][0]["structure"] = structure;
        sidecar["worlds"][0]["structure"]["edges"][0]["attributes"]["kind"] =
            Value::String("gap_junction".to_owned());
        let retyped =
            canonical_structure_sha256(&sidecar["worlds"][0]["structure"], "test.graph")
                .expect("attributed edge is valid");
        assert_ne!(original, retyped);
    }

    #[test]
    fn rejects_duplicate_node_ids_and_dangling_edge_endpoints() {
        let mut sidecar: Value =
            serde_json::from_str(OMITTED_CANDIDATE).expect("valid candidate sidecar");
        let graph = &mut sidecar["worlds"][0]["structure"];
        graph["nodes"][1]["node_id"] = Value::String("n1".to_owned());
        assert!(canonical_structure_sha256(graph, "test.graph").is_err());

        let mut sidecar: Value =
            serde_json::from_str(OMITTED_CANDIDATE).expect("valid candidate sidecar");
        let graph = &mut sidecar["worlds"][0]["structure"];
        graph["edges"][0]["target"] = Value::String("missing-node".to_owned());
        assert!(canonical_structure_sha256(graph, "test.graph").is_err());
    }

    #[test]
    fn duplicate_structure_group_count_counts_groups_not_extra_rows() {
        let digests = vec![
            "a".to_owned(),
            "a".to_owned(),
            "a".to_owned(),
            "b".to_owned(),
            "c".to_owned(),
            "c".to_owned(),
        ];
        assert_eq!(duplicate_group_count(digests.into_iter()), 2);
    }


}
