//! HARDSEC-013 source-bound localization receipt auditor.
//!
//! Scores detector-facing candidate regions against a separately supplied,
//! evaluator-controlled insertion-footprint oracle. The oracle is never part
//! of the detector-facing explanation record.
//!
//! This tool verifies supplied identities and claims; it does not parse RTL,
//! prove circuit equivalence, authenticate the oracle, or certify a design.
//!
//! Usage:
//!   hardsec_localization_audit --explanation explanation.json --graph graph.json \
//!       --oracle evaluator-oracle.json --receipt receipt.json

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const CLAIM_SCOPE: &str = "insertion_footprint_localization_against_supplied_oracle_only";

#[derive(Debug, Clone)]
struct Edge {
    source: String,
    target: String,
}

#[derive(Debug, Clone)]
struct Region {
    id: String,
    rank: usize,
    nodes: Vec<String>,
    edges: Vec<String>,
}

#[derive(Debug, Clone)]
struct Args {
    explanation: PathBuf,
    graph: PathBuf,
    oracle: PathBuf,
    receipt: PathBuf,
}

fn sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}


fn string<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a str, String> {
    value.get(key).and_then(Value::as_str)
        .ok_or_else(|| format!("{context}.{key} must be a string"))
}

fn nonempty_id(value: &str, context: &str) -> Result<(), String> {
    if value.trim().is_empty() || value.chars().any(char::is_control) {
        return Err(format!("{context} must be non-empty and contain no control characters"));
    }
    Ok(())
}

fn string_array(value: &Value, key: &str, context: &str) -> Result<Vec<String>, String> {
    let values = value.get(key).and_then(Value::as_array)
        .ok_or_else(|| format!("{context}.{key} must be an array"))?;
    let mut output = Vec::with_capacity(values.len());
    let mut seen = BTreeSet::new();
    for (index, item) in values.iter().enumerate() {
        let id = item.as_str().ok_or_else(|| format!("{context}.{key}[{index}] must be a string"))?;
        nonempty_id(id, &format!("{context}.{key}[{index}]"))?;
        if !seen.insert(id.to_owned()) {
            return Err(format!("{context}.{key} contains duplicate ID {id}"));
        }
        output.push(id.to_owned());
    }
    Ok(output)
}

fn valid_sha(value: &str, context: &str) -> Result<(), String> {
    if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()) {
        return Err(format!("{context} must be a lowercase SHA-256 hex digest"));
    }
    Ok(())
}

fn require_digest(value: &Value, key: &str, expected: &str, context: &str) -> Result<(), String> {
    let actual = string(value, key, context)?;
    valid_sha(actual, &format!("{context}.{key}"))?;
    if actual != expected {
        return Err(format!("{context}.{key} does not match the supplied bytes"));
    }
    Ok(())
}

fn canonical_graph_digest(nodes: &BTreeSet<String>, edges: &BTreeMap<String, Edge>) -> String {
    let mut canonical = String::new();
    for node in nodes {
        canonical.push_str("N\t");
        canonical.push_str(node);
        canonical.push('\n');
    }
    for (id, edge) in edges {
        canonical.push_str("E\t");
        canonical.push_str(id);
        canonical.push('\t');
        canonical.push_str(&edge.source);
        canonical.push('\t');
        canonical.push_str(&edge.target);
        canonical.push('\n');
    }
    sha256(canonical.as_bytes())
}

fn load_graph(graph: &Value) -> Result<(BTreeSet<String>, BTreeMap<String, Edge>, String), String> {
    let ctx = "graph";
    if string(graph, "schema_version", ctx)? != "hardsec-source-graph-v1" {
        return Err("unsupported graph schema_version".to_owned());
    }
    let node_values = graph.get("nodes").and_then(Value::as_array)
        .ok_or_else(|| "graph.nodes must be an array".to_owned())?;
    if node_values.is_empty() {
        return Err("graph.nodes must not be empty".to_owned());
    }
    let mut nodes = BTreeSet::new();
    for (index, item) in node_values.iter().enumerate() {
        let id = item.as_str().ok_or_else(|| format!("graph.nodes[{index}] must be a string"))?;
        nonempty_id(id, &format!("graph.nodes[{index}]"))?;
        if !nodes.insert(id.to_owned()) {
            return Err(format!("duplicate graph node ID {id}"));
        }
    }

    let edge_values = graph.get("edges").and_then(Value::as_array)
        .ok_or_else(|| "graph.edges must be an array".to_owned())?;
    let mut edges = BTreeMap::new();
    for (index, item) in edge_values.iter().enumerate() {
        let edge_ctx = format!("graph.edges[{index}]");
        let id = string(item, "edge_id", &edge_ctx)?;
        let source = string(item, "source", &edge_ctx)?;
        let target = string(item, "target", &edge_ctx)?;
        nonempty_id(id, &format!("{edge_ctx}.edge_id"))?;
        nonempty_id(source, &format!("{edge_ctx}.source"))?;
        nonempty_id(target, &format!("{edge_ctx}.target"))?;
        if !nodes.contains(source) || !nodes.contains(target) {
            return Err(format!("{edge_ctx} references a node absent from graph.nodes"));
        }
        if edges.insert(id.to_owned(), Edge { source: source.to_owned(), target: target.to_owned() }).is_some() {
            return Err(format!("duplicate graph edge ID {id}"));
        }
    }
    if edges.is_empty() {
        return Err("graph.edges must not be empty".to_owned());
    }
    let map_digest = canonical_graph_digest(&nodes, &edges);
    Ok((nodes, edges, map_digest))
}

fn check_path_edges(
    path_id: &str,
    edge_ids: &[String],
    edges: &BTreeMap<String, Edge>,
    require_nonempty: bool,
) -> Result<(), String> {
    if require_nonempty && edge_ids.is_empty() {
        return Err(format!("path {path_id} must contain at least one edge"));
    }
    let mut previous_target: Option<&str> = None;
    for edge_id in edge_ids {
        let edge = edges.get(edge_id)
            .ok_or_else(|| format!("path {path_id} references missing edge {edge_id}"))?;
        if let Some(previous) = previous_target {
            if previous != edge.source {
                return Err(format!("path {path_id} is discontinuous at edge {edge_id}"));
            }
        }
        previous_target = Some(&edge.target);
    }
    Ok(())
}

fn parse_regions(
    explanation: &Value,
    nodes: &BTreeSet<String>,
    edges: &BTreeMap<String, Edge>,
) -> Result<Vec<Region>, String> {
    let values = explanation.get("candidate_regions").and_then(Value::as_array)
        .ok_or_else(|| "explanation.candidate_regions must be an array".to_owned())?;
    let mut regions = Vec::with_capacity(values.len());
    let mut ids = BTreeSet::new();
    let mut ranks = BTreeSet::new();
    for (index, value) in values.iter().enumerate() {
        let ctx = format!("explanation.candidate_regions[{index}]");
        let id = string(value, "region_id", &ctx)?.to_owned();
        nonempty_id(&id, &format!("{ctx}.region_id"))?;
        if !ids.insert(id.clone()) {
            return Err(format!("duplicate region_id {id}"));
        }
        let rank = value.get("rank").and_then(Value::as_u64)
            .ok_or_else(|| format!("{ctx}.rank must be a positive integer"))? as usize;
        if rank == 0 || !ranks.insert(rank) {
            return Err(format!("{ctx}.rank must be positive and unique"));
        }
        let score = value.get("score").and_then(Value::as_f64)
            .ok_or_else(|| format!("{ctx}.score must be a finite number in [0, 1]"))?;
        if !score.is_finite() || !(0.0..=1.0).contains(&score) {
            return Err(format!("{ctx}.score must be a finite number in [0, 1]"));
        }
        let region_nodes = string_array(value, "node_ids", &ctx)?;
        if region_nodes.is_empty() {
            return Err(format!("{ctx}.node_ids must not be empty"));
        }
        for node in &region_nodes {
            if !nodes.contains(node) {
                return Err(format!("{ctx} references unknown node {node}"));
            }
        }
        let region_node_set: BTreeSet<&str> = region_nodes.iter().map(String::as_str).collect();
        let region_edges = string_array(value, "edge_ids", &ctx)?;
        for edge_id in &region_edges {
            let edge = edges.get(edge_id)
                .ok_or_else(|| format!("{ctx} references unknown edge {edge_id}"))?;
            if !region_node_set.contains(edge.source.as_str()) || !region_node_set.contains(edge.target.as_str()) {
                return Err(format!("{ctx} edge {edge_id} has an endpoint outside the declared candidate region"));
            }
        }
        let _support_claim_ids = string_array(value, "support_claim_ids", &ctx)?;
        regions.push(Region { id, rank, nodes: region_nodes, edges: region_edges });
    }
    regions.sort_by_key(|r| r.rank);
    Ok(regions)
}

fn evaluate_budget(regions: &[Region], edges: &BTreeMap<String, Edge>, positive: &BTreeSet<String>, unit: &str, limit: usize) -> Value {
    let mut selected_nodes = BTreeSet::new();
    let mut selected_edges = BTreeSet::new();
    let mut selected_regions = Vec::new();
    let mut used = 0usize;

    for region in regions {
        if used >= limit {
            break;
        }
        match unit {
            "regions" => {
                selected_nodes.extend(region.nodes.iter().cloned());
                selected_edges.extend(region.edges.iter().cloned());
                selected_regions.push(region.id.clone());
                used += 1;
            }
            "gates" => {
                let mut added = 0usize;
                for node in &region.nodes {
                    if selected_nodes.insert(node.clone()) {
                        added += 1;
                        if used + added >= limit {
                            break;
                        }
                    }
                }
                if added > 0 {
                    selected_regions.push(region.id.clone());
                    used += added;
                }
                for edge_id in &region.edges {
                    if let Some(edge) = edges.get(edge_id) {
                        if selected_nodes.contains(&edge.source) && selected_nodes.contains(&edge.target) {
                            selected_edges.insert(edge_id.clone());
                        }
                    }
                }
            }
            "edges" => {
                let mut added = 0usize;
                for edge_id in &region.edges {
                    if used + added >= limit {
                        break;
                    }
                    if selected_edges.insert(edge_id.clone()) {
                        added += 1;
                        if let Some(edge) = edges.get(edge_id) {
                            selected_nodes.insert(edge.source.clone());
                            selected_nodes.insert(edge.target.clone());
                        }
                    }
                }
                if added > 0 {
                    selected_regions.push(region.id.clone());
                    used += added;
                }
            }
            _ => unreachable!("review-budget unit validated before evaluation"),
        }
    }

    let mut true_positive = 0usize;
    for node in &selected_nodes {
        if positive.contains(node) {
            true_positive += 1;
        }
    }
    let precision = if selected_nodes.is_empty() { 0.0 } else { true_positive as f64 / selected_nodes.len() as f64 };
    let recall = if positive.is_empty() { 0.0 } else { true_positive as f64 / positive.len() as f64 };

    json!({
        "unit": unit,
        "limit": limit,
        "used": used,
        "selected_region_ids": selected_regions,
        "selected_region_count": selected_regions.len(),
        "selected_node_count": selected_nodes.len(),
        "selected_edge_count": selected_edges.len(),
        "true_positive_nodes": true_positive,
        "false_positive_nodes": selected_nodes.len().saturating_sub(true_positive),
        "positive_oracle_nodes": positive.len(),
        "precision": precision,
        "recall": recall
    })
}

fn audit_values(
    explanation: &Value,
    graph: &Value,
    oracle: &Value,
    explanation_bytes: &[u8],
    graph_bytes: &[u8],
    oracle_bytes: &[u8],
) -> Result<Value, String> {
    if string(explanation, "schema_version", "explanation")? != "hardsec-localization-explanation-v1" {
        return Err("unsupported explanation schema_version".to_owned());
    }
    if string(oracle, "schema_version", "oracle")? != "hardsec-localization-oracle-v1" {
        return Err("unsupported oracle schema_version".to_owned());
    }
    let graph_sha = sha256(graph_bytes);
    let explanation_sha = sha256(explanation_bytes);
    let oracle_sha = sha256(oracle_bytes);
    let source_artifact_sha = string(graph, "source_artifact_sha256", "graph")?;
    valid_sha(source_artifact_sha, "graph.source_artifact_sha256")?;
    let (nodes, edges, node_edge_sha) = load_graph(graph)?;

    let design_id = string(explanation.get("grouping").ok_or("explanation.grouping is required")?, "design_id", "explanation.grouping")?;
    if design_id != string(oracle, "design_id", "oracle")? {
        return Err("explanation and oracle design_id do not match".to_owned());
    }
    let grouping = explanation.get("grouping").ok_or("explanation.grouping is required")?;
    for key in ["dataset_id", "split_protocol", "split", "host_family_id", "design_id", "insertion_source_id", "trojan_family_id", "synthesis_toolchain_id", "cell_library_id"] {
        let _ = string(grouping, key, "explanation.grouping")?;
    }
    let source = explanation.get("source").ok_or("explanation.source is required")?;
    let _ = string(source, "artifact_id", "explanation.source")?;
    let _ = string(source, "source_path", "explanation.source")?;
    require_digest(source, "source_sha256", source_artifact_sha, "explanation.source")?;
    require_digest(source, "node_edge_map_sha256", &node_edge_sha, "explanation.source")?;
    require_digest(oracle, "source_sha256", source_artifact_sha, "oracle")?;
    let model = explanation.get("model").ok_or("explanation.model is required")?;
    for key in ["model_sha256", "config_sha256", "training_lineage_sha256"] {
        let value = string(model, key, "explanation.model")?;
        valid_sha(value, &format!("explanation.model.{key}"))?;
    }
    let explainer = explanation.get("explainer").ok_or("explanation.explainer is required")?;
    let _ = string(explainer, "explainer_id", "explanation.explainer")?;
    let _ = string(explainer, "method", "explanation.explainer")?;
    let _ = string(explainer, "version", "explanation.explainer")?;
    let explainer_config = string(explainer, "config_sha256", "explanation.explainer")?;
    valid_sha(explainer_config, "explanation.explainer.config_sha256")?;
    let decision = explanation.get("decision").ok_or("explanation.decision is required")?;
    let decision_value = string(decision, "disposition", "explanation.decision")?;
    if !matches!(decision_value, "clear" | "suspect" | "abstain") {
        return Err("unsupported explanation.decision.disposition".to_owned());
    }
    let decision_score = decision.get("score").ok_or("explanation.decision.score is required")?;
    if !decision_score.is_null() {
        let score = decision_score.as_f64().ok_or("explanation.decision.score must be number or null")?;
        if !score.is_finite() || !(0.0..=1.0).contains(&score) {
            return Err("explanation.decision.score must be finite and in [0, 1]".to_owned());
        }
    }
    let threshold_digest = string(decision, "threshold_profile_sha256", "explanation.decision")?;
    valid_sha(threshold_digest, "explanation.decision.threshold_profile_sha256")?;

    let label_semantics = string(oracle, "label_semantics", "oracle")?;
    if label_semantics != "insertion_footprint" {
        return Err("this evaluator supports insertion_footprint labels only".to_owned());
    }
    let claim_limits = explanation.get("claim_limits").ok_or("explanation.claim_limits is required")?;
    if claim_limits.get("claims_causal_localization").and_then(Value::as_bool) != Some(false)
        || claim_limits.get("claims_mechanism").and_then(Value::as_bool) != Some(false) {
        return Err("insertion-footprint evaluation cannot certify mechanism or causal localization claims".to_owned());
    }

    let positive_values = string_array(oracle, "positive_node_ids", "oracle")?;
    if positive_values.is_empty() {
        return Err("oracle.positive_node_ids must not be empty".to_owned());
    }
    let positive: BTreeSet<String> = positive_values.into_iter().collect();
    for node in &positive {
        if !nodes.contains(node) {
            return Err(format!("oracle references unknown positive node {node}"));
        }
    }

    let oracle_path_values = oracle.get("reference_paths").and_then(Value::as_array)
        .ok_or_else(|| "oracle.reference_paths must be an array".to_owned())?;
    let mut oracle_paths = BTreeMap::new();
    for (index, value) in oracle_path_values.iter().enumerate() {
        let ctx = format!("oracle.reference_paths[{index}]");
        let path_id = string(value, "path_id", &ctx)?.to_owned();
        let edge_ids = string_array(value, "edge_ids", &ctx)?;
        check_path_edges(&path_id, &edge_ids, &edges, true)?;
        if oracle_paths.insert(path_id.clone(), edge_ids).is_some() {
            return Err(format!("duplicate oracle path_id {path_id}"));
        }
    }
    let reference_paths_complete = oracle.get("reference_paths_complete").and_then(Value::as_bool)
        .ok_or_else(|| "oracle.reference_paths_complete must be boolean".to_owned())?;

    let regions = parse_regions(explanation, &nodes, &edges)?;
    let budget = explanation.get("review_budget").ok_or("explanation.review_budget is required")?;
    let unit = string(budget, "unit", "explanation.review_budget")?;
    if !matches!(unit, "regions" | "gates" | "edges") {
        return Err("unsupported review_budget.unit".to_owned());
    }
    let limit = budget.get("limit").and_then(Value::as_u64)
        .filter(|v| *v > 0 && *v <= usize::MAX as u64)
        .ok_or_else(|| "review_budget.limit must be a positive integer".to_owned())? as usize;

    let mut path_matches = BTreeSet::new();
    let mut path_reports = Vec::new();
    let explanation_paths = explanation.get("paths").and_then(Value::as_array)
        .ok_or_else(|| "explanation.paths must be an array".to_owned())?;
    let mut seen_path_ids = BTreeSet::new();
    for (index, path) in explanation_paths.iter().enumerate() {
        let ctx = format!("explanation.paths[{index}]");
        let path_id = string(path, "path_id", &ctx)?.to_owned();
        if !seen_path_ids.insert(path_id.clone()) {
            return Err(format!("duplicate explanation path_id {path_id}"));
        }
        require_digest(path, "source_sha256", source_artifact_sha, &ctx)?;
        let status = string(path, "status", &ctx)?;
        let edge_ids = string_array(path, "edge_ids", &ctx)?;
        let reason = string(path, "reason", &ctx)?;
        if status != "retained" && reason.trim().is_empty() {
            return Err(format!("{ctx}.reason is required for non-retained path status"));
        }
        let all_edges_present = edge_ids.iter().all(|edge_id| edges.contains_key(edge_id));
        let structurally_continuous = all_edges_present
            && check_path_edges(&path_id, &edge_ids, &edges, false).is_ok();
        let matched = match status {
            "retained" => {
                check_path_edges(&path_id, &edge_ids, &edges, true)?;
                match oracle_paths.get(&path_id) {
                    Some(expected) if expected == &edge_ids => {
                        path_matches.insert(path_id.clone());
                        true
                    }
                    Some(_) => return Err(format!("retained path {path_id} differs from the oracle edge sequence")),
                    None => return Err(format!("retained path {path_id} is absent from the evaluator oracle")),
                }
            }
            "cut_and_reported" => {
                if !structurally_continuous || edge_ids.is_empty() {
                    return Err(format!("cut_and_reported path {path_id} must list a non-empty continuous source-edge segment"));
                }
                false
            }
            "unknown" => false,
            "contradicted" => {
                if let Some(expected) = oracle_paths.get(&path_id) {
                    if expected == &edge_ids {
                        return Err(format!("path {path_id} is marked contradicted but exactly matches the evaluator oracle"));
                    }
                }
                false
            }
            _ => return Err(format!("unsupported path status {status}")),
        };
        path_reports.push(json!({
            "path_id": path_id,
            "status": status,
            "oracle_match": matched,
            "all_source_edges_present": all_edges_present,
            "structurally_continuous": structurally_continuous,
            "edge_count": edge_ids.len(),
            "reason": reason
        }));
    }

    let coverage = explanation.get("path_coverage").ok_or("explanation.path_coverage is required")?;
    let coverage_status = string(coverage, "status", "explanation.path_coverage")?;
    if !matches!(coverage_status, "not_claimed" | "partial" | "complete") {
        return Err("unsupported explanation.path_coverage.status".to_owned());
    }
    if coverage_status == "complete" {
        let oracle_id = string(coverage, "oracle_id", "explanation.path_coverage")?;
        nonempty_id(oracle_id, "explanation.path_coverage.oracle_id")?;
        let certificate = string(coverage, "certificate_sha256", "explanation.path_coverage")?;
        valid_sha(certificate, "explanation.path_coverage.certificate_sha256")?;
        if !reference_paths_complete {
            return Err("complete path coverage cannot be claimed against an oracle path set not declared complete".to_owned());
        }
        if path_matches.len() != oracle_paths.len()
            || oracle_paths.is_empty()
            || explanation_paths.len() != oracle_paths.len()
        {
            return Err("complete path coverage requires exactly the supplied reference paths, all retained and matched".to_owned());
        }
    }
    let reference_coverage = if oracle_paths.is_empty() { 0.0 } else { path_matches.len() as f64 / oracle_paths.len() as f64 };

    let mut budgets = BTreeSet::new();
    for value in [1usize, 5, 10, 25, 50, limit] {
        budgets.insert(value);
    }
    let curve: Vec<Value> = budgets.into_iter()
        .map(|k| evaluate_budget(&regions, &edges, &positive, unit, k))
        .collect();
    let requested_result = evaluate_budget(&regions, &edges, &positive, unit, limit);

    Ok(json!({
        "schema_version": "hardsec-localization-receipt-v1",
        "result": "evaluated",
        "claim_scope": CLAIM_SCOPE,
        "inputs": {
            "explanation_sha256": explanation_sha,
            "source_graph_document_sha256": graph_sha,
            "source_artifact_sha256": source_artifact_sha,
            "oracle_sha256": oracle_sha,
            "node_edge_map_sha256": node_edge_sha,
            "model_sha256": explanation.get("model").and_then(|v| v.get("model_sha256")),
            "explainer_id": explanation.get("explainer").and_then(|v| v.get("explainer_id")),
            "design_id": design_id,
            "label_semantics": label_semantics
        },
        "review_budget_result": requested_result,
        "review_budget_curve": curve,
        "path_audit": {
            "oracle_reference_path_count": oracle_paths.len(),
            "retained_matched_path_count": path_matches.len(),
            "reference_path_coverage": reference_coverage,
            "oracle_declares_path_set_complete": reference_paths_complete,
            "claim_status": coverage_status,
            "paths": path_reports
        },
        "candidate_inventory": {
            "region_count": regions.len(),
            "source_node_count": nodes.len(),
            "source_edge_count": edges.len(),
            "oracle_positive_node_count": positive.len()
        },
        "limitations": [
            "The evaluator verifies supplied insertion-footprint labels only; it does not establish causal localization or circuit mechanism.",
            "The evaluator does not parse RTL or independently validate the source graph extractor.",
            "The evaluator cannot authenticate the evaluator-controlled oracle or prove that its labels are complete.",
            "A path-set completeness flag is caller-declared and requires independent provenance outside this utility.",
            "Synthetic fixtures do not establish hardware performance or industrial generalization."
        ]
    }))
}

fn read_input(path: &Path, label: &str) -> Result<(Vec<u8>, Value), String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|e| format!("cannot stat {label} {}: {e}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!("{label} must be a regular non-symlink file"));
    }
    let bytes = fs::read(path).map_err(|e| format!("cannot read {label}: {e}"))?;
    let value = serde_json::from_slice(&bytes).map_err(|e| format!("invalid {label} JSON: {e}"))?;
    Ok((bytes, value))
}

fn parse_args() -> Result<Args, String> {
    let mut values = BTreeMap::new();
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        if !matches!(flag.as_str(), "--explanation" | "--graph" | "--oracle" | "--receipt") {
            return Err(format!("unknown argument {flag}"));
        }
        let value = args.next().ok_or_else(|| format!("{flag} requires a path"))?;
        if values.insert(flag.clone(), PathBuf::from(value)).is_some() {
            return Err(format!("duplicate argument {flag}"));
        }
    }
    let take = |key: &str| values.get(key).cloned().ok_or_else(|| format!("missing required argument {key}"));
    Ok(Args {
        explanation: take("--explanation")?,
        graph: take("--graph")?,
        oracle: take("--oracle")?,
        receipt: take("--receipt")?,
    })
}

fn validate_receipt_location(receipt: &Path, inputs: &[&Path]) -> Result<(), String> {
    if receipt.exists() {
        return Err(format!("refusing to overwrite receipt {}", receipt.display()));
    }
    let parent = receipt.parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let canonical_parent = fs::canonicalize(parent)
        .map_err(|e| format!("receipt parent must exist: {e}"))?;
    for input in inputs {
        let canonical_input = fs::canonicalize(input)
            .map_err(|e| format!("cannot canonicalize input {}: {e}", input.display()))?;
        let input_parent = canonical_input.parent().ok_or("input has no parent directory")?;
        if canonical_parent.starts_with(input_parent) {
            return Err("receipt must be outside every input directory".to_owned());
        }
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let (explanation_bytes, explanation) = read_input(&args.explanation, "explanation")?;
    let (graph_bytes, graph) = read_input(&args.graph, "graph")?;
    let (oracle_bytes, oracle) = read_input(&args.oracle, "oracle")?;
    validate_receipt_location(&args.receipt, &[&args.explanation, &args.graph, &args.oracle])?;
    let receipt = audit_values(
        &explanation, &graph, &oracle,
        &explanation_bytes, &graph_bytes, &oracle_bytes,
    )?;
    let encoded = serde_json::to_vec_pretty(&receipt).map_err(|e| format!("cannot encode receipt: {e}"))?;
    let mut file = OpenOptions::new().write(true).create_new(true).open(&args.receipt)
        .map_err(|e| format!("cannot create receipt without overwrite: {e}"))?;
    file.write_all(&encoded).map_err(|e| format!("cannot write receipt: {e}"))?;
    file.write_all(b"\n").map_err(|e| format!("cannot finish receipt: {e}"))?;
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("HARDSEC-013 localization audit failed: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_values() -> (Value, Value, Value, Vec<u8>, Vec<u8>, Vec<u8>) {
        let source_artifact_sha = "9".repeat(64);
        let graph = json!({
            "schema_version":"hardsec-source-graph-v1",
            "source_artifact_sha256":source_artifact_sha,
            "nodes":["g17","g18","g20","g99","g100"],
            "edges":[
                {"edge_id":"e17-18","source":"g17","target":"g18"},
                {"edge_id":"e99-100","source":"g99","target":"g100"}
            ]
        });
        let graph_bytes = serde_json::to_vec(&graph).unwrap();
        let (nodes, edges, map_sha) = load_graph(&graph).unwrap();
        let _ = (nodes, edges);
        let explanation = json!({
            "schema_version":"hardsec-localization-explanation-v1",
            "grouping":{
                "dataset_id":"fixture-dataset","split_protocol":"not_applicable","split":"not_applicable",
                "host_family_id":"fixture-host","design_id":"design-alpha-01",
                "insertion_source_id":"fixture-author","trojan_family_id":"fixture-family",
                "synthesis_toolchain_id":"fixture-toolchain","cell_library_id":"fixture-library"
            },
            "source":{
                "artifact_id":"fixture-source-01","source_path":"fixture-source-01.json",
                "source_sha256":source_artifact_sha,
                "node_edge_map_sha256":map_sha
            },
            "model":{
                "model_sha256":"c".repeat(64),"config_sha256":"d".repeat(64),
                "training_lineage_sha256":"e".repeat(64)
            },
            "explainer":{
                "explainer_id":"fixture-domain-property","method":"domain_property",
                "version":"0.1.0","config_sha256":"f".repeat(64)
            },
            "decision":{
                "disposition":"suspect","score":0.87,"threshold_profile_sha256":"1".repeat(64)
            },
            "review_budget":{"unit":"regions","limit":1},
            "candidate_regions":[
                {"region_id":"region-r1","rank":1,"score":0.87,"node_ids":["g17","g18"],"edge_ids":["e17-18"],"support_claim_ids":["fixture-claim-1"]},
                {"region_id":"region-r2","rank":2,"score":0.31,"node_ids":["g99","g100"],"edge_ids":["e99-100"],"support_claim_ids":["fixture-claim-2"]}
            ],
            "paths":[
                {"path_id":"path-trigger-observe-01","status":"retained","source_sha256":source_artifact_sha,"edge_ids":["e17-18"],"reason":""}
            ],
            "path_coverage":{"status":"partial"},
            "claim_limits":{"claims_mechanism":false,"claims_causal_localization":false}
        });
        let oracle = json!({
            "schema_version":"hardsec-localization-oracle-v1",
            "design_id":"design-alpha-01",
            "source_sha256":source_artifact_sha,
            "label_semantics":"insertion_footprint",
            "reference_paths_complete":false,
            "positive_node_ids":["g17","g18","g20"],
            "reference_paths":[
                {"path_id":"path-trigger-observe-01","edge_ids":["e17-18"]}
            ]
        });
        let explanation_bytes = serde_json::to_vec(&explanation).unwrap();
        let oracle_bytes = serde_json::to_vec(&oracle).unwrap();
        (explanation, graph, oracle, explanation_bytes, graph_bytes, oracle_bytes)
    }

    #[test]
    fn known_answer_review_budget_scores_exactly() {
        let (e,g,o,eb,gb,ob) = fixture_values();
        let receipt = audit_values(&e,&g,&o,&eb,&gb,&ob).unwrap();
        assert_eq!(receipt["result"], "evaluated");
        assert_eq!(receipt["review_budget_result"]["selected_node_count"], 2);
        assert_eq!(receipt["review_budget_result"]["true_positive_nodes"], 2);
        assert_eq!(receipt["review_budget_result"]["precision"], 1.0);
        let recall = receipt["review_budget_result"]["recall"].as_f64().unwrap();
        assert!((recall - (2.0/3.0)).abs() < 1e-12);
        assert_eq!(receipt["path_audit"]["retained_matched_path_count"], 1);
    }

    #[test]
    fn rejects_source_digest_mismatch() {
        let (mut e,g,o,eb,gb,ob) = fixture_values();
        e["source"]["source_sha256"] = json!("0".repeat(64));
        let changed = serde_json::to_vec(&e).unwrap();
        assert!(audit_values(&e,&g,&o,&changed,&gb,&ob).unwrap_err().contains("source_sha256"));
        let _ = eb;
    }

    #[test]
    fn rejects_unknown_candidate_node() {
        let (mut e,g,o,_,gb,ob) = fixture_values();
        e["candidate_regions"][0]["node_ids"] = json!(["not-in-source"]);
        let eb = serde_json::to_vec(&e).unwrap();
        assert!(audit_values(&e,&g,&o,&eb,&gb,&ob).unwrap_err().contains("unknown node"));
    }

    #[test]
    fn rejects_path_discontinuity() {
        let (mut e,g,o,_,gb,ob) = fixture_values();
        e["paths"][0]["edge_ids"] = json!(["e17-18","e99-100"]);
        let eb = serde_json::to_vec(&e).unwrap();
        assert!(audit_values(&e,&g,&o,&eb,&gb,&ob).unwrap_err().contains("discontinuous"));
    }

    #[test]
    fn rejects_complete_claim_against_incomplete_oracle_path_set() {
        let (mut e,g,o,_,gb,ob) = fixture_values();
        e["path_coverage"]["status"] = json!("complete");
        let eb = serde_json::to_vec(&e).unwrap();
        assert!(audit_values(&e,&g,&o,&eb,&gb,&ob).unwrap_err().contains("not declared complete"));
    }

    #[test]
    fn rejects_causal_claim_from_insertion_footprint_oracle() {
        let (mut e,g,o,_,gb,ob) = fixture_values();
        e["claim_limits"]["claims_causal_localization"] = json!(true);
        let eb = serde_json::to_vec(&e).unwrap();
        assert!(audit_values(&e,&g,&o,&eb,&gb,&ob).unwrap_err().contains("cannot certify"));
    }

    #[test]
    fn rejects_duplicate_region_rank() {
        let (mut e,g,o,_,gb,ob) = fixture_values();
        e["candidate_regions"][1]["rank"] = json!(1);
        let eb = serde_json::to_vec(&e).unwrap();
        assert!(audit_values(&e,&g,&o,&eb,&gb,&ob).unwrap_err().contains("rank must be positive and unique"));
    }

    #[test]
    fn rejects_unknown_oracle_positive_node() {
        let (mut e,g,mut o,_,gb,_) = fixture_values();
        o["positive_node_ids"] = json!(["not-in-source"]);
        let ob = serde_json::to_vec(&o).unwrap();
        let eb = serde_json::to_vec(&e).unwrap();
        assert!(audit_values(&e,&g,&o,&eb,&gb,&ob).unwrap_err().contains("unknown positive node"));
    }

    #[test]
    fn rejects_retained_path_that_differs_from_oracle() {
        let (mut e,g,mut o,_,gb,_) = fixture_values();
        o["reference_paths"][0]["edge_ids"] = json!(["e99-100"]);
        let ob = serde_json::to_vec(&o).unwrap();
        let eb = serde_json::to_vec(&e).unwrap();
        assert!(audit_values(&e,&g,&o,&eb,&gb,&ob).unwrap_err().contains("differs from the oracle"));
    }

    #[test]
    fn budget_curve_contains_requested_budget() {
        let (e,g,o,eb,gb,ob) = fixture_values();
        let receipt = audit_values(&e,&g,&o,&eb,&gb,&ob).unwrap();
        let curve = receipt["review_budget_curve"].as_array().unwrap();
        assert!(curve.iter().any(|row| row["limit"] == 1));
    }

    #[test]
    fn edge_budget_counts_the_referenced_edge_endpoints() {
        let (mut e,g,o,_,gb,ob) = fixture_values();
        e["review_budget"]["unit"] = json!("edges");
        e["review_budget"]["limit"] = json!(1);
        let eb = serde_json::to_vec(&e).unwrap();
        let receipt = audit_values(&e,&g,&o,&eb,&gb,&ob).unwrap();
        let result = &receipt["review_budget_result"];
        assert_eq!(result["selected_edge_count"], 1);
        assert_eq!(result["selected_node_count"], 2);
        assert_eq!(result["true_positive_nodes"], 2);
    }

    #[test]
    fn unknown_path_can_preserve_missing_edge_uncertainty() {
        let (mut e,g,o,_,gb,ob) = fixture_values();
        e["paths"][0]["status"] = json!("unknown");
        e["paths"][0]["edge_ids"] = json!(["edge-not-observed"]);
        e["paths"][0]["reason"] = json!("Source mapping is incomplete");
        let eb = serde_json::to_vec(&e).unwrap();
        let receipt = audit_values(&e,&g,&o,&eb,&gb,&ob).unwrap();
        assert_eq!(receipt["path_audit"]["retained_matched_path_count"], 0);
        assert_eq!(receipt["path_audit"]["paths"][0]["all_source_edges_present"], false);
    }

    #[test]
    fn complete_claim_passes_only_for_exact_complete_reference_set() {
        let (mut e,g,mut o,_,gb,_) = fixture_values();
        o["reference_paths_complete"] = json!(true);
        e["path_coverage"]["status"] = json!("complete");
        e["path_coverage"]["oracle_id"] = json!("independent-oracle-v1");
        e["path_coverage"]["certificate_sha256"] = json!("9".repeat(64));
        let eb = serde_json::to_vec(&e).unwrap();
        let ob = serde_json::to_vec(&o).unwrap();
        let receipt = audit_values(&e,&g,&o,&eb,&gb,&ob).unwrap();
        assert_eq!(receipt["path_audit"]["reference_path_coverage"], 1.0);
    }

    #[test]
    fn rejects_candidate_region_edge_with_endpoint_outside_region() {
        let (mut e,g,o,_,gb,ob) = fixture_values();
        e["candidate_regions"][0]["node_ids"] = json!(["g17"]);
        let eb = serde_json::to_vec(&e).unwrap();
        assert!(audit_values(&e,&g,&o,&eb,&gb,&ob).unwrap_err().contains("endpoint outside"));
    }

    #[test]
    fn rejects_malformed_model_digest() {
        let (mut e,g,o,_,gb,ob) = fixture_values();
        e["model"]["model_sha256"] = json!("not-a-digest");
        let eb = serde_json::to_vec(&e).unwrap();
        assert!(audit_values(&e,&g,&o,&eb,&gb,&ob).unwrap_err().contains("lowercase SHA-256"));
    }

    #[test]
    fn rejects_out_of_range_decision_score() {
        let (mut e,g,o,_,gb,ob) = fixture_values();
        e["decision"]["score"] = json!(1.5);
        let eb = serde_json::to_vec(&e).unwrap();
        assert!(audit_values(&e,&g,&o,&eb,&gb,&ob).unwrap_err().contains("finite and in [0, 1]"));
    }

    #[test]
    fn rejects_unsupported_path_coverage_state() {
        let (mut e,g,o,_,gb,ob) = fixture_values();
        e["path_coverage"]["status"] = json!("probably_complete");
        let eb = serde_json::to_vec(&e).unwrap();
        assert!(audit_values(&e,&g,&o,&eb,&gb,&ob).unwrap_err().contains("unsupported explanation.path_coverage.status"));
    }

    #[test]
    fn empty_reference_path_set_does_not_divide_by_zero() {
        let (mut e,g,mut o,_,gb,_) = fixture_values();
        o["reference_paths"] = json!([]);
        o["reference_paths_complete"] = json!(false);
        e["paths"] = json!([]);
        let eb = serde_json::to_vec(&e).unwrap();
        let ob = serde_json::to_vec(&o).unwrap();
        let receipt = audit_values(&e,&g,&o,&eb,&gb,&ob).unwrap();
        assert_eq!(receipt["path_audit"]["reference_path_coverage"], 0.0);
    }
}
