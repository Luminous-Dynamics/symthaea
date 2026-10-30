//! Audits whether directed critical paths survive graph partitioning.
//!
//! The fixture uses generic directed graphs. A hierarchical boundary relation
//! counts only when it references the exact original edge ID in the exact
//! ordered path. This is a representation-retention audit, not a security proof.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(Debug, Clone)]
struct Node {
    id: String,
    partition: String,
}

#[derive(Debug, Clone)]
struct Edge {
    id: String,
    from: String,
    to: String,
}

#[derive(Debug, Clone)]
struct CriticalPath {
    id: String,
    source: String,
    target: String,
    edge_ids: Vec<String>,
}

#[derive(Debug, Clone)]
struct Args {
    spec: PathBuf,
    receipt: PathBuf,
}

#[derive(Debug, Clone)]
struct PathResult {
    id: String,
    edge_count: usize,
    boundary_edge_count: usize,
    all_endpoints_retained: bool,
    local_only_preserved: bool,
    hierarchical_preserved: bool,
}

#[derive(Debug, Clone)]
struct Summary {
    spec_sha256: String,
    fixture_id: String,
    node_count: usize,
    edge_count: usize,
    retained_node_count: usize,
    retained_local_edge_count: usize,
    boundary_relation_count: usize,
    paths: Vec<PathResult>,
}

fn parse_args() -> Result<Option<Args>, String> {
    let mut spec = None;
    let mut receipt = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            return Ok(None);
        }
        let value = args.next().ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--spec" => spec = Some(PathBuf::from(value)),
            "--receipt" => receipt = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown argument: {flag}")),
        }
    }
    Ok(Some(Args {
        spec: spec.ok_or("--spec is required")?,
        receipt: receipt.ok_or("--receipt is required")?,
    }))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn required_string<'a>(value: &'a Value, field: &str) -> Result<&'a str, String> {
    let text = value
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{field} must be a string"))?;
    if text.trim().is_empty() || text.trim() != text {
        return Err(format!("{field} must be non-empty and have no surrounding whitespace"));
    }
    Ok(text)
}

fn string_array(value: &Value, field: &str) -> Result<Vec<String>, String> {
    let items = value
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{field} must be an array"))?;
    let mut result = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        let text = item
            .as_str()
            .ok_or_else(|| format!("{field}[{index}] must be a string"))?;
        if text.trim().is_empty() || text.trim() != text {
            return Err(format!("{field}[{index}] must be a non-empty string"));
        }
        result.push(text.to_owned());
    }
    Ok(result)
}

fn unique_strings(value: &Value, field: &str) -> Result<BTreeSet<String>, String> {
    let values = string_array(value, field)?;
    let mut result = BTreeSet::new();
    for item in values {
        if !result.insert(item.clone()) {
            return Err(format!("{field} contains duplicate item {item}"));
        }
    }
    Ok(result)
}

fn parse_nodes(value: &Value) -> Result<BTreeMap<String, Node>, String> {
    let raw = value
        .get("nodes")
        .and_then(Value::as_array)
        .ok_or("nodes must be an array")?;
    if raw.is_empty() {
        return Err("nodes must not be empty".to_owned());
    }
    let mut nodes = BTreeMap::new();
    for item in raw {
        let id = required_string(item, "node_id")?.to_owned();
        let partition = required_string(item, "partition_id")?.to_owned();
        if nodes
            .insert(
                id.clone(),
                Node {
                    id: id.clone(),
                    partition,
                },
            )
            .is_some()
        {
            return Err(format!("duplicate node_id {id}"));
        }
    }
    Ok(nodes)
}

fn parse_edges(value: &Value, nodes: &BTreeMap<String, Node>) -> Result<BTreeMap<String, Edge>, String> {
    let raw = value
        .get("edges")
        .and_then(Value::as_array)
        .ok_or("edges must be an array")?;
    if raw.is_empty() {
        return Err("edges must not be empty".to_owned());
    }
    let mut edges = BTreeMap::new();
    for item in raw {
        let id = required_string(item, "edge_id")?.to_owned();
        let from = required_string(item, "from")?.to_owned();
        let to = required_string(item, "to")?.to_owned();
        if !nodes.contains_key(&from) {
            return Err(format!("edge {id}: unknown source node {from}"));
        }
        if !nodes.contains_key(&to) {
            return Err(format!("edge {id}: unknown destination node {to}"));
        }
        if edges
            .insert(
                id.clone(),
                Edge {
                    id: id.clone(),
                    from,
                    to,
                },
            )
            .is_some()
        {
            return Err(format!("duplicate edge_id {id}"));
        }
    }
    Ok(edges)
}

fn parse_paths(value: &Value, nodes: &BTreeMap<String, Node>, edges: &BTreeMap<String, Edge>) -> Result<Vec<CriticalPath>, String> {
    let raw = value
        .get("critical_paths")
        .and_then(Value::as_array)
        .ok_or("critical_paths must be an array")?;
    if raw.is_empty() {
        return Err("critical_paths must not be empty".to_owned());
    }
    let mut paths = Vec::with_capacity(raw.len());
    let mut ids = BTreeSet::new();
    for item in raw {
        let id = required_string(item, "path_id")?.to_owned();
        if !ids.insert(id.clone()) {
            return Err(format!("duplicate path_id {id}"));
        }
        let source = required_string(item, "source")?.to_owned();
        let target = required_string(item, "target")?.to_owned();
        if !nodes.contains_key(&source) || !nodes.contains_key(&target) {
            return Err(format!("path {id}: source or target node is unknown"));
        }
        let edge_ids = string_array(item, "edge_ids")?;
        if edge_ids.is_empty() {
            return Err(format!("path {id}: edge_ids must not be empty"));
        }

        let mut current = source.clone();
        for (index, edge_id) in edge_ids.iter().enumerate() {
            let edge = edges
                .get(edge_id)
                .ok_or_else(|| format!("path {id} step {index}: unknown edge {edge_id}"))?;
            if edge.from != current {
                return Err(format!(
                    "path {id} step {index}: edge {edge_id} starts at {}, expected {current}",
                    edge.from
                ));
            }
            current = edge.to.clone();
        }
        if current != target {
            return Err(format!(
                "path {id}: final edge ends at {current}, expected target {target}"
            ));
        }

        paths.push(CriticalPath {
            id,
            source,
            target,
            edge_ids,
        });
    }
    paths.sort_by(|left, right| left.id.cmp(&right.id));
    Ok(paths)
}

fn validate_spec(value: &Value, spec_sha256: String) -> Result<Summary, String> {
    if required_string(value, "schema_version")? != "hardsec-path-retention-v1" {
        return Err("unsupported schema_version".to_owned());
    }
    let fixture_id = required_string(value, "fixture_id")?.to_owned();
    let nodes = parse_nodes(value)?;
    let edges = parse_edges(value, &nodes)?;
    let paths = parse_paths(value, &nodes, &edges)?;

    let retained_node_ids = unique_strings(value, "retained_node_ids")?;
    for node_id in &retained_node_ids {
        if !nodes.contains_key(node_id) {
            return Err(format!("retained_node_ids references unknown node {node_id}"));
        }
    }

    let retained_edge_ids = unique_strings(value, "retained_edge_ids")?;
    let boundary_edge_ids = unique_strings(value, "boundary_edge_ids")?;
    if !retained_edge_ids.is_disjoint(&boundary_edge_ids) {
        return Err("retained_edge_ids and boundary_edge_ids must be disjoint".to_owned());
    }
    for edge_id in &retained_edge_ids {
        let edge = edges
            .get(edge_id)
            .ok_or_else(|| format!("retained_edge_ids references unknown edge {edge_id}"))?;
        let from = &nodes[&edge.from];
        let to = &nodes[&edge.to];
        if from.partition != to.partition {
            return Err(format!(
                "retained local edge {edge_id} crosses partitions; represent it as a boundary relation"
            ));
        }
        if !retained_node_ids.contains(&edge.from) || !retained_node_ids.contains(&edge.to) {
            return Err(format!("retained edge {edge_id} has an endpoint absent from retained_node_ids"));
        }
    }
    for edge_id in &boundary_edge_ids {
        let edge = edges
            .get(edge_id)
            .ok_or_else(|| format!("boundary_edge_ids references unknown edge {edge_id}"))?;
        let from = &nodes[&edge.from];
        let to = &nodes[&edge.to];
        if from.partition == to.partition {
            return Err(format!(
                "boundary relation {edge_id} is intra-partition; it must represent a cross-partition edge"
            ));
        }
        if !retained_node_ids.contains(&edge.from) || !retained_node_ids.contains(&edge.to) {
            return Err(format!("boundary relation {edge_id} has an endpoint absent from retained_node_ids"));
        }
    }

    let mut results = Vec::with_capacity(paths.len());
    for path in &paths {
        let mut boundary_edge_count = 0;
        let mut local_only_preserved = true;
        let mut hierarchical_preserved = true;
        let all_endpoints_retained = retained_node_ids.contains(&path.source)
            && retained_node_ids.contains(&path.target);

        for edge_id in &path.edge_ids {
            if retained_edge_ids.contains(edge_id) {
                continue;
            }
            local_only_preserved = false;
            if boundary_edge_ids.contains(edge_id) {
                boundary_edge_count += 1;
            } else {
                hierarchical_preserved = false;
            }
        }

        results.push(PathResult {
            id: path.id.clone(),
            edge_count: path.edge_ids.len(),
            boundary_edge_count,
            all_endpoints_retained,
            local_only_preserved: local_only_preserved && all_endpoints_retained,
            hierarchical_preserved: hierarchical_preserved && all_endpoints_retained,
        });
    }

    Ok(Summary {
        spec_sha256,
        fixture_id,
        node_count: nodes.len(),
        edge_count: edges.len(),
        retained_node_count: retained_node_ids.len(),
        retained_local_edge_count: retained_edge_ids.len(),
        boundary_relation_count: boundary_edge_ids.len(),
        paths: results,
    })
}

fn fraction(numerator: usize, denominator: usize) -> f64 {
    numerator as f64 / denominator as f64
}

fn receipt(summary: &Summary) -> Value {
    let local_preserved = summary
        .paths
        .iter()
        .filter(|path| path.local_only_preserved)
        .count();
    let hierarchical_preserved = summary
        .paths
        .iter()
        .filter(|path| path.hierarchical_preserved)
        .count();
    let endpoints_retained = summary
        .paths
        .iter()
        .filter(|path| path.all_endpoints_retained)
        .count();

    let paths = summary
        .paths
        .iter()
        .map(|path| {
            json!({
                "path_id": path.id,
                "edge_count": path.edge_count,
                "boundary_edge_count": path.boundary_edge_count,
                "all_endpoints_retained": path.all_endpoints_retained,
                "local_only_preserved": path.local_only_preserved,
                "hierarchical_preserved": path.hierarchical_preserved,
            })
        })
        .collect::<Vec<_>>();

    json!({
        "schema_version": "hardsec-path-retention-receipt-v1",
        "result": "pass",
        "claim_scope": "exact_declared_critical_path_edge_retention_in_finite_directed_graph",
        "fixture_id": summary.fixture_id,
        "spec_sha256": summary.spec_sha256,
        "node_count": summary.node_count,
        "edge_count": summary.edge_count,
        "retained_node_count": summary.retained_node_count,
        "retained_local_edge_count": summary.retained_local_edge_count,
        "boundary_relation_count": summary.boundary_relation_count,
        "critical_path_count": summary.paths.len(),
        "endpoint_retained_path_count": endpoints_retained,
        "local_only_preserved_path_count": local_preserved,
        "local_only_path_recall": fraction(local_preserved, summary.paths.len()),
        "hierarchical_preserved_path_count": hierarchical_preserved,
        "hierarchical_path_recall": fraction(hierarchical_preserved, summary.paths.len()),
        "paths": paths,
        "security_verdict": null,
        "limitations": [
            "A path is preserved only when every ordered reference edge is retained locally or by exact boundary-edge identity.",
            "Endpoint retention alone does not count as path retention.",
            "This fixture is a finite graph abstraction, not a hardware trigger witness or a detector result.",
            "No claim is made about graph-scale throughput or industrial netlist capacity."
        ]
    })
}

fn main_run(args: Args) -> Result<(), String> {
    let metadata = fs::symlink_metadata(&args.spec)
        .map_err(|error| format!("stat spec {}: {error}", args.spec.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err("spec must be a regular non-symlink file".to_owned());
    }
    let bytes = fs::read(&args.spec)
        .map_err(|error| format!("read spec {}: {error}", args.spec.display()))?;
    let spec_sha256 = format!("{:x}", Sha256::digest(&bytes));
    let value: Value =
        serde_json::from_slice(&bytes).map_err(|error| format!("parse spec JSON: {error}"))?;
    let summary = validate_spec(&value, spec_sha256)?;
    let receipt_value = receipt(&summary);

    let parent = args
        .receipt
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let parent = fs::canonicalize(parent)
        .map_err(|error| format!("canonicalize receipt directory: {error}"))?;
    let filename = args
        .receipt
        .file_name()
        .ok_or("receipt path must include a filename")?;
    let output_path = parent.join(filename);
    let serialized = serde_json::to_vec_pretty(&receipt_value)
        .map_err(|error| format!("serialize receipt: {error}"))?;

    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output_path)
        .map_err(|error| format!("create receipt {}: {error}", output_path.display()))?;
    let result = output
        .write_all(&serialized)
        .and_then(|_| output.write_all(b"\n"))
        .and_then(|_| output.sync_all());
    drop(output);
    if let Err(error) = result {
        let _ = fs::remove_file(&output_path);
        return Err(format!("write receipt {}: {error}", output_path.display()));
    }

    println!("HARDSEC path-retention audit: PASS");
    println!("fixture: {}", summary.fixture_id);
    println!(
        "paths local/hierarchical: {}/{} / {}/{}",
        summary.paths.iter().filter(|path| path.local_only_preserved).count(),
        summary.paths.len(),
        summary.paths.iter().filter(|path| path.hierarchical_preserved).count(),
        summary.paths.len()
    );
    println!("receipt: {}", output_path.display());
    Ok(())
}

fn parse_args() -> Result<Option<Args>, String> {
    let mut spec = None;
    let mut receipt = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            return Ok(None);
        }
        let value = args.next().ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--spec" => spec = Some(PathBuf::from(value)),
            "--receipt" => receipt = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown argument: {flag}")),
        }
    }
    Ok(Some(Args {
        spec: spec.ok_or("--spec is required")?,
        receipt: receipt.ok_or("--receipt is required")?,
    }))
}

fn main() -> ExitCode {
    match parse_args() {
        Ok(Some(args)) => match main_run(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("HARDSEC path-retention audit failed: {error}");
                ExitCode::FAILURE
            }
        },
        Ok(None) => {
            println!("usage: hardsec_path_retention_audit --spec FILE --receipt NEW_FILE");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("HARDSEC path-retention audit failed: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Value {
        serde_json::from_str(include_str!("../tests/fixtures/hardsec/path-retention-boundary-graph.json"))
            .expect("valid path-retention fixture JSON")
    }

    #[test]
    fn local_only_partitioning_severs_long_multi_boundary_path() {
        let value = fixture();
        let summary = validate_spec(&value, "0".repeat(64)).unwrap();
        assert_eq!(summary.paths.len(), 2);
        assert!(summary.paths[0].local_only_preserved);
        assert!(!summary.paths[1].local_only_preserved);
        assert!(summary.paths[1].all_endpoints_retained);
    }

    #[test]
    fn exact_boundary_edge_index_recovers_both_declared_paths() {
        let summary = validate_spec(&fixture(), "0".repeat(64)).unwrap();
        assert!(summary.paths.iter().all(|path| path.hierarchical_preserved));
        let result = receipt(&summary);
        assert_eq!(result["local_only_path_recall"], json!(0.5));
        assert_eq!(result["hierarchical_path_recall"], json!(1.0));
    }

    #[test]
    fn reports_boundary_relation_cost_separately_from_local_edges() {
        let summary = validate_spec(&fixture(), "0".repeat(64)).unwrap();
        assert_eq!(summary.retained_local_edge_count, 4);
        assert_eq!(summary.boundary_relation_count, 2);
    }

    #[test]
    fn rejects_unknown_retained_node() {
        let mut value = fixture();
        value["retained_node_ids"].as_array_mut().unwrap().push(json!("missing"));
        assert!(
            validate_spec(&value, "0".repeat(64))
                .unwrap_err()
                .contains("unknown node")
        );
    }

    #[test]
    fn rejects_unknown_retained_edge() {
        let mut value = fixture();
        value["retained_edge_ids"].as_array_mut().unwrap().push(json!("missing"));
        assert!(
            validate_spec(&value, "0".repeat(64))
                .unwrap_err()
                .contains("unknown edge")
        );
    }

    #[test]
    fn rejects_cross_partition_edge_in_local_edge_set() {
        let mut value = fixture();
        value["retained_edge_ids"].as_array_mut().unwrap().push(json!("e_bc"));
        assert!(
            validate_spec(&value, "0".repeat(64))
                .unwrap_err()
                .contains("crosses partitions")
        );
    }

    #[test]
    fn rejects_intra_partition_edge_falsely_claimed_as_boundary() {
        let mut value = fixture();
        value["boundary_edge_ids"] = json!(["e_ab"]);
        assert!(
            validate_spec(&value, "0".repeat(64))
                .unwrap_err()
                .contains("must represent a cross-partition edge")
        );
    }

    #[test]
    fn rejects_overlapping_local_and_boundary_edge_sets() {
        let mut value = fixture();
        value["boundary_edge_ids"].as_array_mut().unwrap().push(json!("e_ab"));
        assert!(
            validate_spec(&value, "0".repeat(64))
                .unwrap_err()
                .contains("must be disjoint")
        );
    }

    #[test]
    fn rejects_path_with_discontinuous_edge_order() {
        let mut value = fixture();
        value["critical_paths"][1]["edge_ids"] = json!(["e_ab", "e_de", "e_ef"]);
        assert!(
            validate_spec(&value, "0".repeat(64))
                .unwrap_err()
                .contains("starts at")
        );
    }

    #[test]
    fn rejects_path_with_wrong_terminal_target() {
        let mut value = fixture();
        value["critical_paths"][1]["target"] = json!("d");
        assert!(
            validate_spec(&value, "0".repeat(64))
                .unwrap_err()
                .contains("expected target")
        );
    }

    #[test]
    fn missing_boundary_relation_fails_hierarchical_path_retention() {
        let mut value = fixture();
        value["boundary_edge_ids"] = json!(["e_bc"]);
        let summary = validate_spec(&value, "0".repeat(64)).unwrap();
        assert!(!summary.paths[1].hierarchical_preserved);
        assert_eq!(receipt(&summary)["hierarchical_path_recall"], json!(0.5));
    }

    #[test]
    fn spurious_alternate_edge_does_not_preserve_declared_ordered_path() {
        let mut value = fixture();
        value["boundary_edge_ids"] = json!(["e_bc", "e_de"]);
        value["edges"].as_array_mut().unwrap().push(json!({
            "edge_id": "spurious",
            "from": "b",
            "to": "e"
        }));
        let summary = validate_spec(&value, "0".repeat(64)).unwrap();
        assert!(!summary.paths[1].local_only_preserved);
        assert!(summary.paths[1].hierarchical_preserved);
    }

    #[test]
    fn deterministic_receipt_for_same_fixture() {
        let a = validate_spec(&fixture(), "0".repeat(64)).unwrap();
        let b = validate_spec(&fixture(), "0".repeat(64)).unwrap();
        assert_eq!(receipt(&a), receipt(&b));
    }
}
