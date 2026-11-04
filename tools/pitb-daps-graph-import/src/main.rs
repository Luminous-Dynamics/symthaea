use serde::Serialize;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    io::Write,
    path::{Path, PathBuf},
    process::ExitCode,
};

const GRAPH_SCHEMA: &str = "pitb.proof-graph-evidence.v0";
const SNAPSHOT_SCHEMA: &str = "pitb.daps-import-summary.v0";

#[derive(Debug, Clone)]
struct ProvenanceArgs {
    daps_repository_commit: String,
    source_graph_blob_sha1: String,
    extractor_blob_sha1: String,
    lake_manifest_blob_sha1: String,
    mathlib_commit: String,
    lean_toolchain: String,
}

#[derive(Debug, Serialize)]
struct Graph {
    schema_version: String,
    evidence_class: String,
    snapshot: Snapshot,
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    artifacts: Vec<Artifact>,
    upstream_provenance: Value,
}

#[derive(Debug, Serialize)]
struct Snapshot {
    snapshot_id: String,
    repository_commit: String,
    environment_sha256: String,
    extractor_id: String,
    extractor_version: String,
    extractor_config_sha256: String,
}

#[derive(Debug, Serialize)]
struct Node {
    node_id: String,
    kind: String,
    label: String,
    statement_sha256: String,
    statement_artifact_id: String,
    source_locator: String,
    formal_name: Value,
    upstream_metadata: Value,
}

#[derive(Debug, Serialize)]
struct Edge {
    edge_id: String,
    source_node_id: String,
    target_node_id: String,
    relation: String,
    evidence_class: String,
    status: String,
    evidence_artifact_id: String,
    confidence: Value,
    review_artifact_id: Value,
    reviewer_key_id: Value,
}

#[derive(Debug, Serialize)]
struct Artifact {
    artifact_id: String,
    kind: String,
    path: String,
    sha256: String,
}

#[derive(Debug)]
struct Conversion {
    graph: Graph,
    evidence_files: Vec<(String, Vec<u8>)>,
    summary: Value,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn valid_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn require_sha1(value: &str, label: &str) -> Result<(), String> {
    if valid_hex(value, 40) {
        Ok(())
    } else {
        Err(format!("{label} must be a 40-character lowercase Git SHA-1"))
    }
}

fn required_string<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| format!("{context}.{key} must be a non-empty string"))
}

fn optional_string(value: &Value, key: &str) -> Option<String> {
    value.get(key).and_then(Value::as_str).map(str::to_owned)
}

fn bool_field(value: &Value, key: &str, default: bool) -> bool {
    value.get(key).and_then(Value::as_bool).unwrap_or(default)
}

fn json_bytes(value: &Value) -> Result<Vec<u8>, String> {
    serde_json::to_vec(value).map_err(|e| format!("serialize evidence JSON: {e}"))
}

fn node_id(name: &str) -> String {
    format!("daps-node-{}", &sha256(name.as_bytes())[..32])
}

fn edge_id(index: usize, bytes: &[u8]) -> String {
    format!("daps-edge-{index:06}-{}", &sha256(bytes)[..24])
}

fn clean_path_component(value: &str) -> String {
    value
        .bytes()
        .map(|b| if b.is_ascii_alphanumeric() || b == b'-' || b == b'_' { b as char } else { '_' })
        .collect()
}

fn source_locator(node: &Value) -> String {
    let path = optional_string(node, "source_path").unwrap_or_default();
    let line = node.get("lean4_start").and_then(Value::as_u64)
        .or_else(|| node.get("line").and_then(Value::as_u64))
        .unwrap_or(0);
    let end = node.get("lean4_end").and_then(Value::as_u64).unwrap_or(line);
    if !path.is_empty() && line > 0 {
        format!("{path}:{line}-{end}")
    } else if !path.is_empty() {
        path
    } else {
        optional_string(node, "module").unwrap_or_else(|| "daps-export:location-unavailable".into())
    }
}

fn statement_text(node: &Value, name: &str, is_file: bool) -> String {
    if is_file {
        return optional_string(node, "statement")
            .or_else(|| optional_string(node, "module"))
            .unwrap_or_else(|| name.to_owned());
    }
    optional_string(node, "statement")
        .filter(|s| !s.trim().is_empty())
        .or_else(|| optional_string(node, "signature").filter(|s| !s.trim().is_empty()))
        .unwrap_or_else(|| name.to_owned())
}

fn canonical_environment(args: &ProvenanceArgs) -> Result<String, String> {
    let value = json!({
        "mathlib_commit": args.mathlib_commit,
        "lake_manifest_blob_sha1": args.lake_manifest_blob_sha1,
        "lean_toolchain": args.lean_toolchain
    });
    Ok(sha256(&json_bytes(&value)?))
}

fn canonical_extractor_config(args: &ProvenanceArgs, input_sha256: &str, metadata: &Value) -> Result<String, String> {
    let value = json!({
        "upstream_repository": "DAPS-ETH/DAPS",
        "upstream_repository_commit": args.daps_repository_commit,
        "source_graph_blob_sha1": args.source_graph_blob_sha1,
        "source_graph_input_sha256": input_sha256,
        "extractor_path": "build_mathlib_dataset/LeanScripts/ExtractConstants.lean",
        "extractor_blob_sha1": args.extractor_blob_sha1,
        "mathlib_commit": args.mathlib_commit,
        "lake_manifest_blob_sha1": args.lake_manifest_blob_sha1,
        "lean_toolchain": args.lean_toolchain,
        "upstream_metadata": metadata
    });
    Ok(sha256(&json_bytes(&value)?))
}

fn upstream_node_metadata(raw: &Value) -> Value {
    let keys = [
        "category", "subcategory", "module", "source_path", "line", "column",
        "lean4_start", "lean4_end", "isProp", "isHumanTheorem", "signature",
        "doc", "proof", "in_blacklist", "lean4_code",
    ];
    let mut map = Map::new();
    for key in keys {
        if let Some(value) = raw.get(key) {
            map.insert(key.to_owned(), value.clone());
        }
    }
    Value::Object(map)
}

fn edge_relation(raw: &Value, source_kind: &str, target_kind: &str) -> Result<&'static str, String> {
    let upstream_type = required_string(raw, "type", "edge")?;
    if upstream_type == "CONTAINS" {
        if source_kind == "module" && target_kind == "formal_declaration" {
            return Ok("daps_membership_candidate");
        }
        return Err(format!(
            "DAPS CONTAINS edge has unsupported orientation/kinds: {source_kind} -> {target_kind}"
        ));
    }
    if source_kind == "module" && target_kind == "module" {
        return Ok("daps_import_candidate");
    }
    if source_kind == "formal_declaration" && target_kind == "formal_declaration" {
        return Ok("daps_dependency_candidate");
    }
    Err(format!(
        "DAPS edge type {upstream_type} joins unsupported node kinds {source_kind} -> {target_kind}; refusing to infer a semantic edge"
    ))
}

fn convert(input: &Value, input_bytes: &[u8], args: &ProvenanceArgs) -> Result<Conversion, String> {
    require_sha1(&args.daps_repository_commit, "DAPS repository commit")?;
    require_sha1(&args.source_graph_blob_sha1, "source graph Git blob ID")?;
    require_sha1(&args.extractor_blob_sha1, "extractor Git blob ID")?;
    require_sha1(&args.lake_manifest_blob_sha1, "Lake manifest Git blob ID")?;
    require_sha1(&args.mathlib_commit, "Mathlib commit")?;
    if args.lean_toolchain.trim().is_empty() {
        return Err("Lean toolchain must be non-empty".into());
    }

    let upstream_nodes = input.get("nodes").and_then(Value::as_array)
        .ok_or("DAPS graph.nodes must be an array")?;
    let upstream_edges = input.get("edges").and_then(Value::as_array)
        .ok_or("DAPS graph.edges must be an array")?;
    let metadata = input.get("metadata").cloned().unwrap_or_else(|| json!({}));
    if !metadata.is_object() {
        return Err("DAPS graph.metadata must be an object when present".into());
    }
    if upstream_nodes.is_empty() {
        return Err("DAPS graph contains no nodes".into());
    }

    let input_digest = sha256(input_bytes);
    let env_digest = canonical_environment(args)?;
    let extractor_config_digest = canonical_extractor_config(args, &input_digest, &metadata)?;
    let mut nodes = Vec::<Node>::new();
    let mut artifacts = Vec::<Artifact>::new();
    let mut evidence_files = Vec::<(String, Vec<u8>)>::new();
    let mut name_to_id = BTreeMap::<String, String>::new();
    let mut name_to_kind = BTreeMap::<String, String>::new();
    let mut seen_names = BTreeSet::<String>::new();

    for (index, raw) in upstream_nodes.iter().enumerate() {
        let name = required_string(raw, "name", &format!("nodes[{index}]"))?;
        if !seen_names.insert(name.to_owned()) {
            return Err(format!("duplicate DAPS node name {name}"));
        }
        let category = optional_string(raw, "category").unwrap_or_else(|| "unknown".into());
        let is_file = category == "file";
        let kind = if is_file { "module" } else { "formal_declaration" };
        let id = node_id(name);
        let statement = statement_text(raw, name, is_file);
        let statement_bytes = statement.as_bytes().to_vec();
        let statement_digest = sha256(&statement_bytes);
        let statement_artifact_id = format!("daps-statement-{}", &sha256(name.as_bytes())[..24]);
        let statement_path = format!("statements/{}.txt", clean_path_component(&id));
        evidence_files.push((statement_path.clone(), statement_bytes));
        artifacts.push(Artifact {
            artifact_id: statement_artifact_id.clone(),
            kind: "statement".into(),
            path: statement_path,
            sha256: statement_digest.clone(),
        });

        let formal_name = if is_file { Value::Null } else { Value::String(name.to_owned()) };
        nodes.push(Node {
            node_id: id.clone(),
            kind: kind.into(),
            label: name.to_owned(),
            statement_sha256: statement_digest,
            statement_artifact_id,
            source_locator: source_locator(raw),
            formal_name,
            upstream_metadata: upstream_node_metadata(raw),
        });
        name_to_id.insert(name.to_owned(), id);
        name_to_kind.insert(name.to_owned(), kind.to_owned());
    }

    let mut relation_counts = BTreeMap::<String, usize>::new();
    for (index, raw) in upstream_edges.iter().enumerate() {
        let source_name = required_string(raw, "source_name", &format!("edges[{index}]"))?;
        let target_name = required_string(raw, "target_name", &format!("edges[{index}]"))?;
        let source = name_to_id.get(source_name)
            .ok_or_else(|| format!("DAPS edge {index} references unknown source {source_name}"))?;
        let target = name_to_id.get(target_name)
            .ok_or_else(|| format!("DAPS edge {index} references unknown target {target_name}"))?;
        let source_kind = name_to_kind.get(source_name).expect("node kind map synchronized");
        let target_kind = name_to_kind.get(target_name).expect("node kind map synchronized");
        let relation = edge_relation(raw, source_kind, target_kind)?;
        let raw_bytes = json_bytes(raw)?;
        let id = edge_id(index, &raw_bytes);
        let evidence_path = format!("edges/{id}.json");
        let evidence_id = format!("daps-evidence-{}", &sha256(&raw_bytes)[..24]);
        let evidence_record = json!({
            "upstream": "DAPS-ETH/DAPS",
            "upstream_edge_index": index,
            "upstream_edge": raw,
            "interpretation": "candidate-only export relation; not a typed Lean proof dependency"
        });
        let evidence_bytes = json_bytes(&evidence_record)?;
        let evidence_digest = sha256(&evidence_bytes);
        evidence_files.push((evidence_path.clone(), evidence_bytes));
        artifacts.push(Artifact {
            artifact_id: evidence_id.clone(),
            kind: "edge_evidence".into(),
            path: evidence_path,
            sha256: evidence_digest,
        });

        *relation_counts.entry(relation.to_owned()).or_default() += 1;
        nodes_edge_placeholder();
        // Every DAPS edge remains a candidate. Its exported relation labels
        // are coarse target-category labels, not a claim about Lean Expr roles.
        // DAPS does not export calibrated edge confidence, so null is retained.
        // No edge is eligible for scheduler admission.
        let edge = Edge {
            edge_id: id,
            source_node_id: source.clone(),
            target_node_id: target.clone(),
            relation: relation.into(),
            evidence_class: "daps_graph_export".into(),
            status: "candidate".into(),
            evidence_artifact_id: evidence_id,
            confidence: Value::Null,
            review_artifact_id: Value::Null,
            reviewer_key_id: Value::Null,
        };
        // Store in the local vector after all per-edge provenance is complete.
        EDGES.push(edge);
    }

    let upstream_provenance = json!({
        "upstream": "DAPS-ETH/DAPS",
        "upstream_repository_commit": args.daps_repository_commit,
        "source_graph_blob_sha1": args.source_graph_blob_sha1,
        "source_graph_input_sha256": input_digest,
        "extractor_path": "build_mathlib_dataset/LeanScripts/ExtractConstants.lean",
        "extractor_blob_sha1": args.extractor_blob_sha1,
        "lake_manifest_blob_sha1": args.lake_manifest_blob_sha1,
        "mathlib_commit": args.mathlib_commit,
        "lean_toolchain": args.lean_toolchain,
        "upstream_graph_metadata": metadata,
        "trust_boundary": "all DAPS dependency and membership edges are candidate retrieval hints; labels do not distinguish type, proof-term, definition-body or instance semantics"
    });
    let snapshot = Snapshot {
        snapshot_id: format!("daps-{}-{}", &input_digest[..12], &args.mathlib_commit[..12]),
        repository_commit: args.mathlib_commit.clone(),
        environment_sha256: env_digest,
        extractor_id: "daps-lean-graph-export".into(),
        extractor_version: args.daps_repository_commit.clone(),
        extractor_config_sha256: extractor_config_digest,
    };
    let graph = Graph {
        schema_version: GRAPH_SCHEMA.into(),
        evidence_class: "external_candidate_snapshot".into(),
        snapshot,
        nodes,
        edges: EDGES.into_inner(),
        artifacts,
        upstream_provenance,
    };
    let summary = json!({
        "schema_version": SNAPSHOT_SCHEMA,
        "upstream_node_count": upstream_nodes.len(),
        "upstream_edge_count": upstream_edges.len(),
        "imported_node_count": graph.nodes.len(),
        "imported_edge_count": graph.edges.len(),
        "relation_counts": relation_counts,
        "all_edges_candidate": graph.edges.iter().all(|e| e.status == "candidate"),
        "scheduler_admission": false,
        "confidence_policy": "null: DAPS sample graph exposes no calibrated edge confidence",
        "source_graph_input_sha256": input_digest,
        "environment_sha256": graph.snapshot.environment_sha256
    });
    Ok(Conversion { graph, evidence_files, summary })
}

// A local thread-local staging vector keeps the conversion function simple
// without exposing a mutable output buffer in the public CLI.
thread_local! {
    static EDGES: std::cell::RefCell<Vec<Edge>> = const { std::cell::RefCell::new(Vec::new()) };
}

struct EdgeGuard;
impl EdgeGuard {
    fn clear() {
        EDGES.with(|edges| edges.borrow_mut().clear());
    }
    fn push(edge: Edge) {
        EDGES.with(|edges| edges.borrow_mut().push(edge));
    }
    fn take() -> Vec<Edge> {
        EDGES.with(|edges| std::mem::take(&mut *edges.borrow_mut()))
    }
}

fn nodes_edge_placeholder() {
    // Intentionally empty; retained in v0 to mark the node/edge construction boundary.
}

fn write_conversion(conversion: &Conversion, output_dir: &Path) -> Result<(), String> {
    fs::create_dir_all(output_dir).map_err(|e| format!("create output directory: {e}"))?;
    let evidence_root = output_dir.join("evidence");
    for (relative, bytes) in &conversion.evidence_files {
        let path = evidence_root.join(relative);
        if !path.starts_with(&evidence_root) {
            return Err("generated artifact path escaped evidence root".into());
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("create evidence directory: {e}"))?;
        }
        let mut file = fs::File::create(&path).map_err(|e| format!("create evidence file: {e}"))?;
        file.write_all(bytes).map_err(|e| format!("write evidence file: {e}"))?;
        file.sync_all().map_err(|e| format!("sync evidence file: {e}"))?;
    }
    let graph_bytes = serde_json::to_vec_pretty(&conversion.graph)
        .map_err(|e| format!("serialize imported graph: {e}"))?;
    fs::write(output_dir.join("graph.json"), graph_bytes)
        .map_err(|e| format!("write graph.json: {e}"))?;
    let summary_bytes = serde_json::to_vec_pretty(&conversion.summary)
        .map_err(|e| format!("serialize import summary: {e}"))?;
    fs::write(output_dir.join("summary.json"), summary_bytes)
        .map_err(|e| format!("write summary.json: {e}"))?;
    Ok(())
}

fn parse_args(args: &[String]) -> Result<(PathBuf, PathBuf, ProvenanceArgs), String> {
    if args.len() != 10 || args[1] != "--convert" {
        return Err(format!(
            "usage: {} --convert <daps-graph.json> <output-dir> <daps-repo-commit> <source-graph-blob-sha1> <extractor-blob-sha1> <lake-manifest-blob-sha1> <mathlib-commit> <lean-toolchain>",
            args.first().map(String::as_str).unwrap_or("pitb-daps-graph-import")
        ));
    }
    let provenance = ProvenanceArgs {
        daps_repository_commit: args[4].clone(),
        source_graph_blob_sha1: args[5].clone(),
        extractor_blob_sha1: args[6].clone(),
        lake_manifest_blob_sha1: args[7].clone(),
        mathlib_commit: args[8].clone(),
        lean_toolchain: args[9].clone(),
    };
    Ok((PathBuf::from(&args[2]), PathBuf::from(&args[3]), provenance))
}

fn run_self_test() -> Result<(), String> {
    EdgeGuard::clear();
    let raw = json!({
        "nodes": [
            {"name":"A","category":"theorem","statement":"theorem A : True","signature":"True","module":"Demo","line":1,"source_path":"Demo.lean"},
            {"name":"B","category":"def","subcategory":"definition","statement":"def B : Nat := 1","signature":"Nat","module":"Demo","line":2,"source_path":"Demo.lean"},
            {"name":"file_Demo","category":"file","subcategory":"file","statement":"Demo","module":"Demo","line":0,"source_path":"Demo.lean"}
        ],
        "edges": [
            {"source_name":"A","target_name":"B","type":"USES_DEFINITION","explicit":false,"direct":true,"in_blacklist":false},
            {"source_name":"file_Demo","target_name":"A","type":"CONTAINS","explicit":true,"direct":true,"in_blacklist":false}
        ],
        "metadata":{"version":"synthetic","node_count":3,"edge_count":2}
    });
    let raw_bytes = json_bytes(&raw)?;
    let args = ProvenanceArgs {
        daps_repository_commit: "1".repeat(40),
        source_graph_blob_sha1: "2".repeat(40),
        extractor_blob_sha1: "3".repeat(40),
        lake_manifest_blob_sha1: "4".repeat(40),
        mathlib_commit: "5".repeat(40),
        lean_toolchain: "leanprover/lean4:v4.22.0".into(),
    };
    let result = convert(&raw, &raw_bytes, &args)?;
    if result.graph.edges.len() != 2
        || result.graph.edges.iter().any(|edge| edge.status != "candidate")
        || result.graph.edges.iter().any(|edge| edge.confidence != Value::Null)
        || result.graph.edges.iter().any(|edge| edge.evidence_class != "daps_graph_export")
        || result.graph.edges.iter().filter(|edge| edge.relation == "daps_dependency_candidate").count() != 1
        || result.graph.edges.iter().filter(|edge| edge.relation == "daps_membership_candidate").count() != 1
        || result.summary["scheduler_admission"] != Value::Bool(false)
    {
        return Err("DAPS importer trust-boundary self-test failed".into());
    }
    let raw_edge = &result.evidence_files.iter().find(|(path, _)| path.starts_with("edges/")).ok_or("missing edge evidence file")?.1;
    let edge_evidence: Value = serde_json::from_slice(raw_edge).map_err(|e| e.to_string())?;
    if edge_evidence["upstream_edge"]["type"] != "USES_DEFINITION" {
        return Err("original DAPS edge type was not preserved in evidence".into());
    }
    println!("SELF_TEST_PASS all external DAPS edges remain candidate-only");
    println!("SELF_TEST_PASS DAPS dependency and source-membership edges remain distinct");
    println!("SELF_TEST_PASS upstream edge fields and null confidence are preserved");
    println!("PITB_DAPS_GRAPH_IMPORT_SELF_TEST_PASS");
    Ok(())
}

fn execute() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() == 2 && args[1] == "--self-test" {
        EdgeGuard::clear();
        run_self_test()?;
        return Ok(());
    }
    EdgeGuard::clear();
    let (input_path, output_dir, provenance) = parse_args(&args)?;
    let input_bytes = fs::read(&input_path).map_err(|e| format!("read DAPS graph: {e}"))?;
    let input: Value = serde_json::from_slice(&input_bytes).map_err(|e| format!("parse DAPS graph JSON: {e}"))?;
    let conversion = convert(&input, &input_bytes, &provenance)?;
    write_conversion(&conversion, &output_dir)?;
    println!("{}", serde_json::to_string_pretty(&conversion.summary).map_err(|e| e.to_string())?);
    println!("PITB_DAPS_GRAPH_IMPORT_OK");
    Ok(())
}

fn main() -> ExitCode {
    match execute() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("INVALID: {message}");
            ExitCode::FAILURE
        }
    }
}
