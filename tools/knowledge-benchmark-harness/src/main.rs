use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const TOOL_VERSION: &str = "knowledge-benchmark-harness-v0.1.0";
const DRAFT_2020_12: &str = "https://json-schema.org/draft/2020-12/schema";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Case,
    Graph,
    Claim,
}

impl Kind {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "case" => Ok(Self::Case),
            "graph" => Ok(Self::Graph),
            "claim" => Ok(Self::Claim),
            _ => Err(format!("unknown --kind {raw:?}; expected case, graph, or claim")),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Case => "case",
            Self::Graph => "graph",
            Self::Claim => "claim",
        }
    }
}

struct Config {
    schema: PathBuf,
    input: PathBuf,
    output: Option<PathBuf>,
    kind: Kind,
    subject_revision: Option<String>,
}

#[derive(Debug, Serialize)]
struct RecordResult {
    line: usize,
    record_id: String,
    schema_valid: bool,
    domain_contract_valid: bool,
    errors: Vec<String>,
}

#[derive(Debug, Serialize)]
struct Report {
    report_schema_version: &'static str,
    tool_version: &'static str,
    kind: &'static str,
    subject_revision: Option<String>,
    schema_sha256: String,
    input_sha256: String,
    dependency_lock_sha256: String,
    declared_schema_dialect: Option<String>,
    schema_meta_valid: bool,
    schema_compiled: bool,
    record_count: usize,
    schema_valid_records: usize,
    domain_valid_records: usize,
    passed: bool,
    records: Vec<RecordResult>,
    nonclaims: Vec<&'static str>,
}

fn usage() -> &'static str {
    "Usage: knowledge-benchmark-harness --schema FILE --input JSONL --kind case|graph|claim [--out FILE] [--subject-revision SHA]\n\nValidates local data only. It never fetches source URLs or remote schemas."
}

fn parse_args() -> Result<Config, String> {
    let mut args = env::args().skip(1);
    let mut schema = None;
    let mut input = None;
    let mut output = None;
    let mut kind = None;
    let mut subject_revision = None;

    while let Some(arg) = args.next() {
        let value = |args: &mut std::iter::Skip<std::env::Args>, flag: &str| {
            args.next().ok_or_else(|| format!("{flag} requires a value"))
        };
        match arg.as_str() {
            "--schema" => schema = Some(PathBuf::from(value(&mut args, "--schema")?)),
            "--input" => input = Some(PathBuf::from(value(&mut args, "--input")?)),
            "--out" => output = Some(PathBuf::from(value(&mut args, "--out")?)),
            "--kind" => kind = Some(Kind::parse(&value(&mut args, "--kind")?)?),
            "--subject-revision" => {
                subject_revision = Some(value(&mut args, "--subject-revision")?)
            }
            "--help" | "-h" => return Err(usage().to_string()),
            other => return Err(format!("unknown argument {other:?}\n{}", usage())),
        }
    }

    Ok(Config {
        schema: schema.ok_or_else(|| format!("missing --schema\n{}", usage()))?,
        input: input.ok_or_else(|| format!("missing --input\n{}", usage()))?,
        output,
        kind: kind.ok_or_else(|| format!("missing --kind\n{}", usage()))?,
        subject_revision,
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

/// Reject remote and filesystem reference targets before asking the validator
/// library to compile a schema. Local fragment references remain permitted.
fn find_external_schema_refs(value: &Value, path: &str, errors: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                if matches!(key.as_str(), "$ref" | "$dynamicRef" | "$recursiveRef") {
                    match child.as_str() {
                        Some(target) if target.starts_with('#') => {}
                        Some(target) => errors.push(format!(
                            "{path}/{key}: non-local schema reference {target:?} is refused"
                        )),
                        None => errors.push(format!(
                            "{path}/{key}: schema reference must be a string"
                        )),
                    }
                }
                let next = format!("{path}/{}", key.replace('~', "~0").replace('/', "~1"));
                find_external_schema_refs(child, &next, errors);
            }
        }
        Value::Array(values) => {
            for (index, child) in values.iter().enumerate() {
                find_external_schema_refs(child, &format!("{path}/{index}"), errors);
            }
        }
        _ => {}
    }
}

fn string_field<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn string_array<'a>(value: &'a Value, key: &str) -> Vec<&'a str> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(|items| items.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default()
}

fn validate_graph_contract(graph: &Value) -> Vec<String> {
    let mut errors = Vec::new();
    let sources = graph
        .get("source_records")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let nodes = graph
        .get("nodes")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let edges = graph
        .get("edges")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let synthetic_only = graph
        .get("synthetic_only")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let mut source_ids = HashSet::new();
    for source in &sources {
        if let Some(id) = string_field(source, "source_id") {
            if !source_ids.insert(id.to_string()) {
                errors.push(format!("duplicate source_id {id:?}"));
            }
        }
        if synthetic_only && string_field(source, "source_kind") != Some("synthetic_control") {
            errors.push("synthetic_only graph contains a non-synthetic source record".into());
        }
    }

    let mut node_ids = HashSet::new();
    for node in &nodes {
        if let Some(id) = string_field(node, "node_id") {
            if !node_ids.insert(id.to_string()) {
                errors.push(format!("duplicate node_id {id:?}"));
            }
        }
        for source_id in string_array(node, "source_ids") {
            if !source_ids.contains(source_id) {
                errors.push(format!(
                    "node {:?} references unknown source_id {:?}",
                    string_field(node, "node_id").unwrap_or("<missing>"),
                    source_id
                ));
            }
        }
    }

    let mut edge_ids = HashSet::new();
    for edge in &edges {
        let edge_id = string_field(edge, "edge_id").unwrap_or("<missing>");
        if !edge_ids.insert(edge_id.to_string()) {
            errors.push(format!("duplicate edge_id {edge_id:?}"));
        }
        let from = string_field(edge, "source_node").unwrap_or("<missing>");
        let to = string_field(edge, "target_node").unwrap_or("<missing>");
        if !node_ids.contains(from) {
            errors.push(format!("edge {edge_id:?} has unknown source node {from:?}"));
        }
        if !node_ids.contains(to) {
            errors.push(format!("edge {edge_id:?} has unknown target node {to:?}"));
        }
        for source_id in string_array(edge, "source_ids") {
            if !source_ids.contains(source_id) {
                errors.push(format!(
                    "edge {edge_id:?} references unknown source_id {source_id:?}"
                ));
            }
        }

        let relation = string_field(edge, "relation").unwrap_or("");
        let basis = string_field(edge, "evidence_basis").unwrap_or("");
        let epistemic_status = string_field(edge, "epistemic_status").unwrap_or("");
        if synthetic_only && relation == "historically_influences" {
            errors.push(format!(
                "synthetic graph edge {edge_id:?} cannot assert historical influence"
            ));
        }
        if relation == "historically_influences" {
            let has_non_synthetic_source = string_array(edge, "source_ids").iter().any(|id| {
                sources.iter().any(|source| {
                    string_field(source, "source_id") == Some(id)
                        && matches!(
                            string_field(source, "source_kind"),
                            Some("primary_source" | "scholarly_analysis" | "community_authorized_source")
                        )
                })
            });
            if !has_non_synthetic_source
                || !matches!(basis, "direct_source" | "scholarly_argument")
                || !matches!(epistemic_status, "source_attested" | "scholarly_interpretation")
            {
                errors.push(format!(
                    "edge {edge_id:?}: historically_influences requires admissible non-synthetic evidence, a direct_source/scholarly_argument basis, and an attested or scholarly status"
                ));
            }
        }
        if relation == "causes" && !matches!(basis, "direct_source" | "scholarly_argument") {
            errors.push(format!(
                "edge {edge_id:?}: causes requires direct source evidence or an identified scholarly argument; temporal order alone is insufficient"
            ));
        }
        if synthetic_only && !matches!(basis, "synthetic_assertion" | "explicit_inference") {
            errors.push(format!(
                "synthetic graph edge {edge_id:?} has non-synthetic evidence_basis {basis:?}"
            ));
        }
    }
    errors
}

fn record_identity(kind: Kind, value: &Value, line: usize) -> String {
    let key = match kind {
        Kind::Graph => "graph_id",
        Kind::Case | Kind::Claim => "id",
    };
    string_field(value, key)
        .map(str::to_string)
        .unwrap_or_else(|| format!("line-{line}"))
}

fn collect_unique_object_ids(
    value: &Value,
    array_key: &str,
    id_key: &str,
    context: &str,
    errors: &mut Vec<String>,
) -> HashSet<String> {
    let mut ids = HashSet::new();
    let items = value
        .get(array_key)
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    for item in items {
        if let Some(id) = string_field(&item, id_key) {
            if !ids.insert(id.to_string()) {
                errors.push(format!("duplicate {context} identifier {id:?}"));
            }
        }
    }
    ids
}

fn validate_domain_contract(kind: Kind, value: &Value, seen_ids: &mut HashSet<String>) -> Vec<String> {
    let mut errors = Vec::new();
    let id = record_identity(kind, value, 0);
    if !seen_ids.insert(id.clone()) {
        errors.push(format!("duplicate record ID {id:?}"));
    }

    match kind {
        Kind::Case => {
            collect_unique_object_ids(value, "source_refs", "source_id", "source", &mut errors);
            collect_unique_object_ids(value, "evidence_pack", "evidence_id", "evidence", &mut errors);
        }
        Kind::Claim => {
            let source_ids =
                collect_unique_object_ids(value, "sources", "source_id", "claim source", &mut errors);
            let counterclaims = value
                .get("counterclaims")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            for counterclaim in counterclaims {
                for source_id in string_array(&counterclaim, "source_ids") {
                    if !source_ids.contains(source_id) {
                        errors.push(format!(
                            "counterclaim references unknown claim source_id {source_id:?}"
                        ));
                    }
                }
            }
        }
        Kind::Graph => errors.extend(validate_graph_contract(value)),
    }
    errors
}

fn run(config: Config) -> Result<(Report, bool), String> {
    let schema_bytes = fs::read(&config.schema)
        .map_err(|e| format!("cannot read schema {}: {e}", config.schema.display()))?;
    let input_bytes = fs::read(&config.input)
        .map_err(|e| format!("cannot read input {}: {e}", config.input.display()))?;
    let schema_hash = sha256_hex(&schema_bytes);
    let input_hash = sha256_hex(&input_bytes);
    let lock_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.lock");
    let lock_bytes = fs::read(&lock_path).map_err(|e| {
        format!(
            "cannot read dependency lockfile {}: {e}; generate and review Cargo.lock before qualifying this harness",
            lock_path.display()
        )
    })?;
    let lock_hash = sha256_hex(&lock_bytes);

    let schema: Value = serde_json::from_slice(&schema_bytes)
        .map_err(|e| format!("schema is not valid JSON: {e}"))?;
    let dialect = string_field(&schema, "$schema").map(str::to_string);
    if dialect.as_deref() != Some(DRAFT_2020_12) {
        return Err(format!(
            "schema must declare JSON Schema Draft 2020-12 exactly; found {dialect:?}"
        ));
    }

    let mut ref_errors = Vec::new();
    find_external_schema_refs(&schema, "$", &mut ref_errors);
    if !ref_errors.is_empty() {
        return Err(format!("schema reference policy failed: {}", ref_errors.join("; ")));
    }

    let meta_validator = jsonschema::meta::validator_for(&schema)
        .map_err(|e| format!("cannot build Draft 2020-12 meta-validator: {e}"))?;
    let meta_valid = meta_validator.evaluate(&schema).flag().valid;
    if !meta_valid {
        return Err("schema does not validate against its detected JSON Schema meta-schema".into());
    }

    let validator = jsonschema::draft202012::options()
        .should_validate_formats(true)
        .build(&schema)
        .map_err(|e| format!("cannot compile case schema: {e}"))?;

    let input_text = std::str::from_utf8(&input_bytes)
        .map_err(|e| format!("input is not valid UTF-8: {e}"))?;
    let mut seen_ids = HashSet::new();
    let mut results = Vec::new();

    for (zero_index, raw_line) in input_text.lines().enumerate() {
        let line = zero_index + 1;
        if raw_line.trim().is_empty() {
            results.push(RecordResult {
                line,
                record_id: format!("line-{line}"),
                schema_valid: false,
                domain_contract_valid: false,
                errors: vec!["blank JSONL line is not permitted".into()],
            });
            continue;
        }
        let value: Value = match serde_json::from_str(raw_line) {
            Ok(value) => value,
            Err(error) => {
                results.push(RecordResult {
                    line,
                    record_id: format!("line-{line}"),
                    schema_valid: false,
                    domain_contract_valid: false,
                    errors: vec![format!("invalid JSON: {error}")],
                });
                continue;
            }
        };

        let id = record_identity(config.kind, &value, line);
        let mut errors: Vec<String> = validator
            .iter_errors(&value)
            .map(|error| error.to_string())
            .collect();
        let schema_valid = errors.is_empty();
        let domain_errors = validate_domain_contract(config.kind, &value, &mut seen_ids);
        let domain_contract_valid = domain_errors.is_empty();
        errors.extend(domain_errors);
        results.push(RecordResult {
            line,
            record_id: id,
            schema_valid,
            domain_contract_valid,
            errors,
        });
    }

    if results.is_empty() {
        return Err("input contains no JSONL records".into());
    }

    let schema_valid_records = results.iter().filter(|r| r.schema_valid).count();
    let domain_valid_records = results
        .iter()
        .filter(|r| r.domain_contract_valid)
        .count();
    let passed = results
        .iter()
        .all(|r| r.schema_valid && r.domain_contract_valid);

    let report = Report {
        report_schema_version: "knowledge-benchmark-validation-v0",
        tool_version: TOOL_VERSION,
        kind: config.kind.as_str(),
        subject_revision: config.subject_revision,
        schema_sha256: schema_hash,
        input_sha256: input_hash,
        dependency_lock_sha256: lock_hash,
        declared_schema_dialect: dialect,
        schema_meta_valid: meta_valid,
        schema_compiled: true,
        record_count: results.len(),
        schema_valid_records,
        domain_valid_records,
        passed,
        records: results,
        nonclaims: vec![
            "Schema and domain-contract validation do not establish historical truth.",
            "Validation does not establish rights, permissions, community authorization, or empirical validity.",
            "Validation does not measure model quality or state-of-the-art performance.",
            "A passed fixture validation is not a model benchmark run.",
        ],
    };
    Ok((report, passed))
}

fn write_report(path: Option<&Path>, report: &Report) -> Result<(), String> {
    let json = serde_json::to_string_pretty(report)
        .map_err(|e| format!("cannot serialize validation report: {e}"))?;
    match path {
        Some(path) => fs::write(path, format!("{json}\n"))
            .map_err(|e| format!("cannot write report {}: {e}", path.display())),
        None => {
            println!("{json}");
            Ok(())
        }
    }
}

fn main() -> ExitCode {
    let config = match parse_args() {
        Ok(config) => config,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::from(2);
        }
    };
    let output = config.output.clone();
    match run(config) {
        Ok((report, passed)) => {
            if let Err(error) = write_report(output.as_deref(), &report) {
                eprintln!("{error}");
                return ExitCode::from(2);
            }
            if passed { ExitCode::SUCCESS } else { ExitCode::from(1) }
        }
        Err(error) => {
            eprintln!("validation could not be completed: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn hashes_are_stable_for_identical_bytes() {
        assert_eq!(sha256_hex(b"same"), sha256_hex(b"same"));
        assert_ne!(sha256_hex(b"same"), sha256_hex(b"changed"));
    }

    #[test]
    fn external_schema_references_are_rejected() {
        let schema = json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "properties": {
                "nested": { "$ref": "https://example.invalid/schema.json" }
            }
        });
        let mut errors = Vec::new();
        find_external_schema_refs(&schema, "$", &mut errors);
        assert_eq!(errors.len(), 1);
    }

    #[test]
    fn local_fragment_references_are_allowed() {
        let schema = json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$defs": {"id": {"type": "string"}},
            "properties": {"id": {"$ref": "#/$defs/id"}}
        });
        let mut errors = Vec::new();
        find_external_schema_refs(&schema, "$", &mut errors);
        assert!(errors.is_empty());
    }

    #[test]
    fn graph_rejects_dangling_node_and_source_references() {
        let graph = json!({
            "synthetic_only": true,
            "source_records": [{
                "source_id":"s1",
                "source_kind":"synthetic_control"
            }],
            "nodes": [{
                "node_id":"n1",
                "source_ids":["missing-source"]
            }],
            "edges": [{
                "edge_id":"e1",
                "source_node":"missing-node",
                "target_node":"n1",
                "relation":"resembles",
                "source_ids":["s1"],
                "evidence_basis":"synthetic_assertion"
            }]
        });
        let errors = validate_graph_contract(&graph);
        assert!(errors.iter().any(|e| e.contains("unknown source_id")));
        assert!(errors.iter().any(|e| e.contains("unknown source node")));
    }

    #[test]
    fn synthetic_graph_cannot_claim_historical_influence() {
        let graph = json!({
            "synthetic_only": true,
            "source_records": [{
                "source_id":"s1",
                "source_kind":"synthetic_control"
            }],
            "nodes": [
                {"node_id":"a","source_ids":["s1"]},
                {"node_id":"b","source_ids":["s1"]}
            ],
            "edges": [{
                "edge_id":"e1",
                "source_node":"a",
                "target_node":"b",
                "relation":"historically_influences",
                "source_ids":["s1"],
                "evidence_basis":"synthetic_assertion",
                "epistemic_status":"inference"
            }]
        });
        let errors = validate_graph_contract(&graph);
        assert!(errors.iter().any(|e| e.contains("cannot assert historical influence")));
    }

    #[test]
    fn case_contract_rejects_duplicate_evidence_ids() {
        let value = json!({
            "id":"case-1",
            "evidence_pack":[
                {"evidence_id":"same","statement":"A","kind":"synthetic_control"},
                {"evidence_id":"same","statement":"B","kind":"synthetic_control"}
            ]
        });
        let errors = validate_domain_contract(Kind::Case, &value, &mut HashSet::new());
        assert!(errors.iter().any(|e| e.contains("duplicate evidence identifier")));
    }

    #[test]
    fn claim_contract_rejects_dangling_counterclaim_sources() {
        let value = json!({
            "id":"claim-1",
            "sources":[{"source_id":"known"}],
            "counterclaims":[{"statement":"disputes claim","source_ids":["missing"]}]
        });
        let errors = validate_domain_contract(Kind::Claim, &value, &mut HashSet::new());
        assert!(errors.iter().any(|e| e.contains("unknown claim source_id")));
    }

    #[test]
    fn narrative_sequence_does_not_imply_causation() {
        let graph = json!({
            "synthetic_only": true,
            "source_records": [{
                "source_id":"s1",
                "source_kind":"synthetic_control"
            }],
            "nodes": [
                {"node_id":"a","source_ids":["s1"]},
                {"node_id":"b","source_ids":["s1"]}
            ],
            "edges": [{
                "edge_id":"e1",
                "source_node":"a",
                "target_node":"b",
                "relation":"precedes",
                "source_ids":["s1"],
                "evidence_basis":"synthetic_assertion"
            }]
        });
        assert!(validate_graph_contract(&graph).is_empty());
    }
}
