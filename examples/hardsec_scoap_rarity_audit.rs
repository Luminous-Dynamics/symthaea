//! HARDSEC-009 per-design SCOAP rarity audit.
//!
//! Consumes a JSONL manifest whose first line is a dataset/analyzer header and
//! whose remaining lines are per-net CC0/CC1/CO observations. Computes
//! R(n) = max(CC0, CC1) + alpha * CO using fixed-point arithmetic, ranks each
//! design independently, and evaluates labels only after ranking is frozen.
//!
//! This tool audits supplied SCOAP metrics. It does not parse HDL, calculate
//! SCOAP from a netlist, prove sequential reachability, or certify a design.
//!
//! Example:
//!   cargo run --example hardsec_scoap_rarity_audit --features hardsec-scoap-audit -- \
//!     --manifest /path/to/scoap-manifest.jsonl \
//!     --ranking-output /path/outside/manifest/rarity-ranking.tsv \
//!     --receipt /path/outside/manifest/rarity-audit.json
//!
//! Optional --evidence-root verifies every design's source netlist bytes against
//! the SHA-256 declared in the header. Path traversal and symlinks are rejected.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Component, Path, PathBuf};
use std::process::ExitCode;

const CLAIM_SCOPE: &str =
    "supplied_scoap_metric_integrity_and_per_design_rarity_ranking_only";
const ALGORITHM: &str = "scoap-rarity-fixed-point-cdf-percentile-v1";
const ALPHA_MIN_MILLI: u64 = 500;
const ALPHA_MAX_MILLI: u64 = 1_000;

#[derive(Debug, Clone)]
struct Args {
    manifest: PathBuf,
    ranking_output: PathBuf,
    receipt: PathBuf,
    evidence_root: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct DesignMeta {
    design_id: String,
    base_design_family_id: String,
    split: String,
    netlist_sha256: String,
    netlist_path: Option<String>,
}

#[derive(Debug, Clone)]
struct Header {
    dataset_id: String,
    analyzer_name: String,
    analyzer_version: String,
    analyzer_config_sha256: String,
    sequential_cut_policy: String,
    split_protocol: String,
    alpha_milli: u64,
    designs: BTreeMap<String, DesignMeta>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeLabel {
    Trigger,
    Payload,
    Benign,
    Unknown,
}

impl NodeLabel {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "trigger" => Ok(Self::Trigger),
            "payload" => Ok(Self::Payload),
            "benign" => Ok(Self::Benign),
            "unknown" => Ok(Self::Unknown),
            _ => Err(format!("unsupported node label {value:?}")),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Trigger => "trigger",
            Self::Payload => "payload",
            Self::Benign => "benign",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Node {
    node_id: u64,
    score_milli: u128,
    label: NodeLabel,
    rarity_percentile_bps: u32,
}

#[derive(Debug, Default)]
struct DesignData {
    nodes: Vec<Node>,
}

#[derive(Debug, Clone)]
struct TailMetrics {
    threshold_bps: u32,
    selected_nodes: usize,
    known_selected_nodes: usize,
    trigger_total: usize,
    trigger_selected: usize,
    payload_total: usize,
    payload_selected: usize,
    trigger_recall: Option<f64>,
    trigger_precision: Option<f64>,
    payload_recall: Option<f64>,
    payload_precision: Option<f64>,
}

#[derive(Debug, Clone)]
struct DesignSummary {
    design_id: String,
    base_design_family_id: String,
    split: String,
    node_count: usize,
    trigger_count: usize,
    payload_count: usize,
    benign_count: usize,
    unknown_label_count: usize,
    unknown_metric_count: usize,
    tails: Vec<TailMetrics>,
}

fn usage() -> &'static str {
    "usage: hardsec_scoap_rarity_audit --manifest FILE --ranking-output NEW_FILE --receipt NEW_FILE [--evidence-root DIR]"
}

fn parse_args() -> Result<Option<Args>, String> {
    let mut manifest = None;
    let mut ranking_output = None;
    let mut receipt = None;
    let mut evidence_root = None;
    let mut args = env::args().skip(1);

    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            return Ok(None);
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--manifest" => manifest = Some(PathBuf::from(value)),
            "--ranking-output" => ranking_output = Some(PathBuf::from(value)),
            "--receipt" => receipt = Some(PathBuf::from(value)),
            "--evidence-root" => evidence_root = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown argument: {flag}")),
        }
    }

    Ok(Some(Args {
        manifest: manifest.ok_or("--manifest is required")?,
        ranking_output: ranking_output.ok_or("--ranking-output is required")?,
        receipt: receipt.ok_or("--receipt is required")?,
        evidence_root,
    }))
}

fn required_string<'a>(value: &'a Value, field: &str) -> Result<&'a str, String> {
    let result = value
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{field} must be a string"))?;
    if result.trim().is_empty() || result.trim() != result {
        return Err(format!("{field} must be non-empty and have no surrounding whitespace"));
    }
    Ok(result)
}

fn optional_string(value: &Value, field: &str) -> Result<Option<String>, String> {
    match value.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(item)) if !item.trim().is_empty() && item.trim() == item => {
            Ok(Some(item.clone()))
        }
        _ => Err(format!("{field} must be a non-empty string when present")),
    }
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn safe_relative_path(value: &str, field: &str) -> Result<(), String> {
    if value.contains('\\') || value.contains('\0') {
        return Err(format!("{field} must use portable relative path separators"));
    }
    let path = Path::new(value);
    if path.is_absolute() || path.as_os_str().is_empty() {
        return Err(format!("{field} must be a non-empty relative path"));
    }
    if path
        .components()
        .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!("{field} must not contain '.', '..', root or prefix components"));
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn verify_source_file(root: &Path, design: &DesignMeta) -> Result<(), String> {
    let relative = design
        .netlist_path
        .as_deref()
        .ok_or_else(|| format!("design {}: netlist_path is required with --evidence-root", design.design_id))?;
    safe_relative_path(relative, "netlist_path")?;
    let canonical_root = fs::canonicalize(root)
        .map_err(|error| format!("canonicalize evidence root {}: {error}", root.display()))?;
    let components = Path::new(relative).components().collect::<Vec<_>>();
    let mut candidate = canonical_root.clone();
    for (index, component) in components.iter().enumerate() {
        let Component::Normal(part) = component else {
            return Err(format!("design {}: non-normal path component", design.design_id));
        };
        candidate.push(part);
        let metadata = fs::symlink_metadata(&candidate)
            .map_err(|error| format!("stat {}: {error}", candidate.display()))?;
        if metadata.file_type().is_symlink() {
            return Err(format!("design {}: symlink is forbidden at {}", design.design_id, candidate.display()));
        }
        if index + 1 == components.len() {
            if !metadata.is_file() {
                return Err(format!("design {}: netlist is not a regular file", design.design_id));
            }
        } else if !metadata.is_dir() {
            return Err(format!("design {}: netlist parent is not a directory", design.design_id));
        }
    }
    let resolved = fs::canonicalize(&candidate)
        .map_err(|error| format!("canonicalize netlist {}: {error}", candidate.display()))?;
    if !resolved.starts_with(&canonical_root) {
        return Err(format!("design {}: netlist path escapes evidence root", design.design_id));
    }
    let actual = sha256_file(&resolved)?;
    if actual != design.netlist_sha256 {
        return Err(format!(
            "design {}: netlist SHA-256 mismatch (expected {}, got {actual})",
            design.design_id, design.netlist_sha256
        ));
    }
    Ok(())
}

fn parse_header(value: &Value) -> Result<Header, String> {
    if required_string(value, "schema_version")? != "hardsec-scoap-manifest-v1" {
        return Err("unsupported schema_version".to_owned());
    }
    let dataset_id = required_string(value, "dataset_id")?.to_owned();
    let analyzer = value
        .get("analyzer")
        .and_then(Value::as_object)
        .ok_or("analyzer must be an object")?;
    let analyzer_value = Value::Object(analyzer.clone());
    let analyzer_name = required_string(&analyzer_value, "name")?.to_owned();
    let analyzer_version = required_string(&analyzer_value, "version")?.to_owned();
    let analyzer_config_sha256 =
        required_string(&analyzer_value, "config_sha256")?.to_ascii_lowercase();
    if !valid_sha256(&analyzer_config_sha256) {
        return Err("analyzer.config_sha256 must be a SHA-256 digest".to_owned());
    }
    let sequential_cut_policy = required_string(&analyzer_value, "sequential_cut_policy")?.to_owned();
    let split_protocol = required_string(value, "split_protocol")?.to_owned();
    if !["per_design", "leave_one_design_out", "leave_one_base_family_out"]
        .contains(&split_protocol.as_str())
    {
        return Err(format!("unsupported split_protocol {split_protocol}"));
    }

    let alpha_milli = value
        .get("alpha_milli")
        .and_then(Value::as_u64)
        .ok_or("alpha_milli must be an integer")?;
    if !(ALPHA_MIN_MILLI..=ALPHA_MAX_MILLI).contains(&alpha_milli) {
        return Err(format!(
            "alpha_milli must be between {ALPHA_MIN_MILLI} and {ALPHA_MAX_MILLI}"
        ));
    }

    let design_values = value
        .get("designs")
        .and_then(Value::as_array)
        .ok_or("designs must be an array")?;
    if design_values.is_empty() {
        return Err("designs must not be empty".to_owned());
    }
    let mut designs = BTreeMap::new();
    let mut family_splits: BTreeMap<String, String> = BTreeMap::new();
    for item in design_values {
        let design_id = required_string(item, "design_id")?.to_owned();
        let base_design_family_id = required_string(item, "base_design_family_id")?.to_owned();
        let split = required_string(item, "split")?.to_owned();
        if !["train", "validation", "test"].contains(&split.as_str()) {
            return Err(format!("design {design_id}: split must be train, validation or test"));
        }
        let netlist_sha256 = required_string(item, "netlist_sha256")?.to_ascii_lowercase();
        if !valid_sha256(&netlist_sha256) {
            return Err(format!("design {design_id}: netlist_sha256 is invalid"));
        }
        let netlist_path = optional_string(item, "netlist_path")?;
        if let Some(path) = netlist_path.as_deref() {
            safe_relative_path(path, "netlist_path")?;
        }
        if split_protocol == "leave_one_design_out" || split_protocol == "leave_one_base_family_out" {
            if let Some(previous) = family_splits.insert(
                if split_protocol == "leave_one_design_out" {
                    format!("design:{design_id}")
                } else {
                    format!("family:{base_design_family_id}")
                },
                split.clone(),
            ) {
                if previous != split {
                    return Err(format!(
                        "split leakage: design/family {design_id}/{base_design_family_id} crosses splits"
                    ));
                }
            }
        }
        let metadata = DesignMeta {
            design_id: design_id.clone(),
            base_design_family_id,
            split,
            netlist_sha256,
            netlist_path,
        };
        if designs.insert(design_id.clone(), metadata).is_some() {
            return Err(format!("duplicate design_id {design_id}"));
        }
    }

    for split in ["train", "validation", "test"] {
        if !designs.values().any(|design| design.split == split) {
            return Err(format!("design registry is missing {split}"));
        }
    }

    Ok(Header {
        dataset_id,
        analyzer_name,
        analyzer_version,
        analyzer_config_sha256,
        sequential_cut_policy,
        split_protocol,
        alpha_milli,
        designs,
    })
}

fn parse_node(value: &Value, header: &Header) -> Result<(String, Node), String> {
    let design_id = required_string(value, "design_id")?.to_owned();
    if !header.designs.contains_key(&design_id) {
        return Err(format!("node references unknown design_id {design_id}"));
    }
    let node_id = value
        .get("node_id")
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("node {design_id}: node_id must be a non-negative integer"))?;
    let label = NodeLabel::parse(required_string(value, "label")?)?;
    let metric_status = required_string(value, "metric_status")?;
    if metric_status == "unknown" {
        return Ok((
            design_id,
            Node {
                node_id,
                score_milli: 0,
                label,
                rarity_percentile_bps: 0,
            },
        ));
    }
    if metric_status != "measured" {
        return Err(format!("node {design_id}/{node_id}: metric_status must be measured or unknown"));
    }
    let cc0 = value
        .get("cc0")
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("node {design_id}/{node_id}: cc0 must be a non-negative integer"))?;
    let cc1 = value
        .get("cc1")
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("node {design_id}/{node_id}: cc1 must be a non-negative integer"))?;
    let co = value
        .get("co")
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("node {design_id}/{node_id}: co must be a non-negative integer"))?;
    let max_cc = u128::from(cc0.max(cc1));
    let score_milli = max_cc
        .checked_mul(1_000)
        .and_then(|base| {
            u128::from(co)
                .checked_mul(u128::from(header.alpha_milli))
                .and_then(|observability| base.checked_add(observability))
        })
        .ok_or_else(|| format!("node {design_id}/{node_id}: SCOAP rarity score overflow"))?;

    Ok((
        design_id,
        Node {
            node_id,
            score_milli,
            label,
            rarity_percentile_bps: 0,
        },
    ))
}

fn rank_design(nodes: &mut [Node]) -> Result<usize, String> {
    let unknown_metric_count = nodes
        .iter()
        .filter(|node| node.score_milli == 0 && node.label == NodeLabel::Unknown)
        .count();
    nodes.sort_by_key(|node| node.node_id);
    for pair in nodes.windows(2) {
        if pair[0].node_id == pair[1].node_id {
            return Err(format!("duplicate node_id {}", pair[0].node_id));
        }
    }
    let mut measured: Vec<Node> = nodes
        .iter()
        .copied()
        .filter(|node| !(node.score_milli == 0 && node.label == NodeLabel::Unknown))
        .collect();
    // A measured all-zero SCOAP record is valid, so unknown status is kept in
    // a dedicated side table in the source stream; see parse_node metadata.
    if measured.is_empty() {
        return Ok(unknown_metric_count);
    }
    measured.sort_by(|left, right| {
        left.score_milli
            .cmp(&right.score_milli)
            .then(left.node_id.cmp(&right.node_id))
    });
    let total = measured.len() as u128;
    let mut start = 0usize;
    while start < measured.len() {
        let score = measured[start].score_milli;
        let mut end = start + 1;
        while end < measured.len() && measured[end].score_milli == score {
            end += 1;
        }
        let percentile_bps = ((end as u128) * 10_000 / total) as u32;
        for node in &mut measured[start..end] {
            node.rarity_percentile_bps = percentile_bps;
        }
        start = end;
    }
    let mut lookup = BTreeMap::new();
    for node in measured {
        lookup.insert(node.node_id, node.rarity_percentile_bps);
    }
    for node in nodes.iter_mut() {
        if let Some(percentile) = lookup.get(&node.node_id) {
            node.rarity_percentile_bps = *percentile;
        }
    }
    Ok(unknown_metric_count)
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    if denominator == 0 {
        None
    } else {
        Some(numerator as f64 / denominator as f64)
    }
}

fn evaluate_tails(nodes: &[Node]) -> Vec<TailMetrics> {
    [9_900_u32, 9_500, 9_000, 8_000]
        .into_iter()
        .map(|threshold_bps| {
            let selected: Vec<&Node> = nodes
                .iter()
                .filter(|node| {
                    node.rarity_percentile_bps >= threshold_bps
                        && node.label != NodeLabel::Unknown
                })
                .collect();
            let trigger_total = nodes
                .iter()
                .filter(|node| node.label == NodeLabel::Trigger)
                .count();
            let payload_total = nodes
                .iter()
                .filter(|node| node.label == NodeLabel::Payload)
                .count();
            let trigger_selected = selected
                .iter()
                .filter(|node| node.label == NodeLabel::Trigger)
                .count();
            let payload_selected = selected
                .iter()
                .filter(|node| node.label == NodeLabel::Payload)
                .count();
            let known_selected_nodes = selected.len();
            let selected_nodes = nodes
                .iter()
                .filter(|node| node.rarity_percentile_bps >= threshold_bps)
                .count();
            TailMetrics {
                threshold_bps,
                selected_nodes,
                known_selected_nodes,
                trigger_total,
                trigger_selected,
                payload_total,
                payload_selected,
                trigger_recall: ratio(trigger_selected, trigger_total),
                trigger_precision: ratio(trigger_selected, known_selected_nodes),
                payload_recall: ratio(payload_selected, payload_total),
                payload_precision: ratio(payload_selected, known_selected_nodes),
            }
        })
        .collect()
}

fn write_ranking(path: &Path, header: &Header, data: &BTreeMap<String, DesignData>) -> Result<String, String> {
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("create ranking output {}: {error}", path.display()))?;
    writeln!(
        output,
        "design_id\tbase_design_family_id\tsplit\tnode_id\tscoap_score_milli\trarity_percentile_bps\tlabel"
    )
    .map_err(|error| format!("write ranking header: {error}"))?;

    for (design_id, design_data) in data {
        let metadata = header
            .designs
            .get(design_id)
            .ok_or_else(|| format!("missing design metadata {design_id}"))?;
        let mut nodes = design_data.nodes.clone();
        nodes.sort_by_key(|node| node.node_id);
        for node in nodes {
            if node.label == NodeLabel::Unknown && node.rarity_percentile_bps == 0 {
                writeln!(
                    output,
                    "{}\t{}\t{}\t{}\tunknown\tunknown\t{}",
                    metadata.design_id,
                    metadata.base_design_family_id,
                    metadata.split,
                    node.node_id,
                    node.label.as_str()
                )
                .map_err(|error| format!("write unknown ranking row: {error}"))?;
            } else {
                writeln!(
                    output,
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    metadata.design_id,
                    metadata.base_design_family_id,
                    metadata.split,
                    node.node_id,
                    node.score_milli,
                    node.rarity_percentile_bps,
                    node.label.as_str()
                )
                .map_err(|error| format!("write ranking row: {error}"))?;
            }
        }
    }
    output
        .sync_all()
        .map_err(|error| format!("sync ranking output: {error}"))?;
    drop(output);
    sha256_file(path)
}

fn target_path_outside(path: &Path, inputs: &[&Path]) -> Result<PathBuf, String> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let parent = fs::canonicalize(parent)
        .map_err(|error| format!("canonicalize output parent {}: {error}", parent.display()))?;
    let filename = path.file_name().ok_or("output path must have a filename")?;
    let target = parent.join(filename);
    for input in inputs {
        let input = fs::canonicalize(input)
            .map_err(|error| format!("canonicalize input {}: {error}", input.display()))?;
        if target == input || target.starts_with(input.parent().unwrap_or(Path::new("/"))) {
            return Err(format!(
                "output {} must be outside input directory {}",
                target.display(),
                input.display()
            ));
        }
    }
    Ok(target)
}

fn run(args: Args) -> Result<(), String> {
    let manifest_meta = fs::symlink_metadata(&args.manifest)
        .map_err(|error| format!("stat manifest {}: {error}", args.manifest.display()))?;
    if manifest_meta.file_type().is_symlink() || !manifest_meta.is_file() {
        return Err("manifest must be a regular non-symlink file".to_owned());
    }
    let manifest_bytes = fs::read(&args.manifest)
        .map_err(|error| format!("read manifest {}: {error}", args.manifest.display()))?;
    let manifest_sha256 = format!("{:x}", Sha256::digest(&manifest_bytes));
    let text = std::str::from_utf8(&manifest_bytes)
        .map_err(|error| format!("manifest is not UTF-8: {error}"))?;
    let mut lines = text.lines();
    let header_line = lines.next().ok_or("manifest is empty")?;
    let header_value: Value =
        serde_json::from_str(header_line).map_err(|error| format!("invalid JSONL header: {error}"))?;
    let header = parse_header(&header_value)?;
    let mut data: BTreeMap<String, DesignData> = header
        .designs
        .keys()
        .map(|design_id| (design_id.clone(), DesignData::default()))
        .collect();
    let mut unknown_metric_counts: BTreeMap<String, usize> = BTreeMap::new();

    for (index, line) in lines.enumerate() {
        if line.trim().is_empty() {
            return Err(format!("blank JSONL node record at line {}", index + 2));
        }
        let value: Value = serde_json::from_str(line)
            .map_err(|error| format!("invalid node JSON at line {}: {error}", index + 2))?;
        let (design_id, node) = parse_node(&value, &header)?;
        if node.score_milli == 0 && required_string(&value, "metric_status")? == "unknown" {
            *unknown_metric_counts.entry(design_id.clone()).or_insert(0) += 1;
        }
        data.get_mut(&design_id)
            .ok_or_else(|| format!("unknown design_id {design_id}"))?
            .nodes
            .push(node);
    }

    for (design_id, metadata) in &header.designs {
        if let Some(root) = args.evidence_root.as_deref() {
            verify_source_file(root, metadata)?;
        }
    }

    let mut summaries = Vec::new();
    for (design_id, design_data) in data.iter_mut() {
        if design_data.nodes.is_empty() {
            return Err(format!("design {design_id} has no node records"));
        }
        let unknown_count = unknown_metric_counts.get(design_id).copied().unwrap_or(0);
        rank_design(&mut design_data.nodes)?;
        let metadata = header
            .designs
            .get(design_id)
            .ok_or_else(|| format!("missing design metadata {design_id}"))?;
        let trigger_count = design_data
            .nodes
            .iter()
            .filter(|node| node.label == NodeLabel::Trigger)
            .count();
        let payload_count = design_data
            .nodes
            .iter()
            .filter(|node| node.label == NodeLabel::Payload)
            .count();
        let benign_count = design_data
            .nodes
            .iter()
            .filter(|node| node.label == NodeLabel::Benign)
            .count();
        let unknown_label_count = design_data
            .nodes
            .iter()
            .filter(|node| node.label == NodeLabel::Unknown)
            .count();
        summaries.push(DesignSummary {
            design_id: design_id.clone(),
            base_design_family_id: metadata.base_design_family_id.clone(),
            split: metadata.split.clone(),
            node_count: design_data.nodes.len(),
            trigger_count,
            payload_count,
            benign_count,
            unknown_label_count,
            unknown_metric_count: unknown_count,
            tails: evaluate_tails(&design_data.nodes),
        });
    }

    let ranking_path = target_path_outside(
        &args.ranking_output,
        &[&args.manifest],
    )?;
    let receipt_path = target_path_outside(
        &args.receipt,
        &[&args.manifest, &args.ranking_output],
    )?;
    let ranking_sha256 = write_ranking(&ranking_path, &header, &data)?;

    let receipt = json!({
        "schema_version": "hardsec-scoap-rarity-audit-v1",
        "result": "pass",
        "claim_scope": CLAIM_SCOPE,
        "algorithm": ALGORITHM,
        "dataset_id": header.dataset_id,
        "manifest_sha256": manifest_sha256,
        "analyzer": {
            "name": header.analyzer_name,
            "version": header.analyzer_version,
            "config_sha256": header.analyzer_config_sha256,
            "sequential_cut_policy": header.sequential_cut_policy
        },
        "split_protocol": header.split_protocol,
        "alpha_milli": header.alpha_milli,
        "design_count": summaries.len(),
        "total_node_count": summaries.iter().map(|item| item.node_count).sum::<usize>(),
        "total_unknown_metric_count": summaries.iter().map(|item| item.unknown_metric_count).sum::<usize>(),
        "ranking_output": {
            "path": ranking_path,
            "sha256": ranking_sha256
        },
        "designs": summaries.iter().map(|item| json!({
            "design_id": item.design_id,
            "base_design_family_id": item.base_design_family_id,
            "split": item.split,
            "node_count": item.node_count,
            "trigger_count": item.trigger_count,
            "payload_count": item.payload_count,
            "benign_count": item.benign_count,
            "unknown_label_count": item.unknown_label_count,
            "unknown_metric_count": item.unknown_metric_count,
            "tails": item.tails.iter().map(|tail| json!({
                "rarity_percentile_threshold_bps": tail.threshold_bps,
                "selected_nodes_including_unknown": tail.selected_nodes,
                "selected_labeled_nodes": tail.known_selected_nodes,
                "trigger_total": tail.trigger_total,
                "trigger_selected": tail.trigger_selected,
                "trigger_recall": tail.trigger_recall,
                "trigger_precision": tail.trigger_precision,
                "payload_total": tail.payload_total,
                "payload_selected": tail.payload_selected,
                "payload_recall": tail.payload_recall,
                "payload_precision": tail.payload_precision
            })).collect::<Vec<_>>()
        })).collect::<Vec<_>>(),
        "security_verdict": null,
        "notes": [
            "CC0/CC1/CO values are supplied by the declared analyzer and are not recomputed from HDL.",
            "Rarity is ranked independently within each design; scores are not pooled across designs.",
            "Node labels are excluded from score computation and ordering; they are used only for post-ranking descriptive metrics.",
            "SCOAP uses the analyzer's declared sequential cut policy and does not prove sequential trigger reachability.",
            "A high SCOAP rarity percentile is a candidate-prioritization signal, not a Trojan verdict."
        ]
    });
    let serialized = serde_json::to_vec_pretty(&receipt)
        .map_err(|error| format!("serialize receipt: {error}"))?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&receipt_path)
        .map_err(|error| format!("create receipt {}: {error}", receipt_path.display()))?;
    let write_result = output
        .write_all(&serialized)
        .and_then(|_| output.write_all(b"\n"))
        .and_then(|_| output.sync_all());
    drop(output);
    if let Err(error) = write_result {
        let _ = fs::remove_file(&receipt_path);
        let _ = fs::remove_file(&ranking_path);
        return Err(format!("write receipt {}: {error}", receipt_path.display()));
    }

    println!("HARDSEC-009 SCOAP rarity audit: PASS");
    println!("designs: {}", summaries.len());
    println!("nodes: {}", summaries.iter().map(|item| item.node_count).sum::<usize>());
    println!("unknown SCOAP records: {}", summaries.iter().map(|item| item.unknown_metric_count).sum::<usize>());
    println!("ranking output SHA-256: {ranking_sha256}");
    println!("receipt: {}", receipt_path.display());
    println!("claim scope: {}", CLAIM_SCOPE);
    Ok(())
}

fn main() -> ExitCode {
    match parse_args() {
        Ok(Some(args)) => match run(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("HARDSEC-009 SCOAP audit failed: {error}");
                ExitCode::FAILURE
            }
        },
        Ok(None) => {
            println!("{}", usage());
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}\n{}", usage());
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT: AtomicUsize = AtomicUsize::new(0);

    fn header(alpha_milli: u64) -> Header {
        let design = DesignMeta {
            design_id: "design-a".to_owned(),
            base_design_family_id: "family-a".to_owned(),
            split: "train".to_owned(),
            netlist_sha256: "a".repeat(64),
            netlist_path: None,
        };
        Header {
            dataset_id: "fixture".to_owned(),
            analyzer_name: "fixture-scoap".to_owned(),
            analyzer_version: "1.0".to_owned(),
            analyzer_config_sha256: "b".repeat(64),
            sequential_cut_policy: "register_cut".to_owned(),
            split_protocol: "per_design".to_owned(),
            alpha_milli,
            designs: BTreeMap::from([("design-a".to_owned(), design)]),
        }
    }

    fn node(node_id: u64, cc0: u64, cc1: u64, co: u64, label: NodeLabel, alpha: u64) -> Node {
        let max_cc = u128::from(cc0.max(cc1));
        Node {
            node_id,
            score_milli: max_cc * 1_000 + u128::from(co) * u128::from(alpha),
            label,
            rarity_percentile_bps: 0,
        }
    }

    #[test]
    fn matches_published_scoap_example_at_alpha_one() {
        let header = header(1_000);
        let (design_id, value) = parse_node(
            &json!({
                "design_id": "design-a",
                "node_id": 1,
                "metric_status": "measured",
                "cc0": 1,
                "cc1": 2,
                "co": 10,
                "label": "trigger"
            }),
            &header,
        )
        .unwrap();
        assert_eq!(design_id, "design-a");
        assert_eq!(value.score_milli, 12_000);
    }

    #[test]
    fn supports_half_weight_observability() {
        let header = header(500);
        let (_, value) = parse_node(
            &json!({
                "design_id": "design-a",
                "node_id": 1,
                "metric_status": "measured",
                "cc0": 1,
                "cc1": 2,
                "co": 10,
                "label": "trigger"
            }),
            &header,
        )
        .unwrap();
        assert_eq!(value.score_milli, 7_000);
    }

    #[test]
    fn groups_equal_scores_into_the_same_percentile() {
        let mut nodes = vec![
            node(1, 1, 2, 10, NodeLabel::Trigger, 1_000),
            node(2, 1, 2, 10, NodeLabel::Benign, 1_000),
            node(3, 2, 2, 0, NodeLabel::Payload, 1_000),
        ];
        rank_design(&mut nodes).unwrap();
        let first = nodes.iter().find(|item| item.node_id == 1).unwrap();
        let second = nodes.iter().find(|item| item.node_id == 2).unwrap();
        assert_eq!(first.rarity_percentile_bps, second.rarity_percentile_bps);
        assert_eq!(first.rarity_percentile_bps, 10_000);
    }

    #[test]
    fn computes_percentiles_separately_per_design() {
        let mut left = vec![
            node(1, 1, 1, 1, NodeLabel::Trigger, 1_000),
            node(2, 2, 2, 2, NodeLabel::Benign, 1_000),
        ];
        let mut right = vec![
            node(1, 100, 100, 100, NodeLabel::Benign, 1_000),
            node(2, 200, 200, 200, NodeLabel::Payload, 1_000),
        ];
        rank_design(&mut left).unwrap();
        rank_design(&mut right).unwrap();
        assert_eq!(
            left.iter().find(|node| node.node_id == 2).unwrap().rarity_percentile_bps,
            10_000
        );
        assert_eq!(
            right.iter().find(|node| node.node_id == 2).unwrap().rarity_percentile_bps,
            10_000
        );
    }

    #[test]
    fn ignores_labels_when_computing_score_and_rank() {
        let mut trigger = node(1, 1, 4, 10, NodeLabel::Trigger, 1_000);
        let mut benign = trigger;
        benign.label = NodeLabel::Benign;
        let mut left = vec![trigger, node(2, 1, 1, 1, NodeLabel::Benign, 1_000)];
        let mut right = vec![benign, node(2, 1, 1, 1, NodeLabel::Trigger, 1_000)];
        rank_design(&mut left).unwrap();
        rank_design(&mut right).unwrap();
        let a = left.iter().find(|node| node.node_id == 1).unwrap();
        let b = right.iter().find(|node| node.node_id == 1).unwrap();
        assert_eq!(a.score_milli, b.score_milli);
        assert_eq!(a.rarity_percentile_bps, b.rarity_percentile_bps);
    }

    #[test]
    fn unknown_metric_rows_are_not_imputed() {
        let header = header(1_000);
        let (_, value) = parse_node(
            &json!({
                "design_id": "design-a",
                "node_id": 5,
                "metric_status": "unknown",
                "label": "unknown"
            }),
            &header,
        )
        .unwrap();
        assert_eq!(value.rarity_percentile_bps, 0);
        assert_eq!(value.score_milli, 0);
    }

    #[test]
    fn rejects_invalid_alpha() {
        let mut value = json!({
            "schema_version": "hardsec-scoap-manifest-v1",
            "dataset_id": "fixture",
            "alpha_milli": 499,
            "analyzer": {
                "name": "fixture",
                "version": "1.0",
                "config_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "sequential_cut_policy": "register_cut"
            },
            "split_protocol": "per_design",
            "designs": [{
                "design_id": "design-a",
                "base_design_family_id": "family-a",
                "split": "train",
                "netlist_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            }]
        });
        assert!(parse_header(&value).unwrap_err().contains("alpha_milli"));
        value["alpha_milli"] = json!(1_001);
        assert!(parse_header(&value).is_err());
    }

    #[test]
    fn detects_duplicate_node_ids() {
        let mut nodes = vec![
            node(7, 1, 2, 3, NodeLabel::Trigger, 1_000),
            node(7, 2, 3, 4, NodeLabel::Payload, 1_000),
        ];
        assert!(rank_design(&mut nodes).unwrap_err().contains("duplicate node_id"));
    }

    #[test]
    fn rejects_malformed_metric_status_and_labels() {
        let header = header(1_000);
        let result = parse_node(
            &json!({
                "design_id": "design-a",
                "node_id": 1,
                "metric_status": "not-measured",
                "label": "trigger"
            }),
            &header,
        );
        assert!(result.is_err());
        let result = parse_node(
            &json!({
                "design_id": "design-a",
                "node_id": 1,
                "metric_status": "measured",
                "cc0": 1,
                "cc1": 1,
                "co": 1,
                "label": "maybe"
            }),
            &header,
        );
        assert!(result.is_err());
    }
}
