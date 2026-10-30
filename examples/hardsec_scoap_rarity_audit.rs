//! HARDSEC-009 per-design SCOAP rarity audit.
//!
//! Reads a JSONL stream: one provenance header followed by node records sorted
//! by (design_id, node_id). It retains one design's nodes at a time, giving
//! O(max nodes in one design) working memory rather than loading the source
//! manifest or all designs into memory.
//!
//! R(n) = max(CC0(n), CC1(n)) + alpha * CO(n)
//!
//! CC0/CC1/CO are supplied by a named analyzer. This tool does not parse HDL,
//! calculate SCOAP from a netlist, prove sequential reachability, or certify a
//! design. Labels are excluded from ranking and used only for aggregate
//! post-ranking metrics. The ranking TSV contains no labels.
//!
//! Required header: schema_version, record_order, dataset_id, split_protocol,
//! alpha_milli, analyzer, designs[].
//!
//! Example:
//!   cargo run --example hardsec_scoap_rarity_audit --features hardsec-scoap-audit -- \
//!     --manifest /data/scoap.jsonl --ranking-output /output/ranking.tsv \
//!     --receipt /output/audit.json --evidence-root /data/netlists

#![recursion_limit = "256"]

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::ExitCode;

const CLAIM_SCOPE: &str = "supplied_scoap_metric_integrity_and_per_design_rarity_ranking_only";
const ALGORITHM: &str = "scoap-rarity-fixed-point-midrank-percentile-v2";
const RECORD_ORDER: &str = "design_id_then_node_id_ascending";
const ALPHA_MIN_MILLI: u64 = 500;
const ALPHA_MAX_MILLI: u64 = 1_000;
const TAIL_THRESHOLDS_BPS: [u32; 4] = [9_900, 9_500, 9_000, 8_000];

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
}

#[derive(Debug, Clone, Copy)]
struct Node {
    node_id: u64,
    score_milli: u128,
    label: NodeLabel,
    metric_measured: bool,
    rarity_percentile_bps: u32,
}

#[derive(Debug, Default)]
struct DesignData {
    nodes: Vec<Node>,
    last_node_id: Option<u64>,
}

#[derive(Debug, Clone)]
struct TailMetrics {
    threshold_bps: u32,
    selected_measured_nodes: usize,
    selected_labeled_nodes: usize,
    selected_unknown_label_nodes: usize,
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
    netlist_sha256: String,
    netlist_path: Option<String>,
    node_count: usize,
    measured_metric_count: usize,
    trigger_label_count: usize,
    payload_label_count: usize,
    benign_label_count: usize,
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
        return Err(format!(
            "{field} must be non-empty and have no surrounding whitespace"
        ));
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
        return Err(format!(
            "{field} must use portable relative path separators"
        ));
    }
    let path = Path::new(value);
    if path.is_absolute() || path.as_os_str().is_empty() {
        return Err(format!("{field} must be a non-empty relative path"));
    }
    if path
        .components()
        .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(format!(
            "{field} must not contain '.', '..', root or prefix components"
        ));
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| format!("read {}: {error}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn verify_source_file(root: &Path, design: &DesignMeta) -> Result<(), String> {
    let root_meta = fs::symlink_metadata(root)
        .map_err(|error| format!("stat evidence root {}: {error}", root.display()))?;
    if root_meta.file_type().is_symlink() || !root_meta.is_dir() {
        return Err("evidence root must be a regular non-symlink directory".to_owned());
    }
    let relative = design.netlist_path.as_deref().ok_or_else(|| {
        format!(
            "design {}: netlist_path is required with --evidence-root",
            design.design_id
        )
    })?;
    safe_relative_path(relative, "netlist_path")?;
    let canonical_root = fs::canonicalize(root)
        .map_err(|error| format!("canonicalize evidence root {}: {error}", root.display()))?;
    let components = Path::new(relative).components().collect::<Vec<_>>();
    let mut candidate = canonical_root.clone();

    for (index, component) in components.iter().enumerate() {
        let Component::Normal(part) = component else {
            return Err(format!(
                "design {}: non-normal path component",
                design.design_id
            ));
        };
        candidate.push(part);
        let metadata = fs::symlink_metadata(&candidate)
            .map_err(|error| format!("stat {}: {error}", candidate.display()))?;
        if metadata.file_type().is_symlink() {
            return Err(format!(
                "design {}: symlink is forbidden at {}",
                design.design_id,
                candidate.display()
            ));
        }
        if index + 1 == components.len() {
            if !metadata.is_file() {
                return Err(format!(
                    "design {}: netlist is not a regular file",
                    design.design_id
                ));
            }
        } else if !metadata.is_dir() {
            return Err(format!(
                "design {}: netlist parent is not a directory",
                design.design_id
            ));
        }
    }

    let resolved = fs::canonicalize(&candidate)
        .map_err(|error| format!("canonicalize netlist {}: {error}", candidate.display()))?;
    if !resolved.starts_with(&canonical_root) {
        return Err(format!(
            "design {}: netlist path escapes evidence root",
            design.design_id
        ));
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
    if required_string(value, "record_order")? != RECORD_ORDER {
        return Err(format!("record_order must equal {RECORD_ORDER}"));
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
    let sequential_cut_policy =
        required_string(&analyzer_value, "sequential_cut_policy")?.to_owned();
    let split_protocol = required_string(value, "split_protocol")?.to_owned();
    if ![
        "per_design",
        "leave_one_design_out",
        "leave_one_base_family_out",
    ]
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
    let mut netlist_splits: BTreeMap<String, String> = BTreeMap::new();

    for item in design_values {
        let design_id = required_string(item, "design_id")?.to_owned();
        let base_design_family_id = required_string(item, "base_design_family_id")?.to_owned();
        let split = required_string(item, "split")?.to_owned();
        if !["train", "validation", "test"].contains(&split.as_str()) {
            return Err(format!(
                "design {design_id}: split must be train, validation or test"
            ));
        }
        let netlist_sha256 = required_string(item, "netlist_sha256")?.to_ascii_lowercase();
        if !valid_sha256(&netlist_sha256) {
            return Err(format!("design {design_id}: netlist_sha256 is invalid"));
        }
        let netlist_path = optional_string(item, "netlist_path")?;
        if let Some(path) = netlist_path.as_deref() {
            safe_relative_path(path, "netlist_path")?;
        }

        if split_protocol == "leave_one_base_family_out" {
            if let Some(previous) =
                family_splits.insert(base_design_family_id.clone(), split.clone())
            {
                if previous != split {
                    return Err(format!(
                        "base family {base_design_family_id} appears in {previous} and {split}"
                    ));
                }
            }
        }
        if let Some(previous) = netlist_splits.insert(netlist_sha256.clone(), split.clone()) {
            if previous != split {
                return Err(format!(
                    "identical netlist SHA-256 appears in {previous} and {split}"
                ));
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
        for field in ["cc0", "cc1", "co"] {
            if value.get(field).is_some() {
                return Err(format!(
                    "node {design_id}/{node_id}: unknown metric_status must omit {field}"
                ));
            }
        }
        return Ok((
            design_id,
            Node {
                node_id,
                score_milli: 0,
                label,
                metric_measured: false,
                rarity_percentile_bps: 0,
            },
        ));
    }
    if metric_status != "measured" {
        return Err(format!(
            "node {design_id}/{node_id}: metric_status must be measured or unknown"
        ));
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
            metric_measured: true,
            rarity_percentile_bps: 0,
        },
    ))
}

fn rank_design(nodes: &mut [Node]) -> Result<usize, String> {
    nodes.sort_by_key(|node| node.node_id);
    for pair in nodes.windows(2) {
        if pair[0].node_id == pair[1].node_id {
            return Err(format!("duplicate node_id {}", pair[0].node_id));
        }
    }

    let unknown_metric_count = nodes.iter().filter(|node| !node.metric_measured).count();
    let mut order: Vec<usize> = nodes
        .iter()
        .enumerate()
        .filter_map(|(index, node)| node.metric_measured.then_some(index))
        .collect();
    order.sort_by(|left, right| {
        nodes[*left]
            .score_milli
            .cmp(&nodes[*right].score_milli)
            .then(nodes[*left].node_id.cmp(&nodes[*right].node_id))
    });

    let total = order.len() as u128;
    let mut start = 0usize;
    while start < order.len() {
        let score = nodes[order[start]].score_milli;
        let mut end = start + 1;
        while end < order.len() && nodes[order[end]].score_milli == score {
            end += 1;
        }
        // Midrank percentile: ties share the empirical mid-distribution rank.
        let rank_twice = (start as u128) * 2 + (end - start) as u128;
        let percentile_bps = (rank_twice * 10_000 / (total * 2)) as u32;
        for index in &order[start..end] {
            nodes[*index].rarity_percentile_bps = percentile_bps;
        }
        start = end;
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
    TAIL_THRESHOLDS_BPS
        .into_iter()
        .map(|threshold_bps| {
            let selected_measured: Vec<&Node> = nodes
                .iter()
                .filter(|node| node.metric_measured && node.rarity_percentile_bps >= threshold_bps)
                .collect();
            let selected_labeled: Vec<&Node> = selected_measured
                .iter()
                .copied()
                .filter(|node| node.label != NodeLabel::Unknown)
                .collect();
            let trigger_total = nodes
                .iter()
                .filter(|node| node.metric_measured && node.label == NodeLabel::Trigger)
                .count();
            let payload_total = nodes
                .iter()
                .filter(|node| node.metric_measured && node.label == NodeLabel::Payload)
                .count();
            let trigger_selected = selected_labeled
                .iter()
                .filter(|node| node.label == NodeLabel::Trigger)
                .count();
            let payload_selected = selected_labeled
                .iter()
                .filter(|node| node.label == NodeLabel::Payload)
                .count();

            TailMetrics {
                threshold_bps,
                selected_measured_nodes: selected_measured.len(),
                selected_labeled_nodes: selected_labeled.len(),
                selected_unknown_label_nodes: selected_measured.len() - selected_labeled.len(),
                trigger_total,
                trigger_selected,
                payload_total,
                payload_selected,
                trigger_recall: ratio(trigger_selected, trigger_total),
                trigger_precision: ratio(trigger_selected, selected_labeled.len()),
                payload_recall: ratio(payload_selected, payload_total),
                payload_precision: ratio(payload_selected, selected_labeled.len()),
            }
        })
        .collect()
}

fn summarize_design(metadata: &DesignMeta, data: &mut DesignData) -> Result<DesignSummary, String> {
    if data.nodes.is_empty() {
        return Err(format!("design {} has no node records", metadata.design_id));
    }
    let unknown_metric_count = rank_design(&mut data.nodes)?;
    let measured_metric_count = data.nodes.len() - unknown_metric_count;

    Ok(DesignSummary {
        design_id: metadata.design_id.clone(),
        base_design_family_id: metadata.base_design_family_id.clone(),
        split: metadata.split.clone(),
        netlist_sha256: metadata.netlist_sha256.clone(),
        netlist_path: metadata.netlist_path.clone(),
        node_count: data.nodes.len(),
        measured_metric_count,
        trigger_label_count: data
            .nodes
            .iter()
            .filter(|node| node.label == NodeLabel::Trigger)
            .count(),
        payload_label_count: data
            .nodes
            .iter()
            .filter(|node| node.label == NodeLabel::Payload)
            .count(),
        benign_label_count: data
            .nodes
            .iter()
            .filter(|node| node.label == NodeLabel::Benign)
            .count(),
        unknown_label_count: data
            .nodes
            .iter()
            .filter(|node| node.label == NodeLabel::Unknown)
            .count(),
        unknown_metric_count,
        tails: evaluate_tails(&data.nodes),
    })
}

fn flush_design(
    design_id: &str,
    data: &mut DesignData,
    header: &Header,
    ranking: &mut BufWriter<&mut File>,
) -> Result<DesignSummary, String> {
    let metadata = header
        .designs
        .get(design_id)
        .ok_or_else(|| format!("missing design metadata {design_id}"))?;
    let summary = summarize_design(metadata, data)?;

    for node in &data.nodes {
        if node.metric_measured {
            writeln!(
                ranking,
                "{}\t{}\t{}\t{}\t{}\t{}",
                metadata.design_id,
                metadata.base_design_family_id,
                metadata.split,
                node.node_id,
                node.score_milli,
                node.rarity_percentile_bps
            )
            .map_err(|error| format!("write ranking row: {error}"))?;
        } else {
            writeln!(
                ranking,
                "{}\t{}\t{}\t{}\tunknown\tunknown",
                metadata.design_id, metadata.base_design_family_id, metadata.split, node.node_id
            )
            .map_err(|error| format!("write unknown ranking row: {error}"))?;
        }
    }

    data.nodes.clear();
    data.last_node_id = None;
    Ok(summary)
}

fn target_path_outside(
    path: &Path,
    file_inputs: &[&Path],
    directory_inputs: &[&Path],
) -> Result<PathBuf, String> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let parent = fs::canonicalize(parent)
        .map_err(|error| format!("canonicalize output parent {}: {error}", parent.display()))?;
    let filename = path.file_name().ok_or("output path must have a filename")?;
    let target = parent.join(filename);

    for input in file_inputs {
        let canonical = fs::canonicalize(input)
            .map_err(|error| format!("canonicalize input {}: {error}", input.display()))?;
        if target == canonical {
            return Err(format!("output {} aliases an input file", target.display()));
        }
    }
    for input in directory_inputs {
        let canonical = fs::canonicalize(input).map_err(|error| {
            format!("canonicalize input directory {}: {error}", input.display())
        })?;
        if target.starts_with(&canonical) {
            return Err(format!(
                "output {} must be outside input directory {}",
                target.display(),
                canonical.display()
            ));
        }
    }
    Ok(target)
}

fn parse_jsonl_line<R: BufRead>(
    reader: &mut R,
    hasher: &mut Sha256,
    line_number: &mut usize,
) -> Result<Option<Value>, String> {
    let mut bytes = Vec::new();
    let count = reader
        .read_until(b'\n', &mut bytes)
        .map_err(|error| format!("read JSONL line {}: {error}", *line_number + 1))?;
    if count == 0 {
        return Ok(None);
    }
    *line_number += 1;
    hasher.update(&bytes);
    if bytes.last() == Some(&b'\n') {
        bytes.pop();
    }
    if bytes.last() == Some(&b'\r') {
        bytes.pop();
    }
    if bytes.is_empty() {
        return Err(format!("blank JSONL record at line {}", *line_number));
    }
    let text = std::str::from_utf8(&bytes)
        .map_err(|error| format!("JSONL line {} is not UTF-8: {error}", *line_number))?;
    let value = serde_json::from_str(text)
        .map_err(|error| format!("invalid JSON at line {}: {error}", *line_number))?;
    Ok(Some(value))
}

fn run(args: Args) -> Result<(), String> {
    let manifest_meta = fs::symlink_metadata(&args.manifest)
        .map_err(|error| format!("stat manifest {}: {error}", args.manifest.display()))?;
    if manifest_meta.file_type().is_symlink() || !manifest_meta.is_file() {
        return Err("manifest must be a regular non-symlink file".to_owned());
    }

    let manifest_parent = args
        .manifest
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut directory_inputs = vec![manifest_parent];
    if let Some(root) = args.evidence_root.as_deref() {
        directory_inputs.push(root);
    }
    let ranking_path =
        target_path_outside(&args.ranking_output, &[&args.manifest], &directory_inputs)?;
    let receipt_path = target_path_outside(&args.receipt, &[&args.manifest], &directory_inputs)?;
    if ranking_path == receipt_path {
        return Err("ranking output and receipt paths must differ".to_owned());
    }

    let input_file = File::open(&args.manifest)
        .map_err(|error| format!("open manifest {}: {error}", args.manifest.display()))?;
    let opened_metadata = input_file
        .metadata()
        .map_err(|error| format!("stat opened manifest: {error}"))?;
    if !opened_metadata.is_file() {
        return Err("opened manifest is not a regular file".to_owned());
    }
    let mut reader = BufReader::new(input_file);
    let mut manifest_hasher = Sha256::new();
    let mut line_number = 0usize;
    let header_value = parse_jsonl_line(&mut reader, &mut manifest_hasher, &mut line_number)?
        .ok_or("manifest is empty")?;
    let header = parse_header(&header_value)?;

    // Verify source design artifacts before creating output files.
    let mut verified_netlists = Vec::new();
    let source_integrity_status = if let Some(root) = args.evidence_root.as_deref() {
        for metadata in header.designs.values() {
            verify_source_file(root, metadata)?;
            verified_netlists.push(json!({
                "design_id": metadata.design_id,
                "path": metadata.netlist_path,
                "sha256": metadata.netlist_sha256
            }));
        }
        "verified"
    } else {
        "not_requested"
    };

    let mut ranking_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&ranking_path)
        .map_err(|error| format!("create ranking output {}: {error}", ranking_path.display()))?;

    let processing = (|| -> Result<Vec<DesignSummary>, String> {
        let mut ranking = BufWriter::new(&mut ranking_file);
        writeln!(
            ranking,
            "design_id\tbase_design_family_id\tsplit\tnode_id\tscoap_score_milli\trarity_percentile_bps"
        )
        .map_err(|error| format!("write ranking header: {error}"))?;

        let design_ids: Vec<&String> = header.designs.keys().collect();
        let mut next_design_index = 0usize;
        let mut current_design: Option<String> = None;
        let mut current_data = DesignData::default();
        let mut summaries = Vec::with_capacity(header.designs.len());

        while let Some(value) =
            parse_jsonl_line(&mut reader, &mut manifest_hasher, &mut line_number)?
        {
            let (design_id, node) = parse_node(&value, &header)?;
            match current_design.as_deref() {
                None => {
                    let expected = design_ids
                        .get(next_design_index)
                        .ok_or_else(|| format!("unexpected extra design {design_id}"))?;
                    if expected.as_str() != design_id {
                        return Err(format!(
                            "record order violation: expected design {}, received {design_id}",
                            expected
                        ));
                    }
                    current_design = Some(design_id.clone());
                }
                Some(current) if design_id == current => {}
                Some(current) if design_id.as_str() > current => {
                    summaries.push(flush_design(
                        current,
                        &mut current_data,
                        &header,
                        &mut ranking,
                    )?);
                    next_design_index += 1;
                    let expected = design_ids
                        .get(next_design_index)
                        .ok_or_else(|| format!("unexpected extra design {design_id}"))?;
                    if expected.as_str() != design_id {
                        return Err(format!(
                            "record order violation: expected design {}, received {design_id}",
                            expected
                        ));
                    }
                    current_design = Some(design_id.clone());
                }
                Some(current) => {
                    return Err(format!(
                        "record order violation: design {design_id} follows {current}"
                    ));
                }
            }

            if let Some(previous) = current_data.last_node_id {
                if node.node_id <= previous {
                    return Err(format!(
                        "design {design_id}: node_id {} is not greater than previous node_id {previous}",
                        node.node_id
                    ));
                }
            }
            current_data.last_node_id = Some(node.node_id);
            current_data.nodes.push(node);
        }

        if let Some(current) = current_design.as_deref() {
            summaries.push(flush_design(
                current,
                &mut current_data,
                &header,
                &mut ranking,
            )?);
            next_design_index += 1;
        }

        if next_design_index != design_ids.len() {
            let missing = design_ids
                .get(next_design_index)
                .map(|value| value.as_str())
                .unwrap_or("<unknown>");
            return Err(format!(
                "manifest missing node records for design {missing}"
            ));
        }

        ranking
            .flush()
            .map_err(|error| format!("flush ranking output: {error}"))?;
        Ok(summaries)
    })();

    let summaries = match processing {
        Ok(value) => value,
        Err(error) => {
            drop(ranking_file);
            let _ = fs::remove_file(&ranking_path);
            return Err(error);
        }
    };
    if let Err(error) = ranking_file.sync_all() {
        drop(ranking_file);
        let _ = fs::remove_file(&ranking_path);
        return Err(format!("sync ranking output: {error}"));
    }
    drop(ranking_file);

    let manifest_sha256 = format!("{:x}", manifest_hasher.finalize());

    let ranking_sha256 = match sha256_file(&ranking_path) {
        Ok(value) => value,
        Err(error) => {
            let _ = fs::remove_file(&ranking_path);
            return Err(error);
        }
    };

    let receipt = json!({
        "schema_version": "hardsec-scoap-rarity-audit-v2",
        "result": "pass",
        "claim_scope": CLAIM_SCOPE,
        "algorithm": ALGORITHM,
        "record_order": RECORD_ORDER,
        "dataset_id": header.dataset_id,
        "manifest_sha256": manifest_sha256,
        "manifest_line_count": line_number,
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
        "total_measured_metric_count": summaries.iter().map(|item| item.measured_metric_count).sum::<usize>(),
        "total_unknown_metric_count": summaries.iter().map(|item| item.unknown_metric_count).sum::<usize>(),
        "source_integrity": {
            "status": source_integrity_status,
            "verified_netlist_count": verified_netlists.len(),
            "netlists": verified_netlists
        },
        "ranking_output": {
            "path": ranking_path,
            "sha256": ranking_sha256,
            "contains_labels": false
        },
        "designs": summaries.iter().map(|item| json!({
            "design_id": item.design_id,
            "base_design_family_id": item.base_design_family_id,
            "split": item.split,
            "netlist_sha256": item.netlist_sha256,
            "netlist_path": item.netlist_path,
            "node_count": item.node_count,
            "measured_metric_count": item.measured_metric_count,
            "trigger_label_count": item.trigger_label_count,
            "payload_label_count": item.payload_label_count,
            "benign_label_count": item.benign_label_count,
            "unknown_label_count": item.unknown_label_count,
            "unknown_metric_count": item.unknown_metric_count,
            "tails": item.tails.iter().map(|tail| json!({
                "rarity_percentile_threshold_bps": tail.threshold_bps,
                "selected_measured_nodes": tail.selected_measured_nodes,
                "selected_labeled_nodes": tail.selected_labeled_nodes,
                "selected_unknown_label_nodes": tail.selected_unknown_label_nodes,
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
            "Ranking output contains no node labels; labels are used only for post-ranking descriptive metrics.",
            "Percentiles use empirical midranks; tied SCOAP scores receive identical percentiles.",
            "SCOAP uses the analyzer's declared sequential cut policy and does not prove sequential trigger reachability.",
            "A high SCOAP rarity percentile is a candidate-prioritization signal, not a Trojan verdict.",
            "JSONL input is streamed; only one design's nodes are held in memory at a time."
        ]
    });

    let serialized = serde_json::to_vec_pretty(&receipt)
        .map_err(|error| format!("serialize receipt: {error}"))?;
    let mut output = match OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&receipt_path)
    {
        Ok(value) => value,
        Err(error) => {
            let _ = fs::remove_file(&ranking_path);
            return Err(format!(
                "create receipt {}: {error}",
                receipt_path.display()
            ));
        }
    };
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
    println!(
        "nodes: {}",
        summaries.iter().map(|item| item.node_count).sum::<usize>()
    );
    println!(
        "unknown SCOAP records: {}",
        summaries
            .iter()
            .map(|item| item.unknown_metric_count)
            .sum::<usize>()
    );
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
                eprintln!("HARDSEC-009 SCOAP rarity audit failed: {error}");
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
        let designs = [
            ("design-a", "family-a", "train", "1"),
            ("design-b", "family-b", "validation", "2"),
            ("design-c", "family-c", "test", "3"),
        ]
        .into_iter()
        .map(|(design_id, family, split, hash_digit)| {
            (
                design_id.to_owned(),
                DesignMeta {
                    design_id: design_id.to_owned(),
                    base_design_family_id: family.to_owned(),
                    split: split.to_owned(),
                    netlist_sha256: hash_digit.repeat(64),
                    netlist_path: None,
                },
            )
        })
        .collect();

        Header {
            dataset_id: "fixture".to_owned(),
            analyzer_name: "fixture-scoap".to_owned(),
            analyzer_version: "1.0".to_owned(),
            analyzer_config_sha256: "b".repeat(64),
            sequential_cut_policy: "register_cut".to_owned(),
            split_protocol: "per_design".to_owned(),
            alpha_milli,
            designs,
        }
    }

    fn node(node_id: u64, cc0: u64, cc1: u64, co: u64, label: NodeLabel, alpha: u64) -> Node {
        let max_cc = u128::from(cc0.max(cc1));
        Node {
            node_id,
            score_milli: max_cc * 1_000 + u128::from(co) * u128::from(alpha),
            label,
            metric_measured: true,
            rarity_percentile_bps: 0,
        }
    }

    fn header_value(alpha_milli: u64, protocol: &str) -> Value {
        json!({
            "schema_version": "hardsec-scoap-manifest-v1",
            "record_order": RECORD_ORDER,
            "dataset_id": "fixture",
            "alpha_milli": alpha_milli,
            "split_protocol": protocol,
            "analyzer": {
                "name": "fixture-scoap",
                "version": "1.0",
                "config_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "sequential_cut_policy": "register_cut"
            },
            "designs": [
                {
                    "design_id": "design-a",
                    "base_design_family_id": "family-a",
                    "split": "train",
                    "netlist_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                },
                {
                    "design_id": "design-b",
                    "base_design_family_id": "family-b",
                    "split": "validation",
                    "netlist_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
                },
                {
                    "design_id": "design-c",
                    "base_design_family_id": "family-c",
                    "split": "test",
                    "netlist_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
                }
            ]
        })
    }

    struct Fixture {
        base: PathBuf,
        manifest: PathBuf,
        ranking: PathBuf,
        receipt: PathBuf,
    }

    impl Fixture {
        fn new(node_rows: &[(&str, u64, &str, Option<(u64, u64, u64)>, &str)]) -> Self {
            let id = NEXT.fetch_add(1, Ordering::Relaxed);
            let base =
                env::temp_dir().join(format!("hardsec-scoap-audit-{}-{id}", std::process::id()));
            let input_dir = base.join("input");
            let output_dir = base.join("output");
            fs::create_dir_all(&input_dir).unwrap();
            fs::create_dir_all(&output_dir).unwrap();
            let manifest = input_dir.join("scoap.jsonl");
            let ranking = output_dir.join("ranking.tsv");
            let receipt = output_dir.join("receipt.json");

            let header = json!({
                "schema_version": "hardsec-scoap-manifest-v1",
                "record_order": RECORD_ORDER,
                "dataset_id": "fixture",
                "alpha_milli": 1000,
                "split_protocol": "per_design",
                "analyzer": {
                    "name": "fixture-scoap",
                    "version": "1.0",
                    "config_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                    "sequential_cut_policy": "register_cut"
                },
                "designs": [
                    {"design_id":"design-a","base_design_family_id":"family-a","split":"train","netlist_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
                    {"design_id":"design-b","base_design_family_id":"family-b","split":"validation","netlist_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},
                    {"design_id":"design-c","base_design_family_id":"family-c","split":"test","netlist_sha256":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"}
                ]
            });

            let mut content = serde_json::to_string(&header).unwrap();
            content.push('\n');
            for (design, node_id, label, metrics, metric_status) in node_rows {
                let value = match metrics {
                    Some((cc0, cc1, co)) => json!({
                        "design_id": design,
                        "node_id": node_id,
                        "label": label,
                        "metric_status": metric_status,
                        "cc0": cc0,
                        "cc1": cc1,
                        "co": co
                    }),
                    None => json!({
                        "design_id": design,
                        "node_id": node_id,
                        "label": label,
                        "metric_status": metric_status
                    }),
                };
                content.push_str(&serde_json::to_string(&value).unwrap());
                content.push('\n');
            }
            fs::write(&manifest, content).unwrap();
            Self {
                base,
                manifest,
                ranking,
                receipt,
            }
        }

        fn args(&self) -> Args {
            Args {
                manifest: self.manifest.clone(),
                ranking_output: self.ranking.clone(),
                receipt: self.receipt.clone(),
                evidence_root: None,
            }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.base);
        }
    }

    #[test]
    fn matches_composite_score_at_alpha_one() {
        let header = header(1_000);
        let value = json!({
            "design_id": "design-a",
            "node_id": 1,
            "metric_status": "measured",
            "cc0": 1,
            "cc1": 2,
            "co": 10,
            "label": "trigger"
        });
        let (_, value) = parse_node(&value, &header).unwrap();
        assert_eq!(value.score_milli, 12_000);
    }

    #[test]
    fn supports_half_weight_observability() {
        let header = header(500);
        let value = json!({
            "design_id": "design-a",
            "node_id": 1,
            "metric_status": "measured",
            "cc0": 1,
            "cc1": 2,
            "co": 10,
            "label": "trigger"
        });
        let (_, value) = parse_node(&value, &header).unwrap();
        assert_eq!(value.score_milli, 7_000);
    }

    #[test]
    fn ties_receive_equal_midrank_percentiles() {
        let mut nodes = vec![
            node(1, 1, 2, 10, NodeLabel::Trigger, 1_000),
            node(2, 1, 2, 10, NodeLabel::Benign, 1_000),
            node(3, 2, 2, 0, NodeLabel::Payload, 1_000),
        ];
        rank_design(&mut nodes).unwrap();
        let first = nodes.iter().find(|node| node.node_id == 1).unwrap();
        let second = nodes.iter().find(|node| node.node_id == 2).unwrap();
        assert_eq!(first.rarity_percentile_bps, second.rarity_percentile_bps);
        assert_eq!(first.rarity_percentile_bps, 6_666);
    }

    #[test]
    fn all_equal_scores_receive_midpoint_percentile() {
        let mut nodes = vec![
            node(1, 1, 1, 1, NodeLabel::Trigger, 1_000),
            node(2, 1, 1, 1, NodeLabel::Benign, 1_000),
            node(3, 1, 1, 1, NodeLabel::Payload, 1_000),
        ];
        rank_design(&mut nodes).unwrap();
        assert!(nodes.iter().all(|item| item.rarity_percentile_bps == 5_000));
    }

    #[test]
    fn percentile_ranking_is_per_design_not_pooled() {
        let mut low = vec![
            node(1, 1, 1, 1, NodeLabel::Trigger, 1_000),
            node(2, 2, 2, 2, NodeLabel::Benign, 1_000),
        ];
        let mut high = vec![
            node(1, 100, 100, 100, NodeLabel::Benign, 1_000),
            node(2, 200, 200, 200, NodeLabel::Payload, 1_000),
        ];
        rank_design(&mut low).unwrap();
        rank_design(&mut high).unwrap();
        assert_eq!(
            low.iter()
                .find(|node| node.node_id == 2)
                .unwrap()
                .rarity_percentile_bps,
            7_500
        );
        assert_eq!(
            high.iter()
                .find(|node| node.node_id == 2)
                .unwrap()
                .rarity_percentile_bps,
            7_500
        );
    }

    #[test]
    fn label_changes_do_not_change_scores_or_ranks() {
        let mut original = vec![
            node(1, 1, 4, 10, NodeLabel::Trigger, 1_000),
            node(2, 1, 1, 1, NodeLabel::Benign, 1_000),
        ];
        let mut changed = original.clone();
        changed[0].label = NodeLabel::Benign;
        changed[1].label = NodeLabel::Trigger;
        rank_design(&mut original).unwrap();
        rank_design(&mut changed).unwrap();
        for id in [1, 2] {
            let left = original.iter().find(|node| node.node_id == id).unwrap();
            let right = changed.iter().find(|node| node.node_id == id).unwrap();
            assert_eq!(left.score_milli, right.score_milli);
            assert_eq!(left.rarity_percentile_bps, right.rarity_percentile_bps);
        }
    }

    #[test]
    fn unknown_metrics_are_not_imputed() {
        let header = header(1_000);
        let value = json!({
            "design_id": "design-a",
            "node_id": 5,
            "metric_status": "unknown",
            "label": "unknown"
        });
        let (_, node) = parse_node(&value, &header).unwrap();
        assert!(!node.metric_measured);
        assert_eq!(node.score_milli, 0);
        assert_eq!(node.rarity_percentile_bps, 0);
    }

    #[test]
    fn measured_zero_is_distinct_from_unknown() {
        let header = header(1_000);
        let value = json!({
            "design_id": "design-a",
            "node_id": 6,
            "metric_status": "measured",
            "cc0": 0,
            "cc1": 0,
            "co": 0,
            "label": "unknown"
        });
        let (_, node) = parse_node(&value, &header).unwrap();
        assert!(node.metric_measured);
        let mut nodes = vec![node];
        assert_eq!(rank_design(&mut nodes).unwrap(), 0);
        assert_eq!(nodes[0].rarity_percentile_bps, 5_000);
    }

    #[test]
    fn unknown_labels_are_not_treated_as_benign() {
        let nodes = vec![
            Node {
                node_id: 1,
                score_milli: 1,
                label: NodeLabel::Unknown,
                metric_measured: true,
                rarity_percentile_bps: 10_000,
            },
            Node {
                node_id: 2,
                score_milli: 2,
                label: NodeLabel::Trigger,
                metric_measured: true,
                rarity_percentile_bps: 10_000,
            },
        ];
        let tails = evaluate_tails(&nodes);
        assert_eq!(tails[0].selected_measured_nodes, 2);
        assert_eq!(tails[0].selected_labeled_nodes, 1);
        assert_eq!(tails[0].selected_unknown_label_nodes, 1);
        assert_eq!(tails[0].trigger_precision, Some(1.0));
    }

    #[test]
    fn rejects_null_metric_fields_when_metric_status_is_unknown() {
        let header = header(1_000);
        let value = json!({
            "design_id": "design-a",
            "node_id": 5,
            "metric_status": "unknown",
            "cc0": null,
            "label": "unknown"
        });
        assert!(
            parse_node(&value, &header)
                .unwrap_err()
                .contains("unknown metric_status must omit cc0")
        );
    }

    #[test]
    fn verifies_exact_source_netlist_bytes() {
        let base = env::temp_dir().join(format!(
            "hardsec-scoap-source-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&base).unwrap();
        let path = base.join("design.v");
        fs::write(&path, b"module fixture; endmodule\\n").unwrap();
        let digest = sha256_file(&path).unwrap();
        let design = DesignMeta {
            design_id: "design-a".to_owned(),
            base_design_family_id: "family-a".to_owned(),
            split: "train".to_owned(),
            netlist_sha256: digest,
            netlist_path: Some("design.v".to_owned()),
        };
        verify_source_file(&base, &design).unwrap();
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn rejects_source_netlist_hash_mismatch() {
        let base = env::temp_dir().join(format!(
            "hardsec-scoap-source-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&base).unwrap();
        fs::write(base.join("design.v"), b"module fixture; endmodule\\n").unwrap();
        let design = DesignMeta {
            design_id: "design-a".to_owned(),
            base_design_family_id: "family-a".to_owned(),
            split: "train".to_owned(),
            netlist_sha256: "a".repeat(64),
            netlist_path: Some("design.v".to_owned()),
        };
        let error = verify_source_file(&base, &design).unwrap_err();
        assert!(error.contains("SHA-256 mismatch"));
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn rejects_source_path_traversal() {
        let base = env::temp_dir().join(format!(
            "hardsec-scoap-source-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&base).unwrap();
        let design = DesignMeta {
            design_id: "design-a".to_owned(),
            base_design_family_id: "family-a".to_owned(),
            split: "train".to_owned(),
            netlist_sha256: "a".repeat(64),
            netlist_path: Some("../outside.v".to_owned()),
        };
        assert!(
            verify_source_file(&base, &design)
                .unwrap_err()
                .contains("must not contain")
        );
        fs::remove_dir_all(base).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn rejects_source_netlist_symlink() {
        let base = env::temp_dir().join(format!(
            "hardsec-scoap-source-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&base).unwrap();
        let actual = base.join("actual.v");
        fs::write(&actual, b"module fixture; endmodule\\n").unwrap();
        std::os::unix::fs::symlink(&actual, base.join("design.v")).unwrap();
        let design = DesignMeta {
            design_id: "design-a".to_owned(),
            base_design_family_id: "family-a".to_owned(),
            split: "train".to_owned(),
            netlist_sha256: sha256_file(&actual).unwrap(),
            netlist_path: Some("design.v".to_owned()),
        };
        assert!(
            verify_source_file(&base, &design)
                .unwrap_err()
                .contains("symlink is forbidden")
        );
        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn rejects_invalid_alpha() {
        assert!(
            parse_header(&header_value(499, "per_design"))
                .unwrap_err()
                .contains("alpha_milli")
        );
        assert!(parse_header(&header_value(1_001, "per_design")).is_err());
    }

    #[test]
    fn rejects_family_leakage_and_duplicate_netlists_across_splits() {
        let mut family = header_value(1_000, "leave_one_base_family_out");
        family["designs"][1]["base_design_family_id"] = json!("family-a");
        assert!(
            parse_header(&family)
                .unwrap_err()
                .contains("appears in train and validation")
        );

        let mut duplicated = header_value(1_000, "leave_one_design_out");
        duplicated["designs"][1]["netlist_sha256"] =
            duplicated["designs"][0]["netlist_sha256"].clone();
        assert!(
            parse_header(&duplicated)
                .unwrap_err()
                .contains("identical netlist SHA-256")
        );
    }

    #[test]
    fn streamed_run_writes_label_free_rankings_and_receipt() {
        let fixture = Fixture::new(&[
            ("design-a", 1, "trigger", Some((1, 2, 10)), "measured"),
            ("design-a", 2, "benign", Some((1, 1, 1)), "measured"),
            ("design-b", 1, "payload", Some((2, 2, 2)), "measured"),
            ("design-c", 1, "unknown", None, "unknown"),
        ]);
        run(fixture.args()).unwrap();
        let ranking = fs::read_to_string(&fixture.ranking).unwrap();
        assert!(ranking.starts_with("design_id\tbase_design_family_id\tsplit\tnode_id"));
        assert!(!ranking.lines().any(|line| line.ends_with("\ttrigger")));
        assert!(!ranking.lines().any(|line| line.ends_with("\tbenign")));
        assert!(ranking.contains("unknown\tunknown"));

        let receipt: Value = serde_json::from_slice(&fs::read(&fixture.receipt).unwrap()).unwrap();
        assert_eq!(receipt["result"], "pass");
        assert_eq!(receipt["ranking_output"]["contains_labels"], false);
        assert_eq!(receipt["design_count"], 3);
        assert_eq!(receipt["total_unknown_metric_count"], 1);
    }

    #[test]
    fn rejects_out_of_order_designs_and_cleans_partial_ranking() {
        let fixture = Fixture::new(&[
            ("design-b", 1, "trigger", Some((1, 2, 10)), "measured"),
            ("design-a", 1, "benign", Some((1, 1, 1)), "measured"),
            ("design-c", 1, "unknown", None, "unknown"),
        ]);
        let error = run(fixture.args()).unwrap_err();
        assert!(error.contains("record order violation"));
        assert!(!fixture.ranking.exists());
        assert!(!fixture.receipt.exists());
    }

    #[test]
    fn rejects_nonincreasing_node_ids_within_design() {
        let fixture = Fixture::new(&[
            ("design-a", 2, "trigger", Some((1, 2, 10)), "measured"),
            ("design-a", 1, "benign", Some((1, 1, 1)), "measured"),
            ("design-b", 1, "payload", Some((2, 2, 2)), "measured"),
            ("design-c", 1, "unknown", None, "unknown"),
        ]);
        assert!(
            run(fixture.args())
                .unwrap_err()
                .contains("is not greater than previous node_id")
        );
    }

    #[test]
    fn rejects_output_inside_manifest_directory() {
        let mut fixture = Fixture::new(&[
            ("design-a", 1, "trigger", Some((1, 2, 10)), "measured"),
            ("design-b", 1, "payload", Some((2, 2, 2)), "measured"),
            ("design-c", 1, "unknown", None, "unknown"),
        ]);
        fixture.ranking = fixture.manifest.parent().unwrap().join("ranking.tsv");
        assert!(
            run(fixture.args())
                .unwrap_err()
                .contains("outside input directory")
        );
    }
}
