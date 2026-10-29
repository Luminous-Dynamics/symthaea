//! Fail-closed validator and tail-stability diagnostic for connectome proofreading snapshots.
//!
//! This tool never authorizes stopping. Convergent graph statistics are proxies; this report
//! always returns CONTINUE_REVIEW until a separately validated error-risk gate exists.

use clap::Parser;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug, Parser)]
#[command(
    name = "connectome-stopping-audit",
    version,
    about = "Validate sequential connectome snapshots and report query-specific tail stability without authorizing stopping"
)]
struct Args {
    /// JSONL file containing one versioned snapshot object per line.
    #[arg(long)]
    input: PathBuf,

    /// Number of final snapshots used for the descriptive stability window.
    #[arg(long, default_value_t = 3)]
    tail_window: usize,

    /// Relative range tolerance for each metric across the tail window.
    #[arg(long, default_value_t = 0.01)]
    relative_tolerance: f64,

    /// Absolute range tolerance; can classify near-zero metrics as stable.
    #[arg(long, default_value_t = 1e-9)]
    absolute_tolerance: f64,

    /// Optional output JSON path; stdout when omitted.
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Snapshot {
    schema_version: u32,
    dataset: DatasetIdentity,
    snapshot: SnapshotIdentity,
    edits: EditCounts,
    graph: GraphSummary,
    query_evidence: QueryEvidence,
    provenance: Provenance,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct DatasetIdentity {
    dataset_id: String,
    dataset_revision: String,
    dataset_revision_commit_sha: String,
    source_volume_id: String,
    species: Option<String>,
    region: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SnapshotIdentity {
    snapshot_id: String,
    sequence_index: u64,
    captured_at: String,
    graph_sha256: String,
    previous_graph_sha256: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct EditCounts {
    cumulative_committed: u64,
    by_operation_class: BTreeMap<String, u64>,
    edits_since_previous_snapshot: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct GraphSummary {
    directed: bool,
    node_count: u64,
    edge_count: u64,
    invariants: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct QueryEvidence {
    query_id: String,
    query_definition_sha256: String,
    metrics: BTreeMap<String, QueryMetric>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct QueryMetric {
    value: f64,
    unit: String,
    lower: Option<f64>,
    upper: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Provenance {
    graph_computation_profile_sha256: String,
    edit_log_sha256: String,
    query_evaluation_profile_sha256: String,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    schema_version: u32,
    tool: &'static str,
    input_sha256: String,
    snapshot_count: usize,
    first_sequence_index: Option<u64>,
    last_sequence_index: Option<u64>,
    cumulative_edits: Option<u64>,
    dataset_id: Option<String>,
    dataset_revision: Option<String>,
    source_volume_id: Option<String>,
    query_id: Option<String>,
    invariant_tail: BTreeMap<String, MetricStability>,
    query_metric_tail: BTreeMap<String, MetricStability>,
    graph_invariants_stable_in_tail: Option<bool>,
    query_metrics_stable_in_tail: Option<bool>,
    recommendation: &'static str,
    status: &'static str,
    errors: Vec<String>,
    scope_note: &'static str,
}

#[derive(Debug, Serialize)]
struct MetricStability {
    tail_window_snapshots: usize,
    first_value: Option<f64>,
    last_value: Option<f64>,
    absolute_range: Option<f64>,
    relative_range: Option<f64>,
    stable_under_declared_tolerance: Option<bool>,
}

fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn parse_jsonl(bytes: &[u8]) -> Result<Vec<Snapshot>, String> {
    let text = std::str::from_utf8(bytes).map_err(|error| format!("input is not UTF-8: {error}"))?;
    let mut snapshots = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let snapshot = serde_json::from_str::<Snapshot>(line).map_err(|error| {
            format!("line {} is invalid or contains an unknown field: {error}", line_index + 1)
        })?;
        snapshots.push(snapshot);
    }
    if snapshots.is_empty() {
        return Err("input contains no snapshot records".into());
    }
    Ok(snapshots)
}

fn sum_operation_counts(snapshot: &Snapshot) -> Result<u64, String> {
    snapshot.edits.by_operation_class.values().try_fold(0_u64, |total, count| {
        total.checked_add(*count).ok_or_else(|| {
            format!("snapshot {} operation-class counts overflow u64", snapshot.snapshot.snapshot_id)
        })
    })
}

fn validate_snapshot(snapshot: &Snapshot, position: usize) -> Vec<String> {
    let mut errors = Vec::new();
    let prefix = format!("snapshot {:?} (position {position})", snapshot.snapshot.snapshot_id);

    if snapshot.schema_version != 1 {
        errors.push(format!("{prefix}: schema_version must be 1"));
    }
    for (field, value, length) in [
        ("dataset_revision_commit_sha", snapshot.dataset.dataset_revision_commit_sha.as_str(), 40),
        ("graph_sha256", snapshot.snapshot.graph_sha256.as_str(), 64),
        ("query_definition_sha256", snapshot.query_evidence.query_definition_sha256.as_str(), 64),
        ("graph_computation_profile_sha256", snapshot.provenance.graph_computation_profile_sha256.as_str(), 64),
        ("edit_log_sha256", snapshot.provenance.edit_log_sha256.as_str(), 64),
        ("query_evaluation_profile_sha256", snapshot.provenance.query_evaluation_profile_sha256.as_str(), 64),
    ] {
        if !is_lower_hex(value, length) {
            errors.push(format!("{prefix}: {field} must be {length} lowercase hexadecimal characters"));
        }
    }
    if snapshot.dataset.dataset_id.trim().is_empty()
        || snapshot.dataset.dataset_revision.trim().is_empty()
        || snapshot.dataset.source_volume_id.trim().is_empty()
        || snapshot.snapshot.snapshot_id.trim().is_empty()
        || snapshot.snapshot.captured_at.trim().is_empty()
        || snapshot.query_evidence.query_id.trim().is_empty()
    {
        errors.push(format!("{prefix}: dataset, snapshot, timestamp, and query identifiers must be non-empty"));
    }
    if snapshot.edits.by_operation_class.is_empty()
        || snapshot.edits.by_operation_class.keys().any(|name| name.trim().is_empty())
    {
        errors.push(format!("{prefix}: by_operation_class must contain non-empty operation names"));
    }
    match sum_operation_counts(snapshot) {
        Ok(total) if total != snapshot.edits.cumulative_committed => errors.push(format!(
            "{prefix}: operation-class sum {total} differs from cumulative_committed {}",
            snapshot.edits.cumulative_committed
        )),
        Err(error) => errors.push(error),
        _ => {}
    }
    if snapshot.graph.invariants.is_empty() {
        errors.push(format!("{prefix}: at least one graph invariant is required"));
    }
    for (name, value) in &snapshot.graph.invariants {
        if name.trim().is_empty() || !value.is_finite() {
            errors.push(format!("{prefix}: graph invariant names must be non-empty and values finite"));
        }
    }
    if snapshot.query_evidence.metrics.is_empty() {
        errors.push(format!("{prefix}: at least one query metric is required"));
    }
    for (name, metric) in &snapshot.query_evidence.metrics {
        if name.trim().is_empty()
            || metric.unit.trim().is_empty()
            || !metric.value.is_finite()
            || metric.lower.is_some_and(|value| !value.is_finite())
            || metric.upper.is_some_and(|value| !value.is_finite())
            || matches!((metric.lower, metric.upper), (Some(lower), Some(upper)) if lower > upper)
        {
            errors.push(format!("{prefix}: query metric {name:?} has invalid name, unit, bounds, or value"));
        }
    }
    if snapshot.snapshot.sequence_index > 0 && snapshot.snapshot.previous_graph_sha256.is_none() {
        errors.push(format!("{prefix}: non-initial snapshot must declare previous_graph_sha256"));
    }
    if snapshot.snapshot.sequence_index == 0 && snapshot.snapshot.previous_graph_sha256.is_some() {
        errors.push(format!("{prefix}: initial snapshot must not declare previous_graph_sha256"));
    }
    errors
}

fn validate_series(snapshots: &[Snapshot]) -> Vec<String> {
    let mut errors = Vec::new();
    let mut ids = BTreeSet::new();

    for (index, snapshot) in snapshots.iter().enumerate() {
        errors.extend(validate_snapshot(snapshot, index));
        if !ids.insert(snapshot.snapshot.snapshot_id.clone()) {
            errors.push(format!("duplicate snapshot_id {:?}", snapshot.snapshot.snapshot_id));
        }
        if index == 0 {
            continue;
        }
        let previous = &snapshots[index - 1];
        let current = snapshot;
        if current.snapshot.sequence_index <= previous.snapshot.sequence_index {
            errors.push(format!(
                "snapshot {:?}: sequence_index {} must be strictly greater than prior index {}",
                current.snapshot.snapshot_id,
                current.snapshot.sequence_index,
                previous.snapshot.sequence_index
            ));
        }
        if current.snapshot.previous_graph_sha256.as_deref() != Some(previous.snapshot.graph_sha256.as_str()) {
            errors.push(format!(
                "snapshot {:?}: previous_graph_sha256 does not match preceding graph digest",
                current.snapshot.snapshot_id
            ));
        }
        if current.edits.cumulative_committed < previous.edits.cumulative_committed {
            errors.push(format!(
                "snapshot {:?}: cumulative_committed decreased from {} to {}",
                current.snapshot.snapshot_id,
                previous.edits.cumulative_committed,
                current.edits.cumulative_committed
            ));
        }
        if let Some(delta) = current.edits.edits_since_previous_snapshot {
            if previous.edits.cumulative_committed.checked_add(delta) != Some(current.edits.cumulative_committed) {
                errors.push(format!(
                    "snapshot {:?}: edits_since_previous_snapshot is inconsistent with cumulative counts",
                    current.snapshot.snapshot_id
                ));
            }
        }
        if current.dataset != previous.dataset {
            errors.push(format!("snapshot {:?}: dataset identity changed inside a single series", current.snapshot.snapshot_id));
        }
        if current.query_evidence.query_id != previous.query_evidence.query_id
            || current.query_evidence.query_definition_sha256 != previous.query_evidence.query_definition_sha256
        {
            errors.push(format!("snapshot {:?}: query definition changed inside a single series", current.snapshot.snapshot_id));
        }
        if current.provenance.graph_computation_profile_sha256 != previous.provenance.graph_computation_profile_sha256
            || current.provenance.query_evaluation_profile_sha256 != previous.provenance.query_evaluation_profile_sha256
        {
            errors.push(format!("snapshot {:?}: graph/query computation profile changed inside a series", current.snapshot.snapshot_id));
        }
        if current.graph.directed != previous.graph.directed
            || current.graph.invariants.keys().ne(previous.graph.invariants.keys())
            || current.query_evidence.metrics.keys().ne(previous.query_evidence.metrics.keys())
        {
            errors.push(format!("snapshot {:?}: graph direction or metric schema changed inside a series", current.snapshot.snapshot_id));
        }
        if current.query_evidence.metrics.iter().any(|(name, metric)| {
            previous.query_evidence.metrics.get(name).is_none_or(|prior| prior.unit != metric.unit)
        }) {
            errors.push(format!("snapshot {:?}: query metric units changed inside a series", current.snapshot.snapshot_id));
        }
    }
    errors
}

fn stability(values: &[f64], tail_window: usize, rel_tol: f64, abs_tol: f64) -> MetricStability {
    if values.len() < tail_window || tail_window < 2 {
        return MetricStability {
            tail_window_snapshots: tail_window,
            first_value: None,
            last_value: values.last().copied(),
            absolute_range: None,
            relative_range: None,
            stable_under_declared_tolerance: None,
        };
    }
    let tail = &values[values.len() - tail_window..];
    let min = tail.iter().copied().fold(f64::INFINITY, f64::min);
    let max = tail.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;
    let scale = tail.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
    let relative = if scale <= f64::MIN_POSITIVE { 0.0 } else { range / scale };
    MetricStability {
        tail_window_snapshots: tail_window,
        first_value: tail.first().copied(),
        last_value: tail.last().copied(),
        absolute_range: Some(range),
        relative_range: Some(relative),
        stable_under_declared_tolerance: Some(range <= abs_tol || relative <= rel_tol),
    }
}

fn aggregate_stability(metrics: &BTreeMap<String, MetricStability>) -> Option<bool> {
    if metrics.is_empty() || metrics.values().any(|metric| metric.stable_under_declared_tolerance.is_none()) {
        return None;
    }
    Some(metrics.values().all(|metric| metric.stable_under_declared_tolerance == Some(true)))
}

fn build_report(bytes: &[u8], snapshots: &[Snapshot], args: &Args) -> AuditReport {
    let errors = validate_series(snapshots);
    let mut invariant_tail = BTreeMap::new();
    let mut query_metric_tail = BTreeMap::new();

    let invariant_names: BTreeSet<String> = snapshots.iter()
        .flat_map(|snapshot| snapshot.graph.invariants.keys().cloned())
        .collect();
    for name in invariant_names {
        let values: Vec<f64> = snapshots.iter()
            .filter_map(|snapshot| snapshot.graph.invariants.get(&name).copied())
            .collect();
        invariant_tail.insert(name, stability(&values, args.tail_window, args.relative_tolerance, args.absolute_tolerance));
    }
    let query_names: BTreeSet<String> = snapshots.iter()
        .flat_map(|snapshot| snapshot.query_evidence.metrics.keys().cloned())
        .collect();
    for name in query_names {
        let values: Vec<f64> = snapshots.iter()
            .filter_map(|snapshot| snapshot.query_evidence.metrics.get(&name).map(|metric| metric.value))
            .collect();
        query_metric_tail.insert(name, stability(&values, args.tail_window, args.relative_tolerance, args.absolute_tolerance));
    }

    let dataset = snapshots.first().map(|snapshot| snapshot.dataset.clone());
    let query_id = snapshots.first().map(|snapshot| snapshot.query_evidence.query_id.clone());
    AuditReport {
        schema_version: 1,
        tool: "connectome-stopping-audit",
        input_sha256: sha256(bytes),
        snapshot_count: snapshots.len(),
        first_sequence_index: snapshots.first().map(|snapshot| snapshot.snapshot.sequence_index),
        last_sequence_index: snapshots.last().map(|snapshot| snapshot.snapshot.sequence_index),
        cumulative_edits: snapshots.last().map(|snapshot| snapshot.edits.cumulative_committed),
        dataset_id: dataset.as_ref().map(|value| value.dataset_id.clone()),
        dataset_revision: dataset.as_ref().map(|value| value.dataset_revision.clone()),
        source_volume_id: dataset.map(|value| value.source_volume_id),
        query_id,
        graph_invariants_stable_in_tail: aggregate_stability(&invariant_tail),
        query_metrics_stable_in_tail: aggregate_stability(&query_metric_tail),
        invariant_tail,
        query_metric_tail,
        // The snapshot schema has no independently estimated residual-error risk. Never authorize stopping.
        recommendation: "CONTINUE_REVIEW",
        status: if errors.is_empty() { "VALIDATED_NOT_STOPPED" } else { "FAIL" },
        errors,
        scope_note: "Tail stability is a descriptive diagnostic, not a forecast or proof of biological correctness. The tool always recommends CONTINUE_REVIEW because this schema does not establish residual high-impact error risk or a valid stopping threshold. The caller-declared cluster/snapshot independence is not proven.",
    }
}

fn run() -> Result<i32, String> {
    let args = Args::parse();
    if args.tail_window < 2 {
        return Err("--tail-window must be at least 2".into());
    }
    if !args.relative_tolerance.is_finite() || !(0.0..=1.0).contains(&args.relative_tolerance) {
        return Err("--relative-tolerance must be finite and in [0,1]".into());
    }
    if !args.absolute_tolerance.is_finite() || args.absolute_tolerance < 0.0 {
        return Err("--absolute-tolerance must be finite and non-negative".into());
    }
    let bytes = fs::read(&args.input).map_err(|error| format!("failed to read {}: {error}", args.input.display()))?;
    let snapshots = parse_jsonl(&bytes)?;
    let report = build_report(&bytes, &snapshots, &args);
    let rendered = serde_json::to_string_pretty(&report).map_err(|error| format!("failed to serialize report: {error}"))? + "\n";
    if let Some(output) = args.output {
        fs::write(&output, rendered.as_bytes()).map_err(|error| format!("failed to write {}: {error}", output.display()))?;
    } else {
        print!("{rendered}");
    }
    Ok(if report.status == "VALIDATED_NOT_STOPPED" { 0 } else { 1 })
}

fn main() -> ExitCode {
    match run() {
        Ok(0) => ExitCode::SUCCESS,
        Ok(_) => ExitCode::from(1),
        Err(message) => {
            eprintln!("connectome-stopping-audit: {message}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(index: u64, edits: u64, graph_digest: &str, previous: Option<&str>, invariant: f64, metric: f64) -> Snapshot {
        let mut operation_counts = BTreeMap::new();
        operation_counts.insert("merge".to_string(), edits);
        let mut invariants = BTreeMap::new();
        invariants.insert("edge_density".to_string(), invariant);
        let mut query_metrics = BTreeMap::new();
        query_metrics.insert("path_count".to_string(), QueryMetric {
            value: metric,
            unit: "count".into(),
            lower: None,
            upper: None,
        });
        Snapshot {
            schema_version: 1,
            dataset: DatasetIdentity {
                dataset_id: "connectome-fixture".into(),
                dataset_revision: "v2".into(),
                dataset_revision_commit_sha: "a".repeat(40),
                source_volume_id: "volume-1".into(),
                species: Some("fly".into()),
                region: Some("hemibrain".into()),
            },
            snapshot: SnapshotIdentity {
                snapshot_id: format!("snapshot-{index}"),
                sequence_index: index,
                captured_at: format!("2026-10-29T00:00:{index:02}Z"),
                graph_sha256: graph_digest.into(),
                previous_graph_sha256: previous.map(str::to_owned),
            },
            edits: EditCounts {
                cumulative_committed: edits,
                by_operation_class: operation_counts,
                edits_since_previous_snapshot: if index == 0 { None } else { Some(1) },
            },
            graph: GraphSummary {
                directed: true,
                node_count: 100,
                edge_count: 200,
                invariants,
            },
            query_evidence: QueryEvidence {
                query_id: "shortest-path-study".into(),
                query_definition_sha256: "b".repeat(64),
                metrics: query_metrics,
            },
            provenance: Provenance {
                graph_computation_profile_sha256: "c".repeat(64),
                edit_log_sha256: format!("{:0>64}", format!("{:x}", index + 1)),
                query_evaluation_profile_sha256: "d".repeat(64),
            },
        }
    }

    fn jsonl(snapshots: &[Snapshot]) -> Vec<u8> {
        snapshots.iter().map(|snapshot| serde_json::to_string(snapshot).unwrap())
            .collect::<Vec<_>>().join("\n").into_bytes()
    }

    fn args(tail_window: usize) -> Args {
        Args {
            input: PathBuf::from("unused.jsonl"),
            tail_window,
            relative_tolerance: 0.01,
            absolute_tolerance: 1e-9,
            output: None,
        }
    }

    #[test]
    fn checked_in_example_is_valid_and_shows_query_instability() {
        let bytes = include_str!("../../../../docs/neuroscience/examples/CONNECTOME_STOPPING_SNAPSHOTS.example.jsonl").as_bytes();
        let snapshots = parse_jsonl(bytes).expect("example should parse");
        let report = build_report(bytes, &snapshots, &args(3));
        assert_eq!(report.status, "VALIDATED_NOT_STOPPED", "{:?}", report.errors);
        assert_eq!(report.graph_invariants_stable_in_tail, Some(true));
        assert_eq!(report.query_metrics_stable_in_tail, Some(false));
        assert_eq!(report.recommendation, "CONTINUE_REVIEW");
    }

    #[test]
    fn valid_series_reports_validated_but_never_stopped() {
        let a = snapshot(0, 0, &"1".repeat(64), None, 0.5, 12.0);
        let b = snapshot(1, 1, &"2".repeat(64), Some(&"1".repeat(64)), 0.501, 12.01);
        let c = snapshot(2, 2, &"3".repeat(64), Some(&"2".repeat(64)), 0.502, 12.02);
        let bytes = jsonl(&[a.clone(), b.clone(), c.clone()]);
        let snapshots = parse_jsonl(&bytes).unwrap();
        let report = build_report(&bytes, &snapshots, &args(3));
        assert_eq!(report.status, "VALIDATED_NOT_STOPPED");
        assert_eq!(report.recommendation, "CONTINUE_REVIEW");
        assert_eq!(report.snapshot_count, 3);
        assert_eq!(report.graph_invariants_stable_in_tail, Some(true));
        assert_eq!(report.query_metrics_stable_in_tail, Some(true));
    }

    #[test]
    fn broken_parent_digest_fails_closed() {
        let a = snapshot(0, 0, &"1".repeat(64), None, 1.0, 4.0);
        let b = snapshot(1, 1, &"2".repeat(64), Some(&"f".repeat(64)), 1.0, 4.0);
        let snapshots = vec![a, b];
        let report = build_report(&jsonl(&snapshots), &snapshots, &args(2));
        assert!(report.errors.iter().any(|error| error.contains("previous_graph_sha256")));
    }

    #[test]
    fn decreasing_edit_count_fails_closed() {
        let a = snapshot(0, 5, &"1".repeat(64), None, 1.0, 4.0);
        let b = snapshot(1, 4, &"2".repeat(64), Some(&"1".repeat(64)), 1.0, 4.0);
        let snapshots = vec![a, b];
        let report = build_report(&jsonl(&snapshots), &snapshots, &args(2));
        assert!(report.errors.iter().any(|error| error.contains("cumulative_committed decreased")));
    }

    #[test]
    fn mismatched_operation_counts_fail_closed() {
        let mut a = snapshot(0, 3, &"1".repeat(64), None, 1.0, 4.0);
        a.edits.by_operation_class.insert("split".into(), 1);
        let report = build_report(&jsonl(&[a.clone()]), &[a], &args(2));
        assert!(report.errors.iter().any(|error| error.contains("operation-class sum")));
    }

    #[test]
    fn query_change_inside_series_fails_closed() {
        let a = snapshot(0, 0, &"1".repeat(64), None, 1.0, 4.0);
        let mut b = snapshot(1, 1, &"2".repeat(64), Some(&"1".repeat(64)), 1.0, 4.0);
        b.query_evidence.query_id = "different-query".into();
        let report = build_report(&jsonl(&[a.clone(), b.clone()]), &[a, b], &args(2));
        assert!(report.errors.iter().any(|error| error.contains("query definition changed")));
    }

    #[test]
    fn tail_instability_is_reported_not_hidden_by_graph_invariants() {
        let a = snapshot(0, 0, &"1".repeat(64), None, 1.0, 10.0);
        let b = snapshot(1, 1, &"2".repeat(64), Some(&"1".repeat(64)), 1.0, 10.1);
        let c = snapshot(2, 2, &"3".repeat(64), Some(&"2".repeat(64)), 1.0, 13.0);
        let bytes = jsonl(&[a, b, c]);
        let snapshots = parse_jsonl(&bytes).unwrap();
        let report = build_report(&bytes, &snapshots, &args(3));
        assert_eq!(report.graph_invariants_stable_in_tail, Some(true));
        assert_eq!(report.query_metrics_stable_in_tail, Some(false));
        assert_eq!(report.recommendation, "CONTINUE_REVIEW");
    }

    #[test]
    fn insufficient_history_marks_stability_not_estimable() {
        let a = snapshot(0, 0, &"1".repeat(64), None, 1.0, 4.0);
        let b = snapshot(1, 1, &"2".repeat(64), Some(&"1".repeat(64)), 1.0, 4.0);
        let bytes = jsonl(&[a, b]);
        let snapshots = parse_jsonl(&bytes).unwrap();
        let report = build_report(&bytes, &snapshots, &args(3));
        assert_eq!(report.graph_invariants_stable_in_tail, None);
        assert_eq!(report.query_metrics_stable_in_tail, None);
    }
}
