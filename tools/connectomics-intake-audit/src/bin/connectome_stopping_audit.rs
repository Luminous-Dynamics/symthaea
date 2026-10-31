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

    /// Minimum prior snapshots required before issuing a one-step-ahead forecast.
    #[arg(long, default_value_t = 4)]
    forecast_min_training_snapshots: usize,

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
    /// Explicit numerator/denominator evidence; never inferred from graph invariants.
    #[serde(default)]
    synaptic_completeness: Option<SynapticCompletenessEvidence>,
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

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SynapticCompletenessEvidence {
    /// Stable ID for the declared synapse population being measured.
    population_id: String,
    /// Digest of the population definition, including spatial and inclusion scope.
    population_definition_sha256: String,
    /// Digest of the exact synapse IDs included in the denominator at this snapshot.
    synapse_population_sha256: String,
    /// Digest of the exact cells considered fully proofread at this snapshot.
    fully_proofread_cell_set_sha256: String,
    /// Denominator: all synapses in the declared synapse population.
    total_synapse_count: u64,
    /// Numerator: synapses whose pre- and postsynaptic locations both belong to fully proofread cells.
    synapses_with_both_partners_in_fully_proofread_cells: u64,
}

impl SynapticCompletenessEvidence {
    fn fraction(&self) -> Option<f64> {
        (self.total_synapse_count > 0).then_some(
            self.synapses_with_both_partners_in_fully_proofread_cells as f64
                / self.total_synapse_count as f64
        )
    }
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
    /// Separate biological completeness observation; never inferred from graph invariants.
    synaptic_completeness: SynapticCompletenessTail,
    graph_invariants_stable_in_tail: Option<bool>,
    query_metrics_stable_in_tail: Option<bool>,
    forecast_min_training_snapshots: usize,
    walk_forward_forecast_status: &'static str,
    walk_forward_forecasts: Vec<WalkForwardForecast>,
    forecast_summaries: Vec<ForecastSeriesSummary>,
    recommendation: &'static str,
    status: &'static str,
    errors: Vec<String>,
    scope_note: &'static str,
}

#[derive(Debug, Serialize)]
struct SynapticCompletenessTail {
    status: &'static str,
    population_id: Option<String>,
    population_definition_sha256: Option<String>,
    observed_snapshot_count: usize,
    tail_window_snapshot_count: usize,
    latest_total_synapse_count: Option<u64>,
    latest_fully_proofread_partner_synapse_count: Option<u64>,
    latest_fraction: Option<f64>,
    stability: MetricStability,
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

#[derive(Debug, Serialize)]
struct WalkForwardForecast {
    metric_id: String,
    metric_kind: &'static str,
    training_snapshot_count: usize,
    training_through_snapshot_id: String,
    target_snapshot_id: String,
    target_sequence_index: u64,
    target_cumulative_edits: u64,
    actual_value: f64,
    persistence_prediction: f64,
    linear_prediction: Option<f64>,
    asymptotic_prediction: Option<f64>,
    fitted_asymptote: Option<f64>,
    fitted_decay_per_normalized_edit_span: Option<f64>,
    asymptotic_training_rmse: Option<f64>,
    persistence_absolute_error: f64,
    linear_absolute_error: Option<f64>,
    asymptotic_absolute_error: Option<f64>,
    asymptotic_status: &'static str,
}

#[derive(Debug, Serialize)]
struct ForecastSeriesSummary {
    metric_id: String,
    metric_kind: &'static str,
    forecast_count: usize,
    asymptotic_estimable_count: usize,
    asymptotic_mean_absolute_error: Option<f64>,
    persistence_mean_absolute_error: Option<f64>,
    linear_mean_absolute_error: Option<f64>,
    /// Baseline MAEs on the exact target snapshots where the asymptotic model was estimable.
    persistence_mae_on_asymptotic_estimable_cases: Option<f64>,
    linear_mae_on_asymptotic_estimable_cases: Option<f64>,
    asymptotic_relative_mae_improvement_over_persistence: Option<f64>,
    asymptotic_relative_mae_improvement_over_linear: Option<f64>,
}

#[derive(Debug, Clone, Copy)]
struct AsymptoticFit {
    origin_edits: f64,
    edit_span: f64,
    limit: f64,
    amplitude: f64,
    decay: f64,
    training_rmse: f64,
}

impl AsymptoticFit {
    fn predict(&self, cumulative_edits: u64) -> Option<f64> {
        let x = (cumulative_edits as f64 - self.origin_edits) / self.edit_span;
        let prediction = self.limit + self.amplitude * (-self.decay * x).exp();
        prediction.is_finite().then_some(prediction)
    }
}

#[derive(Debug, Clone, Copy)]
struct LinearFit {
    origin_edits: f64,
    edit_span: f64,
    intercept: f64,
    slope: f64,
}

impl LinearFit {
    fn predict(&self, cumulative_edits: u64) -> Option<f64> {
        let x = (cumulative_edits as f64 - self.origin_edits) / self.edit_span;
        let prediction = self.intercept + self.slope * x;
        prediction.is_finite().then_some(prediction)
    }
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
    if let Some(completeness) = &snapshot.synaptic_completeness {
        if completeness.population_id.trim().is_empty() {
            errors.push(format!("{prefix}: synaptic completeness population_id must be non-empty"));
        }
        for (field, digest) in [
            ("population_definition_sha256", completeness.population_definition_sha256.as_str()),
            ("synapse_population_sha256", completeness.synapse_population_sha256.as_str()),
            ("fully_proofread_cell_set_sha256", completeness.fully_proofread_cell_set_sha256.as_str()),
        ] {
            if !is_lower_hex(digest, 64) {
                errors.push(format!("{prefix}: synaptic completeness {field} must be 64 lowercase hexadecimal characters"));
            }
        }
        if completeness.synapses_with_both_partners_in_fully_proofread_cells
            > completeness.total_synapse_count
        {
            errors.push(format!("{prefix}: fully-proofread-partner synapse count exceeds the declared synapse population"));
        }
    }
    if let Some(previous_digest) = snapshot.snapshot.previous_graph_sha256.as_deref() {
        if !is_lower_hex(previous_digest, 64) {
            errors.push(format!("{prefix}: previous_graph_sha256 must be 64 lowercase hexadecimal characters"));
        }
    }
    if snapshot.snapshot.sequence_index > 0 && snapshot.snapshot.previous_graph_sha256.is_none() {
        errors.push(format!("{prefix}: non-initial snapshot must declare previous_graph_sha256"));
    }
    if snapshot.snapshot.sequence_index == 0 && snapshot.snapshot.previous_graph_sha256.is_some() {
        errors.push(format!("{prefix}: initial snapshot must not declare previous_graph_sha256"));
    }
    if snapshot.snapshot.sequence_index > 0 && snapshot.edits.edits_since_previous_snapshot.is_none() {
        errors.push(format!("{prefix}: non-initial snapshot must declare edits_since_previous_snapshot"));
    }
    if snapshot.snapshot.sequence_index == 0 && snapshot.edits.edits_since_previous_snapshot.is_some() {
        errors.push(format!("{prefix}: initial snapshot must not declare edits_since_previous_snapshot"));
    }
    errors
}

fn validate_series(snapshots: &[Snapshot]) -> Vec<String> {
    let mut errors = Vec::new();
    let mut ids = BTreeSet::new();

    if let Some(first) = snapshots.first() {
        if first.snapshot.sequence_index != 0 {
            errors.push(format!(
                "first snapshot sequence_index must be 0, found {}",
                first.snapshot.sequence_index
            ));
        }
        if first.snapshot.previous_graph_sha256.is_some() {
            errors.push("first snapshot must not declare previous_graph_sha256".into());
        }
    }

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
        if let (Some(current_completeness), Some(previous_completeness)) = (
            current.synaptic_completeness.as_ref(),
            previous.synaptic_completeness.as_ref(),
        ) {
            if current_completeness.population_id != previous_completeness.population_id
                || current_completeness.population_definition_sha256
                    != previous_completeness.population_definition_sha256
            {
                errors.push(format!(
                    "snapshot {:?}: synaptic completeness population identity/definition changed inside a series",
                    current.snapshot.snapshot_id
                ));
            }
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

fn fit_asymptotic(edit_counts: &[u64], values: &[f64]) -> Option<AsymptoticFit> {
    if edit_counts.len() != values.len() || edit_counts.len() < 4 || values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let origin = *edit_counts.first()? as f64;
    let span = *edit_counts.last()? as f64 - origin;
    if !span.is_finite() || span <= 0.0 {
        return None;
    }
    let xs: Vec<f64> = edit_counts.iter().map(|count| (*count as f64 - origin) / span).collect();
    let y_mean = values.iter().sum::<f64>() / values.len() as f64;
    let min_log_k = 0.01_f64.ln();
    let max_log_k = 100.0_f64.ln();
    let grid_size = 256_usize;
    let mut best: Option<AsymptoticFit> = None;

    for index in 0..grid_size {
        let fraction = index as f64 / (grid_size - 1) as f64;
        let decay = (min_log_k + fraction * (max_log_k - min_log_k)).exp();
        let zs: Vec<f64> = xs.iter().map(|x| (-decay * x).exp()).collect();
        let z_mean = zs.iter().sum::<f64>() / zs.len() as f64;
        let variance = zs.iter().map(|z| (z - z_mean).powi(2)).sum::<f64>();
        if variance <= 1e-14 {
            continue;
        }
        let covariance = zs.iter().zip(values).map(|(z, y)| (z - z_mean) * (y - y_mean)).sum::<f64>();
        let amplitude = covariance / variance;
        let limit = y_mean - amplitude * z_mean;
        let squared_error = zs.iter().zip(values).map(|(z, y)| {
            let predicted = limit + amplitude * z;
            (y - predicted).powi(2)
        }).sum::<f64>();
        if !squared_error.is_finite() || !limit.is_finite() || !amplitude.is_finite() {
            continue;
        }
        let fit = AsymptoticFit {
            origin_edits: origin,
            edit_span: span,
            limit,
            amplitude,
            decay,
            training_rmse: (squared_error / values.len() as f64).sqrt(),
        };
        if best.is_none_or(|old| fit.training_rmse < old.training_rmse) {
            best = Some(fit);
        }
    }
    best
}

fn fit_linear(edit_counts: &[u64], values: &[f64]) -> Option<LinearFit> {
    if edit_counts.len() != values.len() || edit_counts.len() < 2 || values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let origin = *edit_counts.first()? as f64;
    let span = *edit_counts.last()? as f64 - origin;
    if !span.is_finite() || span <= 0.0 {
        return None;
    }
    let xs: Vec<f64> = edit_counts.iter().map(|count| (*count as f64 - origin) / span).collect();
    let x_mean = xs.iter().sum::<f64>() / xs.len() as f64;
    let y_mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = xs.iter().map(|x| (x - x_mean).powi(2)).sum::<f64>();
    if variance <= 1e-14 {
        return None;
    }
    let covariance = xs.iter().zip(values).map(|(x, y)| (x - x_mean) * (y - y_mean)).sum::<f64>();
    let slope = covariance / variance;
    let intercept = y_mean - slope * x_mean;
    (slope.is_finite() && intercept.is_finite()).then_some(LinearFit {
        origin_edits: origin,
        edit_span: span,
        intercept,
        slope,
    })
}

fn mean_absolute_error(values: impl Iterator<Item = f64>) -> Option<f64> {
    let values: Vec<f64> = values.collect();
    if values.is_empty() { None } else { Some(values.iter().sum::<f64>() / values.len() as f64) }
}

fn relative_mae_improvement(baseline: Option<f64>, model: Option<f64>) -> Option<f64> {
    match (baseline, model) {
        (Some(base), Some(candidate)) if base.is_finite() && base > 0.0 && candidate.is_finite() =>
            Some((base - candidate) / base),
        _ => None,
    }
}

fn walk_forward_forecasts(
    snapshots: &[Snapshot],
    min_training_snapshots: usize,
) -> (Vec<WalkForwardForecast>, Vec<ForecastSeriesSummary>) {
    if snapshots.len() <= min_training_snapshots || min_training_snapshots < 4 {
        return (Vec::new(), Vec::new());
    }
    let edits: Vec<u64> = snapshots.iter().map(|snapshot| snapshot.edits.cumulative_committed).collect();
    let mut series: Vec<(String, &'static str, Vec<f64>)> = Vec::new();

    if let Some(first) = snapshots.first() {
        for name in first.graph.invariants.keys() {
            let values: Vec<f64> = snapshots.iter().filter_map(|snapshot| snapshot.graph.invariants.get(name).copied()).collect();
            if values.len() == snapshots.len() {
                series.push((format!("graph:{name}"), "graph_invariant", values));
            }
        }
        for (name, metric) in &first.query_evidence.metrics {
            let values: Vec<f64> = snapshots.iter().filter_map(|snapshot| snapshot.query_evidence.metrics.get(name).map(|value| value.value)).collect();
            if values.len() == snapshots.len() {
                series.push((format!("query:{name}"), "query_metric", values));
            }
            let _ = metric;
        }
    }

    let mut forecasts = Vec::new();
    let mut summaries = Vec::new();
    for (metric_id, metric_kind, values) in series {
        let series_forecasts_start = forecasts.len();
        for target_index in min_training_snapshots..snapshots.len() {
            let train_edits = &edits[..target_index];
            let train_values = &values[..target_index];
            let target = &snapshots[target_index];
            let previous_value = train_values[train_values.len() - 1];
            let asymptotic = fit_asymptotic(train_edits, train_values);
            let linear = fit_linear(train_edits, train_values);
            let asymptotic_prediction = asymptotic.and_then(|fit| fit.predict(target.edits.cumulative_committed));
            let linear_prediction = linear.and_then(|fit| fit.predict(target.edits.cumulative_committed));
            let actual_value = values[target_index];
            forecasts.push(WalkForwardForecast {
                metric_id: metric_id.clone(),
                metric_kind,
                training_snapshot_count: target_index,
                training_through_snapshot_id: snapshots[target_index - 1].snapshot.snapshot_id.clone(),
                target_snapshot_id: target.snapshot.snapshot_id.clone(),
                target_sequence_index: target.snapshot.sequence_index,
                target_cumulative_edits: target.edits.cumulative_committed,
                actual_value,
                persistence_prediction: previous_value,
                linear_prediction,
                asymptotic_prediction,
                fitted_asymptote: asymptotic.map(|fit| fit.limit),
                fitted_decay_per_normalized_edit_span: asymptotic.map(|fit| fit.decay),
                asymptotic_training_rmse: asymptotic.map(|fit| fit.training_rmse),
                persistence_absolute_error: (actual_value - previous_value).abs(),
                linear_absolute_error: linear_prediction.map(|prediction| (actual_value - prediction).abs()),
                asymptotic_absolute_error: asymptotic_prediction.map(|prediction| (actual_value - prediction).abs()),
                asymptotic_status: if asymptotic_prediction.is_some() { "FORECASTED_NOT_PROMOTED" } else { "NOT_ESTIMABLE" },
            });
        }

        let observations = &forecasts[series_forecasts_start..];
        let asymptotic_mae_rows: Vec<&WalkForwardForecast> = observations.iter()
            .filter(|forecast| forecast.asymptotic_absolute_error.is_some())
            .collect();
        let asym_mae = mean_absolute_error(asymptotic_mae_rows.iter().filter_map(|forecast| forecast.asymptotic_absolute_error));
        let persistence_mae = mean_absolute_error(observations.iter().map(|forecast| forecast.persistence_absolute_error));
        let linear_mae = mean_absolute_error(observations.iter().filter_map(|forecast| forecast.linear_absolute_error));
        // Model deltas must be paired on the same target snapshots to avoid estimability-selection bias.
        let paired_persistence_mae = mean_absolute_error(asymptotic_mae_rows.iter().map(|forecast| forecast.persistence_absolute_error));
        let paired_linear_mae = mean_absolute_error(asymptotic_mae_rows.iter().filter_map(|forecast| forecast.linear_absolute_error));
        summaries.push(ForecastSeriesSummary {
            metric_id,
            metric_kind,
            forecast_count: observations.len(),
            asymptotic_estimable_count: asymptotic_mae_rows.len(),
            asymptotic_mean_absolute_error: asym_mae,
            persistence_mean_absolute_error: persistence_mae,
            linear_mean_absolute_error: linear_mae,
            persistence_mae_on_asymptotic_estimable_cases: paired_persistence_mae,
            linear_mae_on_asymptotic_estimable_cases: paired_linear_mae,
            asymptotic_relative_mae_improvement_over_persistence: relative_mae_improvement(paired_persistence_mae, asym_mae),
            asymptotic_relative_mae_improvement_over_linear: relative_mae_improvement(paired_linear_mae, asym_mae),
        });
    }
    (forecasts, summaries)
}

fn synaptic_completeness_tail(
    snapshots: &[Snapshot],
    tail_window: usize,
    relative_tolerance: f64,
    absolute_tolerance: f64,
) -> SynapticCompletenessTail {
    let observed = snapshots.iter()
        .filter_map(|snapshot| snapshot.synaptic_completeness.as_ref())
        .collect::<Vec<_>>();
    let latest = snapshots.last().and_then(|snapshot| snapshot.synaptic_completeness.as_ref());
    let tail_count = snapshots.len().min(tail_window);
    let tail_records = if tail_count == 0 {
        &[][..]
    } else {
        &snapshots[snapshots.len() - tail_count..]
    };
    let tail_values = tail_records.iter()
        .map(|snapshot| snapshot.synaptic_completeness.as_ref().and_then(SynapticCompletenessEvidence::fraction))
        .collect::<Vec<_>>();
    let tail_complete = snapshots.len() >= tail_window
        && tail_values.len() == tail_window
        && tail_values.iter().all(Option::is_some);
    let stability_report = if tail_complete {
        let values = tail_values.into_iter().flatten().collect::<Vec<_>>();
        stability(&values, tail_window, relative_tolerance, absolute_tolerance)
    } else {
        stability(&[], tail_window, relative_tolerance, absolute_tolerance)
    };
    let status = if observed.is_empty() {
        "NOT_PROVIDED"
    } else if snapshots.len() < tail_window {
        "INSUFFICIENT_TAIL_HISTORY"
    } else if !tail_complete {
        "INCOMPLETE_TAIL"
    } else {
        "AVAILABLE"
    };
    SynapticCompletenessTail {
        status,
        population_id: latest.map(|value| value.population_id.clone())
            .or_else(|| observed.first().map(|value| value.population_id.clone())),
        population_definition_sha256: latest.map(|value| value.population_definition_sha256.clone())
            .or_else(|| observed.first().map(|value| value.population_definition_sha256.clone())),
        observed_snapshot_count: observed.len(),
        tail_window_snapshot_count: tail_count,
        latest_total_synapse_count: latest.map(|value| value.total_synapse_count),
        latest_fully_proofread_partner_synapse_count: latest
            .map(|value| value.synapses_with_both_partners_in_fully_proofread_cells),
        latest_fraction: latest.and_then(SynapticCompletenessEvidence::fraction),
        stability: stability_report,
        scope_note: "Fraction is the declared numerator of synapses whose pre- and postsynaptic locations both belong to fully proofread cells divided by the declared total synapse population. Counts are accepted as independently supplied observations; this tool does not inspect synapse endpoints, validate fully-proofread cell membership, or infer completeness from graph invariants. Missing snapshots or zero denominators make tail stability not estimable.",
    }
}

fn build_report(bytes: &[u8], snapshots: &[Snapshot], args: &Args) -> AuditReport {
    let errors = validate_series(snapshots);
    let (walk_forward_forecasts, forecast_summaries) = if errors.is_empty() {
        walk_forward_forecasts(snapshots, args.forecast_min_training_snapshots)
    } else {
        (Vec::new(), Vec::new())
    };
    let forecast_status = if walk_forward_forecasts.iter().any(|forecast| forecast.asymptotic_prediction.is_some()) {
        "WALK_FORWARD_FORECASTS_RECORDED_NOT_PROMOTED"
    } else {
        "NOT_ESTIMABLE"
    };
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
        schema_version: 2,
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
        synaptic_completeness: synaptic_completeness_tail(
            snapshots,
            args.tail_window,
            args.relative_tolerance,
            args.absolute_tolerance,
        ),
        forecast_min_training_snapshots: args.forecast_min_training_snapshots,
        walk_forward_forecast_status: forecast_status,
        walk_forward_forecasts,
        forecast_summaries,
        // The snapshot schema has no independently estimated residual-error risk. Never authorize stopping.
        recommendation: "CONTINUE_REVIEW",
        status: if errors.is_empty() { "VALIDATED_NOT_STOPPED" } else { "FAIL" },
        errors,
        scope_note: "Tail stability is descriptive and the optional asymptotic model is a one-step-ahead walk-forward diagnostic compared with persistence and linear-trend baselines. Evaluation conditions on the target snapshot cumulative edit count, which is only known retrospectively; it does not forecast elapsed time or edits required to meet a quality threshold, and does not yet provide calibrated prediction intervals. Neither stability nor forecast fit proves biological correctness. The tool always recommends CONTINUE_REVIEW because residual high-impact-error risk and a valid stopping threshold are not established.",
    }
}

fn run() -> Result<i32, String> {
    let args = Args::parse();
    if args.tail_window < 2 {
        return Err("--tail-window must be at least 2".into());
    }
    if args.forecast_min_training_snapshots < 4 {
        return Err("--forecast-min-training-snapshots must be at least 4".into());
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

    fn completeness(
        index: u64,
        total_synapse_count: u64,
        fully_proofread_partner_count: u64,
    ) -> SynapticCompletenessEvidence {
        SynapticCompletenessEvidence {
            population_id: "synthetic-whole-volume-synapses".into(),
            population_definition_sha256: "e".repeat(64),
            synapse_population_sha256: "f".repeat(64),
            fully_proofread_cell_set_sha256: format!("{:0>64}", format!("{:x}", index + 10)),
            total_synapse_count,
            synapses_with_both_partners_in_fully_proofread_cells: fully_proofread_partner_count,
        }
    }

    fn attach_completeness(
        snapshots: &mut [Snapshot],
        total_synapse_count: u64,
        fully_proofread_partner_count: u64,
    ) {
        for snapshot in snapshots {
            snapshot.synaptic_completeness = Some(completeness(
                snapshot.snapshot.sequence_index,
                total_synapse_count,
                fully_proofread_partner_count,
            ));
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
            forecast_min_training_snapshots: 4,
            output: None,
        }
    }

    #[test]
    fn checked_in_false_stop_counterexample_separates_invariants_from_synaptic_completeness() {
        let bytes = include_str!("../../../../docs/neuroscience/examples/CONNECTOME_STOPPING_FALSE_STOP_COUNTEREXAMPLE.example.jsonl").as_bytes();
        let snapshots = parse_jsonl(bytes).expect("false-stop counterexample should parse");
        let report = build_report(bytes, &snapshots, &args(3));
        assert_eq!(report.status, "VALIDATED_NOT_STOPPED", "{:?}", report.errors);
        assert_eq!(report.graph_invariants_stable_in_tail, Some(true));
        assert_eq!(report.query_metrics_stable_in_tail, Some(false));
        assert_eq!(report.synaptic_completeness.status, "AVAILABLE");
        assert_eq!(report.synaptic_completeness.observed_snapshot_count, 3);
        assert_eq!(report.synaptic_completeness.latest_total_synapse_count, Some(10_000));
        assert_eq!(report.synaptic_completeness.latest_fully_proofread_partner_synapse_count, Some(2_000));
        assert!((report.synaptic_completeness.latest_fraction.unwrap() - 0.20).abs() < 1.0e-12);
        assert_eq!(report.synaptic_completeness.stability.stable_under_declared_tolerance, Some(true));
        assert_eq!(report.recommendation, "CONTINUE_REVIEW");
    }

    #[test]
    fn missing_synaptic_completeness_in_tail_is_not_silently_filtered() {
        let mut snapshots = vec![
            snapshot(0, 0, &"1".repeat(64), None, 1.0, 12.0),
            snapshot(1, 1, &"2".repeat(64), Some(&"1".repeat(64)), 1.0, 12.0),
            snapshot(2, 2, &"3".repeat(64), Some(&"2".repeat(64)), 1.0, 12.0),
        ];
        snapshots[0].synaptic_completeness = Some(completeness(0, 100, 20));
        snapshots[1].synaptic_completeness = Some(completeness(1, 100, 20));
        let report = build_report(&jsonl(&snapshots), &snapshots, &args(3));
        assert_eq!(report.synaptic_completeness.status, "INCOMPLETE_TAIL");
        assert_eq!(report.synaptic_completeness.latest_fraction, None);
        assert_eq!(report.synaptic_completeness.stability.stable_under_declared_tolerance, None);
        assert_eq!(report.recommendation, "CONTINUE_REVIEW");
    }

    #[test]
    fn synaptic_completeness_absence_is_reported_not_inferred_from_graph_invariants() {
        let snapshots = vec![
            snapshot(0, 0, &"1".repeat(64), None, 1.0, 12.0),
            snapshot(1, 1, &"2".repeat(64), Some(&"1".repeat(64)), 1.0, 12.0),
            snapshot(2, 2, &"3".repeat(64), Some(&"2".repeat(64)), 1.0, 12.0),
        ];
        let report = build_report(&jsonl(&snapshots), &snapshots, &args(3));
        assert_eq!(report.graph_invariants_stable_in_tail, Some(true));
        assert_eq!(report.synaptic_completeness.status, "NOT_PROVIDED");
        assert_eq!(report.synaptic_completeness.latest_fraction, None);
        assert_eq!(report.synaptic_completeness.stability.stable_under_declared_tolerance, None);
        assert_eq!(report.recommendation, "CONTINUE_REVIEW");
    }

    #[test]
    fn invalid_synaptic_completeness_counts_and_digests_fail_closed() {
        let mut item = snapshot(0, 0, &"1".repeat(64), None, 1.0, 12.0);
        let mut evidence = completeness(0, 100, 20);
        evidence.synapses_with_both_partners_in_fully_proofread_cells = 101;
        evidence.synapse_population_sha256 = "bad".into();
        item.synaptic_completeness = Some(evidence);
        let report = build_report(&jsonl(&[item.clone()]), &[item], &args(2));
        assert!(report.errors.iter().any(|error| error.contains("synaptic completeness synapse_population_sha256")));
        assert!(report.errors.iter().any(|error| error.contains("count exceeds the declared synapse population")));
        assert_eq!(report.recommendation, "CONTINUE_REVIEW");
    }

    #[test]
    fn synaptic_completeness_population_definition_cannot_change_inside_series() {
        let mut a = snapshot(0, 0, &"1".repeat(64), None, 1.0, 12.0);
        let mut b = snapshot(1, 1, &"2".repeat(64), Some(&"1".repeat(64)), 1.0, 12.0);
        a.synaptic_completeness = Some(completeness(0, 100, 20));
        let mut second = completeness(1, 100, 21);
        second.population_definition_sha256 = "9".repeat(64);
        b.synaptic_completeness = Some(second);
        let report = build_report(&jsonl(&[a.clone(), b.clone()]), &[a, b], &args(2));
        assert!(report.errors.iter().any(|error| error.contains("population identity/definition changed")));
    }

    #[test]
    fn checked_in_walk_forward_fixture_emits_forecasts_and_baseline_metrics() {
        let bytes = include_str!("../../../../docs/neuroscience/examples/CONNECTOME_WALK_FORWARD_SNAPSHOTS.example.jsonl").as_bytes();
        let snapshots = parse_jsonl(bytes).expect("forecast fixture should parse");
        let report = build_report(bytes, &snapshots, &args(3));
        assert_eq!(report.status, "VALIDATED_NOT_STOPPED", "{:?}", report.errors);
        assert_eq!(report.walk_forward_forecast_status, "WALK_FORWARD_FORECASTS_RECORDED_NOT_PROMOTED");
        assert_eq!(report.walk_forward_forecasts.len(), 6);
        assert!(report.forecast_summaries.iter().any(|summary| {
            summary.metric_id == "query:path_count"
                && summary.asymptotic_estimable_count == summary.forecast_count
                && summary.persistence_mean_absolute_error.is_some()
                && summary.linear_mean_absolute_error.is_some()
        }));
        assert_eq!(report.recommendation, "CONTINUE_REVIEW");
    }

    #[test]
    fn walk_forward_forecast_does_not_leak_the_target_value() {
        let values = [10.0, 8.2, 7.0, 6.1, 5.7, 5.3];
        let mut snapshots: Vec<Snapshot> = Vec::new();
        for (index, value) in values.iter().enumerate() {
            let graph_digest = format!("{:064x}", index + 1);
            let previous_digest = if index == 0 { None } else { Some(format!("{:064x}", index)) };
            let mut item = snapshot(
                index as u64,
                (index as u64) * 10,
                &graph_digest,
                previous_digest.as_deref(),
                0.5 + index as f64 * 0.0001,
                *value,
            );
            if index > 0 {
                item.edits.edits_since_previous_snapshot = Some(10);
            }
            snapshots.push(item);
        }
        let before = walk_forward_forecasts(&snapshots, 4).0;
        let original = before.iter().find(|forecast| forecast.metric_id == "query:path_count" && forecast.target_sequence_index == 4).unwrap();
        assert!(original.asymptotic_prediction.is_some());
        assert!(original.linear_prediction.is_some());
        snapshots[4].query_evidence.metrics.get_mut("path_count").unwrap().value = 500.0;
        let after = walk_forward_forecasts(&snapshots, 4).0;
        let changed_target = after.iter().find(|forecast| forecast.metric_id == "query:path_count" && forecast.target_sequence_index == 4).unwrap();
        assert_eq!(original.asymptotic_prediction, changed_target.asymptotic_prediction);
        assert_eq!(original.linear_prediction, changed_target.linear_prediction);
        assert_ne!(original.actual_value, changed_target.actual_value);
    }

    #[test]
    fn asymptotic_fit_is_not_estimable_without_four_snapshots_or_edit_span() {
        assert!(fit_asymptotic(&[0, 1, 2], &[3.0, 2.0, 1.0]).is_none());
        assert!(fit_asymptotic(&[5, 5, 5, 5], &[4.0, 3.0, 2.0, 1.0]).is_none());
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
        assert_eq!(report.schema_version, 2);
        assert_eq!(report.status, "VALIDATED_NOT_STOPPED");
        assert_eq!(report.recommendation, "CONTINUE_REVIEW");
        assert_eq!(report.snapshot_count, 3);
        assert_eq!(report.graph_invariants_stable_in_tail, Some(true));
        assert_eq!(report.query_metrics_stable_in_tail, Some(true));
    }

    #[test]
    fn series_starting_after_zero_fails_closed() {
        let later = snapshot(4, 4, &"4".repeat(64), Some(&"3".repeat(64)), 1.0, 4.0);
        let report = build_report(&jsonl(&[later.clone()]), &[later], &args(2));
        assert!(report.errors.iter().any(|error| error.contains("first snapshot sequence_index must be 0")));
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
    fn missing_edit_delta_and_malformed_parent_digest_fail_closed() {
        let first = snapshot(0, 0, &"1".repeat(64), None, 1.0, 4.0);
        let mut second = snapshot(1, 1, &"2".repeat(64), Some(&"1".repeat(64)), 1.0, 4.0);
        second.edits.edits_since_previous_snapshot = None;
        second.snapshot.previous_graph_sha256 = Some("bad-digest".into());
        let snapshots = vec![first, second];
        let report = build_report(&jsonl(&snapshots), &snapshots, &args(2));
        assert!(report.errors.iter().any(|error| error.contains("previous_graph_sha256 must be 64")));
        assert!(report.errors.iter().any(|error| error.contains("must declare edits_since_previous_snapshot")));
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
