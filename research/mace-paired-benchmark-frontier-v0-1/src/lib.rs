// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cluster-paired benchmark effects and a non-scalar quality/resource frontier.
//!
//! This evaluator consumes the matched-run contract. It requires identical
//! (independent_cluster_id, task_id) support across arms, averages within
//! clusters first, and bootstraps clusters rather than correlated task rows.
//! It never combines different benchmark tracks and never creates a weighted
//! composite score.

use std::collections::{BTreeMap, BTreeSet};

use mace_matched_benchmark_contract::{
    BenchmarkRunManifestV1, MatchedComparisonReceiptV1, MetricDirectionV1,
    validate_matched_comparison_v1,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const REPORT_SCHEMA_VERSION_V1: &str = "mace_paired_benchmark_frontier_v1";
pub const BOOTSTRAP_REPLICATES_V1: usize = 2_000;
pub const CONFIRMATORY_CLUSTER_FLOOR_V1: usize = 30;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeCustodyV1 {
    PublicDiagnosticOnly,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskOutcomeV1 {
    pub run_id: String,
    pub arm_id: String,
    pub independent_cluster_id: String,
    pub task_id: String,
    pub outcome_receipt_sha256: String,
    pub evaluator_custody: OutcomeCustodyV1,
    pub candidate_saw_evaluator_labels: bool,
    /// Exact metric IDs declared in the run manifest; values use the declared unit.
    pub metric_values: BTreeMap<String, f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfidenceIntervalV1 {
    pub lower: f64,
    pub upper: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrontierInferenceStatusV1 {
    Exploratory,
    ConfirmatoryClusterFloorMet,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArmFrontierPointV1 {
    pub arm_id: String,
    pub run_id: String,
    pub memory_method_id: String,
    pub independent_cluster_count: usize,
    pub paired_task_count: usize,
    pub inference_status: FrontierInferenceStatusV1,
    /// Equal-cluster-weighted means. Every metric remains in its native unit.
    pub metric_means: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairedMetricEffectV1 {
    pub baseline_arm_id: String,
    pub candidate_arm_id: String,
    pub metric_id: String,
    pub unit: String,
    pub direction: MetricDirectionV1,
    /// Candidate minus baseline in the metric's native unit.
    pub mean_candidate_minus_baseline: f64,
    /// Positive is better for Higher/Lower metrics; None for DiagnosticOnly.
    pub mean_directional_gain: Option<f64>,
    /// Interval for directionally oriented gain, or raw difference for diagnostics.
    pub confidence_interval_95: ConfidenceIntervalV1,
    pub worst_cluster_directional_gain: Option<f64>,
    pub independent_cluster_count: usize,
    pub paired_task_count: usize,
    pub inference_status: FrontierInferenceStatusV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairedBenchmarkFrontierReportV1 {
    pub schema_version: String,
    pub record_type: String,
    pub comparison_group_id: String,
    pub benchmark_id: String,
    pub track_id: String,
    pub matched_comparison_receipt_sha256: String,
    pub metric_definition_set_sha256: String,
    pub outcome_set_sha256: String,
    pub outcome_count: usize,
    pub evaluator_custody: OutcomeCustodyV1,
    pub candidate_saw_evaluator_labels: bool,
    pub baseline_arm_id: String,
    pub bootstrap_seed: u64,
    pub bootstrap_replicates: usize,
    pub confirmatory_cluster_floor: usize,
    pub inference_status: FrontierInferenceStatusV1,
    pub inference_unit: String,
    pub cluster_weighting: String,
    pub arm_points: Vec<ArmFrontierPointV1>,
    pub pairwise_effects: Vec<PairedMetricEffectV1>,
    /// Pareto-efficient arms over all Higher/Lower objectives; no weighted score.
    pub pareto_frontier_arm_ids: Vec<String>,
    pub objective_metric_ids: Vec<String>,
    pub diagnostic_metric_ids: Vec<String>,
    pub canonical_sha256: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrontierRejectV1 {
    InvalidReceipt,
    InvalidManifestSet,
    InvalidBaselineArm,
    InvalidOutcome,
    InvalidDigest,
    InvalidOutcomeReceipt,
    OutcomePayloadMismatch,
    CandidateSawEvaluatorLabels,
    UnknownRun,
    UnknownArm,
    DuplicateTaskOutcome,
    MetricSetMismatch,
    NonFiniteMetric,
    MissingArmOutcomes,
    MismatchedTaskCoverage,
    TaskCountMismatch,
    ClusterCountMismatch,
    InsufficientIndependentClusters,
    NoParetoObjectives,
    SerializationFailure,
}

#[derive(Clone, Copy, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed ^ 0x9e37_79b9_7f4a_7c15 }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn index(&mut self, length: usize) -> usize {
        (((self.next() as u128) * (length as u128)) >> 64) as usize
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (97..=102).contains(&byte))
}

fn digest_serialized<T: Serialize>(value: &T) -> Result<String, FrontierRejectV1> {
    serde_json::to_vec(value)
        .map(|bytes| sha256_hex(&bytes))
        .map_err(|_| FrontierRejectV1::SerializationFailure)
}

fn percentile(sorted: &[f64], probability: f64) -> f64 {
    let position = (sorted.len() - 1) as f64 * probability;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;
    sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction
}

fn cluster_bootstrap_ci(values: &[f64], seed: u64) -> ConfidenceIntervalV1 {
    let mut rng = SplitMix64::new(seed);
    let mut draws = Vec::with_capacity(BOOTSTRAP_REPLICATES_V1);
    for _ in 0..BOOTSTRAP_REPLICATES_V1 {
        let mut total = 0.0;
        for _ in 0..values.len() {
            total += values[rng.index(values.len())];
        }
        draws.push(total / values.len() as f64);
    }
    draws.sort_by(f64::total_cmp);
    ConfidenceIntervalV1 {
        lower: percentile(&draws, 0.025),
        upper: percentile(&draws, 0.975),
    }
}

fn metric_seed(base: u64, left: &str, right: &str, metric: &str) -> u64 {
    let text = format!("{left}\0{right}\0{metric}");
    text.bytes().fold(base ^ 0xcbf2_9ce4_8422_2325, |state, byte| {
        (state ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

type TaskKey = (String, String);
type ArmRows = BTreeMap<TaskKey, BTreeMap<String, f64>>;

/// Compute the canonical payload digest for one task outcome. This binds
/// task identity and metric values; it is an integrity digest, not an evaluator signature.
pub fn task_outcome_payload_sha256_v1(
    outcome: &TaskOutcomeV1,
) -> Result<String, FrontierRejectV1> {
    digest_serialized(&(
        outcome.run_id.as_str(),
        outcome.arm_id.as_str(),
        outcome.independent_cluster_id.as_str(),
        outcome.task_id.as_str(),
        &outcome.metric_values,
    ))
}

fn validate_and_group_outcomes(
    manifests: &[BenchmarkRunManifestV1],
    outcomes: &[TaskOutcomeV1],
) -> Result<BTreeMap<String, ArmRows>, FrontierRejectV1> {
    let by_run: BTreeMap<&str, &BenchmarkRunManifestV1> = manifests
        .iter()
        .map(|manifest| (manifest.run_id.as_str(), manifest))
        .collect();
    let mut grouped: BTreeMap<String, ArmRows> = BTreeMap::new();

    for row in outcomes {
        let manifest = by_run.get(row.run_id.as_str()).copied().ok_or(FrontierRejectV1::UnknownRun)?;
        let expected_arm = manifest.arm_id.as_deref().ok_or(FrontierRejectV1::UnknownArm)?;
        if row.arm_id != expected_arm {
            return Err(FrontierRejectV1::UnknownArm);
        }
        if row.independent_cluster_id.trim().is_empty() || row.task_id.trim().is_empty() {
            return Err(FrontierRejectV1::InvalidOutcome);
        }
        if !valid_sha256(&row.outcome_receipt_sha256) {
            return Err(FrontierRejectV1::InvalidOutcomeReceipt);
        }
        if task_outcome_payload_sha256_v1(row)? != row.outcome_receipt_sha256 {
            return Err(FrontierRejectV1::OutcomePayloadMismatch);
        }
        if row.evaluator_custody != OutcomeCustodyV1::PublicDiagnosticOnly {
            return Err(FrontierRejectV1::InvalidOutcome);
        }
        if row.candidate_saw_evaluator_labels {
            return Err(FrontierRejectV1::CandidateSawEvaluatorLabels);
        }
        let definitions: BTreeSet<&str> = manifest.metrics.iter().map(|metric| metric.metric_id.as_str()).collect();
        let actual: BTreeSet<&str> = row.metric_values.keys().map(String::as_str).collect();
        if definitions != actual {
            return Err(FrontierRejectV1::MetricSetMismatch);
        }
        if row.metric_values.values().any(|value| !value.is_finite()) {
            return Err(FrontierRejectV1::NonFiniteMetric);
        }
        let rows = grouped.entry(row.arm_id.clone()).or_default();
        let key = (row.independent_cluster_id.clone(), row.task_id.clone());
        if rows.insert(key, row.metric_values.clone()).is_some() {
            return Err(FrontierRejectV1::DuplicateTaskOutcome);
        }
    }

    if grouped.len() != manifests.len() {
        return Err(FrontierRejectV1::MissingArmOutcomes);
    }

    let first_manifest = &manifests[0];
    let first_arm = first_manifest.arm_id.as_deref().ok_or(FrontierRejectV1::UnknownArm)?;
    let first_rows = grouped.get(first_arm).ok_or(FrontierRejectV1::MissingArmOutcomes)?;
    let first_keys: BTreeSet<&TaskKey> = first_rows.keys().collect();

    for manifest in manifests {
        let arm_id = manifest.arm_id.as_deref().ok_or(FrontierRejectV1::UnknownArm)?;
        let rows = grouped.get(arm_id).ok_or(FrontierRejectV1::MissingArmOutcomes)?;
        if rows.len() as u64 != manifest.scored_task_count {
            return Err(FrontierRejectV1::TaskCountMismatch);
        }
        let clusters: BTreeSet<&str> = rows.keys().map(|(cluster, _)| cluster.as_str()).collect();
        if clusters.len() as u64 != manifest.independent_cluster_count {
            return Err(FrontierRejectV1::ClusterCountMismatch);
        }
        if clusters.len() < 2 {
            return Err(FrontierRejectV1::InsufficientIndependentClusters);
        }
        let keys: BTreeSet<&TaskKey> = rows.keys().collect();
        if keys != first_keys {
            return Err(FrontierRejectV1::MismatchedTaskCoverage);
        }
    }
    Ok(grouped)
}

fn cluster_means(
    rows: &ArmRows,
    metric_id: &str,
) -> BTreeMap<String, f64> {
    let mut grouped: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for ((cluster, _task), values) in rows {
        if let Some(value) = values.get(metric_id) {
            grouped.entry(cluster.as_str()).or_default().push(*value);
        }
    }
    grouped.into_iter()
        .map(|(cluster, values)| {
            (cluster.to_owned(), values.iter().sum::<f64>() / values.len() as f64)
        })
        .collect()
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let collected: Vec<f64> = values.collect();
    collected.iter().sum::<f64>() / collected.len() as f64
}

fn oriented_gain(direction: MetricDirectionV1, candidate: f64, baseline: f64) -> Option<f64> {
    match direction {
        MetricDirectionV1::HigherIsBetter => Some(candidate - baseline),
        MetricDirectionV1::LowerIsBetter => Some(baseline - candidate),
        MetricDirectionV1::DiagnosticOnly => None,
    }
}

fn arm_frontier_point(
    manifest: &BenchmarkRunManifestV1,
    rows: &ArmRows,
) -> ArmFrontierPointV1 {
    let metric_means = manifest.metrics.iter().map(|metric| {
        let clusters = cluster_means(rows, &metric.metric_id);
        let value = mean(clusters.values().copied());
        (metric.metric_id.clone(), value)
    }).collect();
    let cluster_count = rows.keys().map(|(cluster, _)| cluster).collect::<BTreeSet<_>>().len();
    ArmFrontierPointV1 {
        arm_id: manifest.arm_id.clone().unwrap_or_default(),
        run_id: manifest.run_id.clone(),
        memory_method_id: manifest.memory_method_id.clone(),
        independent_cluster_count: cluster_count,
        paired_task_count: rows.len(),
        inference_status: if cluster_count >= CONFIRMATORY_CLUSTER_FLOOR_V1 {
            FrontierInferenceStatusV1::ConfirmatoryClusterFloorMet
        } else {
            FrontierInferenceStatusV1::Exploratory
        },
        metric_means,
    }
}

fn paired_metric_effects(
    baseline_manifest: &BenchmarkRunManifestV1,
    candidate_manifest: &BenchmarkRunManifestV1,
    baseline_rows: &ArmRows,
    candidate_rows: &ArmRows,
    seed: u64,
) -> Vec<PairedMetricEffectV1> {
    let baseline_arm = baseline_manifest.arm_id.clone().unwrap_or_default();
    let candidate_arm = candidate_manifest.arm_id.clone().unwrap_or_default();
    let cluster_ids: BTreeSet<String> = baseline_rows.keys().map(|(cluster, _)| cluster.clone()).collect();
    baseline_manifest.metrics.iter().map(|metric| {
        let mut cluster_raw = Vec::with_capacity(cluster_ids.len());
        let mut cluster_gain = Vec::with_capacity(cluster_ids.len());
        for cluster_id in &cluster_ids {
            let mut raw_deltas = Vec::new();
            let mut gains = Vec::new();
            for ((cluster, task), baseline_values) in baseline_rows {
                if cluster != cluster_id { continue; }
                let candidate_values = &candidate_rows[&(cluster.clone(), task.clone())];
                let baseline_value = baseline_values[&metric.metric_id];
                let candidate_value = candidate_values[&metric.metric_id];
                raw_deltas.push(candidate_value - baseline_value);
                if let Some(gain) = oriented_gain(metric.direction, candidate_value, baseline_value) {
                    gains.push(gain);
                }
            }
            cluster_raw.push(raw_deltas.iter().sum::<f64>() / raw_deltas.len() as f64);
            let values = if gains.is_empty() { &raw_deltas } else { &gains };
            cluster_gain.push(values.iter().sum::<f64>() / values.len() as f64);
        }
        let mean_raw = mean(cluster_raw.iter().copied());
        let mean_gain = if metric.direction == MetricDirectionV1::DiagnosticOnly {
            None
        } else {
            Some(mean(cluster_gain.iter().copied()))
        };
        let ci = cluster_bootstrap_ci(
            &cluster_gain,
            metric_seed(seed, &baseline_arm, &candidate_arm, &metric.metric_id),
        );
        let worst = cluster_gain.iter().copied().min_by(f64::total_cmp);
        PairedMetricEffectV1 {
            baseline_arm_id: baseline_arm.clone(),
            candidate_arm_id: candidate_arm.clone(),
            metric_id: metric.metric_id.clone(),
            unit: metric.unit.clone(),
            direction: metric.direction,
            mean_candidate_minus_baseline: mean_raw,
            mean_directional_gain: mean_gain,
            confidence_interval_95: ci,
            worst_cluster_directional_gain: if metric.direction == MetricDirectionV1::DiagnosticOnly { None } else { worst },
            independent_cluster_count: cluster_ids.len(),
            paired_task_count: baseline_rows.len(),
            inference_status: if cluster_ids.len() >= CONFIRMATORY_CLUSTER_FLOOR_V1 {
                FrontierInferenceStatusV1::ConfirmatoryClusterFloorMet
            } else {
                FrontierInferenceStatusV1::Exploratory
            },
        }
    }).collect()
}

fn dominates(left: &ArmFrontierPointV1, right: &ArmFrontierPointV1, manifests: &[BenchmarkRunManifestV1]) -> bool {
    let metrics = &manifests[0].metrics;
    let objectives: Vec<_> = metrics.iter().filter(|metric| metric.direction != MetricDirectionV1::DiagnosticOnly).collect();
    if objectives.is_empty() { return false; }
    let mut strictly_better = false;
    for metric in objectives {
        let left_value = left.metric_means[&metric.metric_id];
        let right_value = right.metric_means[&metric.metric_id];
        let better = match metric.direction {
            MetricDirectionV1::HigherIsBetter => left_value.total_cmp(&right_value),
            MetricDirectionV1::LowerIsBetter => right_value.total_cmp(&left_value),
            MetricDirectionV1::DiagnosticOnly => std::cmp::Ordering::Equal,
        };
        if better.is_lt() { return false; }
        if better.is_gt() { strictly_better = true; }
    }
    strictly_better
}

fn pareto_frontier(
    points: &[ArmFrontierPointV1],
    manifests: &[BenchmarkRunManifestV1],
) -> Vec<String> {
    points.iter().filter(|candidate| {
        !points.iter().any(|other| {
            other.arm_id != candidate.arm_id && dominates(other, candidate, manifests)
        })
    }).map(|point| point.arm_id.clone()).collect()
}

/// Analyze all arms in one matched comparison. The receipt is recomputed from
/// manifests and must match exactly. Every arm must provide identical task and
/// independent-cluster keys; missing cases are not silently dropped.
pub fn analyze_matched_frontier_v1(
    receipt: &MatchedComparisonReceiptV1,
    manifests: &[BenchmarkRunManifestV1],
    outcomes: &[TaskOutcomeV1],
    baseline_arm_id: &str,
    bootstrap_seed: u64,
) -> Result<PairedBenchmarkFrontierReportV1, FrontierRejectV1> {
    if !receipt.verify_integrity() { return Err(FrontierRejectV1::InvalidReceipt); }
    let recomputed = validate_matched_comparison_v1(manifests)
        .map_err(|_| FrontierRejectV1::InvalidManifestSet)?;
    if recomputed.receipt_sha256 != receipt.receipt_sha256 {
        return Err(FrontierRejectV1::InvalidReceipt);
    }
    let baseline_manifest = manifests.iter()
        .find(|manifest| manifest.arm_id.as_deref() == Some(baseline_arm_id))
        .ok_or(FrontierRejectV1::InvalidBaselineArm)?;
    let grouped = validate_and_group_outcomes(manifests, outcomes)?;
    let first_custody = outcomes.first().ok_or(FrontierRejectV1::InvalidOutcome)?;
    if first_custody.evaluator_custody != OutcomeCustodyV1::PublicDiagnosticOnly {
        return Err(FrontierRejectV1::InvalidOutcome);
    }
    if outcomes.iter().any(|row| row.candidate_saw_evaluator_labels) {
        return Err(FrontierRejectV1::CandidateSawEvaluatorLabels);
    }
    let mut canonical_outcomes = outcomes.to_vec();
    canonical_outcomes.sort_by(|left, right| {
        left.run_id.cmp(&right.run_id)
            .then_with(|| left.arm_id.cmp(&right.arm_id))
            .then_with(|| left.independent_cluster_id.cmp(&right.independent_cluster_id))
            .then_with(|| left.task_id.cmp(&right.task_id))
    });
    let outcome_set_sha256 = digest_serialized(&canonical_outcomes)?;
    let baseline_rows = grouped.get(baseline_arm_id).ok_or(FrontierRejectV1::MissingArmOutcomes)?;

    let mut arm_points = Vec::with_capacity(manifests.len());
    for manifest in manifests {
        let arm = manifest.arm_id.as_deref().ok_or(FrontierRejectV1::InvalidManifestSet)?;
        arm_points.push(arm_frontier_point(
            manifest,
            grouped.get(arm).ok_or(FrontierRejectV1::MissingArmOutcomes)?,
        ));
    }
    arm_points.sort_by(|left, right| left.arm_id.cmp(&right.arm_id));

    let mut pairwise_effects = Vec::new();
    for manifest in manifests {
        let candidate_arm = manifest.arm_id.as_deref().ok_or(FrontierRejectV1::InvalidManifestSet)?;
        if candidate_arm == baseline_arm_id { continue; }
        pairwise_effects.extend(paired_metric_effects(
            baseline_manifest,
            manifest,
            baseline_rows,
            grouped.get(candidate_arm).ok_or(FrontierRejectV1::MissingArmOutcomes)?,
            bootstrap_seed,
        ));
    }

    let objectives = baseline_manifest.metrics.iter()
        .filter(|metric| metric.direction != MetricDirectionV1::DiagnosticOnly)
        .map(|metric| metric.metric_id.clone())
        .collect::<Vec<_>>();
    if objectives.is_empty() { return Err(FrontierRejectV1::NoParetoObjectives); }
    let diagnostic_metric_ids = baseline_manifest.metrics.iter()
        .filter(|metric| metric.direction == MetricDirectionV1::DiagnosticOnly)
        .map(|metric| metric.metric_id.clone())
        .collect::<Vec<_>>();
    let frontier = pareto_frontier(&arm_points, manifests);
    let mut report = PairedBenchmarkFrontierReportV1 {
        schema_version: REPORT_SCHEMA_VERSION_V1.to_owned(),
        record_type: "paired_benchmark_frontier_report".to_owned(),
        comparison_group_id: receipt.comparison_group_id.clone(),
        benchmark_id: receipt.benchmark_id.clone(),
        track_id: receipt.track_id.clone(),
        matched_comparison_receipt_sha256: receipt.receipt_sha256.clone(),
        metric_definition_set_sha256: receipt.metric_definition_set_sha256.clone(),
        outcome_set_sha256,
        outcome_count: canonical_outcomes.len(),
        evaluator_custody: first_custody.evaluator_custody,
        candidate_saw_evaluator_labels: false,
        baseline_arm_id: baseline_arm_id.to_owned(),
        bootstrap_seed,
        bootstrap_replicates: BOOTSTRAP_REPLICATES_V1,
        confirmatory_cluster_floor: CONFIRMATORY_CLUSTER_FLOOR_V1,
        inference_status: if baseline_rows.keys().map(|(cluster, _)| cluster).collect::<BTreeSet<_>>().len() >= CONFIRMATORY_CLUSTER_FLOOR_V1 {
            FrontierInferenceStatusV1::ConfirmatoryClusterFloorMet
        } else {
            FrontierInferenceStatusV1::Exploratory
        },
        inference_unit: "independent_cluster".to_owned(),
        cluster_weighting: "equal_cluster_weight_after_within_cluster_task_mean".to_owned(),
        arm_points,
        pairwise_effects,
        pareto_frontier_arm_ids: frontier,
        objective_metric_ids: objectives,
        diagnostic_metric_ids,
        canonical_sha256: String::new(),
    };
    let mut unsigned = report.clone();
    unsigned.canonical_sha256.clear();
    report.canonical_sha256 = digest_serialized(&unsigned)?;
    Ok(report)
}

impl PairedBenchmarkFrontierReportV1 {
    pub fn verify_integrity(&self) -> bool {
        let mut unsigned = self.clone();
        unsigned.canonical_sha256.clear();
        digest_serialized(&unsigned)
            .map(|digest| digest == self.canonical_sha256)
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mace_matched_benchmark_contract::{
        compute_matched_controls_sha256_v1, EvidenceKindV1, MetricObservationV1,
        ResourceObservationV1,
    };

    fn digest(value: &str) -> String {
        sha256_hex(value.as_bytes())
    }

    fn manifest(arm: &str, method: &str) -> BenchmarkRunManifestV1 {
        let mut value = BenchmarkRunManifestV1 {
            schema_version: "mace_benchmark_run_manifest_v1".into(),
            record_type: "benchmark_run_manifest".into(),
            run_id: format!("run-{arm}"),
            evidence_kind: EvidenceKindV1::LocalExperiment,
            benchmark_id: "timlmembench".into(),
            track_id: "raw_native_en".into(),
            protocol_version: "timlmembench-v1".into(),
            protocol_sha256: digest("protocol"),
            evaluator_sha256: digest("evaluator"),
            dataset_id: "timlmembench".into(),
            dataset_revision_sha256: digest("dataset"),
            split_id: "dev".into(),
            split_sha256: digest("split"),
            seed_set_sha256: digest("seeds"),
            scored_task_count: 4,
            independent_cluster_count: 2,
            dependence_unit: "sample_id".into(),
            system_name: method.into(),
            system_revision_sha256: Some(digest(&format!("system-{method}"))),
            adapter_sha256: Some(digest(&format!("adapter-{method}"))),
            memory_method_id: method.into(),
            memory_method_family: method.into(),
            reader_model_id: "fixed-reader".into(),
            reader_model_revision_sha256: digest("reader"),
            reader_prompt_sha256: digest("prompt"),
            judge_model_id: "fixed-judge".into(),
            judge_model_revision_sha256: digest("judge"),
            tool_contract_sha256: digest("tools"),
            resource_budget_sha256: digest("budget"),
            context_token_budget: 16000,
            max_model_calls: 3,
            wall_clock_budget_ms: 200000,
            cpu_budget_microseconds: Some(1_000_000),
            gpu_budget_microseconds: Some(2_000_000),
            energy_budget_microwatt_hours: None,
            comparison_group_id: Some("paired-timl-raw-native".into()),
            arm_id: Some(arm.into()),
            matched_controls_sha256: None,
            candidate_saw_evaluator_labels: false,
            evaluator_isolated: true,
            evidence_uri: None,
            result_artifact_sha256: Some(digest(&format!("result-{arm}"))),
            resource_receipt_sha256: Some(digest(&format!("resource-{arm}"))),
            metrics: vec![
                MetricObservationV1 {
                    metric_id: "governance_score".into(),
                    value: 0.0,
                    unit: "fraction".into(),
                    direction: MetricDirectionV1::HigherIsBetter,
                    definition_sha256: digest("governance-definition"),
                    denominator: Some(4),
                },
                MetricObservationV1 {
                    metric_id: "p95_latency_ms".into(),
                    value: 100.0,
                    unit: "ms".into(),
                    direction: MetricDirectionV1::LowerIsBetter,
                    definition_sha256: digest("latency-definition"),
                    denominator: Some(4),
                },
                MetricObservationV1 {
                    metric_id: "index_bytes".into(),
                    value: 1000.0,
                    unit: "bytes".into(),
                    direction: MetricDirectionV1::LowerIsBetter,
                    definition_sha256: digest("index-definition"),
                    denominator: None,
                },
                MetricObservationV1 {
                    metric_id: "abstention_count".into(),
                    value: 0.0,
                    unit: "count".into(),
                    direction: MetricDirectionV1::DiagnosticOnly,
                    definition_sha256: digest("abstention-definition"),
                    denominator: Some(4),
                },
            ],
            resources: ResourceObservationV1 {
                p50_latency_ms: Some(50.0),
                p95_latency_ms: Some(100.0),
                p99_latency_ms: Some(150.0),
                input_tokens: Some(1000),
                output_tokens: Some(500),
                context_tokens_peak: Some(800),
                persistent_memory_bytes: Some(10000),
                index_bytes: Some(1000),
                provenance_sidecar_bytes: Some(500),
                tombstone_journal_bytes: Some(100),
                cpu_microseconds: Some(1000),
                gpu_microseconds: Some(2000),
                energy_microwatt_hours: None,
                monetary_micro_usd: Some(100),
            },
        };
        value.matched_controls_sha256 = Some(compute_matched_controls_sha256_v1(&value).unwrap());
        value
    }

    fn outcomes(arms: &[(&str, f64, f64, f64)]) -> Vec<TaskOutcomeV1> {
        let mut rows = Vec::new();
        for (arm, accuracy, latency, index_bytes) in arms {
            for cluster in ["sample-a", "sample-b"] {
                for task in ["case-1", "case-2"] {
                    rows.push(TaskOutcomeV1 {
                        run_id: format!("run-{arm}"),
                        arm_id: (*arm).into(),
                        independent_cluster_id: cluster.into(),
                        task_id: task.into(),
                        outcome_receipt_sha256: String::new(),
                        evaluator_custody: OutcomeCustodyV1::PublicDiagnosticOnly,
                        candidate_saw_evaluator_labels: false,
                        metric_values: BTreeMap::from([
                            ("governance_score".into(), *accuracy),
                            ("p95_latency_ms".into(), *latency),
                            ("index_bytes".into(), *index_bytes),
                            ("abstention_count".into(), 0.0),
                        ]),
                    });
                    let last = rows.last_mut().expect("just pushed an outcome");
                    last.outcome_receipt_sha256 = task_outcome_payload_sha256_v1(last).unwrap();
                }
            }
        }
        rows
    }

    fn paired(manifests: Vec<BenchmarkRunManifestV1>, rows: Vec<TaskOutcomeV1>, baseline: &str) -> PairedBenchmarkFrontierReportV1 {
        let receipt = validate_matched_comparison_v1(&manifests).unwrap();
        analyze_matched_frontier_v1(&receipt, &manifests, &rows, baseline, 42).unwrap()
    }

    #[test]
    fn paired_effects_orient_accuracy_latency_and_storage_correctly() {
        let manifests = vec![manifest("bm25", "bm25"), manifest("mace", "mace")];
        let rows = outcomes(&[("bm25",0.6,100.0,1000.0),("mace",0.8,80.0,700.0)]);
        let report = paired(manifests, rows, "bm25");
        let accuracy = report.pairwise_effects.iter().find(|e| e.metric_id=="governance_score").unwrap();
        let latency = report.pairwise_effects.iter().find(|e| e.metric_id=="p95_latency_ms").unwrap();
        let storage = report.pairwise_effects.iter().find(|e| e.metric_id=="index_bytes").unwrap();
        assert!((accuracy.mean_directional_gain.unwrap()-0.2).abs()<1e-12);
        assert!((latency.mean_directional_gain.unwrap()-20.0).abs()<1e-12);
        assert!((storage.mean_directional_gain.unwrap()-300.0).abs()<1e-12);
        assert!(accuracy.confidence_interval_95.lower>0.0);
        assert!(report.verify_integrity());
    }

    #[test]
    fn pareto_frontier_keeps_quality_latency_tradeoffs_without_weighting() {
        let manifests=vec![
            manifest("base","base"),
            manifest("balanced","balanced"),
            manifest("fast","fast"),
        ];
        let rows=outcomes(&[
            ("base",0.6,100.0,1000.0),
            ("balanced",0.8,90.0,800.0),
            ("fast",0.6,60.0,500.0),
        ]);
        let report=paired(manifests,rows,"base");
        assert!(!report.pareto_frontier_arm_ids.contains(&"base".to_owned()));
        assert!(report.pareto_frontier_arm_ids.contains(&"balanced".to_owned()));
        assert!(report.pareto_frontier_arm_ids.contains(&"fast".to_owned()));
        assert!(!report.objective_metric_ids.contains(&"abstention_count".to_owned()));
        assert!(report.diagnostic_metric_ids.contains(&"abstention_count".to_owned()));
    }

    #[test]
    fn repeated_tasks_do_not_inflate_independent_cluster_count() {
        let manifests=vec![manifest("base","base"),manifest("candidate","candidate")];
        let mut rows=outcomes(&[("base",0.6,100.0,1000.0),("candidate",0.8,90.0,800.0)]);
        for row in rows.clone() {
            let mut repeated=row;
            repeated.task_id=format!("repeat-{}",repeated.task_id);
            rows.push(repeated);
        }
        let mut adjusted=manifests.clone();
        for m in &mut adjusted { m.scored_task_count=8; }
        for m in &mut adjusted {
            m.matched_controls_sha256=Some(compute_matched_controls_sha256_v1(m).unwrap());
        }
        let report=paired(adjusted,rows,"base");
        assert_eq!(report.arm_points[0].independent_cluster_count,2);
        assert_eq!(report.arm_points[0].paired_task_count,8);
        assert!(report.pairwise_effects.iter().all(|effect|effect.independent_cluster_count==2));
    }

    #[test]
    fn task_coverage_must_match_exactly_across_arms() {
        let manifests=vec![manifest("base","base"),manifest("candidate","candidate")];
        let mut rows=outcomes(&[("base",0.6,100.0,1000.0),("candidate",0.8,90.0,800.0)]);
        rows.retain(|row| !(row.arm_id=="candidate" && row.task_id=="case-2" && row.independent_cluster_id=="sample-b"));
        assert_eq!(
            validate_matched_comparison_v1(&manifests).unwrap().receipt_sha256.len(),64
        );
        let receipt=validate_matched_comparison_v1(&manifests).unwrap();
        assert_eq!(
            analyze_matched_frontier_v1(&receipt,&manifests,&rows,"base",7),
            Err(FrontierRejectV1::TaskCountMismatch)
        );
    }

    #[test]
    fn duplicate_rows_nonfinite_values_and_wrong_metric_sets_fail() {
        let manifests=vec![manifest("base","base"),manifest("candidate","candidate")];
        let mut rows=outcomes(&[("base",0.6,100.0,1000.0),("candidate",0.8,90.0,800.0)]);
        rows.push(rows[0].clone());
        assert_eq!(
            validate_and_group_outcomes(&manifests,&rows),
            Err(FrontierRejectV1::DuplicateTaskOutcome)
        );
        let mut rows=outcomes(&[("base",0.6,100.0,1000.0),("candidate",0.8,90.0,800.0)]);
        rows[0].metric_values.insert("governance_score".into(),f64::NAN);
        assert_eq!(
            validate_and_group_outcomes(&manifests,&rows),
            Err(FrontierRejectV1::NonFiniteMetric)
        );
        let mut rows=outcomes(&[("base",0.6,100.0,1000.0),("candidate",0.8,90.0,800.0)]);
        rows[0].metric_values.remove("index_bytes");
        assert_eq!(
            validate_and_group_outcomes(&manifests,&rows),
            Err(FrontierRejectV1::MetricSetMismatch)
        );
    }

    #[test]
    fn wrong_receipt_and_baseline_arm_are_rejected() {
        let manifests=vec![manifest("base","base"),manifest("candidate","candidate")];
        let receipt=validate_matched_comparison_v1(&manifests).unwrap();
        let rows=outcomes(&[("base",0.6,100.0,1000.0),("candidate",0.8,90.0,800.0)]);
        let mut tampered=receipt.clone();
        tampered.track_id="other".into();
        assert_eq!(analyze_matched_frontier_v1(&tampered,&manifests,&rows,"base",7),Err(FrontierRejectV1::InvalidReceipt));
        assert_eq!(analyze_matched_frontier_v1(&receipt,&manifests,&rows,"absent",7),Err(FrontierRejectV1::InvalidBaselineArm));
    }

    #[test]
    fn every_arm_must_match_declared_task_and_cluster_counts() {
        let manifests=vec![manifest("base","base"),manifest("candidate","candidate")];
        let mut rows=outcomes(&[("base",0.6,100.0,1000.0),("candidate",0.8,90.0,800.0)]);
        rows.pop();
        assert_eq!(validate_and_group_outcomes(&manifests,&rows),Err(FrontierRejectV1::TaskCountMismatch));
    }

    #[test]
    fn diagnostic_only_metrics_have_no_directional_improvement_claim() {
        let manifests=vec![manifest("base","base"),manifest("candidate","candidate")];
        let rows=outcomes(&[("base",0.6,100.0,1000.0),("candidate",0.8,90.0,800.0)]);
        let report=paired(manifests,rows,"base");
        let diagnostic=report.pairwise_effects.iter().find(|e|e.metric_id=="abstention_count").unwrap();
        assert_eq!(diagnostic.mean_directional_gain,None);
        assert_eq!(diagnostic.worst_cluster_directional_gain,None);
    }

    #[test]
    fn task_outcomes_require_bound_receipts_and_hide_evaluator_labels() {
        let manifests=vec![manifest("base","base"),manifest("candidate","candidate")];
        let mut rows=outcomes(&[("base",0.6,100.0,1000.0),("candidate",0.8,90.0,800.0)]);
        rows[0].outcome_receipt_sha256="BAD".into();
        assert_eq!(
            validate_and_group_outcomes(&manifests,&rows),
            Err(FrontierRejectV1::InvalidOutcomeReceipt)
        );
        let mut rows=outcomes(&[("base",0.6,100.0,1000.0),("candidate",0.8,90.0,800.0)]);
        rows[0].candidate_saw_evaluator_labels=true;
        assert_eq!(
            validate_and_group_outcomes(&manifests,&rows),
            Err(FrontierRejectV1::CandidateSawEvaluatorLabels)
        );
    }

    #[test]
    fn report_binds_sorted_outcome_set_and_custody_class() {
        let manifests=vec![manifest("base","base"),manifest("candidate","candidate")];
        let mut rows=outcomes(&[("base",0.6,100.0,1000.0),("candidate",0.8,90.0,800.0)]);
        let first=paired(manifests.clone(),rows.clone(),"base");
        rows.reverse();
        let second=paired(manifests,rows,"base");
        assert_eq!(first.outcome_set_sha256,second.outcome_set_sha256);
        assert_eq!(first.canonical_sha256,second.canonical_sha256);
        assert_eq!(first.outcome_count,8);
        assert_eq!(first.evaluator_custody,OutcomeCustodyV1::PublicDiagnosticOnly);
        assert!(!first.candidate_saw_evaluator_labels);
    }

    #[test]
    fn outcome_payload_tampering_is_rejected_by_receipt_digest() {
        let manifests=vec![manifest("base","base"),manifest("candidate","candidate")];
        let mut rows=outcomes(&[("base",0.6,100.0,1000.0),("candidate",0.8,90.0,800.0)]);
        rows[0].metric_values.insert("governance_score".into(),0.99);
        assert_eq!(
            validate_and_group_outcomes(&manifests,&rows),
            Err(FrontierRejectV1::OutcomePayloadMismatch)
        );
    }

    #[test]
    fn evaluator_custody_remains_explicitly_diagnostic_only() {
        let manifests=vec![manifest("base","base"),manifest("candidate","candidate")];
        let rows=outcomes(&[("base",0.6,100.0,1000.0),("candidate",0.8,90.0,800.0)]);
        let report=paired(manifests,rows,"base");
        assert_eq!(report.evaluator_custody,OutcomeCustodyV1::PublicDiagnosticOnly);
        assert!(!report.candidate_saw_evaluator_labels);
    }

}
