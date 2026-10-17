// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Design-weighted scoring for the NANO-013 policy-blind probability sample.
//!
//! The input must first pass the structural audit. This scorer independently checks the
//! sample/result identity and digest bindings it relies on. It does not attest that the
//! reference provider is truthful, the receipt was externally timestamped, costs were billed,
//! or the benchmark is scientifically representative.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

const BOOTSTRAP_REPLICATES: usize = 2_000;
const MIN_GROUPS_FOR_INTERVAL: usize = 10;
const REQUIRED_VALID_BOOTSTRAP_FRACTION: f64 = 0.90;
const MAX_RESULTS_BYTES: u64 = 256 * 1024 * 1024;
const MAX_RESULTS_ROWS: usize = 1_000_000;

#[derive(Debug, Deserialize)]
struct SamplingManifest {
    schema_version: String,
    run_id: String,
    policy_revision_sha256: String,
    sampled_windows: Vec<SampledWindow>,
}

#[derive(Debug, Deserialize)]
struct SampledWindow {
    window_id: String,
    parent_group_id: String,
    trajectory_id: String,
    stratum_id: String,
    inclusion_probability: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DecisionRecord {
    schema_version: String,
    run_id: String,
    trajectory_id: String,
    parent_group_id: String,
    window_id: String,
    sequence: u64,
    decision_step: u64,
    max_observed_step: u64,
    action: String,
    future_data_read: bool,
    heldout_label_read: bool,
    observed_state_sha256: String,
    available_inputs_manifest_sha256: String,
    policy_revision_sha256: String,
    trigger_reason: String,
    risk_score: Value,
    reference_request_id: Value,
    policy_cost: f64,
    policy_cost_unit: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AuditResult {
    schema_version: String,
    run_id: String,
    sampling_manifest_sha256: String,
    window_id: String,
    parent_group_id: String,
    trajectory_id: String,
    stratum_id: String,
    inclusion_probability: f64,
    reference_status: String,
    severe_event: Option<bool>,
    physical_integrity_pass: Option<bool>,
    policy_triggered_reference: bool,
    labels_revealed_to_policy: bool,
    audit_reference_cost: f64,
    audit_cost_unit: String,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
struct SufficientStats {
    sampled_windows: usize,
    evaluated_reference_windows: usize,
    reference_failed_windows: usize,
    reference_unavailable_windows: usize,
    known_integrity_windows: usize,
    integrity_breach_windows: usize,
    weight_total: f64,
    weight_query: f64,
    weight_intervention: f64,
    weight_evaluated: f64,
    weight_severe: f64,
    weight_severe_query: f64,
    weight_severe_intervention: f64,
    weight_nonsevere: f64,
    weight_nonsevere_query: f64,
    weight_nonsevere_intervention: f64,
    weight_integrity_known: f64,
    weight_integrity_failure: f64,
    weighted_reference_cost: f64,
}

impl SufficientStats {
    fn add(&mut self, other: Self) {
        self.sampled_windows += other.sampled_windows;
        self.evaluated_reference_windows += other.evaluated_reference_windows;
        self.reference_failed_windows += other.reference_failed_windows;
        self.reference_unavailable_windows += other.reference_unavailable_windows;
        self.known_integrity_windows += other.known_integrity_windows;
        self.integrity_breach_windows += other.integrity_breach_windows;
        self.weight_total += other.weight_total;
        self.weight_query += other.weight_query;
        self.weight_intervention += other.weight_intervention;
        self.weight_evaluated += other.weight_evaluated;
        self.weight_severe += other.weight_severe;
        self.weight_severe_query += other.weight_severe_query;
        self.weight_severe_intervention += other.weight_severe_intervention;
        self.weight_nonsevere += other.weight_nonsevere;
        self.weight_nonsevere_query += other.weight_nonsevere_query;
        self.weight_nonsevere_intervention += other.weight_nonsevere_intervention;
        self.weight_integrity_known += other.weight_integrity_known;
        self.weight_integrity_failure += other.weight_integrity_failure;
        self.weighted_reference_cost += other.weighted_reference_cost;
    }

    fn from_row(row: &AuditResult, intervention: bool) -> Self {
        let weight = 1.0 / row.inclusion_probability;
        let evaluated = row.reference_status == "evaluated";
        let severe = row.severe_event == Some(true);
        let nonsevere = row.severe_event == Some(false);
        let integrity_known = row.physical_integrity_pass.is_some();
        let integrity_failure = row.physical_integrity_pass == Some(false);

        Self {
            sampled_windows: 1,
            evaluated_reference_windows: if evaluated { 1 } else { 0 },
            reference_failed_windows: if row.reference_status == "reference_failed" { 1 } else { 0 },
            reference_unavailable_windows: if row.reference_status == "reference_unavailable" { 1 } else { 0 },
            known_integrity_windows: if integrity_known { 1 } else { 0 },
            integrity_breach_windows: if integrity_failure { 1 } else { 0 },
            weight_total: weight,
            weight_query: if row.policy_triggered_reference { weight } else { 0.0 },
            weight_intervention: if intervention { weight } else { 0.0 },
            weight_evaluated: if evaluated { weight } else { 0.0 },
            weight_severe: if evaluated && severe { weight } else { 0.0 },
            weight_severe_query: if evaluated && severe && row.policy_triggered_reference {
                weight
            } else {
                0.0
            },
            weight_severe_intervention: if evaluated && severe && intervention {
                weight
            } else {
                0.0
            },
            weight_nonsevere: if evaluated && nonsevere { weight } else { 0.0 },
            weight_nonsevere_query: if evaluated && nonsevere && row.policy_triggered_reference {
                weight
            } else {
                0.0
            },
            weight_nonsevere_intervention: if evaluated && nonsevere && intervention {
                weight
            } else {
                0.0
            },
            weight_integrity_known: if integrity_known { weight } else { 0.0 },
            weight_integrity_failure: if integrity_failure { weight } else { 0.0 },
            weighted_reference_cost: weight * row.audit_reference_cost,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Metric {
    QueryFraction,
    InterventionFraction,
    ReferenceOutcomeCoverage,
    SevereEventInterventionRecall,
    SevereEventQueryRecall,
    FalseInterventionRate,
    FalseQueryRate,
    InterventionFalseAlarmBurden,
    QueryFalseAlarmBurden,
    PhysicalIntegrityFailureRate,
}

impl Metric {
    fn value(self, stats: &SufficientStats) -> Option<f64> {
        let (numerator, denominator) = match self {
            Self::QueryFraction => (stats.weight_query, stats.weight_total),
            Self::InterventionFraction => (stats.weight_intervention, stats.weight_total),
            Self::ReferenceOutcomeCoverage => (stats.weight_evaluated, stats.weight_total),
            Self::SevereEventInterventionRecall => (
                stats.weight_severe_intervention,
                stats.weight_severe,
            ),
            Self::SevereEventQueryRecall => (stats.weight_severe_query, stats.weight_severe),
            Self::FalseInterventionRate => (
                stats.weight_nonsevere_intervention,
                stats.weight_nonsevere,
            ),
            Self::FalseQueryRate => (stats.weight_nonsevere_query, stats.weight_nonsevere),
            Self::InterventionFalseAlarmBurden => (
                stats.weight_nonsevere_intervention,
                stats.weight_total,
            ),
            Self::QueryFalseAlarmBurden => (stats.weight_nonsevere_query, stats.weight_total),
            Self::PhysicalIntegrityFailureRate => (
                stats.weight_integrity_failure,
                stats.weight_integrity_known,
            ),
        };
        (denominator > 0.0).then_some(numerator / denominator)
    }
}

#[derive(Debug, Serialize)]
struct MetricEstimate {
    estimate: Option<f64>,
    cluster_bootstrap_95_ci: Option<[f64; 2]>,
    valid_bootstrap_replicates: usize,
    interpretation: String,
}

#[derive(Debug, Serialize)]
struct ParentGroupReport {
    parent_group_id: String,
    sampled_windows: usize,
    evaluated_reference_windows: usize,
    reference_failed_windows: usize,
    reference_unavailable_windows: usize,
    known_integrity_windows: usize,
    observed_integrity_breach_windows: usize,
    policy_reference_query_fraction: Option<f64>,
    policy_intervention_fraction: Option<f64>,
    severe_event_intervention_recall: Option<f64>,
    severe_event_reference_query_recall: Option<f64>,
    false_intervention_rate: Option<f64>,
    false_reference_query_rate: Option<f64>,
    physical_integrity_failure_rate: Option<f64>,
}

impl ParentGroupReport {
    fn from_stats(parent_group_id: &str, stats: &SufficientStats) -> Self {
        Self {
            parent_group_id: parent_group_id.to_owned(),
            sampled_windows: stats.sampled_windows,
            evaluated_reference_windows: stats.evaluated_reference_windows,
            reference_failed_windows: stats.reference_failed_windows,
            reference_unavailable_windows: stats.reference_unavailable_windows,
            known_integrity_windows: stats.known_integrity_windows,
            observed_integrity_breach_windows: stats.integrity_breach_windows,
            policy_reference_query_fraction: Metric::QueryFraction.value(stats),
            policy_intervention_fraction: Metric::InterventionFraction.value(stats),
            severe_event_intervention_recall: Metric::SevereEventInterventionRecall.value(stats),
            severe_event_reference_query_recall: Metric::SevereEventQueryRecall.value(stats),
            false_intervention_rate: Metric::FalseInterventionRate.value(stats),
            false_reference_query_rate: Metric::FalseQueryRate.value(stats),
            physical_integrity_failure_rate: Metric::PhysicalIntegrityFailureRate.value(stats),
        }
    }
}

#[derive(Debug, Serialize)]
struct ScoreReport {
    schema_version: &'static str,
    run_id: String,
    sampling_manifest_sha256: String,
    sampled_windows: usize,
    evaluated_reference_windows: usize,
    reference_failed_windows: usize,
    reference_unavailable_windows: usize,
    independent_parent_groups: usize,
    independent_trajectories: usize,
    parent_groups_with_observed_integrity_breach: usize,
    parent_groups_with_known_integrity_outcomes: usize,
    parent_groups_without_evaluated_reference_outcome: usize,
    worst_parent_group_physical_integrity_failure_rate: Option<f64>,
    worst_parent_group_false_intervention_rate: Option<f64>,
    lowest_parent_group_severe_event_intervention_recall: Option<f64>,
    parent_groups: Vec<ParentGroupReport>,
    policy_decision_count: usize,
    policy_reference_query_count: usize,
    policy_intervention_count: usize,
    weighted_estimated_population_windows: f64,
    observed_policy_cost: f64,
    policy_cost_unit: String,
    observed_audit_reference_cost: f64,
    horvitz_thompson_if_all_eligible_windows_reference_cost: f64,
    audit_cost_unit: String,
    reference_outcome_coverage: MetricEstimate,
    policy_reference_query_fraction: MetricEstimate,
    policy_intervention_fraction: MetricEstimate,
    severe_event_intervention_recall_on_evaluated_references: MetricEstimate,
    severe_event_reference_query_recall_on_evaluated_references: MetricEstimate,
    false_intervention_rate_among_evaluated_nonsevere_windows: MetricEstimate,
    false_reference_query_rate_among_evaluated_nonsevere_windows: MetricEstimate,
    known_label_intervention_false_alarm_burden_per_estimated_population_window: MetricEstimate,
    known_label_reference_query_false_alarm_burden_per_estimated_population_window: MetricEstimate,
    physical_integrity_failure_rate_on_evaluated_references: MetricEstimate,
    interval_method: String,
    warnings: Vec<String>,
}

#[derive(Debug, Default)]
struct DeterministicRng(u64);

impl DeterministicRng {
    fn next(&mut self) -> u64 {
        if self.0 == 0 {
            self.0 = 0x9e37_79b9_7f4a_7c15;
        }
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn index(&mut self, upper: usize) -> usize {
        (self.next() % upper as u64) as usize
    }
}

fn read_regular_file(path: &Path) -> Result<Vec<u8>, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Err(format!("{} is not a regular, non-symlink file", path.display()));
    }
    if metadata.len() > MAX_RESULTS_BYTES {
        return Err(format!("{} exceeds the {}-byte limit", path.display(), MAX_RESULTS_BYTES));
    }
    fs::read(path).map_err(|error| format!("{}: {error}", path.display()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn close_probability(left: f64, right: f64) -> bool {
    left.is_finite() && right.is_finite() && (left - right).abs() <= 1.0e-12
}

/// Returns true when a parent cluster is already assigned to a different stratum.
/// It never overwrites the original assignment after a conflict.
fn parent_cluster_crosses_strata(
    parent_strata: &mut BTreeMap<String, String>,
    parent_group_id: &str,
    stratum_id: &str,
) -> bool {
    match parent_strata.get(parent_group_id) {
        Some(existing) => existing != stratum_id,
        None => {
            parent_strata.insert(parent_group_id.to_owned(), stratum_id.to_owned());
            false
        }
    }
}

fn parse_decisions(bytes: &[u8]) -> Result<Vec<DecisionRecord>, String> {
    let text = std::str::from_utf8(bytes).map_err(|error| format!("decision ledger is not UTF-8: {error}"))?;
    let mut rows = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            return Err(format!("blank decision JSONL record at line {}", index + 1));
        }
        if rows.len() >= MAX_RESULTS_ROWS {
            return Err(format!("decision rows exceed the {MAX_RESULTS_ROWS}-row limit"));
        }
        let row: DecisionRecord = serde_json::from_str(line)
            .map_err(|error| format!("invalid decision record at line {}: {error}", index + 1))?;
        rows.push(row);
    }
    if rows.is_empty() {
        return Err("policy decision ledger contains no records".to_owned());
    }
    Ok(rows)
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn parse_results(bytes: &[u8]) -> Result<Vec<AuditResult>, String> {
    let text = std::str::from_utf8(bytes).map_err(|error| format!("results are not UTF-8: {error}"))?;
    let mut rows = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            return Err(format!("blank JSONL record at line {}", index + 1));
        }
        if rows.len() >= MAX_RESULTS_ROWS {
            return Err(format!("result rows exceed the {MAX_RESULTS_ROWS}-row limit"));
        }
        let row: AuditResult = serde_json::from_str(line)
            .map_err(|error| format!("invalid audit result at line {}: {error}", index + 1))?;
        rows.push(row);
    }
    if rows.is_empty() {
        return Err("blinded audit results contain no records".to_owned());
    }
    Ok(rows)
}

fn estimate(
    metric: Metric,
    groups: &BTreeMap<String, SufficientStats>,
    stratified_groups: Option<&BTreeMap<String, BTreeMap<String, SufficientStats>>>,
    total: &SufficientStats,
    seed: u64,
    interpretation: &str,
) -> MetricEstimate {
    let point = metric.value(total);
    let mut report = MetricEstimate {
        estimate: point,
        cluster_bootstrap_95_ci: None,
        valid_bootstrap_replicates: 0,
        interpretation: interpretation.to_owned(),
    };
    if groups.len() < MIN_GROUPS_FOR_INTERVAL || stratified_groups.is_none() {
        return report;
    }

    // Resample independent parent clusters within each frozen stratum. Pooling clusters
    // across strata would alter the declared allocation and can distort uncertainty.
    let strata = stratified_groups
        .expect("checked above")
        .values()
        .map(|stratum| stratum.values().copied().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    if strata.iter().any(Vec::is_empty) {
        return report;
    }
    let mut rng = DeterministicRng(seed);
    let mut values = Vec::with_capacity(BOOTSTRAP_REPLICATES);
    for _ in 0..BOOTSTRAP_REPLICATES {
        let mut replicate = SufficientStats::default();
        for stratum_groups in &strata {
            for _ in 0..stratum_groups.len() {
                replicate.add(stratum_groups[rng.index(stratum_groups.len())]);
            }
        }
        if let Some(value) = metric.value(&replicate) {
            if value.is_finite() {
                values.push(value);
            }
        }
    }
    report.valid_bootstrap_replicates = values.len();
    if values.len() as f64 >= BOOTSTRAP_REPLICATES as f64 * REQUIRED_VALID_BOOTSTRAP_FRACTION {
        values.sort_by(f64::total_cmp);
        let low = ((values.len() - 1) as f64 * 0.025).floor() as usize;
        let high = ((values.len() - 1) as f64 * 0.975).ceil() as usize;
        report.cluster_bootstrap_95_ci = Some([values[low], values[high]]);
    }
    report
}

/// Score the independently sampled NANO-013 outcome records in a run directory.
pub fn score_run(run_dir: &Path) -> Result<String, String> {
    let root = run_dir.join("rescue-controller");
    let manifest_path = root.join("audit-sampling-manifest.json");
    let results_path = root.join("blinded-audit-results.jsonl");
    let manifest_bytes = read_regular_file(&manifest_path)?;
    let manifest_digest = sha256_hex(&manifest_bytes);
    let manifest: SamplingManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| format!("invalid sampling manifest JSON: {error}"))?;
    if manifest.schema_version != "nano-rescue-audit-sampling-v1" {
        return Err("unsupported sampling manifest schema_version".to_owned());
    }
    if manifest.run_id.trim().is_empty()
        || manifest.sampled_windows.is_empty()
        || !is_sha256(&manifest.policy_revision_sha256)
    {
        return Err("sampling manifest must declare a run_id, non-empty sample, and valid policy revision digest".to_owned());
    }

    let decisions_path = root.join("decision-ledger.jsonl");
    let decisions_bytes = read_regular_file(&decisions_path)?;
    let decisions = parse_decisions(&decisions_bytes)?;
    if decisions.is_empty() {
        return Err("policy decision ledger is empty".to_owned());
    }
    let mut decision_by_id = BTreeMap::new();
    let mut policy_cost_unit: Option<String> = None;
    let mut observed_policy_cost = 0.0;
    let mut policy_reference_query_count = 0usize;
    let mut policy_intervention_count = 0usize;
    let mut sequences_by_trajectory = BTreeMap::<String, Vec<u64>>::new();
    for decision in &decisions {
        if decision.schema_version != "nano-rescue-decision-v1"
            || decision.run_id != manifest.run_id
            || decision.policy_revision_sha256 != manifest.policy_revision_sha256
        {
            return Err(format!("decision {} has mismatched schema, run, or policy revision", decision.window_id));
        }
        if decision.window_id.trim().is_empty()
            || decision.parent_group_id.trim().is_empty()
            || decision.trajectory_id.trim().is_empty()
            || decision.trigger_reason.trim().is_empty()
        {
            return Err("policy decision is missing required identity fields".to_owned());
        }
        if decision.max_observed_step > decision.decision_step {
            return Err(format!("decision {} observes a future step", decision.window_id));
        }
        if !matches!(
            decision.trigger_reason.as_str(),
            "nominal" | "uncertainty" | "shift_alarm" | "budget_guard" | "reference_result" | "integrity_guard" | "operator_override"
        ) {
            return Err(format!("decision {} has an unsupported trigger reason", decision.window_id));
        }
        if !decision.risk_score.is_null()
            && decision.risk_score.as_f64().map_or(true, |value| {
                !value.is_finite() || !(0.0..=1.0).contains(&value)
            })
        {
            return Err(format!("decision {} has an invalid risk score", decision.window_id));
        }
        if decision.future_data_read || decision.heldout_label_read {
            return Err(format!("decision {} declares evaluation leakage", decision.window_id));
        }
        if !matches!(
            decision.action.as_str(),
            "continue" | "conservative_continue" | "abstain_stop" | "request_reference"
        ) {
            return Err(format!("decision {} has an unsupported action", decision.window_id));
        }
        if !is_sha256(&decision.observed_state_sha256)
            || !is_sha256(&decision.available_inputs_manifest_sha256)
        {
            return Err(format!("decision {} contains invalid state/input digest", decision.window_id));
        }
        if !decision.policy_cost.is_finite()
            || decision.policy_cost < 0.0
            || decision.policy_cost_unit.trim().is_empty()
        {
            return Err(format!("decision {} has invalid policy cost/unit", decision.window_id));
        }
        match &policy_cost_unit {
            Some(unit) if unit != &decision.policy_cost_unit => {
                return Err("policy costs use mixed units in the decision ledger".to_owned());
            }
            None => policy_cost_unit = Some(decision.policy_cost_unit.clone()),
            _ => {}
        }
        if decision.action == "request_reference" {
            if !decision.reference_request_id.as_str().is_some_and(|id| !id.trim().is_empty()) {
                return Err(format!("reference request {} has no request ID", decision.window_id));
            }
            policy_reference_query_count += 1;
        } else if !decision.reference_request_id.is_null() {
            return Err(format!("non-query action {} contains a reference request ID", decision.window_id));
        }
        if decision.action != "continue" {
            policy_intervention_count += 1;
        }
        observed_policy_cost += decision.policy_cost;
        if !observed_policy_cost.is_finite() {
            return Err("policy cost sum overflowed or became non-finite".to_owned());
        }
        sequences_by_trajectory
            .entry(decision.trajectory_id.clone())
            .or_default()
            .push(decision.sequence);
        if decision_by_id.insert(decision.window_id.as_str(), decision).is_some() {
            return Err(format!("duplicate window in policy decision ledger: {}", decision.window_id));
        }
    }
    for (trajectory, sequences) in &mut sequences_by_trajectory {
        sequences.sort_unstable();
        for (index, sequence) in sequences.iter().enumerate() {
            if *sequence != index as u64 + 1 {
                return Err(format!(
                    "decision sequence for trajectory {trajectory} is not contiguous from 1"
                ));
            }
        }
    }

    let mut sample_by_id = BTreeMap::new();
    for sample in &manifest.sampled_windows {
        if sample.window_id.trim().is_empty()
            || sample.parent_group_id.trim().is_empty()
            || sample.trajectory_id.trim().is_empty()
            || sample.stratum_id.trim().is_empty()
            || !sample.inclusion_probability.is_finite()
            || sample.inclusion_probability <= 0.0
            || sample.inclusion_probability > 1.0
        {
            return Err(format!("invalid sampled-window identity/probability: {}", sample.window_id));
        }
        if sample_by_id.insert(sample.window_id.as_str(), sample).is_some() {
            return Err(format!("duplicate sampled window in manifest: {}", sample.window_id));
        }
    }

    let results_bytes = read_regular_file(&results_path)?;
    let rows = parse_results(&results_bytes)?;
    if rows.len() != manifest.sampled_windows.len() {
        return Err(format!(
            "sample/result cardinality mismatch: {} manifest windows, {} result records",
            manifest.sampled_windows.len(),
            rows.len()
        ));
    }
    let result_digest_binding = manifest_digest.clone();
    let mut seen = BTreeSet::new();
    let mut groups = BTreeMap::<String, SufficientStats>::new();
    let mut groups_by_stratum =
        BTreeMap::<String, BTreeMap<String, SufficientStats>>::new();
    let mut parent_stratum = BTreeMap::<String, String>::new();
    let mut parent_spans_strata = false;
    let mut trajectories = BTreeSet::new();
    let mut evaluated = 0usize;
    let mut failed = 0usize;
    let mut unavailable = 0usize;
    let mut observed_cost = 0.0;
    let mut cost_unit: Option<String> = None;
    let mut total = SufficientStats::default();

    for row in &rows {
        if row.schema_version != "nano-rescue-audit-result-v1"
            || row.run_id != manifest.run_id
            || row.sampling_manifest_sha256 != result_digest_binding
        {
            return Err(format!("result {} has mismatched schema, run, or manifest digest", row.window_id));
        }
        if row.labels_revealed_to_policy {
            return Err(format!("result {} declares label leakage to policy", row.window_id));
        }
        if !seen.insert(row.window_id.as_str()) {
            return Err(format!("duplicate result window: {}", row.window_id));
        }
        let sample = sample_by_id
            .get(row.window_id.as_str())
            .ok_or_else(|| format!("result window {} is absent from sampling manifest", row.window_id))?;
        if row.parent_group_id != sample.parent_group_id
            || row.trajectory_id != sample.trajectory_id
            || row.stratum_id != sample.stratum_id
            || !close_probability(row.inclusion_probability, sample.inclusion_probability)
        {
            return Err(format!("result {} does not match sampled identity/probability", row.window_id));
        }
        let decision = decision_by_id
            .get(row.window_id.as_str())
            .ok_or_else(|| format!("sampled window {} has no logged policy decision", row.window_id))?;
        if decision.parent_group_id != row.parent_group_id
            || decision.trajectory_id != row.trajectory_id
            || (decision.action == "request_reference") != row.policy_triggered_reference
        {
            return Err(format!("decision/action does not reconcile with audited outcome for {}", row.window_id));
        }
        let intervention = decision.action != "continue";
        if !row.inclusion_probability.is_finite()
            || row.inclusion_probability <= 0.0
            || row.inclusion_probability > 1.0
        {
            return Err(format!("invalid inclusion probability for {}", row.window_id));
        }
        if !row.audit_reference_cost.is_finite()
            || row.audit_reference_cost < 0.0
            || row.audit_cost_unit.trim().is_empty()
        {
            return Err(format!("invalid audit reference cost/unit for {}", row.window_id));
        }
        match &cost_unit {
            Some(unit) if unit != &row.audit_cost_unit => {
                return Err("audit reference costs use mixed units".to_owned());
            }
            None => cost_unit = Some(row.audit_cost_unit.clone()),
            _ => {}
        }
        match row.reference_status.as_str() {
            "evaluated" if row.severe_event.is_some() && row.physical_integrity_pass.is_some() => {
                evaluated += 1;
            }
            "reference_failed" if row.severe_event.is_none() && row.physical_integrity_pass.is_none() => {
                failed += 1;
            }
            "reference_unavailable" if row.severe_event.is_none() && row.physical_integrity_pass.is_none() => {
                unavailable += 1;
            }
            _ => {
                return Err(format!("reference status/outcome labels disagree for {}", row.window_id));
            }
        }
        trajectories.insert(row.trajectory_id.clone());
        observed_cost += row.audit_reference_cost;
        if !observed_cost.is_finite() {
            return Err("observed audit-reference cost sum overflowed or became non-finite".to_owned());
        }
        let row_stats = SufficientStats::from_row(row, intervention);
        total.add(row_stats);
        groups.entry(row.parent_group_id.clone()).or_default().add(row_stats);
        parent_spans_strata |= parent_cluster_crosses_strata(
            &mut parent_stratum,
            &row.parent_group_id,
            &row.stratum_id,
        );
        groups_by_stratum
            .entry(row.stratum_id.clone())
            .or_default()
            .entry(row.parent_group_id.clone())
            .or_default()
            .add(row_stats);
    }

    if seen.len() != sample_by_id.len() {
        return Err("one or more preregistered sample windows have no outcome record".to_owned());
    }
    let parent_groups = groups
        .iter()
        .map(|(parent, stats)| ParentGroupReport::from_stats(parent, stats))
        .collect::<Vec<_>>();
    let parent_groups_with_observed_integrity_breach = groups
        .values()
        .filter(|stats| stats.integrity_breach_windows > 0)
        .count();
    let parent_groups_with_known_integrity_outcomes = groups
        .values()
        .filter(|stats| stats.known_integrity_windows > 0)
        .count();
    let parent_groups_without_evaluated_reference_outcome = groups
        .values()
        .filter(|stats| stats.evaluated_reference_windows == 0)
        .count();
    let worst_parent_group_physical_integrity_failure_rate = parent_groups
        .iter()
        .filter_map(|group| group.physical_integrity_failure_rate)
        .max_by(f64::total_cmp);
    let worst_parent_group_false_intervention_rate = parent_groups
        .iter()
        .filter_map(|group| group.false_intervention_rate)
        .max_by(f64::total_cmp);
    let lowest_parent_group_severe_event_intervention_recall = parent_groups
        .iter()
        .filter_map(|group| group.severe_event_intervention_recall)
        .min_by(f64::total_cmp);

    let bootstrap_groups = if parent_spans_strata {
        None
    } else {
        Some(&groups_by_stratum)
    };

    let mut warnings = Vec::new();
    if parent_spans_strata {
        warnings.push(
            "One or more parent systems span multiple sampling strata; the simple stratified cluster bootstrap is not appropriate for this crossed design, so confidence intervals are withheld. Use a preregistered design-based or multiway-cluster variance method.".to_owned()
        );
    }
    if parent_groups_with_observed_integrity_breach > 0 {
        warnings.push(format!(
            "{} parent group(s) contain at least one observed physical-integrity breach; pooled averages must not mask these failures.",
            parent_groups_with_observed_integrity_breach
        ));
    }
    if parent_groups_without_evaluated_reference_outcome > 0 {
        warnings.push(format!(
            "{} parent group(s) have no successfully evaluated reference outcome; these groups cannot be classified as physically safe.",
            parent_groups_without_evaluated_reference_outcome
        ));
    }
    if groups.len() < MIN_GROUPS_FOR_INTERVAL {
        warnings.push(format!(
            "Only {} independent parent groups are represented; cluster-bootstrap intervals are withheld until at least {} groups are available.",
            groups.len(),
            MIN_GROUPS_FOR_INTERVAL
        ));
    }
    if failed + unavailable > 0 {
        warnings.push(
            "Some sampled windows lack reference labels. Recall and physical-integrity estimates are conditional on successful reference outcomes; known-label false-alarm burdens are lower bounds, not complete-population estimates. Do not silently treat missing outcomes as safe.".to_owned()
        );
    }
    if groups.len() < 20 {
        warnings.push(
            "Fewer than 20 independent parent groups: even when an interval is emitted, tail-risk and worst-group estimates may be unstable.".to_owned()
        );
    }

    let report = ScoreReport {
        schema_version: "nano-rescue-score-v1",
        run_id: manifest.run_id,
        sampling_manifest_sha256: manifest_digest,
        sampled_windows: rows.len(),
        evaluated_reference_windows: evaluated,
        reference_failed_windows: failed,
        reference_unavailable_windows: unavailable,
        independent_parent_groups: groups.len(),
        independent_trajectories: trajectories.len(),
        parent_groups_with_observed_integrity_breach,
        parent_groups_with_known_integrity_outcomes,
        parent_groups_without_evaluated_reference_outcome,
        worst_parent_group_physical_integrity_failure_rate,
        worst_parent_group_false_intervention_rate,
        lowest_parent_group_severe_event_intervention_recall,
        parent_groups,
        policy_decision_count: decisions.len(),
        policy_reference_query_count,
        policy_intervention_count,
        weighted_estimated_population_windows: total.weight_total,
        observed_policy_cost,
        policy_cost_unit: policy_cost_unit.unwrap_or_else(|| "unknown".to_owned()),
        observed_audit_reference_cost: observed_cost,
        horvitz_thompson_if_all_eligible_windows_reference_cost: total.weighted_reference_cost,
        audit_cost_unit: cost_unit.unwrap_or_else(|| "unknown".to_owned()),
        reference_outcome_coverage: estimate(
            Metric::ReferenceOutcomeCoverage,
            &groups,
            bootstrap_groups,
            &total,
            0x8ad1_ee31_15ca_77a1,
            "Horvitz-Thompson weighted evaluated-outcome share of the eligible-window population.",
        ),
        policy_reference_query_fraction: estimate(
            Metric::QueryFraction,
            &groups,
            bootstrap_groups,
            &total,
            0xa904_2d31_99c7_10ef,
            "Design-weighted fraction of eligible windows for which the policy requested a reference; independent of whether the evaluator reference later succeeded.",
        ),
        policy_intervention_fraction: estimate(
            Metric::InterventionFraction,
            &groups,
            bootstrap_groups,
            &total,
            0xe3b1_7a62_5c49_08df,
            "Design-weighted fraction of eligible windows where policy action was not plain continue; includes conservative continuation, stop/abstain, and reference request.",
        ),
        severe_event_intervention_recall_on_evaluated_references: estimate(
            Metric::SevereEventInterventionRecall,
            &groups,
            bootstrap_groups,
            &total,
            0x17d3_5a2c_91b4_06ef,
            "Design-weighted recall of any intervention (conservative_continue, abstain_stop, or request_reference) among successfully evaluated severe outcomes; conditional if any reference labels are missing.",
        ),
        severe_event_reference_query_recall_on_evaluated_references: estimate(
            Metric::SevereEventQueryRecall,
            &groups,
            bootstrap_groups,
            &total,
            0x7ce4_1b90_2d65_a83f,
            "Design-weighted recall of explicit reference queries among successfully evaluated severe outcomes; conditional if any reference labels are missing.",
        ),
        false_intervention_rate_among_evaluated_nonsevere_windows: estimate(
            Metric::FalseInterventionRate,
            &groups,
            bootstrap_groups,
            &total,
            0x36c8_21af_73d9_450b,
            "Design-weighted fraction of successfully evaluated non-severe windows receiving any intervention; conditional on observed reference labels.",
        ),
        false_reference_query_rate_among_evaluated_nonsevere_windows: estimate(
            Metric::FalseQueryRate,
            &groups,
            bootstrap_groups,
            &total,
            0x11a8_2d74_c59e_630f,
            "Design-weighted fraction of successfully evaluated non-severe windows receiving an explicit reference query; conditional on observed reference labels.",
        ),
        known_label_intervention_false_alarm_burden_per_estimated_population_window: estimate(
            Metric::InterventionFalseAlarmBurden,
            &groups,
            bootstrap_groups,
            &total,
            0xc0f1_25b8_794e_631d,
            "Known non-severe windows receiving any intervention divided by estimated eligible-window population. With missing labels this is a lower bound on total intervention false-alarm burden.",
        ),
        known_label_reference_query_false_alarm_burden_per_estimated_population_window: estimate(
            Metric::QueryFalseAlarmBurden,
            &groups,
            bootstrap_groups,
            &total,
            0x62ef_a137_9d45_80bc,
            "Known non-severe windows receiving a reference query divided by estimated eligible-window population. With missing labels this is a lower bound on query false-alarm burden.",
        ),
        physical_integrity_failure_rate_on_evaluated_references: estimate(
            Metric::PhysicalIntegrityFailureRate,
            &groups,
            bootstrap_groups,
            &total,
            0x5e20_4f7a_d193_86bc,
            "Design-weighted physical-integrity failure rate among outcomes with a known physical-integrity label; conditional if any labels are missing.",
        ),
        interval_method: format!(
            "Approximate percentile bootstrap resampling parent_group_id clusters with replacement within each declared sampling stratum, preserving the number of sampled clusters per stratum; {BOOTSTRAP_REPLICATES} deterministic replicates. A 95% interval is emitted only with at least {MIN_GROUPS_FOR_INTERVAL} parent groups, no parent crossing strata, and at least {:.0}% valid replicates. This is not an exact design-based finite-population variance estimator.",
            REQUIRED_VALID_BOOTSTRAP_FRACTION * 100.0
        ),
        warnings,
    };
    serde_json::to_string_pretty(&report).map_err(|error| format!("serialize score report: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(
        window: &str,
        parent: &str,
        pi: f64,
        status: &str,
        severe: Option<bool>,
        trigger: bool,
    ) -> AuditResult {
        AuditResult {
            schema_version: "nano-rescue-audit-result-v1".to_owned(),
            run_id: "test".to_owned(),
            sampling_manifest_sha256: "a".repeat(64),
            window_id: window.to_owned(),
            parent_group_id: parent.to_owned(),
            trajectory_id: format!("trajectory-{parent}"),
            stratum_id: "family".to_owned(),
            inclusion_probability: pi,
            reference_status: status.to_owned(),
            severe_event: severe,
            physical_integrity_pass: severe.map(|is_severe| !is_severe),
            policy_triggered_reference: trigger,
            labels_revealed_to_policy: false,
            audit_reference_cost: 1.0,
            audit_cost_unit: "USD".to_owned(),
        }
    }

    #[test]
    fn design_weights_change_the_recall_estimate() {
        let severe_high_weight = row("a", "parent-a", 0.1, "evaluated", Some(true), true);
        let severe_low_weight = row("b", "parent-b", 0.5, "evaluated", Some(true), false);
        let mut stats = SufficientStats::default();
        stats.add(SufficientStats::from_row(&severe_high_weight, true));
        stats.add(SufficientStats::from_row(&severe_low_weight, false));
        let recall = Metric::SevereEventQueryRecall.value(&stats).unwrap();
        assert!((recall - (10.0 / 12.0)).abs() < 1.0e-12);
    }

    #[test]
    fn missing_reference_labels_are_not_counted_as_negative_events() {
        let evaluated = row("a", "parent-a", 0.5, "evaluated", Some(true), true);
        let missing = row("b", "parent-b", 0.5, "reference_failed", None, false);
        let mut stats = SufficientStats::default();
        stats.add(SufficientStats::from_row(&evaluated, true));
        stats.add(SufficientStats::from_row(&missing, false));
        assert!((Metric::ReferenceOutcomeCoverage.value(&stats).unwrap() - 0.5).abs() < 1.0e-12);
        assert_eq!(Metric::SevereEventQueryRecall.value(&stats), Some(1.0));
        assert_eq!(Metric::FalseQueryRate.value(&stats), None);
    }


    #[test]
    fn conservative_action_counts_as_intervention_without_counting_as_reference_query() {
        let severe = row("s", "parent-s", 0.25, "evaluated", Some(true), false);
        let stats = SufficientStats::from_row(&severe, true);
        assert_eq!(Metric::SevereEventInterventionRecall.value(&stats), Some(1.0));
        assert_eq!(Metric::SevereEventQueryRecall.value(&stats), Some(0.0));
        assert_eq!(Metric::InterventionFraction.value(&stats), Some(1.0));
        assert_eq!(Metric::QueryFraction.value(&stats), Some(0.0));
    }

    #[test]
    fn false_alarm_distinguishes_intervention_cost_from_reference_query_cost() {
        let nonsevere = row("n", "parent-n", 0.5, "evaluated", Some(false), false);
        let stats = SufficientStats::from_row(&nonsevere, true);
        assert_eq!(Metric::FalseInterventionRate.value(&stats), Some(1.0));
        assert_eq!(Metric::FalseQueryRate.value(&stats), Some(0.0));
        assert_eq!(Metric::InterventionFalseAlarmBurden.value(&stats), Some(1.0));
        assert_eq!(Metric::QueryFalseAlarmBurden.value(&stats), Some(0.0));
    }



    #[test]
    fn synthetic_bundle_scores_queries_interventions_and_worst_group_separately() {
        let report = score_run(
            &Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("fixtures")
                .join("nano-rescue-valid"),
        )
        .expect("structurally valid synthetic NANO-013 bundle should be scoreable");
        let value: serde_json::Value =
            serde_json::from_str(&report).expect("score report must be valid JSON");

        assert_eq!(value["policy_decision_count"], 4);
        assert_eq!(value["policy_reference_query_count"], 1);
        assert_eq!(value["policy_intervention_count"], 2);
        assert_eq!(value["policy_reference_query_fraction"]["estimate"], 0.5);
        assert_eq!(value["policy_intervention_fraction"]["estimate"], 0.5);
        assert_eq!(
            value["severe_event_intervention_recall_on_evaluated_references"]["estimate"],
            0.0
        );
        assert_eq!(
            value["parent_groups_with_observed_integrity_breach"],
            1
        );
        // A well-formed contract still scores the deliberate policy miss; integrity is not performance.
    }

    #[test]
    fn parent_group_report_preserves_localized_breach_hidden_by_pooling() {
        let severe = row("bad", "parent-bad", 0.5, "evaluated", Some(true), false);
        let safe = row("safe", "parent-safe", 0.5, "evaluated", Some(false), false);
        let severe_stats = SufficientStats::from_row(&severe, true);
        let safe_stats = SufficientStats::from_row(&safe, false);
        let mut pooled = severe_stats;
        pooled.add(safe_stats);

        assert_eq!(Metric::PhysicalIntegrityFailureRate.value(&pooled), Some(0.5));
        let worst_group = ParentGroupReport::from_stats("parent-bad", &severe_stats);
        let safe_group = ParentGroupReport::from_stats("parent-safe", &safe_stats);
        assert_eq!(worst_group.observed_integrity_breach_windows, 1);
        assert_eq!(worst_group.physical_integrity_failure_rate, Some(1.0));
        assert_eq!(safe_group.physical_integrity_failure_rate, Some(0.0));
        assert_eq!(worst_group.severe_event_intervention_recall, Some(1.0));
    }



    #[test]
    fn detects_parent_cluster_assigned_to_multiple_strata() {
        let mut parent_strata = BTreeMap::new();
        assert!(!parent_cluster_crosses_strata(
            &mut parent_strata,
            "parent-A",
            "surface"
        ));
        assert!(!parent_cluster_crosses_strata(
            &mut parent_strata,
            "parent-A",
            "surface"
        ));
        assert!(parent_cluster_crosses_strata(
            &mut parent_strata,
            "parent-A",
            "interface"
        ));
        assert_eq!(parent_strata.get("parent-A").map(String::as_str), Some("surface"));
    }

    #[test]
    fn cluster_bootstrap_preserves_sample_allocation_within_each_stratum() {
        let mut groups = BTreeMap::<String, SufficientStats>::new();
        let mut strata = BTreeMap::<String, BTreeMap<String, SufficientStats>>::new();
        let mut total = SufficientStats::default();

        for index in 0..10 {
            let mut queried = row(
                &format!("a-window-{index}"),
                &format!("a-parent-{index}"),
                1.0,
                "evaluated",
                Some(false),
                true,
            );
            queried.stratum_id = "always-query".to_owned();
            let queried_stats = SufficientStats::from_row(&queried, true);
            groups.insert(queried.parent_group_id.clone(), queried_stats);
            strata
                .entry(queried.stratum_id.clone())
                .or_default()
                .insert(queried.parent_group_id.clone(), queried_stats);
            total.add(queried_stats);

            let mut not_queried = row(
                &format!("b-window-{index}"),
                &format!("b-parent-{index}"),
                1.0,
                "evaluated",
                Some(false),
                false,
            );
            not_queried.stratum_id = "never-query".to_owned();
            let not_queried_stats = SufficientStats::from_row(&not_queried, false);
            groups.insert(not_queried.parent_group_id.clone(), not_queried_stats);
            strata
                .entry(not_queried.stratum_id.clone())
                .or_default()
                .insert(not_queried.parent_group_id.clone(), not_queried_stats);
            total.add(not_queried_stats);
        }

        let estimate = estimate(
            Metric::QueryFraction,
            &groups,
            Some(&strata),
            &total,
            0x5a11_f1ed,
            "stratified fixture",
        );
        assert_eq!(estimate.estimate, Some(0.5));
        assert_eq!(estimate.cluster_bootstrap_95_ci, Some([0.5, 0.5]));
        assert_eq!(estimate.valid_bootstrap_replicates, BOOTSTRAP_REPLICATES);
    }

    #[test]
    fn intervals_are_withheld_without_a_stratified_cluster_design() {
        let mut groups = BTreeMap::<String, SufficientStats>::new();
        let mut stats = BTreeMap::<String, SufficientStats>::new();
        for index in 0..MIN_GROUPS_FOR_INTERVAL {
            let row = row(
                &format!("window-{index}"),
                &format!("parent-{index}"),
                1.0,
                "evaluated",
                Some(false),
                false,
            );
            let value = SufficientStats::from_row(&row, false);
            groups.insert(row.parent_group_id, value);
            stats.insert(format!("parent-{index}"), value);
        }
        let total = groups.values().copied().fold(
            SufficientStats::default(),
            |mut sum, value| {
                sum.add(value);
                sum
            },
        );
        let result = estimate(
            Metric::QueryFraction,
            &groups,
            None,
            &total,
            0x5a11_f1ed,
            "crossed cluster/strata fixture",
        );
        assert_eq!(result.cluster_bootstrap_95_ci, None);
        assert_eq!(result.valid_bootstrap_replicates, 0);
    }

    #[test]
    fn intervals_are_withheld_for_too_few_independent_groups() {
        let row = row("a", "parent-a", 0.5, "evaluated", Some(true), true);
        let mut groups = BTreeMap::new();
        groups.insert("parent-a".to_owned(), SufficientStats::from_row(&row, true));
        let total = *groups.values().next().unwrap();
        let result = estimate(
            Metric::SevereEventQueryRecall,
            &groups,
            None,
            &total,
            1234,
            "test",
        );
        assert_eq!(result.cluster_bootstrap_95_ci, None);
        assert_eq!(result.valid_bootstrap_replicates, 0);
    }
}
