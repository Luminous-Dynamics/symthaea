//! Deterministic offline evaluator for frozen connectomics proofreading predictions.
//!
//! The scorer consumes predictions and adjudicated labels together only after inference
//! is frozen. Never give this file to the runtime ranker as an input. The positive class
//! is normalized by the cited mapping reference to mean "a genuine reconstruction error
//! requiring expert correction."

use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

const DATASET_ID: &str = "jeffbbrown2/ConnectomeBench2";
const DATASET_REVISION: &str = "v2";
const UPSTREAM_COMMIT: &str = "5679b39aa2641d8d0afeed88b441e696f87a66df";

#[derive(Debug, Clone, Copy, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum ReviewTimeBasis {
    ProspectiveMeasured,
    SyntheticProxy,
}

#[derive(Debug, Clone, Copy, ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum EvaluationPartition {
    Development,
    Confirmatory,
    Lockbox,
}

#[derive(Debug, Parser)]
#[command(
    name = "proofreading-score",
    version,
    about = "Score frozen connectomics proofreading predictions under review and coverage budgets"
)]
struct Args {
    /// JSONL file with one frozen prediction and independent adjudicated label per operation.
    #[arg(long)]
    input: PathBuf,

    /// Citation/path to the reviewed task/label normalization contract.
    #[arg(long)]
    mapping_reference: String,

    /// Exact dataset revision commit SHA used for this run.
    #[arg(long)]
    dataset_revision_commit_sha: String,

    /// Stable identifier for the frozen model/policy baseline being scored.
    #[arg(long)]
    policy_id: String,

    /// SHA-256 of the frozen model/policy configuration and preprocessing profile.
    #[arg(long)]
    prediction_profile_sha256: String,

    /// Data partition used for this scoring report.
    #[arg(long, value_enum)]
    evaluation_partition: EvaluationPartition,

    /// State whether review minutes were prospectively measured or are only synthetic proxies.
    #[arg(long, value_enum)]
    review_time_basis: ReviewTimeBasis,

    /// Comma-separated fractions of candidate items to review, e.g. 0.01,0.05,0.1,0.2.
    #[arg(long, default_value = "0.01,0.05,0.1,0.2,0.5")]
    item_budget_fractions: String,

    /// Comma-separated review-minute budgets, e.g. 10,30,60.
    #[arg(long, default_value = "10,30,60")]
    review_budgets_minutes: String,

    /// Comma-separated automatic-coverage targets, e.g. 0.5,0.7,0.8,0.9,0.95.
    #[arg(long, default_value = "0.5,0.7,0.8,0.9,0.95")]
    coverage_targets: String,

    /// Number of deterministic cluster-bootstrap replicates; at least 100 required.
    #[arg(long, default_value_t = 500)]
    bootstrap_replicates: usize,

    /// Require complete domain metadata for calibration/shift diagnostics.
    #[arg(long, default_value_t = false)]
    require_domain_shift_audit: bool,

    /// Stable identifier for the frozen novelty/OOD detector that produced ood_score.
    #[arg(long)]
    ood_detector_id: Option<String>,

    /// SHA-256 of the frozen novelty detector, representation, and preprocessing profile.
    #[arg(long)]
    ood_detector_profile_sha256: Option<String>,

    /// Frozen OOD novelty-score threshold; never selected from confirmatory labels.
    #[arg(long)]
    ood_score_threshold: Option<f64>,

    /// Write report to this JSON file (stdout if omitted).
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Prediction {
    /// Stable adjudication-unit ID, ideally source_archive_sample_hash / source operation ID.
    sample_id: String,
    /// Frozen grouping, preferably species:error_family (e.g. mouse:merge).
    group: String,
    /// Dependency cluster for uncertainty estimation (e.g. volume/cube/neuron), not the row ID.
    cluster_id: String,
    /// Independent adjudication: true iff a real reconstruction error requiring correction exists.
    label_is_error: bool,
    /// Frozen model probability that this candidate is a genuine error, calibrated if claimed.
    p_error: f64,
    /// Frozen policy ranking score; define upstream and keep fixed before the test labels are scored.
    review_priority_score: f64,
    /// Estimated or prospectively measured expert minutes to inspect this candidate.
    review_minutes: f64,
    /// Independent evaluator weight for scientific impact; must never be available to the runtime ranker.
    impact_weight: f64,
    /// Optional predeclared evaluation domain (separate from task/error-family group).
    #[serde(default)]
    domain_id: Option<String>,
    /// One shift axis per input run: species, region, source_volume, source_archive,
    /// acquisition_protocol, segmentation_lineage, modality_availability, or morphology_regime.
    #[serde(default)]
    shift_axis: Option<String>,
    /// True only when the frozen split protocol held this domain out of model/calibrator fitting.
    #[serde(default)]
    is_ood_domain: Option<bool>,
    /// Frozen model novelty score, separate from p_error and review_priority_score.
    #[serde(default)]
    ood_score: Option<f64>,
}

#[derive(Debug, Serialize)]
struct CalibrationBin {
    index: usize,
    lower: f64,
    upper: f64,
    upper_inclusive: bool,
    count: usize,
    mean_predicted_probability: Option<f64>,
    observed_error_rate: Option<f64>,
    absolute_calibration_gap: Option<f64>,
}

#[derive(Debug, Serialize)]
struct ConfidenceThresholdRisk {
    threshold: f64,
    selected_count: usize,
    false_accept_count: usize,
    false_accept_fraction: Option<f64>,
    precision: Option<f64>,
}

#[derive(Debug, Serialize)]
struct LowConfidenceErrorMass {
    threshold: f64,
    missed_error_count: usize,
    missed_error_impact_weight: f64,
}

#[derive(Debug, Serialize)]
struct CalibrationDiagnostics {
    fixed_bin_count: usize,
    adaptive_confidence_bin_count_requested: usize,
    adaptive_confidence_bins: Vec<ConfidenceCalibrationBin>,
    adaptive_confidence_expected_calibration_error: Option<f64>,
    adaptive_confidence_maximum_calibration_error: Option<f64>,
    reliability_curve_bin_count_requested: usize,
    reliability_curve_bins: Vec<ConfidenceCalibrationBin>,
    reliability_curve_expected_calibration_error: Option<f64>,
    brier_score: Option<f64>,
    log_loss: Option<f64>,
    expected_calibration_error: Option<f64>,
    maximum_calibration_error: Option<f64>,
    bins: Vec<CalibrationBin>,
    high_confidence_at_90: ConfidenceThresholdRisk,
    high_confidence_at_95: ConfidenceThresholdRisk,
    low_confidence_below_10: LowConfidenceErrorMass,
}

#[derive(Debug, Serialize)]
struct ConfidenceCalibrationBin {
    index: usize,
    count: usize,
    minimum_confidence: f64,
    maximum_confidence: f64,
    mean_confidence: f64,
    observed_accuracy: f64,
    absolute_calibration_gap: f64,
}

#[derive(Debug, Serialize)]
struct CalibrationClusterBootstrap {
    requested_replicates: usize,
    valid_replicates: usize,
    minimum_valid_replicates_for_interval: usize,
    expected_calibration_error_95: Option<Interval>,
    maximum_calibration_error_95: Option<Interval>,
    adaptive_confidence_ece_15_95: Option<Interval>,
    reliability_curve_ece_12_95: Option<Interval>,
    reliability_curve_observed_accuracy_95: Vec<ConfidenceBinClusterBootstrap>,
    brier_score_95: Option<Interval>,
}

#[derive(Debug, Serialize)]
struct ConfidenceBinClusterBootstrap {
    index: usize,
    full_sample_count: usize,
    requested_replicates: usize,
    valid_replicates: usize,
    minimum_valid_replicates_for_interval: usize,
    observed_accuracy_95: Option<Interval>,
    mean_confidence_95: Option<Interval>,
}

#[derive(Debug, Serialize)]
struct OodClusterBootstrap {
    requested_replicates: usize,
    valid_replicates: usize,
    minimum_valid_replicates_for_interval: usize,
    roc_auc_95: Option<Interval>,
    average_precision_95: Option<Interval>,
}

#[derive(Debug, Serialize)]
struct DomainEvaluation {
    domain_id: String,
    shift_axis: String,
    is_ood_domain: bool,
    n: usize,
    independent_cluster_count: usize,
    classification: ClassificationMetrics,
    calibration: CalibrationDiagnostics,
    calibration_cluster_bootstrap_95: CalibrationClusterBootstrap,
}

#[derive(Debug, Serialize)]
struct OodDetectionMetrics {
    status: &'static str,
    n: usize,
    independent_cluster_count: usize,
    in_domain_count: usize,
    held_out_domain_count: usize,
    roc_auc: Option<f64>,
    average_precision: Option<f64>,
    cluster_bootstrap_95: OodClusterBootstrap,
    frozen_threshold: Option<f64>,
    false_ood_count: Option<usize>,
    false_ood_rate_on_in_domain: Option<f64>,
    missed_ood_count: Option<usize>,
    held_out_domain_recall: Option<f64>,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    scorer: &'static str,
    scorer_version: &'static str,
    dataset_id: &'static str,
    dataset_revision: &'static str,
    dataset_revision_commit_sha: String,
    pinned_upstream_commit: &'static str,
    mapping_reference: String,
    policy_id: String,
    prediction_profile_sha256: String,
    evaluation_partition: EvaluationPartition,
    review_time_basis: ReviewTimeBasis,
    bootstrap_replicates: usize,
    input_sha256: String,
    n_candidates: usize,
    positive_errors: usize,
    negative_controls: usize,
    status: &'static str,
    overall: ClassificationMetrics,
    overall_cluster_bootstrap_95: ClusterBootstrap,
    per_group: BTreeMap<String, GroupEvaluation>,
    item_budget_results: Vec<ReviewYield>,
    review_time_budget_results: Vec<ReviewYield>,
    risk_coverage_results: Vec<RiskCoveragePoint>,
    /// Matched-budget controls generated from the same frozen rows: uncertainty, probability, and deterministic random.
    baseline_comparisons: Vec<PolicyEvaluation>,
    domain_shift_audit_status: &'static str,
    domain_shift_axis: Option<String>,
    overall_calibration: Option<CalibrationDiagnostics>,
    overall_calibration_cluster_bootstrap_95: Option<CalibrationClusterBootstrap>,
    per_domain: Option<BTreeMap<String, DomainEvaluation>>,
    ood_detector_id: Option<String>,
    ood_detector_profile_sha256: Option<String>,
    ood_detection: Option<OodDetectionMetrics>,
    scope_note: &'static str,
}

#[derive(Debug, Serialize)]
struct GroupEvaluation {
    classification: ClassificationMetrics,
    balanced_accuracy_cluster_bootstrap_95: ClusterBootstrap,
    /// Local review yield at the same item-budget fractions used for the pooled report.
    /// Counts are computed only within this group, never by borrowing other groups' cases.
    item_budget_results: Vec<ReviewYield>,
    /// Local review yield at the same minute budgets used for the pooled report.
    review_time_budget_results: Vec<ReviewYield>,
    /// Local selective-risk curve; pooled coverage must not mask a weak species/error family.
    risk_coverage_results: Vec<RiskCoveragePoint>,
    /// Matched-budget policy controls computed using only this group's rows.
    baseline_comparisons: Vec<PolicyEvaluation>,
}

#[derive(Debug, Serialize)]
struct PolicyEvaluation {
    policy: &'static str,
    item_budget_results: Vec<ReviewYield>,
    review_time_budget_results: Vec<ReviewYield>,
    /// Paired cluster-bootstrap intervals for submitted-priority minus this baseline.
    submitted_minus_baseline_cluster_bootstrap_95: Vec<PairedReviewDelta>,
}

#[derive(Debug, Serialize)]
struct PairedReviewDelta {
    budget_kind: &'static str,
    requested_budget: f64,
    requested_item_fraction: Option<f64>,
    impact_weight_recall_delta: DeltaInterval,
    error_recall_delta: DeltaInterval,
    precision_delta: DeltaInterval,
}

#[derive(Debug, Serialize)]
struct DeltaInterval {
    requested_replicates: usize,
    valid_replicates: usize,
    minimum_valid_replicates_for_interval: usize,
    interval_95: Option<Interval>,
}

#[derive(Debug, Serialize)]
struct ClusterBootstrap {
    requested_replicates: usize,
    valid_replicates: usize,
    minimum_valid_replicates_for_interval: usize,
    balanced_accuracy_95: Option<Interval>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
struct Interval {
    lower: f64,
    upper: f64,
}

#[derive(Debug, Default, Serialize)]
struct ClassificationMetrics {
    n: usize,
    positive_errors: usize,
    negative_controls: usize,
    accuracy: Option<f64>,
    balanced_accuracy: Option<f64>,
    precision_at_half: Option<f64>,
    recall_at_half: Option<f64>,
    f1_at_half: Option<f64>,
    roc_auc: Option<f64>,
    average_precision: Option<f64>,
    brier_score: Option<f64>,
    log_loss: Option<f64>,
}

#[derive(Debug, Serialize)]
struct ReviewYield {
    budget_kind: &'static str,
    requested_budget: f64,
    requested_item_fraction: Option<f64>,
    selected_count: usize,
    actual_review_minutes: f64,
    true_errors_found: usize,
    false_alarms: usize,
    precision: Option<f64>,
    error_recall: Option<f64>,
    impact_weight_captured: f64,
    impact_weight_recall: Option<f64>,
    true_errors_per_review_minute: Option<f64>,
    selection_policy: &'static str,
}

#[derive(Debug, Serialize)]
struct RiskCoveragePoint {
    requested_coverage: f64,
    achieved_coverage: f64,
    confidence_threshold: Option<f64>,
    accepted_count: usize,
    incorrect_accepted_predictions: usize,
    accepted_case_risk: Option<f64>,
    accepted_case_accuracy: Option<f64>,
}

fn sha256(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn is_lower_hex(value: &str, exact_len: usize) -> bool {
    value.len() == exact_len
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn parse_f64_list(raw: &str, name: &str, min: f64, max: f64) -> Result<Vec<f64>, String> {
    let mut values = Vec::new();
    for token in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let value: f64 = token.parse().map_err(|_| format!("{name}: invalid number {token:?}"))?;
        if !value.is_finite() || value < min || value > max {
            return Err(format!("{name}: value {token:?} must be finite and in [{min}, {max}]"));
        }
        values.push(value);
    }
    values.sort_by(f64::total_cmp);
    values.dedup_by(|a, b| a.total_cmp(b).is_eq());
    if values.is_empty() {
        return Err(format!("{name}: at least one value is required"));
    }
    Ok(values)
}

fn parse_jsonl(bytes: &[u8]) -> Result<Vec<Prediction>, String> {
    let text = std::str::from_utf8(bytes).map_err(|e| format!("input is not valid UTF-8: {e}"))?;
    let mut records = Vec::new();
    let mut ids = BTreeSet::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: Prediction = serde_json::from_str(line)
            .map_err(|e| format!("line {} is invalid or has unknown fields: {e}", index + 1))?;
        if row.sample_id.trim().is_empty() {
            return Err(format!("line {} has an empty sample_id", index + 1));
        }
        if row.group.trim().is_empty() {
            return Err(format!("line {} has an empty group", index + 1));
        }
        if row.cluster_id.trim().is_empty() {
            return Err(format!("line {} has an empty cluster_id; cluster-aware intervals require a dependency unit", index + 1));
        }
        if !ids.insert(row.sample_id.clone()) {
            return Err(format!(
                "duplicate sample_id {:?}; aggregate derived views at the source-operation level before scoring",
                row.sample_id
            ));
        }
        if !row.p_error.is_finite() || !(0.0..=1.0).contains(&row.p_error) {
            return Err(format!("line {} p_error must be finite and in [0,1]", index + 1));
        }
        if !row.review_priority_score.is_finite()
            || !(0.0..=1.0).contains(&row.review_priority_score)
        {
            return Err(format!(
                "line {} review_priority_score must be finite and in [0,1]",
                index + 1
            ));
        }
        if !row.review_minutes.is_finite() || row.review_minutes <= 0.0 {
            return Err(format!("line {} review_minutes must be finite and greater than 0", index + 1));
        }
        if !row.impact_weight.is_finite() || row.impact_weight < 0.0 {
            return Err(format!("line {} impact_weight must be finite and non-negative", index + 1));
        }
        let domain_fields = [
            row.domain_id.is_some(),
            row.shift_axis.is_some(),
            row.is_ood_domain.is_some(),
        ];
        let has_domain_metadata = domain_fields.iter().any(|present| *present);
        if has_domain_metadata && domain_fields.iter().any(|present| !present) {
            return Err(format!("line {} must provide domain_id, shift_axis, and is_ood_domain together", index + 1));
        }
        if let Some(domain_id) = &row.domain_id {
            if domain_id.trim().is_empty() {
                return Err(format!("line {} domain_id must be non-empty", index + 1));
            }
        }
        if let Some(axis) = &row.shift_axis {
            let allowed = [
                "species", "region", "source_volume", "source_archive",
                "acquisition_protocol", "segmentation_lineage",
                "modality_availability", "morphology_regime",
            ];
            if !allowed.contains(&axis.as_str()) {
                return Err(format!("line {} has unsupported shift_axis {:?}", index + 1, axis));
            }
        }
        if let Some(score) = row.ood_score {
            if !score.is_finite() || !(0.0..=1.0).contains(&score) {
                return Err(format!("line {} ood_score must be finite and in [0,1]", index + 1));
            }
            if !has_domain_metadata {
                return Err(format!("line {} ood_score requires complete domain metadata", index + 1));
            }
        }
        records.push(row);
    }
    if records.is_empty() {
        return Err("input contains no prediction records".into());
    }
    let domain_rows = records.iter().filter(|row| row.domain_id.is_some()).count();
    if domain_rows != 0 && domain_rows != records.len() {
        return Err("domain metadata must be present for every row or for none; partial domain coverage is invalid".into());
    }
    let ood_score_rows = records.iter().filter(|row| row.ood_score.is_some()).count();
    if ood_score_rows != 0 && ood_score_rows != records.len() {
        return Err("ood_score must be present for every row or for none; partial novelty scores are invalid".into());
    }
    let mut domain_contracts = BTreeMap::<String, (String, bool)>::new();
    let mut cluster_domain_contracts = BTreeMap::<String, String>::new();
    let mut axes = BTreeSet::<String>::new();
    for row in &records {
        if let (Some(domain), Some(axis), Some(is_ood)) = (&row.domain_id, &row.shift_axis, row.is_ood_domain) {
            axes.insert(axis.clone());
            let contract = (axis.clone(), is_ood);
            if let Some(previous) = domain_contracts.insert(domain.clone(), contract.clone()) {
                if previous != contract {
                    return Err(format!("domain_id {:?} has inconsistent shift_axis or is_ood_domain declarations", domain));
                }
            }
            if let Some(previous_domain) =
                cluster_domain_contracts.insert(row.cluster_id.clone(), domain.clone())
            {
                if previous_domain.as_str() != domain.as_str() {
                    return Err(format!(
                        "cluster_id {:?} crosses domain_id values {:?} and {:?}; use a domain-qualified dependency-cluster identity or a split that preserves the dependency block",
                        row.cluster_id, previous_domain, domain
                    ));
                }
            }
        }
    }
    if axes.len() > 1 {
        return Err("one scoring input may contain only one shift_axis; run separate audits for each axis".into());
    }
    let total_review_minutes: f64 = records.iter().map(|row| row.review_minutes).sum();
    if !total_review_minutes.is_finite() {
        return Err("sum of review_minutes overflows finite f64; check units and input values".into());
    }
    let total_positive_impact: f64 = records.iter()
        .filter(|row| row.label_is_error)
        .map(|row| row.impact_weight)
        .sum();
    if !total_positive_impact.is_finite() {
        return Err("sum of positive-class impact_weight overflows finite f64; check units and input values".into());
    }
    Ok(records)
}

fn classification(rows: &[&Prediction]) -> ClassificationMetrics {
    if rows.is_empty() {
        return ClassificationMetrics::default();
    }
    let n = rows.len();
    let positives = rows.iter().filter(|r| r.label_is_error).count();
    let negatives = n - positives;
    let mut tp = 0_usize;
    let mut fp = 0_usize;
    let mut tn = 0_usize;
    let mut fn_count = 0_usize;
    let mut brier = 0.0;
    let mut log_loss = 0.0;
    for r in rows {
        let prediction = r.p_error >= 0.5;
        match (prediction, r.label_is_error) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, false) => tn += 1,
            (false, true) => fn_count += 1,
        }
        let target = if r.label_is_error { 1.0 } else { 0.0 };
        brier += (r.p_error - target).powi(2);
        let p = r.p_error.clamp(1e-15, 1.0 - 1e-15);
        log_loss -= target * p.ln() + (1.0 - target) * (1.0 - p).ln();
    }
    let precision = ratio(tp, tp + fp);
    let recall = ratio(tp, tp + fn_count);
    let specificity = ratio(tn, tn + fp);
    let balanced_accuracy = match (recall, specificity) {
        (Some(sensitivity), Some(specificity)) => Some((sensitivity + specificity) / 2.0),
        _ => None,
    };
    let f1 = match (precision, recall) {
        (Some(p), Some(r)) if p + r > 0.0 => Some(2.0 * p * r / (p + r)),
        (Some(_), Some(_)) => Some(0.0),
        _ => None,
    };

    let mut asc: Vec<&Prediction> = rows.to_vec();
    asc.sort_by(|a, b| a.p_error.total_cmp(&b.p_error).then_with(|| a.sample_id.cmp(&b.sample_id)));
    let roc_auc = if positives == 0 || negatives == 0 {
        None
    } else {
        let mut neg_before = 0_usize;
        let mut concordant = 0.0;
        let mut i = 0;
        while i < asc.len() {
            let mut j = i + 1;
            while j < asc.len() && asc[j].p_error.total_cmp(&asc[i].p_error).is_eq() {
                j += 1;
            }
            let group_pos = asc[i..j].iter().filter(|r| r.label_is_error).count();
            let group_neg = (j - i) - group_pos;
            concordant += group_pos as f64 * neg_before as f64
                + 0.5 * group_pos as f64 * group_neg as f64;
            neg_before += group_neg;
            i = j;
        }
        Some(concordant / (positives as f64 * negatives as f64))
    };

    let average_precision = if positives == 0 {
        None
    } else {
        let mut desc: Vec<&Prediction> = rows.to_vec();
        desc.sort_by(|a, b| b.p_error.total_cmp(&a.p_error).then_with(|| a.sample_id.cmp(&b.sample_id)));
        let mut tp_seen = 0_usize;
        let mut fp_seen = 0_usize;
        let mut ap = 0.0;
        let mut i = 0;
        while i < desc.len() {
            let mut j = i + 1;
            while j < desc.len() && desc[j].p_error.total_cmp(&desc[i].p_error).is_eq() {
                j += 1;
            }
            let group_pos = desc[i..j].iter().filter(|r| r.label_is_error).count();
            let group_neg = (j - i) - group_pos;
            tp_seen += group_pos;
            fp_seen += group_neg;
            if group_pos > 0 {
                ap += (group_pos as f64 / positives as f64)
                    * (tp_seen as f64 / (tp_seen + fp_seen) as f64);
            }
            i = j;
        }
        Some(ap)
    };

    ClassificationMetrics {
        n,
        positive_errors: positives,
        negative_controls: negatives,
        accuracy: Some((tp + tn) as f64 / n as f64),
        balanced_accuracy,
        precision_at_half: precision,
        recall_at_half: recall,
        f1_at_half: f1,
        roc_auc,
        average_precision,
        brier_score: Some(brier / n as f64),
        log_loss: Some(log_loss / n as f64),
    }
}

fn balanced_accuracy_only(rows: &[&Prediction]) -> Option<f64> {
    let positives = rows.iter().filter(|r| r.label_is_error).count();
    let negatives = rows.len().saturating_sub(positives);
    if positives == 0 || negatives == 0 {
        return None;
    }
    let tp = rows.iter().filter(|r| r.label_is_error && r.p_error >= 0.5).count();
    let tn = rows.iter().filter(|r| !r.label_is_error && r.p_error < 0.5).count();
    Some((tp as f64 / positives as f64 + tn as f64 / negatives as f64) / 2.0)
}

fn next_u64(state: &mut u64) -> u64 {
    if *state == 0 {
        *state = 0x9e3779b97f4a7c15;
    }
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn percentile(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let index = (p * (sorted.len() - 1) as f64).floor() as usize;
    sorted.get(index).copied()
}

fn cluster_bootstrap(rows: &[&Prediction], replicates: usize, seed_material: &str) -> ClusterBootstrap {
    let mut clusters: BTreeMap<String, Vec<&Prediction>> = BTreeMap::new();
    for row in rows {
        clusters.entry(row.cluster_id.clone()).or_default().push(*row);
    }
    let cluster_rows: Vec<Vec<&Prediction>> = clusters.into_values().collect();
    let minimum_valid = ((replicates as f64) * 0.8).ceil() as usize;
    if cluster_rows.len() < 2 || replicates == 0 {
        return ClusterBootstrap {
            requested_replicates: replicates,
            valid_replicates: 0,
            minimum_valid_replicates_for_interval: minimum_valid,
            balanced_accuracy_95: None,
        };
    }

    let digest = sha256(seed_material.as_bytes());
    let mut state = u64::from_str_radix(&digest[..16], 16).unwrap_or(1);
    if state == 0 {
        state = 1;
    }
    let mut scores = Vec::with_capacity(replicates);
    for _ in 0..replicates {
        let mut sample = Vec::with_capacity(rows.len());
        for _ in 0..cluster_rows.len() {
            let index = (next_u64(&mut state) % cluster_rows.len() as u64) as usize;
            sample.extend(cluster_rows[index].iter().copied());
        }
        if let Some(score) = balanced_accuracy_only(&sample) {
            scores.push(score);
        }
    }
    scores.sort_by(f64::total_cmp);
    let interval = if scores.len() >= minimum_valid {
        Some(Interval {
            lower: percentile(&scores, 0.025).unwrap_or(0.0),
            upper: percentile(&scores, 0.975).unwrap_or(1.0),
        })
    } else {
        None
    };
    ClusterBootstrap {
        requested_replicates: replicates,
        valid_replicates: scores.len(),
        minimum_valid_replicates_for_interval: minimum_valid,
        balanced_accuracy_95: interval,
    }
}

fn scalar_percentile_interval(mut values: Vec<f64>, requested_replicates: usize) -> Option<Interval> {
    let minimum_valid = ((requested_replicates as f64) * 0.8).ceil() as usize;
    if values.len() < minimum_valid || values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    Some(Interval {
        lower: percentile(&values, 0.025)?,
        upper: percentile(&values, 0.975)?,
    })
}

fn calibration_cluster_bootstrap(
    rows: &[&Prediction],
    replicates: usize,
    seed_material: &str,
) -> CalibrationClusterBootstrap {
    let mut clusters: BTreeMap<String, Vec<&Prediction>> = BTreeMap::new();
    for row in rows {
        clusters.entry(row.cluster_id.clone()).or_default().push(*row);
    }
    let cluster_rows = clusters.into_values().collect::<Vec<_>>();
    let minimum_valid = ((replicates as f64) * 0.8).ceil() as usize;
    if cluster_rows.len() < 2 || replicates == 0 {
        return CalibrationClusterBootstrap {
            requested_replicates: replicates,
            valid_replicates: 0,
            minimum_valid_replicates_for_interval: minimum_valid,
            expected_calibration_error_95: None,
            maximum_calibration_error_95: None,
            adaptive_confidence_ece_15_95: None,
            reliability_curve_ece_12_95: None,
            reliability_curve_observed_accuracy_95: Vec::new(),
            brier_score_95: None,
        };
    }

    let digest = sha256(seed_material.as_bytes());
    let mut state = u64::from_str_radix(&digest[..16], 16).unwrap_or(1);
    if state == 0 { state = 1; }
    let mut ece = Vec::with_capacity(replicates);
    let mut mce = Vec::with_capacity(replicates);
    let mut adaptive_ece = Vec::with_capacity(replicates);
    let mut reliability_ece = Vec::with_capacity(replicates);
    let reliability_reference = adaptive_confidence_bins(rows, 12);
    let mut reliability_accuracy_by_bin = vec![Vec::with_capacity(replicates); reliability_reference.len()];
    let mut reliability_confidence_by_bin = vec![Vec::with_capacity(replicates); reliability_reference.len()];
    let mut brier = Vec::with_capacity(replicates);
    for _ in 0..replicates {
        let mut sample = Vec::with_capacity(rows.len());
        for _ in 0..cluster_rows.len() {
            let index = (next_u64(&mut state) % cluster_rows.len() as u64) as usize;
            sample.extend(cluster_rows[index].iter().copied());
        }
        let metrics = calibration_diagnostics(&sample);
        if !reliability_reference.is_empty() {
            let mut bin_counts = vec![0_usize; reliability_reference.len()];
            let mut bin_correct = vec![0_usize; reliability_reference.len()];
            let mut bin_confidence_sums = vec![0.0_f64; reliability_reference.len()];
            for row in &sample {
                let confidence = predicted_confidence(row);
                let bin_index = reliability_reference.iter()
                    .position(|bin| confidence <= bin.maximum_confidence)
                    .unwrap_or(reliability_reference.len() - 1);
                bin_counts[bin_index] += 1;
                bin_confidence_sums[bin_index] += confidence;
                if (row.p_error >= 0.5) == row.label_is_error {
                    bin_correct[bin_index] += 1;
                }
            }
            for bin_index in 0..reliability_reference.len() {
                if bin_counts[bin_index] > 0 {
                    reliability_accuracy_by_bin[bin_index].push(
                        bin_correct[bin_index] as f64 / bin_counts[bin_index] as f64
                    );
                    reliability_confidence_by_bin[bin_index].push(
                        bin_confidence_sums[bin_index] / bin_counts[bin_index] as f64
                    );
                }
            }
        }
        if let (Some(e), Some(m), Some(a), Some(r), Some(b)) = (
            metrics.expected_calibration_error,
            metrics.maximum_calibration_error,
            metrics.adaptive_confidence_expected_calibration_error,
            metrics.reliability_curve_expected_calibration_error,
            metrics.brier_score,
        ) {
            if e.is_finite() && m.is_finite() && a.is_finite() && r.is_finite() && b.is_finite() {
                ece.push(e);
                mce.push(m);
                adaptive_ece.push(a);
                reliability_ece.push(r);
                brier.push(b);
            }
        }
    }
    let per_bin_intervals = reliability_reference.iter().enumerate().map(|(index, bin)| {
        let accuracy_values = std::mem::take(&mut reliability_accuracy_by_bin[index]);
        let confidence_values = std::mem::take(&mut reliability_confidence_by_bin[index]);
        ConfidenceBinClusterBootstrap {
            index: bin.index,
            full_sample_count: bin.count,
            requested_replicates: replicates,
            valid_replicates: accuracy_values.len(),
            minimum_valid_replicates_for_interval: minimum_valid,
            observed_accuracy_95: scalar_percentile_interval(accuracy_values, replicates),
            mean_confidence_95: scalar_percentile_interval(confidence_values, replicates),
        }
    }).collect();
    CalibrationClusterBootstrap {
        requested_replicates: replicates,
        valid_replicates: ece.len(),
        minimum_valid_replicates_for_interval: minimum_valid,
        expected_calibration_error_95: scalar_percentile_interval(ece, replicates),
        maximum_calibration_error_95: scalar_percentile_interval(mce, replicates),
        adaptive_confidence_ece_15_95: scalar_percentile_interval(adaptive_ece, replicates),
        reliability_curve_ece_12_95: scalar_percentile_interval(reliability_ece, replicates),
        reliability_curve_observed_accuracy_95: per_bin_intervals,
        brier_score_95: scalar_percentile_interval(brier, replicates),
    }
}

fn ood_cluster_bootstrap(
    rows: &[&Prediction],
    replicates: usize,
    seed_material: &str,
) -> OodClusterBootstrap {
    let mut clusters: BTreeMap<String, Vec<&Prediction>> = BTreeMap::new();
    for row in rows {
        clusters.entry(row.cluster_id.clone()).or_default().push(*row);
    }
    let cluster_rows = clusters.into_values().collect::<Vec<_>>();
    let minimum_valid = ((replicates as f64) * 0.8).ceil() as usize;
    if cluster_rows.len() < 2 || replicates == 0 {
        return OodClusterBootstrap {
            requested_replicates: replicates,
            valid_replicates: 0,
            minimum_valid_replicates_for_interval: minimum_valid,
            roc_auc_95: None,
            average_precision_95: None,
        };
    }

    let digest = sha256(seed_material.as_bytes());
    let mut state = u64::from_str_radix(&digest[..16], 16).unwrap_or(1);
    if state == 0 { state = 1; }
    let mut auc = Vec::with_capacity(replicates);
    let mut average_precision = Vec::with_capacity(replicates);
    for _ in 0..replicates {
        let mut sample = Vec::with_capacity(rows.len());
        for _ in 0..cluster_rows.len() {
            let index = (next_u64(&mut state) % cluster_rows.len() as u64) as usize;
            sample.extend(cluster_rows[index].iter().copied());
        }
        let scores = sample.iter().filter_map(|row| {
            Some((row.is_ood_domain?, row.ood_score?))
        }).collect::<Vec<_>>();
        let (sample_auc, sample_ap) = binary_ranking_metrics(&scores);
        if let (Some(a), Some(ap)) = (sample_auc, sample_ap) {
            if a.is_finite() && ap.is_finite() {
                auc.push(a);
                average_precision.push(ap);
            }
        }
    }
    OodClusterBootstrap {
        requested_replicates: replicates,
        valid_replicates: auc.len(),
        minimum_valid_replicates_for_interval: minimum_valid,
        roc_auc_95: scalar_percentile_interval(auc, replicates),
        average_precision_95: scalar_percentile_interval(average_precision, replicates),
    }
}

fn ratio(num: usize, den: usize) -> Option<f64> {
    if den == 0 { None } else { Some(num as f64 / den as f64) }
}

#[derive(Debug, Clone, Copy)]
enum RankingPolicy {
    UncertaintyOnly,
    ProbabilityOnly,
    DeterministicRandom,
    PriorityScore,
}

impl RankingPolicy {
    fn id(self) -> &'static str {
        match self {
            Self::UncertaintyOnly => "uncertainty_only",
            Self::ProbabilityOnly => "probability_only",
            Self::DeterministicRandom => "deterministic_random",
            Self::PriorityScore => "submitted_priority_score",
        }
    }
}

fn ranked_for_policy<'a>(
    rows: &[&'a Prediction],
    policy: RankingPolicy,
    seed_material: &str,
) -> Vec<&'a Prediction> {
    let mut result = rows.to_vec();
    match policy {
        RankingPolicy::PriorityScore => result.sort_by(|a, b| {
            b.review_priority_score
                .total_cmp(&a.review_priority_score)
                .then_with(|| a.sample_id.cmp(&b.sample_id))
        }),
        RankingPolicy::UncertaintyOnly => result.sort_by(|a, b| {
            let confidence_a = a.p_error.max(1.0 - a.p_error);
            let confidence_b = b.p_error.max(1.0 - b.p_error);
            confidence_a.total_cmp(&confidence_b)
                .then_with(|| a.sample_id.cmp(&b.sample_id))
        }),
        RankingPolicy::ProbabilityOnly => result.sort_by(|a, b| {
            b.p_error.total_cmp(&a.p_error)
                .then_with(|| a.sample_id.cmp(&b.sample_id))
        }),
        RankingPolicy::DeterministicRandom => {
            let mut keyed: Vec<(String, &'a Prediction)> = result.into_iter()
                .map(|row| {
                    let key = sha256(format!("{seed_material}:{}", row.sample_id).as_bytes());
                    (key, row)
                })
                .collect();
            keyed.sort_by(|(key_a, row_a), (key_b, row_b)| {
                key_a.cmp(key_b).then_with(|| row_a.sample_id.cmp(&row_b.sample_id))
            });
            result = keyed.into_iter().map(|(_, row)| row).collect();
        }
    }
    result
}

fn review_yield(
    rows: &[&Prediction],
    requested_budget: f64,
    requested_item_fraction: Option<f64>,
    budget_kind: &'static str,
) -> ReviewYield {
    review_yield_for_policy(
        rows,
        requested_budget,
        requested_item_fraction,
        budget_kind,
        RankingPolicy::PriorityScore,
        "primary",
    )
}

fn review_yield_for_policy(
    rows: &[&Prediction],
    requested_budget: f64,
    requested_item_fraction: Option<f64>,
    budget_kind: &'static str,
    policy: RankingPolicy,
    seed_material: &str,
) -> ReviewYield {
    let ordered = ranked_for_policy(rows, policy, seed_material);
    let total_errors = rows.iter().filter(|r| r.label_is_error).count();
    let total_impact: f64 = rows.iter().filter(|r| r.label_is_error).map(|r| r.impact_weight).sum();
    let mut selected = Vec::new();
    let mut used_minutes = 0.0;
    if budget_kind == "items" {
        let count = (requested_budget.ceil() as usize).min(ordered.len());
        selected.extend(ordered.into_iter().take(count));
        used_minutes = selected.iter().map(|r| r.review_minutes).sum();
    } else {
        for row in ordered {
            if used_minutes + row.review_minutes <= requested_budget + 1e-9 {
                used_minutes += row.review_minutes;
                selected.push(row);
            }
        }
    }
    let found = selected.iter().filter(|r| r.label_is_error).count();
    let false_alarms = selected.iter().filter(|r| !r.label_is_error).count();
    let captured_impact: f64 = selected.iter()
        .filter(|r| r.label_is_error)
        .map(|r| r.impact_weight)
        .sum();
    ReviewYield {
        budget_kind,
        requested_budget,
        requested_item_fraction,
        selected_count: selected.len(),
        actual_review_minutes: used_minutes,
        true_errors_found: found,
        false_alarms,
        precision: ratio(found, selected.len()),
        error_recall: ratio(found, total_errors),
        impact_weight_captured: captured_impact,
        impact_weight_recall: if total_impact > 0.0 { Some(captured_impact / total_impact) } else { None },
        true_errors_per_review_minute: if used_minutes > 0.0 { Some(found as f64 / used_minutes) } else { None },
        selection_policy: policy.id(),
    }
}

fn risk_coverage(rows: &[&Prediction], targets: &[f64]) -> Vec<RiskCoveragePoint> {
    let mut ordered = rows.to_vec();
    ordered.sort_by(|a, b| {
        let ca = a.p_error.max(1.0 - a.p_error);
        let cb = b.p_error.max(1.0 - b.p_error);
        cb.total_cmp(&ca).then_with(|| a.sample_id.cmp(&b.sample_id))
    });
    let n = ordered.len();
    targets.iter().map(|target| {
        let desired = ((*target * n as f64).ceil() as usize).max(1).min(n);
        let mut accepted = 0_usize;
        let mut incorrect = 0_usize;
        let mut i = 0;
        let mut last_threshold = None;
        while i < n && accepted < desired {
            let confidence = ordered[i].p_error.max(1.0 - ordered[i].p_error);
            let mut j = i + 1;
            while j < n && ordered[j].p_error.max(1.0 - ordered[j].p_error).total_cmp(&confidence).is_eq() {
                j += 1;
            }
            for row in &ordered[i..j] {
                let predicted_error = row.p_error >= 0.5;
                if predicted_error != row.label_is_error {
                    incorrect += 1;
                }
            }
            accepted += j - i;
            last_threshold = Some(confidence);
            i = j;
        }
        RiskCoveragePoint {
            requested_coverage: *target,
            achieved_coverage: accepted as f64 / n as f64,
            confidence_threshold: last_threshold,
            accepted_count: accepted,
            incorrect_accepted_predictions: incorrect,
            accepted_case_risk: ratio(incorrect, accepted),
            accepted_case_accuracy: ratio(accepted - incorrect, accepted),
        }
    }).collect()
}

fn evaluate_group(
    rows: &[&Prediction],
    item_fractions: &[f64],
    minute_budgets: &[f64],
    coverage_targets: &[f64],
    bootstrap_replicates: usize,
    seed_material: &str,
) -> GroupEvaluation {
    let item_budget_results = item_fractions.iter().map(|fraction| {
        let count = (*fraction * rows.len() as f64).ceil().max(1.0);
        review_yield(rows, count, Some(*fraction), "items")
    }).collect();
    let review_time_budget_results = minute_budgets.iter().map(|minutes| {
        review_yield(rows, *minutes, None, "minutes")
    }).collect();
    GroupEvaluation {
        classification: classification(rows),
        balanced_accuracy_cluster_bootstrap_95: cluster_bootstrap(
            rows,
            bootstrap_replicates,
            seed_material,
        ),
        item_budget_results,
        review_time_budget_results,
        risk_coverage_results: risk_coverage(rows, coverage_targets),
        baseline_comparisons: baseline_policy_results(
            rows,
            item_fractions,
            minute_budgets,
            bootstrap_replicates,
            seed_material,
        ),
    }
}

fn percentile_interval(mut values: Vec<f64>, replicates: usize) -> DeltaInterval {
    values.sort_by(f64::total_cmp);
    let valid = values.len();
    let minimum_valid = ((replicates as f64) * 0.8).ceil() as usize;
    let interval_95 = if valid >= minimum_valid {
        Some(Interval {
            lower: percentile(&values, 0.025).unwrap_or(0.0),
            upper: percentile(&values, 0.975).unwrap_or(1.0),
        })
    } else {
        None
    };
    DeltaInterval {
        requested_replicates: replicates,
        valid_replicates: valid,
        minimum_valid_replicates_for_interval: minimum_valid,
        interval_95,
    }
}

fn paired_delta_bootstrap(
    rows: &[&Prediction],
    baseline: RankingPolicy,
    budget_kind: &'static str,
    requested_budget: f64,
    item_fraction: Option<f64>,
    replicates: usize,
    seed_material: &str,
) -> PairedReviewDelta {
    let mut clusters: BTreeMap<String, Vec<&Prediction>> = BTreeMap::new();
    for row in rows {
        clusters.entry(row.cluster_id.clone()).or_default().push(*row);
    }
    let cluster_rows: Vec<Vec<&Prediction>> = clusters.into_values().collect();
    let mut impact_deltas = Vec::with_capacity(replicates);
    let mut recall_deltas = Vec::with_capacity(replicates);
    let mut precision_deltas = Vec::with_capacity(replicates);

    if cluster_rows.len() >= 2 && replicates > 0 {
        let digest = sha256(seed_material.as_bytes());
        let mut state = u64::from_str_radix(&digest[..16], 16).unwrap_or(1);
        if state == 0 { state = 1; }

        for _ in 0..replicates {
            // Resample dependency clusters as paired units; both policies see exactly the same sample.
            let mut sample = Vec::with_capacity(rows.len());
            for _ in 0..cluster_rows.len() {
                let index = (next_u64(&mut state) % cluster_rows.len() as u64) as usize;
                sample.extend(cluster_rows[index].iter().copied());
            }

            let budget = if budget_kind == "items" {
                (item_fraction.unwrap_or(0.0) * sample.len() as f64).ceil().max(1.0)
            } else {
                requested_budget
            };
            let primary = review_yield_for_policy(
                &sample, budget, item_fraction, budget_kind,
                RankingPolicy::PriorityScore, seed_material,
            );
            let comparator = review_yield_for_policy(
                &sample, budget, item_fraction, budget_kind, baseline, seed_material,
            );

            if let (Some(a), Some(b)) = (primary.impact_weight_recall, comparator.impact_weight_recall) {
                impact_deltas.push(a - b);
            }
            if let (Some(a), Some(b)) = (primary.error_recall, comparator.error_recall) {
                recall_deltas.push(a - b);
            }
            if let (Some(a), Some(b)) = (primary.precision, comparator.precision) {
                precision_deltas.push(a - b);
            }
        }
    }

    PairedReviewDelta {
        budget_kind,
        requested_budget,
        requested_item_fraction: item_fraction,
        impact_weight_recall_delta: percentile_interval(impact_deltas, replicates),
        error_recall_delta: percentile_interval(recall_deltas, replicates),
        precision_delta: percentile_interval(precision_deltas, replicates),
    }
}

fn paired_deltas_for_policy(
    rows: &[&Prediction],
    baseline: RankingPolicy,
    item_fractions: &[f64],
    minute_budgets: &[f64],
    replicates: usize,
    seed_prefix: &str,
) -> Vec<PairedReviewDelta> {
    let mut results = Vec::with_capacity(item_fractions.len() + minute_budgets.len());
    for fraction in item_fractions {
        let count = (*fraction * rows.len() as f64).ceil().max(1.0);
        results.push(paired_delta_bootstrap(
            rows, baseline, "items", count, Some(*fraction), replicates,
            seed_prefix,
        ));
    }
    for minutes in minute_budgets {
        results.push(paired_delta_bootstrap(
            rows, baseline, "minutes", *minutes, None, replicates,
            seed_prefix,
        ));
    }
    results
}

fn baseline_policy_results(
    rows: &[&Prediction],
    item_fractions: &[f64],
    minute_budgets: &[f64],
    bootstrap_replicates: usize,
    seed_prefix: &str,
) -> Vec<PolicyEvaluation> {
    let policies = [
        RankingPolicy::UncertaintyOnly,
        RankingPolicy::ProbabilityOnly,
        RankingPolicy::DeterministicRandom,
    ];
    policies.iter().map(|policy| {
        let item_budget_results = item_fractions.iter().map(|fraction| {
            let count = (*fraction * rows.len() as f64).ceil().max(1.0);
            review_yield_for_policy(
                rows, count, Some(*fraction), "items", *policy, seed_prefix,
            )
        }).collect();
        let review_time_budget_results = minute_budgets.iter().map(|minutes| {
            review_yield_for_policy(
                rows, *minutes, None, "minutes", *policy, seed_prefix,
            )
        }).collect();
        PolicyEvaluation {
            policy: policy.id(),
            item_budget_results,
            review_time_budget_results,
            submitted_minus_baseline_cluster_bootstrap_95: paired_deltas_for_policy(
                rows, *policy, item_fractions, minute_budgets, bootstrap_replicates, seed_prefix,
            ),
        }
    }).collect()
}

fn calibration_threshold_risk(rows: &[&Prediction], threshold: f64) -> ConfidenceThresholdRisk {
    let selected = rows.iter().filter(|row| row.p_error >= threshold).collect::<Vec<_>>();
    let false_accepts = selected.iter().filter(|row| !row.label_is_error).count();
    ConfidenceThresholdRisk {
        threshold,
        selected_count: selected.len(),
        false_accept_count: false_accepts,
        false_accept_fraction: ratio(false_accepts, selected.len()),
        precision: ratio(selected.len() - false_accepts, selected.len()),
    }
}

fn predicted_confidence(row: &Prediction) -> f64 {
    row.p_error.max(1.0 - row.p_error)
}

fn confidence_calibration_bin(
    rows: &[&Prediction],
    index: usize,
) -> ConfidenceCalibrationBin {
    let minimum_confidence = rows.iter()
        .map(|row| predicted_confidence(row))
        .fold(f64::INFINITY, f64::min);
    let maximum_confidence = rows.iter()
        .map(|row| predicted_confidence(row))
        .fold(f64::NEG_INFINITY, f64::max);
    let confidence_sum = rows.iter().map(|row| predicted_confidence(row)).sum::<f64>();
    let correct_count = rows.iter().filter(|row| {
        (row.p_error >= 0.5) == row.label_is_error
    }).count();
    let mean_confidence = confidence_sum / rows.len() as f64;
    let observed_accuracy = correct_count as f64 / rows.len() as f64;
    ConfidenceCalibrationBin {
        index,
        count: rows.len(),
        minimum_confidence,
        maximum_confidence,
        mean_confidence,
        observed_accuracy,
        absolute_calibration_gap: (mean_confidence - observed_accuracy).abs(),
    }
}

fn adaptive_confidence_bins(
    rows: &[&Prediction],
    requested_bins: usize,
) -> Vec<ConfidenceCalibrationBin> {
    if rows.is_empty() || requested_bins == 0 {
        return Vec::new();
    }
    let mut ordered = rows.to_vec();
    ordered.sort_by(|a, b| {
        predicted_confidence(a).total_cmp(&predicted_confidence(b))
            .then_with(|| a.sample_id.cmp(&b.sample_id))
    });
    let mut bins = Vec::with_capacity(requested_bins.min(rows.len()));
    let mut start = 0_usize;
    for bin_number in 1..=requested_bins {
        if start >= ordered.len() {
            break;
        }
        // ceil(N * k / B) without multiplying N*k directly.
        let quotient = ordered.len() / requested_bins;
        let remainder = ordered.len() % requested_bins;
        let target = quotient * bin_number
            + (remainder * bin_number + requested_bins - 1) / requested_bins;
        if target <= start {
            continue;
        }
        let mut end = target.min(ordered.len());
        // Never split equal-confidence ties across quantile bins. Tied predictions
        // cannot acquire different calibration treatment from sample-ID ordering.
        while end < ordered.len()
            && predicted_confidence(ordered[end - 1]).total_cmp(&predicted_confidence(ordered[end])).is_eq()
        {
            end += 1;
        }
        bins.push(confidence_calibration_bin(&ordered[start..end], bins.len()));
        start = end;
    }
    if start < ordered.len() {
        bins.push(confidence_calibration_bin(&ordered[start..], bins.len()));
    }
    bins
}

fn confidence_bins_ece(bins: &[ConfidenceCalibrationBin], total_rows: usize) -> Option<f64> {
    if total_rows == 0 || bins.is_empty() {
        return None;
    }
    Some(bins.iter()
        .map(|bin| bin.count as f64 / total_rows as f64 * bin.absolute_calibration_gap)
        .sum())
}

fn calibration_diagnostics(rows: &[&Prediction]) -> CalibrationDiagnostics {
    const BIN_COUNT: usize = 10;
    let mut counts = [0_usize; BIN_COUNT];
    let mut probability_sums = [0.0_f64; BIN_COUNT];
    let mut positive_counts = [0_usize; BIN_COUNT];
    let mut brier_sum = 0.0_f64;
    let mut log_loss_sum = 0.0_f64;
    let mut low_confidence_error_count = 0_usize;
    let mut low_confidence_error_impact = 0.0_f64;

    for row in rows {
        let bin = ((row.p_error * BIN_COUNT as f64).floor() as usize).min(BIN_COUNT - 1);
        counts[bin] += 1;
        probability_sums[bin] += row.p_error;
        if row.label_is_error {
            positive_counts[bin] += 1;
        }
        let target = if row.label_is_error { 1.0 } else { 0.0 };
        brier_sum += (row.p_error - target).powi(2);
        let p = row.p_error.clamp(1e-15, 1.0 - 1e-15);
        log_loss_sum -= target * p.ln() + (1.0 - target) * (1.0 - p).ln();
        if row.label_is_error && row.p_error < 0.10 {
            low_confidence_error_count += 1;
            low_confidence_error_impact += row.impact_weight;
        }
    }

    let adaptive_confidence_bins_15 = adaptive_confidence_bins(rows, 15);
    let reliability_curve_bins_12 = adaptive_confidence_bins(rows, 12);
    let adaptive_confidence_ece = confidence_bins_ece(&adaptive_confidence_bins_15, rows.len());
    let adaptive_confidence_mce = adaptive_confidence_bins_15.iter()
        .map(|bin| bin.absolute_calibration_gap)
        .reduce(f64::max);
    let reliability_curve_ece = confidence_bins_ece(&reliability_curve_bins_12, rows.len());

    let mut bins = Vec::with_capacity(BIN_COUNT);
    let mut ece = 0.0_f64;
    let mut max_gap = 0.0_f64;
    for index in 0..BIN_COUNT {
        let count = counts[index];
        let mean = if count == 0 { None } else { Some(probability_sums[index] / count as f64) };
        let observed = if count == 0 { None } else { Some(positive_counts[index] as f64 / count as f64) };
        let gap = match (mean, observed) {
            (Some(predicted), Some(actual)) => {
                let value = (predicted - actual).abs();
                ece += count as f64 / rows.len().max(1) as f64 * value;
                max_gap = max_gap.max(value);
                Some(value)
            }
            _ => None,
        };
        bins.push(CalibrationBin {
            index,
            lower: index as f64 / BIN_COUNT as f64,
            upper: (index + 1) as f64 / BIN_COUNT as f64,
            upper_inclusive: index == BIN_COUNT - 1,
            count,
            mean_predicted_probability: mean,
            observed_error_rate: observed,
            absolute_calibration_gap: gap,
        });
    }

    CalibrationDiagnostics {
        fixed_bin_count: BIN_COUNT,
        adaptive_confidence_bin_count_requested: 15,
        adaptive_confidence_bins: adaptive_confidence_bins_15,
        adaptive_confidence_expected_calibration_error: adaptive_confidence_ece,
        adaptive_confidence_maximum_calibration_error: adaptive_confidence_mce,
        reliability_curve_bin_count_requested: 12,
        reliability_curve_bins: reliability_curve_bins_12,
        reliability_curve_expected_calibration_error: reliability_curve_ece,
        brier_score: if rows.is_empty() { None } else { Some(brier_sum / rows.len() as f64) },
        log_loss: if rows.is_empty() { None } else { Some(log_loss_sum / rows.len() as f64) },
        expected_calibration_error: if rows.is_empty() { None } else { Some(ece) },
        maximum_calibration_error: if rows.is_empty() { None } else { Some(max_gap) },
        bins,
        high_confidence_at_90: calibration_threshold_risk(rows, 0.90),
        high_confidence_at_95: calibration_threshold_risk(rows, 0.95),
        low_confidence_below_10: LowConfidenceErrorMass {
            threshold: 0.10,
            missed_error_count: low_confidence_error_count,
            missed_error_impact_weight: low_confidence_error_impact,
        },
    }
}

fn binary_ranking_metrics(labels_and_scores: &[(bool, f64)]) -> (Option<f64>, Option<f64>) {
    let positives = labels_and_scores.iter().filter(|(label, _)| *label).count();
    let negatives = labels_and_scores.len() - positives;
    if positives == 0 || negatives == 0 {
        return (None, None);
    }

    let mut ascending = labels_and_scores.to_vec();
    ascending.sort_by(|a, b| a.1.total_cmp(&b.1));
    let mut negatives_before = 0_usize;
    let mut concordant = 0.0_f64;
    let mut index = 0_usize;
    while index < ascending.len() {
        let mut end = index + 1;
        while end < ascending.len() && ascending[end].1.total_cmp(&ascending[index].1).is_eq() {
            end += 1;
        }
        let group_positives = ascending[index..end].iter().filter(|(label, _)| *label).count();
        let group_negatives = end - index - group_positives;
        concordant += group_positives as f64 * negatives_before as f64
            + 0.5 * group_positives as f64 * group_negatives as f64;
        negatives_before += group_negatives;
        index = end;
    }
    let roc_auc = Some(concordant / (positives as f64 * negatives as f64));

    let mut descending = labels_and_scores.to_vec();
    descending.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut seen_positive = 0_usize;
    let mut seen_negative = 0_usize;
    let mut average_precision = 0.0_f64;
    index = 0;
    while index < descending.len() {
        let mut end = index + 1;
        while end < descending.len() && descending[end].1.total_cmp(&descending[index].1).is_eq() {
            end += 1;
        }
        let group_positives = descending[index..end].iter().filter(|(label, _)| *label).count();
        let group_negatives = end - index - group_positives;
        seen_positive += group_positives;
        seen_negative += group_negatives;
        if group_positives > 0 {
            average_precision += (group_positives as f64 / positives as f64)
                * (seen_positive as f64 / (seen_positive + seen_negative) as f64);
        }
        index = end;
    }
    (roc_auc, Some(average_precision))
}

fn ood_detection_metrics(
    rows: &[&Prediction],
    threshold: Option<f64>,
    bootstrap_replicates: usize,
    seed_material: &str,
) -> OodDetectionMetrics {
    let labelled_scores = rows.iter().filter_map(|row| {
        Some((row.is_ood_domain?, row.ood_score?))
    }).collect::<Vec<_>>();
    let (roc_auc, average_precision) = binary_ranking_metrics(&labelled_scores);
    let in_domain_count = rows.iter().filter(|row| row.is_ood_domain == Some(false)).count();
    let held_out_domain_count = rows.iter().filter(|row| row.is_ood_domain == Some(true)).count();
    let operating = threshold.map(|cutoff| {
        let false_ood_count = rows.iter().filter(|row| {
            row.is_ood_domain == Some(false) && row.ood_score.unwrap_or(0.0) >= cutoff
        }).count();
        let true_ood_count = rows.iter().filter(|row| {
            row.is_ood_domain == Some(true) && row.ood_score.unwrap_or(0.0) >= cutoff
        }).count();
        let missed_ood_count = held_out_domain_count - true_ood_count;
        (
            false_ood_count,
            missed_ood_count,
            ratio(false_ood_count, in_domain_count),
            ratio(true_ood_count, held_out_domain_count),
        )
    });

    let cluster_count = rows.iter().map(|row| row.cluster_id.as_str()).collect::<BTreeSet<_>>().len();
    let status = if in_domain_count > 0 && held_out_domain_count > 0
        && roc_auc.is_some() && average_precision.is_some()
    {
        "ESTIMABLE"
    } else {
        "NOT_ESTIMABLE"
    };
    OodDetectionMetrics {
        status,
        n: rows.len(),
        independent_cluster_count: cluster_count,
        in_domain_count,
        held_out_domain_count,
        roc_auc,
        average_precision,
        cluster_bootstrap_95: ood_cluster_bootstrap(
            rows,
            bootstrap_replicates,
            seed_material,
        ),
        frozen_threshold: threshold,
        false_ood_count: operating.map(|value| value.0),
        false_ood_rate_on_in_domain: operating.map(|value| value.2),
        missed_ood_count: operating.map(|value| value.1),
        held_out_domain_recall: operating.map(|value| value.3),
    }
}

fn domain_shift_evaluation(
    records: &[Prediction],
    required: bool,
    ood_threshold: Option<f64>,
    bootstrap_replicates: usize,
    seed_material: &str,
) -> Result<(
    &'static str,
    Option<String>,
    Option<CalibrationDiagnostics>,
    Option<CalibrationClusterBootstrap>,
    Option<BTreeMap<String, DomainEvaluation>>,
    Option<OodDetectionMetrics>,
), String> {
    let tagged = records.iter().any(|row| row.domain_id.is_some());
    if !tagged {
        if required {
            return Err("--require-domain-shift-audit was set but no domain metadata is present".into());
        }
        if ood_threshold.is_some() {
            return Err("--ood-score-threshold requires complete domain metadata and novelty scores".into());
        }
        return Ok(("NOT_PROVIDED", None, None, None, None, None));
    }

    let axis = records.first().and_then(|row| row.shift_axis.clone())
        .ok_or_else(|| "domain-shift audit is missing shift_axis".to_string())?;
    let mut grouped = BTreeMap::<String, Vec<&Prediction>>::new();
    for row in records {
        let domain = row.domain_id.as_ref().ok_or_else(|| "domain-shift audit has partial domain IDs".to_string())?;
        grouped.entry(domain.clone()).or_default().push(row);
    }
    let mut per_domain = BTreeMap::new();
    for (domain_id, rows) in grouped {
        let first = rows[0];
        let clusters = rows.iter().map(|row| row.cluster_id.as_str()).collect::<BTreeSet<_>>();
        let refs = rows.iter().copied().collect::<Vec<_>>();
        per_domain.insert(domain_id.clone(), DomainEvaluation {
            domain_id: domain_id.clone(),
            shift_axis: axis.clone(),
            is_ood_domain: first.is_ood_domain.unwrap_or(false),
            n: rows.len(),
            independent_cluster_count: clusters.len(),
            classification: classification(&refs),
            calibration: calibration_diagnostics(&refs),
            calibration_cluster_bootstrap_95: calibration_cluster_bootstrap(
                &refs,
                bootstrap_replicates,
                &format!("{seed_material}:domain:{domain_id}:calibration"),
            ),
        });
    }

    let novelty_present = records.iter().all(|row| row.ood_score.is_some());
    if ood_threshold.is_some() && !novelty_present {
        return Err("--ood-score-threshold requires an ood_score for every row".into());
    }
    let refs = records.iter().collect::<Vec<_>>();
    let overall_calibration = Some(calibration_diagnostics(&refs));
    let overall_calibration_bootstrap = Some(calibration_cluster_bootstrap(
        &refs,
        bootstrap_replicates,
        &format!("{seed_material}:overall-calibration"),
    ));
    let ood_detection = if novelty_present {
        Some(ood_detection_metrics(
            &refs,
            ood_threshold,
            bootstrap_replicates,
            &format!("{seed_material}:ood-detection"),
        ))
    } else {
        None
    };
    let status = match ood_detection.as_ref() {
        Some(metrics) if metrics.status == "ESTIMABLE" => "OOD_SCORED",
        Some(_) => "OOD_NOT_ESTIMABLE",
        None => "DOMAIN_STRATIFIED_ONLY",
    };
    Ok((
        status,
        Some(axis),
        overall_calibration,
        overall_calibration_bootstrap,
        Some(per_domain),
        ood_detection,
    ))
}


fn validate_ood_detector_provenance(
    records: &[Prediction],
    detector_id: Option<&str>,
    detector_profile_sha256: Option<&str>,
) -> Result<(), String> {
    let novelty_present = !records.is_empty() && records.iter().all(|row| row.ood_score.is_some());
    match (detector_id, detector_profile_sha256) {
        (Some(id), Some(digest)) => {
            if id.trim().is_empty() {
                return Err("--ood-detector-id must be non-empty".into());
            }
            if !is_lower_hex(digest, 64) {
                return Err("--ood-detector-profile-sha256 must be a lowercase 64-character SHA-256 digest".into());
            }
            if !novelty_present {
                return Err("OOD detector provenance was supplied but the input contains no complete ood_score column".into());
            }
        }
        (None, None) => {
            if novelty_present {
                return Err("every ood_score run requires --ood-detector-id and --ood-detector-profile-sha256".into());
            }
        }
        _ => return Err("--ood-detector-id and --ood-detector-profile-sha256 must be supplied together".into()),
    }
    Ok(())
}

fn run() -> Result<i32, String> {
    let args = Args::parse();
    if args.mapping_reference.trim().is_empty() {
        return Err("--mapping-reference must identify the reviewed label normalization contract".into());
    }
    if args.bootstrap_replicates < 100 {
        return Err("--bootstrap-replicates must be at least 100".into());
    }
    if let Some(threshold) = args.ood_score_threshold {
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err("--ood-score-threshold must be finite and in [0,1]".into());
        }
    }
    if !is_lower_hex(&args.dataset_revision_commit_sha, 40) {
        return Err("--dataset-revision-commit-sha must be a lowercase 40-character Git SHA".into());
    }
    if !is_lower_hex(&args.prediction_profile_sha256, 64) {
        return Err("--prediction-profile-sha256 must be a lowercase 64-character SHA-256 digest".into());
    }
    if args.policy_id.trim().is_empty() {
        return Err("--policy-id must be a non-empty stable identifier".into());
    }
    let item_fractions = parse_f64_list(&args.item_budget_fractions, "item_budget_fractions", f64::MIN_POSITIVE, 1.0)?;
    let minute_budgets = parse_f64_list(&args.review_budgets_minutes, "review_budgets_minutes", f64::MIN_POSITIVE, f64::MAX)?;
    let coverage_targets = parse_f64_list(&args.coverage_targets, "coverage_targets", f64::MIN_POSITIVE, 1.0)?;
    let input_bytes = fs::read(&args.input)
        .map_err(|e| format!("failed to read input {}: {e}", args.input.display()))?;
    let input_hash = sha256(&input_bytes);
    let records = parse_jsonl(&input_bytes)?;
    validate_ood_detector_provenance(
        &records,
        args.ood_detector_id.as_deref(),
        args.ood_detector_profile_sha256.as_deref(),
    )?;
    let (
        domain_shift_audit_status,
        domain_shift_axis,
        overall_calibration,
        overall_calibration_cluster_bootstrap_95,
        per_domain,
        ood_detection,
    ) = domain_shift_evaluation(
        &records,
        args.require_domain_shift_audit,
        args.ood_score_threshold,
        args.bootstrap_replicates,
        &input_hash,
    )?;
    let refs: Vec<&Prediction> = records.iter().collect();
    let overall = classification(&refs);
    let overall_cluster_bootstrap_95 = cluster_bootstrap(
        &refs,
        args.bootstrap_replicates,
        &format!("{input_hash}:overall"),
    );
    let mut by_group: BTreeMap<String, Vec<&Prediction>> = BTreeMap::new();
    for row in &records {
        by_group.entry(row.group.clone()).or_default().push(row);
    }
    let per_group = by_group.iter().map(|(group, rows)| {
        (
            group.clone(),
            evaluate_group(
                rows,
                &item_fractions,
                &minute_budgets,
                &coverage_targets,
                args.bootstrap_replicates,
                &format!("{input_hash}:{group}"),
            ),
        )
    }).collect();
    let item_budget_results = item_fractions.iter().map(|fraction| {
        let count = (*fraction * records.len() as f64).ceil().max(1.0);
        review_yield(&refs, count, Some(*fraction), "items")
    }).collect();
    let review_time_budget_results = minute_budgets.iter().map(|minutes| {
        review_yield(&refs, *minutes, None, "minutes")
    }).collect();
    let risk_coverage_results = risk_coverage(&refs, &coverage_targets);

    let report = Report {
        schema_version: 5,
        scorer: "proofreading-score",
        scorer_version: env!("CARGO_PKG_VERSION"),
        dataset_id: DATASET_ID,
        dataset_revision: DATASET_REVISION,
        dataset_revision_commit_sha: args.dataset_revision_commit_sha,
        pinned_upstream_commit: UPSTREAM_COMMIT,
        mapping_reference: args.mapping_reference,
        policy_id: args.policy_id,
        prediction_profile_sha256: args.prediction_profile_sha256,
        evaluation_partition: args.evaluation_partition,
        review_time_basis: args.review_time_basis,
        bootstrap_replicates: args.bootstrap_replicates,
        input_sha256: input_hash.clone(),
        n_candidates: overall.n,
        positive_errors: overall.positive_errors,
        negative_controls: overall.negative_controls,
        status: "SCORED_NOT_PROMOTED",
        overall,
        overall_cluster_bootstrap_95,
        per_group,
        item_budget_results,
        review_time_budget_results,
        risk_coverage_results,
        baseline_comparisons: baseline_policy_results(
            &refs,
            &item_fractions,
            &minute_budgets,
            args.bootstrap_replicates,
            &format!("{input_hash}:overall"),
        ),
        domain_shift_audit_status,
        domain_shift_axis,
        overall_calibration,
        overall_calibration_cluster_bootstrap_95,
        per_domain,
        ood_detector_id: args.ood_detector_id.clone(),
        ood_detector_profile_sha256: args.ood_detector_profile_sha256.clone(),
        ood_detection,
        scope_note: "Metrics describe this frozen prediction file only. Cluster-bootstrap intervals resample the caller-declared cluster_id; they do not prove that the chosen cluster unit is independent. This scorer does not validate dataset labels/mapping truth or establish prospective review-time savings. Use a locked evaluation protocol before claims.",
    };
    let json = serde_json::to_string_pretty(&report)
        .map_err(|e| format!("failed to serialize report: {e}"))? + "\n";
    if let Some(output) = args.output {
        fs::write(&output, json.as_bytes())
            .map_err(|e| format!("failed to write output {}: {e}", output.display()))?;
    } else {
        print!("{json}");
    }
    Ok(0)
}

fn main() -> ExitCode {
    match run() {
        Ok(0) => ExitCode::SUCCESS,
        Ok(_) => ExitCode::from(1),
        Err(message) => {
            eprintln!("proofreading-score: {message}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: &str, label: bool, p: f64, priority: f64, mins: f64, impact: f64, group: &str) -> Prediction {
        Prediction {
            sample_id: id.into(),
            group: group.into(),
            cluster_id: format!("cluster-{id}"),
            label_is_error: label,
            p_error: p,
            review_priority_score: priority,
            review_minutes: mins,
            impact_weight: impact,
            domain_id: None,
            shift_axis: None,
            is_ood_domain: None,
            ood_score: None,
        }
    }

    fn tagged_row(
        id: &str,
        label: bool,
        p: f64,
        domain: &str,
        is_ood: bool,
        novelty: f64,
    ) -> Prediction {
        let mut prediction = row(id, label, p, p, 1.0, 1.0, "merge");
        prediction.domain_id = Some(domain.into());
        prediction.shift_axis = Some("species".into());
        prediction.is_ood_domain = Some(is_ood);
        prediction.ood_score = Some(novelty);
        prediction
    }

    #[test]
    fn calibration_cluster_bootstrap_is_deterministic_and_uses_clusters() {
        let mut rows = Vec::new();
        for cluster in 0..6 {
            for within in 0..2 {
                let label = within == 0;
                let mut item = row(
                    &format!("cb-{cluster}-{within}"),
                    label,
                    if label { 0.8 } else { 0.2 },
                    0.5,
                    1.0,
                    1.0,
                    "merge",
                );
                item.cluster_id = format!("psu-{cluster}");
                rows.push(item);
            }
        }
        let refs = rows.iter().collect::<Vec<_>>();
        let first = calibration_cluster_bootstrap(&refs, 200, "calibration-seed");
        let second = calibration_cluster_bootstrap(&refs, 200, "calibration-seed");
        assert_eq!(first.valid_replicates, 200);
        assert_eq!(
            first.expected_calibration_error_95.map(|x| (x.lower, x.upper)),
            second.expected_calibration_error_95.map(|x| (x.lower, x.upper))
        );
        assert_eq!(first.brier_score_95, Some(Interval { lower: 0.04, upper: 0.04 }));
        assert_eq!(first.adaptive_confidence_ece_15_95, Some(Interval { lower: 0.2, upper: 0.2 }));
        assert_eq!(first.reliability_curve_ece_12_95, Some(Interval { lower: 0.2, upper: 0.2 }));
    }

    #[test]
    fn calibration_bootstrap_reports_cluster_intervals_for_each_reliability_bin() {
        let mut rows = Vec::new();
        for cluster in 0..6 {
            for within in 0..2 {
                let label = within == 0;
                let mut item = row(
                    &format!("rel-{cluster}-{within}"),
                    label,
                    if label { 0.8 } else { 0.2 },
                    0.5,
                    1.0,
                    1.0,
                    "merge",
                );
                item.cluster_id = format!("reliability-psu-{cluster}");
                rows.push(item);
            }
        }
        let refs = rows.iter().collect::<Vec<_>>();
        let result = calibration_cluster_bootstrap(&refs, 200, "reliability-per-bin");
        assert_eq!(result.reliability_curve_observed_accuracy_95.len(), 1);
        let bin = &result.reliability_curve_observed_accuracy_95[0];
        assert_eq!(bin.full_sample_count, 12);
        assert_eq!(bin.valid_replicates, 200);
        assert_eq!(bin.observed_accuracy_95, Some(Interval { lower: 1.0, upper: 1.0 }));
    }

    #[test]
    fn calibration_cluster_bootstrap_refuses_one_cluster() {
        let mut a = row("one-a", true, 0.9, 0.9, 1.0, 1.0, "merge");
        let mut b = row("one-b", false, 0.1, 0.1, 1.0, 1.0, "merge");
        a.cluster_id = "one-psu".into();
        b.cluster_id = "one-psu".into();
        let interval = calibration_cluster_bootstrap(&[&a, &b], 200, "one-cluster");
        assert_eq!(interval.valid_replicates, 0);
        assert_eq!(interval.expected_calibration_error_95, None);
        assert_eq!(interval.brier_score_95, None);
    }

    #[test]
    fn ood_cluster_bootstrap_refuses_single_class_cluster_resamples() {
        let mut in_domain = row("in", false, 0.1, 0.1, 1.0, 1.0, "merge");
        let mut ood = row("out", true, 0.9, 0.9, 1.0, 1.0, "merge");
        in_domain.cluster_id = "one-in-cluster".into();
        ood.cluster_id = "one-out-cluster".into();
        in_domain.domain_id = Some("mouse".into());
        in_domain.shift_axis = Some("species".into());
        in_domain.is_ood_domain = Some(false);
        in_domain.ood_score = Some(0.1);
        ood.domain_id = Some("zebrafish".into());
        ood.shift_axis = Some("species".into());
        ood.is_ood_domain = Some(true);
        ood.ood_score = Some(0.9);
        let interval = ood_cluster_bootstrap(&[&in_domain, &ood], 500, "sparse-ood");
        assert!(interval.valid_replicates < 400);
        assert_eq!(interval.roc_auc_95, None);
        assert_eq!(interval.average_precision_95, None);
    }

    #[test]
    fn confidence_reliability_bins_never_split_equal_confidence_ties() {
        let mut rows = Vec::new();
        for i in 0..6 {
            let label = i % 2 == 0;
            let probability = if label { 0.9 } else { 0.1 };
            rows.push(row(&format!("tie-{i}"), label, probability, 0.5, 1.0, 1.0, "merge"));
        }
        let refs = rows.iter().collect::<Vec<_>>();
        let bins = adaptive_confidence_bins(&refs, 12);
        assert_eq!(bins.len(), 1);
        assert_eq!(bins[0].count, 6);
        assert_eq!(bins[0].mean_confidence, 0.9);
        assert_eq!(bins[0].observed_accuracy, 1.0);
        assert!((bins[0].absolute_calibration_gap - 0.1).abs() < 1e-12);
    }

    #[test]
    fn confidence_reliability_bins_are_deterministic_and_cover_every_row() {
        let rows = vec![
            row("q1", true, 0.99, 0.5, 1.0, 1.0, "merge"),
            row("q2", false, 0.95, 0.5, 1.0, 1.0, "merge"),
            row("q3", false, 0.80, 0.5, 1.0, 1.0, "merge"),
            row("q4", true, 0.20, 0.5, 1.0, 1.0, "merge"),
            row("q5", true, 0.05, 0.5, 1.0, 1.0, "merge"),
            row("q6", false, 0.01, 0.5, 1.0, 1.0, "merge"),
        ];
        let refs = rows.iter().collect::<Vec<_>>();
        let first = adaptive_confidence_bins(&refs, 15);
        let second = adaptive_confidence_bins(&refs, 15);
        assert_eq!(first.len(), second.len());
        assert_eq!(first.iter().map(|bin| bin.count).sum::<usize>(), rows.len());
        assert_eq!(
            first.iter().map(|bin| bin.mean_confidence).collect::<Vec<_>>(),
            second.iter().map(|bin| bin.mean_confidence).collect::<Vec<_>>()
        );
    }

    #[test]
    fn calibration_bins_expose_counts_and_expected_error() {
        let rows = vec![
            row("c1", true, 0.95, 0.5, 1.0, 1.0, "merge"),
            row("c2", false, 0.05, 0.5, 1.0, 1.0, "merge"),
            row("c3", true, 0.85, 0.5, 1.0, 1.0, "merge"),
            row("c4", false, 0.15, 0.5, 1.0, 1.0, "merge"),
        ];
        let refs = rows.iter().collect::<Vec<_>>();
        let diagnostics = calibration_diagnostics(&refs);
        assert_eq!(diagnostics.fixed_bin_count, 10);
        assert_eq!(diagnostics.adaptive_confidence_bin_count_requested, 15);
        assert_eq!(diagnostics.reliability_curve_bin_count_requested, 12);
        assert_eq!(diagnostics.adaptive_confidence_bins.iter().map(|bin| bin.count).sum::<usize>(), 4);
        assert_eq!(diagnostics.reliability_curve_bins.iter().map(|bin| bin.count).sum::<usize>(), 4);
        assert!(diagnostics.reliability_curve_expected_calibration_error.is_some());
        assert_eq!(diagnostics.bins.iter().map(|bin| bin.count).sum::<usize>(), 4);
        assert!((diagnostics.expected_calibration_error.unwrap() - 0.10).abs() < 1e-12);
        assert_eq!(diagnostics.high_confidence_at_90.false_accept_count, 0);
    }

    #[test]
    fn ood_detection_scores_are_separate_from_error_probabilities() {
        let rows = vec![
            tagged_row("d1", true, 0.9, "mouse", false, 0.1),
            tagged_row("d2", false, 0.1, "mouse", false, 0.2),
            tagged_row("d3", true, 0.1, "zebrafish", true, 0.9),
            tagged_row("d4", false, 0.9, "zebrafish", true, 0.8),
        ];
        let refs = rows.iter().collect::<Vec<_>>();
        let metrics = ood_detection_metrics(&refs, Some(0.5), 500, "test:ood-bootstrap");
        assert_eq!(metrics.roc_auc, Some(1.0));
        assert_eq!(metrics.average_precision, Some(1.0));
        assert_eq!(metrics.false_ood_count, Some(0));
        assert_eq!(metrics.missed_ood_count, Some(0));
        let zebrafish = refs.iter().filter(|row| row.domain_id.as_deref() == Some("zebrafish")).copied().collect::<Vec<_>>();
        let calibration = calibration_diagnostics(&zebrafish);
        assert_eq!(calibration.high_confidence_at_90.false_accept_count, 1);
        assert_eq!(calibration.low_confidence_below_10.missed_error_count, 0);
    }

    #[test]
    fn novelty_scores_require_detector_provenance() {
        let rows = vec![tagged_row("prov-1", true, 0.9, "mouse", false, 0.1)];
        assert!(validate_ood_detector_provenance(&rows, None, None).unwrap_err().contains("requires --ood-detector-id"));
        assert!(validate_ood_detector_provenance(&rows, Some("mahalanobis"), None).is_err());
        assert!(validate_ood_detector_provenance(&rows, Some("mahalanobis"), Some(&"a".repeat(64))).is_ok());
    }

    #[test]
    fn detector_provenance_without_novelty_scores_is_rejected() {
        let rows = vec![row("prov-2", true, 0.9, 0.9, 1.0, 1.0, "merge")];
        assert!(validate_ood_detector_provenance(&rows, Some("knn-distance"), Some(&"b".repeat(64))).is_err());
    }

    #[test]
    fn invalid_detector_profile_digest_is_rejected() {
        let rows = vec![tagged_row("prov-3", true, 0.9, "mouse", false, 0.1)];
        assert!(validate_ood_detector_provenance(&rows, Some("ensemble-disagreement"), Some("not-a-sha")).is_err());
    }

    #[test]
    fn ood_metric_status_is_not_estimable_without_both_domain_classes() {
        let rows = vec![
            tagged_row("only-in-1", true, 0.8, "mouse", false, 0.1),
            tagged_row("only-in-2", false, 0.2, "mouse", false, 0.2),
        ];
        let refs = rows.iter().collect::<Vec<_>>();
        let metrics = ood_detection_metrics(&refs, Some(0.5), 500, "single-domain-ood");
        assert_eq!(metrics.status, "NOT_ESTIMABLE");
        assert_eq!(metrics.roc_auc, None);
        assert_eq!(metrics.held_out_domain_recall, None);
    }

    #[test]
    fn parser_rejects_dependency_cluster_reused_across_domains() {
        let first = tagged_row("cross-1", true, 0.9, "mouse", false, 0.1);
        let mut second = tagged_row("cross-2", false, 0.1, "zebrafish", true, 0.9);
        second.cluster_id = first.cluster_id.clone();
        let input = format!("{}\n{}", serde_json::to_string(&first).unwrap(), serde_json::to_string(&second).unwrap());
        assert!(parse_jsonl(input.as_bytes()).unwrap_err().contains("crosses domain_id"));
    }

    #[test]
    fn equal_novelty_scores_produce_half_auc() {
        let metrics = binary_ranking_metrics(&[(false, 0.5), (false, 0.5), (true, 0.5), (true, 0.5)]);
        assert_eq!(metrics.0, Some(0.5));
    }

    #[test]
    fn calibration_threshold_reports_zero_denominator_as_none() {
        let risk = calibration_threshold_risk(&[], 0.95);
        assert_eq!(risk.selected_count, 0);
        assert_eq!(risk.false_accept_fraction, None);
        assert_eq!(risk.precision, None);
    }

    #[test]
    fn domain_audit_groups_domains_and_keeps_axis_explicit() {
        let rows = vec![
            tagged_row("e1", true, 0.9, "mouse", false, 0.1),
            tagged_row("e2", false, 0.1, "mouse", false, 0.2),
            tagged_row("e3", true, 0.1, "zebrafish", true, 0.9),
            tagged_row("e4", false, 0.9, "zebrafish", true, 0.8),
        ];
        let (status, axis, overall, overall_bootstrap, domains, ood) =
            domain_shift_evaluation(&rows, true, Some(0.5), 500, "test:domain-bootstrap").unwrap();
        assert_eq!(status, "OOD_SCORED");
        assert_eq!(axis.as_deref(), Some("species"));
        assert!(overall.is_some());
        assert!(overall_bootstrap.is_some());
        assert_eq!(domains.unwrap().len(), 2);
        assert_eq!(ood.unwrap().roc_auc, Some(1.0));
    }

    #[test]
    fn domain_audit_refuses_missing_metadata_when_required() {
        let rows = vec![row("n1", true, 0.9, 0.9, 1.0, 1.0, "merge")];
        assert!(domain_shift_evaluation(&rows, true, None, 100, "test:missing-domain").is_err());
    }

    #[test]
    fn domain_audit_refuses_threshold_without_novelty_scores() {
        let mut item = tagged_row("n2", true, 0.9, "mouse", false, 0.2);
        item.ood_score = None;
        assert!(domain_shift_evaluation(&[item], false, Some(0.5), 100, "test:missing-ood").is_err());
    }

    #[test]
    fn parser_rejects_partial_domain_metadata() {
        let line = r#"{"sample_id":"x","group":"mouse:merge","cluster_id":"c1","label_is_error":true,"p_error":0.9,"review_priority_score":0.9,"review_minutes":1.0,"impact_weight":1.0,"domain_id":"mouse"}"#;
        assert!(parse_jsonl(line.as_bytes()).unwrap_err().contains("provide domain_id"));
    }

    #[test]
    fn parser_rejects_multiple_shift_axes_in_one_run() {
        let first = tagged_row("a1", true, 0.9, "mouse", false, 0.1);
        let mut second = tagged_row("a2", false, 0.1, "fly", false, 0.1);
        second.shift_axis = Some("region".into());
        let input = format!("{}\n{}", serde_json::to_string(&first).unwrap(), serde_json::to_string(&second).unwrap());
        assert!(parse_jsonl(input.as_bytes()).unwrap_err().contains("only one shift_axis"));
    }

    #[test]
    fn parser_rejects_inconsistent_domain_holdout_labels() {
        let first = tagged_row("b1", true, 0.9, "mouse", false, 0.1);
        let second = tagged_row("b2", false, 0.1, "mouse", true, 0.9);
        let input = format!("{}\n{}", serde_json::to_string(&first).unwrap(), serde_json::to_string(&second).unwrap());
        assert!(parse_jsonl(input.as_bytes()).unwrap_err().contains("inconsistent"));
    }

    #[test]
    fn parser_rejects_partial_novelty_scores() {
        let first = tagged_row("p1", true, 0.9, "mouse", false, 0.1);
        let mut second = tagged_row("p2", false, 0.1, "mouse", false, 0.1);
        second.ood_score = None;
        let input = format!("{}\n{}", serde_json::to_string(&first).unwrap(), serde_json::to_string(&second).unwrap());
        assert!(parse_jsonl(input.as_bytes()).unwrap_err().contains("present for every row"));
    }

    #[test]
    fn parser_rejects_out_of_range_novelty_score() {
        let mut item = tagged_row("r1", true, 0.9, "mouse", false, 0.1);
        item.ood_score = Some(1.1);
        let input = serde_json::to_string(&item).unwrap();
        assert!(parse_jsonl(input.as_bytes()).unwrap_err().contains("ood_score must be finite"));
    }

    #[test]
    fn subgroup_operating_metrics_use_only_rows_in_that_subgroup() {
        let high_confidence_error = row("a", true, 0.99, 0.99, 1.0, 5.0, "mouse:merge");
        let lower_confidence_control = row("b", false, 0.20, 0.10, 1.0, 0.0, "mouse:merge");
        let unrelated_confident_false_alarm = row("c", false, 0.001, 1.0, 1.0, 0.0, "human:merge");
        let local_rows = [&high_confidence_error, &lower_confidence_control];
        let _pooled_rows = [&high_confidence_error, &lower_confidence_control, &unrelated_confident_false_alarm];

        let local = evaluate_group(
            &local_rows,
            &[0.5],
            &[1.0],
            &[0.5],
            100,
            "test:mouse:merge",
        );
        assert_eq!(local.item_budget_results[0].selected_count, 1);
        assert_eq!(local.item_budget_results[0].true_errors_found, 1);
        assert_eq!(local.review_time_budget_results[0].selected_count, 1);
        assert_eq!(local.risk_coverage_results[0].accepted_count, 1);
        assert_eq!(local.risk_coverage_results[0].accepted_case_risk, Some(0.0));
        assert_eq!(local.risk_coverage_results[0].achieved_coverage, 0.5);
        assert_eq!(local.classification.n, 2);
    }

    #[test]
    fn uncertainty_baseline_prefers_ambiguous_predictions() {
        let confident = row("confident", true, 0.99, 0.1, 1.0, 1.0, "mouse:merge");
        let ambiguous = row("ambiguous", false, 0.51, 0.9, 1.0, 0.0, "mouse:merge");
        let ordered = ranked_for_policy(
            &[&confident, &ambiguous],
            RankingPolicy::UncertaintyOnly,
            "test",
        );
        assert_eq!(ordered[0].sample_id, "ambiguous");
    }

    #[test]
    fn deterministic_random_baseline_replays_for_same_seed() {
        let a = row("a", true, 0.9, 0.9, 1.0, 1.0, "mouse:merge");
        let b = row("b", false, 0.1, 0.1, 1.0, 0.0, "mouse:merge");
        let c = row("c", true, 0.8, 0.8, 1.0, 1.0, "mouse:merge");
        let rows = [&a, &b, &c];
        let first: Vec<String> = ranked_for_policy(&rows, RankingPolicy::DeterministicRandom, "run-a")
            .iter().map(|row| row.sample_id.clone()).collect();
        let second: Vec<String> = ranked_for_policy(&rows, RankingPolicy::DeterministicRandom, "run-a")
            .iter().map(|row| row.sample_id.clone()).collect();
        assert_eq!(first, second);
        let first_yield = review_yield_for_policy(
            &rows, 1.0, Some(1.0 / 3.0), "items", RankingPolicy::DeterministicRandom, "run-a",
        );
        assert_eq!(first_yield.selection_policy, "deterministic_random");
    }

    #[test]
    fn paired_bootstrap_reports_positive_policy_delta_against_uncertainty() {
        let mut rows = Vec::new();
        // Every cluster contains both classes, so a valid cluster resample cannot erase one class.
        for i in 0..4 {
            let mut error = row(&format!("err-{i}"), true, 0.9, 0.9, 1.0, 2.0, "mouse:merge");
            error.cluster_id = format!("cluster-{i}");
            let mut control = row(&format!("ok-{i}"), false, 0.49, 0.1, 1.0, 0.0, "mouse:merge");
            control.cluster_id = format!("cluster-{i}");
            rows.push(error);
            rows.push(control);
        }
        let refs: Vec<&Prediction> = rows.iter().collect();
        let delta = paired_delta_bootstrap(
            &refs,
            RankingPolicy::UncertaintyOnly,
            "items",
            1.0,
            Some(0.1),
            500,
            "unit-test:paired-delta",
        );
        assert!(delta.error_recall_delta.valid_replicates >= 400);
        let interval = delta.error_recall_delta.interval_95.expect("supported interval");
        assert!(interval.lower > 0.0, "expected submitted ranker to beat uncertainty: {interval:?}");
        assert_eq!(delta.precision_delta.interval_95, Some(Interval { lower: 1.0, upper: 1.0 }));
    }

    #[test]
    fn paired_bootstrap_marks_one_cluster_not_estimable() {
        let mut a = row("a", true, 0.9, 0.9, 1.0, 1.0, "mouse:merge");
        let mut b = row("b", false, 0.1, 0.1, 1.0, 0.0, "mouse:merge");
        a.cluster_id = "one".into();
        b.cluster_id = "one".into();
        let delta = paired_delta_bootstrap(
            &[&a, &b],
            RankingPolicy::UncertaintyOnly,
            "items",
            1.0,
            Some(0.5),
            500,
            "unit-test:one-cluster",
        );
        assert_eq!(delta.error_recall_delta.valid_replicates, 0);
        assert_eq!(delta.error_recall_delta.interval_95, None);
    }

    #[test]
    fn perfect_probabilities_have_perfect_auc_and_ap() {
        let a = row("a", true, 0.9, 0.9, 1.0, 2.0, "mouse:merge");
        let b = row("b", false, 0.1, 0.1, 1.0, 0.0, "mouse:merge");
        let metrics = classification(&[&a, &b]);
        assert_eq!(metrics.roc_auc, Some(1.0));
        assert_eq!(metrics.average_precision, Some(1.0));
        assert_eq!(metrics.accuracy, Some(1.0));
    }

    #[test]
    fn tied_scores_are_scored_without_id_order_advantage() {
        let a = row("a", true, 0.5, 0.5, 1.0, 1.0, "fly:merge");
        let b = row("b", false, 0.5, 0.5, 1.0, 0.0, "fly:merge");
        let metrics = classification(&[&a, &b]);
        assert_eq!(metrics.roc_auc, Some(0.5));
        assert_eq!(metrics.average_precision, Some(0.5));
    }

    #[test]
    fn no_positive_or_negative_class_makes_auc_not_estimable() {
        let a = row("a", true, 0.9, 0.9, 1.0, 1.0, "human:merge");
        let b = row("b", true, 0.8, 0.8, 1.0, 1.0, "human:merge");
        let metrics = classification(&[&a, &b]);
        assert_eq!(metrics.roc_auc, None);
        assert_eq!(metrics.average_precision, Some(1.0));
        assert_eq!(metrics.balanced_accuracy, None);
    }

    #[test]
    fn invalid_probability_fails_closed() {
        let line = r#"{"sample_id":"a","group":"mouse:merge","cluster_id":"cube-a","label_is_error":true,"p_error":1.1,"review_priority_score":0.8,"review_minutes":1.0,"impact_weight":1.0}"#;
        assert!(parse_jsonl(line.as_bytes()).unwrap_err().contains("p_error"));
    }

    #[test]
    fn duplicate_sample_ids_fail_closed() {
        let line = r#"{"sample_id":"a","group":"mouse:merge","cluster_id":"cube-a","label_is_error":true,"p_error":0.9,"review_priority_score":0.8,"review_minutes":1.0,"impact_weight":1.0}"#;
        let input = format!("{line}\n{line}\n");
        assert!(parse_jsonl(input.as_bytes()).unwrap_err().contains("duplicate sample_id"));
    }

    #[test]
    fn review_priority_finds_true_errors_within_budget() {
        let a = row("a", true, 0.6, 0.9, 2.0, 3.0, "mouse:merge");
        let b = row("b", false, 0.1, 0.7, 1.0, 0.0, "mouse:merge");
        let c = row("c", true, 0.8, 0.3, 1.0, 2.0, "mouse:merge");
        let out = review_yield(&[&a, &b, &c], 3.0, None, "minutes");
        assert_eq!(out.selected_count, 2);
        assert_eq!(out.true_errors_found, 1);
        assert_eq!(out.actual_review_minutes, 3.0);
    }

    #[test]
    fn review_budget_skips_expensive_items_and_keeps_fitting_candidates() {
        let a = row("a", true, 0.9, 0.9, 5.0, 4.0, "mouse:merge");
        let b = row("b", true, 0.8, 0.8, 2.0, 2.0, "mouse:merge");
        let c = row("c", false, 0.1, 0.7, 1.0, 0.0, "mouse:merge");
        let out = review_yield(&[&a, &b, &c], 3.0, None, "minutes");
        assert_eq!(out.selected_count, 2);
        assert_eq!(out.true_errors_found, 1);
        assert_eq!(out.actual_review_minutes, 3.0);
    }

    #[test]
    fn confidence_coverage_abstains_on_low_confidence_cases() {
        let a = row("a", true, 0.99, 0.9, 1.0, 1.0, "zebrafish:merge");
        let b = row("b", false, 0.51, 0.8, 1.0, 0.0, "zebrafish:merge");
        let c = row("c", true, 0.52, 0.7, 1.0, 1.0, "zebrafish:merge");
        let result = risk_coverage(&[&a, &b, &c], &[0.33, 0.66]);
        assert_eq!(result[0].accepted_count, 1);
        assert_eq!(result[0].accepted_case_risk, Some(0.0));
        assert!(result[1].achieved_coverage >= 0.66);
    }

    #[test]
    fn cluster_bootstrap_is_deterministic_for_frozen_input_and_seed() {
        let mut a = row("a", true, 0.9, 0.9, 1.0, 2.0, "mouse:merge");
        let mut b = row("b", false, 0.1, 0.1, 1.0, 0.0, "mouse:merge");
        let mut c = row("c", true, 0.8, 0.8, 1.0, 2.0, "mouse:merge");
        let mut d = row("d", false, 0.2, 0.2, 1.0, 0.0, "mouse:merge");
        a.cluster_id = "cube-1".into();
        b.cluster_id = "cube-1".into();
        c.cluster_id = "cube-2".into();
        d.cluster_id = "cube-2".into();
        let rows = [&a, &b, &c, &d];
        let first = cluster_bootstrap(&rows, 100, "frozen-input:overall");
        let second = cluster_bootstrap(&rows, 100, "frozen-input:overall");
        assert_eq!(first.valid_replicates, 100);
        assert_eq!(
            first.balanced_accuracy_95.map(|i| (i.lower, i.upper)),
            second.balanced_accuracy_95.map(|i| (i.lower, i.upper))
        );
    }

    #[test]
    fn cluster_bootstrap_marks_single_cluster_interval_not_estimable() {
        let mut a = row("a", true, 0.9, 0.9, 1.0, 2.0, "human:merge");
        let mut b = row("b", false, 0.1, 0.1, 1.0, 0.0, "human:merge");
        a.cluster_id = "same-cube".into();
        b.cluster_id = "same-cube".into();
        let result = cluster_bootstrap(&[&a, &b], 100, "seed");
        assert_eq!(result.valid_replicates, 0);
        assert_eq!(result.balanced_accuracy_95, None);
    }

    #[test]
    fn cluster_bootstrap_resamples_clusters_not_individual_rows() {
        let mut a = row("a", true, 0.9, 0.9, 1.0, 2.0, "fly:merge");
        let mut b = row("b", false, 0.1, 0.1, 1.0, 0.0, "fly:merge");
        let mut c = row("c", true, 0.8, 0.8, 1.0, 2.0, "fly:merge");
        let mut d = row("d", false, 0.2, 0.2, 1.0, 0.0, "fly:merge");
        a.cluster_id = "cube-1".into();
        b.cluster_id = "cube-1".into();
        c.cluster_id = "cube-2".into();
        d.cluster_id = "cube-2".into();
        let result = cluster_bootstrap(&[&a, &b, &c, &d], 100, "cluster-unit-test");
        assert_eq!(result.valid_replicates, 100);
        assert_eq!(result.balanced_accuracy_95, Some(Interval { lower: 1.0, upper: 1.0 }));
    }

    #[test]
    fn group_breakdown_is_order_stable_by_btreemap_contract() {
        let a = row("a", true, 0.9, 0.9, 1.0, 2.0, "mouse:merge");
        let b = row("b", false, 0.1, 0.1, 1.0, 0.0, "fly:split");
        let groups = BTreeMap::from([
            ("mouse:merge".to_string(), classification(&[&a])),
            ("fly:split".to_string(), classification(&[&b])),
        ]);
        let keys: Vec<_> = groups.keys().map(String::as_str).collect();
        assert_eq!(keys, vec!["fly:split", "mouse:merge"]);
    }
}
