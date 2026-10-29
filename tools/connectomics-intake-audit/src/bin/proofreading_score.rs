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

    /// Write report to this JSON file (stdout if omitted).
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize)]
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
    scope_note: &'static str,
}

#[derive(Debug, Serialize)]
struct GroupEvaluation {
    classification: ClassificationMetrics,
    balanced_accuracy_cluster_bootstrap_95: ClusterBootstrap,
}

#[derive(Debug, Serialize)]
struct ClusterBootstrap {
    requested_replicates: usize,
    valid_replicates: usize,
    minimum_valid_replicates_for_interval: usize,
    balanced_accuracy_95: Option<Interval>,
}

#[derive(Debug, Clone, Copy, Serialize)]
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
        records.push(row);
    }
    if records.is_empty() {
        return Err("input contains no prediction records".into());
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

fn ratio(num: usize, den: usize) -> Option<f64> {
    if den == 0 { None } else { Some(num as f64 / den as f64) }
}

fn ranked<'a>(rows: &[&'a Prediction]) -> Vec<&'a Prediction> {
    let mut result = rows.to_vec();
    result.sort_by(|a, b| {
        b.review_priority_score
            .total_cmp(&a.review_priority_score)
            .then_with(|| a.sample_id.cmp(&b.sample_id))
    });
    result
}

fn review_yield(
    rows: &[&Prediction],
    requested_budget: f64,
    requested_item_fraction: Option<f64>,
    budget_kind: &'static str,
) -> ReviewYield {
    let ordered = ranked(rows);
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
        selection_policy: "descending review_priority_score; lexical sample_id breaks ties; time budgets skip items that do not fit",
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

fn run() -> Result<i32, String> {
    let args = Args::parse();
    if args.mapping_reference.trim().is_empty() {
        return Err("--mapping-reference must identify the reviewed label normalization contract".into());
    }
    if args.bootstrap_replicates < 100 {
        return Err("--bootstrap-replicates must be at least 100".into());
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
            GroupEvaluation {
                classification: classification(rows),
                balanced_accuracy_cluster_bootstrap_95: cluster_bootstrap(
                    rows,
                    args.bootstrap_replicates,
                    &format!("{input_hash}:{group}"),
                ),
            },
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
        schema_version: 1,
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
        input_sha256: input_hash,
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
        }
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
        let mut b = row("b", true, 0.9, 0.9, 1.0, 2.0, "fly:merge");
        let mut c = row("c", false, 0.1, 0.1, 1.0, 0.0, "fly:merge");
        let mut d = row("d", false, 0.1, 0.1, 1.0, 0.0, "fly:merge");
        a.cluster_id = "positive-cube".into();
        b.cluster_id = "positive-cube".into();
        c.cluster_id = "negative-cube".into();
        d.cluster_id = "negative-cube".into();
        let result = cluster_bootstrap(&[&a, &b, &c, &d], 100, "cluster-unit-test");
        assert!(result.valid_replicates >= result.minimum_valid_replicates_for_interval);
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
