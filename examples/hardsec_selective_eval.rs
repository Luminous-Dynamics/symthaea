//! HARDSEC-009 validation-frozen selective-risk evaluator.
//!
//! Evaluates a detector score ledger using three possible decisions:
//! clear, suspect, or abstain/escalate. The clear/alert thresholds are selected
//! on validation only under caller-specified false-alert and false-clear caps.
//! The frozen thresholds are then evaluated once on group-held-out test data.
//!
//! The score is treated as a probability estimate for calibration diagnostics,
//! not as proof that the model is calibrated or as a security verdict.
//!
//! Example:
//!   cargo run --example hardsec_selective_eval --features hardsec-selective-eval -- \
//!     --ledger predictions.json --receipt NEW_receipt.json \
//!     --max-false-alert-rate 0.005 --max-false-clear-rate 0.01
//!
//! Input schema: hardsec-score-ledger-v1. The group_id field is the declared
//! holdout unit (for example host family, Trojan family, insertion source, or
//! library). A group may occur in exactly one split. The tool does not train
//! models, authenticate score producers, or prove a design secure.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const LEDGER_SCHEMA: &str = "hardsec-score-ledger-v1";
const RECEIPT_SCHEMA: &str = "hardsec-selective-eval-receipt-v1";
const CALIBRATION_BINS: usize = 10;

#[derive(Debug, Clone)]
struct Args {
    ledger: PathBuf,
    receipt: PathBuf,
    max_false_alert_rate: f64,
    max_false_clear_rate: f64,
}

#[derive(Debug, Clone, PartialEq)]
struct Prediction {
    sample_id: String,
    group_id: String,
    split: String,
    label: u8,
    score: f64,
}

#[derive(Debug, Clone)]
struct Ledger {
    dataset_id: String,
    grouping_key: String,
    model_sha256: String,
    config_sha256: String,
    predictions: Vec<Prediction>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct OperatingPoint {
    clear_max_score: f64,
    alert_min_score: f64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Counts {
    total: u64,
    positives: u64,
    negatives: u64,
    cleared: u64,
    alerted: u64,
    abstained: u64,
    true_alerts: u64,
    false_alerts: u64,
    true_clears: u64,
    false_clears: u64,
}

#[derive(Debug, Clone, Copy)]
struct Rates {
    counts: Counts,
    coverage: f64,
    abstention_rate: f64,
    false_alert_rate: Option<f64>,
    false_clear_rate: Option<f64>,
    alert_precision: Option<f64>,
    trojan_recall: Option<f64>,
    selective_risk: Option<f64>,
}

#[derive(Debug, Clone, Copy)]
struct Calibration {
    brier_score: f64,
    log_loss: f64,
    expected_calibration_error: f64,
}

#[derive(Debug, Clone)]
struct GroupSummary {
    group_id: String,
    metrics: Rates,
}

fn usage() -> &'static str {
    "Usage:
  hardsec_selective_eval --ledger FILE --receipt NEW_FILE
    --max-false-alert-rate RATE --max-false-clear-rate RATE

RATE must be in [0, 1]. The receipt path must not already exist."
}

fn parse_rate(name: &str, value: &str) -> Result<f64, String> {
    let parsed = value
        .parse::<f64>()
        .map_err(|error| format!("invalid {name} {value:?}: {error}"))?;
    if !parsed.is_finite() || !(0.0..=1.0).contains(&parsed) {
        return Err(format!("{name} must be finite and in [0, 1]"));
    }
    Ok(parsed)
}

fn parse_args() -> Result<Option<Args>, String> {
    let mut ledger = None;
    let mut receipt = None;
    let mut max_false_alert_rate = None;
    let mut max_false_clear_rate = None;
    let mut args = env::args().skip(1);

    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            return Ok(None);
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--ledger" => ledger = Some(PathBuf::from(value)),
            "--receipt" => receipt = Some(PathBuf::from(value)),
            "--max-false-alert-rate" => {
                max_false_alert_rate = Some(parse_rate("max false-alert rate", &value)?)
            }
            "--max-false-clear-rate" => {
                max_false_clear_rate = Some(parse_rate("max false-clear rate", &value)?)
            }
            _ => return Err(format!("unknown argument: {flag}")),
        }
    }

    Ok(Some(Args {
        ledger: ledger.ok_or("--ledger is required")?,
        receipt: receipt.ok_or("--receipt is required")?,
        max_false_alert_rate: max_false_alert_rate
            .ok_or("--max-false-alert-rate is required")?,
        max_false_clear_rate: max_false_clear_rate
            .ok_or("--max-false-clear-rate is required")?,
    }))
}

fn required_string(value: &Value, field: &str) -> Result<String, String> {
    let result = value
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{field} must be a string"))?;
    if result.trim().is_empty() || result.trim() != result {
        return Err(format!(
            "{field} must be non-empty and have no surrounding whitespace"
        ));
    }
    Ok(result.to_owned())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn parse_ledger(bytes: &[u8]) -> Result<Ledger, String> {
    let value: Value =
        serde_json::from_slice(bytes).map_err(|error| format!("parse score ledger JSON: {error}"))?;
    if value.get("schema_version").and_then(Value::as_str) != Some(LEDGER_SCHEMA) {
        return Err(format!("schema_version must be {LEDGER_SCHEMA}"));
    }

    let dataset_id = required_string(&value, "dataset_id")?;
    let grouping_key = required_string(&value, "grouping_key")?;
    let score_semantics = required_string(&value, "score_semantics")?;
    if score_semantics != "probability_of_trojan" {
        return Err("score_semantics must be probability_of_trojan".to_owned());
    }
    let model_sha256 = required_string(&value, "model_sha256")?.to_ascii_lowercase();
    let config_sha256 = required_string(&value, "config_sha256")?.to_ascii_lowercase();
    if !valid_sha256(&model_sha256) || !valid_sha256(&config_sha256) {
        return Err("model_sha256 and config_sha256 must be 64-character SHA-256 digests".to_owned());
    }

    let rows = value
        .get("predictions")
        .and_then(Value::as_array)
        .ok_or("predictions must be a JSON array")?;
    if rows.is_empty() {
        return Err("predictions must not be empty".to_owned());
    }

    let mut predictions = Vec::with_capacity(rows.len());
    let mut sample_ids = BTreeSet::new();
    let mut group_splits = BTreeMap::<String, String>::new();
    let mut split_counts = BTreeMap::<String, Counts>::new();

    for (index, row) in rows.iter().enumerate() {
        let context = format!("predictions[{index}]");
        let sample_id = required_string(row, "sample_id")
            .map_err(|error| format!("{context}: {error}"))?;
        let group_id = required_string(row, "group_id")
            .map_err(|error| format!("{context}: {error}"))?;
        let split = required_string(row, "split")
            .map_err(|error| format!("{context}: {error}"))?;
        if !["train", "validation", "test"].contains(&split.as_str()) {
            return Err(format!("{context}: split must be train, validation, or test"));
        }
        if !sample_ids.insert(sample_id.clone()) {
            return Err(format!("duplicate sample_id {sample_id}"));
        }

        match group_splits.get(&group_id) {
            Some(existing) if existing != &split => {
                return Err(format!(
                    "group_id {group_id} leaks across {existing} and {split} splits"
                ));
            }
            Some(_) => {}
            None => {
                group_splits.insert(group_id.clone(), split.clone());
            }
        }

        let label = row
            .get("label")
            .and_then(Value::as_u64)
            .filter(|label| *label <= 1)
            .ok_or_else(|| format!("{context}: label must be integer 0 or 1"))?
            as u8;
        let score = row
            .get("score")
            .and_then(Value::as_f64)
            .ok_or_else(|| format!("{context}: score must be numeric"))?;
        if !score.is_finite() || !(0.0..=1.0).contains(&score) {
            return Err(format!("{context}: score must be finite and in [0, 1]"));
        }

        let prediction = Prediction {
            sample_id,
            group_id,
            split: split.clone(),
            label,
            score,
        };
        let counts = split_counts.entry(split).or_default();
        counts.total += 1;
        if label == 1 {
            counts.positives += 1;
        } else {
            counts.negatives += 1;
        }
        predictions.push(prediction);
    }

    for split in ["train", "validation", "test"] {
        let counts = split_counts
            .get(split)
            .ok_or_else(|| format!("{split} split is empty"))?;
        if counts.total == 0 {
            return Err(format!("{split} split is empty"));
        }
        if counts.positives == 0 || counts.negatives == 0 {
            return Err(format!("{split} split must contain both labels"));
        }
    }

    Ok(Ledger {
        dataset_id,
        grouping_key,
        model_sha256,
        config_sha256,
        predictions,
    })
}

fn max_allowed_errors(rate: f64, count: u64) -> u64 {
    ((rate * count as f64) + 1e-12).floor() as u64
}

fn unique_sorted_scores(predictions: &[Prediction]) -> Vec<f64> {
    let mut scores = predictions.iter().map(|item| item.score).collect::<Vec<_>>();
    scores.sort_by(f64::total_cmp);
    scores.dedup_by(|left, right| *left == *right);
    scores
}

fn upper_bound(values: &[f64], target: f64) -> usize {
    let mut low = 0usize;
    let mut high = values.len();
    while low < high {
        let middle = low + (high - low) / 2;
        if values[middle] <= target {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    low
}

fn select_operating_point(
    validation: &[Prediction],
    max_false_alert_rate: f64,
    max_false_clear_rate: f64,
) -> Result<OperatingPoint, String> {
    if validation.is_empty() {
        return Err("validation split is empty".to_owned());
    }
    let total_positives = validation.iter().filter(|row| row.label == 1).count() as u64;
    let total_negatives = validation.iter().filter(|row| row.label == 0).count() as u64;
    if total_positives == 0 || total_negatives == 0 {
        return Err("validation split must contain both labels".to_owned());
    }

    let max_false_alerts = max_allowed_errors(max_false_alert_rate, total_negatives);
    let max_false_clears = max_allowed_errors(max_false_clear_rate, total_positives);
    let unique_scores = unique_sorted_scores(validation);

    // Histograms allow the full threshold search in O(n log n), without an
    // O(number_of_unique_scores^2 * n) rescan of the prediction ledger.
    let mut totals = vec![0u64; unique_scores.len()];
    let mut positives = vec![0u64; unique_scores.len()];
    let mut negatives = vec![0u64; unique_scores.len()];
    for row in validation {
        let index = unique_scores
            .binary_search_by(|value| value.total_cmp(&row.score))
            .expect("score came from unique_scores");
        totals[index] += 1;
        if row.label == 1 {
            positives[index] += 1;
        } else {
            negatives[index] += 1;
        }
    }

    let mut prefix_total = vec![0u64; unique_scores.len() + 1];
    let mut prefix_positive = vec![0u64; unique_scores.len() + 1];
    for index in 0..unique_scores.len() {
        prefix_total[index + 1] = prefix_total[index] + totals[index];
        prefix_positive[index + 1] = prefix_positive[index] + positives[index];
    }

    let mut suffix_total = vec![0u64; unique_scores.len() + 1];
    let mut suffix_negative = vec![0u64; unique_scores.len() + 1];
    for index in (0..unique_scores.len()).rev() {
        suffix_total[index] = suffix_total[index + 1] + totals[index];
        suffix_negative[index] = suffix_negative[index + 1] + negatives[index];
    }

    // The all-abstain point is always feasible and is the explicit fallback.
    let mut best = OperatingPoint {
        clear_max_score: -1.0,
        alert_min_score: 2.0,
    };
    let mut best_classified = 0u64;
    let mut best_errors = 0u64;
    let mut best_false_alerts = 0u64;
    let mut best_false_clears = 0u64;

    // The lowest alert threshold satisfying the false-alert cap gives the
    // greatest possible alert coverage. Increasing it can only remove alerts.
    let minimum_alert_index = (0..=unique_scores.len())
        .find(|index| suffix_negative[*index] <= max_false_alerts)
        .ok_or("internal error: all-abstain alert bound missing")?;

    // Candidate clear thresholds are -1 (clear none) and each observed score.
    for clear_index in 0..=unique_scores.len() {
        let clear_threshold = if clear_index == 0 {
            -1.0
        } else {
            unique_scores[clear_index - 1]
        };
        let clear_total = prefix_total[clear_index];
        let false_clears = prefix_positive[clear_index];
        if false_clears > max_false_clears {
            continue;
        }

        // Require a strict gap. A score tied to both boundaries abstains,
        // rather than being simultaneously cleared and alerted.
        let first_separated_alert = upper_bound(&unique_scores, clear_threshold);
        let alert_index = minimum_alert_index.max(first_separated_alert);
        let alert_threshold = if alert_index == unique_scores.len() {
            2.0
        } else {
            unique_scores[alert_index]
        };
        if clear_threshold >= alert_threshold {
            continue;
        }

        let alert_total = suffix_total[alert_index];
        let false_alerts = suffix_negative[alert_index];
        let classified = clear_total + alert_total;
        let errors = false_clears + false_alerts;

        let better = classified > best_classified
            || (classified == best_classified && errors < best_errors)
            || (classified == best_classified
                && errors == best_errors
                && false_alerts < best_false_alerts)
            || (classified == best_classified
                && errors == best_errors
                && false_alerts == best_false_alerts
                && false_clears < best_false_clears)
            || (classified == best_classified
                && errors == best_errors
                && false_alerts == best_false_alerts
                && false_clears == best_false_clears
                && (clear_threshold < best.clear_max_score
                    || (clear_threshold == best.clear_max_score
                        && alert_threshold > best.alert_min_score)));

        if better {
            best = OperatingPoint {
                clear_max_score: clear_threshold,
                alert_min_score: alert_threshold,
            };
            best_classified = classified;
            best_errors = errors;
            best_false_alerts = false_alerts;
            best_false_clears = false_clears;
        }
    }

    Ok(best)
}

fn calculate_counts(predictions: &[Prediction], point: OperatingPoint) -> Counts {
    let mut counts = Counts::default();
    for row in predictions {
        counts.total += 1;
        if row.label == 1 {
            counts.positives += 1;
        } else {
            counts.negatives += 1;
        }

        if row.score <= point.clear_max_score {
            counts.cleared += 1;
            if row.label == 1 {
                counts.false_clears += 1;
            } else {
                counts.true_clears += 1;
            }
        } else if row.score >= point.alert_min_score {
            counts.alerted += 1;
            if row.label == 1 {
                counts.true_alerts += 1;
            } else {
                counts.false_alerts += 1;
            }
        } else {
            counts.abstained += 1;
        }
    }
    counts
}

fn optional_rate(numerator: u64, denominator: u64) -> Option<f64> {
    if denominator == 0 {
        None
    } else {
        Some(numerator as f64 / denominator as f64)
    }
}

fn rates(counts: Counts) -> Rates {
    let classified = counts.cleared + counts.alerted;
    Rates {
        counts,
        coverage: optional_rate(classified, counts.total).unwrap_or(0.0),
        abstention_rate: optional_rate(counts.abstained, counts.total).unwrap_or(0.0),
        false_alert_rate: optional_rate(counts.false_alerts, counts.negatives),
        false_clear_rate: optional_rate(counts.false_clears, counts.positives),
        alert_precision: optional_rate(counts.true_alerts, counts.alerted),
        trojan_recall: optional_rate(counts.true_alerts, counts.positives),
        selective_risk: optional_rate(
            counts.false_alerts + counts.false_clears,
            classified,
        ),
    }
}

fn calibration(predictions: &[Prediction]) -> Result<(Calibration, Vec<Value>), String> {
    if predictions.is_empty() {
        return Err("cannot calculate calibration on an empty split".to_owned());
    }
    let mut brier_sum = 0.0;
    let mut log_loss_sum = 0.0;
    let mut bin_count = [0u64; CALIBRATION_BINS];
    let mut bin_score_sum = [0.0f64; CALIBRATION_BINS];
    let mut bin_label_sum = [0.0f64; CALIBRATION_BINS];

    for row in predictions {
        let target = f64::from(row.label);
        let error = row.score - target;
        brier_sum += error * error;

        let clipped = row.score.clamp(1e-15, 1.0 - 1e-15);
        log_loss_sum -= target * clipped.ln() + (1.0 - target) * (1.0 - clipped).ln();

        let index =
            ((row.score * CALIBRATION_BINS as f64).floor() as usize).min(CALIBRATION_BINS - 1);
        bin_count[index] += 1;
        bin_score_sum[index] += row.score;
        bin_label_sum[index] += target;
    }

    let total = predictions.len() as f64;
    let mut ece = 0.0;
    let mut bins = Vec::with_capacity(CALIBRATION_BINS);
    for index in 0..CALIBRATION_BINS {
        let count = bin_count[index];
        let mean_score = if count == 0 {
            None
        } else {
            Some(bin_score_sum[index] / count as f64)
        };
        let empirical_rate = if count == 0 {
            None
        } else {
            Some(bin_label_sum[index] / count as f64)
        };
        if let (Some(score), Some(rate)) = (mean_score, empirical_rate) {
            ece += (count as f64 / total) * (score - rate).abs();
        }
        bins.push(json!({
            "bin_index": index,
            "lower_inclusive": index as f64 / CALIBRATION_BINS as f64,
            "upper_bound": (index + 1) as f64 / CALIBRATION_BINS as f64,
            "upper_bound_inclusive": index == CALIBRATION_BINS - 1,
            "count": count,
            "mean_score": mean_score,
            "empirical_trojan_rate": empirical_rate,
        }));
    }

    Ok((
        Calibration {
            brier_score: brier_sum / total,
            log_loss: log_loss_sum / total,
            expected_calibration_error: ece,
        },
        bins,
    ))
}

fn groups(predictions: &[Prediction], point: OperatingPoint) -> Vec<GroupSummary> {
    let mut by_group = BTreeMap::<String, Vec<Prediction>>::new();
    for row in predictions {
        by_group
            .entry(row.group_id.clone())
            .or_default()
            .push(row.clone());
    }
    by_group
        .into_iter()
        .map(|(group_id, rows)| GroupSummary {
            group_id,
            metrics: rates(calculate_counts(&rows, point)),
        })
        .collect()
}

fn ratio_within(value: Option<f64>, cap: f64) -> bool {
    value.is_none_or(|rate| rate <= cap + 1e-12)
}
/// Exact one-sided Clopper-Pearson upper confidence limit for a binomial
/// event probability. `alpha` is the upper-tail significance level.
fn clopper_pearson_upper(events: u64, trials: u64, alpha: f64) -> Result<Option<f64>, String> {
    if !alpha.is_finite() || !(0.0..1.0).contains(&alpha) {
        return Err("alpha must be finite and in (0, 1)".to_owned());
    }
    if trials == 0 {
        return Ok(None);
    }
    if events > trials {
        return Err("events cannot exceed trials".to_owned());
    }
    if events == trials {
        return Ok(Some(1.0));
    }
    if events == 0 {
        return Ok(Some(-((alpha.ln() / trials as f64).exp_m1())));
    }

    let mut lower = 0.0;
    let mut upper = 1.0;
    for _ in 0..80 {
        let midpoint = lower + (upper - lower) * 0.5;
        let cdf = binomial_cdf(events, trials, midpoint);
        if cdf > alpha {
            lower = midpoint;
        } else {
            upper = midpoint;
        }
    }
    Ok(Some(upper))
}

fn binomial_cdf(max_events: u64, trials: u64, probability: f64) -> f64 {
    if probability <= 0.0 {
        return 1.0;
    }
    if probability >= 1.0 {
        return if max_events >= trials { 1.0 } else { 0.0 };
    }

    // Sum binomial masses in log space using the recurrence
    // P(X=k+1)/P(X=k) = ((n-k)/(k+1)) * (p/(1-p)).
    let log_p = probability.ln();
    let log_not_p = (-probability).ln_1p();
    let mut log_mass = trials as f64 * log_not_p;
    let mut max_log = f64::NEG_INFINITY;
    let mut scaled_sum = 0.0;

    for events in 0..=max_events {
        if log_mass > max_log {
            scaled_sum = if max_log.is_finite() {
                scaled_sum * (max_log - log_mass).exp() + 1.0
            } else {
                1.0
            };
            max_log = log_mass;
        } else {
            scaled_sum += (log_mass - max_log).exp();
        }

        if events < max_events {
            log_mass += ((trials - events) as f64).ln()
                - ((events + 1) as f64).ln()
                + log_p
                - log_not_p;
        }
    }

    (max_log.exp() * scaled_sum).clamp(0.0, 1.0)
}

fn minimum_zero_failure_groups(target_upper_bound: f64, alpha: f64) -> Result<Option<u64>, String> {
    if !alpha.is_finite() || !(0.0..1.0).contains(&alpha) {
        return Err("alpha must be finite and in (0, 1)".to_owned());
    }
    if !target_upper_bound.is_finite() || !(0.0..=1.0).contains(&target_upper_bound) {
        return Err("target upper bound must be finite and in [0, 1]".to_owned());
    }
    if target_upper_bound == 0.0 {
        return Ok(None);
    }
    if target_upper_bound == 1.0 {
        return Ok(Some(1));
    }
    let required = (alpha.ln() / (-target_upper_bound).ln_1p()).ceil();
    if !required.is_finite() || required > u64::MAX as f64 {
        return Ok(None);
    }
    Ok(Some((required as u64).max(1)))
}


fn metrics_json(value: Rates) -> Value {
    json!({
        "samples": value.counts.total,
        "positives": value.counts.positives,
        "negatives": value.counts.negatives,
        "cleared": value.counts.cleared,
        "alerted": value.counts.alerted,
        "abstained": value.counts.abstained,
        "true_alerts": value.counts.true_alerts,
        "false_alerts": value.counts.false_alerts,
        "true_clears": value.counts.true_clears,
        "false_clears": value.counts.false_clears,
        "coverage": value.coverage,
        "abstention_rate": value.abstention_rate,
        "false_alert_rate": value.false_alert_rate,
        "false_clear_rate": value.false_clear_rate,
        "alert_precision": value.alert_precision,
        "trojan_recall": value.trojan_recall,
        "selective_risk": value.selective_risk,
    })
}

fn build_receipt(
    ledger: &Ledger,
    ledger_sha256: &str,
    point: OperatingPoint,
    validation: &[Prediction],
    test: &[Prediction],
    max_false_alert_rate: f64,
    max_false_clear_rate: f64,
) -> Result<Value, String> {
    let validation_metrics = rates(calculate_counts(validation, point));
    let test_metrics = rates(calculate_counts(test, point));
    let (calibration_metrics, calibration_bins) = calibration(test)?;
    let per_group = groups(test, point);

    let mut split_audit = BTreeMap::<String, (u64, u64, u64, BTreeSet<String>)>::new();
    for row in &ledger.predictions {
        let entry = split_audit
            .entry(row.split.clone())
            .or_insert_with(|| (0, 0, 0, BTreeSet::new()));
        entry.0 += 1;
        if row.label == 1 {
            entry.1 += 1;
        } else {
            entry.2 += 1;
        }
        entry.3.insert(row.group_id.clone());
    }
    let split_summary = split_audit
        .iter()
        .map(|(split, (samples, positives, negatives, groups))| {
            json!({
                "split": split,
                "samples": samples,
                "positives": positives,
                "negatives": negatives,
                "group_count": groups.len(),
            })
        })
        .collect::<Vec<_>>();

    let aggregate_constraints_met = ratio_within(
        test_metrics.false_alert_rate,
        max_false_alert_rate,
    ) && ratio_within(test_metrics.false_clear_rate, max_false_clear_rate);

    let mut worst_group_false_alert_rate: Option<f64> = None;
    let mut worst_group_false_clear_rate: Option<f64> = None;
    let mut group_constraints_met = true;
    let mut group_coverage_sum = 0.0;
    for group in &per_group {
        group_coverage_sum += group.metrics.coverage;
        if let Some(rate) = group.metrics.false_alert_rate {
            worst_group_false_alert_rate = Some(
                worst_group_false_alert_rate.map_or(rate, |current| current.max(rate)),
            );
            if rate > max_false_alert_rate + 1e-12 {
                group_constraints_met = false;
            }
        }
        if let Some(rate) = group.metrics.false_clear_rate {
            worst_group_false_clear_rate = Some(
                worst_group_false_clear_rate.map_or(rate, |current| current.max(rate)),
            );
            if rate > max_false_clear_rate + 1e-12 {
                group_constraints_met = false;
            }
        }
    }
    let macro_group_coverage = if per_group.is_empty() {
        0.0
    } else {
        group_coverage_sum / per_group.len() as f64
    };

    // A group is a failure if it contains >=1 false alert or >=1 false clear.
    // This is a group-level event rate, not a gate-level FPR/FNR bound.
    // Bonferroni allocates alpha=0.05 across the two reported risks.
    let group_trials = per_group.len() as u64;
    let false_alert_groups = per_group
        .iter()
        .filter(|group| group.metrics.counts.false_alerts > 0)
        .count() as u64;
    let false_clear_groups = per_group
        .iter()
        .filter(|group| group.metrics.counts.false_clears > 0)
        .count() as u64;
    let familywise_alpha = 0.05;
    let alpha_per_risk = familywise_alpha / 2.0;
    let false_alert_group_upper =
        clopper_pearson_upper(false_alert_groups, group_trials, alpha_per_risk)?;
    let false_clear_group_upper =
        clopper_pearson_upper(false_clear_groups, group_trials, alpha_per_risk)?;
    let zero_failure_group_targets = json!({
        "one_percent": minimum_zero_failure_groups(0.01, alpha_per_risk)?,
        "half_percent": minimum_zero_failure_groups(0.005, alpha_per_risk)?,
        "zero_percent": minimum_zero_failure_groups(0.0, alpha_per_risk)?,
    });

    Ok(json!({
        "schema_version": RECEIPT_SCHEMA,
        "result": "evaluated",
        "claim_scope": "validation_frozen_selective_risk_metrics_only",
        "dataset_id": ledger.dataset_id,
        "grouping_key": ledger.grouping_key,
        "ledger_sha256": ledger_sha256,
        "model_sha256": ledger.model_sha256,
        "config_sha256": ledger.config_sha256,
        "split_protocol": {
            "group_id_disjoint_across_train_validation_test": true,
            "validation_used_for_threshold_selection": true,
            "test_used_for_threshold_selection": false,
            "test_used_only_for_reporting": true,
        },
        "decision_policy": {
            "clear_if_score_lte": point.clear_max_score,
            "suspect_if_score_gte": point.alert_min_score,
            "otherwise": "abstain_and_escalate",
            "max_false_alert_rate": max_false_alert_rate,
            "max_false_clear_rate": max_false_clear_rate,
            "threshold_selection_objective": "maximize_validation_coverage_subject_to_both_error_caps",
            "tie_breaking": [
                "minimize_total_false_decisions",
                "minimize_false_alerts",
                "minimize_false_clears",
                "maximize_abstention_band_width"
            ],
            "sentinels": {
                "clear_threshold_minus_one": "clear no samples",
                "alert_threshold_two": "alert no samples"
            }
        },
        "split_summary": split_summary,
        "operating_point": {
            "clear_max_score": point.clear_max_score,
            "alert_min_score": point.alert_min_score,
            "validation_metrics": metrics_json(validation_metrics),
        },
        "test_metrics": metrics_json(test_metrics),
        "calibration": {
            "score_semantics": "probability_of_trojan",
            "brier_score": calibration_metrics.brier_score,
            "log_loss": calibration_metrics.log_loss,
            "expected_calibration_error_10_bins": calibration_metrics.expected_calibration_error,
            "bins": calibration_bins,
            "note": "These are diagnostics on untouched test scores; this evaluator does not calibrate the model."
        },
        "group_metrics": per_group.iter().map(|group| json!({
            "group_id": group.group_id,
            "metrics": metrics_json(group.metrics),
        })).collect::<Vec<_>>(),
        "family_level_failure_bounds": {
            "scope": "probability_an_independent_held_out_group_has_any_error_of_the_named_type",
            "assumption": "held-out groups are independent and exchangeable draws from the target family population; this is not established by the evaluator",
            "error_event_definition": "at least one false decision of the named type within a group",
            "method": "one_sided_exact_clopper_pearson_binomial_upper_bound",
            "confidence_level_familywise": 0.95,
            "familywise_alpha": familywise_alpha,
            "bonferroni_alpha_per_risk": alpha_per_risk,
            "group_count": group_trials,
            "false_alert_failure_groups": false_alert_groups,
            "false_alert_group_failure_upper_bound": false_alert_group_upper,
            "false_clear_failure_groups": false_clear_groups,
            "false_clear_group_failure_upper_bound": false_clear_group_upper,
            "minimum_independent_groups_if_zero_failures": zero_failure_group_targets,
            "notes": [
                "These are family-level any-error bounds, not per-gate false-alert or false-clear bounds.",
                "A small number of families produces broad upper bounds even with zero observed group failures.",
                "Group independence and representativeness are assumptions supplied by the experimental design, not verified from the score ledger."
            ]
        },
        "generalization": {
            "held_out_group_count": per_group.len(),
            "macro_group_coverage": macro_group_coverage,
            "worst_group_false_alert_rate": worst_group_false_alert_rate,
            "worst_group_false_clear_rate": worst_group_false_clear_rate,
            "aggregate_constraints_met": aggregate_constraints_met,
            "all_observed_group_constraints_met": group_constraints_met,
        },
        "limits": [
            "A score is not evidence that a Trojan exists or is absent.",
            "A test-set constraint violation is reported; thresholds are never retuned on test.",
            "Calibration scores do not establish calibration under future distribution shift.",
            "The split audit checks group disjointness, not the truth or completeness of group labels.",
            "This tool does not authenticate the model, prediction producer, or input dataset."
        ],
        "signature": null,
        "security_verdict": null,
    }))
}

fn write_receipt(path: &Path, receipt: &Value) -> Result<(), String> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let filename = path
        .file_name()
        .ok_or("receipt path must include a filename")?;
    let canonical_parent = fs::canonicalize(parent)
        .map_err(|error| format!("canonicalize receipt parent {}: {error}", parent.display()))?;
    let destination = canonical_parent.join(filename);
    let bytes = serde_json::to_vec_pretty(receipt)
        .map_err(|error| format!("serialize selective-risk receipt: {error}"))?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&destination)
        .map_err(|error| format!("create new receipt {}: {error}", destination.display()))?;
    let result = file
        .write_all(&bytes)
        .and_then(|_| file.write_all(b"\n"))
        .and_then(|_| file.sync_all());
    drop(file);
    if let Err(error) = result {
        let _ = fs::remove_file(&destination);
        return Err(format!("write receipt {}: {error}", destination.display()));
    }
    Ok(())
}

fn run(args: Args) -> Result<(), String> {
    let metadata = fs::symlink_metadata(&args.ledger)
        .map_err(|error| format!("stat ledger {}: {error}", args.ledger.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err("ledger must be a regular non-symlink file".to_owned());
    }
    let bytes = fs::read(&args.ledger)
        .map_err(|error| format!("read ledger {}: {error}", args.ledger.display()))?;
    let ledger_sha256 = format!("{:x}", Sha256::digest(&bytes));
    let ledger = parse_ledger(&bytes)?;
    let validation = ledger
        .predictions
        .iter()
        .filter(|row| row.split == "validation")
        .cloned()
        .collect::<Vec<_>>();
    let test = ledger
        .predictions
        .iter()
        .filter(|row| row.split == "test")
        .cloned()
        .collect::<Vec<_>>();
    let point = select_operating_point(
        &validation,
        args.max_false_alert_rate,
        args.max_false_clear_rate,
    )?;
    let receipt = build_receipt(
        &ledger,
        &ledger_sha256,
        point,
        &validation,
        &test,
        args.max_false_alert_rate,
        args.max_false_clear_rate,
    )?;
    write_receipt(&args.receipt, &receipt)?;

    println!("HARDSEC-009 selective-risk evaluation: COMPLETE");
    println!("dataset: {}", ledger.dataset_id);
    println!("grouping key: {}", ledger.grouping_key);
    println!(
        "operating point: clear <= {:.6}, suspect >= {:.6}",
        point.clear_max_score, point.alert_min_score
    );
    println!(
        "test coverage: {:.3}; abstention: {:.3}",
        receipt["test_metrics"]["coverage"].as_f64().unwrap_or(0.0),
        receipt["test_metrics"]["abstention_rate"].as_f64().unwrap_or(0.0)
    );
    println!(
        "aggregate test constraints met: {}",
        receipt["generalization"]["aggregate_constraints_met"]
    );
    println!("receipt: {}", args.receipt.display());
    Ok(())
}

fn main() -> ExitCode {
    match parse_args() {
        Ok(Some(args)) => match run(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("HARDSEC-009 selective-risk evaluation failed: {error}");
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

    fn row(
        sample_id: &str,
        group_id: &str,
        split: &str,
        label: u8,
        score: f64,
    ) -> Prediction {
        Prediction {
            sample_id: sample_id.to_owned(),
            group_id: group_id.to_owned(),
            split: split.to_owned(),
            label,
            score,
        }
    }

    fn balanced_validation() -> Vec<Prediction> {
        vec![
            row("v-1", "host-v", "validation", 0, 0.05),
            row("v-2", "host-v", "validation", 0, 0.10),
            row("v-3", "host-v", "validation", 0, 0.90),
            row("v-4", "host-v", "validation", 1, 0.70),
            row("v-5", "host-v", "validation", 1, 0.80),
            row("v-6", "host-v", "validation", 1, 0.95),
        ]
    }

    fn valid_ledger_json() -> Value {
        json!({
            "schema_version": LEDGER_SCHEMA,
            "dataset_id": "fixture-dataset",
            "grouping_key": "host_family",
            "score_semantics": "probability_of_trojan",
            "model_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "config_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "predictions": [
                {"sample_id":"tr-1","group_id":"host-train","split":"train","label":0,"score":0.1},
                {"sample_id":"tr-2","group_id":"host-train","split":"train","label":1,"score":0.9},
                {"sample_id":"va-1","group_id":"host-validation","split":"validation","label":0,"score":0.1},
                {"sample_id":"va-2","group_id":"host-validation","split":"validation","label":1,"score":0.9},
                {"sample_id":"te-1","group_id":"host-test","split":"test","label":0,"score":0.2},
                {"sample_id":"te-2","group_id":"host-test","split":"test","label":1,"score":0.8}
            ]
        })
    }

    #[test]
    fn selects_non_overlapping_thresholds_under_zero_error_caps() {
        let validation = balanced_validation();
        let point = select_operating_point(&validation, 0.0, 0.0).unwrap();
        assert!(point.clear_max_score < point.alert_min_score);
        let result = rates(calculate_counts(&validation, point));
        assert_eq!(result.false_alert_rate, Some(0.0));
        assert_eq!(result.false_clear_rate, Some(0.0));
        assert_eq!(result.coverage, 0.5);
    }

    #[test]
    fn all_abstain_is_the_feasible_fallback_when_scores_are_entangled() {
        let validation = vec![
            row("v-1", "host-v", "validation", 0, 0.5),
            row("v-2", "host-v", "validation", 1, 0.5),
            row("v-3", "host-v", "validation", 0, 0.5),
            row("v-4", "host-v", "validation", 1, 0.5),
        ];
        let point = select_operating_point(&validation, 0.0, 0.0).unwrap();
        assert_eq!(point.clear_max_score, -1.0);
        assert_eq!(point.alert_min_score, 2.0);
        assert_eq!(rates(calculate_counts(&validation, point)).coverage, 0.0);
    }

    #[test]
    fn score_ties_between_decision_thresholds_abstain() {
        let validation = vec![
            row("v-1", "host-v", "validation", 0, 0.1),
            row("v-2", "host-v", "validation", 1, 0.9),
            row("v-3", "host-v", "validation", 0, 0.5),
            row("v-4", "host-v", "validation", 1, 0.5),
        ];
        let point = select_operating_point(&validation, 0.0, 0.0).unwrap();
        let counts = calculate_counts(&validation, point);
        assert_eq!(counts.abstained, 2);
        assert_eq!(counts.false_alerts, 0);
        assert_eq!(counts.false_clears, 0);
    }

    #[test]
    fn exact_zero_and_one_scores_are_supported() {
        let rows = vec![
            row("v-1", "host-v", "validation", 0, 0.0),
            row("v-2", "host-v", "validation", 1, 1.0),
        ];
        let point = OperatingPoint {
            clear_max_score: 0.0,
            alert_min_score: 1.0,
        };
        let metrics = rates(calculate_counts(&rows, point));
        assert_eq!(metrics.coverage, 1.0);
        assert_eq!(metrics.false_alert_rate, Some(0.0));
        assert_eq!(metrics.false_clear_rate, Some(0.0));
    }

    #[test]
    fn zero_class_denominators_are_never_divided() {
        let counts = Counts {
            total: 1,
            positives: 0,
            negatives: 1,
            ..Counts::default()
        };
        assert_eq!(rates(counts).false_clear_rate, None);
    }

    #[test]
    fn computes_brier_log_loss_and_ten_bin_ece() {
        let rows = vec![
            row("t-1", "g", "test", 0, 0.2),
            row("t-2", "g", "test", 1, 0.8),
        ];
        let (cal, bins) = calibration(&rows).unwrap();
        assert!((cal.brier_score - 0.04).abs() < 1e-12);
        assert!((cal.log_loss - (-(0.8f64).ln())).abs() < 1e-12);
        assert!((cal.expected_calibration_error - 0.2).abs() < 1e-12);
        assert_eq!(bins.len(), 10);
        assert_eq!(bins[2]["count"], 1);
        assert_eq!(bins[8]["count"], 1);
    }

    #[test]
    fn group_summary_exposes_worst_group_rates() {
        let point = OperatingPoint {
            clear_max_score: 0.2,
            alert_min_score: 0.8,
        };
        let rows = vec![
            row("t-1", "host-a", "test", 0, 0.1),
            row("t-2", "host-a", "test", 1, 0.9),
            row("t-3", "host-b", "test", 0, 0.85),
            row("t-4", "host-b", "test", 1, 0.95),
        ];
        let per_group = groups(&rows, point);
        assert_eq!(per_group.len(), 2);
        let host_b = per_group
            .iter()
            .find(|item| item.group_id == "host-b")
            .unwrap();
        assert_eq!(host_b.metrics.false_alert_rate, Some(1.0));
    }

    #[test]
    fn rejects_duplicate_sample_ids() {
        let mut ledger = valid_ledger_json();
        ledger["predictions"][1]["sample_id"] = json!("tr-1");
        assert!(parse_ledger(&serde_json::to_vec(&ledger).unwrap())
            .unwrap_err()
            .contains("duplicate sample_id"));
    }

    #[test]
    fn rejects_group_leakage_across_splits() {
        let mut ledger = valid_ledger_json();
        ledger["predictions"][2]["group_id"] = json!("host-train");
        assert!(parse_ledger(&serde_json::to_vec(&ledger).unwrap())
            .unwrap_err()
            .contains("leaks across train and validation"));
    }

    #[test]
    fn rejects_unsupported_score_semantics() {
        let mut ledger = valid_ledger_json();
        ledger["score_semantics"] = json!("arbitrary_rank_score");
        assert!(parse_ledger(&serde_json::to_vec(&ledger).unwrap())
            .unwrap_err()
            .contains("score_semantics"));
    }

    #[test]
    fn rejects_non_probability_scores() {
        let mut ledger = valid_ledger_json();
        ledger["predictions"][5]["score"] = json!(1.1);
        assert!(parse_ledger(&serde_json::to_vec(&ledger).unwrap())
            .unwrap_err()
            .contains("score must be finite and in [0, 1]"));
    }

    #[test]
    fn rejects_missing_classes_in_test() {
        let mut ledger = valid_ledger_json();
        ledger["predictions"][5]["label"] = json!(0);
        assert!(parse_ledger(&serde_json::to_vec(&ledger).unwrap())
            .unwrap_err()
            .contains("test split must contain both labels"));
    }

    #[test]
    fn rejects_invalid_model_hashes() {
        let mut ledger = valid_ledger_json();
        ledger["model_sha256"] = json!("not-a-hash");
        assert!(parse_ledger(&serde_json::to_vec(&ledger).unwrap())
            .unwrap_err()
            .contains("SHA-256"));
    }

    #[test]
    fn test_metrics_use_frozen_validation_thresholds() {
        let validation = balanced_validation();
        let point = select_operating_point(&validation, 0.0, 0.0).unwrap();
        let test = vec![
            row("t-1", "host-test", "test", 0, 0.99),
            row("t-2", "host-test", "test", 1, 0.99),
        ];
        let metrics = rates(calculate_counts(&test, point));
        assert_eq!(metrics.false_alert_rate, Some(1.0));
        assert_eq!(metrics.false_clear_rate, Some(0.0));
        // The evaluator reports the test failure; it does not retune the point.
        assert_eq!(point.alert_min_score, 0.95);
    }

    #[test]
    fn calibration_requires_nonempty_test_data() {
        assert!(calibration(&[]).is_err());
    }

    #[test]
    fn threshold_search_matches_brute_force_coverage_optimum() {
        let scores = [0.1, 0.1, 0.5, 0.5, 0.9, 0.9];
        let caps = [0.0, 0.25, 0.5, 1.0];

        for mask in 1_u32..63 {
            let rows = scores
                .iter()
                .enumerate()
                .map(|(index, score)| {
                    row(
                        &format!("v-{index}"),
                        "host-v",
                        "validation",
                        ((mask >> index) & 1) as u8,
                        *score,
                    )
                })
                .collect::<Vec<_>>();
            if !rows.iter().any(|item| item.label == 0)
                || !rows.iter().any(|item| item.label == 1)
            {
                continue;
            }

            for max_false_alert_rate in caps {
                for max_false_clear_rate in caps {
                    let selected = select_operating_point(
                        &rows,
                        max_false_alert_rate,
                        max_false_clear_rate,
                    )
                    .unwrap();
                    let selected_counts = calculate_counts(&rows, selected);
                    let allowed_alerts =
                        max_allowed_errors(max_false_alert_rate, selected_counts.negatives);
                    let allowed_clears =
                        max_allowed_errors(max_false_clear_rate, selected_counts.positives);
                    assert!(selected_counts.false_alerts <= allowed_alerts);
                    assert!(selected_counts.false_clears <= allowed_clears);

                    let unique = unique_sorted_scores(&rows);
                    let mut best_coverage = 0.0;
                    for clear_threshold in std::iter::once(-1.0).chain(unique.iter().copied()) {
                        for alert_threshold in unique.iter().copied().chain(std::iter::once(2.0)) {
                            if clear_threshold >= alert_threshold {
                                continue;
                            }
                            let candidate = OperatingPoint {
                                clear_max_score: clear_threshold,
                                alert_min_score: alert_threshold,
                            };
                            let counts = calculate_counts(&rows, candidate);
                            if counts.false_alerts <= allowed_alerts
                                && counts.false_clears <= allowed_clears
                            {
                                best_coverage = best_coverage.max(
                                    counts.cleared as f64 / counts.total as f64
                                        + counts.alerted as f64 / counts.total as f64,
                                );
                            }
                        }
                    }
                    assert_eq!(
                        selected_counts.cleared + selected_counts.alerted,
                        (best_coverage * selected_counts.total as f64).round() as u64,
                        "mask={mask}, max_false_alert_rate={max_false_alert_rate}, max_false_clear_rate={max_false_clear_rate}"
                    );
                }
            }
        }
    }

    #[test]
    fn ece_last_bin_includes_score_exactly_one() {
        let rows = vec![
            row("t-1", "g", "test", 0, 0.95),
            row("t-2", "g", "test", 1, 1.0),
        ];
        let (cal, bins) = calibration(&rows).unwrap();
        assert_eq!(bins[9]["count"], 2);
        assert_eq!(bins[9]["upper_bound_inclusive"], true);
        assert!(cal.expected_calibration_error.is_finite());
    }

    #[test]
    fn operating_point_is_deterministic() {
        let validation = balanced_validation();
        let first = select_operating_point(&validation, 0.0, 0.0).unwrap();
        let second = select_operating_point(&validation, 0.0, 0.0).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn parse_args_rejects_rates_outside_zero_one() {
        assert!(parse_rate("alpha", "-0.1").is_err());
        assert!(parse_rate("alpha", "1.1").is_err());
        assert!(parse_rate("alpha", "NaN").is_err());
    }
}
