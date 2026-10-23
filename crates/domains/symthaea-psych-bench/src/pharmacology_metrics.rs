// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Safety-first evaluation metrics for binary hazard prediction.
//!
//! Metrics are computed only on the cases where a model emitted a probability;
//! abstentions are reported separately and reduce coverage. This prevents a
//! model from appearing accurate merely by refusing hard cases. A result is a
//! benchmark measurement, not a certificate that a compound is safe.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Versioned, frozen-threshold evaluation configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinarySafetyMetricConfig {
    pub evaluation_id: String,
    pub input_data_manifest_digest: String,
    pub label_definition_digest: String,
    pub split_id: String,
    /// Fixed before opening the final test set.
    pub hazard_threshold: f64,
    /// Equal-width probability bins for expected calibration error.
    pub calibration_bins: usize,
}

/// Test-set truth and one model decision. The positive class must mean the
/// predeclared hazard label from the versioned label definition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinarySafetyEvaluationCase {
    pub case_id: String,
    pub ground_truth_hazard: bool,
    pub output: BinarySafetyOutput,
}

/// The model either emits a hazard probability or abstains.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BinarySafetyOutput {
    Probability { hazard_probability: f64 },
    Abstained { reason: String },
}

/// Confusion counts at a predeclared threshold. Counts exclude abstentions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfusionCounts {
    pub true_positive: usize,
    pub false_positive: usize,
    pub true_negative: usize,
    pub false_negative: usize,
}

/// Wilson score confidence interval for a binomial proportion. This interval
/// remains non-degenerate at 0/n and n/n, unlike a naive percentile bootstrap.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WilsonInterval {
    pub successes: usize,
    pub trials: usize,
    pub confidence_level: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
}

/// Metrics that may be undefined for a small/degenerate set are represented as
/// None rather than silently replaced with a misleading zero.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinarySafetyMetricReport {
    pub schema_version: String,
    pub evaluation_id: String,
    pub input_data_manifest_digest: String,
    pub label_definition_digest: String,
    pub split_id: String,
    pub hazard_threshold: f64,
    pub calibration_bins: usize,
    pub total_cases: usize,
    pub predicted_cases: usize,
    pub abstained_cases: usize,
    pub abstained_positive_cases: usize,
    pub abstained_negative_cases: usize,
    pub coverage: Option<f64>,
    pub positive_ground_truth_cases: usize,
    pub negative_ground_truth_cases: usize,
    pub confusion: ConfusionCounts,
    /// Conditional sensitivity among cases the model chose to answer.
    pub sensitivity: Option<f64>,
    /// Hazard detection among all known positive cases; abstained positives
    /// count as not automatically detected.
    pub end_to_end_hazard_sensitivity: Option<f64>,
    /// Two-sided 95% Wilson interval for conditional sensitivity.
    pub sensitivity_wilson_95: Option<WilsonInterval>,
    /// Two-sided 95% Wilson interval for end-to-end automatic hazard detection.
    pub end_to_end_hazard_sensitivity_wilson_95: Option<WilsonInterval>,
    pub specificity: Option<f64>,
    pub specificity_wilson_95: Option<WilsonInterval>,
    pub precision: Option<f64>,
    pub precision_wilson_95: Option<WilsonInterval>,
    pub negative_predictive_value: Option<f64>,
    pub negative_predictive_value_wilson_95: Option<WilsonInterval>,
    pub balanced_accuracy: Option<f64>,
    pub auroc: Option<f64>,
    /// Non-interpolated average precision, used as a threshold-swept PR metric.
    pub average_precision: Option<f64>,
    pub brier_score: Option<f64>,
    pub expected_calibration_error: Option<f64>,
    pub notes: Vec<String>,
}

pub const METRIC_SCHEMA_VERSION: &str = "1.0";

/// Calculate predeclared binary safety metrics and retain abstention statistics.
///
/// Validation rejects duplicate case IDs, malformed digests, non-finite or
/// out-of-range probabilities, invalid thresholds, and zero calibration bins.
/// The function does not fit thresholds or calibrators on the test data.
pub fn evaluate_binary_safety(
    config: &BinarySafetyMetricConfig,
    cases: &[BinarySafetyEvaluationCase],
) -> Result<BinarySafetyMetricReport, Vec<String>> {
    let mut errors = Vec::new();
    require_nonempty(&config.evaluation_id, "evaluation_id", &mut errors);
    require_digest(
        &config.input_data_manifest_digest,
        "input_data_manifest_digest",
        &mut errors,
    );
    require_digest(
        &config.label_definition_digest,
        "label_definition_digest",
        &mut errors,
    );
    require_nonempty(&config.split_id, "split_id", &mut errors);
    if !config.hazard_threshold.is_finite() || !(0.0..=1.0).contains(&config.hazard_threshold) {
        errors.push("hazard_threshold must be finite and within [0, 1]".into());
    }
    if config.calibration_bins == 0 || config.calibration_bins > 1_000 {
        errors.push("calibration_bins must be within 1..=1000".into());
    }
    if cases.is_empty() {
        errors.push("evaluation requires at least one case".into());
    }

    let mut seen_ids = BTreeSet::new();
    for case in cases {
        require_nonempty(&case.case_id, "case_id", &mut errors);
        if !seen_ids.insert(case.case_id.as_str()) {
            errors.push(format!("duplicate case_id {:?}", case.case_id));
        }
        match &case.output {
            BinarySafetyOutput::Probability { hazard_probability } => {
                if !hazard_probability.is_finite()
                    || !(0.0..=1.0).contains(hazard_probability)
                {
                    errors.push(format!(
                        "case {:?} hazard_probability must be finite and within [0, 1]",
                        case.case_id
                    ));
                }
            }
            BinarySafetyOutput::Abstained { reason } => {
                require_nonempty(reason, "abstention reason", &mut errors);
            }
        }
    }
    if !errors.is_empty() {
        return Err(errors);
    }

    let mut predicted: Vec<(f64, bool)> = Vec::new();
    let mut abstained_positive_cases = 0;
    let mut abstained_negative_cases = 0;
    let mut confusion = ConfusionCounts::default();
    let mut brier_sum = 0.0;
    let mut bins = vec![CalibrationBin::default(); config.calibration_bins];

    for case in cases {
        match &case.output {
            BinarySafetyOutput::Abstained { .. } => {
                if case.ground_truth_hazard {
                    abstained_positive_cases += 1;
                } else {
                    abstained_negative_cases += 1;
                }
            }
            BinarySafetyOutput::Probability { hazard_probability } => {
                let probability = *hazard_probability;
                predicted.push((probability, case.ground_truth_hazard));
                let target = if case.ground_truth_hazard { 1.0 } else { 0.0 };
                brier_sum += (probability - target).powi(2);

                let bin_index = if probability >= 1.0 {
                    config.calibration_bins - 1
                } else {
                    (probability * config.calibration_bins as f64).floor() as usize
                };
                bins[bin_index].count += 1;
                bins[bin_index].probability_sum += probability;
                bins[bin_index].positive_count += if case.ground_truth_hazard { 1 } else { 0 };

                let predicted_hazard = probability >= config.hazard_threshold;
                match (predicted_hazard, case.ground_truth_hazard) {
                    (true, true) => confusion.true_positive += 1,
                    (true, false) => confusion.false_positive += 1,
                    (false, false) => confusion.true_negative += 1,
                    (false, true) => confusion.false_negative += 1,
                }
            }
        }
    }

    let total_cases = cases.len();
    let predicted_cases = predicted.len();
    let abstained_cases = total_cases - predicted_cases;
    let positive_ground_truth_cases =
        predicted.iter().filter(|item| item.1).count();
    let negative_ground_truth_cases = predicted_cases - positive_ground_truth_cases;

    let sensitivity = ratio(
        confusion.true_positive,
        confusion.true_positive + confusion.false_negative,
    );
    let specificity = ratio(
        confusion.true_negative,
        confusion.true_negative + confusion.false_positive,
    );
    let precision = ratio(
        confusion.true_positive,
        confusion.true_positive + confusion.false_positive,
    );
    let negative_predictive_value = ratio(
        confusion.true_negative,
        confusion.true_negative + confusion.false_negative,
    );
    let all_ground_truth_positives =
        confusion.true_positive + confusion.false_negative + abstained_positive_cases;
    let end_to_end_hazard_sensitivity =
        ratio(confusion.true_positive, all_ground_truth_positives);
    let sensitivity_wilson_95 = wilson_interval(
        confusion.true_positive,
        confusion.true_positive + confusion.false_negative,
        0.95,
    );
    let end_to_end_hazard_sensitivity_wilson_95 =
        wilson_interval(confusion.true_positive, all_ground_truth_positives, 0.95);
    let specificity_wilson_95 = wilson_interval(
        confusion.true_negative,
        confusion.true_negative + confusion.false_positive,
        0.95,
    );
    let precision_wilson_95 = wilson_interval(
        confusion.true_positive,
        confusion.true_positive + confusion.false_positive,
        0.95,
    );
    let negative_predictive_value_wilson_95 = wilson_interval(
        confusion.true_negative,
        confusion.true_negative + confusion.false_negative,
        0.95,
    );
    let balanced_accuracy = match (sensitivity, specificity) {
        (Some(sensitivity), Some(specificity)) => Some((sensitivity + specificity) / 2.0),
        _ => None,
    };

    let ece = if predicted_cases == 0 {
        None
    } else {
        let weighted_error: f64 = bins
            .iter()
            .filter(|bin| bin.count > 0)
            .map(|bin| {
                let mean_probability = bin.probability_sum / bin.count as f64;
                let observed_rate = bin.positive_count as f64 / bin.count as f64;
                (bin.count as f64 / predicted_cases as f64)
                    * (mean_probability - observed_rate).abs()
            })
            .sum();
        Some(weighted_error)
    };

    let (auroc, average_precision) = ranking_metrics(&predicted);
    let coverage = Some(predicted_cases as f64 / total_cases as f64);
    let brier_score = ratio_float(brier_sum, predicted_cases);

    let mut notes = vec![
        "Conditional sensitivity/specificity exclude abstentions; end_to_end_hazard_sensitivity counts abstained positive cases as not automatically detected.".to_string(),
        "Wilson intervals are reported for binomial safety rates because percentile bootstrap intervals may be degenerate when a small sample has zero observed misses.".to_string(),
        "This report measures predictions on the supplied labels and split only; it is not a safety certification or causal conclusion.".to_string(),
        "Expected calibration error depends on the prespecified equal-width bin count; report Brier score and reliability plots as complementary checks.".to_string(),
    ];
    if abstained_cases > 0 {
        notes.push("Abstained cases require a human review path and must not be counted as correct predictions.".to_string());
    }
    if positive_ground_truth_cases == 0 || negative_ground_truth_cases == 0 {
        notes.push("The answered subset has fewer than two ground-truth classes; AUROC and average precision are undefined on this subset.".to_string());
    }

    Ok(BinarySafetyMetricReport {
        schema_version: METRIC_SCHEMA_VERSION.to_string(),
        evaluation_id: config.evaluation_id.clone(),
        input_data_manifest_digest: config.input_data_manifest_digest.clone(),
        label_definition_digest: config.label_definition_digest.clone(),
        split_id: config.split_id.clone(),
        hazard_threshold: config.hazard_threshold,
        calibration_bins: config.calibration_bins,
        total_cases,
        predicted_cases,
        abstained_cases,
        abstained_positive_cases,
        abstained_negative_cases,
        coverage,
        positive_ground_truth_cases,
        negative_ground_truth_cases,
        confusion,
        sensitivity,
        end_to_end_hazard_sensitivity,
        sensitivity_wilson_95,
        end_to_end_hazard_sensitivity_wilson_95,
        specificity,
        specificity_wilson_95,
        precision,
        precision_wilson_95,
        negative_predictive_value,
        negative_predictive_value_wilson_95,
        balanced_accuracy,
        auroc,
        average_precision,
        brier_score,
        expected_calibration_error: ece,
        notes,
    })
}

#[derive(Debug, Clone, Copy, Default)]
struct CalibrationBin {
    count: usize,
    positive_count: usize,
    probability_sum: f64,
}

fn ranking_metrics(predictions: &[(f64, bool)]) -> (Option<f64>, Option<f64>) {
    let positives = predictions.iter().filter(|item| item.1).count();
    let negatives = predictions.len() - positives;
    // Both ranking metrics are non-informative when only one class is
    // represented. Keep them undefined rather than reporting a misleading AP=1.
    if positives == 0 || negatives == 0 {
        return (None, None);
    }

    // AUROC via average ranks: ties receive their shared midrank.
    let mut ascending = predictions.to_vec();
    ascending.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut positive_rank_sum = 0.0;
    let mut start = 0usize;
    while start < ascending.len() {
        let mut end = start + 1;
        while end < ascending.len() && ascending[end].0 == ascending[start].0 {
            end += 1;
        }
        let average_rank = ((start + 1 + end) as f64) / 2.0;
        let tied_positives = ascending[start..end].iter().filter(|item| item.1).count();
        positive_rank_sum += average_rank * tied_positives as f64;
        start = end;
    }
    let auroc = Some(
        (positive_rank_sum - (positives * (positives + 1) / 2) as f64)
            / (positives * negatives) as f64,
    );

    // Non-interpolated average precision using score groups, so tied scores do
    // not depend on the arbitrary input order within the tie.
    let mut descending = predictions.to_vec();
    descending.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut seen_positive = 0usize;
    let mut seen_total = 0usize;
    let mut previous_recall = 0.0;
    let mut ap = 0.0;
    start = 0;
    while start < descending.len() {
        let mut end = start + 1;
        while end < descending.len() && descending[end].0 == descending[start].0 {
            end += 1;
        }
        let group_positives = descending[start..end].iter().filter(|item| item.1).count();
        seen_positive += group_positives;
        seen_total += end - start;
        let recall = seen_positive as f64 / positives as f64;
        let precision = seen_positive as f64 / seen_total as f64;
        ap += (recall - previous_recall) * precision;
        previous_recall = recall;
        start = end;
    }
    (auroc, Some(ap))
}

/// Wilson score interval using the normal quantile for the requested
/// two-sided confidence level. For 95%, z = 1.959963984540054.
/// Returns None when there are no observed trials.
fn wilson_interval(
    successes: usize,
    trials: usize,
    confidence_level: f64,
) -> Option<WilsonInterval> {
    if trials == 0
        || !confidence_level.is_finite()
        || !(0.0 < confidence_level && confidence_level < 1.0)
        || successes > trials
    {
        return None;
    }
    // The initial safety benchmark uses a predeclared 95% interval. Avoid
    // silently approximating arbitrary confidence levels with the wrong z.
    if (confidence_level - 0.95).abs() > f64::EPSILON {
        return None;
    }
    let z = 1.959_963_984_540_054_f64;
    let n = trials as f64;
    let p = successes as f64 / n;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half_width = z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt()) / denominator;
    Some(WilsonInterval {
        successes,
        trials,
        confidence_level,
        lower_bound: (center - half_width).max(0.0),
        upper_bound: (center + half_width).min(1.0),
    })
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    if denominator == 0 {
        None
    } else {
        Some(numerator as f64 / denominator as f64)
    }
}

fn ratio_float(numerator: f64, denominator: usize) -> Option<f64> {
    if denominator == 0 {
        None
    } else {
        Some(numerator / denominator as f64)
    }
}

fn require_nonempty(value: &str, field: &str, errors: &mut Vec<String>) {
    if value.trim().is_empty() {
        errors.push(format!("{field} must not be empty"));
    }
}

fn is_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn require_digest(value: &str, field: &str, errors: &mut Vec<String>) {
    if !is_digest(value) {
        errors.push(format!("{field} must be a 64-character hexadecimal digest"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> BinarySafetyMetricConfig {
        BinarySafetyMetricConfig {
            evaluation_id: "frozen-safety-test-v1".to_string(),
            input_data_manifest_digest: "a".repeat(64),
            label_definition_digest: "b".repeat(64),
            split_id: "temporal-holdout-v1".to_string(),
            hazard_threshold: 0.5,
            calibration_bins: 2,
        }
    }

    fn case(id: &str, label: bool, probability: f64) -> BinarySafetyEvaluationCase {
        BinarySafetyEvaluationCase {
            case_id: id.to_string(),
            ground_truth_hazard: label,
            output: BinarySafetyOutput::Probability {
                hazard_probability: probability,
            },
        }
    }

    #[test]
    fn perfect_ranked_predictions_have_expected_metrics() {
        let cases = vec![
            case("p1", true, 0.9),
            case("p2", true, 0.8),
            case("n1", false, 0.2),
            case("n2", false, 0.1),
        ];
        let report = evaluate_binary_safety(&config(), &cases).expect("evaluation");
        assert_eq!(report.total_cases, 4);
        assert_eq!(report.predicted_cases, 4);
        assert_eq!(report.positive_ground_truth_cases, 2);
        assert_eq!(report.negative_ground_truth_cases, 2);
        assert_eq!(report.abstained_cases, 0);
        assert_eq!(report.confusion.true_positive, 2);
        assert_eq!(report.confusion.true_negative, 2);
        assert_eq!(report.sensitivity, Some(1.0));
        assert_eq!(report.end_to_end_hazard_sensitivity, Some(1.0));
        assert!(report.sensitivity_wilson_95.expect("Wilson CI").lower_bound < 1.0);
        assert!(report.end_to_end_hazard_sensitivity_wilson_95.expect("end-to-end Wilson CI").lower_bound < 1.0);
        assert_eq!(report.specificity, Some(1.0));
        assert_eq!(report.auroc, Some(1.0));
        assert_eq!(report.average_precision, Some(1.0));
        assert!((report.brier_score.expect("Brier") - 0.025).abs() < 1e-12);
    }

    #[test]
    fn ties_receive_neutral_auc_ranks() {
        let cases = vec![
            case("p1", true, 0.5),
            case("n1", false, 0.5),
            case("p2", true, 0.9),
            case("n2", false, 0.1),
        ];
        let report = evaluate_binary_safety(&config(), &cases).expect("evaluation");
        assert!((report.auroc.expect("AUC") - 0.875).abs() < 1e-12);
    }

    #[test]
    fn abstentions_reduce_coverage_and_are_counted_by_class() {
        let mut abstained = case("p2", true, 0.4);
        abstained.output = BinarySafetyOutput::Abstained {
            reason: "insufficient evidence".to_string(),
        };
        let cases = vec![case("p1", true, 0.9), case("n1", false, 0.1), abstained];
        let report = evaluate_binary_safety(&config(), &cases).expect("evaluation");
        assert_eq!(report.predicted_cases, 2);
        assert_eq!(report.abstained_cases, 1);
        assert_eq!(report.abstained_positive_cases, 1);
        assert_eq!(report.coverage, Some(2.0 / 3.0));
        assert_eq!(report.sensitivity, Some(1.0));
        assert_eq!(report.end_to_end_hazard_sensitivity, Some(0.5));
        assert_eq!(report.end_to_end_hazard_sensitivity_wilson_95.expect("Wilson CI").successes, 1);
        assert_eq!(report.end_to_end_hazard_sensitivity_wilson_95.expect("Wilson CI").trials, 2);
        assert!(report.notes.iter().any(|note| note.contains("human review")));
    }

    #[test]
    fn one_class_test_set_does_not_claim_auc_or_specificity() {
        let cases = vec![case("p1", true, 0.9), case("p2", true, 0.4)];
        let report = evaluate_binary_safety(&config(), &cases).expect("evaluation");
        assert_eq!(report.auroc, None);
        assert_eq!(report.average_precision, None);
        assert_eq!(report.specificity, None);
    }

    #[test]
    fn invalid_probability_duplicate_id_threshold_and_manifest_fail_closed() {
        let mut bad_config = config();
        bad_config.hazard_threshold = f64::NAN;
        bad_config.input_data_manifest_digest = "bad".to_string();
        let cases = vec![case("duplicate", true, 1.1), case("duplicate", false, 0.2)];
        let errors = evaluate_binary_safety(&bad_config, &cases).expect_err("invalid input");
        assert!(errors.iter().any(|e| e.contains("hazard_threshold")));
        assert!(errors.iter().any(|e| e.contains("input_data_manifest_digest")));
        assert!(errors.iter().any(|e| e.contains("duplicate case_id")));
        assert!(errors.iter().any(|e| e.contains("hazard_probability")));

        let mut excessive_bins = config();
        excessive_bins.calibration_bins = 1_001;
        assert!(evaluate_binary_safety(&excessive_bins, &[case("ok", true, 0.9)])
            .expect_err("excessive bin count must fail")
            .iter()
            .any(|e| e.contains("calibration_bins")));
    }

    #[test]
    fn empty_prediction_coverage_is_zero_and_score_metrics_are_undefined() {
        let cases = vec![BinarySafetyEvaluationCase {
            case_id: "abstain-all".to_string(),
            ground_truth_hazard: true,
            output: BinarySafetyOutput::Abstained {
                reason: "human review".to_string(),
            },
        }];
        let report = evaluate_binary_safety(&config(), &cases).expect("evaluation");
        assert_eq!(report.coverage, Some(0.0));
        assert_eq!(report.brier_score, None);
        assert_eq!(report.auroc, None);
        assert_eq!(report.sensitivity, None);
    }
}
