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
    pub predicted_positive_cases: usize,
    pub predicted_negative_cases: usize,
    pub confusion: ConfusionCounts,
    pub sensitivity: Option<f64>,
    pub specificity: Option<f64>,
    pub precision: Option<f64>,
    pub negative_predictive_value: Option<f64>,
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
    if config.calibration_bins == 0 {
        errors.push("calibration_bins must be greater than zero".into());
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
                bins[bin_index].positive_count += usize::from(case.ground_truth_hazard);

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
    let predicted_positive_cases =
        predicted.iter().filter(|(_, label)| *label).count();
    let predicted_negative_cases = predicted_cases - predicted_positive_cases;

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
        "All probability-based quality metrics exclude abstentions; coverage and abstained class counts must be read alongside them.".to_string(),
        "This report measures predictions on the supplied labels and split only; it is not a safety certification or causal conclusion.".to_string(),
        "Expected calibration error depends on the prespecified equal-width bin count; report Brier score and reliability plots as complementary checks.".to_string(),
    ];
    if abstained_cases > 0 {
        notes.push("Abstained cases require a human review path and must not be counted as correct predictions.".to_string());
    }
    if predicted_positive_cases == 0 || predicted_negative_cases == 0 {
        notes.push("Only one ground-truth class received a prediction; AUROC is undefined on the predicted subset.".to_string());
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
        predicted_positive_cases,
        predicted_negative_cases,
        confusion,
        sensitivity,
        specificity,
        precision,
        negative_predictive_value,
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
    let positives = predictions.iter().filter(|(_, label)| *label).count();
    let negatives = predictions.len() - positives;
    if positives == 0 || negatives == 0 {
        return (None, if positives == 0 { None } else { Some(1.0) });
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
        let tied_positives = ascending[start..end].iter().filter(|(_, label)| *label).count();
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
        let group_positives = descending[start..end].iter().filter(|(_, label)| *label).count();
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
        assert_eq!(report.abstained_cases, 0);
        assert_eq!(report.confusion.true_positive, 2);
        assert_eq!(report.confusion.true_negative, 2);
        assert_eq!(report.sensitivity, Some(1.0));
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
        assert!(report.notes.iter().any(|note| note.contains("human review")));
    }

    #[test]
    fn one_class_test_set_does_not_claim_auc_or_specificity() {
        let cases = vec![case("p1", true, 0.9), case("p2", true, 0.4)];
        let report = evaluate_binary_safety(&config(), &cases).expect("evaluation");
        assert_eq!(report.auroc, None);
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
