// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Validation-only threshold selection with test-set isolation.
//!
//! The selector maximizes validation recall subject to a declared precision
//! floor. Equal validation scores are admitted together. The chosen threshold
//! is then frozen and applied once to the disjoint test subject.

use super::binary_metrics::{
    BinaryMetricsError, BinaryMetricsReportV1, BinaryScoreSampleV1, BinaryScoresSubjectV1,
    score_binary_subject,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};

pub const INPUT_SCHEMA: &str = "luminous.semi-bench.validation-threshold-input.v1";
pub const OUTPUT_SCHEMA: &str = "luminous.semi-bench.validation-threshold-report.v1";

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ValidationThresholdInputV1 {
    pub schema: String,
    pub case_id: Option<String>,
    pub minimum_validation_precision: f64,
    pub fallback_threshold: f64,
    pub validation_samples: Vec<BinaryScoreSampleV1>,
    pub test_subject: BinaryScoresSubjectV1,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ThresholdSelection {
    pub threshold: f64,
    pub validation_precision: f64,
    pub validation_recall: f64,
    pub validation_tp: usize,
    pub validation_fp: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ValidationThresholdReportV1 {
    pub schema: String,
    pub case_id: Option<String>,
    pub minimum_validation_precision: f64,
    pub fallback_threshold: f64,
    pub fallback_used: bool,
    pub selection: Option<ThresholdSelection>,
    pub test_metrics: BinaryMetricsReportV1,
}

#[derive(Debug, Clone, thiserror::Error, PartialEq)]
pub enum ThresholdPolicyError {
    #[error("unsupported or missing validation-threshold schema")]
    UnsupportedSchema,
    #[error("minimum_validation_precision must be finite and in (0, 1]")]
    InvalidMinimumPrecision,
    #[error("fallback_threshold must be finite and in [0, 1]")]
    InvalidFallbackThreshold,
    #[error("validation_samples must be non-empty")]
    EmptyValidation,
    #[error("validation sample identity is empty or duplicated: {0}")]
    DuplicateOrEmptyValidationId(String),
    #[error("validation score for sample {0} must be finite and in [0, 1]")]
    InvalidValidationScore(String),
    #[error("validation samples must contain both positive and negative labels")]
    SingleClassValidation,
    #[error("validation/test identity overlap: {0}")]
    ValidationTestIdentityOverlap(String),
    #[error("test score subject is invalid: {0}")]
    InvalidTestSubject(#[from] BinaryMetricsError),
}

fn round12(value: f64) -> f64 {
    (value * 1.0e12).round() / 1.0e12
}

fn descending_validation_groups(
    samples: &[BinaryScoreSampleV1],
) -> Vec<(f64, Vec<&BinaryScoreSampleV1>)> {
    let mut groups = BTreeMap::<u64, Vec<&BinaryScoreSampleV1>>::new();
    for sample in samples {
        groups
            .entry(sample.score.to_bits())
            .or_default()
            .push(sample);
    }
    let mut groups: Vec<(f64, Vec<&BinaryScoreSampleV1>)> = groups
        .into_iter()
        .map(|(bits, rows)| (f64::from_bits(bits), rows))
        .collect();
    groups.sort_by(|left, right| right.0.total_cmp(&left.0));
    groups
}

/// Select a threshold on validation only, then apply it to a disjoint test set.
///
/// If no observed validation threshold satisfies the precision floor, the
/// explicit fallback is used and recorded. This does not make a small
/// validation sample statistically reliable; support and uncertainty must
/// still be reported.
pub fn evaluate_with_validation_threshold(
    input: &ValidationThresholdInputV1,
) -> Result<ValidationThresholdReportV1, ThresholdPolicyError> {
    if input.schema != INPUT_SCHEMA {
        return Err(ThresholdPolicyError::UnsupportedSchema);
    }
    if !input.minimum_validation_precision.is_finite()
        || input.minimum_validation_precision <= 0.0
        || input.minimum_validation_precision > 1.0
    {
        return Err(ThresholdPolicyError::InvalidMinimumPrecision);
    }
    if !input.fallback_threshold.is_finite()
        || !(0.0..=1.0).contains(&input.fallback_threshold)
    {
        return Err(ThresholdPolicyError::InvalidFallbackThreshold);
    }
    if input.validation_samples.is_empty() {
        return Err(ThresholdPolicyError::EmptyValidation);
    }

    let mut validation_ids = HashSet::with_capacity(input.validation_samples.len());
    let mut positive_count = 0usize;
    let mut negative_count = 0usize;
    for sample in &input.validation_samples {
        if sample.sample_id.trim().is_empty()
            || !validation_ids.insert(sample.sample_id.as_str())
        {
            return Err(ThresholdPolicyError::DuplicateOrEmptyValidationId(
                sample.sample_id.clone(),
            ));
        }
        if !sample.score.is_finite() || !(0.0..=1.0).contains(&sample.score) {
            return Err(ThresholdPolicyError::InvalidValidationScore(
                sample.sample_id.clone(),
            ));
        }
        if sample.positive {
            positive_count += 1;
        } else {
            negative_count += 1;
        }
    }
    if positive_count == 0 || negative_count == 0 {
        return Err(ThresholdPolicyError::SingleClassValidation);
    }

    // Validate the test subject before checking overlap, so malformed tests
    // cannot be mistaken for a valid disjoint test partition.
    super::binary_metrics::score_binary_subject(&input.test_subject)?;
    let test_ids: HashSet<&str> = input
        .test_subject
        .samples
        .iter()
        .map(|sample| sample.sample_id.as_str())
        .collect();
    if let Some(overlap) = validation_ids.intersection(&test_ids).next() {
        return Err(ThresholdPolicyError::ValidationTestIdentityOverlap(
            (*overlap).to_owned(),
        ));
    }

    let mut cumulative_tp = 0usize;
    let mut cumulative_fp = 0usize;
    let mut best_recall = -1.0f64;
    let mut selection: Option<ThresholdSelection> = None;

    for (threshold, group) in descending_validation_groups(&input.validation_samples) {
        let group_tp = group.iter().filter(|row| row.positive).count();
        let group_fp = group.len() - group_tp;
        cumulative_tp += group_tp;
        cumulative_fp += group_fp;

        let precision = cumulative_tp as f64 / (cumulative_tp + cumulative_fp) as f64;
        let recall = cumulative_tp as f64 / positive_count as f64;
        if precision >= input.minimum_validation_precision && recall > best_recall {
            best_recall = recall;
            selection = Some(ThresholdSelection {
                threshold: round12(threshold),
                validation_precision: round12(precision),
                validation_recall: round12(recall),
                validation_tp: cumulative_tp,
                validation_fp: cumulative_fp,
            });
        }
    }

    let fallback_used = selection.is_none();
    let selected_threshold = selection
        .as_ref()
        .map(|value| value.threshold)
        .unwrap_or(input.fallback_threshold);
    let mut test_subject = input.test_subject.clone();
    test_subject.decision_threshold = selected_threshold;
    let test_metrics = score_binary_subject(&test_subject)?;

    Ok(ValidationThresholdReportV1 {
        schema: OUTPUT_SCHEMA.to_owned(),
        case_id: input.test_subject.case_id.clone(),
        minimum_validation_precision: round12(input.minimum_validation_precision),
        fallback_threshold: round12(input.fallback_threshold),
        fallback_used,
        selection,
        test_metrics,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const INPUT: &str = include_str!(
        "../../../../../benchmarks/semi-bench-001/reference/fixtures/validation-threshold-known-answer-v1.json"
    );
    const EXPECTED: &str = include_str!(
        "../../../../../benchmarks/semi-bench-001/reference/fixtures/validation-threshold-known-answer-v1.expected.json"
    );

    fn read_input() -> ValidationThresholdInputV1 {
        serde_json::from_str(INPUT).expect("validation threshold fixture")
    }

    fn assert_json_numeric_equivalent(actual: &Value, expected: &Value) {
        match (actual, expected) {
            (Value::Number(left), Value::Number(right)) => {
                assert_eq!(left.as_f64(), right.as_f64());
            }
            (Value::Array(left), Value::Array(right)) => {
                assert_eq!(left.len(), right.len());
                for (left, right) in left.iter().zip(right) {
                    assert_json_numeric_equivalent(left, right);
                }
            }
            (Value::Object(left), Value::Object(right)) => {
                assert_eq!(
                    left.keys().collect::<Vec<_>>(),
                    right.keys().collect::<Vec<_>>()
                );
                for (key, left_value) in left {
                    assert_json_numeric_equivalent(
                        left_value,
                        right.get(key).expect("matching key"),
                    );
                }
            }
            _ => assert_eq!(actual, expected),
        }
    }

    #[test]
    fn validation_threshold_report_matches_python_known_answer_exactly() {
        let input = read_input();
        let report = evaluate_with_validation_threshold(&input).expect("valid threshold input");
        let actual = serde_json::to_value(report).expect("serialize threshold report");
        let expected: Value = serde_json::from_str(EXPECTED).expect("expected JSON");
        assert_json_numeric_equivalent(&actual, &expected);
    }

    #[test]
    fn rejects_validation_test_identity_overlap() {
        let mut input = read_input();
        input.test_subject.samples[0].sample_id =
            input.validation_samples[0].sample_id.clone();
        assert!(matches!(
            evaluate_with_validation_threshold(&input),
            Err(ThresholdPolicyError::ValidationTestIdentityOverlap(_))
        ));
    }

    #[test]
    fn records_fallback_when_precision_floor_is_infeasible() {
        let mut input = read_input();
        for sample in &mut input.validation_samples {
            sample.score = if sample.positive { 0.1 } else { 0.9 };
        }
        input.minimum_validation_precision = 1.0;
        let report = evaluate_with_validation_threshold(&input).expect("fallback is explicit");
        assert!(report.fallback_used);
        assert!(report.selection.is_none());
        assert_eq!(report.test_metrics.threshold_metrics.threshold, 0.5);
    }

    #[test]
    fn rejects_single_class_validation() {
        let mut input = read_input();
        for sample in &mut input.validation_samples {
            sample.positive = true;
        }
        assert_eq!(
            evaluate_with_validation_threshold(&input),
            Err(ThresholdPolicyError::SingleClassValidation)
        );
    }

    #[test]
    fn validation_sample_order_does_not_change_selected_threshold() {
        let mut input = read_input();
        let original =
            evaluate_with_validation_threshold(&input).expect("valid threshold input");
        input.validation_samples.reverse();
        let reversed =
            evaluate_with_validation_threshold(&input).expect("valid reordered input");
        assert_eq!(original, reversed);
    }
}
