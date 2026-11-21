// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Rust implementation of the frozen SEMI-BENCH-001 binary metric contract.
//!
//! This module scores supplied probabilities. It does not train a model,
//! impute sensor values, or infer a physical defect mechanism.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use thiserror::Error;

pub const INPUT_SCHEMA_V1: &str = "luminous.semi-bench.binary-scores.v1";
pub const OUTPUT_SCHEMA_V1: &str = "luminous.semi-bench.binary-metrics.v1";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BinaryScoreSampleV1 {
    pub sample_id: String,
    pub positive: bool,
    pub score: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BinaryScoreSubjectV1 {
    pub schema: String,
    #[serde(default)]
    pub case_id: Option<String>,
    pub decision_threshold: f64,
    pub calibration_bins: usize,
    pub recall_target: f64,
    #[serde(default)]
    pub positive_class_semantics: Option<String>,
    pub samples: Vec<BinaryScoreSampleV1>,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct BinaryMetricsError(pub String);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BinaryClassCounts {
    pub positive: usize,
    pub negative: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThresholdMetricsV1 {
    pub threshold: f64,
    pub tp: usize,
    pub fp: usize,
    pub tn: usize,
    #[serde(rename = "fn")]
    pub fn_count: usize,
    pub sensitivity: f64,
    pub specificity: f64,
    pub precision: f64,
    pub accuracy: f64,
    pub balanced_accuracy: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FprAtRecallV1 {
    pub target_recall: f64,
    pub threshold: f64,
    pub achieved_recall: f64,
    pub false_positive_rate: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RankingMetricsV1 {
    pub average_precision: f64,
    pub auroc: f64,
    pub fpr_at_recall: Option<FprAtRecallV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibrationBinV1 {
    pub bin: usize,
    pub lower_inclusive: f64,
    pub upper_exclusive: Option<f64>,
    pub count: usize,
    pub mean_score: Option<f64>,
    pub observed_positive_rate: Option<f64>,
    pub ece_contribution: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProbabilityMetricsV1 {
    pub brier_score: f64,
    pub expected_calibration_error: f64,
    pub calibration_bins: Vec<CalibrationBinV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RiskCoveragePointV1 {
    pub minimum_confidence_inclusive: f64,
    pub accepted: usize,
    pub coverage: f64,
    pub errors: usize,
    pub risk: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BinaryMetricReportV1 {
    pub schema: String,
    pub case_id: Option<String>,
    pub sample_count: usize,
    pub class_counts: BinaryClassCounts,
    pub threshold_metrics: ThresholdMetricsV1,
    pub ranking_metrics: RankingMetricsV1,
    pub probability_metrics: ProbabilityMetricsV1,
    pub risk_coverage: Vec<RiskCoveragePointV1>,
}

fn rounded(value: f64) -> f64 {
    (value * 1_000_000_000_000.0).round() / 1_000_000_000_000.0
}

fn validate(subject: &BinaryScoreSubjectV1) -> Result<(usize, usize), BinaryMetricsError> {
    if subject.schema != INPUT_SCHEMA_V1 {
        return Err(BinaryMetricsError(
            "schema: unsupported or missing input schema".into(),
        ));
    }
    if !subject.decision_threshold.is_finite() || !(0.0..=1.0).contains(&subject.decision_threshold)
    {
        return Err(BinaryMetricsError(
            "decision_threshold: must be finite and in [0, 1]".into(),
        ));
    }
    if !(2..=100).contains(&subject.calibration_bins) {
        return Err(BinaryMetricsError(
            "calibration_bins: expected integer in [2, 100]".into(),
        ));
    }
    if !subject.recall_target.is_finite()
        || !(0.0 < subject.recall_target && subject.recall_target <= 1.0)
    {
        return Err(BinaryMetricsError(
            "recall_target: must be finite and in (0, 1]".into(),
        ));
    }
    if subject.samples.is_empty() {
        return Err(BinaryMetricsError(
            "samples: expected non-empty array".into(),
        ));
    }

    let mut seen = HashSet::with_capacity(subject.samples.len());
    let mut positives = 0;
    for (index, sample) in subject.samples.iter().enumerate() {
        if sample.sample_id.trim().is_empty() {
            return Err(BinaryMetricsError(format!(
                "samples[{index}].sample_id: expected non-empty string"
            )));
        }
        if !seen.insert(sample.sample_id.as_str()) {
            return Err(BinaryMetricsError(format!(
                "samples[{index}].sample_id: duplicate {:?}",
                sample.sample_id
            )));
        }
        if !sample.score.is_finite() {
            return Err(BinaryMetricsError(format!(
                "samples[{index}].score: expected finite number"
            )));
        }
        if !(0.0..=1.0).contains(&sample.score) {
            return Err(BinaryMetricsError(format!(
                "samples[{index}].score: must be in [0, 1]"
            )));
        }
        if sample.positive {
            positives += 1;
        }
    }
    let negatives = subject.samples.len() - positives;
    if positives == 0 || negatives == 0 {
        return Err(BinaryMetricsError(
            "samples: both positive and negative classes are required".into(),
        ));
    }
    Ok((positives, negatives))
}

fn groups_desc(
    samples: &[BinaryScoreSampleV1],
    score: impl Fn(&BinaryScoreSampleV1) -> f64,
) -> Vec<(f64, Vec<&BinaryScoreSampleV1>)> {
    let mut ordered: Vec<&BinaryScoreSampleV1> = samples.iter().collect();
    ordered.sort_by(|a, b| score(b).total_cmp(&score(a)));
    let mut groups: Vec<(f64, Vec<&BinaryScoreSampleV1>)> = Vec::new();
    for sample in ordered {
        let value = score(sample);
        match groups.last_mut() {
            Some((last, members)) if *last == value => members.push(sample),
            _ => groups.push((value, vec![sample])),
        }
    }
    groups
}

/// Scores a version-1 binary-probability subject using tie-grouped metrics.
pub fn score_binary_subject_v1(
    subject: &BinaryScoreSubjectV1,
) -> Result<BinaryMetricReportV1, BinaryMetricsError> {
    let (positive_count, negative_count) = validate(subject)?;
    let n = subject.samples.len();

    let (mut tp, mut fp, mut tn, mut fn_count) = (0, 0, 0, 0);
    for sample in &subject.samples {
        match (sample.positive, sample.score >= subject.decision_threshold) {
            (true, true) => tp += 1,
            (false, true) => fp += 1,
            (false, false) => tn += 1,
            (true, false) => fn_count += 1,
        }
    }
    let sensitivity = tp as f64 / positive_count as f64;
    let specificity = tn as f64 / negative_count as f64;
    let precision = if tp + fp == 0 {
        0.0
    } else {
        tp as f64 / (tp + fp) as f64
    };
    let accuracy = (tp + tn) as f64 / n as f64;
    let balanced_accuracy = (sensitivity + specificity) / 2.0;

    let mut cumulative_tp = 0usize;
    let mut cumulative_fp = 0usize;
    let mut previous_recall = 0.0;
    let mut average_precision = 0.0;
    for (_, group) in groups_desc(&subject.samples, |sample| sample.score) {
        let group_tp = group.iter().filter(|s| s.positive).count();
        cumulative_tp += group_tp;
        cumulative_fp += group.len() - group_tp;
        let recall = cumulative_tp as f64 / positive_count as f64;
        let group_precision = cumulative_tp as f64 / (cumulative_tp + cumulative_fp) as f64;
        average_precision += (recall - previous_recall) * group_precision;
        previous_recall = recall;
    }

    let positive_scores: Vec<f64> = subject
        .samples
        .iter()
        .filter(|s| s.positive)
        .map(|s| s.score)
        .collect();
    let negative_scores: Vec<f64> = subject
        .samples
        .iter()
        .filter(|s| !s.positive)
        .map(|s| s.score)
        .collect();
    let mut wins = 0.0;
    for positive in &positive_scores {
        for negative in &negative_scores {
            if positive > negative {
                wins += 1.0;
            } else if positive == negative {
                wins += 0.5;
            }
        }
    }
    let auroc = wins / (positive_count * negative_count) as f64;

    let brier_score = subject
        .samples
        .iter()
        .map(|s| (s.score - if s.positive { 1.0 } else { 0.0 }).powi(2))
        .sum::<f64>()
        / n as f64;

    let mut calibration_bins = Vec::with_capacity(subject.calibration_bins);
    let mut ece = 0.0;
    for index in 0..subject.calibration_bins {
        let low = index as f64 / subject.calibration_bins as f64;
        let high = (index + 1) as f64 / subject.calibration_bins as f64;
        let members: Vec<&BinaryScoreSampleV1> = subject
            .samples
            .iter()
            .filter(|s| {
                (low <= s.score && s.score < high)
                    || (index + 1 == subject.calibration_bins && s.score == 1.0)
            })
            .collect();
        let count = members.len();
        let (mean, observed, contribution) = if count == 0 {
            (None, None, None)
        } else {
            let mean = members.iter().map(|s| s.score).sum::<f64>() / count as f64;
            let observed = members.iter().filter(|s| s.positive).count() as f64 / count as f64;
            let contribution = (mean - observed).abs() * count as f64 / n as f64;
            ece += contribution;
            (
                Some(rounded(mean)),
                Some(rounded(observed)),
                Some(rounded(contribution)),
            )
        };
        calibration_bins.push(CalibrationBinV1 {
            bin: index,
            lower_inclusive: rounded(low),
            upper_exclusive: if index + 1 < subject.calibration_bins {
                Some(rounded(high))
            } else {
                None
            },
            count,
            mean_score: mean,
            observed_positive_rate: observed,
            ece_contribution: contribution,
        });
    }

    let mut cumulative_tp = 0usize;
    let mut cumulative_fp = 0usize;
    let mut fpr_at_recall = None;
    for (threshold, group) in groups_desc(&subject.samples, |s| s.score) {
        cumulative_tp += group.iter().filter(|s| s.positive).count();
        cumulative_fp += group.iter().filter(|s| !s.positive).count();
        let achieved = cumulative_tp as f64 / positive_count as f64;
        if achieved >= subject.recall_target {
            fpr_at_recall = Some(FprAtRecallV1 {
                target_recall: rounded(subject.recall_target),
                threshold: rounded(threshold),
                achieved_recall: rounded(achieved),
                false_positive_rate: rounded(cumulative_fp as f64 / negative_count as f64),
            });
            break;
        }
    }

    let mut risk_coverage = Vec::new();
    let mut accepted = 0usize;
    let mut errors = 0usize;
    for (confidence, group) in groups_desc(&subject.samples, |s| s.score.max(1.0 - s.score)) {
        accepted += group.len();
        errors += group
            .iter()
            .filter(|s| (s.score >= 0.5) != s.positive)
            .count();
        risk_coverage.push(RiskCoveragePointV1 {
            minimum_confidence_inclusive: rounded(confidence),
            accepted,
            coverage: rounded(accepted as f64 / n as f64),
            errors,
            risk: rounded(errors as f64 / accepted as f64),
        });
    }

    Ok(BinaryMetricReportV1 {
        schema: OUTPUT_SCHEMA_V1.into(),
        case_id: subject.case_id.clone(),
        sample_count: n,
        class_counts: BinaryClassCounts {
            positive: positive_count,
            negative: negative_count,
        },
        threshold_metrics: ThresholdMetricsV1 {
            threshold: rounded(subject.decision_threshold),
            tp,
            fp,
            tn,
            fn_count,
            sensitivity: rounded(sensitivity),
            specificity: rounded(specificity),
            precision: rounded(precision),
            accuracy: rounded(accuracy),
            balanced_accuracy: rounded(balanced_accuracy),
        },
        ranking_metrics: RankingMetricsV1 {
            average_precision: rounded(average_precision),
            auroc: rounded(auroc),
            fpr_at_recall,
        },
        probability_metrics: ProbabilityMetricsV1 {
            brier_score: rounded(brier_score),
            expected_calibration_error: rounded(ece),
            calibration_bins,
        },
        risk_coverage,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const INPUT: &str = include_str!(
        "../../../../../benchmarks/semi-bench-001/reference/fixtures/binary-metrics-known-answer-v1.json"
    );
    const EXPECTED: &str = include_str!(
        "../../../../../benchmarks/semi-bench-001/reference/fixtures/binary-metrics-known-answer-v1.expected.json"
    );

    fn known_answer() -> BinaryScoreSubjectV1 {
        serde_json::from_str(INPUT).expect("known-answer input must deserialize")
    }

    #[test]
    fn rust_metrics_match_frozen_python_report() {
        let report = score_binary_subject_v1(&known_answer()).expect("input must score");
        let actual = serde_json::to_value(report).expect("report serializes");
        let expected: Value = serde_json::from_str(EXPECTED).expect("expected JSON parses");
        assert_eq!(actual, expected);
    }

    #[test]
    fn input_order_does_not_change_any_report_field() {
        let subject = known_answer();
        let expected = score_binary_subject_v1(&subject).unwrap();
        let mut reversed = subject;
        reversed.samples.reverse();
        assert_eq!(score_binary_subject_v1(&reversed).unwrap(), expected);
    }

    #[test]
    fn duplicate_sample_identity_is_rejected() {
        let mut subject = known_answer();
        subject.samples[1].sample_id = subject.samples[0].sample_id.clone();
        assert!(
            score_binary_subject_v1(&subject)
                .unwrap_err()
                .to_string()
                .contains("duplicate")
        );
    }

    #[test]
    fn one_class_only_is_rejected() {
        let mut subject = known_answer();
        for sample in &mut subject.samples {
            sample.positive = true;
        }
        assert!(
            score_binary_subject_v1(&subject)
                .unwrap_err()
                .to_string()
                .contains("both positive and negative")
        );
    }

    #[test]
    fn out_of_range_probability_is_rejected_without_clipping() {
        let mut subject = known_answer();
        subject.samples[0].score = 1.01;
        assert!(
            score_binary_subject_v1(&subject)
                .unwrap_err()
                .to_string()
                .contains("must be in [0, 1]")
        );
    }

    #[test]
    fn invalid_threshold_recall_and_bin_count_are_rejected() {
        let mut subject = known_answer();
        subject.decision_threshold = 1.01;
        assert!(score_binary_subject_v1(&subject).is_err());
        let mut subject = known_answer();
        subject.recall_target = 0.0;
        assert!(score_binary_subject_v1(&subject).is_err());
        let mut subject = known_answer();
        subject.calibration_bins = 1;
        assert!(score_binary_subject_v1(&subject).is_err());
    }
}
