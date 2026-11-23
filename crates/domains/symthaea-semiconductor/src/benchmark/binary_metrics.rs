// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Rust-native reference metrics for binary semiconductor risk scores.
//!
//! Metric conventions intentionally match the independent Python oracle in
//! SEMI-BENCH-001. The positive label is declared by the benchmark subject.
//! Scores are probabilities in [0, 1]; a sample is predicted positive when
//! score >= threshold. Ties are grouped for average precision and risk/coverage.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};

pub const INPUT_SCHEMA: &str = "luminous.semi-bench.binary-scores.v1";
pub const OUTPUT_SCHEMA: &str = "luminous.semi-bench.binary-metrics.v1";

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct BinaryScoresSubjectV1 {
    pub schema: String,
    #[serde(default)]
    pub case_id: Option<String>,
    pub decision_threshold: f64,
    pub calibration_bins: usize,
    pub recall_target: f64,
    pub samples: Vec<BinaryScoreSampleV1>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct BinaryScoreSampleV1 {
    pub sample_id: String,
    pub positive: bool,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BinaryClassCounts {
    pub positive: usize,
    pub negative: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ThresholdMetrics {
    pub threshold: f64,
    pub tp: usize,
    pub fp: usize,
    pub tn: usize,
    #[serde(rename = "fn")]
    pub fn_: usize,
    pub sensitivity: f64,
    pub specificity: f64,
    pub precision: f64,
    pub accuracy: f64,
    pub balanced_accuracy: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FprAtRecall {
    pub target_recall: f64,
    pub threshold: f64,
    pub achieved_recall: f64,
    pub false_positive_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RankingMetrics {
    pub average_precision: f64,
    pub auroc: f64,
    pub fpr_at_recall: Option<FprAtRecall>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CalibrationBin {
    pub bin: usize,
    pub lower_inclusive: f64,
    pub upper_exclusive: Option<f64>,
    pub count: usize,
    pub mean_score: Option<f64>,
    pub observed_positive_rate: Option<f64>,
    pub ece_contribution: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProbabilityMetrics {
    pub brier_score: f64,
    pub expected_calibration_error: f64,
    pub calibration_bins: Vec<CalibrationBin>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RiskCoveragePoint {
    pub minimum_confidence_inclusive: f64,
    pub accepted: usize,
    pub coverage: f64,
    pub errors: usize,
    pub risk: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BinaryMetricsReportV1 {
    pub schema: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub case_id: Option<String>,
    pub sample_count: usize,
    pub class_counts: BinaryClassCounts,
    pub threshold_metrics: ThresholdMetrics,
    pub ranking_metrics: RankingMetrics,
    pub probability_metrics: ProbabilityMetrics,
    pub risk_coverage: Vec<RiskCoveragePoint>,
}

#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum BinaryMetricsError {
    #[error("unsupported or missing input schema")]
    UnsupportedSchema,
    #[error("decision_threshold must be finite and in [0, 1]")]
    InvalidThreshold,
    #[error("calibration_bins must be an integer in [2, 100]")]
    InvalidCalibrationBins,
    #[error("recall_target must be finite and in (0, 1]")]
    InvalidRecallTarget,
    #[error("samples must be a non-empty array")]
    EmptySamples,
    #[error("sample identity is empty or duplicated: {0}")]
    DuplicateOrEmptySampleId(String),
    #[error("score for sample {0} must be finite and in [0, 1]")]
    InvalidScore(String),
    #[error("both positive and negative labels are required")]
    SingleClassOnly,
}

fn round12(value: f64) -> f64 {
    // Rust's round is sufficient for the published golden cases; values are
    // rounded only for stable serialization, not during intermediate sums.
    (value * 1.0e12).round() / 1.0e12
}

fn validate_subject(
    subject: &BinaryScoresSubjectV1,
) -> Result<(usize, usize), BinaryMetricsError> {
    if subject.schema != INPUT_SCHEMA {
        return Err(BinaryMetricsError::UnsupportedSchema);
    }
    if !subject.decision_threshold.is_finite()
        || !(0.0..=1.0).contains(&subject.decision_threshold)
    {
        return Err(BinaryMetricsError::InvalidThreshold);
    }
    if !(2..=100).contains(&subject.calibration_bins) {
        return Err(BinaryMetricsError::InvalidCalibrationBins);
    }
    if !subject.recall_target.is_finite()
        || subject.recall_target <= 0.0
        || subject.recall_target > 1.0
    {
        return Err(BinaryMetricsError::InvalidRecallTarget);
    }
    if subject.samples.is_empty() {
        return Err(BinaryMetricsError::EmptySamples);
    }

    let mut seen = HashSet::with_capacity(subject.samples.len());
    let mut positives = 0usize;
    let mut negatives = 0usize;
    for sample in &subject.samples {
        if sample.sample_id.trim().is_empty() || !seen.insert(sample.sample_id.as_str()) {
            return Err(BinaryMetricsError::DuplicateOrEmptySampleId(
                sample.sample_id.clone(),
            ));
        }
        if !sample.score.is_finite() || !(0.0..=1.0).contains(&sample.score) {
            return Err(BinaryMetricsError::InvalidScore(sample.sample_id.clone()));
        }
        if sample.positive {
            positives += 1;
        } else {
            negatives += 1;
        }
    }
    if positives == 0 || negatives == 0 {
        return Err(BinaryMetricsError::SingleClassOnly);
    }
    Ok((positives, negatives))
}

fn score_groups_desc(samples: &[BinaryScoreSampleV1]) -> Vec<(f64, Vec<&BinaryScoreSampleV1>)> {
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

fn confidence_groups_desc(
    samples: &[BinaryScoreSampleV1],
) -> Vec<(f64, Vec<(&BinaryScoreSampleV1, bool)>)> {
    let mut groups = BTreeMap::<u64, Vec<(&BinaryScoreSampleV1, bool)>>::new();
    for sample in samples {
        let confidence = sample.score.max(1.0 - sample.score);
        let error = (sample.score >= 0.5) != sample.positive;
        groups
            .entry(confidence.to_bits())
            .or_default()
            .push((sample, error));
    }
    let mut groups: Vec<(f64, Vec<(&BinaryScoreSampleV1, bool)>)> = groups
        .into_iter()
        .map(|(bits, rows)| (f64::from_bits(bits), rows))
        .collect();
    groups.sort_by(|left, right| right.0.total_cmp(&left.0));
    groups
}

/// Compute threshold, ranking, calibration and selective-prediction metrics.
pub fn score_binary_subject(
    subject: &BinaryScoresSubjectV1,
) -> Result<BinaryMetricsReportV1, BinaryMetricsError> {
    let (total_positive, total_negative) = validate_subject(subject)?;
    let sample_count = subject.samples.len() as f64;

    let (mut tp, mut fp, mut tn, mut fn_) = (0usize, 0usize, 0usize, 0usize);
    for sample in &subject.samples {
        let predicted_positive = sample.score >= subject.decision_threshold;
        match (sample.positive, predicted_positive) {
            (true, true) => tp += 1,
            (false, true) => fp += 1,
            (false, false) => tn += 1,
            (true, false) => fn_ += 1,
        }
    }
    let sensitivity = tp as f64 / total_positive as f64;
    let specificity = tn as f64 / total_negative as f64;
    let precision = if tp + fp == 0 {
        0.0
    } else {
        tp as f64 / (tp + fp) as f64
    };
    let accuracy = (tp + tn) as f64 / sample_count;
    let balanced_accuracy = (sensitivity + specificity) / 2.0;

    let groups = score_groups_desc(&subject.samples);
    let mut cumulative_tp = 0usize;
    let mut cumulative_fp = 0usize;
    let mut previous_recall = 0.0;
    let mut average_precision = 0.0;
    let mut fpr_at_recall = None;
    for (score, group) in &groups {
        let group_tp = group.iter().filter(|row| row.positive).count();
        let group_fp = group.len() - group_tp;
        cumulative_tp += group_tp;
        cumulative_fp += group_fp;

        let recall = cumulative_tp as f64 / total_positive as f64;
        let group_precision = cumulative_tp as f64 / (cumulative_tp + cumulative_fp) as f64;
        average_precision += (recall - previous_recall) * group_precision;
        previous_recall = recall;

        if fpr_at_recall.is_none() && recall >= subject.recall_target {
            fpr_at_recall = Some(FprAtRecall {
                target_recall: round12(subject.recall_target),
                threshold: round12(*score),
                achieved_recall: round12(recall),
                false_positive_rate: round12(cumulative_fp as f64 / total_negative as f64),
            });
        }
    }

    let positive_scores: Vec<f64> = subject
        .samples
        .iter()
        .filter(|row| row.positive)
        .map(|row| row.score)
        .collect();
    let negative_scores: Vec<f64> = subject
        .samples
        .iter()
        .filter(|row| !row.positive)
        .map(|row| row.score)
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
    let auroc = wins / (total_positive * total_negative) as f64;

    let brier_score = subject
        .samples
        .iter()
        .map(|sample| {
            let target = if sample.positive { 1.0 } else { 0.0 };
            (sample.score - target).powi(2)
        })
        .sum::<f64>()
        / sample_count;

    let mut calibration_bins = Vec::with_capacity(subject.calibration_bins);
    let mut expected_calibration_error = 0.0;
    for bin_index in 0..subject.calibration_bins {
        let low = bin_index as f64 / subject.calibration_bins as f64;
        let high = (bin_index + 1) as f64 / subject.calibration_bins as f64;
        let members: Vec<&BinaryScoreSampleV1> = subject
            .samples
            .iter()
            .filter(|sample| {
                (sample.score >= low && sample.score < high)
                    || (bin_index == subject.calibration_bins - 1 && sample.score == 1.0)
            })
            .collect();
        if members.is_empty() {
            calibration_bins.push(CalibrationBin {
                bin: bin_index,
                lower_inclusive: round12(low),
                upper_exclusive: if bin_index < subject.calibration_bins - 1 {
                    Some(round12(high))
                } else {
                    None
                },
                count: 0,
                mean_score: None,
                observed_positive_rate: None,
                ece_contribution: None,
            });
        } else {
            let count = members.len();
            let mean_score =
                members.iter().map(|sample| sample.score).sum::<f64>() / count as f64;
            let observed_positive_rate = members
                .iter()
                .filter(|sample| sample.positive)
                .count() as f64
                / count as f64;
            let contribution =
                (mean_score - observed_positive_rate).abs() * count as f64 / sample_count;
            expected_calibration_error += contribution;
            calibration_bins.push(CalibrationBin {
                bin: bin_index,
                lower_inclusive: round12(low),
                upper_exclusive: if bin_index < subject.calibration_bins - 1 {
                    Some(round12(high))
                } else {
                    None
                },
                count,
                mean_score: Some(round12(mean_score)),
                observed_positive_rate: Some(round12(observed_positive_rate)),
                ece_contribution: Some(round12(contribution)),
            });
        }
    }

    let mut risk_coverage = Vec::new();
    let mut accepted = 0usize;
    let mut errors = 0usize;
    for (confidence, group) in confidence_groups_desc(&subject.samples) {
        accepted += group.len();
        errors += group.iter().filter(|(_, error)| *error).count();
        risk_coverage.push(RiskCoveragePoint {
            minimum_confidence_inclusive: round12(confidence),
            accepted,
            coverage: round12(accepted as f64 / sample_count),
            errors,
            risk: round12(errors as f64 / accepted as f64),
        });
    }

    Ok(BinaryMetricsReportV1 {
        schema: OUTPUT_SCHEMA.to_owned(),
        case_id: subject.case_id.clone(),
        sample_count: subject.samples.len(),
        class_counts: BinaryClassCounts {
            positive: total_positive,
            negative: total_negative,
        },
        threshold_metrics: ThresholdMetrics {
            threshold: round12(subject.decision_threshold),
            tp,
            fp,
            tn,
            fn_,
            sensitivity: round12(sensitivity),
            specificity: round12(specificity),
            precision: round12(precision),
            accuracy: round12(accuracy),
            balanced_accuracy: round12(balanced_accuracy),
        },
        ranking_metrics: RankingMetrics {
            average_precision: round12(average_precision),
            auroc: round12(auroc),
            fpr_at_recall,
        },
        probability_metrics: ProbabilityMetrics {
            brier_score: round12(brier_score),
            expected_calibration_error: round12(expected_calibration_error),
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

    #[test]
    fn binary_metrics_report_matches_python_known_answer_exactly() {
        let subject: BinaryScoresSubjectV1 =
            serde_json::from_str(INPUT).expect("binary subject fixture");
        let report = score_binary_subject(&subject).expect("valid binary score subject");
        let actual = serde_json::to_value(report).expect("serialize report");
        let expected: Value = serde_json::from_str(EXPECTED).expect("expected report");
        assert_eq!(actual, expected);
    }

    #[test]
    fn tied_scores_are_grouped_for_average_precision() {
        let mut subject: BinaryScoresSubjectV1 =
            serde_json::from_str(INPUT).expect("binary subject fixture");
        let original = score_binary_subject(&subject).expect("valid input");
        subject.samples.reverse();
        let reversed = score_binary_subject(&subject).expect("valid reordered input");
        assert_eq!(
            original.ranking_metrics.average_precision,
            reversed.ranking_metrics.average_precision
        );
    }

    #[test]
    fn rejects_duplicate_sample_identity() {
        let mut subject: BinaryScoresSubjectV1 =
            serde_json::from_str(INPUT).expect("binary subject fixture");
        subject.samples[1].sample_id = subject.samples[0].sample_id.clone();
        assert!(matches!(
            score_binary_subject(&subject),
            Err(BinaryMetricsError::DuplicateOrEmptySampleId(_))
        ));
    }

    #[test]
    fn rejects_non_finite_and_out_of_range_scores() {
        let mut subject: BinaryScoresSubjectV1 =
            serde_json::from_str(INPUT).expect("binary subject fixture");
        subject.samples[0].score = f64::NAN;
        assert!(matches!(
            score_binary_subject(&subject),
            Err(BinaryMetricsError::InvalidScore(_))
        ));
        subject.samples[0].score = 1.01;
        assert!(matches!(
            score_binary_subject(&subject),
            Err(BinaryMetricsError::InvalidScore(_))
        ));
    }

    #[test]
    fn rejects_single_class_subjects() {
        let mut subject: BinaryScoresSubjectV1 =
            serde_json::from_str(INPUT).expect("binary subject fixture");
        for sample in &mut subject.samples {
            sample.positive = true;
        }
        assert_eq!(
            score_binary_subject(&subject),
            Err(BinaryMetricsError::SingleClassOnly)
        );
    }
}
