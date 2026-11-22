// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Class-conditional split conformal classification with explicit abstention.
//!
//! The predictor receives only sample IDs and scores. Truth labels are kept in
//! a separate evaluation API. The class-conditional coverage statement relies
//! on exchangeability within each class and matured calibration labels.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use thiserror::Error;

use crate::benchmark::process_event_graph::{
    InformationCutoffSnapshotV1, SNAPSHOT_SCHEMA_V1,
};

pub const SUBJECT_SCHEMA_V1: &str = "luminous.semi-bench.mondrian-conformal-subject.v1";
pub const REPORT_SCHEMA_V1: &str = "luminous.semi-bench.mondrian-conformal-report.v1";
pub const EVALUATION_SCHEMA_V1: &str = "luminous.semi-bench.mondrian-conformal-evaluation.v1";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConformalCalibrationSampleV1 {
    pub sample_id: String,
    /// Label identity must appear in the information snapshot's matured
    /// validated-label set before it can enter calibration.
    pub label_id: String,
    pub positive: bool,
    pub score: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConformalPredictionSampleV1 {
    pub sample_id: String,
    pub score: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConformalSubjectV1 {
    pub schema: String,
    pub case_id: String,
    pub model_sha256: String,
    pub alpha: f64,
    /// IDs used to fit the underlying model. The conformal calibrator verifies
    /// disjointness; it does not re-train or inspect model internals.
    pub training_sample_ids: Vec<String>,
    pub calibration_samples: Vec<ConformalCalibrationSampleV1>,
    /// Inference samples contain no labels.
    pub prediction_samples: Vec<ConformalPredictionSampleV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConformalClassThresholdV1 {
    pub class_positive: bool,
    pub calibration_count: usize,
    /// One-indexed finite-sample order statistic; n+1 means +infinity.
    pub quantile_rank: usize,
    pub threshold: Option<f64>,
    pub unbounded_threshold: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectiveDecisionV1 {
    PredictNegative,
    PredictPositive,
    Abstain,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConformalPredictionSetV1 {
    pub sample_id: String,
    pub score: f64,
    pub negative_in_set: bool,
    pub positive_in_set: bool,
    pub set_size: usize,
    pub decision: SelectiveDecisionV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConformalClaimLimitsV1 {
    pub class_conditional_coverage_requires_exchangeability_within_each_class: bool,
    pub guarantee_is_marginal_not_per_sample_conditional: bool,
    pub arbitrary_temporal_tool_chamber_recipe_or_fab_shift_not_covered: bool,
    pub calibration_labels_are_required_to_be_matured_and_validated: bool,
    pub prediction_construction_does_not_receive_test_labels: bool,
    pub abstention_does_not_authorize_hold_release_or_process_control: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConformalReportV1 {
    pub schema: String,
    pub case_id: String,
    pub model_sha256: String,
    pub calibration_snapshot_sha256: String,
    pub calibration_cutoff_id: String,
    pub alpha: f64,
    pub training_sample_ids_sha256: String,
    pub calibration_sample_ids_sha256: String,
    pub calibration_label_ids_sha256: String,
    pub prediction_sample_ids_sha256: String,
    pub negative_threshold: ConformalClassThresholdV1,
    pub positive_threshold: ConformalClassThresholdV1,
    pub prediction_set_count: usize,
    pub prediction_sets_sha256: String,
    pub prediction_sets: Vec<ConformalPredictionSetV1>,
    pub claim_limits: ConformalClaimLimitsV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConformalTruthSampleV1 {
    pub sample_id: String,
    pub positive: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConformalClassEvaluationV1 {
    pub sample_count: usize,
    pub covered_count: usize,
    pub empirical_coverage: Option<f64>,
    pub singleton_count: usize,
    pub singleton_correct_count: usize,
    pub singleton_accuracy: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConformalEvaluationV1 {
    pub schema: String,
    pub prediction_sets_sha256: String,
    pub sample_count: usize,
    pub positive: ConformalClassEvaluationV1,
    pub negative: ConformalClassEvaluationV1,
    pub singleton_predictions: usize,
    pub correct_singleton_predictions: usize,
    pub singleton_accuracy: Option<f64>,
    pub abstention_count: usize,
    pub abstention_rate: f64,
    pub empty_set_count: usize,
    pub doubleton_set_count: usize,
    pub mean_set_size: f64,
    /// Empirical test coverage is diagnostic only; conformal construction did
    /// not consume these labels.
    pub empirical_marginal_coverage: f64,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct ConformalError(pub String);

fn require_text(value: &str, field: &str) -> Result<(), ConformalError> {
    if value.trim().is_empty() {
        Err(ConformalError(format!("{field}: expected non-empty string")))
    } else {
        Ok(())
    }
}

fn finite(value: f64, field: &str) -> Result<(), ConformalError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ConformalError(format!("{field}: expected finite number")))
    }
}

fn probability(value: f64, field: &str) -> Result<(), ConformalError> {
    finite(value, field)?;
    if !(0.0..=1.0).contains(&value) {
        Err(ConformalError(format!("{field}: expected value in [0, 1]")))
    } else {
        Ok(())
    }
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn ids_digest<'a>(ids: impl IntoIterator<Item = &'a str>) -> String {
    let mut ordered: Vec<&str> = ids.into_iter().collect();
    ordered.sort_unstable();
    sha256_hex(&serde_json::to_vec(&ordered).expect("sample ID serialization is infallible"))
}

fn validate_unique_ids(
    ids: impl IntoIterator<Item = String>,
    field: &str,
) -> Result<BTreeSet<String>, ConformalError> {
    let mut unique = BTreeSet::new();
    for (index, id) in ids.into_iter().enumerate() {
        require_text(&id, &format!("{field}[{index}]"))?;
        if !unique.insert(id.clone()) {
            return Err(ConformalError(format!("{field}[{index}]: duplicate identity")));
        }
    }
    Ok(unique)
}

fn class_threshold(
    calibration: &[ConformalCalibrationSampleV1],
    class_positive: bool,
    alpha: f64,
) -> Result<ConformalClassThresholdV1, ConformalError> {
    let mut nonconformity: Vec<f64> = calibration
        .iter()
        .filter(|sample| sample.positive == class_positive)
        .map(|sample| {
            if class_positive {
                1.0 - sample.score
            } else {
                sample.score
            }
        })
        .collect();
    if nonconformity.is_empty() {
        return Err(ConformalError(format!(
            "calibration: class {} has no matured calibration samples",
            if class_positive { "positive" } else { "negative" }
        )));
    }
    nonconformity.sort_by(f64::total_cmp);
    let count = nonconformity.len();
    let quantile_rank = (((count + 1) as f64) * (1.0 - alpha)).ceil() as usize;
    let unbounded_threshold = quantile_rank > count;
    let threshold = if unbounded_threshold {
        None
    } else {
        Some(nonconformity[quantile_rank - 1])
    };
    Ok(ConformalClassThresholdV1 {
        class_positive,
        calibration_count: count,
        quantile_rank,
        threshold,
        unbounded_threshold,
    })
}

fn threshold_contains(
    threshold: &ConformalClassThresholdV1,
    nonconformity: f64,
) -> bool {
    match threshold.threshold {
        Some(value) => nonconformity <= value,
        None => true,
    }
}

/// Builds class-conditional conformal prediction sets.
///
/// Calibration labels must be members of the supplied information snapshot's
/// matured validated-label set. The prediction samples have no label field.
pub fn build_mondrian_conformal_sets_v1(
    subject: &ConformalSubjectV1,
    maturity_snapshot: &InformationCutoffSnapshotV1,
) -> Result<ConformalReportV1, ConformalError> {
    if subject.schema != SUBJECT_SCHEMA_V1 {
        return Err(ConformalError("subject.schema: unsupported or missing".into()));
    }
    require_text(&subject.case_id, "case_id")?;
    if !valid_sha256(&subject.model_sha256) {
        return Err(ConformalError("model_sha256: expected lowercase SHA-256".into()));
    }
    finite(subject.alpha, "alpha")?;
    if !(0.0..1.0).contains(&subject.alpha) {
        return Err(ConformalError("alpha: expected value strictly between 0 and 1".into()));
    }
    if maturity_snapshot.schema != SNAPSHOT_SCHEMA_V1 {
        return Err(ConformalError(
            "maturity snapshot schema: unsupported or missing".into(),
        ));
    }
    require_text(&maturity_snapshot.graph_id, "maturity_snapshot.graph_id")?;
    require_text(&maturity_snapshot.cutoff.cutoff_id, "maturity_snapshot.cutoff.cutoff_id")?;
    if !valid_sha256(&maturity_snapshot.graph_sha256) {
        return Err(ConformalError(
            "maturity_snapshot.graph_sha256: expected lowercase SHA-256".into(),
        ));
    }
    if subject.training_sample_ids.is_empty()
        || subject.calibration_samples.is_empty()
        || subject.prediction_samples.is_empty()
    {
        return Err(ConformalError(
            "training, calibration and prediction partitions must be non-empty".into(),
        ));
    }

    let training_ids = validate_unique_ids(
        subject.training_sample_ids.clone(),
        "training_sample_ids",
    )?;
    let calibration_ids = validate_unique_ids(
        subject
            .calibration_samples
            .iter()
            .map(|sample| sample.sample_id.clone()),
        "calibration_sample_ids",
    )?;
    let prediction_ids = validate_unique_ids(
        subject
            .prediction_samples
            .iter()
            .map(|sample| sample.sample_id.clone()),
        "prediction_sample_ids",
    )?;

    if !training_ids.is_disjoint(&calibration_ids)
        || !training_ids.is_disjoint(&prediction_ids)
        || !calibration_ids.is_disjoint(&prediction_ids)
    {
        return Err(ConformalError(
            "training, calibration and prediction sample IDs must be pairwise disjoint".into(),
        ));
    }

    let matured_ids: BTreeSet<&str> = maturity_snapshot
        .matured_validated_label_ids
        .iter()
        .map(String::as_str)
        .collect();
    let mut calibration_label_ids = BTreeSet::new();
    for (index, sample) in subject.calibration_samples.iter().enumerate() {
        require_text(&sample.label_id, &format!("calibration_samples[{index}].label_id"))?;
        probability(sample.score, &format!("calibration_samples[{index}].score"))?;
        if !matured_ids.contains(sample.label_id.as_str()) {
            return Err(ConformalError(format!(
                "calibration_samples[{index}].label_id: label is not matured and validated at cutoff"
            )));
        }
        if !calibration_label_ids.insert(sample.label_id.as_str()) {
            return Err(ConformalError(format!(
                "calibration_samples[{index}].label_id: duplicate label identity"
            )));
        }
    }
    for (index, sample) in subject.prediction_samples.iter().enumerate() {
        probability(sample.score, &format!("prediction_samples[{index}].score"))?;
    }

    let negative_threshold =
        class_threshold(&subject.calibration_samples, false, subject.alpha)?;
    let positive_threshold =
        class_threshold(&subject.calibration_samples, true, subject.alpha)?;

    let mut ordered_predictions = subject.prediction_samples.clone();
    ordered_predictions.sort_by(|left, right| left.sample_id.cmp(&right.sample_id));
    let prediction_sets: Vec<ConformalPredictionSetV1> = ordered_predictions
        .iter()
        .map(|sample| {
            let negative_nonconformity = sample.score;
            let positive_nonconformity = 1.0 - sample.score;
            let negative_in_set =
                threshold_contains(&negative_threshold, negative_nonconformity);
            let positive_in_set =
                threshold_contains(&positive_threshold, positive_nonconformity);
            let set_size = usize::from(negative_in_set) + usize::from(positive_in_set);
            let decision = match (negative_in_set, positive_in_set) {
                (true, false) => SelectiveDecisionV1::PredictNegative,
                (false, true) => SelectiveDecisionV1::PredictPositive,
                _ => SelectiveDecisionV1::Abstain,
            };
            ConformalPredictionSetV1 {
                sample_id: sample.sample_id.clone(),
                score: sample.score,
                negative_in_set,
                positive_in_set,
                set_size,
                decision,
            }
        })
        .collect();

    let prediction_sets_sha256 = sha256_hex(
        &serde_json::to_vec(&prediction_sets)
            .map_err(|error| ConformalError(format!("prediction-set serialization failed: {error}")))?,
    );
    Ok(ConformalReportV1 {
        schema: REPORT_SCHEMA_V1.into(),
        case_id: subject.case_id.clone(),
        model_sha256: subject.model_sha256.clone(),
        calibration_snapshot_sha256: maturity_snapshot.graph_sha256.clone(),
        calibration_cutoff_id: maturity_snapshot.cutoff.cutoff_id.clone(),
        alpha: subject.alpha,
        training_sample_ids_sha256: ids_digest(training_ids.iter().map(String::as_str)),
        calibration_sample_ids_sha256: ids_digest(calibration_ids.iter().map(String::as_str)),
        calibration_label_ids_sha256: ids_digest(calibration_label_ids.iter().copied()),
        prediction_sample_ids_sha256: ids_digest(prediction_ids.iter().map(String::as_str)),
        negative_threshold,
        positive_threshold,
        prediction_set_count: prediction_sets.len(),
        prediction_sets_sha256,
        prediction_sets,
        claim_limits: ConformalClaimLimitsV1 {
            class_conditional_coverage_requires_exchangeability_within_each_class: true,
            guarantee_is_marginal_not_per_sample_conditional: true,
            arbitrary_temporal_tool_chamber_recipe_or_fab_shift_not_covered: true,
            calibration_labels_are_required_to_be_matured_and_validated: true,
            prediction_construction_does_not_receive_test_labels: true,
            abstention_does_not_authorize_hold_release_or_process_control: true,
        },
    })
}

/// Evaluates prediction sets against truth labels supplied only after set
/// construction. Truth cannot change the prediction-set digest.
pub fn evaluate_conformal_sets_v1(
    report: &ConformalReportV1,
    truth: &[ConformalTruthSampleV1],
) -> Result<ConformalEvaluationV1, ConformalError> {
    if report.schema != REPORT_SCHEMA_V1 {
        return Err(ConformalError("conformal report schema unsupported".into()));
    }
    if report.prediction_set_count != report.prediction_sets.len() {
        return Err(ConformalError(
            "prediction-set count does not match payload length".into(),
        ));
    }
    let serialized_sets = serde_json::to_vec(&report.prediction_sets)
        .map_err(|error| ConformalError(format!("prediction-set serialization failed: {error}")))?;
    if sha256_hex(&serialized_sets) != report.prediction_sets_sha256 {
        return Err(ConformalError(
            "prediction-set digest does not match payload".into(),
        ));
    }
    for (index, prediction) in report.prediction_sets.iter().enumerate() {
        let expected_size = usize::from(prediction.negative_in_set)
            + usize::from(prediction.positive_in_set);
        let expected_decision = match (prediction.negative_in_set, prediction.positive_in_set) {
            (true, false) => SelectiveDecisionV1::PredictNegative,
            (false, true) => SelectiveDecisionV1::PredictPositive,
            _ => SelectiveDecisionV1::Abstain,
        };
        if prediction.set_size != expected_size || prediction.decision != expected_decision {
            return Err(ConformalError(format!(
                "prediction_sets[{index}]: set size or decision contradicts membership flags"
            )));
        }
        probability(prediction.score, &format!("prediction_sets[{index}].score"))?;
    }
    let mut ordered_truth = truth.to_vec();
    ordered_truth.sort_by(|left, right| left.sample_id.cmp(&right.sample_id));
    if ordered_truth.len() != report.prediction_sets.len() {
        return Err(ConformalError(
            "prediction-set/truth sample counts differ".into(),
        ));
    }
    let mut seen = BTreeSet::new();
    let mut positive_support = 0usize;
    let mut positive_covered = 0usize;
    let mut positive_singleton = 0usize;
    let mut positive_singleton_correct = 0usize;
    let mut negative_support = 0usize;
    let mut negative_covered = 0usize;
    let mut negative_singleton = 0usize;
    let mut negative_singleton_correct = 0usize;
    let mut singleton_predictions = 0usize;
    let mut correct_singleton_predictions = 0usize;
    let mut abstention_count = 0usize;
    let mut empty_set_count = 0usize;
    let mut doubleton_set_count = 0usize;
    let mut set_size_total = 0usize;
    let mut marginal_covered = 0usize;

    for (index, (prediction, actual)) in report
        .prediction_sets
        .iter()
        .zip(&ordered_truth)
        .enumerate()
    {
        require_text(&actual.sample_id, &format!("truth[{index}].sample_id"))?;
        if !seen.insert(actual.sample_id.as_str()) {
            return Err(ConformalError(format!(
                "truth[{index}].sample_id: duplicate"
            )));
        }
        if prediction.sample_id != actual.sample_id {
            return Err(ConformalError(format!(
                "prediction/truth identity mismatch at sorted row {index}"
            )));
        }
        set_size_total += prediction.set_size;
        if prediction.set_size == 0 {
            empty_set_count += 1;
        } else if prediction.set_size == 2 {
            doubleton_set_count += 1;
        }
        if actual.positive {
            positive_support += 1;
            if prediction.positive_in_set {
                positive_covered += 1;
                marginal_covered += 1;
            }
            if prediction.set_size == 1 {
                positive_singleton += 1;
                if matches!(prediction.decision, SelectiveDecisionV1::PredictPositive) {
                    positive_singleton_correct += 1;
                }
            }
        } else {
            negative_support += 1;
            if prediction.negative_in_set {
                negative_covered += 1;
                marginal_covered += 1;
            }
            if prediction.set_size == 1 {
                negative_singleton += 1;
                if matches!(prediction.decision, SelectiveDecisionV1::PredictNegative) {
                    negative_singleton_correct += 1;
                }
            }
        }
        if prediction.set_size == 1 {
            singleton_predictions += 1;
            let correct = match prediction.decision {
                SelectiveDecisionV1::PredictPositive => actual.positive,
                SelectiveDecisionV1::PredictNegative => !actual.positive,
                SelectiveDecisionV1::Abstain => false,
            };
            if correct {
                correct_singleton_predictions += 1;
            }
        } else {
            abstention_count += 1;
        }
    }
    if positive_support == 0 || negative_support == 0 {
        return Err(ConformalError(
            "evaluation truth must contain both classes".into(),
        ));
    }
    let sample_count = report.prediction_sets.len();
    let percentage = |numerator: usize, denominator: usize| -> f64 {
        numerator as f64 / denominator as f64
    };
    let positive = ConformalClassEvaluationV1 {
        sample_count: positive_support,
        covered_count: positive_covered,
        empirical_coverage: Some(percentage(positive_covered, positive_support)),
        singleton_count: positive_singleton,
        singleton_correct_count: positive_singleton_correct,
        singleton_accuracy: (positive_singleton > 0)
            .then(|| percentage(positive_singleton_correct, positive_singleton)),
    };
    let negative = ConformalClassEvaluationV1 {
        sample_count: negative_support,
        covered_count: negative_covered,
        empirical_coverage: Some(percentage(negative_covered, negative_support)),
        singleton_count: negative_singleton,
        singleton_correct_count: negative_singleton_correct,
        singleton_accuracy: (negative_singleton > 0)
            .then(|| percentage(negative_singleton_correct, negative_singleton)),
    };
    Ok(ConformalEvaluationV1 {
        schema: EVALUATION_SCHEMA_V1.into(),
        prediction_sets_sha256: report.prediction_sets_sha256.clone(),
        sample_count,
        positive,
        negative,
        singleton_predictions,
        correct_singleton_predictions,
        singleton_accuracy: (singleton_predictions > 0)
            .then(|| percentage(correct_singleton_predictions, singleton_predictions)),
        abstention_count,
        abstention_rate: percentage(abstention_count, sample_count),
        empty_set_count,
        doubleton_set_count,
        mean_set_size: set_size_total as f64 / sample_count as f64,
        empirical_marginal_coverage: percentage(marginal_covered, sample_count),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::process_event_graph::{
        ClockBasisV1, PredictionInformationCutoffV1,
    };

    const MODEL_SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const GRAPH_SHA: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn snapshot(label_ids: &[&str]) -> InformationCutoffSnapshotV1 {
        InformationCutoffSnapshotV1 {
            schema: SNAPSHOT_SCHEMA_V1.into(),
            graph_id: "graph-1".into(),
            graph_sha256: GRAPH_SHA.into(),
            cutoff: PredictionInformationCutoffV1 {
                cutoff_id: "cutoff-1".into(),
                cutoff_time: "2026-11-22T12:00:00".into(),
                clock_domain_id: "fab-local".into(),
                clock_basis: ClockBasisV1::SourceLocalNaive,
            },
            available_event_ids: vec![],
            unavailable_event_ids: vec![],
            unaligned_event_ids: vec![],
            future_event_ids: vec![],
            matured_validated_label_ids: label_ids.iter().map(|id| (*id).into()).collect(),
            not_yet_available_label_ids: vec![],
            non_trainable_label_ids: vec![],
            unaligned_label_ids: vec![],
            claim_limits: vec![],
        }
    }

    fn calibration() -> Vec<ConformalCalibrationSampleV1> {
        (0..20)
            .map(|index| {
                let positive = index < 5;
                ConformalCalibrationSampleV1 {
                    sample_id: format!("cal-{index:02}"),
                    label_id: format!("label-{index:02}"),
                    positive,
                    score: if positive {
                        0.75 + (index as f64 * 0.01)
                    } else {
                        0.15 + (index as f64 * 0.005)
                    },
                }
            })
            .collect()
    }

    fn subject() -> ConformalSubjectV1 {
        ConformalSubjectV1 {
            schema: SUBJECT_SCHEMA_V1.into(),
            case_id: "CONFORMAL-KA-001".into(),
            model_sha256: MODEL_SHA.into(),
            alpha: 0.2,
            training_sample_ids: (0..30).map(|i| format!("train-{i:02}")).collect(),
            calibration_samples: calibration(),
            prediction_samples: vec![
                ConformalPredictionSampleV1 {
                    sample_id: "test-a".into(),
                    score: 0.95,
                },
                ConformalPredictionSampleV1 {
                    sample_id: "test-b".into(),
                    score: 0.05,
                },
                ConformalPredictionSampleV1 {
                    sample_id: "test-c".into(),
                    score: 0.50,
                },
            ],
        }
    }

    #[test]
    fn prediction_sets_are_deterministic_and_sample_order_invariant() {
        let input = subject();
        let snapshot = snapshot(
            &(0..20)
                .map(|i| format!("label-{i:02}"))
                .collect::<Vec<_>>()
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
        );
        let first = build_mondrian_conformal_sets_v1(&input, &snapshot).unwrap();
        let mut reversed = subject();
        reversed.calibration_samples.reverse();
        reversed.prediction_samples.reverse();
        reversed.training_sample_ids.reverse();
        let second = build_mondrian_conformal_sets_v1(&reversed, &snapshot).unwrap();
        assert_eq!(first.prediction_sets_sha256, second.prediction_sets_sha256);
        assert_eq!(first.prediction_sets, second.prediction_sets);
    }

    #[test]
    fn immature_or_unvalidated_calibration_label_is_rejected() {
        let mut input = subject();
        input.calibration_samples[0].label_id = "not-matured".into();
        assert!(build_mondrian_conformal_sets_v1(&input, &snapshot(&["other-label"])).is_err());
    }

    #[test]
    fn training_calibration_and_prediction_partitions_must_be_disjoint() {
        let mut input = subject();
        input.calibration_samples[0].sample_id = "train-00".into();
        let ids: Vec<String> = (0..20).map(|i| format!("label-{i:02}")).collect();
        let ids_ref: Vec<&str> = ids.iter().map(String::as_str).collect();
        assert!(build_mondrian_conformal_sets_v1(&input, &snapshot(&ids_ref)).is_err());
    }

    #[test]
    fn both_mondrian_classes_need_calibration_support() {
        let mut input = subject();
        for sample in &mut input.calibration_samples {
            sample.positive = false;
        }
        let ids: Vec<String> = input.calibration_samples.iter().map(|s| s.label_id.clone()).collect();
        let ids_ref: Vec<&str> = ids.iter().map(String::as_str).collect();
        assert!(build_mondrian_conformal_sets_v1(&input, &snapshot(&ids_ref)).is_err());
    }

    #[test]
    fn small_class_support_uses_unbounded_finite_sample_rank() {
        let mut input = subject();
        input.alpha = 0.1;
        input.calibration_samples = vec![
            ConformalCalibrationSampleV1 {
                sample_id: "cal-neg".into(),
                label_id: "label-neg".into(),
                positive: false,
                score: 0.2,
            },
            ConformalCalibrationSampleV1 {
                sample_id: "cal-pos".into(),
                label_id: "label-pos".into(),
                positive: true,
                score: 0.8,
            },
        ];
        let report = build_mondrian_conformal_sets_v1(
            &input,
            &snapshot(&["label-neg", "label-pos"]),
        )
        .unwrap();
        assert!(report.negative_threshold.unbounded_threshold);
        assert!(report.positive_threshold.unbounded_threshold);
        assert!(report.prediction_sets.iter().all(|row| row.set_size == 2));
        assert!(report.prediction_sets.iter().all(|row| matches!(row.decision, SelectiveDecisionV1::Abstain)));
    }

    #[test]
    fn test_truth_changes_evaluation_not_prediction_set_digest() {
        let input = subject();
        let label_ids: Vec<String> = (0..20)
            .map(|i| format!("label-{i:02}"))
            .collect();
        let labels: Vec<&str> = label_ids.iter().map(String::as_str).collect();
        let report = build_mondrian_conformal_sets_v1(&input, &snapshot(&labels)).unwrap();
        let truth_a = vec![
            ConformalTruthSampleV1 { sample_id: "test-a".into(), positive: true },
            ConformalTruthSampleV1 { sample_id: "test-b".into(), positive: false },
            ConformalTruthSampleV1 { sample_id: "test-c".into(), positive: true },
        ];
        let truth_b = vec![
            ConformalTruthSampleV1 { sample_id: "test-a".into(), positive: false },
            ConformalTruthSampleV1 { sample_id: "test-b".into(), positive: true },
            ConformalTruthSampleV1 { sample_id: "test-c".into(), positive: false },
        ];
        let first = evaluate_conformal_sets_v1(&report, &truth_a).unwrap();
        let second = evaluate_conformal_sets_v1(&report, &truth_b).unwrap();
        assert_eq!(first.prediction_sets_sha256, second.prediction_sets_sha256);
        assert_ne!(first.empirical_marginal_coverage, second.empirical_marginal_coverage);
    }

    #[test]
    fn invalid_scores_and_snapshot_digest_reject() {
        let mut input = subject();
        input.prediction_samples[0].score = 1.1;
        let ids: Vec<String> = (0..20).map(|i| format!("label-{i:02}")).collect();
        let ids_ref: Vec<&str> = ids.iter().map(String::as_str).collect();
        assert!(build_mondrian_conformal_sets_v1(&input, &snapshot(&ids_ref)).is_err());

        let mut bad_snapshot = snapshot(&ids_ref);
        bad_snapshot.graph_sha256 = "not-a-sha".into();
        assert!(build_mondrian_conformal_sets_v1(&subject(), &bad_snapshot).is_err());
    }
}
