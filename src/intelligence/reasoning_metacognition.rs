// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! RQ-006 metacognition qualification primitives.
//!
//! This module measures externally auditable metacognitive behavior. It does not define how a
//! reasoner should generate confidence and does not treat confidence as evidence merely because
//! it lies in `[0, 1]`.
//!
//! The evaluator keeps separate:
//! - calibration of predicted correctness,
//! - discrimination between correct and incorrect episodes,
//! - selective risk / abstention behavior,
//! - detection of weak assumptions,
//! - confidence revision after new evidence, and
//! - whether a confidence report is actually bound to the current episode.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;

pub const METACOGNITION_EVALUATOR_VERSION: &str = "rq-006-metacognition-v1";
const LOG_LOSS_EPSILON: f64 = 1.0e-15;

/// One pre-outcome prediction of whether the subject's answer is correct.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorrectnessPrediction {
    /// Episode whose correctness is being evaluated.
    pub episode_id: String,
    /// Episode the confidence value claims to describe. A mismatch exposes stale/cross-episode
    /// confidence rather than silently attributing it to the current episode.
    pub confidence_target_episode_id: String,
    /// Predicted probability that an asserted answer would be correct.
    pub confidence: f64,
    /// Ground-truth correctness supplied by the benchmark after the prediction is frozen.
    pub correct: bool,
    /// Whether the subject actually asserted an answer rather than abstaining.
    pub asserted: bool,
}

/// Binary ground truth for whether a materially weak assumption was present and detected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeakAssumptionObservation {
    pub episode_id: String,
    pub weak_assumption_present: bool,
    pub weak_assumption_detected: bool,
}

/// Direction in which confidence should move after independently specified new evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfidenceRevisionDirection {
    Increase,
    Decrease,
    Stable,
}

/// Paired confidence observation before and after new evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfidenceRevisionObservation {
    pub pair_id: String,
    pub before_episode_id: String,
    pub after_episode_id: String,
    pub before_confidence: f64,
    pub after_confidence: f64,
    pub expected_direction: ConfidenceRevisionDirection,
    /// Absolute delta considered observationally stable for `Stable` expectations.
    pub stable_tolerance: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SelectiveRiskPoint {
    pub threshold: f64,
    /// Fraction of all evaluated episodes on which confidence meets the threshold and the subject
    /// actually asserted an answer.
    pub coverage: f64,
    /// Error rate conditional on selection. `None` when no episode is selected.
    pub risk: Option<f64>,
    pub selected: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinaryDetectionReport {
    pub observations: usize,
    pub true_positive: usize,
    pub false_positive: usize,
    pub true_negative: usize,
    pub false_negative: usize,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f1: Option<f64>,
    pub accuracy: Option<f64>,
}

/// Decomposed RQ-006 measurement report. There is intentionally no single "metacognition score".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetacognitionReport {
    pub evaluator_version: String,
    pub predictions: usize,
    pub empirical_accuracy: Option<f64>,
    pub mean_confidence: Option<f64>,
    pub brier_score: Option<f64>,
    pub log_loss: Option<f64>,
    pub expected_calibration_error: Option<f64>,
    pub maximum_calibration_error: Option<f64>,
    /// Mean confidence minus empirical accuracy. Positive values indicate overconfidence.
    pub confidence_bias: Option<f64>,
    /// Rank discrimination: probability that a randomly chosen correct episode receives higher
    /// confidence than a randomly chosen incorrect episode, with ties worth 0.5.
    pub correctness_auroc: Option<f64>,
    /// Fraction whose confidence target is the episode currently being evaluated.
    pub current_episode_binding_rate: Option<f64>,
    /// Actual assertion rate, independent of correctness.
    pub assertion_rate: Option<f64>,
    /// Error rate among actually asserted answers.
    pub asserted_risk: Option<f64>,
    /// Fraction of abstentions that would have been correct if answered; useful for distinguishing
    /// prudent abstention from indiscriminate refusal on benchmark tasks with known answers.
    pub abstention_opportunity_cost: Option<f64>,
    pub selective_risk: Vec<SelectiveRiskPoint>,
    pub weak_assumption_detection: BinaryDetectionReport,
    pub confidence_revisions: usize,
    pub revision_direction_accuracy: Option<f64>,
    /// Mean signed movement in the expected direction. Positive is favorable; stable pairs use
    /// negative excess movement outside tolerance as their signed contribution.
    pub mean_expected_revision_delta: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MetacognitionEvaluationError {
    InvalidBinCount(usize),
    EmptyEpisodeId(&'static str),
    DuplicatePredictionEpisode(String),
    DuplicateAssumptionEpisode(String),
    DuplicateRevisionPair(String),
    InvalidProbability { field: &'static str, value: f64 },
    InvalidTolerance(f64),
    InvalidThreshold(f64),
}

impl fmt::Display for MetacognitionEvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBinCount(count) => write!(f, "calibration bin count must be in 1..=100, got {count}"),
            Self::EmptyEpisodeId(field) => write!(f, "required episode identifier `{field}` is empty"),
            Self::DuplicatePredictionEpisode(id) => {
                write!(f, "episode `{id}` has more than one correctness prediction")
            }
            Self::DuplicateAssumptionEpisode(id) => {
                write!(f, "episode `{id}` has more than one weak-assumption observation")
            }
            Self::DuplicateRevisionPair(id) => write!(f, "revision pair `{id}` appears more than once"),
            Self::InvalidProbability { field, value } => {
                write!(f, "probability `{field}` must be finite and within [0, 1], got {value}")
            }
            Self::InvalidTolerance(value) => {
                write!(f, "stable tolerance must be finite and within [0, 1], got {value}")
            }
            Self::InvalidThreshold(value) => {
                write!(f, "selective-risk threshold must be finite and within [0, 1], got {value}")
            }
        }
    }
}

impl std::error::Error for MetacognitionEvaluationError {}

/// Evaluate a frozen set of metacognitive observations.
pub fn evaluate_metacognition(
    predictions: &[CorrectnessPrediction],
    assumptions: &[WeakAssumptionObservation],
    revisions: &[ConfidenceRevisionObservation],
    calibration_bins: usize,
    selective_thresholds: &[f64],
) -> Result<MetacognitionReport, MetacognitionEvaluationError> {
    if !(1..=100).contains(&calibration_bins) {
        return Err(MetacognitionEvaluationError::InvalidBinCount(calibration_bins));
    }
    validate_predictions(predictions)?;
    validate_assumptions(assumptions)?;
    validate_revisions(revisions)?;
    for &threshold in selective_thresholds {
        validate_probability("selective_threshold", threshold)
            .map_err(|_| MetacognitionEvaluationError::InvalidThreshold(threshold))?;
    }

    let n = predictions.len();
    let empirical_accuracy = mean_bool(predictions.iter().map(|p| p.correct));
    let mean_confidence = mean_f64(predictions.iter().map(|p| p.confidence));

    let (brier_score, log_loss) = if predictions.is_empty() {
        (None, None)
    } else {
        let mut brier = 0.0;
        let mut log = 0.0;
        for prediction in predictions {
            let target = if prediction.correct { 1.0 } else { 0.0 };
            brier += (prediction.confidence - target).powi(2);
            let p_correct = if prediction.correct {
                prediction.confidence
            } else {
                1.0 - prediction.confidence
            };
            log += -p_correct
                .clamp(LOG_LOSS_EPSILON, 1.0 - LOG_LOSS_EPSILON)
                .ln();
        }
        (Some(brier / n as f64), Some(log / n as f64))
    };

    let (ece, mce) = calibration_errors(predictions, calibration_bins);
    let confidence_bias = match (mean_confidence, empirical_accuracy) {
        (Some(confidence), Some(accuracy)) => Some(confidence - accuracy),
        _ => None,
    };
    let correctness_auroc = auroc(predictions);
    let current_episode_binding_rate = mean_bool(
        predictions
            .iter()
            .map(|p| p.episode_id == p.confidence_target_episode_id),
    );
    let assertion_rate = mean_bool(predictions.iter().map(|p| p.asserted));
    let asserted_risk = conditional_rate(predictions.iter().filter(|p| p.asserted), |p| !p.correct);
    let abstention_opportunity_cost =
        conditional_rate(predictions.iter().filter(|p| !p.asserted), |p| p.correct);

    let selective_risk = selective_thresholds
        .iter()
        .map(|&threshold| {
            let selected: Vec<&CorrectnessPrediction> = predictions
                .iter()
                .filter(|p| p.asserted && p.confidence >= threshold)
                .collect();
            let selected_n = selected.len();
            let errors = selected.iter().filter(|p| !p.correct).count();
            SelectiveRiskPoint {
                threshold,
                coverage: if n == 0 {
                    0.0
                } else {
                    selected_n as f64 / n as f64
                },
                risk: ratio(errors, selected_n),
                selected: selected_n,
            }
        })
        .collect();

    let weak_assumption_detection = detection_report(assumptions);
    let (revision_direction_accuracy, mean_expected_revision_delta) = revision_metrics(revisions);

    Ok(MetacognitionReport {
        evaluator_version: METACOGNITION_EVALUATOR_VERSION.into(),
        predictions: n,
        empirical_accuracy,
        mean_confidence,
        brier_score,
        log_loss,
        expected_calibration_error: ece,
        maximum_calibration_error: mce,
        confidence_bias,
        correctness_auroc,
        current_episode_binding_rate,
        assertion_rate,
        asserted_risk,
        abstention_opportunity_cost,
        selective_risk,
        weak_assumption_detection,
        confidence_revisions: revisions.len(),
        revision_direction_accuracy,
        mean_expected_revision_delta,
    })
}

fn validate_predictions(predictions: &[CorrectnessPrediction]) -> Result<(), MetacognitionEvaluationError> {
    let mut seen = HashSet::new();
    for prediction in predictions {
        if prediction.episode_id.trim().is_empty() {
            return Err(MetacognitionEvaluationError::EmptyEpisodeId("episode_id"));
        }
        if prediction.confidence_target_episode_id.trim().is_empty() {
            return Err(MetacognitionEvaluationError::EmptyEpisodeId(
                "confidence_target_episode_id",
            ));
        }
        validate_probability("confidence", prediction.confidence)?;
        if !seen.insert(prediction.episode_id.as_str()) {
            return Err(MetacognitionEvaluationError::DuplicatePredictionEpisode(
                prediction.episode_id.clone(),
            ));
        }
    }
    Ok(())
}

fn validate_assumptions(
    assumptions: &[WeakAssumptionObservation],
) -> Result<(), MetacognitionEvaluationError> {
    let mut seen = HashSet::new();
    for observation in assumptions {
        if observation.episode_id.trim().is_empty() {
            return Err(MetacognitionEvaluationError::EmptyEpisodeId("episode_id"));
        }
        if !seen.insert(observation.episode_id.as_str()) {
            return Err(MetacognitionEvaluationError::DuplicateAssumptionEpisode(
                observation.episode_id.clone(),
            ));
        }
    }
    Ok(())
}

fn validate_revisions(
    revisions: &[ConfidenceRevisionObservation],
) -> Result<(), MetacognitionEvaluationError> {
    let mut seen = HashSet::new();
    for observation in revisions {
        if observation.pair_id.trim().is_empty() {
            return Err(MetacognitionEvaluationError::EmptyEpisodeId("pair_id"));
        }
        if observation.before_episode_id.trim().is_empty() {
            return Err(MetacognitionEvaluationError::EmptyEpisodeId(
                "before_episode_id",
            ));
        }
        if observation.after_episode_id.trim().is_empty() {
            return Err(MetacognitionEvaluationError::EmptyEpisodeId(
                "after_episode_id",
            ));
        }
        if !seen.insert(observation.pair_id.as_str()) {
            return Err(MetacognitionEvaluationError::DuplicateRevisionPair(
                observation.pair_id.clone(),
            ));
        }
        validate_probability("before_confidence", observation.before_confidence)?;
        validate_probability("after_confidence", observation.after_confidence)?;
        if !observation.stable_tolerance.is_finite()
            || !(0.0..=1.0).contains(&observation.stable_tolerance)
        {
            return Err(MetacognitionEvaluationError::InvalidTolerance(
                observation.stable_tolerance,
            ));
        }
    }
    Ok(())
}

fn validate_probability(field: &'static str, value: f64) -> Result<(), MetacognitionEvaluationError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(MetacognitionEvaluationError::InvalidProbability { field, value })
    }
}

fn calibration_errors(
    predictions: &[CorrectnessPrediction],
    bins: usize,
) -> (Option<f64>, Option<f64>) {
    if predictions.is_empty() {
        return (None, None);
    }
    let mut confidence_sum = vec![0.0; bins];
    let mut correct_sum = vec![0usize; bins];
    let mut count = vec![0usize; bins];
    for prediction in predictions {
        let index = ((prediction.confidence * bins as f64).floor() as usize).min(bins - 1);
        confidence_sum[index] += prediction.confidence;
        correct_sum[index] += usize::from(prediction.correct);
        count[index] += 1;
    }

    let mut ece = 0.0;
    let mut mce: f64 = 0.0;
    for index in 0..bins {
        if count[index] == 0 {
            continue;
        }
        let mean_confidence = confidence_sum[index] / count[index] as f64;
        let accuracy = correct_sum[index] as f64 / count[index] as f64;
        let gap = (mean_confidence - accuracy).abs();
        ece += gap * count[index] as f64 / predictions.len() as f64;
        mce = mce.max(gap);
    }
    (Some(ece), Some(mce))
}

fn auroc(predictions: &[CorrectnessPrediction]) -> Option<f64> {
    let positives: Vec<&CorrectnessPrediction> = predictions.iter().filter(|p| p.correct).collect();
    let negatives: Vec<&CorrectnessPrediction> = predictions.iter().filter(|p| !p.correct).collect();
    if positives.is_empty() || negatives.is_empty() {
        return None;
    }
    let mut wins = 0.0;
    let mut pairs = 0usize;
    for positive in &positives {
        for negative in &negatives {
            pairs += 1;
            if positive.confidence > negative.confidence {
                wins += 1.0;
            } else if positive.confidence == negative.confidence {
                wins += 0.5;
            }
        }
    }
    Some(wins / pairs as f64)
}

fn detection_report(observations: &[WeakAssumptionObservation]) -> BinaryDetectionReport {
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut tn = 0usize;
    let mut fn_ = 0usize;
    for observation in observations {
        match (
            observation.weak_assumption_present,
            observation.weak_assumption_detected,
        ) {
            (true, true) => tp += 1,
            (false, true) => fp += 1,
            (false, false) => tn += 1,
            (true, false) => fn_ += 1,
        }
    }
    let precision = ratio(tp, tp + fp);
    let recall = ratio(tp, tp + fn_);
    let f1 = match (precision, recall) {
        (Some(p), Some(r)) if p + r > 0.0 => Some(2.0 * p * r / (p + r)),
        _ => None,
    };
    BinaryDetectionReport {
        observations: observations.len(),
        true_positive: tp,
        false_positive: fp,
        true_negative: tn,
        false_negative: fn_,
        precision,
        recall,
        f1,
        accuracy: ratio(tp + tn, observations.len()),
    }
}

fn revision_metrics(revisions: &[ConfidenceRevisionObservation]) -> (Option<f64>, Option<f64>) {
    if revisions.is_empty() {
        return (None, None);
    }
    let mut correct_direction = 0usize;
    let mut signed_total = 0.0;
    for revision in revisions {
        let delta = revision.after_confidence - revision.before_confidence;
        let (correct, signed) = match revision.expected_direction {
            ConfidenceRevisionDirection::Increase => (delta > 0.0, delta),
            ConfidenceRevisionDirection::Decrease => (delta < 0.0, -delta),
            ConfidenceRevisionDirection::Stable => {
                let excess = delta.abs() - revision.stable_tolerance;
                (excess <= 0.0, -excess.max(0.0))
            }
        };
        if correct {
            correct_direction += 1;
        }
        signed_total += signed;
    }
    (
        Some(correct_direction as f64 / revisions.len() as f64),
        Some(signed_total / revisions.len() as f64),
    )
}

fn conditional_rate<'a, I, F>(items: I, predicate: F) -> Option<f64>
where
    I: IntoIterator<Item = &'a CorrectnessPrediction>,
    F: Fn(&CorrectnessPrediction) -> bool,
{
    let mut count = 0usize;
    let mut matched = 0usize;
    for item in items {
        count += 1;
        if predicate(item) {
            matched += 1;
        }
    }
    ratio(matched, count)
}

fn mean_bool<I>(values: I) -> Option<f64>
where
    I: IntoIterator<Item = bool>,
{
    let mut count = 0usize;
    let mut sum = 0usize;
    for value in values {
        count += 1;
        sum += usize::from(value);
    }
    ratio(sum, count)
}

fn mean_f64<I>(values: I) -> Option<f64>
where
    I: IntoIterator<Item = f64>,
{
    let mut count = 0usize;
    let mut sum = 0.0;
    for value in values {
        count += 1;
        sum += value;
    }
    (count != 0).then(|| sum / count as f64)
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator != 0).then(|| numerator as f64 / denominator as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prediction(id: &str, confidence: f64, correct: bool, asserted: bool) -> CorrectnessPrediction {
        CorrectnessPrediction {
            episode_id: id.into(),
            confidence_target_episode_id: id.into(),
            confidence,
            correct,
            asserted,
        }
    }

    #[test]
    fn perfect_predictions_are_perfectly_calibrated_and_discriminative() {
        let observations = vec![
            prediction("p1", 1.0, true, true),
            prediction("p2", 1.0, true, true),
            prediction("p3", 0.0, false, false),
            prediction("p4", 0.0, false, false),
        ];
        let report = evaluate_metacognition(&observations, &[], &[], 10, &[0.5, 0.9])
            .unwrap_or_else(|err| panic!("evaluation must succeed: {err}"));
        assert_eq!(report.brier_score, Some(0.0));
        assert_eq!(report.expected_calibration_error, Some(0.0));
        assert_eq!(report.correctness_auroc, Some(1.0));
        assert_eq!(report.current_episode_binding_rate, Some(1.0));
        assert_eq!(report.asserted_risk, Some(0.0));
        assert_eq!(report.abstention_opportunity_cost, Some(0.0));
    }

    #[test]
    fn constant_confidence_has_no_discrimination() {
        let observations = vec![
            prediction("p1", 0.7, true, true),
            prediction("p2", 0.7, false, true),
        ];
        let report = evaluate_metacognition(&observations, &[], &[], 10, &[])
            .unwrap_or_else(|err| panic!("evaluation must succeed: {err}"));
        assert_eq!(report.correctness_auroc, Some(0.5));
        assert!((report.brier_score.unwrap_or_default() - 0.29).abs() < 1.0e-12);
        assert!((report.confidence_bias.unwrap_or_default() - 0.2).abs() < 1.0e-12);
    }

    #[test]
    fn stale_confidence_binding_is_measured_not_hidden() {
        let mut observations = vec![prediction("p1", 0.8, true, true)];
        observations[0].confidence_target_episode_id = "previous-episode".into();
        let report = evaluate_metacognition(&observations, &[], &[], 10, &[])
            .unwrap_or_else(|err| panic!("evaluation must succeed: {err}"));
        assert_eq!(report.current_episode_binding_rate, Some(0.0));
    }

    #[test]
    fn weak_assumption_metrics_keep_false_positives_and_false_negatives_visible() {
        let observations = vec![
            WeakAssumptionObservation {
                episode_id: "a".into(),
                weak_assumption_present: true,
                weak_assumption_detected: true,
            },
            WeakAssumptionObservation {
                episode_id: "b".into(),
                weak_assumption_present: true,
                weak_assumption_detected: false,
            },
            WeakAssumptionObservation {
                episode_id: "c".into(),
                weak_assumption_present: false,
                weak_assumption_detected: true,
            },
            WeakAssumptionObservation {
                episode_id: "d".into(),
                weak_assumption_present: false,
                weak_assumption_detected: false,
            },
        ];
        let report = evaluate_metacognition(&[], &observations, &[], 10, &[])
            .unwrap_or_else(|err| panic!("evaluation must succeed: {err}"));
        assert_eq!(report.weak_assumption_detection.true_positive, 1);
        assert_eq!(report.weak_assumption_detection.false_positive, 1);
        assert_eq!(report.weak_assumption_detection.false_negative, 1);
        assert_eq!(report.weak_assumption_detection.true_negative, 1);
        assert_eq!(report.weak_assumption_detection.accuracy, Some(0.5));
    }

    #[test]
    fn evidence_revision_scores_direction_not_storytelling() {
        let revisions = vec![
            ConfidenceRevisionObservation {
                pair_id: "contradiction".into(),
                before_episode_id: "p1-before".into(),
                after_episode_id: "p1-after".into(),
                before_confidence: 0.9,
                after_confidence: 0.4,
                expected_direction: ConfidenceRevisionDirection::Decrease,
                stable_tolerance: 0.01,
            },
            ConfidenceRevisionObservation {
                pair_id: "confirmation".into(),
                before_episode_id: "p2-before".into(),
                after_episode_id: "p2-after".into(),
                before_confidence: 0.4,
                after_confidence: 0.7,
                expected_direction: ConfidenceRevisionDirection::Increase,
                stable_tolerance: 0.01,
            },
        ];
        let report = evaluate_metacognition(&[], &[], &revisions, 10, &[])
            .unwrap_or_else(|err| panic!("evaluation must succeed: {err}"));
        assert_eq!(report.revision_direction_accuracy, Some(1.0));
        assert!((report.mean_expected_revision_delta.unwrap_or_default() - 0.4).abs() < 1.0e-12);
    }

    #[test]
    fn selective_risk_reports_coverage_and_errors_separately() {
        let observations = vec![
            prediction("p1", 0.95, true, true),
            prediction("p2", 0.90, false, true),
            prediction("p3", 0.60, true, true),
            prediction("p4", 0.40, false, false),
        ];
        let report = evaluate_metacognition(&observations, &[], &[], 10, &[0.5, 0.9])
            .unwrap_or_else(|err| panic!("evaluation must succeed: {err}"));
        assert_eq!(report.selective_risk[0].selected, 3);
        assert_eq!(report.selective_risk[0].coverage, 0.75);
        assert!((report.selective_risk[0].risk.unwrap_or_default() - 1.0 / 3.0).abs() < 1.0e-12);
        assert_eq!(report.selective_risk[1].selected, 2);
        assert_eq!(report.selective_risk[1].risk, Some(0.5));
    }

    #[test]
    fn invalid_probabilities_and_duplicate_episode_predictions_fail_closed() {
        let invalid = prediction("p1", f64::NAN, true, true);
        assert!(matches!(
            evaluate_metacognition(&[invalid], &[], &[], 10, &[]),
            Err(MetacognitionEvaluationError::InvalidProbability { .. })
        ));

        let duplicate = prediction("same", 0.5, true, true);
        assert!(matches!(
            evaluate_metacognition(&[duplicate.clone(), duplicate], &[], &[], 10, &[]),
            Err(MetacognitionEvaluationError::DuplicatePredictionEpisode(_))
        ));
    }
}
