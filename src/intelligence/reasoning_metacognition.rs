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
use std::collections::{BTreeMap, HashSet};
use std::fmt;

pub const METACOGNITION_EVALUATOR_VERSION: &str = "rq-006-metacognition-v4";
const LOG_LOSS_EPSILON: f64 = 1.0e-15;
const SELECTIVE_RISK_FAMILYWISE_ALPHA: f64 = 0.05;
const SELECTIVE_RISK_BOUND_METHOD: &str = "hoeffding-familywise-95-v1";
const SELECTIVE_RISK_BOUND_ASSUMPTIONS: &str = concat!(
    "IID evaluation episodes; frozen scoring and selection rule; ",
    "supplied threshold set predeclared before correctness outcomes; ",
    "task-family taxonomy and equal-width confidence bins predeclared and outcome-independent; ",
    "simultaneous bounds cover pooled and observed task-family by threshold and bin comparisons; ",
    "no distribution-shift guarantee",
);

/// One pre-outcome prediction of whether the subject's answer is correct.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorrectnessPrediction {
    /// Episode whose correctness is being evaluated.
    pub episode_id: String,
    /// Stable task-family identifier from a taxonomy frozen before outcomes are inspected.
    /// Do not assign or revise this label using correctness outcomes.
    pub task_family_id: String,
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
    /// Empirical error rate conditional on selection. `None` when no episode is selected.
    pub risk: Option<f64>,
    /// Conservative one-sided 95% family-wise upper bound over the evaluator's supplied,
    /// predeclared threshold list. Uses Hoeffding + Bonferroni and assumes IID evaluation
    /// episodes and a frozen score/selection rule. `None` when none are selected.
    /// The bound does not cover thresholds searched outside that list and cannot repair shift.
    pub risk_upper_bound_95: Option<f64>,
    pub selected: usize,
}

/// One fixed-width confidence bin with outcome frequency and uncertainty interval.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CalibrationBinReport {
    pub bin_index: usize,
    /// Inclusive lower endpoint of the confidence interval represented by this bin.
    pub confidence_lower: f64,
    /// Upper endpoint; exclusive except that the final bin includes confidence 1.0.
    pub confidence_upper: f64,
    pub episodes: usize,
    pub mean_confidence: Option<f64>,
    pub empirical_accuracy: Option<f64>,
    /// Conservative family-wise adjusted two-sided Hoeffding interval for accuracy in this bin.
    /// None when the bin is empty.
    pub accuracy_lower_95: Option<f64>,
    pub accuracy_upper_95: Option<f64>,
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

/// Calibration report scoped to one predeclared task family.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskFamilyCalibrationReport {
    pub task_family_id: String,
    pub episodes: usize,
    pub empirical_accuracy: Option<f64>,
    pub mean_confidence: Option<f64>,
    pub brier_score: Option<f64>,
    pub log_loss: Option<f64>,
    pub expected_calibration_error: Option<f64>,
    pub maximum_calibration_error: Option<f64>,
    /// Mean confidence minus accuracy; positive means overconfidence in this family.
    pub confidence_bias: Option<f64>,
    pub correctness_auroc: Option<f64>,
    pub current_episode_binding_rate: Option<f64>,
    pub assertion_rate: Option<f64>,
    pub asserted_risk: Option<f64>,
    /// Group-local coverage and risk; upper bounds share family-wise correction
    /// with pooled results.
    pub selective_risk: Vec<SelectiveRiskPoint>,
    /// Equal-width reliability bins with simultaneous accuracy intervals.
    pub calibration_bins: Vec<CalibrationBinReport>,
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
    /// Machine-readable method identity for simultaneous selective-risk upper bounds.
    pub selective_risk_bound_method: String,
    /// Explicit assumptions/limitations shipped with the report for downstream consumers.
    pub selective_risk_bound_assumptions: String,
    /// Deterministically sorted, group-local calibration results; never averaged into a scalar.
    pub task_family_calibration: Vec<TaskFamilyCalibrationReport>,
    /// Pooled equal-width reliability bins with simultaneous accuracy intervals.
    pub calibration_bins: Vec<CalibrationBinReport>,
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
    EmptyTaskFamilyId { episode_id: String },
    DuplicateAssumptionEpisode(String),
    DuplicateRevisionPair(String),
    InvalidProbability { field: &'static str, value: f64 },
    InvalidTolerance(f64),
    InvalidThreshold(f64),
}

impl fmt::Display for MetacognitionEvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBinCount(count) => {
                write!(f, "calibration bin count must be in 1..=100, got {count}")
            }
            Self::EmptyEpisodeId(field) => {
                write!(f, "required episode identifier `{field}` is empty")
            }
            Self::DuplicatePredictionEpisode(id) => {
                write!(f, "episode `{id}` has more than one correctness prediction")
            }
            Self::EmptyTaskFamilyId { episode_id } => {
                write!(f, "episode `{episode_id}` has an empty task-family identifier")
            }
            Self::DuplicateAssumptionEpisode(id) => {
                write!(f, "episode `{id}` has more than one weak-assumption observation")
            }
            Self::DuplicateRevisionPair(id) => {
                write!(f, "revision pair `{id}` appears more than once")
            }
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
    let mut task_family_members: BTreeMap<String, Vec<CorrectnessPrediction>> = BTreeMap::new();
    for prediction in predictions {
        task_family_members
            .entry(prediction.task_family_id.clone())
            .or_default()
            .push(prediction.clone());
    }
    // Correct across pooled + every observed task family, for each predeclared threshold.
    // Task-family labels themselves must come from a frozen, outcome-independent taxonomy.
    let familywise_comparison_count = selective_thresholds
        .len()
        .saturating_add(calibration_bins)
        .saturating_mul(task_family_members.len().saturating_add(1));

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

    // Freeze thresholds, equal-width bins, and group taxonomy before inspecting outcomes.
    // Bonferroni covers every pooled/group threshold bound and every reported calibration bin.
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
                risk_upper_bound_95: hoeffding_familywise_risk_upper_bound(
                    errors,
                    selected_n,
                    familywise_comparison_count,
                ),
                selected: selected_n,
            }
        })
        .collect();

    let task_family_calibration = task_family_members
        .iter()
        .map(|(task_family_id, members)| {
            task_family_report(
                task_family_id,
                members,
                calibration_bins,
                selective_thresholds,
                familywise_comparison_count,
            )
        })
        .collect();
    let pooled_calibration_bins = calibration_bin_reports(
        predictions,
        calibration_bins,
        familywise_comparison_count,
    );

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
        selective_risk_bound_method: SELECTIVE_RISK_BOUND_METHOD.into(),
        selective_risk_bound_assumptions: SELECTIVE_RISK_BOUND_ASSUMPTIONS.into(),
        task_family_calibration,
        calibration_bins: pooled_calibration_bins,
        selective_risk,
        weak_assumption_detection,
        confidence_revisions: revisions.len(),
        revision_direction_accuracy,
        mean_expected_revision_delta,
    })
}

fn validate_predictions(
    predictions: &[CorrectnessPrediction],
) -> Result<(), MetacognitionEvaluationError> {
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
        if prediction.task_family_id.trim().is_empty() {
            return Err(MetacognitionEvaluationError::EmptyTaskFamilyId {
                episode_id: prediction.episode_id.clone(),
            });
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

fn validate_probability(
    field: &'static str,
    value: f64,
) -> Result<(), MetacognitionEvaluationError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(MetacognitionEvaluationError::InvalidProbability { field, value })
    }
}

/// Conservative one-sided Hoeffding bound with Bonferroni correction over a frozen threshold
/// family. It assumes independent evaluation episodes under a fixed selection rule; it is not
/// a distribution-shift guarantee.
fn hoeffding_familywise_risk_upper_bound(
    errors: usize,
    selected: usize,
    comparison_count: usize,
) -> Option<f64> {
    if selected == 0 || comparison_count == 0 || errors > selected {
        return None;
    }
    let empirical_risk = errors as f64 / selected as f64;
    let per_comparison_alpha = SELECTIVE_RISK_FAMILYWISE_ALPHA / comparison_count as f64;
    let radius = ((1.0 / per_comparison_alpha).ln() / (2.0 * selected as f64)).sqrt();
    Some((empirical_risk + radius).min(1.0))
}

fn task_family_report(
    task_family_id: &str,
    predictions: &[CorrectnessPrediction],
    calibration_bins: usize,
    selective_thresholds: &[f64],
    familywise_comparison_count: usize,
) -> TaskFamilyCalibrationReport {
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
            log += -p_correct.clamp(LOG_LOSS_EPSILON, 1.0 - LOG_LOSS_EPSILON).ln();
        }
        (Some(brier / n as f64), Some(log / n as f64))
    };
    let (ece, mce) = calibration_errors(predictions, calibration_bins);
    let confidence_bias = match (mean_confidence, empirical_accuracy) {
        (Some(confidence), Some(accuracy)) => Some(confidence - accuracy),
        _ => None,
    };
    let current_episode_binding_rate = mean_bool(
        predictions.iter().map(|p| p.episode_id == p.confidence_target_episode_id),
    );
    let assertion_rate = mean_bool(predictions.iter().map(|p| p.asserted));
    let asserted_risk = conditional_rate(predictions.iter().filter(|p| p.asserted), |p| !p.correct);
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
                coverage: if n == 0 { 0.0 } else { selected_n as f64 / n as f64 },
                risk: ratio(errors, selected_n),
                risk_upper_bound_95: hoeffding_familywise_risk_upper_bound(
                    errors,
                    selected_n,
                    familywise_comparison_count,
                ),
                selected: selected_n,
            }
        })
        .collect();

    TaskFamilyCalibrationReport {
        task_family_id: task_family_id.to_string(),
        episodes: n,
        empirical_accuracy,
        mean_confidence,
        brier_score,
        log_loss,
        expected_calibration_error: ece,
        maximum_calibration_error: mce,
        confidence_bias,
        correctness_auroc: auroc(predictions),
        current_episode_binding_rate,
        assertion_rate,
        asserted_risk,
        selective_risk,
        calibration_bins: calibration_bin_reports(
            predictions,
            calibration_bins,
            familywise_comparison_count,
        ),
    }
}

fn calibration_bin_reports(
    predictions: &[CorrectnessPrediction],
    bins: usize,
    familywise_comparison_count: usize,
) -> Vec<CalibrationBinReport> {
    let mut counts = vec![0usize; bins];
    let mut confidence_sums = vec![0.0_f64; bins];
    let mut correct_counts = vec![0usize; bins];
    for prediction in predictions {
        let index = ((prediction.confidence * bins as f64).floor() as usize).min(bins - 1);
        counts[index] += 1;
        confidence_sums[index] += prediction.confidence;
        correct_counts[index] += usize::from(prediction.correct);
    }

    (0..bins)
        .map(|bin_index| {
            let episodes = counts[bin_index];
            let empirical_accuracy = ratio(correct_counts[bin_index], episodes);
            let mean_confidence =
                (episodes > 0).then(|| confidence_sums[bin_index] / episodes as f64);
            let interval = empirical_accuracy.map(|accuracy| {
                // Two-sided Hoeffding interval at alpha / familywise_comparison_count.
                // The union bound covers every pooled/group bin and selective-risk threshold.
                let per_comparison_alpha =
                    SELECTIVE_RISK_FAMILYWISE_ALPHA / familywise_comparison_count as f64;
                let radius = ((2.0 / per_comparison_alpha).ln()
                    / (2.0 * episodes as f64))
                    .sqrt();
                ((accuracy - radius).max(0.0), (accuracy + radius).min(1.0))
            });
            CalibrationBinReport {
                bin_index,
                confidence_lower: bin_index as f64 / bins as f64,
                confidence_upper: (bin_index + 1) as f64 / bins as f64,
                episodes,
                mean_confidence,
                empirical_accuracy,
                accuracy_lower_95: interval.map(|(lower, _)| lower),
                accuracy_upper_95: interval.map(|(_, upper)| upper),
            }
        })
        .collect()
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
    let negatives: Vec<&CorrectnessPrediction> = predictions
        .iter()
        .filter(|p| !p.correct)
        .collect();
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

    fn prediction(
        id: &str,
        confidence: f64,
        correct: bool,
        asserted: bool,
    ) -> CorrectnessPrediction {
        CorrectnessPrediction {
            episode_id: id.into(),
            task_family_id: "general".into(),
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
        assert_eq!(report.selective_risk_bound_method, "hoeffding-familywise-95-v1");
        assert!(report.selective_risk_bound_assumptions.contains("IID evaluation episodes"));
        assert!(report
            .selective_risk_bound_assumptions
            .contains("no distribution-shift guarantee"));
        assert!(report
            .selective_risk_bound_assumptions
            .contains("task-family taxonomy predeclared"));
        assert_eq!(report.selective_risk[0].selected, 3);
        assert_eq!(report.selective_risk[0].coverage, 0.75);
        assert!((report.selective_risk[0].risk.unwrap_or_default() - 1.0 / 3.0).abs() < 1.0e-12);
        assert!(report.selective_risk[0].risk_upper_bound_95.unwrap_or_default()
            >= report.selective_risk[0].risk.unwrap_or_default());
        assert_eq!(report.selective_risk[1].selected, 2);
        assert_eq!(report.selective_risk[1].risk, Some(0.5));
        assert!(report.selective_risk[1].risk_upper_bound_95.unwrap_or_default() >= 0.5);
    }

    #[test]
    fn selective_risk_upper_bound_is_finite_sample_conservative_and_multiplicity_corrected() {
        assert_eq!(hoeffding_familywise_risk_upper_bound(0, 0, 1), None);
        assert_eq!(hoeffding_familywise_risk_upper_bound(1, 0, 1), None);
        assert_eq!(hoeffding_familywise_risk_upper_bound(2, 1, 1), None);
        let expected_one = (20.0_f64.ln() / 200.0).sqrt();
        let one_threshold = hoeffding_familywise_risk_upper_bound(0, 100, 1)
            .expect("non-empty selected sample has a bound");
        let two_thresholds = hoeffding_familywise_risk_upper_bound(0, 100, 2)
            .expect("non-empty selected sample has a bound");
        assert!((one_threshold - expected_one).abs() < 1.0e-12);
        assert!(one_threshold > 0.0);
        assert!(two_thresholds > one_threshold,
            "family-wise correction must account for additional thresholds");
        assert_eq!(hoeffding_familywise_risk_upper_bound(100, 100, 4), Some(1.0));
        assert!((0..=100).all(|errors| {
            hoeffding_familywise_risk_upper_bound(errors, 100, 5)
                .is_some_and(|bound| (errors as f64 / 100.0) <= bound && bound <= 1.0)
        }));
    }

    #[test]
    fn task_family_reports_expose_pooled_calibration_cancellation() {
        let mut observations = Vec::new();
        for index in 0..5 {
            let mut p = prediction(&format!("reasoning-{index}"), 0.9, true, true);
            p.task_family_id = "reasoning".into();
            observations.push(p);
        }
        for index in 0..5 {
            let mut p = prediction(&format!("retrieval-{index}"), 0.9, index != 0, true);
            p.task_family_id = "retrieval".into();
            observations.push(p);
        }

        let report = evaluate_metacognition(&observations, &[], &[], 10, &[0.5, 0.9])
            .unwrap_or_else(|err| panic!("evaluation must succeed: {err}"));
        assert_eq!(report.empirical_accuracy, Some(0.9));
        assert!(
            (report.mean_confidence.unwrap_or_default() - 0.9).abs() < 1.0e-12
        );
        assert!(
            report.expected_calibration_error.unwrap_or_default().abs() < 1.0e-12
        );
        assert_eq!(report.task_family_calibration.len(), 2);
        assert_eq!(report.task_family_calibration[0].task_family_id, "reasoning");
        assert!(
            report.task_family_calibration[0]
                .expected_calibration_error
                .unwrap_or_default()
                .abs()
                < 1.0e-12
        );
        assert_eq!(report.task_family_calibration[1].task_family_id, "retrieval");
        assert!(
            (report.task_family_calibration[1]
                .expected_calibration_error
                .unwrap_or_default()
                - 0.1)
                .abs()
                < 1.0e-12
        );
        assert!(report.task_family_calibration.iter().all(|family| {
            family.selective_risk.iter().all(|point| {
                point.risk_upper_bound_95.unwrap_or_default() >= point.risk.unwrap_or_default()
            })
        }));
        assert!(report.task_family_calibration[0].selective_risk[0].risk_upper_bound_95
            >= report.selective_risk[0].risk_upper_bound_95);

        assert_eq!(report.calibration_bins.len(), 10);
        let pooled_top_bin = report.calibration_bins[9];
        assert_eq!(pooled_top_bin.episodes, 10);
        assert!((pooled_top_bin.mean_confidence.unwrap_or_default() - 0.9).abs() < 1.0e-12);
        assert_eq!(pooled_top_bin.empirical_accuracy, Some(0.9));
        assert!(pooled_top_bin.accuracy_lower_95.unwrap_or_default() <= 0.9);
        assert!(pooled_top_bin.accuracy_upper_95.unwrap_or_default() >= 0.9);
        assert_eq!(report.calibration_bins[0].episodes, 0);
        assert_eq!(report.calibration_bins[0].empirical_accuracy, None);
        assert_eq!(report.calibration_bins[0].accuracy_lower_95, None);
        assert_eq!(report.task_family_calibration[1].calibration_bins[9].episodes, 5);
        assert_eq!(
            report.task_family_calibration[1].calibration_bins[9].empirical_accuracy,
            Some(0.8)
        );
        let pooled_width = pooled_top_bin.accuracy_upper_95.unwrap_or_default()
            - pooled_top_bin.accuracy_lower_95.unwrap_or_default();
        let subgroup_bin = report.task_family_calibration[1].calibration_bins[9];
        let subgroup_width = subgroup_bin.accuracy_upper_95.unwrap_or_default()
            - subgroup_bin.accuracy_lower_95.unwrap_or_default();
        assert!(subgroup_width > pooled_width, "smaller subgroup must show greater uncertainty");
    }

    #[test]
    fn empty_task_family_id_fails_closed() {
        let mut observation = prediction("p1", 0.5, true, true);
        observation.task_family_id.clear();
        assert!(matches!(
            evaluate_metacognition(&[observation], &[], &[], 10, &[]),
            Err(MetacognitionEvaluationError::EmptyTaskFamilyId { .. })
        ));
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
