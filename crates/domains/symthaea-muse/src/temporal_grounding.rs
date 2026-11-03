// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Versioned, artifact-bound temporal grounding contracts and scorer.
//!
//! The evaluator deliberately separates four outcomes that are often
//! collapsed by audio-language systems: a query is absent, an event was not
//! detected, the evidence is ambiguous, or the event is not estimable.
//! Intervals use half-open sample-frame coordinates so the score is exact
//! with respect to the declared audio clock. This module scores supplied
//! annotations and predictions; it does not detect events in waveforms.

use crate::evidence_digest::{canonical_json_sha256, decode_hex_32};
use serde::{Deserialize, Serialize};

pub const TEMPORAL_GROUNDING_SCHEMA_V1: &str = "mel-temporal-grounding-v1";
pub const TEMPORAL_GROUNDING_EVALUATOR_V1: &str = "mel-temporal-grounding-evaluator-v1";

/// Identity and clock for the exact audio artifact being queried.
///
/// The SHA-256 must be computed upstream from the exact audio bytes. This
/// structure binds a prediction to that digest and clock; it does not itself
/// prove that the caller hashed the right bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioArtifactClockV1 {
    pub audio_sha256: String,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub total_frames: u64,
}

/// A half-open interval in sample frames: [start_frame, end_frame).
///
/// Half-open intervals make adjacent events unambiguous: [0, 10) and
/// [10, 20) do not overlap. Overlapping/nested intervals are permitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FrameIntervalV1 {
    pub start_frame: u64,
    pub end_frame: u64,
}

impl FrameIntervalV1 {
    pub fn len_frames(self) -> u64 {
        self.end_frame - self.start_frame
    }
}

/// Adjudicated annotation state for a query over one exact audio artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GroundTruthStateV1 {
    Present,
    Absent,
    Ambiguous,
    NotEstimable,
}

/// Ground truth for one natural-language event query.
///
/// Ambiguous and NotEstimable records are retained as data but excluded from
/// interval accuracy denominators. Their reason and annotation-set digest
/// remain attached so exclusion is visible and auditable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalGroundTruthV1 {
    pub schema_version: String,
    pub query_id: String,
    pub query_text: String,
    pub artifact: AudioArtifactClockV1,
    pub annotation_policy_version: String,
    pub annotation_set_sha256: String,
    pub annotator_count: u16,
    pub state: GroundTruthStateV1,
    pub intervals: Vec<FrameIntervalV1>,
    pub status_reason: Option<String>,
}

/// Prediction disposition for one temporal grounding query.
///
/// QueryAbsent is an explicit negative claim. NotDetected means the system
/// attempted detection but returned no event. Ambiguous and NotEstimable
/// are explicit abstentions, not negative claims.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GroundingPredictionStateV1 {
    Occurrences {
        intervals: Vec<FrameIntervalV1>,
    },
    QueryAbsent,
    NotDetected,
    Ambiguous {
        candidate_intervals: Vec<FrameIntervalV1>,
        reason: String,
    },
    NotEstimable {
        reason: String,
    },
}

/// Prediction bound to one query and the exact audio artifact clock.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalGroundingPredictionV1 {
    pub schema_version: String,
    pub query_id: String,
    pub artifact: AudioArtifactClockV1,
    pub state: GroundingPredictionStateV1,
}

/// Frozen scoring policy. Threshold is interval IoU in (0, 1].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalGroundingPlanV1 {
    pub evaluator_version: String,
    pub minimum_interval_iou: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalGroundingIssue {
    WrongGroundTruthSchema { found: String },
    WrongPredictionSchema { found: String },
    WrongEvaluatorVersion { found: String },
    EmptyQueryId,
    EmptyQueryText,
    EmptyAnnotationPolicyVersion,
    InvalidAnnotationDigest,
    ZeroAnnotators,
    InvalidAudioDigest,
    InvalidAudioClock { field: String },
    InvalidInterval { index: usize },
    IntervalBeyondArtifact { index: usize },
    UnsortedIntervals { index: usize },
    DuplicateInterval { index: usize },
    PresentTruthHasNoIntervals,
    NonPresentTruthHasIntervals,
    MissingTruthStatusReason,
    EmptyOccurrencePrediction,
    EmptyAmbiguousCandidateSet,
    MissingPredictionStatusReason,
    InvalidIoUThreshold,
    QueryIdMismatch,
    ArtifactMismatch { field: String },
}

/// A specific one-to-one match between predicted and annotated occurrences.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalIntervalMatchV1 {
    pub prediction_index: usize,
    pub annotation_index: usize,
    pub interval_iou: f64,
}

/// Whether a sample contributes to positive-event or negative-query metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalGroundingOutcomeV1 {
    ScoredPresentQuery,
    ScoredAbsentQuery,
    ExcludedAmbiguousGroundTruth,
    ExcludedNotEstimableGroundTruth,
}

/// Prediction disposition copied into the score for aggregate analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GroundingPredictionKindV1 {
    Occurrences,
    QueryAbsent,
    NotDetected,
    Ambiguous,
    NotEstimable,
}

/// Per-query score. Positive-query P/R/F1 are None for negative or excluded
/// queries; negative-query correctness is None when the system abstains.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalGroundingScoreV1 {
    pub evaluator_version: String,
    pub query_id: String,
    pub audio_sha256: String,
    pub outcome: TemporalGroundingOutcomeV1,
    pub prediction_kind: GroundingPredictionKindV1,
    pub scored: bool,
    pub abstained: bool,
    pub ground_truth_occurrences: usize,
    pub predicted_occurrences: usize,
    pub ambiguous_candidate_count: usize,
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f1: Option<f64>,
    pub matched_interval_iou_mean: Option<f64>,
    pub count_absolute_error: Option<usize>,
    pub exact_occurrence_count: Option<bool>,
    pub correct_absence_claim: Option<bool>,
    pub matches: Vec<TemporalIntervalMatchV1>,
}

/// Return all validation issues. This intentionally validates the caller's
/// declared digest syntax and clock, not the underlying WAV bytes.
pub fn validate_temporal_ground_truth(
    truth: &TemporalGroundTruthV1,
) -> Vec<TemporalGroundingIssue> {
    let mut issues = Vec::new();
    if truth.schema_version != TEMPORAL_GROUNDING_SCHEMA_V1 {
        issues.push(TemporalGroundingIssue::WrongGroundTruthSchema {
            found: truth.schema_version.clone(),
        });
    }
    if truth.query_id.trim().is_empty() {
        issues.push(TemporalGroundingIssue::EmptyQueryId);
    }
    if truth.query_text.trim().is_empty() {
        issues.push(TemporalGroundingIssue::EmptyQueryText);
    }
    if truth.annotation_policy_version.trim().is_empty() {
        issues.push(TemporalGroundingIssue::EmptyAnnotationPolicyVersion);
    }
    if !is_lower_hex_digest(&truth.annotation_set_sha256) {
        issues.push(TemporalGroundingIssue::InvalidAnnotationDigest);
    }
    if truth.annotator_count == 0 {
        issues.push(TemporalGroundingIssue::ZeroAnnotators);
    }
    validate_artifact(&truth.artifact, &mut issues);
    validate_intervals(&truth.intervals, &truth.artifact, &mut issues);

    match truth.state {
        GroundTruthStateV1::Present if truth.intervals.is_empty() => {
            issues.push(TemporalGroundingIssue::PresentTruthHasNoIntervals);
        }
        GroundTruthStateV1::Present => {}
        _ if !truth.intervals.is_empty() => {
            issues.push(TemporalGroundingIssue::NonPresentTruthHasIntervals);
        }
        _ => {}
    }
    if matches!(
        truth.state,
        GroundTruthStateV1::Ambiguous | GroundTruthStateV1::NotEstimable
    ) && truth
        .status_reason
        .as_deref()
        .is_none_or(|reason| reason.trim().is_empty())
    {
        issues.push(TemporalGroundingIssue::MissingTruthStatusReason);
    }
    issues
}

/// Return all prediction validation issues.
pub fn validate_temporal_grounding_prediction(
    prediction: &TemporalGroundingPredictionV1,
) -> Vec<TemporalGroundingIssue> {
    let mut issues = Vec::new();
    if prediction.schema_version != TEMPORAL_GROUNDING_SCHEMA_V1 {
        issues.push(TemporalGroundingIssue::WrongPredictionSchema {
            found: prediction.schema_version.clone(),
        });
    }
    if prediction.query_id.trim().is_empty() {
        issues.push(TemporalGroundingIssue::EmptyQueryId);
    }
    validate_artifact(&prediction.artifact, &mut issues);

    match &prediction.state {
        GroundingPredictionStateV1::Occurrences { intervals } => {
            if intervals.is_empty() {
                issues.push(TemporalGroundingIssue::EmptyOccurrencePrediction);
            }
            validate_intervals(intervals, &prediction.artifact, &mut issues);
        }
        GroundingPredictionStateV1::Ambiguous {
            candidate_intervals,
            reason,
        } => {
            if candidate_intervals.is_empty() {
                issues.push(TemporalGroundingIssue::EmptyAmbiguousCandidateSet);
            }
            if reason.trim().is_empty() {
                issues.push(TemporalGroundingIssue::MissingPredictionStatusReason);
            }
            validate_intervals(candidate_intervals, &prediction.artifact, &mut issues);
        }
        GroundingPredictionStateV1::NotEstimable { reason } if reason.trim().is_empty() => {
            issues.push(TemporalGroundingIssue::MissingPredictionStatusReason);
        }
        _ => {}
    }
    issues
}

pub fn validate_temporal_grounding_plan(
    plan: &TemporalGroundingPlanV1,
) -> Vec<TemporalGroundingIssue> {
    let mut issues = Vec::new();
    if plan.evaluator_version != TEMPORAL_GROUNDING_EVALUATOR_V1 {
        issues.push(TemporalGroundingIssue::WrongEvaluatorVersion {
            found: plan.evaluator_version.clone(),
        });
    }
    if !plan.minimum_interval_iou.is_finite()
        || plan.minimum_interval_iou <= 0.0
        || plan.minimum_interval_iou > 1.0
    {
        issues.push(TemporalGroundingIssue::InvalidIoUThreshold);
    }
    issues
}

/// Exact intersection-over-union for two intervals in the same frame clock.
pub fn frame_interval_iou(left: FrameIntervalV1, right: FrameIntervalV1) -> f64 {
    let start = left.start_frame.max(right.start_frame);
    let end = left.end_frame.min(right.end_frame);
    let intersection = u128::from(end.saturating_sub(start));
    if intersection == 0 {
        return 0.0;
    }
    let left_len = u128::from(left.end_frame - left.start_frame);
    let right_len = u128::from(right.end_frame - right.start_frame);
    let union = left_len + right_len - intersection;
    intersection as f64 / union as f64
}

/// Score one query. Invalid schemas, artifact mismatch and invalid clocks
/// fail closed. Ambiguous/unestimable ground truth is retained but excluded.
pub fn score_temporal_grounding(
    truth: &TemporalGroundTruthV1,
    prediction: &TemporalGroundingPredictionV1,
    plan: &TemporalGroundingPlanV1,
) -> Result<TemporalGroundingScoreV1, Vec<TemporalGroundingIssue>> {
    let mut issues = validate_temporal_ground_truth(truth);
    issues.extend(validate_temporal_grounding_prediction(prediction));
    issues.extend(validate_temporal_grounding_plan(plan));
    if truth.query_id != prediction.query_id {
        issues.push(TemporalGroundingIssue::QueryIdMismatch);
    }
    if truth.artifact.audio_sha256 != prediction.artifact.audio_sha256 {
        issues.push(TemporalGroundingIssue::ArtifactMismatch {
            field: "audio_sha256".into(),
        });
    }
    if truth.artifact.sample_rate_hz != prediction.artifact.sample_rate_hz {
        issues.push(TemporalGroundingIssue::ArtifactMismatch {
            field: "sample_rate_hz".into(),
        });
    }
    if truth.artifact.channels != prediction.artifact.channels {
        issues.push(TemporalGroundingIssue::ArtifactMismatch {
            field: "channels".into(),
        });
    }
    if truth.artifact.total_frames != prediction.artifact.total_frames {
        issues.push(TemporalGroundingIssue::ArtifactMismatch {
            field: "total_frames".into(),
        });
    }
    if !issues.is_empty() {
        return Err(issues);
    }

    let prediction_kind = prediction_kind(&prediction.state);
    let (outcome, scored) = match truth.state {
        GroundTruthStateV1::Present => {
            (TemporalGroundingOutcomeV1::ScoredPresentQuery, true)
        }
        GroundTruthStateV1::Absent => {
            (TemporalGroundingOutcomeV1::ScoredAbsentQuery, true)
        }
        GroundTruthStateV1::Ambiguous => (
            TemporalGroundingOutcomeV1::ExcludedAmbiguousGroundTruth,
            false,
        ),
        GroundTruthStateV1::NotEstimable => (
            TemporalGroundingOutcomeV1::ExcludedNotEstimableGroundTruth,
            false,
        ),
    };

    let predicted_intervals: &[FrameIntervalV1] = match &prediction.state {
        GroundingPredictionStateV1::Occurrences { intervals } => intervals,
        _ => &[],
    };
    let ambiguous_candidate_count = match &prediction.state {
        GroundingPredictionStateV1::Ambiguous {
            candidate_intervals,
            ..
        } => candidate_intervals.len(),
        _ => 0,
    };
    let abstained = matches!(
        prediction.state,
        GroundingPredictionStateV1::Ambiguous { .. }
            | GroundingPredictionStateV1::NotEstimable { .. }
    ) || (truth.state == GroundTruthStateV1::Absent
        && prediction.state == GroundingPredictionStateV1::NotDetected);
    let predicted_occurrences = predicted_intervals.len();

    if !scored {
        return Ok(TemporalGroundingScoreV1 {
            evaluator_version: plan.evaluator_version.clone(),
            query_id: truth.query_id.clone(),
            audio_sha256: truth.artifact.audio_sha256.clone(),
            outcome,
            prediction_kind,
            scored: false,
            abstained,
            ground_truth_occurrences: 0,
            predicted_occurrences,
            ambiguous_candidate_count,
            true_positives: 0,
            false_positives: 0,
            false_negatives: 0,
            precision: None,
            recall: None,
            f1: None,
            matched_interval_iou_mean: None,
            count_absolute_error: None,
            exact_occurrence_count: None,
            correct_absence_claim: None,
            matches: Vec::new(),
        });
    }

    if truth.state == GroundTruthStateV1::Absent {
        let correct_absence_claim = match prediction.state {
            GroundingPredictionStateV1::QueryAbsent => Some(true),
            GroundingPredictionStateV1::Occurrences { .. } => Some(false),
            GroundingPredictionStateV1::NotDetected
            | GroundingPredictionStateV1::Ambiguous { .. }
            | GroundingPredictionStateV1::NotEstimable { .. } => None,
        };
        let false_positives = predicted_occurrences;
        return Ok(TemporalGroundingScoreV1 {
            evaluator_version: plan.evaluator_version.clone(),
            query_id: truth.query_id.clone(),
            audio_sha256: truth.artifact.audio_sha256.clone(),
            outcome,
            prediction_kind,
            scored: true,
            abstained,
            ground_truth_occurrences: 0,
            predicted_occurrences,
            ambiguous_candidate_count,
            true_positives: 0,
            false_positives,
            false_negatives: 0,
            precision: if predicted_occurrences > 0 {
                Some(0.0)
            } else {
                None
            },
            recall: None,
            f1: if predicted_occurrences > 0 {
                Some(0.0)
            } else {
                None
            },
            matched_interval_iou_mean: None,
            count_absolute_error: None,
            exact_occurrence_count: None,
            correct_absence_claim,
            matches: Vec::new(),
        });
    }

    let (matches, matched_iou_mean) = maximum_cardinality_matches(
        predicted_intervals,
        &truth.intervals,
        plan.minimum_interval_iou,
    );
    let true_positives = matches.len();
    let false_positives = predicted_occurrences - true_positives;
    let false_negatives = truth.intervals.len() - true_positives;
    let precision = if predicted_occurrences == 0 {
        0.0
    } else {
        true_positives as f64 / predicted_occurrences as f64
    };
    let recall = true_positives as f64 / truth.intervals.len() as f64;
    let f1 = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    let count_absolute_error = predicted_occurrences.abs_diff(truth.intervals.len());

    Ok(TemporalGroundingScoreV1 {
        evaluator_version: plan.evaluator_version.clone(),
        query_id: truth.query_id.clone(),
        audio_sha256: truth.artifact.audio_sha256.clone(),
        outcome,
        prediction_kind,
        scored: true,
        abstained,
        ground_truth_occurrences: truth.intervals.len(),
        predicted_occurrences,
        ambiguous_candidate_count,
        true_positives,
        false_positives,
        false_negatives,
        precision: Some(precision),
        recall: Some(recall),
        f1: Some(f1),
        matched_interval_iou_mean: Some(matched_iou_mean),
        count_absolute_error: Some(count_absolute_error),
        exact_occurrence_count: Some(count_absolute_error == 0 && !abstained),
        correct_absence_claim: None,
        matches,
    })
}

/// Bind the exact truth, prediction and scoring plan into one canonical
/// commitment. The repository's canonical JSON encoder is used; this is not
/// a cross-language canonicalization standard.
pub fn temporal_grounding_input_commitment_sha256(
    truth: &TemporalGroundTruthV1,
    prediction: &TemporalGroundingPredictionV1,
    plan: &TemporalGroundingPlanV1,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&(truth, prediction, plan))
}

fn prediction_kind(state: &GroundingPredictionStateV1) -> GroundingPredictionKindV1 {
    match state {
        GroundingPredictionStateV1::Occurrences { .. } => {
            GroundingPredictionKindV1::Occurrences
        }
        GroundingPredictionStateV1::QueryAbsent => GroundingPredictionKindV1::QueryAbsent,
        GroundingPredictionStateV1::NotDetected => GroundingPredictionKindV1::NotDetected,
        GroundingPredictionStateV1::Ambiguous { .. } => GroundingPredictionKindV1::Ambiguous,
        GroundingPredictionStateV1::NotEstimable { .. } => {
            GroundingPredictionKindV1::NotEstimable
        }
    }
}

fn validate_artifact(
    artifact: &AudioArtifactClockV1,
    issues: &mut Vec<TemporalGroundingIssue>,
) {
    if !is_lower_hex_digest(&artifact.audio_sha256) {
        issues.push(TemporalGroundingIssue::InvalidAudioDigest);
    }
    if artifact.sample_rate_hz == 0 {
        issues.push(TemporalGroundingIssue::InvalidAudioClock {
            field: "sample_rate_hz".into(),
        });
    }
    if artifact.channels == 0 {
        issues.push(TemporalGroundingIssue::InvalidAudioClock {
            field: "channels".into(),
        });
    }
    if artifact.total_frames == 0 {
        issues.push(TemporalGroundingIssue::InvalidAudioClock {
            field: "total_frames".into(),
        });
    }
}

fn validate_intervals(
    intervals: &[FrameIntervalV1],
    artifact: &AudioArtifactClockV1,
    issues: &mut Vec<TemporalGroundingIssue>,
) {
    let mut previous: Option<FrameIntervalV1> = None;
    for (index, interval) in intervals.iter().copied().enumerate() {
        if interval.start_frame >= interval.end_frame {
            issues.push(TemporalGroundingIssue::InvalidInterval { index });
            continue;
        }
        if interval.end_frame > artifact.total_frames {
            issues.push(TemporalGroundingIssue::IntervalBeyondArtifact { index });
        }
        if let Some(prior) = previous {
            if interval == prior {
                issues.push(TemporalGroundingIssue::DuplicateInterval { index });
            } else if interval < prior {
                issues.push(TemporalGroundingIssue::UnsortedIntervals { index });
            }
        }
        previous = Some(interval);
    }
}

fn is_lower_hex_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn maximum_cardinality_matches(
    predictions: &[FrameIntervalV1],
    annotations: &[FrameIntervalV1],
    threshold: f64,
) -> (Vec<TemporalIntervalMatchV1>, f64) {
    let mut annotation_to_prediction = vec![None; annotations.len()];
    for prediction_index in 0..predictions.len() {
        let mut visited_annotations = vec![false; annotations.len()];
        let _ = augment_match(
            prediction_index,
            predictions,
            annotations,
            threshold,
            &mut visited_annotations,
            &mut annotation_to_prediction,
        );
    }

    let mut matches = Vec::new();
    for (annotation_index, prediction_index) in
        annotation_to_prediction.into_iter().enumerate()
    {
        if let Some(prediction_index) = prediction_index {
            matches.push(TemporalIntervalMatchV1 {
                prediction_index,
                annotation_index,
                interval_iou: frame_interval_iou(
                    predictions[prediction_index],
                    annotations[annotation_index],
                ),
            });
        }
    }
    matches.sort_by(|left, right| {
        left.prediction_index
            .cmp(&right.prediction_index)
            .then_with(|| left.annotation_index.cmp(&right.annotation_index))
    });
    let mean_iou = if matches.is_empty() {
        0.0
    } else {
        matches.iter().map(|item| item.interval_iou).sum::<f64>()
            / matches.len() as f64
    };
    (matches, mean_iou)
}

fn augment_match(
    prediction_index: usize,
    predictions: &[FrameIntervalV1],
    annotations: &[FrameIntervalV1],
    threshold: f64,
    visited_annotations: &mut [bool],
    annotation_to_prediction: &mut [Option<usize>],
) -> bool {
    let mut candidates: Vec<_> = annotations
        .iter()
        .enumerate()
        .filter_map(|(annotation_index, annotation)| {
            let iou = frame_interval_iou(predictions[prediction_index], *annotation);
            (iou >= threshold).then_some((annotation_index, iou))
        })
        .collect();
    candidates.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });

    for (annotation_index, _) in candidates {
        if visited_annotations[annotation_index] {
            continue;
        }
        visited_annotations[annotation_index] = true;
        let previous_prediction = annotation_to_prediction[annotation_index];
        let can_assign = match previous_prediction {
            None => true,
            Some(previous_index) => augment_match(
                previous_index,
                predictions,
                annotations,
                threshold,
                visited_annotations,
                annotation_to_prediction,
            ),
        };
        if can_assign {
            annotation_to_prediction[annotation_index] = Some(prediction_index);
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_digest::sha256_hex;

    fn artifact() -> AudioArtifactClockV1 {
        AudioArtifactClockV1 {
            audio_sha256: sha256_hex(b"exact rendered audio bytes"),
            sample_rate_hz: 48_000,
            channels: 2,
            total_frames: 480_000,
        }
    }

    fn truth(state: GroundTruthStateV1, intervals: Vec<FrameIntervalV1>) -> TemporalGroundTruthV1 {
        TemporalGroundTruthV1 {
            schema_version: TEMPORAL_GROUNDING_SCHEMA_V1.into(),
            query_id: "query-entry-01".into(),
            query_text: "Find every entrance of the violin".into(),
            artifact: artifact(),
            annotation_policy_version: "human-adjudication-v1".into(),
            annotation_set_sha256: sha256_hex(b"adjudicated annotation set"),
            annotator_count: 3,
            state,
            intervals,
            status_reason: if matches!(
                state,
                GroundTruthStateV1::Ambiguous | GroundTruthStateV1::NotEstimable
            ) {
                Some("annotators could not resolve the event boundary".into())
            } else {
                None
            },
        }
    }

    fn prediction(state: GroundingPredictionStateV1) -> TemporalGroundingPredictionV1 {
        TemporalGroundingPredictionV1 {
            schema_version: TEMPORAL_GROUNDING_SCHEMA_V1.into(),
            query_id: "query-entry-01".into(),
            artifact: artifact(),
            state,
        }
    }

    fn plan() -> TemporalGroundingPlanV1 {
        TemporalGroundingPlanV1 {
            evaluator_version: TEMPORAL_GROUNDING_EVALUATOR_V1.into(),
            minimum_interval_iou: 0.5,
        }
    }

    #[test]
    fn interval_iou_uses_half_open_frame_intervals() {
        let left = FrameIntervalV1 {
            start_frame: 0,
            end_frame: 10,
        };
        let right = FrameIntervalV1 {
            start_frame: 5,
            end_frame: 15,
        };
        assert!((frame_interval_iou(left, right) - (1.0 / 3.0)).abs() < 1e-12);
    }

    #[test]
    fn one_to_many_occurrences_are_matched_independently() {
        let truth = truth(
            GroundTruthStateV1::Present,
            vec![
                FrameIntervalV1 {
                    start_frame: 100,
                    end_frame: 200,
                },
                FrameIntervalV1 {
                    start_frame: 500,
                    end_frame: 600,
                },
            ],
        );
        let prediction = prediction(GroundingPredictionStateV1::Occurrences {
            intervals: vec![
                FrameIntervalV1 {
                    start_frame: 100,
                    end_frame: 200,
                },
                FrameIntervalV1 {
                    start_frame: 500,
                    end_frame: 600,
                },
                FrameIntervalV1 {
                    start_frame: 800,
                    end_frame: 900,
                },
            ],
        });
        let score = score_temporal_grounding(&truth, &prediction, &plan()).unwrap();
        assert_eq!(score.true_positives, 2);
        assert_eq!(score.false_positives, 1);
        assert_eq!(score.false_negatives, 0);
        assert_eq!(score.count_absolute_error, Some(1));
        assert_eq!(score.exact_occurrence_count, Some(false));
        assert!((score.f1.unwrap() - 0.8).abs() < 1e-12);
    }

    #[test]
    fn duplicate_like_predictions_cannot_claim_one_annotation_twice() {
        let truth = truth(
            GroundTruthStateV1::Present,
            vec![FrameIntervalV1 {
                start_frame: 100,
                end_frame: 200,
            }],
        );
        let prediction = prediction(GroundingPredictionStateV1::Occurrences {
            intervals: vec![
                FrameIntervalV1 {
                    start_frame: 100,
                    end_frame: 200,
                },
                FrameIntervalV1 {
                    start_frame: 110,
                    end_frame: 190,
                },
            ],
        });
        let score = score_temporal_grounding(&truth, &prediction, &plan()).unwrap();
        assert_eq!(score.true_positives, 1);
        assert_eq!(score.false_positives, 1);
        assert_eq!(score.matches.len(), 1);
    }

    #[test]
    fn absence_and_failed_detection_are_not_equivalent() {
        let truth = truth(GroundTruthStateV1::Absent, Vec::new());
        let absent = prediction(GroundingPredictionStateV1::QueryAbsent);
        let not_detected = prediction(GroundingPredictionStateV1::NotDetected);

        let absent_score = score_temporal_grounding(&truth, &absent, &plan()).unwrap();
        let missed_score = score_temporal_grounding(&truth, &not_detected, &plan()).unwrap();

        assert_eq!(absent_score.correct_absence_claim, Some(true));
        assert_eq!(missed_score.correct_absence_claim, None);
        assert!(!absent_score.abstained);
        assert!(missed_score.abstained);
    }

    #[test]
    fn explicit_ambiguous_and_unestimable_truth_is_excluded_not_relabelled_absent() {
        for state in [GroundTruthStateV1::Ambiguous, GroundTruthStateV1::NotEstimable] {
            let truth = truth(state, Vec::new());
            let prediction = prediction(GroundingPredictionStateV1::QueryAbsent);
            let score = score_temporal_grounding(&truth, &prediction, &plan()).unwrap();
            assert!(!score.scored);
            assert_eq!(score.correct_absence_claim, None);
            assert_eq!(score.f1, None);
        }
    }

    #[test]
    fn mismatched_audio_artifact_is_rejected() {
        let truth = truth(
            GroundTruthStateV1::Present,
            vec![FrameIntervalV1 {
                start_frame: 10,
                end_frame: 20,
            }],
        );
        let mut prediction = prediction(GroundingPredictionStateV1::Occurrences {
            intervals: vec![FrameIntervalV1 {
                start_frame: 10,
                end_frame: 20,
            }],
        });
        prediction.artifact.audio_sha256 = sha256_hex(b"different audio bytes");
        let issues = score_temporal_grounding(&truth, &prediction, &plan()).unwrap_err();
        assert!(issues.contains(&TemporalGroundingIssue::ArtifactMismatch {
            field: "audio_sha256".into()
        }));
    }

    #[test]
    fn interval_past_audio_duration_is_rejected() {
        let truth = truth(
            GroundTruthStateV1::Present,
            vec![FrameIntervalV1 {
                start_frame: 479_999,
                end_frame: 480_001,
            }],
        );
        let issues = validate_temporal_ground_truth(&truth);
        assert!(issues.contains(&TemporalGroundingIssue::IntervalBeyondArtifact {
            index: 0
        }));
    }

    #[test]
    fn invalid_clock_and_zero_length_intervals_fail_closed() {
        let mut truth = truth(
            GroundTruthStateV1::Present,
            vec![FrameIntervalV1 {
                start_frame: 10,
                end_frame: 10,
            }],
        );
        truth.artifact.sample_rate_hz = 0;
        let issues = validate_temporal_ground_truth(&truth);
        assert!(issues.contains(&TemporalGroundingIssue::InvalidAudioClock {
            field: "sample_rate_hz".into()
        }));
        assert!(issues.contains(&TemporalGroundingIssue::InvalidInterval {
            index: 0
        }));
    }

    #[test]
    fn unsorted_and_duplicate_intervals_are_rejected() {
        let base = truth(
            GroundTruthStateV1::Present,
            vec![
                FrameIntervalV1 {
                    start_frame: 200,
                    end_frame: 300,
                },
                FrameIntervalV1 {
                    start_frame: 100,
                    end_frame: 150,
                },
            ],
        );
        assert!(validate_temporal_ground_truth(&base)
            .contains(&TemporalGroundingIssue::UnsortedIntervals { index: 1 }));

        let duplicate = truth(
            GroundTruthStateV1::Present,
            vec![
                FrameIntervalV1 {
                    start_frame: 100,
                    end_frame: 150,
                },
                FrameIntervalV1 {
                    start_frame: 100,
                    end_frame: 150,
                },
            ],
        );
        assert!(validate_temporal_ground_truth(&duplicate)
            .contains(&TemporalGroundingIssue::DuplicateInterval { index: 1 }));
    }

    #[test]
    fn ambiguous_prediction_is_an_explicit_abstention() {
        let truth = truth(
            GroundTruthStateV1::Present,
            vec![FrameIntervalV1 {
                start_frame: 100,
                end_frame: 200,
            }],
        );
        let prediction = prediction(GroundingPredictionStateV1::Ambiguous {
            candidate_intervals: vec![FrameIntervalV1 {
                start_frame: 95,
                end_frame: 205,
            }],
            reason: "two overlapping sources cannot be separated".into(),
        });
        let score = score_temporal_grounding(&truth, &prediction, &plan()).unwrap();
        assert!(score.abstained);
        assert_eq!(score.ambiguous_candidate_count, 1);
        assert_eq!(score.predicted_occurrences, 0);
        assert_eq!(score.false_negatives, 1);
    }

    #[test]
    fn input_commitment_binds_prediction_and_threshold() {
        let truth = truth(
            GroundTruthStateV1::Present,
            vec![FrameIntervalV1 {
                start_frame: 100,
                end_frame: 200,
            }],
        );
        let prediction = prediction(GroundingPredictionStateV1::Occurrences {
            intervals: vec![FrameIntervalV1 {
                start_frame: 100,
                end_frame: 200,
            }],
        });
        let original = temporal_grounding_input_commitment_sha256(
            &truth,
            &prediction,
            &plan(),
        )
        .unwrap();
        let mut changed_plan = plan();
        changed_plan.minimum_interval_iou = 0.7;
        let changed = temporal_grounding_input_commitment_sha256(
            &truth,
            &prediction,
            &changed_plan,
        )
        .unwrap();
        assert_ne!(original, changed);
    }
}
