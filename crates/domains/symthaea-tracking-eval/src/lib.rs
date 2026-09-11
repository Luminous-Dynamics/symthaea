// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic point-track evaluation for trackers that do not yet emit
//! benchmark-grade bounding boxes.
//!
//! This crate intentionally does **not** label its metrics HOTA, IDF1, or MOTA.
//! Those standard metrics require representation and association semantics that
//! must be implemented faithfully. Here we measure exactly what a centroid-only
//! tracker can support: TP/FP/FN under a reviewed spatial gate, identity switches,
//! unmatched tracks, and negative-only false-alarm behavior.

#![deny(unsafe_code)]

use std::collections::{BTreeSet, HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2 {
    pub x_norm: f64,
    pub y_norm: f64,
}

impl Point2 {
    pub fn validate(self) -> bool {
        self.x_norm.is_finite()
            && self.y_norm.is_finite()
            && (0.0..=1.0).contains(&self.x_norm)
            && (0.0..=1.0).contains(&self.y_norm)
    }

    pub fn distance(self, other: Self) -> f64 {
        let dx = self.x_norm - other.x_norm;
        let dy = self.y_norm - other.y_norm;
        (dx * dx + dy * dy).sqrt()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GroundTruthPoint {
    pub object_id: u64,
    pub point: Point2,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PredictedPoint {
    pub track_id: u64,
    pub point: Point2,
}

/// Explicit frame record. Keeping empty frames is important because long runs of
/// "nothing happened" are where false-positive behavior becomes visible.
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationFrame {
    pub frame_index: u64,
    pub ground_truth: Vec<GroundTruthPoint>,
    pub predictions: Vec<PredictedPoint>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackingEvaluationPolicy {
    /// Maximum normalized image-plane Euclidean distance for an admissible match.
    pub maximum_match_distance_norm: f64,
}

impl TrackingEvaluationPolicy {
    pub fn validate(self) -> bool {
        self.maximum_match_distance_norm.is_finite()
            && self.maximum_match_distance_norm > 0.0
            && self.maximum_match_distance_norm <= std::f64::consts::SQRT_2
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackingEvaluationError {
    InvalidPolicy,
    DuplicateFrameIndex,
    DuplicateGroundTruthId,
    DuplicatePredictionTrackId,
    InvalidCoordinate,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PointMatch {
    pub ground_truth_id: u64,
    pub prediction_track_id: u64,
    pub distance_norm: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FrameEvaluation {
    pub frame_index: u64,
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub matches: Vec<PointMatch>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrackingEvaluationReport {
    pub evaluated_frames: usize,
    pub ground_truth_points: usize,
    pub prediction_points: usize,
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub identity_switches: usize,
    /// Distinct prediction track IDs that never matched any ground-truth object.
    pub false_track_count: usize,
    pub unique_prediction_tracks: usize,
    pub frames_with_false_positives: usize,
    pub negative_frames: usize,
    pub negative_frames_with_false_positives: usize,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f1: Option<f64>,
    pub mean_match_distance_norm: Option<f64>,
    pub false_positives_per_frame: f64,
    pub clean_negative_frame_fraction: Option<f64>,
    pub false_track_fraction: Option<f64>,
    pub negative_only_sequence: bool,
    pub frames: Vec<FrameEvaluation>,
}

impl TrackingEvaluationReport {
    /// Evaluation metrics are evidence about a tracker, never physical authority.
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

pub fn evaluate_sequence(
    frames: &[EvaluationFrame],
    policy: TrackingEvaluationPolicy,
) -> Result<TrackingEvaluationReport, TrackingEvaluationError> {
    if !policy.validate() {
        return Err(TrackingEvaluationError::InvalidPolicy);
    }

    let mut ordered = frames.to_vec();
    ordered.sort_by_key(|frame| frame.frame_index);
    if ordered
        .windows(2)
        .any(|pair| pair[0].frame_index == pair[1].frame_index)
    {
        return Err(TrackingEvaluationError::DuplicateFrameIndex);
    }

    let mut report_frames = Vec::with_capacity(ordered.len());
    let mut total_gt = 0usize;
    let mut total_predictions = 0usize;
    let mut total_tp = 0usize;
    let mut total_fp = 0usize;
    let mut total_fn = 0usize;
    let mut identity_switches = 0usize;
    let mut match_distance_sum = 0.0;
    let mut last_track_for_gt = HashMap::<u64, u64>::new();
    let mut all_prediction_tracks = HashSet::<u64>::new();
    let mut matched_prediction_tracks = HashSet::<u64>::new();
    let mut frames_with_fp = 0usize;
    let mut negative_frames = 0usize;
    let mut negative_frames_with_fp = 0usize;

    for frame in &ordered {
        validate_frame(frame)?;
        let evaluation = evaluate_frame(frame, policy.maximum_match_distance_norm);

        total_gt += frame.ground_truth.len();
        total_predictions += frame.predictions.len();
        total_tp += evaluation.true_positives;
        total_fp += evaluation.false_positives;
        total_fn += evaluation.false_negatives;
        all_prediction_tracks.extend(frame.predictions.iter().map(|prediction| prediction.track_id));

        if evaluation.false_positives > 0 {
            frames_with_fp += 1;
        }
        if frame.ground_truth.is_empty() {
            negative_frames += 1;
            if evaluation.false_positives > 0 {
                negative_frames_with_fp += 1;
            }
        }

        for matched in &evaluation.matches {
            match_distance_sum += matched.distance_norm;
            matched_prediction_tracks.insert(matched.prediction_track_id);
            if let Some(previous_track) = last_track_for_gt.insert(
                matched.ground_truth_id,
                matched.prediction_track_id,
            ) {
                if previous_track != matched.prediction_track_id {
                    identity_switches += 1;
                }
            }
        }

        report_frames.push(evaluation);
    }

    let precision = ratio(total_tp, total_tp + total_fp);
    let recall = ratio(total_tp, total_tp + total_fn);
    let f1 = match (precision, recall) {
        (Some(p), Some(r)) if p + r > 0.0 => Some(2.0 * p * r / (p + r)),
        (Some(_), Some(_)) => Some(0.0),
        _ => None,
    };
    let false_track_count = all_prediction_tracks
        .difference(&matched_prediction_tracks)
        .count();
    let evaluated_frames = ordered.len();

    Ok(TrackingEvaluationReport {
        evaluated_frames,
        ground_truth_points: total_gt,
        prediction_points: total_predictions,
        true_positives: total_tp,
        false_positives: total_fp,
        false_negatives: total_fn,
        identity_switches,
        false_track_count,
        unique_prediction_tracks: all_prediction_tracks.len(),
        frames_with_false_positives: frames_with_fp,
        negative_frames,
        negative_frames_with_false_positives,
        precision,
        recall,
        f1,
        mean_match_distance_norm: (total_tp > 0).then_some(match_distance_sum / total_tp as f64),
        false_positives_per_frame: if evaluated_frames == 0 {
            0.0
        } else {
            total_fp as f64 / evaluated_frames as f64
        },
        clean_negative_frame_fraction: (negative_frames > 0).then_some(
            (negative_frames - negative_frames_with_fp) as f64 / negative_frames as f64,
        ),
        false_track_fraction: (!all_prediction_tracks.is_empty())
            .then_some(false_track_count as f64 / all_prediction_tracks.len() as f64),
        negative_only_sequence: total_gt == 0,
        frames: report_frames,
    })
}

fn validate_frame(frame: &EvaluationFrame) -> Result<(), TrackingEvaluationError> {
    let mut gt_ids = BTreeSet::new();
    for truth in &frame.ground_truth {
        if !truth.point.validate() {
            return Err(TrackingEvaluationError::InvalidCoordinate);
        }
        if !gt_ids.insert(truth.object_id) {
            return Err(TrackingEvaluationError::DuplicateGroundTruthId);
        }
    }

    let mut prediction_ids = BTreeSet::new();
    for prediction in &frame.predictions {
        if !prediction.point.validate() {
            return Err(TrackingEvaluationError::InvalidCoordinate);
        }
        if !prediction_ids.insert(prediction.track_id) {
            return Err(TrackingEvaluationError::DuplicatePredictionTrackId);
        }
    }
    Ok(())
}

fn evaluate_frame(frame: &EvaluationFrame, maximum_distance: f64) -> FrameEvaluation {
    let mut truths = frame.ground_truth.clone();
    let mut predictions = frame.predictions.clone();
    truths.sort_by_key(|truth| truth.object_id);
    predictions.sort_by_key(|prediction| prediction.track_id);

    let mut candidates = Vec::<Vec<usize>>::with_capacity(truths.len());
    for truth in &truths {
        let mut options = predictions
            .iter()
            .enumerate()
            .filter_map(|(prediction_index, prediction)| {
                let distance = truth.point.distance(prediction.point);
                (distance <= maximum_distance).then_some((prediction_index, distance, prediction.track_id))
            })
            .collect::<Vec<_>>();
        options.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.2.cmp(&b.2)));
        candidates.push(options.into_iter().map(|entry| entry.0).collect());
    }

    // Deterministic maximum-cardinality bipartite matching. Candidate lists are
    // nearest-first, but this is not advertised as a globally minimum-distance
    // assignment. That distinction matters when interpreting distance metrics.
    let mut prediction_to_truth = vec![None::<usize>; predictions.len()];
    for truth_index in 0..truths.len() {
        let mut visited_predictions = vec![false; predictions.len()];
        let _ = augment_match(
            truth_index,
            &candidates,
            &mut visited_predictions,
            &mut prediction_to_truth,
        );
    }

    let mut matches = prediction_to_truth
        .iter()
        .enumerate()
        .filter_map(|(prediction_index, truth_index)| {
            truth_index.map(|truth_index| PointMatch {
                ground_truth_id: truths[truth_index].object_id,
                prediction_track_id: predictions[prediction_index].track_id,
                distance_norm: truths[truth_index]
                    .point
                    .distance(predictions[prediction_index].point),
            })
        })
        .collect::<Vec<_>>();
    matches.sort_by_key(|matched| (matched.ground_truth_id, matched.prediction_track_id));

    let true_positives = matches.len();
    FrameEvaluation {
        frame_index: frame.frame_index,
        true_positives,
        false_positives: predictions.len().saturating_sub(true_positives),
        false_negatives: truths.len().saturating_sub(true_positives),
        matches,
    }
}

fn augment_match(
    truth_index: usize,
    candidates: &[Vec<usize>],
    visited_predictions: &mut [bool],
    prediction_to_truth: &mut [Option<usize>],
) -> bool {
    for &prediction_index in &candidates[truth_index] {
        if visited_predictions[prediction_index] {
            continue;
        }
        visited_predictions[prediction_index] = true;
        let can_take = match prediction_to_truth[prediction_index] {
            None => true,
            Some(previous_truth) => augment_match(
                previous_truth,
                candidates,
                visited_predictions,
                prediction_to_truth,
            ),
        };
        if can_take {
            prediction_to_truth[prediction_index] = Some(truth_index);
            return true;
        }
    }
    false
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then_some(numerator as f64 / denominator as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(x: f64, y: f64) -> Point2 {
        Point2 {
            x_norm: x,
            y_norm: y,
        }
    }

    fn policy() -> TrackingEvaluationPolicy {
        TrackingEvaluationPolicy {
            maximum_match_distance_norm: 0.08,
        }
    }

    #[test]
    fn perfect_sequence_is_exact() {
        let frames = vec![
            EvaluationFrame {
                frame_index: 1,
                ground_truth: vec![GroundTruthPoint {
                    object_id: 10,
                    point: point(0.2, 0.2),
                }],
                predictions: vec![PredictedPoint {
                    track_id: 7,
                    point: point(0.2, 0.2),
                }],
            },
            EvaluationFrame {
                frame_index: 2,
                ground_truth: vec![GroundTruthPoint {
                    object_id: 10,
                    point: point(0.3, 0.2),
                }],
                predictions: vec![PredictedPoint {
                    track_id: 7,
                    point: point(0.3, 0.2),
                }],
            },
        ];
        let report = evaluate_sequence(&frames, policy()).unwrap();
        assert_eq!(report.true_positives, 2);
        assert_eq!(report.false_positives, 0);
        assert_eq!(report.false_negatives, 0);
        assert_eq!(report.identity_switches, 0);
        assert_eq!(report.precision, Some(1.0));
        assert_eq!(report.recall, Some(1.0));
        assert!(!report.grants_physical_authority());
    }

    #[test]
    fn negative_only_sequence_exposes_false_alarms() {
        let frames = vec![
            EvaluationFrame {
                frame_index: 1,
                ground_truth: vec![],
                predictions: vec![],
            },
            EvaluationFrame {
                frame_index: 2,
                ground_truth: vec![],
                predictions: vec![PredictedPoint {
                    track_id: 99,
                    point: point(0.5, 0.5),
                }],
            },
            EvaluationFrame {
                frame_index: 3,
                ground_truth: vec![],
                predictions: vec![],
            },
        ];
        let report = evaluate_sequence(&frames, policy()).unwrap();
        assert!(report.negative_only_sequence);
        assert_eq!(report.false_positives, 1);
        assert_eq!(report.false_track_count, 1);
        assert_eq!(report.negative_frames, 3);
        assert_eq!(report.negative_frames_with_false_positives, 1);
        assert_eq!(report.clean_negative_frame_fraction, Some(2.0 / 3.0));
        assert_eq!(report.precision, Some(0.0));
        assert_eq!(report.recall, None);
    }

    #[test]
    fn missed_and_extra_points_are_counted() {
        let frame = EvaluationFrame {
            frame_index: 1,
            ground_truth: vec![
                GroundTruthPoint {
                    object_id: 1,
                    point: point(0.1, 0.1),
                },
                GroundTruthPoint {
                    object_id: 2,
                    point: point(0.9, 0.9),
                },
            ],
            predictions: vec![
                PredictedPoint {
                    track_id: 10,
                    point: point(0.1, 0.1),
                },
                PredictedPoint {
                    track_id: 11,
                    point: point(0.5, 0.5),
                },
            ],
        };
        let report = evaluate_sequence(&[frame], policy()).unwrap();
        assert_eq!(report.true_positives, 1);
        assert_eq!(report.false_positives, 1);
        assert_eq!(report.false_negatives, 1);
        assert_eq!(report.false_track_count, 1);
    }

    #[test]
    fn track_reassignment_counts_identity_switch() {
        let frames = vec![
            EvaluationFrame {
                frame_index: 1,
                ground_truth: vec![GroundTruthPoint {
                    object_id: 1,
                    point: point(0.2, 0.2),
                }],
                predictions: vec![PredictedPoint {
                    track_id: 10,
                    point: point(0.2, 0.2),
                }],
            },
            EvaluationFrame {
                frame_index: 2,
                ground_truth: vec![GroundTruthPoint {
                    object_id: 1,
                    point: point(0.25, 0.2),
                }],
                predictions: vec![PredictedPoint {
                    track_id: 11,
                    point: point(0.25, 0.2),
                }],
            },
        ];
        let report = evaluate_sequence(&frames, policy()).unwrap();
        assert_eq!(report.identity_switches, 1);
    }

    #[test]
    fn matching_maximizes_cardinality_under_gate() {
        // GT 1 can match either prediction; GT 2 can match only prediction 10.
        // A purely greedy one-pass matcher could waste prediction 10 on GT 1.
        let frame = EvaluationFrame {
            frame_index: 1,
            ground_truth: vec![
                GroundTruthPoint {
                    object_id: 1,
                    point: point(0.50, 0.50),
                },
                GroundTruthPoint {
                    object_id: 2,
                    point: point(0.44, 0.50),
                },
            ],
            predictions: vec![
                PredictedPoint {
                    track_id: 10,
                    point: point(0.45, 0.50),
                },
                PredictedPoint {
                    track_id: 11,
                    point: point(0.56, 0.50),
                },
            ],
        };
        let report = evaluate_sequence(&[frame], policy()).unwrap();
        assert_eq!(report.true_positives, 2);
        assert_eq!(report.false_positives, 0);
        assert_eq!(report.false_negatives, 0);
    }

    #[test]
    fn duplicate_ids_fail_closed() {
        let frame = EvaluationFrame {
            frame_index: 1,
            ground_truth: vec![],
            predictions: vec![
                PredictedPoint {
                    track_id: 10,
                    point: point(0.1, 0.1),
                },
                PredictedPoint {
                    track_id: 10,
                    point: point(0.2, 0.2),
                },
            ],
        };
        assert_eq!(
            evaluate_sequence(&[frame], policy()),
            Err(TrackingEvaluationError::DuplicatePredictionTrackId)
        );
    }
}
