// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Tie-correct receiver-operating-characteristic curves and AUC.
//!
//! Scores are interpreted as evidence for the positive/signal class; larger
//! scores predict signal. All equal scores are admitted at the same threshold,
//! so ties are handled as groups rather than being ordered arbitrarily.
//!
//! The trapezoidal area is equivalent to the Mann–Whitney probability-of-
//! superiority estimator with half credit for tied positive/negative pairs.
//! AUC is a discrimination summary, not calibration, causal evidence, or a
//! substitute for a preregistered threshold/decision rule.

/// One scored binary observation. is_signal is the ground-truth class.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RocObservation {
    pub score: f64,
    pub is_signal: bool,
}

/// A point on a ROC curve. The threshold uses score >= threshold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RocPoint {
    /// False-positive rate: FP / (FP + TN).
    pub false_positive_rate: f64,
    /// True-positive rate: TP / (TP + FN).
    pub true_positive_rate: f64,
    /// Decision threshold. Positive infinity is the all-negative endpoint;
    /// negative infinity is the all-positive endpoint.
    pub threshold: f64,
}

/// Error returned when a ROC curve is not mathematically evaluable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RocError {
    EmptyInput,
    NonFiniteScore,
    MissingSignalClass,
    MissingNoiseClass,
}

/// A ROC curve and its trapezoidal area.
#[derive(Debug, Clone, PartialEq)]
pub struct RocCurve {
    pub signal_count: usize,
    pub noise_count: usize,
    /// Includes (0,0) and (1,1) endpoints, with score ties grouped.
    pub points: Vec<RocPoint>,
    /// Area under the ROC curve, in [0,1].
    pub auc: f64,
}

impl RocCurve {
    /// Compute ROC AUC using trapezoidal integration of ascending FPR points.
    pub fn compute_auc(points: &[RocPoint]) -> f64 {
        if points.len() < 2 {
            return 0.0;
        }
        points
            .windows(2)
            .map(|pair| {
                let dx = pair[1].false_positive_rate - pair[0].false_positive_rate;
                let mean_tpr =
                    0.5 * (pair[0].true_positive_rate + pair[1].true_positive_rate);
                dx * mean_tpr
            })
            .sum::<f64>()
            .clamp(0.0, 1.0)
    }
}

/// Construct a tie-correct ROC curve from scored observations.
///
/// Input requirements:
/// - at least one signal and one noise observation;
/// - every score must be finite;
/// - larger scores indicate the signal/positive class.
///
/// The implementation sorts once (O(n log n)), groups exact score ties,
/// and updates cumulative TP/FP counts per group (O(n)). It does not perform
/// an O(n × number_of_thresholds) threshold rescan.
pub fn try_roc_curve(observations: &[RocObservation]) -> Result<RocCurve, RocError> {
    if observations.is_empty() {
        return Err(RocError::EmptyInput);
    }
    if observations.iter().any(|item| !item.score.is_finite()) {
        return Err(RocError::NonFiniteScore);
    }

    let signal_count = observations.iter().filter(|item| item.is_signal).count();
    let noise_count = observations.len() - signal_count;
    if signal_count == 0 {
        return Err(RocError::MissingSignalClass);
    }
    if noise_count == 0 {
        return Err(RocError::MissingNoiseClass);
    }

    // Sort descending by score. total_cmp is well-defined because non-finite
    // scores have already been rejected. Equal numeric scores (including
    // signed zero) are grouped by == below.
    let mut sorted = observations.to_vec();
    sorted.sort_by(|a, b| b.score.total_cmp(&a.score));

    let mut points = Vec::with_capacity(sorted.len() + 1);
    points.push(RocPoint {
        false_positive_rate: 0.0,
        true_positive_rate: 0.0,
        threshold: f64::INFINITY,
    });

    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut index = 0usize;

    while index < sorted.len() {
        let threshold = sorted[index].score;
        let mut end = index + 1;
        while end < sorted.len() && sorted[end].score == threshold {
            end += 1;
        }

        // Admit all tied observations at this threshold together.
        for item in &sorted[index..end] {
            if item.is_signal {
                tp += 1;
            } else {
                fp += 1;
            }
        }

        points.push(RocPoint {
            false_positive_rate: fp as f64 / noise_count as f64,
            true_positive_rate: tp as f64 / signal_count as f64,
            threshold,
        });
        index = end;
    }

    // With both classes present this is exactly (1,1), but keep the explicit
    // endpoint to make the contract clear and stable.
    if points.last().is_some_and(|point| {
        point.false_positive_rate != 1.0 || point.true_positive_rate != 1.0
    }) {
        points.push(RocPoint {
            false_positive_rate: 1.0,
            true_positive_rate: 1.0,
            threshold: f64::NEG_INFINITY,
        });
    }

    let auc = RocCurve::compute_auc(&points);
    Ok(RocCurve {
        signal_count,
        noise_count,
        points,
        auc,
    })
}

/// One point on a precision–recall curve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrecisionRecallPoint {
    /// Positive predictive value: TP / (TP + FP).
    pub precision: f64,
    /// Recall / sensitivity: TP / (TP + FN).
    pub recall: f64,
    /// Decision threshold. Positive infinity is the no-positive-call endpoint.
    pub threshold: f64,
}

/// A precision–recall curve and non-interpolated average precision.
#[derive(Debug, Clone, PartialEq)]
pub struct PrecisionRecallCurve {
    pub signal_count: usize,
    pub noise_count: usize,
    /// Includes the conventional (precision=1, recall=0) starting point.
    pub points: Vec<PrecisionRecallPoint>,
    /// Sum of each recall increment times precision at that operating point.
    ///
    /// This is non-interpolated average precision, not trapezoidal PR area.
    pub average_precision: f64,
}

/// Construct a tie-correct precision–recall curve from scored observations.
///
/// This is particularly useful when event-present intervals are rare. It uses
/// the same score direction and exact tie grouping as the ROC routine, and
/// requires both classes to be present. Average precision is calculated as
/// sum(delta_recall * precision_at_group), with no interpolation.
pub fn try_precision_recall_curve(
    observations: &[RocObservation],
) -> Result<PrecisionRecallCurve, RocError> {
    if observations.is_empty() {
        return Err(RocError::EmptyInput);
    }
    if observations.iter().any(|item| !item.score.is_finite()) {
        return Err(RocError::NonFiniteScore);
    }

    let signal_count = observations.iter().filter(|item| item.is_signal).count();
    let noise_count = observations.len() - signal_count;
    if signal_count == 0 {
        return Err(RocError::MissingSignalClass);
    }
    if noise_count == 0 {
        return Err(RocError::MissingNoiseClass);
    }

    let mut sorted = observations.to_vec();
    sorted.sort_by(|a, b| b.score.total_cmp(&a.score));

    let mut points = Vec::with_capacity(sorted.len() + 1);
    points.push(PrecisionRecallPoint {
        precision: 1.0,
        recall: 0.0,
        threshold: f64::INFINITY,
    });

    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut index = 0usize;
    let mut previous_recall = 0.0;
    let mut average_precision = 0.0;

    while index < sorted.len() {
        let threshold = sorted[index].score;
        let mut end = index + 1;
        while end < sorted.len() && sorted[end].score == threshold {
            end += 1;
        }

        for item in &sorted[index..end] {
            if item.is_signal {
                tp += 1;
            } else {
                fp += 1;
            }
        }

        let recall = tp as f64 / signal_count as f64;
        let precision = tp as f64 / (tp + fp) as f64;
        average_precision += (recall - previous_recall) * precision;
        points.push(PrecisionRecallPoint {
            precision,
            recall,
            threshold,
        });
        previous_recall = recall;
        index = end;
    }

    Ok(PrecisionRecallCurve {
        signal_count,
        noise_count,
        points,
        average_precision: average_precision.clamp(0.0, 1.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(score: f64, is_signal: bool) -> RocObservation {
        RocObservation { score, is_signal }
    }

    #[test]
    fn perfect_ranking_has_auc_one() {
        let curve = try_roc_curve(&[
            item(0.9, true),
            item(0.8, true),
            item(0.2, false),
            item(0.1, false),
        ])
        .unwrap();

        assert_eq!(curve.signal_count, 2);
        assert_eq!(curve.noise_count, 2);
        assert!((curve.auc - 1.0).abs() < 1e-12);
        assert_eq!(curve.points.first().unwrap().false_positive_rate, 0.0);
        assert_eq!(curve.points.first().unwrap().true_positive_rate, 0.0);
        assert_eq!(curve.points.last().unwrap().false_positive_rate, 1.0);
        assert_eq!(curve.points.last().unwrap().true_positive_rate, 1.0);
    }

    #[test]
    fn reversed_ranking_has_auc_zero() {
        let curve = try_roc_curve(&[
            item(0.1, true),
            item(0.2, true),
            item(0.8, false),
            item(0.9, false),
        ])
        .unwrap();
        assert!((curve.auc - 0.0).abs() < 1e-12);
    }

    #[test]
    fn tied_positive_and_negative_scores_receive_half_credit() {
        let curve = try_roc_curve(&[item(1.0, true), item(1.0, false)]).unwrap();
        assert!((curve.auc - 0.5).abs() < 1e-12);
        // Both observations enter at one threshold: no arbitrary intermediate
        // point may claim that one tied class scored ahead of the other.
        assert_eq!(curve.points.len(), 2);
    }

    #[test]
    fn non_tied_partial_ranking_matches_pairwise_reference() {
        let curve = try_roc_curve(&[
            item(0.9, true),
            item(0.8, false),
            item(0.7, true),
            item(0.6, false),
        ])
        .unwrap();
        assert!((curve.auc - 0.75).abs() < 1e-12);
    }

    #[test]
    fn perfect_ranking_has_average_precision_one() {
        let curve = try_precision_recall_curve(&[
            item(0.9, true),
            item(0.8, true),
            item(0.2, false),
            item(0.1, false),
        ])
        .unwrap();
        assert!((curve.average_precision - 1.0).abs() < 1e-12);
        assert_eq!(curve.points.first().unwrap().precision, 1.0);
        assert_eq!(curve.points.first().unwrap().recall, 0.0);
        assert_eq!(curve.points.last().unwrap().recall, 1.0);
    }

    #[test]
    fn tied_positive_and_negative_scores_have_half_average_precision() {
        let curve = try_precision_recall_curve(&[item(1.0, true), item(1.0, false)]).unwrap();
        assert!((curve.average_precision - 0.5).abs() < 1e-12);
        assert_eq!(curve.points.len(), 2);
    }

    #[test]
    fn average_precision_penalizes_early_false_alarms() {
        let clean = try_precision_recall_curve(&[
            item(0.9, true),
            item(0.8, true),
            item(0.2, false),
            item(0.1, false),
        ])
        .unwrap();
        let noisy = try_precision_recall_curve(&[
            item(0.9, false),
            item(0.8, true),
            item(0.7, false),
            item(0.6, true),
        ])
        .unwrap();
        assert!(clean.average_precision > noisy.average_precision);
        assert!((noisy.average_precision - 0.5).abs() < 1e-12);
    }

    #[test]
    fn rejects_single_class_for_precision_recall() {
        assert_eq!(
            try_precision_recall_curve(&[item(0.9, true), item(0.8, true)]),
            Err(RocError::MissingNoiseClass)
        );
        assert_eq!(
            try_precision_recall_curve(&[item(0.9, false), item(0.8, false)]),
            Err(RocError::MissingSignalClass)
        );
    }

    #[test]
    fn rejects_empty_nonfinite_and_single_class_inputs() {
        assert_eq!(try_roc_curve(&[]), Err(RocError::EmptyInput));
        assert_eq!(
            try_roc_curve(&[item(f64::NAN, true), item(0.0, false)]),
            Err(RocError::NonFiniteScore)
        );
        assert_eq!(
            try_roc_curve(&[item(0.9, true), item(0.8, true)]),
            Err(RocError::MissingNoiseClass)
        );
        assert_eq!(
            try_roc_curve(&[item(0.9, false), item(0.8, false)]),
            Err(RocError::MissingSignalClass)
        );
    }
}
