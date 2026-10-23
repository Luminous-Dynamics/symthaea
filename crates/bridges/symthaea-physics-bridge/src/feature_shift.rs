// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Fixed-reference feature-distribution shift scores, separate from predictive residual adequacy.
//!
//! A predictor can remain accurate on a batch while its inputs move outside the training/reference
//! distribution. Residual-only auditing does not detect this when the conditional relationship
//! remains correct. This module computes a batch-level feature-shift score from standardized mean,
//! scale, and pairwise-correlation changes against a fixed feature reference.
//!
//! The returned score is not a p-value and makes no analytic distributional guarantee. For
//! finite-sample marginal false-alarm control, pass scores from independent clean calibration
//! batches through a ConformalAdequacyCalibrator, then evaluate independent future batches against
//! the same fixed FeatureShiftScorer. The reference rows, feature order, scale floor, audit batch
//! size, and score definition must remain fixed throughout that calibration epoch.

use crate::adequacy_calibration::{
    CalibrationError, ConformalAdequacyCalibrator, ConformalAdequacyResult,
};
use crate::model_adequacy::{AdequacyAssessment, AdequacyDiagnostic};

/// Components of one fixed-reference feature-shift assessment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FeatureShiftScore {
    /// Maximum per-feature mean difference, standardized by reference scale and sqrt(batch size).
    pub normalized_mean_shift: f64,
    /// Maximum log-scale change, multiplied by sqrt(batch size / 2).
    pub normalized_scale_shift: f64,
    /// Maximum pairwise-correlation change multiplied by sqrt(batch size).
    pub normalized_correlation_shift: f64,
    /// Maximum of the three components; calibrate this scalar if a single alarm is required.
    pub aggregate: f64,
}

/// Errors from scoring and calibrating a feature-shift audit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FeatureShiftError {
    /// The audit batch is empty, malformed, non-finite, or has incompatible feature width.
    InvalidAuditBatch,
    /// The shared conformal calibrator rejected the audit contract.
    Calibration(CalibrationError),
}

/// Classified combination of model-adequacy and feature-distribution evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShiftTriageDecision {
    /// One or both channels are missing, uncalibrated, non-independent, or mislabeled.
    EvidenceInsufficient,
    /// Both channels are calibrated/independent and neither raised an alarm; not proof of truth.
    NoDetectedShift,
    /// A trusted model-adequacy alarm without a feature-distribution alarm.
    ModelAdequacySignalOnly,
    /// A trusted feature-distribution alarm without a model-adequacy alarm.
    FeatureDistributionShiftOnly,
    /// Both trusted channels raised alarms.
    JointAdequacyAndFeatureShift,
}

/// Classify two independently calibrated channels without conflating their meanings.
///
/// The first input is a model-adequacy assessment. The second is the feature-shift assessment
/// returned by `FeatureShiftScorer::assess`. That method labels an alarm as
/// `QueryDistributionShift`; this function rejects using that label in the model channel or using
/// another diagnostic label in the feature channel. A no-alarm result is not a proof of adequacy.
pub fn classify_shift_evidence(
    model_adequacy: AdequacyAssessment,
    feature_shift: AdequacyAssessment,
) -> ShiftTriageDecision {
    let model_alarm = match model_adequacy {
        AdequacyAssessment::Clear {
            calibrated: true,
            independent_audit: true,
        } => false,
        AdequacyAssessment::Alarm {
            diagnostic,
            calibrated: true,
            independent_audit: true,
        } if diagnostic != AdequacyDiagnostic::QueryDistributionShift => true,
        _ => return ShiftTriageDecision::EvidenceInsufficient,
    };

    let feature_alarm = match feature_shift {
        AdequacyAssessment::Clear {
            calibrated: true,
            independent_audit: true,
        } => false,
        AdequacyAssessment::Alarm {
            diagnostic: AdequacyDiagnostic::QueryDistributionShift,
            calibrated: true,
            independent_audit: true,
        } => true,
        _ => return ShiftTriageDecision::EvidenceInsufficient,
    };

    match (model_alarm, feature_alarm) {
        (false, false) => ShiftTriageDecision::NoDetectedShift,
        (true, false) => ShiftTriageDecision::ModelAdequacySignalOnly,
        (false, true) => ShiftTriageDecision::FeatureDistributionShiftOnly,
        (true, true) => ShiftTriageDecision::JointAdequacyAndFeatureShift,
    }
}

/// Fixed feature reference used to score independent calibration and audit batches.
#[derive(Clone, Debug, PartialEq)]
pub struct FeatureShiftScorer {
    reference_means: Vec<f64>,
    reference_scales: Vec<f64>,
    reference_correlations: Vec<f64>,
    scale_floor: f64,
}

impl FeatureShiftScorer {
    /// Fit a reference summary from at least two finite, equal-width feature rows.
    ///
    /// `scale_floor` is a positive minimum feature scale in the caller's declared feature units.
    /// It prevents division by zero for constant reference features; feature normalization and the
    /// floor are part of the frozen score contract and should be selected without audit data.
    pub fn fit(reference_rows: &[Vec<f64>], scale_floor: f64) -> Option<Self> {
        if !scale_floor.is_finite() || scale_floor <= 0.0 || reference_rows.len() < 2 {
            return None;
        }
        let feature_count = reference_rows.first()?.len();
        if feature_count == 0
            || reference_rows
                .iter()
                .any(|row| row.len() != feature_count || row.iter().any(|value| !value.is_finite()))
        {
            return None;
        }

        let reference_means = column_means(reference_rows, feature_count)?;
        let reference_scales = column_scales(reference_rows, &reference_means, feature_count)?;
        let reference_correlations = pairwise_correlations(
            reference_rows,
            &reference_means,
            &reference_scales,
            feature_count,
        )?;

        Some(Self {
            reference_means,
            reference_scales,
            reference_correlations,
            scale_floor,
        })
    }

    /// Score one fresh feature batch against this fixed reference.
    ///
    /// The audit needs at least two rows and exactly the same feature layout as the reference.
    /// A non-finite intermediate, incompatible batch, or invalid input fails closed with None.
    pub fn score(&self, audit_rows: &[Vec<f64>]) -> Option<FeatureShiftScore> {
        if audit_rows.len() < 2
            || audit_rows.iter().any(|row| {
                row.len() != self.reference_means.len()
                    || row.iter().any(|value| !value.is_finite())
            })
        {
            return None;
        }

        let feature_count = self.reference_means.len();
        let means = column_means(audit_rows, feature_count)?;
        let scales = column_scales(audit_rows, &means, feature_count)?;
        let correlations = pairwise_correlations(audit_rows, &means, &scales, feature_count)?;
        let n = audit_rows.len() as f64;

        let mut normalized_mean_shift = 0.0_f64;
        let mut normalized_scale_shift = 0.0_f64;
        for index in 0..feature_count {
            let reference_scale = self.reference_scales[index].max(self.scale_floor);
            let difference = (means[index] - self.reference_means[index]).abs();
            let mean_shift = difference / reference_scale * n.sqrt();
            let audit_log_scale = scales[index].max(self.scale_floor).ln();
            let reference_log_scale = self.reference_scales[index].max(self.scale_floor).ln();
            let scale_shift = (audit_log_scale - reference_log_scale).abs() * (n / 2.0).sqrt();
            if !mean_shift.is_finite() || !scale_shift.is_finite() {
                return None;
            }
            normalized_mean_shift = normalized_mean_shift.max(mean_shift);
            normalized_scale_shift = normalized_scale_shift.max(scale_shift);
        }

        let mut normalized_correlation_shift = 0.0_f64;
        for (reference, audit) in self.reference_correlations.iter().zip(correlations.iter()) {
            let shift = (audit - reference).abs() * n.sqrt();
            if !shift.is_finite() {
                return None;
            }
            normalized_correlation_shift = normalized_correlation_shift.max(shift);
        }

        let aggregate = normalized_mean_shift
            .max(normalized_scale_shift)
            .max(normalized_correlation_shift);
        aggregate.is_finite().then_some(FeatureShiftScore {
            normalized_mean_shift,
            normalized_scale_shift,
            normalized_correlation_shift,
            aggregate,
        })
    }

    /// Score and conformally assess one feature batch against this fixed reference.
    ///
    /// The supplied calibrator must have been fitted on independent clean batches scored against
    /// this exact reference profile. A feature alarm is labeled `QueryDistributionShift` so it
    /// remains distinct from predictive residual/model-adequacy failures.
    pub fn assess(
        &self,
        calibrator: &ConformalAdequacyCalibrator,
        audit_rows: &[Vec<f64>],
        independent_audit: bool,
    ) -> Result<ConformalAdequacyResult, FeatureShiftError> {
        let aggregate = self
            .score(audit_rows)
            .ok_or(FeatureShiftError::InvalidAuditBatch)?
            .aggregate;
        calibrator
            .assess_audit_score_with_diagnostic(
                aggregate,
                independent_audit,
                AdequacyDiagnostic::QueryDistributionShift,
            )
            .map_err(FeatureShiftError::Calibration)
    }

    /// Number of features represented by the fixed reference.
    pub fn feature_count(&self) -> usize {
        self.reference_means.len()
    }

    /// Number of pairwise correlations represented in the fixed reference.
    pub fn correlation_count(&self) -> usize {
        self.reference_correlations.len()
    }
}

fn column_means(rows: &[Vec<f64>], feature_count: usize) -> Option<Vec<f64>> {
    let mut sums = vec![0.0_f64; feature_count];
    for row in rows {
        for (sum, value) in sums.iter_mut().zip(row.iter()) {
            *sum += *value;
            if !sum.is_finite() {
                return None;
            }
        }
    }
    let denominator = rows.len() as f64;
    let means = sums
        .into_iter()
        .map(|sum| sum / denominator)
        .collect::<Vec<_>>();
    means.iter().all(|value| value.is_finite()).then_some(means)
}

fn column_scales(rows: &[Vec<f64>], means: &[f64], feature_count: usize) -> Option<Vec<f64>> {
    let mut sums_of_squares = vec![0.0_f64; feature_count];
    for row in rows {
        for index in 0..feature_count {
            let centered = row[index] - means[index];
            let square = centered * centered;
            if !centered.is_finite() || !square.is_finite() {
                return None;
            }
            sums_of_squares[index] += square;
            if !sums_of_squares[index].is_finite() {
                return None;
            }
        }
    }
    let denominator = rows.len() as f64;
    let scales = sums_of_squares
        .into_iter()
        .map(|sum| (sum / denominator).sqrt())
        .collect::<Vec<_>>();
    scales
        .iter()
        .all(|value| value.is_finite())
        .then_some(scales)
}

fn pairwise_correlations(
    rows: &[Vec<f64>],
    means: &[f64],
    scales: &[f64],
    feature_count: usize,
) -> Option<Vec<f64>> {
    let correlation_count = feature_count.saturating_mul(feature_count.saturating_sub(1)) / 2;
    let mut correlations = Vec::with_capacity(correlation_count);
    for left in 0..feature_count {
        for right in (left + 1)..feature_count {
            let left_scale = scales[left];
            let right_scale = scales[right];
            if left_scale == 0.0 || right_scale == 0.0 {
                correlations.push(0.0);
                continue;
            }

            let mut sum = 0.0_f64;
            for row in rows {
                let left_centered = (row[left] - means[left]) / left_scale;
                let right_centered = (row[right] - means[right]) / right_scale;
                sum += left_centered * right_centered;
                if !sum.is_finite() {
                    return None;
                }
            }
            let correlation = (sum / rows.len() as f64).clamp(-1.0, 1.0);
            if !correlation.is_finite() {
                return None;
            }
            correlations.push(correlation);
        }
    }
    Some(correlations)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows_with_correlation(len: usize, rho: f64) -> Vec<Vec<f64>> {
        (0..len)
            .map(|index| {
                // Balanced orthogonal +/-1 columns yield exactly zero reference correlation.
                let x = if index % 2 == 0 { 1.0 } else { -1.0 };
                let independent = if (index / 2) % 2 == 0 { 1.0 } else { -1.0 };
                let y = rho * x + (1.0 - rho * rho).sqrt() * independent;
                vec![x, y]
            })
            .collect()
    }

    #[test]
    fn fixed_reference_scores_detect_correlation_change_without_mean_shift() {
        let reference = rows_with_correlation(128, 0.0);
        let scorer = FeatureShiftScorer::fit(&reference, 1e-9).unwrap();
        let shifted = rows_with_correlation(128, 0.8);
        let score = scorer.score(&shifted).unwrap();
        assert!(score.normalized_correlation_shift > score.normalized_mean_shift);
        assert!(score.normalized_correlation_shift > 2.0);
        assert_eq!(scorer.feature_count(), 2);
        assert_eq!(scorer.correlation_count(), 1);
    }

    #[test]
    fn fixed_reference_separates_mean_and_scale_changes() {
        let reference = rows_with_correlation(128, 0.0);
        let scorer = FeatureShiftScorer::fit(&reference, 1e-9).unwrap();

        let mean_shift = reference
            .iter()
            .map(|row| vec![row[0] + 0.5, row[1]])
            .collect::<Vec<_>>();
        let mean_score = scorer.score(&mean_shift).unwrap();
        assert!(mean_score.normalized_mean_shift > 5.0);
        assert!(mean_score.normalized_mean_shift > mean_score.normalized_scale_shift);

        let scale_shift = reference
            .iter()
            .map(|row| vec![row[0] * 3.0, row[1]])
            .collect::<Vec<_>>();
        let scale_score = scorer.score(&scale_shift).unwrap();
        assert!(scale_score.normalized_scale_shift > 8.0);
        assert!(scale_score.normalized_scale_shift > scale_score.normalized_mean_shift);
    }

    fn clear() -> AdequacyAssessment {
        AdequacyAssessment::Clear {
            calibrated: true,
            independent_audit: true,
        }
    }

    fn alarm(diagnostic: AdequacyDiagnostic) -> AdequacyAssessment {
        AdequacyAssessment::Alarm {
            diagnostic,
            calibrated: true,
            independent_audit: true,
        }
    }

    #[test]
    fn calibrated_feature_alarm_keeps_its_distinct_diagnostic_label() {
        let reference = rows_with_correlation(128, 0.0);
        let scorer = FeatureShiftScorer::fit(&reference, 1e-9).unwrap();
        let calibrator = ConformalAdequacyCalibrator::new(&[0.0; 99], 0.05).unwrap();
        let shifted = rows_with_correlation(128, 0.8);
        let result = scorer.assess(&calibrator, &shifted, true).unwrap();
        assert!(matches!(
            result.assessment,
            AdequacyAssessment::Alarm {
                diagnostic: AdequacyDiagnostic::QueryDistributionShift,
                calibrated: true,
                independent_audit: true,
            }
        ));
    }

    #[test]
    fn shift_triage_keeps_model_and_feature_failures_separate() {
        assert_eq!(
            classify_shift_evidence(clear(), clear()),
            ShiftTriageDecision::NoDetectedShift
        );
        assert_eq!(
            classify_shift_evidence(alarm(AdequacyDiagnostic::PredictiveMisfit), clear()),
            ShiftTriageDecision::ModelAdequacySignalOnly
        );
        assert_eq!(
            classify_shift_evidence(clear(), alarm(AdequacyDiagnostic::QueryDistributionShift)),
            ShiftTriageDecision::FeatureDistributionShiftOnly
        );
        assert_eq!(
            classify_shift_evidence(
                alarm(AdequacyDiagnostic::CalibrationFailure),
                alarm(AdequacyDiagnostic::QueryDistributionShift)
            ),
            ShiftTriageDecision::JointAdequacyAndFeatureShift
        );
        assert_eq!(
            classify_shift_evidence(
                AdequacyAssessment::NotEvaluated,
                alarm(AdequacyDiagnostic::QueryDistributionShift)
            ),
            ShiftTriageDecision::EvidenceInsufficient
        );
        assert_eq!(
            classify_shift_evidence(alarm(AdequacyDiagnostic::QueryDistributionShift), clear()),
            ShiftTriageDecision::EvidenceInsufficient
        );
        assert_eq!(
            classify_shift_evidence(
                clear(),
                AdequacyAssessment::Alarm {
                    diagnostic: AdequacyDiagnostic::QueryDistributionShift,
                    calibrated: false,
                    independent_audit: true,
                }
            ),
            ShiftTriageDecision::EvidenceInsufficient
        );
    }

    #[test]
    fn fixed_reference_rejects_malformed_or_nonfinite_rows() {
        assert!(FeatureShiftScorer::fit(&[], 1e-9).is_none());
        assert!(FeatureShiftScorer::fit(&[vec![1.0]], 1e-9).is_none());
        assert!(FeatureShiftScorer::fit(&[vec![1.0, 2.0], vec![1.0]], 1e-9).is_none());
        assert!(FeatureShiftScorer::fit(&[vec![f64::NAN], vec![0.0]], 1e-9).is_none());
        assert!(FeatureShiftScorer::fit(&[vec![1.0], vec![2.0]], 0.0).is_none());

        let scorer = FeatureShiftScorer::fit(&[vec![0.0, 1.0], vec![1.0, 0.0]], 1e-9).unwrap();
        assert!(scorer.score(&[]).is_none());
        assert!(scorer.score(&[vec![0.0]]).is_none());
        assert!(
            scorer
                .score(&[vec![f64::INFINITY, 1.0], vec![0.0, 1.0]])
                .is_none()
        );
    }
}
