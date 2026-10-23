// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Finite-sample conformal calibration for predictive model-adequacy audits.
//!
//! Input scores must be computed by the same predeclared statistic on independent, clean
//! calibration batches and on a fresh audit batch. Under exchangeability of those batch-level
//! scores, the rank p-value controls marginal false alarms without assuming a Gaussian residual
//! distribution. It does not detect every type of misspecification: a wrong law can have an
//! ordinary residual score on a particular audit batch, and covariate shift can invalidate
//! exchangeability. Use target-distribution audit batches or a separately justified shift-aware
//! method when the target distribution differs.
//!
//! This module never sees hidden ground truth or discrepancy amplitudes. It only evaluates a
//! caller-supplied batch score and independent calibration-score reference set.

use crate::model_adequacy::{AdequacyAssessment, AdequacyDiagnostic};

/// Errors from constructing or using a conformal adequacy calibrator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CalibrationError {
    /// The requested false-alarm level must be finite and strictly between 0 and 1.
    InvalidSignificanceLevel,
    /// The calibration reference set is empty.
    EmptyCalibrationSet,
    /// Too few calibration batches for the requested significance resolution.
    TooFewCalibrationBatches {
        actual: usize,
        minimum_required: usize,
    },
    /// A calibration score is negative or non-finite.
    InvalidCalibrationScore { index: usize },
    /// The audit batch score is negative or non-finite.
    InvalidAuditScore,
    /// The audit observations were not independent of model fitting/tuning data.
    AuditNotIndependent,
}

/// Result of comparing an independent audit batch with clean calibration batches.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConformalAdequacyResult {
    /// Nonconformity score for this independent audit batch.
    pub audit_score: f64,
    /// Conservative conformal rank p-value. Smaller values are stronger evidence against
    /// exchangeability with the clean calibration-score distribution.
    pub p_value: f64,
    /// Predeclared significance level.
    pub alpha: f64,
    /// Adequacy state suitable for the fail-closed model-adequacy decision layer.
    pub assessment: AdequacyAssessment,
}

/// Calibrator over batch-level predictive nonconformity scores.
///
/// Calibration scores must come from the same score function and a distribution intended
/// to represent clean future audit batches. They must be generated independently from data
/// used to fit/tune the candidate and must not be reused as the audit batch. Calibration
/// validity is a caller/data-protocol obligation; this type checks numeric and sample-size
/// preconditions but cannot prove exchangeability.
#[derive(Clone, Debug, PartialEq)]
pub struct ConformalAdequacyCalibrator {
    calibration_scores: Vec<f64>,
    alpha: f64,
}

impl ConformalAdequacyCalibrator {
    /// Construct a calibrator.
    ///
    /// The minimum sample size enforces that the smallest attainable p-value, 1/(n+1),
    /// can reach the declared alpha. At alpha=0.05, at least 19 independent clean calibration
    /// batches are required. This is only a p-value resolution requirement; larger samples
    /// may be needed for useful power across diverse scientific worlds.
    pub fn new(calibration_scores: &[f64], alpha: f64) -> Result<Self, CalibrationError> {
        if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
            return Err(CalibrationError::InvalidSignificanceLevel);
        }
        if calibration_scores.is_empty() {
            return Err(CalibrationError::EmptyCalibrationSet);
        }

        for (index, score) in calibration_scores.iter().copied().enumerate() {
            if !score.is_finite() || score < 0.0 {
                return Err(CalibrationError::InvalidCalibrationScore { index });
            }
        }

        let minimum_required = (1.0 / alpha).ceil() as usize - 1;
        if calibration_scores.len() < minimum_required {
            return Err(CalibrationError::TooFewCalibrationBatches {
                actual: calibration_scores.len(),
                minimum_required,
            });
        }

        Ok(Self {
            calibration_scores: calibration_scores.to_vec(),
            alpha,
        })
    }

    /// Compare an independent audit batch against the calibration reference.
    ///
    /// The p-value is (1 + number of calibration scores >= audit score) / (n + 1).
    /// Counting ties against the audit score makes the procedure conservative. An audit
    /// that is not independent of model fitting/tuning is rejected instead of being marked
    /// as calibrated.
    pub fn assess_audit_score(
        &self,
        audit_score: f64,
        independent_audit: bool,
    ) -> Result<ConformalAdequacyResult, CalibrationError> {
        let p_value = self.audit_p_value(audit_score, independent_audit)?;
        let assessment = if p_value <= self.alpha {
            AdequacyAssessment::Alarm {
                diagnostic: AdequacyDiagnostic::PredictiveMisfit,
                calibrated: true,
                independent_audit: true,
            }
        } else {
            AdequacyAssessment::Clear {
                calibrated: true,
                independent_audit: true,
            }
        };

        Ok(ConformalAdequacyResult {
            audit_score,
            p_value,
            alpha: self.alpha,
            assessment,
        })
    }

    /// Compute the conservative rank p-value without applying this calibrator's one-shot
    /// significance threshold. This is useful for a separately declared sequential or
    /// multiplicity-corrected policy that spends its own error budget.
    ///
    /// This method does not make repeated p-values independent or anytime-valid. The caller
    /// must provide an independent audit and preserve the same calibration/scoring protocol.
    pub fn audit_p_value(
        &self,
        audit_score: f64,
        independent_audit: bool,
    ) -> Result<f64, CalibrationError> {
        if !independent_audit {
            return Err(CalibrationError::AuditNotIndependent);
        }
        if !audit_score.is_finite() || audit_score < 0.0 {
            return Err(CalibrationError::InvalidAuditScore);
        }

        let no_less_extreme = self
            .calibration_scores
            .iter()
            .filter(|calibration_score| **calibration_score >= audit_score)
            .count();
        Ok((1 + no_less_extreme) as f64 / (self.calibration_scores.len() + 1) as f64)
    }

    /// Number of independent calibration batches represented.
    pub fn calibration_batch_count(&self) -> usize {
        self.calibration_scores.len()
    }

    /// The predeclared marginal false-alarm threshold.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }
}

/// Compute a batch-level RMS standardized residual score.
///
/// predictive_scales[i] is a positive scale declared by the predictive model for observation
/// i (for example, a predictive standard deviation). The result is a nonconformity score,
/// not a p-value; feed one score per independent calibration/audit batch to
/// ConformalAdequacyCalibrator. Return None on mismatched/empty arrays, invalid scales,
/// non-finite values, or arithmetic overflow. Caller must ensure observations represent the
/// intended audit distribution.
pub fn standardized_rms_residual(
    observed: &[f64],
    predicted: &[f64],
    predictive_scales: &[f64],
) -> Option<f64> {
    if observed.is_empty()
        || observed.len() != predicted.len()
        || observed.len() != predictive_scales.len()
    {
        return None;
    }

    let mut standardized = Vec::with_capacity(observed.len());
    for ((observed_value, predicted_value), scale) in observed
        .iter()
        .copied()
        .zip(predicted.iter().copied())
        .zip(predictive_scales.iter().copied())
    {
        if !observed_value.is_finite()
            || !predicted_value.is_finite()
            || !scale.is_finite()
            || scale <= 0.0
        {
            return None;
        }
        let residual = observed_value - predicted_value;
        if !residual.is_finite() {
            return None;
        }
        let score = residual / scale;
        if !score.is_finite() {
            return None;
        }
        standardized.push(score);
    }

    // Scaled sum-of-squares avoids premature overflow when the RMS itself is representable.
    let max_abs = standardized
        .iter()
        .map(|score| score.abs())
        .fold(0.0_f64, f64::max);
    if max_abs == 0.0 {
        return Some(0.0);
    }
    let mean_scaled_squares = standardized
        .iter()
        .map(|score| {
            let scaled = score / max_abs;
            scaled * scaled
        })
        .sum::<f64>()
        / standardized.len() as f64;
    let rms = max_abs * mean_scaled_squares.sqrt();
    rms.is_finite().then_some(rms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standardized_rms_residual_matches_hand_calculation() {
        let score = standardized_rms_residual(&[3.0, 2.0], &[1.0, 2.0], &[2.0, 1.0])
            .expect("valid residual batch");
        assert!((score - 0.5_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn standardized_rms_residual_is_zero_for_perfect_predictions() {
        assert_eq!(
            standardized_rms_residual(&[1.0, -2.0], &[1.0, -2.0], &[0.1, 100.0]),
            Some(0.0)
        );
    }

    #[test]
    fn standardized_rms_residual_fails_closed_on_invalid_inputs() {
        assert_eq!(standardized_rms_residual(&[], &[], &[]), None);
        assert_eq!(standardized_rms_residual(&[1.0], &[], &[1.0]), None);
        assert_eq!(standardized_rms_residual(&[1.0], &[0.0], &[0.0]), None);
        assert_eq!(standardized_rms_residual(&[f64::NAN], &[0.0], &[1.0]), None);
        assert_eq!(
            standardized_rms_residual(&[f64::MAX], &[-f64::MAX], &[1.0]),
            None
        );
    }

    #[test]
    fn alpha_point_zero_five_requires_nineteen_calibration_batches() {
        assert_eq!(
            ConformalAdequacyCalibrator::new(&[1.0; 18], 0.05),
            Err(CalibrationError::TooFewCalibrationBatches {
                actual: 18,
                minimum_required: 19,
            })
        );
        let calibrator = ConformalAdequacyCalibrator::new(&[1.0; 19], 0.05)
            .expect("19 batches resolve alpha=0.05");
        assert_eq!(calibrator.calibration_batch_count(), 19);
    }

    #[test]
    fn more_extreme_audit_score_can_raise_a_calibrated_alarm() {
        let calibration_scores = (1..=19).map(|value| value as f64).collect::<Vec<_>>();
        let calibrator = ConformalAdequacyCalibrator::new(&calibration_scores, 0.05)
            .expect("enough valid calibration batches");
        let result = calibrator
            .assess_audit_score(100.0, true)
            .expect("independent audit batch");
        assert!((result.p_value - 0.05).abs() < 1e-12);
        assert_eq!(
            result.assessment,
            AdequacyAssessment::Alarm {
                diagnostic: AdequacyDiagnostic::PredictiveMisfit,
                calibrated: true,
                independent_audit: true,
            }
        );
    }

    #[test]
    fn ordinary_audit_score_is_clear_and_ties_are_conservative() {
        let calibrator = ConformalAdequacyCalibrator::new(&[1.0; 19], 0.05)
            .expect("enough valid calibration batches");
        let result = calibrator
            .assess_audit_score(1.0, true)
            .expect("independent audit batch");
        assert_eq!(result.p_value, 1.0);
        assert_eq!(
            result.assessment,
            AdequacyAssessment::Clear {
                calibrated: true,
                independent_audit: true,
            }
        );
    }

    #[test]
    fn audit_must_be_independent_of_model_fitting_and_tuning() {
        let calibrator = ConformalAdequacyCalibrator::new(&[1.0; 19], 0.05)
            .expect("enough valid calibration batches");
        assert_eq!(
            calibrator.assess_audit_score(100.0, false),
            Err(CalibrationError::AuditNotIndependent)
        );
    }

    #[test]
    fn empty_calibration_set_is_rejected() {
        assert_eq!(
            ConformalAdequacyCalibrator::new(&[], 0.05),
            Err(CalibrationError::EmptyCalibrationSet)
        );
    }

    #[test]
    fn rank_p_value_is_bounded_and_monotone_in_audit_score() {
        let calibration_scores = (1..=50).map(|value| value as f64).collect::<Vec<_>>();
        let calibrator = ConformalAdequacyCalibrator::new(&calibration_scores, 0.05)
            .expect("enough valid calibration batches");
        let mut previous = 1.0;
        for score in [0.0, 10.0, 25.0, 50.0, 100.0] {
            let result = calibrator
                .assess_audit_score(score, true)
                .expect("independent audit batch");
            assert!((1.0 / 51.0..=1.0).contains(&result.p_value));
            assert!(result.p_value <= previous);
            previous = result.p_value;
        }
    }

    #[test]
    fn invalid_scores_and_significance_levels_are_rejected() {
        assert_eq!(
            ConformalAdequacyCalibrator::new(&[0.0, f64::NAN], 0.05),
            Err(CalibrationError::InvalidCalibrationScore { index: 1 })
        );
        assert_eq!(
            ConformalAdequacyCalibrator::new(&[1.0; 19], 0.0),
            Err(CalibrationError::InvalidSignificanceLevel)
        );
        let calibrator = ConformalAdequacyCalibrator::new(&[1.0; 19], 0.05).unwrap();
        assert_eq!(
            calibrator.assess_audit_score(f64::INFINITY, true),
            Err(CalibrationError::InvalidAuditScore)
        );
    }
}
