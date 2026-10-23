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
//! exchangeability. A single RMS score can be insensitive to localized outliers or residual
//! dependence. The multi-view score below combines predeclared batch diagnostics into one scalar
//! before conformal calibration; calibration is then applied to that aggregate statistic, not to
//! each component separately. Use target-distribution audit batches or a separately justified
//! shift-aware method when the target distribution differs.
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
        self.assess_audit_score_with_diagnostic(
            audit_score,
            independent_audit,
            AdequacyDiagnostic::PredictiveMisfit,
        )
    }

    /// Calibrate an audit score while preserving the meaning of the diagnostic channel.
    ///
    /// For example, residual scores should use `PredictiveMisfit`, while a separately
    /// calibrated input-feature score should use `QueryDistributionShift`. The score function
    /// and diagnostic label must be predeclared; changing a label cannot repair a mismatched or
    /// non-exchangeable calibration protocol.
    pub fn assess_audit_score_with_diagnostic(
        &self,
        audit_score: f64,
        independent_audit: bool,
        diagnostic: AdequacyDiagnostic,
    ) -> Result<ConformalAdequacyResult, CalibrationError> {
        let p_value = self.audit_p_value(audit_score, independent_audit)?;
        let assessment = if p_value <= self.alpha {
            AdequacyAssessment::Alarm {
                diagnostic,
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

/// Predeclared component diagnostics for one standardized-residual batch.
///
/// Every field is a non-negative score; larger is more anomalous. The normalizers are fixed
/// sensitivity heuristics, not analytic p-values. Use the *same* batch size, ordering convention,
/// fields, and implementation for independent clean calibration batches and audit batches.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResidualDiagnosticScores {
    /// Root-mean-square standardized residual across the batch.
    pub rms_standardized_residual: f64,
    /// Maximum absolute standardized residual divided by sqrt(2 ln(n + 1)).
    pub normalized_peak_residual: f64,
    /// sqrt(n) times absolute mean standardized residual.
    pub normalized_mean_bias: f64,
    /// sqrt(n - 1) times absolute lag-one residual correlation (zero for n < 2 or undefined variance).
    pub normalized_lag_one_dependence: f64,
}

impl ResidualDiagnosticScores {
    /// Return the component scores in the stable order used by score-fusion calibration.
    pub fn as_array(self) -> [f64; 4] {
        [
            self.rms_standardized_residual,
            self.normalized_peak_residual,
            self.normalized_mean_bias,
            self.normalized_lag_one_dependence,
        ]
    }

    /// Maximum of the predeclared components.
    pub fn maximum(self) -> f64 {
        self.as_array().into_iter().fold(0.0_f64, f64::max)
    }
}

/// Compute complementary residual diagnostics for one fixed-size audit batch.
///
/// The returned components target broad misfit, localized outliers, systematic residual bias,
/// and lag-one dependence. The normalizers do not rely on a Gaussian analytic null: finite-sample
/// validity, if required, comes from calibrating a frozen scalar aggregation on independent clean
/// batches. Do not threshold these components analytically or interpret them as p-values.
pub fn standardized_residual_diagnostics(
    observed: &[f64],
    predicted: &[f64],
    predictive_scales: &[f64],
) -> Option<ResidualDiagnosticScores> {
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

    let n = standardized.len();
    let max_abs = standardized
        .iter()
        .map(|score| score.abs())
        .fold(0.0_f64, f64::max);
    if max_abs == 0.0 {
        return Some(ResidualDiagnosticScores {
            rms_standardized_residual: 0.0,
            normalized_peak_residual: 0.0,
            normalized_mean_bias: 0.0,
            normalized_lag_one_dependence: 0.0,
        });
    }

    // Normalize before summing to avoid premature overflow in the moments.
    let normalized = standardized
        .iter()
        .map(|score| score / max_abs)
        .collect::<Vec<_>>();
    let mean_normalized = normalized.iter().sum::<f64>() / n as f64;
    let mean_scaled_squares = normalized.iter().map(|score| score * score).sum::<f64>() / n as f64;
    let rms = max_abs * mean_scaled_squares.sqrt();

    // Fixed scale factors make the raw components more comparable; they are not null p-values.
    let peak_normalizer = (2.0 * ((n as f64) + 1.0).ln()).sqrt();
    let peak = max_abs / peak_normalizer;
    let bias = max_abs * mean_normalized.abs() * (n as f64).sqrt();

    let serial = if n < 2 {
        0.0
    } else {
        let centered = normalized
            .iter()
            .map(|score| score - mean_normalized)
            .collect::<Vec<_>>();
        let numerator = centered
            .windows(2)
            .map(|pair| pair[0] * pair[1])
            .sum::<f64>();
        let left_sum_squares = centered[..n - 1]
            .iter()
            .map(|score| score * score)
            .sum::<f64>();
        let right_sum_squares = centered[1..].iter().map(|score| score * score).sum::<f64>();
        let denominator = left_sum_squares.sqrt() * right_sum_squares.sqrt();
        if denominator == 0.0 {
            0.0
        } else {
            (numerator / denominator).abs().min(1.0) * ((n - 1) as f64).sqrt()
        }
    };

    let scores = ResidualDiagnosticScores {
        rms_standardized_residual: rms,
        normalized_peak_residual: peak,
        normalized_mean_bias: bias,
        normalized_lag_one_dependence: serial,
    };
    scores
        .as_array()
        .iter()
        .all(|score| score.is_finite() && *score >= 0.0)
        .then_some(scores)
}

/// Maximum-statistic score for the four predeclared residual diagnostics.
///
/// Calibrate this *single returned score* on independent clean batches using
/// `ConformalAdequacyCalibrator` before thresholding it. The maximum retains a single scalar
/// calibration contract but can be less powerful than RMS for some alternatives; compare it with
/// each constituent score at matched false-alarm and measurement budgets.
pub fn multi_view_standardized_residual_score(
    observed: &[f64],
    predicted: &[f64],
    predictive_scales: &[f64],
) -> Option<f64> {
    standardized_residual_diagnostics(observed, predicted, predictive_scales)
        .map(ResidualDiagnosticScores::maximum)
}

/// Fisher-like score fusion from component-calibrated tail p-values.
///
/// Returns `sum(-ln(p_i))` over the predeclared component family. This is a score, *not* a
/// Fisher-reference p-value: the component p-values may be dependent and discrete. For finite-sample
/// marginal validity, calibrate these fused scores on an untouched second set of clean batches
/// using `ConformalAdequacyCalibrator`, then use that outer calibrator for the final audit.
/// The two-stage protocol avoids independence assumptions between residual diagnostics at the cost
/// of a separate normalization and aggregate-calibration split.
///
/// Larger values mean more combined tail evidence. Freeze the component set/order and split roles
/// before viewing audit data; never use the audit batch to tune components or fusion weights.
pub fn sum_log_component_p_fusion_score(
    component_calibrators: &[ConformalAdequacyCalibrator],
    component_scores: &[f64],
    independent_audit: bool,
) -> Result<f64, CalibrationError> {
    if !independent_audit {
        return Err(CalibrationError::AuditNotIndependent);
    }
    if component_calibrators.is_empty() || component_calibrators.len() != component_scores.len() {
        return Err(CalibrationError::InvalidAuditScore);
    }

    let mut aggregate = 0.0_f64;
    for (calibrator, score) in component_calibrators
        .iter()
        .zip(component_scores.iter().copied())
    {
        let component_p = calibrator.audit_p_value(score, true)?;
        aggregate -= component_p.ln();
    }
    if !aggregate.is_finite() || aggregate < 0.0 {
        return Err(CalibrationError::InvalidAuditScore);
    }
    Ok(aggregate)
}

/// Combine already-calibrated component scores into one score for a *second* conformal calibration.
///
/// The component calibrators must be fitted on an independent normalization split. This function
/// computes `-ln(min component p-value)`; larger values mean at least one component is unusual.
/// To retain finite-sample marginal validity without assuming independence between components,
/// compute this aggregate for a separate, untouched set of clean aggregate-calibration batches and
/// fit a new `ConformalAdequacyCalibrator` to those aggregate scores. Use that second calibrator
/// for the final audit. The two split roles must not overlap, and the component family/order must
/// be frozen before looking at audit scores.
///
/// This two-stage score-fusion primitive is intentionally explicit about the extra calibration
/// cost. It is not itself a final calibrated p-value and does not implement anytime-valid monitoring.
pub fn minimum_component_p_fusion_score(
    component_calibrators: &[ConformalAdequacyCalibrator],
    component_scores: &[f64],
    independent_audit: bool,
) -> Result<f64, CalibrationError> {
    if !independent_audit {
        return Err(CalibrationError::AuditNotIndependent);
    }
    if component_calibrators.is_empty() || component_calibrators.len() != component_scores.len() {
        return Err(CalibrationError::InvalidAuditScore);
    }

    let mut minimum_p = 1.0_f64;
    for (calibrator, score) in component_calibrators
        .iter()
        .zip(component_scores.iter().copied())
    {
        minimum_p = minimum_p.min(calibrator.audit_p_value(score, true)?);
    }
    let aggregate = -minimum_p.ln();
    if !aggregate.is_finite() || aggregate < 0.0 {
        return Err(CalibrationError::InvalidAuditScore);
    }
    Ok(aggregate)
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
    fn multi_view_score_increases_raw_statistic_for_localized_outlier() {
        let mut observed = vec![0.0; 100];
        observed[50] = 1.0;
        let predicted = vec![0.0; 100];
        let scales = vec![1.0; 100];

        let rms = standardized_rms_residual(&observed, &predicted, &scales).unwrap();
        let multi_view =
            multi_view_standardized_residual_score(&observed, &predicted, &scales).unwrap();

        assert!((rms - 0.1).abs() < 1e-12);
        assert!(multi_view > rms * 3.0);
    }

    #[test]
    fn multi_view_score_increases_raw_statistic_for_systematic_bias() {
        let observed = vec![0.25; 64];
        let predicted = vec![0.0; 64];
        let scales = vec![1.0; 64];

        let rms = standardized_rms_residual(&observed, &predicted, &scales).unwrap();
        let multi_view =
            multi_view_standardized_residual_score(&observed, &predicted, &scales).unwrap();

        assert!((rms - 0.25).abs() < 1e-12);
        assert!(multi_view >= 2.0 - 1e-12);
    }

    #[test]
    fn multi_view_score_increases_raw_statistic_for_alternating_residual_structure() {
        let observed = (0..32)
            .map(|index| if index % 2 == 0 { 1.0 } else { -1.0 })
            .collect::<Vec<_>>();
        let predicted = vec![0.0; 32];
        let scales = vec![1.0; 32];

        let rms = standardized_rms_residual(&observed, &predicted, &scales).unwrap();
        let multi_view =
            multi_view_standardized_residual_score(&observed, &predicted, &scales).unwrap();

        assert!((rms - 1.0).abs() < 1e-12);
        assert!(multi_view > 5.0);
    }

    #[test]
    fn multi_view_score_preserves_zero_and_fails_closed() {
        assert_eq!(
            multi_view_standardized_residual_score(&[1.0, -2.0], &[1.0, -2.0], &[0.1, 100.0]),
            Some(0.0)
        );
        assert_eq!(multi_view_standardized_residual_score(&[], &[], &[]), None);
        assert_eq!(
            multi_view_standardized_residual_score(&[1.0], &[0.0], &[0.0]),
            None
        );
        assert_eq!(
            multi_view_standardized_residual_score(&[f64::NAN], &[0.0], &[1.0]),
            None
        );
        assert_eq!(
            multi_view_standardized_residual_score(&[f64::MAX], &[-f64::MAX], &[1.0]),
            None
        );
    }

    #[test]
    fn conformal_calibration_applies_to_the_frozen_aggregate_score() {
        let calibration_scores = (1..=19)
            .map(|value| {
                multi_view_standardized_residual_score(
                    &[value as f64, 0.0, 0.0],
                    &[0.0, 0.0, 0.0],
                    &[1.0, 1.0, 1.0],
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let calibrator = ConformalAdequacyCalibrator::new(&calibration_scores, 0.05).unwrap();
        let audit_score =
            multi_view_standardized_residual_score(&[100.0, 0.0, 0.0], &[0.0; 3], &[1.0; 3])
                .unwrap();
        let result = calibrator.assess_audit_score(audit_score, true).unwrap();
        assert!((result.p_value - 0.05).abs() < 1e-12);
        assert!(matches!(
            result.assessment,
            AdequacyAssessment::Alarm { .. }
        ));
    }

    fn component_calibrators() -> Vec<ConformalAdequacyCalibrator> {
        let scores = (1..=19).map(|value| value as f64).collect::<Vec<_>>();
        let calibrator = ConformalAdequacyCalibrator::new(&scores, 0.5).unwrap();
        vec![
            calibrator.clone(),
            calibrator.clone(),
            calibrator.clone(),
            calibrator,
        ]
    }

    #[test]
    fn component_p_fusion_is_monotone_and_checks_contract() {
        let components = component_calibrators();
        let usual = minimum_component_p_fusion_score(&components, &[10.0; 4], true).unwrap();
        let unusual =
            minimum_component_p_fusion_score(&components, &[100.0, 10.0, 10.0, 10.0], true)
                .unwrap();
        assert!(unusual > usual);
        assert_eq!(
            minimum_component_p_fusion_score(&components, &[1.0], true),
            Err(CalibrationError::InvalidAuditScore)
        );
        assert_eq!(
            minimum_component_p_fusion_score(&components, &[1.0; 4], false),
            Err(CalibrationError::AuditNotIndependent)
        );
    }

    #[test]
    fn sum_log_fusion_accumulates_multiple_component_tail_signals() {
        let components = component_calibrators();
        let usual = sum_log_component_p_fusion_score(&components, &[10.0; 4], true).unwrap();
        let one_unusual =
            sum_log_component_p_fusion_score(&components, &[100.0, 10.0, 10.0, 10.0], true)
                .unwrap();
        let all_unusual = sum_log_component_p_fusion_score(&components, &[100.0; 4], true).unwrap();
        assert!(one_unusual > usual);
        assert!(all_unusual > one_unusual);
        assert_eq!(
            sum_log_component_p_fusion_score(&components, &[1.0], true),
            Err(CalibrationError::InvalidAuditScore)
        );
    }

    #[test]
    fn sum_log_fusion_requires_its_own_outer_calibration_split() {
        let components = component_calibrators();
        let aggregate_reference = (1..=19)
            .map(|value| {
                sum_log_component_p_fusion_score(&components, &[value as f64, 0.0, 0.0, 0.0], true)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let final_calibrator =
            ConformalAdequacyCalibrator::new(&aggregate_reference, 0.05).unwrap();
        let audit_aggregate =
            sum_log_component_p_fusion_score(&components, &[100.0, 0.0, 0.0, 0.0], true).unwrap();
        let result = final_calibrator
            .assess_audit_score(audit_aggregate, true)
            .unwrap();
        assert!((result.p_value - 0.05).abs() < 1e-12);
        assert!(matches!(
            result.assessment,
            AdequacyAssessment::Alarm { .. }
        ));
    }

    #[test]
    fn component_p_fusion_uses_a_separate_outer_conformal_calibrator() {
        let components = component_calibrators();
        let aggregate_reference = (1..=19)
            .map(|value| {
                minimum_component_p_fusion_score(&components, &[value as f64; 4], true).unwrap()
            })
            .collect::<Vec<_>>();
        let final_calibrator =
            ConformalAdequacyCalibrator::new(&aggregate_reference, 0.05).unwrap();
        let audit_aggregate =
            minimum_component_p_fusion_score(&components, &[100.0, 10.0, 10.0, 10.0], true)
                .unwrap();
        let result = final_calibrator
            .assess_audit_score(audit_aggregate, true)
            .unwrap();
        assert!((result.p_value - 0.05).abs() < 1e-12);
        assert!(matches!(
            result.assessment,
            AdequacyAssessment::Alarm { .. }
        ));
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
    fn calibrated_assessment_preserves_the_declared_diagnostic_channel() {
        let calibrator = ConformalAdequacyCalibrator::new(
            &(1..=99).map(|value| value as f64).collect::<Vec<_>>(),
            0.05,
        )
        .unwrap();
        let feature_result = calibrator
            .assess_audit_score_with_diagnostic(
                100.0,
                true,
                AdequacyDiagnostic::QueryDistributionShift,
            )
            .unwrap();
        assert_eq!(
            feature_result.assessment,
            AdequacyAssessment::Alarm {
                diagnostic: AdequacyDiagnostic::QueryDistributionShift,
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
