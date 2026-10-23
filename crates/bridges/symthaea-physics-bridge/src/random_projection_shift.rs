// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Fixed-reference random-projection feature-shift scoring for wide input spaces.
//!
//! The exact moment/correlation scorer in feature_shift computes all feature-pair
//! correlations and therefore costs O(n d²) for a batch with n rows and d features.
//! This scorer freezes k < d Rademacher projections at fit time, then measures mean,
//! scale, and pairwise-correlation changes in that k-dimensional sketch. Scoring costs
//! O(n d k + n k²), with no dense d-by-k matrix retained.
//!
//! Projection compresses information and can miss changes orthogonal to the chosen sketch.
//! It is an explicitly uncalibrated nonconformity score, not a p-value or a guarantee.
//! For marginal false-alarm control, calibrate scores from independent clean batches
//! with the exact same fitted scorer and batch protocol, then assess independent audits.
//! The projection count, seed, feature order, scale floor, reference rows, and score
//! definition must be fixed before calibration and unchanged for the entire epoch.

use crate::adequacy_calibration::{
    CalibrationError, ConformalAdequacyCalibrator, ConformalAdequacyResult,
};
use crate::feature_shift::FeatureShiftError;
use crate::feature_shift::FeatureShiftScore;
use crate::model_adequacy::AdequacyDiagnostic;

/// Numerical floor for scales in the standardized projection space.
const PROJECTED_SCALE_FLOOR: f64 = 1e-9;

/// Fixed-reference feature-shift scorer using a deterministic random projection sketch.
#[derive(Clone, Debug, PartialEq)]
pub struct RandomProjectionFeatureShiftScorer {
    feature_means: Vec<f64>,
    feature_scales: Vec<f64>,
    reference_projection_means: Vec<f64>,
    reference_projection_scales: Vec<f64>,
    reference_projection_correlations: Vec<f64>,
    projection_count: usize,
    seed: u64,
    scale_floor: f64,
}

impl RandomProjectionFeatureShiftScorer {
    /// Fit a frozen sketch from at least two finite, equal-width reference rows.
    ///
    /// projection_count must be positive and strictly less than the input feature count;
    /// otherwise this is not a dimensionality reduction and the exact scorer is preferable.
    /// scale_floor is measured in the original input feature units. seed deterministically
    /// defines the Rademacher projection matrix; do not select it using audit outcomes.
    pub fn fit(
        reference_rows: &[Vec<f64>],
        projection_count: usize,
        scale_floor: f64,
        seed: u64,
    ) -> Option<Self> {
        if !scale_floor.is_finite() || scale_floor <= 0.0 || reference_rows.len() < 2 {
            return None;
        }
        let feature_count = reference_rows.first()?.len();
        if feature_count < 2
            || projection_count == 0
            || projection_count >= feature_count
            || reference_rows
                .iter()
                .any(|row| row.len() != feature_count || row.iter().any(|x| !x.is_finite()))
        {
            return None;
        }

        let feature_means = column_means(reference_rows, feature_count)?;
        let feature_scales = column_scales(reference_rows, &feature_means, feature_count)?;
        let projected_reference = project_rows(
            reference_rows,
            &feature_means,
            &feature_scales,
            scale_floor,
            projection_count,
            seed,
        )?;
        let reference_projection_means = column_means(&projected_reference, projection_count)?;
        let reference_projection_scales = column_scales(
            &projected_reference,
            &reference_projection_means,
            projection_count,
        )?;
        let reference_projection_correlations = pairwise_correlations(
            &projected_reference,
            &reference_projection_means,
            &reference_projection_scales,
            projection_count,
        )?;

        Some(Self {
            feature_means,
            feature_scales,
            reference_projection_means,
            reference_projection_scales,
            reference_projection_correlations,
            projection_count,
            seed,
            scale_floor,
        })
    }

    /// Score an independent audit batch against the exact frozen reference and sketch.
    ///
    /// The result is a nonconformity statistic, not a p-value. Every non-finite intermediate,
    /// empty/malformed batch, or feature-layout mismatch fails closed with None.
    pub fn score(&self, audit_rows: &[Vec<f64>]) -> Option<FeatureShiftScore> {
        if audit_rows.len() < 2
            || audit_rows.iter().any(|row| {
                row.len() != self.feature_means.len()
                    || row.iter().any(|value| !value.is_finite())
            })
        {
            return None;
        }

        let projected_audit = project_rows(
            audit_rows,
            &self.feature_means,
            &self.feature_scales,
            self.scale_floor,
            self.projection_count,
            self.seed,
        )?;
        let audit_means = column_means(&projected_audit, self.projection_count)?;
        let audit_scales = column_scales(&projected_audit, &audit_means, self.projection_count)?;
        let audit_correlations = pairwise_correlations(
            &projected_audit,
            &audit_means,
            &audit_scales,
            self.projection_count,
        )?;
        let n = audit_rows.len() as f64;

        let mut normalized_mean_shift = 0.0_f64;
        let mut normalized_scale_shift = 0.0_f64;
        for index in 0..self.projection_count {
            let reference_scale =
                self.reference_projection_scales[index].max(PROJECTED_SCALE_FLOOR);
            let mean_shift = (audit_means[index] - self.reference_projection_means[index]).abs()
                / reference_scale
                * n.sqrt();
            let scale_shift = (audit_scales[index].max(PROJECTED_SCALE_FLOOR).ln()
                - reference_scale.ln())
            .abs()
                * (n / 2.0).sqrt();
            if !mean_shift.is_finite() || !scale_shift.is_finite() {
                return None;
            }
            normalized_mean_shift = normalized_mean_shift.max(mean_shift);
            normalized_scale_shift = normalized_scale_shift.max(scale_shift);
        }

        let mut normalized_correlation_shift = 0.0_f64;
        for (reference, audit) in self
            .reference_projection_correlations
            .iter()
            .zip(audit_correlations.iter())
        {
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

    /// Score and conformally assess one audit batch using independent clean calibration scores.
    ///
    /// The calibrator must have been fit on batches scored by this exact scorer with the same
    /// batch size/protocol. The alarm remains typed as QueryDistributionShift, not predictive
    /// model misfit. This wrapper cannot prove that the caller honored the data split.
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

    /// Number of original input features.
    pub fn input_feature_count(&self) -> usize {
        self.feature_means.len()
    }

    /// Number of retained random-projection dimensions.
    pub fn projection_count(&self) -> usize {
        self.projection_count
    }

    /// Number of pairwise correlations compared in the compressed sketch.
    pub fn projected_correlation_count(&self) -> usize {
        self.reference_projection_correlations.len()
    }
}

fn project_rows(
    rows: &[Vec<f64>],
    feature_means: &[f64],
    feature_scales: &[f64],
    scale_floor: f64,
    projection_count: usize,
    seed: u64,
) -> Option<Vec<Vec<f64>>> {
    if rows.is_empty()
        || feature_means.is_empty()
        || feature_means.len() != feature_scales.len()
        || projection_count == 0
    {
        return None;
    }
    let feature_count = feature_means.len();
    let projection_scale = (projection_count as f64).sqrt().recip();
    let mut projected_rows = Vec::with_capacity(rows.len());

    for row in rows {
        if row.len() != feature_count || row.iter().any(|value| !value.is_finite()) {
            return None;
        }
        let mut projected = vec![0.0_f64; projection_count];
        for (feature, value) in row.iter().enumerate() {
            let standardized =
                (*value - feature_means[feature]) / feature_scales[feature].max(scale_floor);
            if !standardized.is_finite() {
                return None;
            }
            for (component, sum) in projected.iter_mut().enumerate() {
                *sum += standardized * rademacher_sign(seed, component, feature);
                if !sum.is_finite() {
                    return None;
                }
            }
        }
        for value in &mut projected {
            *value *= projection_scale;
            if !value.is_finite() {
                return None;
            }
        }
        projected_rows.push(projected);
    }
    Some(projected_rows)
}

fn rademacher_sign(seed: u64, component: usize, feature: usize) -> f64 {
    let key = seed
        ^ (component as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (feature as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
    if mix64(key) & 1 == 0 { 1.0 } else { -1.0 }
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn column_means(rows: &[Vec<f64>], feature_count: usize) -> Option<Vec<f64>> {
    let mut sums = vec![0.0_f64; feature_count];
    for row in rows {
        if row.len() != feature_count {
            return None;
        }
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
    if rows.is_empty() || means.len() != feature_count {
        return None;
    }
    let mut sums_of_squares = vec![0.0_f64; feature_count];
    for row in rows {
        if row.len() != feature_count {
            return None;
        }
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
    scales.iter().all(|value| value.is_finite()).then_some(scales)
}

fn pairwise_correlations(
    rows: &[Vec<f64>],
    means: &[f64],
    scales: &[f64],
    feature_count: usize,
) -> Option<Vec<f64>> {
    let correlation_count = feature_count
        .checked_mul(feature_count.saturating_sub(1))?
        / 2;
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
    use crate::model_adequacy::AdequacyAssessment;

    fn orthogonal_reference() -> Vec<Vec<f64>> {
        (0..128)
            .map(|row| {
                (1..=127)
                    .map(|feature| {
                        if ((row & feature).count_ones() & 1) == 0 {
                            1.0
                        } else {
                            -1.0
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[test]
    fn random_projection_feature_shift_compresses_wide_correlation_summary() {
        let reference = orthogonal_reference();
        let scorer =
            RandomProjectionFeatureShiftScorer::fit(&reference, 16, 1e-8, 0xA4D2_9187).unwrap();

        assert_eq!(scorer.input_feature_count(), 127);
        assert_eq!(scorer.projection_count(), 16);
        assert_eq!(scorer.projected_correlation_count(), 120);
        assert_eq!(127 * 126 / 2, 8_001);

        let reference_score = scorer.score(&reference).unwrap();
        assert!(reference_score.aggregate < 1e-8);

        let diffuse_shift = reference
            .iter()
            .map(|row| row.iter().map(|value| value + 0.5).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let shifted_score = scorer.score(&diffuse_shift).unwrap();
        assert!(
            shifted_score.normalized_mean_shift > 3.0,
            "diffuse mean-shift score was {}",
            shifted_score.normalized_mean_shift
        );
    }

    #[test]
    fn random_projection_feature_shift_rejects_bad_contracts() {
        let reference = orthogonal_reference();
        assert!(RandomProjectionFeatureShiftScorer::fit(&[], 8, 1e-8, 7).is_none());
        assert!(RandomProjectionFeatureShiftScorer::fit(&reference, 0, 1e-8, 7).is_none());
        assert!(RandomProjectionFeatureShiftScorer::fit(&reference, 127, 1e-8, 7).is_none());
        assert!(RandomProjectionFeatureShiftScorer::fit(&reference, 8, 0.0, 7).is_none());

        let scorer = RandomProjectionFeatureShiftScorer::fit(&reference, 8, 1e-8, 7).unwrap();
        assert!(scorer.score(&[]).is_none());
        assert!(scorer.score(&[vec![f64::NAN; 127]]).is_none());
        assert!(scorer.score(&[vec![0.0; 126], vec![0.0; 126]]).is_none());
    }

    #[test]
    fn random_projection_feature_shift_uses_distinct_calibrated_diagnostic() {
        let reference = orthogonal_reference();
        let scorer =
            RandomProjectionFeatureShiftScorer::fit(&reference, 16, 1e-8, 0xA4D2_9187).unwrap();
        let calibrator = ConformalAdequacyCalibrator::new(&[0.0; 99], 0.05).unwrap();
        let shifted = reference
            .iter()
            .map(|row| row.iter().map(|value| value + 0.5).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let result = scorer.assess(&calibrator, &shifted, true).unwrap();
        assert!(result.p_value <= 0.05);
        assert!(matches!(
            result.assessment,
            AdequacyAssessment::Alarm {
                diagnostic: AdequacyDiagnostic::QueryDistributionShift,
                calibrated: true,
                independent_audit: true,
            }
        ));
    }
}
