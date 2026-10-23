// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Fixed-reference random-projection feature-shift scoring for wide input spaces.
//!
//! The exact moment/correlation scorer in feature_shift computes all feature-pair
//! correlations and therefore costs O(n d²) for a batch with n rows and d features.
//! This scorer preserves exact per-feature mean/scale checks in O(n d) time and freezes
//! k < d Rademacher projections to approximate pairwise-correlation changes. Scoring costs
//! O(n d k + n k²). Projection signs are precomputed as compact one-byte coefficients, avoiding
//! repeated hash mixing for every audit observation and avoiding an f64 projection matrix.
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
    projection_signs: Vec<i8>,
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
        let projection_signs = make_projection_signs(feature_count, projection_count, seed)?;
        let projected_reference = project_rows(
            reference_rows,
            &feature_means,
            &feature_scales,
            scale_floor,
            projection_count,
            &projection_signs,
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
            projection_signs,
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
                row.len() != self.feature_means.len() || row.iter().any(|value| !value.is_finite())
            })
        {
            return None;
        }

        // Preserve sparse marginal-shift sensitivity at O(n d) cost rather than relying
        // only on sketches, where a one-coordinate mean/scale change can be diluted.
        let audit_feature_means = column_means(audit_rows, self.feature_means.len())?;
        let audit_feature_scales =
            column_scales(audit_rows, &audit_feature_means, self.feature_means.len())?;
        let n = audit_rows.len() as f64;
        let mut normalized_mean_shift = 0.0_f64;
        let mut normalized_scale_shift = 0.0_f64;
        for index in 0..self.feature_means.len() {
            let reference_scale = self.feature_scales[index].max(self.scale_floor);
            let mean_shift = (audit_feature_means[index] - self.feature_means[index]).abs()
                / reference_scale
                * n.sqrt();
            let scale_shift = (audit_feature_scales[index].max(self.scale_floor).ln()
                - reference_scale.ln())
            .abs()
                * (n / 2.0).sqrt();
            if !mean_shift.is_finite() || !scale_shift.is_finite() {
                return None;
            }
            normalized_mean_shift = normalized_mean_shift.max(mean_shift);
            normalized_scale_shift = normalized_scale_shift.max(scale_shift);
        }

        let projected_audit = project_rows(
            audit_rows,
            &self.feature_means,
            &self.feature_scales,
            self.scale_floor,
            self.projection_count,
            &self.projection_signs,
        )?;
        let audit_means = column_means(&projected_audit, self.projection_count)?;
        let audit_scales = column_scales(&projected_audit, &audit_means, self.projection_count)?;
        let audit_correlations = pairwise_correlations(
            &projected_audit,
            &audit_means,
            &audit_scales,
            self.projection_count,
        )?;

        // Projected mean/scale views can add power for diffuse alternatives. Max with the
        // exact marginal views above so sketches cannot erase large one-feature shifts.
        for index in 0..self.projection_count {
            let reference_scale =
                self.reference_projection_scales[index].max(PROJECTED_SCALE_FLOOR);
            let mean_shift = (audit_means[index] - self.reference_projection_means[index]).abs()
                / reference_scale
                * n.sqrt();
            let scale_shift =
                (audit_scales[index].max(PROJECTED_SCALE_FLOOR).ln() - reference_scale.ln()).abs()
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
            .map_err(map_calibration_error)
    }

    /// Number of original input features.
    pub fn input_feature_count(&self) -> usize {
        self.feature_means.len()
    }

    /// Number of retained random-projection dimensions.
    pub fn projection_count(&self) -> usize {
        self.projection_count
    }

    /// Fixed projection seed, useful for an auditable calibration receipt.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Number of pairwise correlations compared in the compressed sketch.
    pub fn projected_correlation_count(&self) -> usize {
        self.reference_projection_correlations.len()
    }
}

fn map_calibration_error(error: CalibrationError) -> FeatureShiftError {
    FeatureShiftError::Calibration(error)
}

fn project_rows(
    rows: &[Vec<f64>],
    feature_means: &[f64],
    feature_scales: &[f64],
    scale_floor: f64,
    projection_count: usize,
    projection_signs: &[i8],
) -> Option<Vec<Vec<f64>>> {
    if rows.is_empty()
        || feature_means.is_empty()
        || feature_means.len() != feature_scales.len()
        || projection_count == 0
        || feature_means.len().checked_mul(projection_count)? != projection_signs.len()
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
            let sign_offset = feature.checked_mul(projection_count)?;
            for (component, sum) in projected.iter_mut().enumerate() {
                *sum += standardized * projection_signs[sign_offset + component] as f64;
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

fn make_projection_signs(
    feature_count: usize,
    projection_count: usize,
    seed: u64,
) -> Option<Vec<i8>> {
    let coefficient_count = feature_count.checked_mul(projection_count)?;
    let mut signs = Vec::with_capacity(coefficient_count);
    for feature in 0..feature_count {
        for component in 0..projection_count {
            signs.push(if rademacher_sign(seed, component, feature) > 0.0 {
                1
            } else {
                -1
            });
        }
    }
    Some(signs)
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
    let correlation_count = feature_count.checked_mul(feature_count.saturating_sub(1))? / 2;
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
    use crate::feature_shift::FeatureShiftScorer;
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
    fn random_projection_feature_shift_preserves_sparse_marginal_shift_sensitivity() {
        let reference = orthogonal_reference();
        let projected =
            RandomProjectionFeatureShiftScorer::fit(&reference, 16, 1e-8, 0xA4D2_9187).unwrap();
        let exact = FeatureShiftScorer::fit(&reference, 1e-8).unwrap();
        let sparse_shift = reference
            .iter()
            .map(|row| {
                let mut shifted = row.clone();
                shifted[0] += 0.5;
                shifted
            })
            .collect::<Vec<_>>();

        let exact_score = exact.score(&sparse_shift).unwrap();
        let projected_score = projected.score(&sparse_shift).unwrap();

        // The low-cost exact marginal pass protects sparse first/second-moment alternatives.
        assert!(exact_score.normalized_mean_shift > 5.0);
        assert!(projected_score.normalized_mean_shift > 5.0);
        assert!(
            (projected_score.normalized_mean_shift - exact_score.normalized_mean_shift).abs()
                < 1e-8
        );

        let sparse_scale_shift = reference
            .iter()
            .map(|row| {
                let mut shifted = row.clone();
                shifted[0] *= 3.0;
                shifted
            })
            .collect::<Vec<_>>();
        let exact_scale_score = exact.score(&sparse_scale_shift).unwrap();
        let projected_scale_score = projected.score(&sparse_scale_shift).unwrap();
        assert!(exact_scale_score.normalized_scale_shift > 8.0);
        assert!(projected_scale_score.normalized_scale_shift > 8.0);
        assert!(
            (projected_scale_score.normalized_scale_shift
                - exact_scale_score.normalized_scale_shift)
                .abs()
                < 1e-8
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

    fn benchmark_rows(row_count: usize, feature_count: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut state = seed;
        (0..row_count)
            .map(|_| {
                (0..feature_count)
                    .map(|_| {
                        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                        if mix64(state) & 1 == 0 { 1.0 } else { -1.0 }
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn timing_percentile(samples: &mut [u128], percentile: usize) -> u128 {
        samples.sort_unstable();
        let index = ((samples.len() - 1) * percentile).div_ceil(100);
        samples[index.min(samples.len() - 1)]
    }

    /// Manual timing harness. Run with:
    /// cargo test -p symthaea-physics-bridge random_projection_feature_shift_reports_timing --lib -- --ignored --nocapture
    #[test]
    #[ignore = "manual microbenchmark; timings are environment-dependent"]
    fn random_projection_feature_shift_reports_timing() {
        use std::hint::black_box;
        use std::time::Instant;

        const ROWS: usize = 64;
        const FEATURES: usize = 512;
        const PROJECTIONS: usize = 32;
        const REPEATS: usize = 9;
        let reference = benchmark_rows(ROWS, FEATURES, 0x0123_4567_89AB_CDEF);
        let audit = benchmark_rows(ROWS, FEATURES, 0xFEDC_BA98_7654_3210)
            .into_iter()
            .map(|row| row.into_iter().map(|value| value + 0.1).collect())
            .collect::<Vec<Vec<f64>>>();

        let exact_fit_start = Instant::now();
        let exact = FeatureShiftScorer::fit(&reference, 1e-8).unwrap();
        let exact_fit_us = exact_fit_start.elapsed().as_micros();
        let projected_fit_start = Instant::now();
        let projected =
            RandomProjectionFeatureShiftScorer::fit(&reference, PROJECTIONS, 1e-8, 0xA4D2_9187)
                .unwrap();
        let projected_fit_us = projected_fit_start.elapsed().as_micros();

        black_box(exact.score(&audit).unwrap());
        black_box(projected.score(&audit).unwrap());
        let mut exact_times = Vec::with_capacity(REPEATS);
        let mut projected_times = Vec::with_capacity(REPEATS);
        for _ in 0..REPEATS {
            let start = Instant::now();
            black_box(exact.score(black_box(&audit)).unwrap());
            exact_times.push(start.elapsed().as_micros());

            let start = Instant::now();
            black_box(projected.score(black_box(&audit)).unwrap());
            projected_times.push(start.elapsed().as_micros());
        }

        let exact_p50 = timing_percentile(&mut exact_times, 50);
        let exact_p95 = timing_percentile(&mut exact_times, 95);
        let projected_p50 = timing_percentile(&mut projected_times, 50);
        let projected_p95 = timing_percentile(&mut projected_times, 95);
        let exact_pairs = exact.correlation_count();
        let projected_pairs = projected.projected_correlation_count();
        let exact_reference_bytes =
            FEATURES * 2 * std::mem::size_of::<f64>() + exact_pairs * std::mem::size_of::<f64>();
        let projected_reference_bytes = FEATURES * 2 * std::mem::size_of::<f64>()
            + FEATURES * PROJECTIONS * std::mem::size_of::<i8>()
            + PROJECTIONS * 2 * std::mem::size_of::<f64>()
            + projected_pairs * std::mem::size_of::<f64>();

        println!(
            "feature-shift timing: rows={ROWS}, features={FEATURES}, projections={PROJECTIONS}, repeats={REPEATS}"
        );
        println!(
            "exact: fit_us={exact_fit_us}, p50_score_us={exact_p50}, p95_score_us={exact_p95}, correlations={exact_pairs}, approx_reference_bytes={exact_reference_bytes}"
        );
        println!(
            "projected: fit_us={projected_fit_us}, p50_score_us={projected_p50}, p95_score_us={projected_p95}, correlations={projected_pairs}, approx_reference_bytes={projected_reference_bytes}"
        );
        println!(
            "observed_median_scoring_speedup={:.3}x",
            exact_p50 as f64 / projected_p50.max(1) as f64
        );
    }
}
