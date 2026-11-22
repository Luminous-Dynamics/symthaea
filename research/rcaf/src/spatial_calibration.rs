//! Variance-calibrated scaling for spatial nuisance fields.
//!
//! An additive loading is not comparable across graph ranks, diffusion scales, or seed sets.
//! This module calibrates a generated field against the clean observations to a declared
//! component-variance fraction. It reports signal/artifact covariance separately because
//! aligned nuisance can change the variance of the sum even at a fixed component fraction.

use crate::spatial_nuisance::SpatialNuisanceField;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct NuisanceVarianceCalibrationReport {
    pub schema_version: &'static str,
    /// Requested Var(A) / (Var(X) + Var(A)); excludes covariance by definition.
    pub requested_component_variance_fraction: f64,
    pub realized_component_variance_fraction: f64,
    pub clean_component_variance: f64,
    pub nuisance_component_variance: f64,
    pub clean_nuisance_covariance: f64,
    pub covariance_contribution_fraction: f64,
    pub combined_observed_variance: f64,
    pub applied_scale_multiplier: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpatialVarianceCalibrationError {
    InvalidTargetFraction,
    DimensionMismatch,
    NonFiniteInput,
    ZeroCleanVariance,
    ZeroNuisanceVariance,
    NonFiniteScale,
}

impl fmt::Display for SpatialVarianceCalibrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTargetFraction => {
                write!(
                    f,
                    "target nuisance component-variance fraction must be finite in [0, 1)"
                )
            }
            Self::DimensionMismatch => {
                write!(
                    f,
                    "clean observations must match spatial nuisance field dimensions"
                )
            }
            Self::NonFiniteInput => {
                write!(f, "clean observations and nuisance values must be finite")
            }
            Self::ZeroCleanVariance => {
                write!(
                    f,
                    "clean observations must have positive pooled within-ROI variance"
                )
            }
            Self::ZeroNuisanceVariance => {
                write!(
                    f,
                    "a positive target fraction requires a non-zero nuisance field"
                )
            }
            Self::NonFiniteScale => {
                write!(f, "variance calibration produced a non-finite scale")
            }
        }
    }
}

impl std::error::Error for SpatialVarianceCalibrationError {}

/// Backwards-compatible public names for existing spatial callers.
pub type SpatialVarianceCalibrationReport = NuisanceVarianceCalibrationReport;
pub type NuisanceVarianceCalibrationError = SpatialVarianceCalibrationError;

/// Calibrate any time x ROI nuisance matrix to a requested component-variance fraction.
///
/// Clean and nuisance time series are centered separately within each ROI. Population
/// variances and covariance are pooled equally across ROIs. The target is Var(A)/(Var(X)+Var(A));
/// Cov(X,A) is reported separately and is included when computing Var(X+A).
///
/// The input nuisance matrix is never modified. All nuisance families can use this function,
/// including common-mode, spatially structured, transient, physiological, and future empirical
/// fields. Calibration changes magnitude only; it does not make a synthetic distribution
/// empirically representative.
pub fn calibrate_nuisance_matrix_to_component_fraction(
    clean_observations: &[Vec<f64>],
    nuisance_matrix: &[Vec<f64>],
    target_fraction: f64,
) -> Result<
    (Vec<Vec<f64>>, NuisanceVarianceCalibrationReport),
    NuisanceVarianceCalibrationError,
> {
    if !target_fraction.is_finite() || !(0.0..1.0).contains(&target_fraction) {
        return Err(SpatialVarianceCalibrationError::InvalidTargetFraction);
    }

    let time_points = clean_observations.len();
    if time_points < 3 || clean_observations.first().map_or(0, Vec::len) < 2 {
        return Err(SpatialVarianceCalibrationError::DimensionMismatch);
    }
    let roi_count = clean_observations[0].len();
    if clean_observations.iter().any(|row| row.len() != roi_count)
        || nuisance_matrix.len() != time_points
        || nuisance_matrix.iter().any(|row| row.len() != roi_count)
    {
        return Err(SpatialVarianceCalibrationError::DimensionMismatch);
    }
    if clean_observations
        .iter()
        .flatten()
        .chain(nuisance_matrix.iter().flatten())
        .any(|value| !value.is_finite())
    {
        return Err(SpatialVarianceCalibrationError::NonFiniteInput);
    }

    let (clean_variance, nuisance_variance, _) =
        pooled_within_roi_moments(clean_observations, nuisance_matrix);
    if !clean_variance.is_finite() || !nuisance_variance.is_finite() {
        return Err(SpatialVarianceCalibrationError::NonFiniteInput);
    }
    if clean_variance <= 1e-30 {
        return Err(SpatialVarianceCalibrationError::ZeroCleanVariance);
    }
    if target_fraction > 0.0 && nuisance_variance <= 1e-30 {
        return Err(SpatialVarianceCalibrationError::ZeroNuisanceVariance);
    }

    let scale_multiplier = if target_fraction == 0.0 {
        0.0
    } else {
        ((target_fraction / (1.0 - target_fraction)) * (clean_variance / nuisance_variance))
            .sqrt()
    };
    if !scale_multiplier.is_finite() {
        return Err(SpatialVarianceCalibrationError::NonFiniteScale);
    }

    let mut calibrated = nuisance_matrix.to_vec();
    for value in calibrated.iter_mut().flatten() {
        *value *= scale_multiplier;
    }
    if calibrated.iter().flatten().any(|value| !value.is_finite()) {
        return Err(SpatialVarianceCalibrationError::NonFiniteScale);
    }

    let (calibrated_clean_variance, calibrated_nuisance_variance, calibrated_covariance) =
        pooled_within_roi_moments(clean_observations, &calibrated);
    let denominator = calibrated_clean_variance + calibrated_nuisance_variance;
    let realized_fraction = if denominator > 0.0 {
        calibrated_nuisance_variance / denominator
    } else {
        0.0
    };
    let covariance_contribution_fraction = if denominator > 0.0 {
        2.0 * calibrated_covariance / denominator
    } else {
        0.0
    };
    let combined_variance =
        calibrated_clean_variance + calibrated_nuisance_variance + 2.0 * calibrated_covariance;

    if [
        realized_fraction,
        covariance_contribution_fraction,
        combined_variance,
    ]
    .iter()
    .any(|value| !value.is_finite())
    {
        return Err(SpatialVarianceCalibrationError::NonFiniteScale);
    }

    Ok((
        calibrated,
        NuisanceVarianceCalibrationReport {
            schema_version: "rcaf.nuisance-variance-calibration/0.1.0",
            requested_component_variance_fraction: target_fraction,
            realized_component_variance_fraction: realized_fraction,
            clean_component_variance: calibrated_clean_variance,
            nuisance_component_variance: calibrated_nuisance_variance,
            clean_nuisance_covariance: calibrated_covariance,
            covariance_contribution_fraction,
            combined_observed_variance: combined_variance.max(0.0),
            applied_scale_multiplier: scale_multiplier,
        },
    ))
}

/// Calibrate a generated graph-spatial nuisance field while preserving its factor metadata.
///
/// This compatibility wrapper delegates magnitude calibration to the family-agnostic matrix
/// calibrator, then applies the same scale multiplier to the stored factor series. Its graph basis
/// is unchanged.
/// Calibrate a generated graph-spatial nuisance field while preserving factor metadata.
///
/// The generic matrix calibrator scales the stored time x ROI values. This wrapper applies the
/// same multiplier to the latent factor series and retains the graph basis and ROI labels.
pub fn calibrate_spatial_nuisance_to_component_fraction(
    clean_observations: &[Vec<f64>],
    nuisance_field: &SpatialNuisanceField,
    target_fraction: f64,
) -> Result<
    (SpatialNuisanceField, SpatialVarianceCalibrationReport),
    SpatialVarianceCalibrationError,
> {
    let time_points = clean_observations.len();
    let roi_count = nuisance_field.roi_labels.len();
    let mode_count = nuisance_field.spatial_basis.first().map_or(0, Vec::len);
    if time_points < 3
        || roi_count < 2
        || mode_count == 0
        || nuisance_field.values.len() != time_points
        || clean_observations.iter().any(|row| row.len() != roi_count)
        || nuisance_field.values.iter().any(|row| row.len() != roi_count)
        || nuisance_field.spatial_basis.len() != roi_count
        || nuisance_field.spatial_basis.iter().any(|row| row.len() != mode_count)
        || nuisance_field.factor_series.len() != mode_count
        || nuisance_field.factor_series.iter().any(|series| series.len() != time_points)
    {
        return Err(SpatialVarianceCalibrationError::DimensionMismatch);
    }
    if nuisance_field
        .spatial_basis
        .iter()
        .flatten()
        .chain(nuisance_field.factor_series.iter().flatten())
        .any(|value| !value.is_finite())
    {
        return Err(SpatialVarianceCalibrationError::NonFiniteInput);
    }

    let (calibrated_values, mut report) = calibrate_nuisance_matrix_to_component_fraction(
        clean_observations,
        &nuisance_field.values,
        target_fraction,
    )?;
    report.schema_version = "rcaf.spatial-variance-calibration/0.1.0";

    let mut calibrated = nuisance_field.clone();
    calibrated.values = calibrated_values;
    for value in calibrated.factor_series.iter_mut().flatten() {
        *value *= report.applied_scale_multiplier;
    }
    if calibrated.factor_series.iter().flatten().any(|value| !value.is_finite()) {
        return Err(SpatialVarianceCalibrationError::NonFiniteScale);
    }

    Ok((calibrated, report))
}

fn pooled_within_roi_moments(clean: &[Vec<f64>], nuisance: &[Vec<f64>]) -> (f64, f64, f64) {
    let time_points = clean.len();
    let roi_count = clean[0].len();
    let divisor = (time_points * roi_count) as f64;
    let mut clean_squares = 0.0;
    let mut nuisance_squares = 0.0;
    let mut cross_products = 0.0;

    for roi in 0..roi_count {
        let clean_mean = clean.iter().map(|row| row[roi]).sum::<f64>() / time_points as f64;
        let nuisance_mean = nuisance.iter().map(|row| row[roi]).sum::<f64>() / time_points as f64;
        for time in 0..time_points {
            let clean_centered = clean[time][roi] - clean_mean;
            let nuisance_centered = nuisance[time][roi] - nuisance_mean;
            clean_squares += clean_centered * clean_centered;
            nuisance_squares += nuisance_centered * nuisance_centered;
            cross_products += clean_centered * nuisance_centered;
        }
    }

    (
        clean_squares / divisor,
        nuisance_squares / divisor,
        cross_products / divisor,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial_nuisance::{
        generate_spatial_nuisance_field, SpatialGraph, SpatialNuisanceConfig, SpatialNuisanceFamily,
    };

    fn graph() -> SpatialGraph {
        let labels = (0..6).map(|index| format!("roi-{index}")).collect();
        let mut adjacency = vec![vec![0.0; 6]; 6];
        for index in 0..5 {
            adjacency[index][index + 1] = 1.0;
            adjacency[index + 1][index] = 1.0;
        }
        SpatialGraph::new(labels, adjacency).unwrap()
    }

    fn config(rank: usize, dose: f64, seed: u64) -> SpatialNuisanceConfig {
        SpatialNuisanceConfig {
            family: SpatialNuisanceFamily::GraphSmoothLowRank {
                rank,
                diffusion_time: 0.35,
            },
            time_points: 128,
            dose,
            temporal_ar1: 0.35,
            temporal_coherence: 0.0,
            seed,
        }
    }

    fn clean_observations() -> Vec<Vec<f64>> {
        (0..128)
            .map(|time| {
                (0..6)
                    .map(|roi| {
                        ((time as f64 + 1.0) * (roi as f64 + 1.0) * 0.037).sin()
                            + 0.2 * ((time as f64 + roi as f64) * 0.11).cos()
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn generic_matrix_calibrator_handles_common_mode_and_transient_fields() {
        let clean = clean_observations();
        let common_mode = (0..clean.len())
            .map(|time| {
                let value = 0.4 * ((time as f64 + 1.0) * 0.19).sin();
                vec![value; clean[0].len()]
            })
            .collect::<Vec<_>>();
        let transient = (0..clean.len())
            .map(|time| {
                let value = if time < 20 {
                    0.0
                } else {
                    0.7 * (-((time - 20) as f64) / 9.0).exp()
                };
                (0..clean[0].len())
                    .map(|roi| value * (1.0 + roi as f64 * 0.1))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for nuisance in [&common_mode, &transient] {
            let (calibrated, report) =
                calibrate_nuisance_matrix_to_component_fraction(&clean, nuisance, 0.10)
                    .unwrap();
            assert!((report.realized_component_variance_fraction - 0.10).abs() < 1e-12);
            assert!(calibrated.iter().flatten().any(|value| *value != 0.0));
            assert!(report.clean_nuisance_covariance.is_finite());
            assert!(report.combined_observed_variance.is_finite());
        }
    }

    #[test]
    fn generic_matrix_calibrator_reports_covariance_sign_and_zero_target() {
        let clean = clean_observations();
        let aligned = clean
            .iter()
            .map(|row| row.iter().map(|value| value * 0.1).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let anti_aligned = clean
            .iter()
            .map(|row| row.iter().map(|value| -value * 0.1).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let (_, positive) =
            calibrate_nuisance_matrix_to_component_fraction(&clean, &aligned, 0.10).unwrap();
        let (_, negative) =
            calibrate_nuisance_matrix_to_component_fraction(&clean, &anti_aligned, 0.10).unwrap();
        assert!(positive.clean_nuisance_covariance > 0.0);
        assert!(negative.clean_nuisance_covariance < 0.0);
        let (zero, zero_report) =
            calibrate_nuisance_matrix_to_component_fraction(&clean, &aligned, 0.0).unwrap();
        assert!(zero.iter().flatten().all(|value| *value == 0.0));
        assert_eq!(zero_report.realized_component_variance_fraction, 0.0);
    }

    #[test]
    fn generic_matrix_calibrator_rejects_invalid_and_degenerate_inputs() {
        let clean = clean_observations();
        let zero = (0..clean.len())
            .map(|_| vec![0.0; clean[0].len()])
            .collect::<Vec<_>>();
        assert!(matches!(
            calibrate_nuisance_matrix_to_component_fraction(&clean, &zero, 0.10),
            Err(SpatialVarianceCalibrationError::ZeroNuisanceVariance)
        ));
        assert!(matches!(
            calibrate_nuisance_matrix_to_component_fraction(&clean, &zero[..2], 0.10),
            Err(SpatialVarianceCalibrationError::DimensionMismatch)
        ));
        let mut invalid = clean.clone();
        invalid[0][0] = f64::NAN;
        assert!(matches!(
            calibrate_nuisance_matrix_to_component_fraction(&invalid, &zero, 0.10),
            Err(SpatialVarianceCalibrationError::NonFiniteInput)
        ));
    }

    #[test]
    fn calibrates_to_requested_component_variance_fraction() {
        let clean = clean_observations();
        let field = generate_spatial_nuisance_field(&graph(), &config(2, 0.2, 17), None).unwrap();
        let (calibrated, report) =
            calibrate_spatial_nuisance_to_component_fraction(&clean, &field, 0.15).unwrap();
        assert!((report.realized_component_variance_fraction - 0.15).abs() < 1e-12);
        assert!(report.applied_scale_multiplier.is_finite());
        assert!(calibrated
            .values
            .iter()
            .flatten()
            .any(|value| *value != 0.0));
    }

    #[test]
    fn target_zero_fraction_removes_the_component_exactly() {
        let clean = clean_observations();
        let field = generate_spatial_nuisance_field(&graph(), &config(2, 0.2, 18), None).unwrap();
        let (calibrated, report) =
            calibrate_spatial_nuisance_to_component_fraction(&clean, &field, 0.0).unwrap();
        assert!(calibrated
            .values
            .iter()
            .flatten()
            .all(|value| *value == 0.0));
        assert!(calibrated
            .factor_series
            .iter()
            .flatten()
            .all(|value| *value == 0.0));
        assert_eq!(report.realized_component_variance_fraction, 0.0);
        assert_eq!(report.applied_scale_multiplier, 0.0);
    }

    #[test]
    fn invalid_target_fractions_fail_closed() {
        let clean = clean_observations();
        let field = generate_spatial_nuisance_field(&graph(), &config(2, 0.2, 19), None).unwrap();
        for fraction in [-0.01, 1.0, 1.01, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                calibrate_spatial_nuisance_to_component_fraction(&clean, &field, fraction),
                Err(SpatialVarianceCalibrationError::InvalidTargetFraction)
            ));
        }
    }

    #[test]
    fn component_fraction_calibration_is_invariant_to_initial_additive_dose() {
        let clean = clean_observations();
        let graph = graph();
        let field_a = generate_spatial_nuisance_field(&graph, &config(1, 0.05, 20), None).unwrap();
        let field_b = generate_spatial_nuisance_field(&graph, &config(3, 0.40, 21), None).unwrap();
        let (_, report_a) =
            calibrate_spatial_nuisance_to_component_fraction(&clean, &field_a, 0.20).unwrap();
        let (_, report_b) =
            calibrate_spatial_nuisance_to_component_fraction(&clean, &field_b, 0.20).unwrap();
        assert!((report_a.realized_component_variance_fraction - 0.20).abs() < 1e-12);
        assert!((report_b.realized_component_variance_fraction - 0.20).abs() < 1e-12);
    }

    #[test]
    fn temporal_alignment_is_reported_as_covariance_not_hidden_in_dose() {
        let graph = graph();
        let base_config = config(2, 0.1, 22);
        let first = generate_spatial_nuisance_field(&graph, &base_config, None).unwrap();
        let clean = first.values.clone();
        let (_, report) =
            calibrate_spatial_nuisance_to_component_fraction(&clean, &first, 0.25).unwrap();
        assert!(report.clean_nuisance_covariance > 0.0);
        assert!(report.covariance_contribution_fraction > 0.0);
        assert!(
            report.combined_observed_variance
                > report.clean_component_variance + report.nuisance_component_variance
        );
    }

    #[test]
    fn negative_signal_nuisance_covariance_is_reported_separately() {
        let field = generate_spatial_nuisance_field(&graph(), &config(2, 0.2, 24), None).unwrap();
        let clean = field
            .values
            .iter()
            .map(|row| row.iter().map(|value| -*value).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let (_, report) =
            calibrate_spatial_nuisance_to_component_fraction(&clean, &field, 0.25).unwrap();
        assert!(report.clean_nuisance_covariance < 0.0);
        assert!(report.covariance_contribution_fraction < 0.0);
        assert!(
            report.combined_observed_variance
                < report.clean_component_variance + report.nuisance_component_variance
        );
    }

    #[test]
    fn malformed_factor_metadata_fails_closed() {
        let clean = clean_observations();
        let mut field =
            generate_spatial_nuisance_field(&graph(), &config(2, 0.2, 25), None).unwrap();
        field.factor_series[0].pop();
        assert!(matches!(
            calibrate_spatial_nuisance_to_component_fraction(&clean, &field, 0.2),
            Err(SpatialVarianceCalibrationError::DimensionMismatch)
        ));
    }

    #[test]
    fn mismatched_or_non_finite_inputs_fail_closed() {
        let field = generate_spatial_nuisance_field(&graph(), &config(2, 0.2, 23), None).unwrap();
        assert!(matches!(
            calibrate_spatial_nuisance_to_component_fraction(
                &(0..128).map(|_| vec![0.0; 6]).collect::<Vec<_>>(),
                &field,
                0.2
            ),
            Err(SpatialVarianceCalibrationError::ZeroCleanVariance)
        ));
        assert!(matches!(
            calibrate_spatial_nuisance_to_component_fraction(
                &(0..128).map(|_| vec![0.0; 5]).collect::<Vec<_>>(),
                &field,
                0.2
            ),
            Err(SpatialVarianceCalibrationError::DimensionMismatch)
        ));
        let mut invalid = clean_observations();
        invalid[0][0] = f64::NAN;
        assert!(matches!(
            calibrate_spatial_nuisance_to_component_fraction(&invalid, &field, 0.2),
            Err(SpatialVarianceCalibrationError::NonFiniteInput)
        ));
    }
}
