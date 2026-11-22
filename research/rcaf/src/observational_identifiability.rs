//! Constructive observational-identifiability and abstention reference for RCAF.
//!
//! This module does not infer causality. It constructs two known counterfactual worlds whose
//! observations are equal when target and artifact contributions are observationally identical,
//! then enforces an abstention disposition at the declared observation resolution.

use std::fmt;

/// A pair of worlds with identical intended spatial-temporal contributions but different causes.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservationallyEquivalentPair {
    /// Clean observations plus a latent target-change contribution.
    pub target_change_world: Vec<Vec<f64>>,
    /// The same clean observations plus an observation-artifact contribution.
    pub artifact_only_world: Vec<Vec<f64>>,
}

/// Disposition after comparing two observation matrices at a predeclared resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservationalDisposition {
    /// The worlds cannot be distinguished at the declared resolution; causal attribution must
    /// abstain.
    NotIdentifiableAbstain,
    /// The matrices differ beyond the declared resolution. This is not a causal verdict.
    ObservationallyDistinctAtDeclaredResolution,
}

impl ObservationalDisposition {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotIdentifiableAbstain => "not_identifiable_abstain",
            Self::ObservationallyDistinctAtDeclaredResolution => {
                "observationally_distinct_at_declared_resolution"
            }
        }
    }
}

/// Report for a pairwise observational-equivalence check.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservationalIdentifiabilityReport {
    pub schema_version: &'static str,
    pub time_points: usize,
    pub roi_count: usize,
    /// Predeclared maximum absolute observation difference accepted as indistinguishable.
    pub resolution_tolerance: f64,
    pub max_absolute_difference: f64,
    pub root_mean_square_difference: f64,
    pub compared_values: usize,
    pub observationally_equivalent_at_resolution: bool,
    pub disposition: ObservationalDisposition,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObservationalIdentifiabilityError {
    EmptyObservations,
    InvalidObservationShape,
    DimensionMismatch,
    InvalidResolutionTolerance,
    NonFiniteInput,
    NonFiniteOutput,
    InvalidComponentAmplitude,
}

impl fmt::Display for ObservationalIdentifiabilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyObservations => write!(
                f,
                "observations must contain at least one time point and ROI"
            ),
            Self::InvalidObservationShape => write!(f, "observations must be rectangular"),
            Self::DimensionMismatch => write!(
                f,
                "counterfactual observations or component dimensions differ"
            ),
            Self::InvalidResolutionTolerance => {
                write!(
                    f,
                    "observation-resolution tolerance must be finite and non-negative"
                )
            }
            Self::NonFiniteInput => write!(f, "observations and component inputs must be finite"),
            Self::NonFiniteOutput => write!(
                f,
                "generated counterfactual observations must remain finite"
            ),
            Self::InvalidComponentAmplitude => write!(f, "component amplitude must be finite"),
        }
    }
}

impl std::error::Error for ObservationalIdentifiabilityError {}

/// Construct two worlds that are observationally identical by design.
///
/// For clean data X, spatial loading u, temporal process z, and amplitude a:
///
/// - target world: X + a z uᵀ, where the contribution is declared to be a latent target change;
/// - artifact world: X + a z uᵀ, where the same contribution is declared to be an observation artifact.
///
/// The generating cause differs, while every observed value is identical. This is a constructive
/// non-identifiability control, not a model of any particular biological or scanner artifact.
pub fn construct_observationally_equivalent_pair(
    clean_observations: &[Vec<f64>],
    spatial_loading: &[f64],
    temporal_process: &[f64],
    amplitude: f64,
) -> Result<ObservationallyEquivalentPair, ObservationalIdentifiabilityError> {
    validate_component_inputs(
        clean_observations,
        spatial_loading,
        temporal_process,
        amplitude,
    )?;

    let target_change_world = add_separable_component(
        clean_observations,
        spatial_loading,
        temporal_process,
        amplitude,
    );
    let artifact_only_world = add_separable_component(
        clean_observations,
        spatial_loading,
        temporal_process,
        amplitude,
    );
    if target_change_world
        .iter()
        .flatten()
        .chain(artifact_only_world.iter().flatten())
        .any(|value| !value.is_finite())
    {
        return Err(ObservationalIdentifiabilityError::NonFiniteOutput);
    }

    Ok(ObservationallyEquivalentPair {
        target_change_world,
        artifact_only_world,
    })
}

/// Compare two observation matrices at a declared instrument/data-resolution tolerance.
///
/// If the maximum absolute difference is at or below the tolerance, the disposition is
/// NOT_IDENTIFIABLE_ABSTAIN. A value above the tolerance only means the matrices are
/// distinguishable at that resolution; it does not establish whether a difference is biological,
/// artifactual, causal, or scientifically meaningful.
pub fn assess_observational_identifiability(
    left: &[Vec<f64>],
    right: &[Vec<f64>],
    resolution_tolerance: f64,
) -> Result<ObservationalIdentifiabilityReport, ObservationalIdentifiabilityError> {
    if !resolution_tolerance.is_finite() || resolution_tolerance < 0.0 {
        return Err(ObservationalIdentifiabilityError::InvalidResolutionTolerance);
    }
    let (time_points, roi_count) = validate_observation_matrix(left)?;
    let (right_time_points, right_roi_count) = validate_observation_matrix(right)?;
    if time_points != right_time_points || roi_count != right_roi_count {
        return Err(ObservationalIdentifiabilityError::DimensionMismatch);
    }

    let mut max_absolute_difference = 0.0f64;
    let mut sum_squared_difference = 0.0;
    let compared_values = time_points * roi_count;
    for (left_row, right_row) in left.iter().zip(right.iter()) {
        for (&left_value, &right_value) in left_row.iter().zip(right_row.iter()) {
            let difference = (left_value - right_value).abs();
            if !difference.is_finite() {
                return Err(ObservationalIdentifiabilityError::NonFiniteInput);
            }
            max_absolute_difference = max_absolute_difference.max(difference);
            sum_squared_difference += difference * difference;
        }
    }

    let root_mean_square_difference = (sum_squared_difference / compared_values as f64).sqrt();
    if !root_mean_square_difference.is_finite() {
        return Err(ObservationalIdentifiabilityError::NonFiniteInput);
    }

    let observationally_equivalent_at_resolution = max_absolute_difference <= resolution_tolerance;
    let disposition = if observationally_equivalent_at_resolution {
        ObservationalDisposition::NotIdentifiableAbstain
    } else {
        ObservationalDisposition::ObservationallyDistinctAtDeclaredResolution
    };

    Ok(ObservationalIdentifiabilityReport {
        schema_version: "rcaf.observational-identifiability/0.1.0",
        time_points,
        roi_count,
        resolution_tolerance,
        max_absolute_difference,
        root_mean_square_difference,
        compared_values,
        observationally_equivalent_at_resolution,
        disposition,
    })
}

fn validate_component_inputs(
    clean_observations: &[Vec<f64>],
    spatial_loading: &[f64],
    temporal_process: &[f64],
    amplitude: f64,
) -> Result<(), ObservationalIdentifiabilityError> {
    let (time_points, roi_count) = validate_observation_matrix(clean_observations)?;
    if spatial_loading.len() != roi_count || temporal_process.len() != time_points {
        return Err(ObservationalIdentifiabilityError::DimensionMismatch);
    }
    if !amplitude.is_finite() {
        return Err(ObservationalIdentifiabilityError::InvalidComponentAmplitude);
    }
    if spatial_loading
        .iter()
        .chain(temporal_process.iter())
        .any(|value| !value.is_finite())
    {
        return Err(ObservationalIdentifiabilityError::NonFiniteInput);
    }
    Ok(())
}

fn validate_observation_matrix(
    observations: &[Vec<f64>],
) -> Result<(usize, usize), ObservationalIdentifiabilityError> {
    if observations.is_empty() || observations[0].is_empty() {
        return Err(ObservationalIdentifiabilityError::EmptyObservations);
    }
    let roi_count = observations[0].len();
    if observations.iter().any(|row| row.len() != roi_count) {
        return Err(ObservationalIdentifiabilityError::InvalidObservationShape);
    }
    if observations
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(ObservationalIdentifiabilityError::NonFiniteInput);
    }
    Ok((observations.len(), roi_count))
}

fn add_separable_component(
    clean_observations: &[Vec<f64>],
    spatial_loading: &[f64],
    temporal_process: &[f64],
    amplitude: f64,
) -> Vec<Vec<f64>> {
    clean_observations
        .iter()
        .zip(temporal_process.iter())
        .map(|(row, &temporal_value)| {
            row.iter()
                .zip(spatial_loading.iter())
                .map(|(&clean, &spatial_value)| clean + amplitude * temporal_value * spatial_value)
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clean() -> Vec<Vec<f64>> {
        (0..12)
            .map(|time| {
                (0..4)
                    .map(|roi| ((time as f64 + 1.0) * (roi as f64 + 1.0) * 0.17).sin())
                    .collect()
            })
            .collect()
    }

    #[test]
    fn exactly_equivalent_target_and_artifact_worlds_require_abstention() {
        let clean = clean();
        let spatial = [0.5, -0.5, 0.5, -0.5];
        let temporal = (0..12)
            .map(|time| (time as f64 * 0.23).sin())
            .collect::<Vec<_>>();
        let pair =
            construct_observationally_equivalent_pair(&clean, &spatial, &temporal, 0.3).unwrap();
        assert_eq!(pair.target_change_world, pair.artifact_only_world);

        let report = assess_observational_identifiability(
            &pair.target_change_world,
            &pair.artifact_only_world,
            0.0,
        )
        .unwrap();
        assert!(report.observationally_equivalent_at_resolution);
        assert_eq!(
            report.disposition,
            ObservationalDisposition::NotIdentifiableAbstain
        );
        assert_eq!(report.max_absolute_difference, 0.0);
    }

    #[test]
    fn differences_below_declared_resolution_require_abstention() {
        let left = vec![vec![0.0, 1.0], vec![2.0, 3.0]];
        let right = vec![vec![1e-5, 1.0], vec![2.0, 3.0]];
        let report = assess_observational_identifiability(&left, &right, 1e-4).unwrap();
        assert_eq!(
            report.disposition,
            ObservationalDisposition::NotIdentifiableAbstain
        );
        assert!(report.max_absolute_difference <= report.resolution_tolerance);
    }

    #[test]
    fn differences_above_declared_resolution_are_only_labelled_distinct() {
        let left = vec![vec![0.0, 1.0], vec![2.0, 3.0]];
        let right = vec![vec![0.01, 1.0], vec![2.0, 3.0]];
        let report = assess_observational_identifiability(&left, &right, 1e-4).unwrap();
        assert_eq!(
            report.disposition,
            ObservationalDisposition::ObservationallyDistinctAtDeclaredResolution
        );
        assert!(report.max_absolute_difference > report.resolution_tolerance);
    }

    #[test]
    fn invalid_tolerance_and_mismatched_shapes_fail_closed() {
        let left = vec![vec![0.0, 1.0], vec![2.0, 3.0]];
        for tolerance in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                assess_observational_identifiability(&left, &left, tolerance),
                Err(ObservationalIdentifiabilityError::InvalidResolutionTolerance)
            ));
        }
        assert!(matches!(
            assess_observational_identifiability(&left, &[vec![0.0, 1.0]], 0.0),
            Err(ObservationalIdentifiabilityError::DimensionMismatch)
        ));
    }

    #[test]
    fn invalid_component_dimensions_and_non_finite_values_fail_closed() {
        let clean = clean();
        let spatial = [1.0, 0.0, -1.0, 0.0];
        let temporal = vec![1.0; 12];
        assert!(matches!(
            construct_observationally_equivalent_pair(&clean, &[1.0, 0.0], &temporal, 1.0),
            Err(ObservationalIdentifiabilityError::DimensionMismatch)
        ));
        let mut invalid_temporal = temporal.clone();
        invalid_temporal[0] = f64::NAN;
        assert!(matches!(
            construct_observationally_equivalent_pair(&clean, &spatial, &invalid_temporal, 1.0),
            Err(ObservationalIdentifiabilityError::NonFiniteInput)
        ));
        assert!(matches!(
            construct_observationally_equivalent_pair(&clean, &spatial, &temporal, f64::INFINITY),
            Err(ObservationalIdentifiabilityError::InvalidComponentAmplitude)
        ));
    }

    #[test]
    fn finite_inputs_that_overflow_the_generated_world_fail_closed() {
        let clean = clean();
        let spatial = [2.0, 2.0, 2.0, 2.0];
        let temporal = vec![2.0; 12];
        assert!(matches!(
            construct_observationally_equivalent_pair(&clean, &spatial, &temporal, f64::MAX),
            Err(ObservationalIdentifiabilityError::NonFiniteOutput)
        ));
    }

    #[test]
    fn report_never_upgrades_distinguishability_to_a_causal_label() {
        let left = vec![vec![0.0, 1.0], vec![2.0, 3.0]];
        let right = vec![vec![0.0, 1.0], vec![2.0, 3.1]];
        let report = assess_observational_identifiability(&left, &right, 1e-6).unwrap();
        assert_eq!(
            report.disposition.as_str(),
            "observationally_distinct_at_declared_resolution"
        );
        assert!(!report.disposition.as_str().contains("causal"));
        assert!(!report.disposition.as_str().contains("biological"));
    }
}
