//! Crossed factorial signal-strength × nuisance-variance response surface for RCAF.
//!
//! One clean matrix, one target-signal innovation and one nuisance innovation are frozen before
//! the sweep. Signal scales and nuisance component-variance fractions are varied without redrawing
//! either matrix. This is a common-random-numbers construction that isolates declared dose changes.

use crate::estimator_response::{
    evaluate_factorial_estimator_response, ConnectivityDirectionReport, EstimatorResponseError,
    EstimatorResponseSummary,
};
use crate::factorial_counterfactual::{
    run_factorial_counterfactual_experiment, FactorialCellSummary, FactorialExperimentError,
    FactorialExperimentMetadata, FactorialInvariantReport,
};
use crate::spatial_calibration::{
    calibrate_nuisance_matrix_to_component_fraction, SpatialVarianceCalibrationError,
};
use std::fmt;

/// Configuration and stable provenance for one raw signal × nuisance realization.
#[derive(Debug, Clone, PartialEq)]
pub struct FactorialDoseResponseConfig {
    pub metadata: FactorialExperimentMetadata,
    /// Strictly increasing, finite, non-negative signal amplitude multipliers.
    pub signal_scales: Vec<f64>,
    /// Strictly increasing component-variance fractions in [0, 1).
    pub nuisance_component_variance_fractions: Vec<f64>,
}

/// One point in the crossed response surface.
#[derive(Debug, Clone, PartialEq)]
pub struct FactorialDoseResponsePoint {
    pub signal_scale_index: usize,
    pub nuisance_fraction_index: usize,
    pub signal_scale: f64,
    pub requested_nuisance_component_variance_fraction: f64,
    pub realized_nuisance_component_variance_fraction: f64,
    pub nuisance_scale_multiplier: f64,
    pub clean_component_variance: f64,
    pub nuisance_component_variance: f64,
    pub clean_nuisance_covariance: f64,
    pub combined_observed_variance: f64,
    pub signal_component_rms: f64,
    pub nuisance_component_rms: f64,
    pub estimators: Vec<EstimatorResponseSummary>,
    pub connectivity_direction: ConnectivityDirectionReport,
    pub cells: Vec<FactorialCellSummary>,
    pub invariants: FactorialInvariantReport,
}

/// Machine-readable crossed response surface.
#[derive(Debug, Clone, PartialEq)]
pub struct FactorialDoseResponseReport {
    pub schema_version: &'static str,
    pub experiment_id: String,
    pub signal_id: String,
    pub nuisance_family_id: String,
    pub nuisance_instance_id: String,
    pub time_points: usize,
    pub roi_count: usize,
    pub signal_scales: Vec<f64>,
    pub nuisance_component_variance_fractions: Vec<f64>,
    /// Fingerprints of the raw, unscaled innovations; they are constant across the surface.
    pub raw_signal_fingerprint: String,
    pub raw_nuisance_fingerprint: String,
    pub points: Vec<FactorialDoseResponsePoint>,
}

impl FactorialDoseResponseReport {
    /// Render deterministic JSON without adding a serialization dependency to the standalone crate.
    pub fn to_json_pretty(&self) -> String {
        let signal_scales = number_array_json(&self.signal_scales);
        let nuisance_fractions = number_array_json(&self.nuisance_component_variance_fractions);
        let points = self
            .points
            .iter()
            .map(render_point)
            .collect::<Vec<_>>()
            .join(",\n");
        format!(
            concat!(
                "{{\n",
                "  \"schema_version\": \"{}\",\n",
                "  \"source_status\": \"synthetic_factorial_sweep\",\n",
                "  \"experiment_id\": \"{}\",\n",
                "  \"signal_id\": \"{}\",\n",
                "  \"nuisance_family_id\": \"{}\",\n",
                "  \"nuisance_instance_id\": \"{}\",\n",
                "  \"time_points\": {},\n",
                "  \"roi_count\": {},\n",
                "  \"signal_scales\": [{}],\n",
                "  \"nuisance_component_variance_fractions\": [{}],\n",
                "  \"raw_signal_fingerprint\": \"{}\",\n",
                "  \"raw_nuisance_fingerprint\": \"{}\",\n",
                "  \"point_count\": {},\n",
                "  \"points\": [\n{}\n  ],\n",
                "  \"claim_limits\": [\n",
                "    \"fixed raw synthetic signal and nuisance innovations; no redraw across sweep points\",\n",
                "    \"nuisance fraction is Var(A)/(Var(X)+Var(A)) and excludes covariance by definition\",\n",
                "    \"the covariance and combined variance are reported independently at every point\",\n",
                "    \"construction and calibration response surface only; no estimator or empirical pipeline is qualified\",\n",
                "    \"FNV-1a fingerprints are diagnostics, not cryptographic digests; workflow receipt supplies content addressing\"\n",
                "  ]\n",
                "}}\n"
            ),
            self.schema_version,
            self.experiment_id,
            self.signal_id,
            self.nuisance_family_id,
            self.nuisance_instance_id,
            self.time_points,
            self.roi_count,
            signal_scales,
            nuisance_fractions,
            self.raw_signal_fingerprint,
            self.raw_nuisance_fingerprint,
            self.points.len(),
            points,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FactorialDoseResponseError {
    EmptySignalScaleGrid,
    EmptyNuisanceFractionGrid,
    InvalidSignalScaleGrid,
    InvalidNuisanceFractionGrid,
    EmptyMatrix,
    RaggedMatrix(&'static str),
    InsufficientMatrixShape,
    DimensionMismatch(&'static str),
    NonFiniteInput(&'static str),
    NonFiniteScaledSignal,
    FactorialInvariantFailure,
    Factorial(FactorialExperimentError),
    Calibration(SpatialVarianceCalibrationError),
    EstimatorResponse(EstimatorResponseError),
}

impl fmt::Display for FactorialDoseResponseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySignalScaleGrid => write!(f, "signal scale grid must not be empty"),
            Self::EmptyNuisanceFractionGrid => {
                write!(f, "nuisance variance-fraction grid must not be empty")
            }
            Self::InvalidSignalScaleGrid => write!(
                f,
                "signal scales must be finite, non-negative and strictly increasing"
            ),
            Self::InvalidNuisanceFractionGrid => write!(
                f,
                "nuisance fractions must be finite, in [0, 1), and strictly increasing"
            ),
            Self::EmptyMatrix => write!(f, "sweep input matrices must be non-empty"),
            Self::RaggedMatrix(name) => write!(f, "{name} must be rectangular"),
            Self::InsufficientMatrixShape => {
                write!(
                    f,
                    "sweep inputs require at least three time points and two ROIs"
                )
            }
            Self::DimensionMismatch(name) => write!(f, "sweep dimension mismatch: {name}"),
            Self::NonFiniteInput(name) => write!(f, "{name} contains non-finite values"),
            Self::NonFiniteScaledSignal => {
                write!(f, "signal scaling produced non-finite observations")
            }
            Self::FactorialInvariantFailure => {
                write!(
                    f,
                    "factorial matched-component invariants were not preserved"
                )
            }
            Self::Factorial(error) => write!(f, "factorial cell construction failed: {error}"),
            Self::Calibration(error) => write!(f, "nuisance calibration failed: {error}"),
            Self::EstimatorResponse(error) => write!(f, "estimator response failed: {error}"),
        }
    }
}

impl std::error::Error for FactorialDoseResponseError {}

impl From<FactorialExperimentError> for FactorialDoseResponseError {
    fn from(value: FactorialExperimentError) -> Self {
        Self::Factorial(value)
    }
}

impl From<SpatialVarianceCalibrationError> for FactorialDoseResponseError {
    fn from(value: SpatialVarianceCalibrationError) -> Self {
        Self::Calibration(value)
    }
}

impl From<EstimatorResponseError> for FactorialDoseResponseError {
    fn from(value: EstimatorResponseError) -> Self {
        Self::EstimatorResponse(value)
    }
}

/// Build a full signal-scale × nuisance-fraction factorial response surface.
///
/// The raw signal and nuisance matrices are frozen once. Each signal scale multiplies the same
/// signal innovation; each nuisance fraction calibrates the same raw nuisance innovation against
/// the same clean observations. Every grid point then runs the canonical four-cell engine.
///
/// This function evaluates construction and nuisance calibration only. Downstream estimator
/// recovery, null false-positive rate, confidence coverage and abstention must be computed by
/// separate estimator qualification stages.
pub fn run_factorial_dose_response_sweep(
    clean_observations: &[Vec<f64>],
    raw_signal_component: &[Vec<f64>],
    raw_nuisance_component: &[Vec<f64>],
    config: &FactorialDoseResponseConfig,
) -> Result<FactorialDoseResponseReport, FactorialDoseResponseError> {
    validate_signal_scales(&config.signal_scales)?;
    validate_nuisance_fractions(&config.nuisance_component_variance_fractions)?;
    let (time_points, roi_count) = validate_matrix(clean_observations, "clean observations")?;
    validate_matching_matrix(raw_signal_component, time_points, roi_count, "raw signal")?;
    validate_matching_matrix(
        raw_nuisance_component,
        time_points,
        roi_count,
        "raw nuisance",
    )?;

    let raw_signal_fingerprint = matrix_fingerprint(raw_signal_component);
    let raw_nuisance_fingerprint = matrix_fingerprint(raw_nuisance_component);

    // Precompute each intervention level once: the innovations remain fixed across the cross-grid.
    let mut scaled_signals = Vec::with_capacity(config.signal_scales.len());
    for &signal_scale in &config.signal_scales {
        let scaled = scale_matrix(raw_signal_component, signal_scale)?;
        scaled_signals.push(scaled);
    }

    let mut calibrated_nuisances =
        Vec::with_capacity(config.nuisance_component_variance_fractions.len());
    let mut calibration_reports =
        Vec::with_capacity(config.nuisance_component_variance_fractions.len());
    for &target_fraction in &config.nuisance_component_variance_fractions {
        let (calibrated, calibration_report) = calibrate_nuisance_matrix_to_component_fraction(
            clean_observations,
            raw_nuisance_component,
            target_fraction,
        )?;
        calibrated_nuisances.push(calibrated);
        calibration_reports.push(calibration_report);
    }

    let mut points = Vec::with_capacity(
        config.signal_scales.len() * config.nuisance_component_variance_fractions.len(),
    );
    for (signal_index, (signal_scale, signal_matrix)) in config
        .signal_scales
        .iter()
        .zip(scaled_signals.iter())
        .enumerate()
    {
        for (fraction_index, (target_fraction, nuisance_matrix)) in config
            .nuisance_component_variance_fractions
            .iter()
            .zip(calibrated_nuisances.iter())
            .enumerate()
        {
            let mut metadata = config.metadata.clone();
            metadata.experiment_id = format!(
                "{}-s{:02}-n{:02}",
                config.metadata.experiment_id, signal_index, fraction_index
            );
            metadata.target_component_variance_fraction = Some(*target_fraction);

            let experiment = run_factorial_counterfactual_experiment(
                clean_observations,
                signal_matrix,
                nuisance_matrix,
                &metadata,
            )?;
            if !experiment.report.invariants.all_invariants_within_tolerance {
                return Err(FactorialDoseResponseError::FactorialInvariantFailure);
            }
            let estimator_response = evaluate_factorial_estimator_response(&experiment)?;
            let calibration = &calibration_reports[fraction_index];
            points.push(FactorialDoseResponsePoint {
                signal_scale_index: signal_index,
                nuisance_fraction_index: fraction_index,
                signal_scale: *signal_scale,
                requested_nuisance_component_variance_fraction: *target_fraction,
                realized_nuisance_component_variance_fraction: experiment
                    .report
                    .realized_component_variance_fraction,
                nuisance_scale_multiplier: calibration.applied_scale_multiplier,
                clean_component_variance: experiment.report.clean_component_variance,
                nuisance_component_variance: experiment.report.nuisance_component_variance,
                clean_nuisance_covariance: experiment.report.clean_nuisance_covariance,
                combined_observed_variance: experiment.report.combined_observed_variance,
                signal_component_rms: experiment.report.signal_component_rms,
                nuisance_component_rms: experiment.report.nuisance_component_rms,
                estimators: estimator_response.estimators,
                connectivity_direction: estimator_response.connectivity_direction,
                cells: experiment.report.cells,
                invariants: experiment.report.invariants,
            });
        }
    }

    Ok(FactorialDoseResponseReport {
        schema_version: "rcaf.factorial-dose-response/0.1.0",
        experiment_id: config.metadata.experiment_id.clone(),
        signal_id: config.metadata.signal_id.clone(),
        nuisance_family_id: config.metadata.nuisance_family_id.clone(),
        nuisance_instance_id: config.metadata.nuisance_instance_id.clone(),
        time_points,
        roi_count,
        signal_scales: config.signal_scales.clone(),
        nuisance_component_variance_fractions: config.nuisance_component_variance_fractions.clone(),
        raw_signal_fingerprint,
        raw_nuisance_fingerprint,
        points,
    })
}

fn validate_signal_scales(scales: &[f64]) -> Result<(), FactorialDoseResponseError> {
    if scales.is_empty() {
        return Err(FactorialDoseResponseError::EmptySignalScaleGrid);
    }
    if scales
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
        || scales.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(FactorialDoseResponseError::InvalidSignalScaleGrid);
    }
    Ok(())
}

fn validate_nuisance_fractions(fractions: &[f64]) -> Result<(), FactorialDoseResponseError> {
    if fractions.is_empty() {
        return Err(FactorialDoseResponseError::EmptyNuisanceFractionGrid);
    }
    if fractions
        .iter()
        .any(|value| !value.is_finite() || !(0.0..1.0).contains(value))
        || fractions.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(FactorialDoseResponseError::InvalidNuisanceFractionGrid);
    }
    Ok(())
}

fn validate_matrix(
    matrix: &[Vec<f64>],
    name: &'static str,
) -> Result<(usize, usize), FactorialDoseResponseError> {
    if matrix.is_empty() || matrix.first().is_none_or(Vec::is_empty) {
        return Err(FactorialDoseResponseError::EmptyMatrix);
    }
    let roi_count = matrix[0].len();
    if matrix.iter().any(|row| row.len() != roi_count) {
        return Err(FactorialDoseResponseError::RaggedMatrix(name));
    }
    if matrix.len() < 3 || roi_count < 2 {
        return Err(FactorialDoseResponseError::InsufficientMatrixShape);
    }
    if matrix.iter().flatten().any(|value| !value.is_finite()) {
        return Err(FactorialDoseResponseError::NonFiniteInput(name));
    }
    Ok((matrix.len(), roi_count))
}

fn validate_matching_matrix(
    matrix: &[Vec<f64>],
    time_points: usize,
    roi_count: usize,
    name: &'static str,
) -> Result<(), FactorialDoseResponseError> {
    let (actual_time_points, actual_roi_count) = validate_matrix(matrix, name)?;
    if actual_time_points != time_points || actual_roi_count != roi_count {
        return Err(FactorialDoseResponseError::DimensionMismatch(name));
    }
    Ok(())
}

fn scale_matrix(
    matrix: &[Vec<f64>],
    scale: f64,
) -> Result<Vec<Vec<f64>>, FactorialDoseResponseError> {
    let scaled = matrix
        .iter()
        .map(|row| row.iter().map(|value| value * scale).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    if scaled.iter().flatten().any(|value| !value.is_finite()) {
        return Err(FactorialDoseResponseError::NonFiniteScaledSignal);
    }
    Ok(scaled)
}

fn matrix_fingerprint(matrix: &[Vec<f64>]) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for dimension in [matrix.len(), matrix.first().map_or(0, Vec::len)] {
        for byte in (dimension as u64).to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    for row in matrix {
        for value in row {
            for byte in value.to_bits().to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
    format!("{hash:016x}")
}

fn number_array_json(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| format!("{value:.12}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn render_point(point: &FactorialDoseResponsePoint) -> String {
    let estimators = point
        .estimators
        .iter()
        .map(|item| {
            let nuisance_ratio = item
                .nuisance_to_signal_distance_ratio
                .map(|value| format!("{value:.12}"))
                .unwrap_or_else(|| "null".to_string());
            let retention = item
                .signal_retention_fraction
                .map(|value| format!("{value:.12}"))
                .unwrap_or_else(|| "null".to_string());
            format!(
                concat!(
                    "        {{\n",
                    "          \"estimator_id\": \"{}\",\n",
                    "          \"clean_to_signal_similarity\": {:.12},\n",
                    "          \"clean_to_nuisance_similarity\": {:.12},\n",
                    "          \"signal_to_combined_similarity\": {:.12},\n",
                    "          \"clean_to_combined_similarity\": {:.12},\n",
                    "          \"clean_signal_distance\": {:.12},\n",
                    "          \"nuisance_only_distance\": {:.12},\n",
                    "          \"combined_signal_residual_distance\": {:.12},\n",
                    "          \"nuisance_to_signal_distance_ratio\": {},\n",
                    "          \"signal_retention_fraction\": {}\n",
                    "        }}"
                ),
                item.estimator_id,
                item.clean_to_signal_similarity,
                item.clean_to_nuisance_similarity,
                item.signal_to_combined_similarity,
                item.clean_to_combined_similarity,
                item.clean_signal_distance,
                item.nuisance_only_distance,
                item.combined_signal_residual_distance,
                nuisance_ratio,
                retention,
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");
    let direction = &point.connectivity_direction;
    let direction_gain = direction
        .signal_projection_gain
        .map(|value| format!("{value:.12}"))
        .unwrap_or_else(|| "null".to_string());
    let direction_cosine = direction
        .signal_directional_cosine
        .map(|value| format!("{value:.12}"))
        .unwrap_or_else(|| "null".to_string());
    let direction_residual = direction
        .signal_change_residual_fraction
        .map(|value| format!("{value:.12}"))
        .unwrap_or_else(|| "null".to_string());

    let cells = point
        .cells
        .iter()
        .map(|cell| {
            format!(
                concat!(
                    "        {\n",
                    "          \"cell_id\": \"{}\",\n",
                    "          \"rms\": {:.12},\n",
                    "          \"max_absolute_value\": {:.12},\n",
                    "          \"fnv1a64_fingerprint\": \"{}\"\n",
                    "        }"
                ),
                cell.cell_id, cell.rms, cell.max_absolute_value, cell.fnv1a64_fingerprint
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");

    format!(
        concat!(
            "    {{\n",
            "      \"signal_scale_index\": {},\n",
            "      \"nuisance_fraction_index\": {},\n",
            "      \"signal_scale\": {:.12},\n",
            "      \"requested_nuisance_component_variance_fraction\": {:.12},\n",
            "      \"realized_nuisance_component_variance_fraction\": {:.12},\n",
            "      \"nuisance_scale_multiplier\": {:.12},\n",
            "      \"clean_component_variance\": {:.12},\n",
            "      \"nuisance_component_variance\": {:.12},\n",
            "      \"clean_nuisance_covariance\": {:.12},\n",
            "      \"combined_observed_variance\": {:.12},\n",
            "      \"signal_component_rms\": {:.12},\n",
            "      \"nuisance_component_rms\": {:.12},\n",
            "      \"estimators\": [\n{}\n      ],\n",
            "      \"connectivity_direction\": {{\n",
            "        \"signal_projection_gain\": {},\n",
            "        \"signal_directional_cosine\": {},\n",
            "        \"signal_change_residual_fraction\": {},\n",
            "        \"clean_signal_change_norm\": {:.12},\n",
            "        \"combined_signal_change_norm\": {:.12}\n",
            "      }},\n",
            "      \"cells\": [\n{}\n      ],\n",
            "      \"invariants\": {{\n",
            "        \"matched_nuisance_delta_max_abs\": {:.12},\n",
            "        \"matched_signal_delta_max_abs\": {:.12},\n",
            "        \"additive_interaction_max_abs\": {:.12},\n",
            "        \"numerical_tolerance\": {:.12},\n",
            "        \"all_invariants_within_tolerance\": {}\n",
            "      }}\n",
            "    }}"
        ),
        point.signal_scale_index,
        point.nuisance_fraction_index,
        point.signal_scale,
        point.requested_nuisance_component_variance_fraction,
        point.realized_nuisance_component_variance_fraction,
        point.nuisance_scale_multiplier,
        point.clean_component_variance,
        point.nuisance_component_variance,
        point.clean_nuisance_covariance,
        point.combined_observed_variance,
        point.signal_component_rms,
        point.nuisance_component_rms,
        estimators,
        direction_gain,
        direction_cosine,
        direction_residual,
        direction.clean_signal_change_norm,
        direction.combined_signal_change_norm,
        cells,
        point.invariants.matched_nuisance_delta_max_abs,
        point.invariants.matched_signal_delta_max_abs,
        point.invariants.additive_interaction_max_abs,
        point.invariants.numerical_tolerance,
        point.invariants.all_invariants_within_tolerance,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata() -> FactorialExperimentMetadata {
        FactorialExperimentMetadata {
            experiment_id: "sweep-fixture-01".to_string(),
            signal_id: "target-network-up".to_string(),
            nuisance_family_id: "motion_transient".to_string(),
            nuisance_instance_id: "fixed-transient-instance-01".to_string(),
            target_component_variance_fraction: None,
        }
    }

    fn clean() -> Vec<Vec<f64>> {
        (0..48)
            .map(|time| {
                (0..4)
                    .map(|roi| {
                        ((time as f64 + 1.0) * (roi as f64 + 1.0) * 0.071).sin()
                            + 0.13 * ((time as f64 + roi as f64) * 0.17).cos()
                    })
                    .collect()
            })
            .collect()
    }

    fn signal() -> Vec<Vec<f64>> {
        (0..48)
            .map(|time| {
                (0..4)
                    .map(|roi| 0.08 * ((time as f64 + 2.0) * (roi as f64 + 1.0) * 0.043).sin())
                    .collect()
            })
            .collect()
    }

    fn nuisance() -> Vec<Vec<f64>> {
        (0usize..48)
            .map(|time| {
                (0..4)
                    .map(|roi| {
                        let elapsed = time.saturating_sub(13) as f64;
                        if time < 13 {
                            0.0
                        } else {
                            0.4 * (-(elapsed / 5.0)).exp() * (1.0 + roi as f64 * 0.1)
                        }
                    })
                    .collect()
            })
            .collect()
    }

    fn config() -> FactorialDoseResponseConfig {
        FactorialDoseResponseConfig {
            metadata: metadata(),
            signal_scales: vec![0.0, 0.5, 1.0, 2.0],
            nuisance_component_variance_fractions: vec![0.0, 0.05, 0.10, 0.20],
        }
    }

    #[test]
    fn grid_contains_cartesian_product_in_stable_row_major_order() {
        let report =
            run_factorial_dose_response_sweep(&clean(), &signal(), &nuisance(), &config()).unwrap();
        assert_eq!(report.points.len(), 16);
        let grid = config();
        for (signal_index, &signal_scale) in grid.signal_scales.iter().enumerate() {
            for (nuisance_index, &nuisance_fraction) in grid
                .nuisance_component_variance_fractions
                .iter()
                .enumerate()
            {
                let point = &report.points[signal_index * 4 + nuisance_index];
                assert_eq!(point.signal_scale_index, signal_index);
                assert_eq!(point.nuisance_fraction_index, nuisance_index);
                assert_eq!(point.signal_scale, signal_scale);
                assert_eq!(
                    point.requested_nuisance_component_variance_fraction,
                    nuisance_fraction
                );
                assert!(point.invariants.all_invariants_within_tolerance);
            }
        }
    }

    #[test]
    fn requested_and_realized_nuisance_fraction_follow_the_dose_axis() {
        let report =
            run_factorial_dose_response_sweep(&clean(), &signal(), &nuisance(), &config()).unwrap();
        for row in report.points.chunks_exact(4) {
            for point in row {
                assert!(
                    (point.realized_nuisance_component_variance_fraction
                        - point.requested_nuisance_component_variance_fraction)
                        .abs()
                        < 1e-12
                );
            }
            assert!(row.windows(2).all(|pair| {
                pair[0].nuisance_component_variance < pair[1].nuisance_component_variance
            }));
        }
    }

    #[test]
    fn fixed_nuisance_innovation_is_identical_across_signal_strengths() {
        let report =
            run_factorial_dose_response_sweep(&clean(), &signal(), &nuisance(), &config()).unwrap();
        for nuisance_index in 0..4 {
            let fingerprints = report
                .points
                .chunks_exact(4)
                .map(|row| row[nuisance_index].cells[2].fnv1a64_fingerprint.as_str())
                .collect::<Vec<_>>();
            assert!(fingerprints.windows(2).all(|pair| pair[0] == pair[1]));
        }
        assert_eq!(report.raw_signal_fingerprint, matrix_fingerprint(&signal()));
        assert_eq!(
            report.raw_nuisance_fingerprint,
            matrix_fingerprint(&nuisance())
        );
    }

    #[test]
    fn zero_signal_and_zero_nuisance_are_exact_factorial_controls() {
        let report =
            run_factorial_dose_response_sweep(&clean(), &signal(), &nuisance(), &config()).unwrap();
        for row in report.points.chunks_exact(4) {
            let zero_nuisance = &row[0];
            assert_eq!(zero_nuisance.nuisance_component_variance, 0.0);
            assert_eq!(
                zero_nuisance.cells[0].fnv1a64_fingerprint,
                zero_nuisance.cells[2].fnv1a64_fingerprint
            );
            assert_eq!(
                zero_nuisance.cells[1].fnv1a64_fingerprint,
                zero_nuisance.cells[3].fnv1a64_fingerprint
            );
        }
        for point in &report.points[..4] {
            assert_eq!(point.signal_component_rms, 0.0);
            assert_eq!(
                point.cells[0].fnv1a64_fingerprint,
                point.cells[1].fnv1a64_fingerprint
            );
            assert_eq!(
                point.cells[2].fnv1a64_fingerprint,
                point.cells[3].fnv1a64_fingerprint
            );
        }
    }

    #[test]
    fn repeated_sweep_invocation_has_identical_report() {
        let clean = clean();
        let signal = signal();
        let nuisance = nuisance();
        let config = config();
        assert_eq!(
            run_factorial_dose_response_sweep(&clean, &signal, &nuisance, &config).unwrap(),
            run_factorial_dose_response_sweep(&clean, &signal, &nuisance, &config).unwrap()
        );
    }

    #[test]
    fn every_dose_point_contains_all_estimator_response_scores() {
        let report =
            run_factorial_dose_response_sweep(&clean(), &signal(), &nuisance(), &config()).unwrap();
        for point in &report.points {
            assert_eq!(
                point
                    .estimators
                    .iter()
                    .map(|item| item.estimator_id)
                    .collect::<Vec<_>>(),
                vec![
                    "absolute_weighted_jaccard",
                    "signed_shift_weighted_jaccard",
                    "normalized_mean_absolute_edge_error",
                ]
            );
            assert!(point.estimators.iter().all(|item| {
                (0.0..=1.0).contains(&item.clean_to_signal_similarity)
                    && (0.0..=1.0).contains(&item.clean_to_nuisance_similarity)
                    && (0.0..=1.0).contains(&item.signal_to_combined_similarity)
            }));
            assert!(point.connectivity_direction.clean_signal_change_norm >= 0.0);
            assert!(point.connectivity_direction.combined_signal_change_norm >= 0.0);
        }
    }

    #[test]
    fn zero_signal_axis_produces_undefined_estimator_retention() {
        let report =
            run_factorial_dose_response_sweep(&clean(), &signal(), &nuisance(), &config()).unwrap();
        for point in &report.points[..4] {
            assert!(point
                .estimators
                .iter()
                .all(|item| item.signal_retention_fraction.is_none()));
            assert!(point
                .connectivity_direction
                .signal_projection_gain
                .is_none());
        }
    }

    #[test]
    fn invalid_or_unsorted_grids_fail_closed() {
        let clean = clean();
        let signal = signal();
        let nuisance = nuisance();
        for scales in [
            vec![],
            vec![0.0, 1.0, 0.5],
            vec![-0.1, 0.5],
            vec![0.0, f64::NAN],
        ] {
            let mut config = config();
            config.signal_scales = scales;
            assert!(
                run_factorial_dose_response_sweep(&clean, &signal, &nuisance, &config).is_err()
            );
        }
        for fractions in [
            vec![],
            vec![0.0, 0.1, 0.1],
            vec![-0.01, 0.1],
            vec![0.0, 1.0],
            vec![0.0, f64::NAN],
        ] {
            let mut config = config();
            config.nuisance_component_variance_fractions = fractions;
            assert!(
                run_factorial_dose_response_sweep(&clean, &signal, &nuisance, &config).is_err()
            );
        }
    }

    #[test]
    fn shape_and_non_finite_inputs_fail_closed() {
        let clean = clean();
        let mut invalid_signal = signal();
        invalid_signal[0][0] = f64::NAN;
        assert!(matches!(
            run_factorial_dose_response_sweep(&clean, &invalid_signal, &nuisance(), &config()),
            Err(FactorialDoseResponseError::NonFiniteInput("raw signal"))
        ));
        assert!(matches!(
            run_factorial_dose_response_sweep(&clean, &signal(), &nuisance()[..47], &config()),
            Err(FactorialDoseResponseError::DimensionMismatch(
                "raw nuisance"
            ))
        ));
    }

    #[test]
    fn zero_raw_nuisance_cannot_support_a_positive_fraction() {
        let zero = vec![vec![0.0; 4]; 48];
        assert!(matches!(
            run_factorial_dose_response_sweep(&clean(), &signal(), &zero, &config()),
            Err(FactorialDoseResponseError::Calibration(
                SpatialVarianceCalibrationError::ZeroNuisanceVariance
            ))
        ));
    }

    #[test]
    fn signal_scale_overflow_fails_closed() {
        let mut config = config();
        config.signal_scales = vec![0.0, 1e308];
        let huge = vec![vec![1e100; 4]; 48];
        assert!(matches!(
            run_factorial_dose_response_sweep(&clean(), &huge, &nuisance(), &config),
            Err(FactorialDoseResponseError::NonFiniteScaledSignal)
        ));
    }
}
