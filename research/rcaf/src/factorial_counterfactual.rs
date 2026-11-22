//! Four-cell matched counterfactual engine for RCAF.
//!
//! This module is nuisance-family agnostic. Every nuisance generator supplies one already
//! generated time x ROI field. The same immutable field is applied to nuisance-only and
//! signal-plus-nuisance cells, preventing accidental redraws between matched counterfactuals.

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct FactorialExperimentMetadata {
    pub experiment_id: String,
    pub signal_id: String,
    pub nuisance_family_id: String,
    pub nuisance_instance_id: String,
    pub target_component_variance_fraction: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FactorialCellSummary {
    pub cell_id: &'static str,
    pub rms: f64,
    pub max_absolute_value: f64,
    /// FNV-1a-64 diagnostic fingerprint, not a cryptographic digest.
    pub fnv1a64_fingerprint: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FactorialInvariantReport {
    pub matched_nuisance_delta_max_abs: f64,
    pub matched_signal_delta_max_abs: f64,
    pub additive_interaction_max_abs: f64,
    pub numerical_tolerance: f64,
    pub all_invariants_within_tolerance: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FactorialCounterfactualReport {
    pub schema_version: &'static str,
    pub experiment_id: String,
    pub signal_id: String,
    pub nuisance_family_id: String,
    pub nuisance_instance_id: String,
    pub time_points: usize,
    pub roi_count: usize,
    pub compared_values: usize,
    pub target_component_variance_fraction: Option<f64>,
    pub realized_component_variance_fraction: f64,
    pub clean_component_variance: f64,
    pub nuisance_component_variance: f64,
    pub clean_nuisance_covariance: f64,
    pub combined_observed_variance: f64,
    pub signal_component_rms: f64,
    pub nuisance_component_rms: f64,
    pub cells: Vec<FactorialCellSummary>,
    pub invariants: FactorialInvariantReport,
}

impl FactorialCounterfactualReport {
    pub fn to_json_pretty(&self) -> String {
        let cells = self
            .cells
            .iter()
            .map(|cell| {
                format!(
                    concat!(
                        "    {{\n",
                        "      \"cell_id\": \"{}\",\n",
                        "      \"rms\": {:.12},\n",
                        "      \"max_absolute_value\": {:.12},\n",
                        "      \"fnv1a64_fingerprint\": \"{}\"\n",
                        "    }}"
                    ),
                    cell.cell_id, cell.rms, cell.max_absolute_value, cell.fnv1a64_fingerprint,
                )
            })
            .collect::<Vec<_>>()
            .join(",\n");
        let target_fraction = self
            .target_component_variance_fraction
            .map(|value| format!("{value:.12}"))
            .unwrap_or_else(|| "null".to_string());
        format!(
            concat!(
                "{{\n",
                "  \"schema_version\": \"{}\",\n",
                "  \"experiment_id\": \"{}\",\n",
                "  \"signal_id\": \"{}\",\n",
                "  \"nuisance_family_id\": \"{}\",\n",
                "  \"nuisance_instance_id\": \"{}\",\n",
                "  \"time_points\": {},\n",
                "  \"roi_count\": {},\n",
                "  \"compared_values\": {},\n",
                "  \"target_component_variance_fraction\": {},\n",
                "  \"realized_component_variance_fraction\": {:.12},\n",
                "  \"clean_component_variance\": {:.12},\n",
                "  \"nuisance_component_variance\": {:.12},\n",
                "  \"clean_nuisance_covariance\": {:.12},\n",
                "  \"combined_observed_variance\": {:.12},\n",
                "  \"signal_component_rms\": {:.12},\n",
                "  \"nuisance_component_rms\": {:.12},\n",
                "  \"cells\": [\n{}\n  ],\n",
                "  \"invariants\": {{\n",
                "    \"matched_nuisance_delta_max_abs\": {:.12},\n",
                "    \"matched_signal_delta_max_abs\": {:.12},\n",
                "    \"additive_interaction_max_abs\": {:.12},\n",
                "    \"numerical_tolerance\": {:.12},\n",
                "    \"all_invariants_within_tolerance\": {}\n",
                "  }}\n",
                "}}\n"
            ),
            self.schema_version,
            self.experiment_id,
            self.signal_id,
            self.nuisance_family_id,
            self.nuisance_instance_id,
            self.time_points,
            self.roi_count,
            self.compared_values,
            target_fraction,
            self.realized_component_variance_fraction,
            self.clean_component_variance,
            self.nuisance_component_variance,
            self.clean_nuisance_covariance,
            self.combined_observed_variance,
            self.signal_component_rms,
            self.nuisance_component_rms,
            cells,
            self.invariants.matched_nuisance_delta_max_abs,
            self.invariants.matched_signal_delta_max_abs,
            self.invariants.additive_interaction_max_abs,
            self.invariants.numerical_tolerance,
            self.invariants.all_invariants_within_tolerance,
        )
    }
}

/// Full output matrices for the four counterfactual cells, all in time x ROI layout.
#[derive(Debug, Clone, PartialEq)]
pub struct FactorialCounterfactualExperiment {
    pub clean_null: Vec<Vec<f64>>,
    pub signal_only: Vec<Vec<f64>>,
    pub nuisance_only: Vec<Vec<f64>>,
    pub signal_plus_nuisance: Vec<Vec<f64>>,
    pub report: FactorialCounterfactualReport,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FactorialExperimentError {
    EmptyIdentifier(&'static str),
    InvalidIdentifier(&'static str),
    InvalidTargetVarianceFraction,
    EmptyMatrix,
    InsufficientMatrixShape,
    RaggedMatrix(&'static str),
    DimensionMismatch(&'static str),
    NonFiniteInput(&'static str),
    NonFiniteOutput(&'static str),
    NonFiniteMoment,
}

impl fmt::Display for FactorialExperimentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyIdentifier(field) => write!(f, "{field} must not be empty"),
            Self::InvalidIdentifier(field) => write!(
                f,
                "{field} must use only ASCII letters, digits, '.', '_', ':' or '-'"
            ),
            Self::InvalidTargetVarianceFraction => write!(
                f,
                "target component-variance fraction must be finite and in [0, 1)"
            ),
            Self::EmptyMatrix => write!(f, "counterfactual matrices must be non-empty"),
            Self::InsufficientMatrixShape => write!(
                f,
                "counterfactual matrices require at least three time points and two ROIs"
            ),
            Self::RaggedMatrix(name) => write!(f, "{name} matrix must be rectangular"),
            Self::DimensionMismatch(reason) => write!(f, "factorial dimension mismatch: {reason}"),
            Self::NonFiniteInput(name) => write!(f, "{name} contains non-finite values"),
            Self::NonFiniteOutput(name) => {
                write!(f, "factorial construction produced non-finite {name}")
            }
            Self::NonFiniteMoment => write!(f, "factorial pooled moments are non-finite"),
        }
    }
}

impl std::error::Error for FactorialExperimentError {}

/// Build clean-null, signal-only, nuisance-only and signal-plus-nuisance cells.
///
/// The nuisance matrix is borrowed once and applied unchanged to the two nuisance-containing
/// cells. This function never calls a nuisance generator or redraws random innovations.
pub fn run_factorial_counterfactual_experiment(
    clean_observations: &[Vec<f64>],
    signal_component: &[Vec<f64>],
    nuisance_component: &[Vec<f64>],
    metadata: &FactorialExperimentMetadata,
) -> Result<FactorialCounterfactualExperiment, FactorialExperimentError> {
    validate_metadata(metadata)?;
    let (time_points, roi_count) = validate_matrix(clean_observations, "clean observations")?;
    validate_matching_matrix(signal_component, time_points, roi_count, "signal component")?;
    validate_matching_matrix(
        nuisance_component,
        time_points,
        roi_count,
        "nuisance component",
    )?;

    let mut signal_only = Vec::with_capacity(time_points);
    let mut nuisance_only = Vec::with_capacity(time_points);
    let mut signal_plus_nuisance = Vec::with_capacity(time_points);
    for time in 0..time_points {
        let mut signal_row = Vec::with_capacity(roi_count);
        let mut nuisance_row = Vec::with_capacity(roi_count);
        let mut combined_row = Vec::with_capacity(roi_count);
        for roi in 0..roi_count {
            let clean = clean_observations[time][roi];
            let signal = signal_component[time][roi];
            let nuisance = nuisance_component[time][roi];
            let signal_value = clean + signal;
            let nuisance_value = clean + nuisance;
            let combined_value = clean + signal + nuisance;
            if !signal_value.is_finite() {
                return Err(FactorialExperimentError::NonFiniteOutput(
                    "signal-only cell",
                ));
            }
            if !nuisance_value.is_finite() {
                return Err(FactorialExperimentError::NonFiniteOutput(
                    "nuisance-only cell",
                ));
            }
            if !combined_value.is_finite() {
                return Err(FactorialExperimentError::NonFiniteOutput(
                    "signal-plus-nuisance cell",
                ));
            }
            signal_row.push(signal_value);
            nuisance_row.push(nuisance_value);
            combined_row.push(combined_value);
        }
        signal_only.push(signal_row);
        nuisance_only.push(nuisance_row);
        signal_plus_nuisance.push(combined_row);
    }

    let clean_null = clean_observations.to_vec();
    let (clean_variance, nuisance_variance, clean_nuisance_covariance) =
        pooled_within_roi_moments(clean_observations, nuisance_component);
    if [clean_variance, nuisance_variance, clean_nuisance_covariance]
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(FactorialExperimentError::NonFiniteMoment);
    }
    let component_denominator = clean_variance + nuisance_variance;
    let realized_component_variance_fraction = if component_denominator > 0.0 {
        nuisance_variance / component_denominator
    } else {
        0.0
    };
    let combined_observed_variance =
        clean_variance + nuisance_variance + 2.0 * clean_nuisance_covariance;
    if !realized_component_variance_fraction.is_finite() || !combined_observed_variance.is_finite()
    {
        return Err(FactorialExperimentError::NonFiniteMoment);
    }

    let scale = [
        max_abs(&clean_null),
        max_abs(&signal_only),
        max_abs(&nuisance_only),
        max_abs(&signal_plus_nuisance),
    ]
    .into_iter()
    .fold(1.0f64, f64::max);
    let numerical_tolerance = 1e-12 * scale;
    let matched_nuisance_delta_max_abs = max_matched_delta(
        &nuisance_only,
        &clean_null,
        &signal_plus_nuisance,
        &signal_only,
    );
    let matched_signal_delta_max_abs = max_matched_delta(
        &signal_only,
        &clean_null,
        &signal_plus_nuisance,
        &nuisance_only,
    );
    let additive_interaction_max_abs = max_additive_interaction(
        &clean_null,
        &signal_only,
        &nuisance_only,
        &signal_plus_nuisance,
    );
    let invariants = FactorialInvariantReport {
        matched_nuisance_delta_max_abs,
        matched_signal_delta_max_abs,
        additive_interaction_max_abs,
        numerical_tolerance,
        all_invariants_within_tolerance: matched_nuisance_delta_max_abs <= numerical_tolerance
            && matched_signal_delta_max_abs <= numerical_tolerance
            && additive_interaction_max_abs <= numerical_tolerance,
    };

    let cells = vec![
        summarize_cell("clean_null", &clean_null),
        summarize_cell("signal_only", &signal_only),
        summarize_cell("nuisance_only", &nuisance_only),
        summarize_cell("signal_plus_nuisance", &signal_plus_nuisance),
    ];
    let signal_component_rms = rms(signal_component);
    let nuisance_component_rms = rms(nuisance_component);
    if !signal_component_rms.is_finite()
        || !nuisance_component_rms.is_finite()
        || cells
            .iter()
            .any(|cell| !cell.rms.is_finite() || !cell.max_absolute_value.is_finite())
    {
        return Err(FactorialExperimentError::NonFiniteMoment);
    }
    let report = FactorialCounterfactualReport {
        schema_version: "rcaf.factorial-counterfactual/0.1.0",
        experiment_id: metadata.experiment_id.clone(),
        signal_id: metadata.signal_id.clone(),
        nuisance_family_id: metadata.nuisance_family_id.clone(),
        nuisance_instance_id: metadata.nuisance_instance_id.clone(),
        time_points,
        roi_count,
        compared_values: time_points * roi_count,
        target_component_variance_fraction: metadata.target_component_variance_fraction,
        realized_component_variance_fraction,
        clean_component_variance: clean_variance,
        nuisance_component_variance: nuisance_variance,
        clean_nuisance_covariance,
        combined_observed_variance: combined_observed_variance.max(0.0),
        signal_component_rms,
        nuisance_component_rms,
        cells,
        invariants,
    };

    Ok(FactorialCounterfactualExperiment {
        clean_null,
        signal_only,
        nuisance_only,
        signal_plus_nuisance,
        report,
    })
}

fn validate_metadata(
    metadata: &FactorialExperimentMetadata,
) -> Result<(), FactorialExperimentError> {
    for (name, value) in [
        ("experiment_id", metadata.experiment_id.as_str()),
        ("signal_id", metadata.signal_id.as_str()),
        ("nuisance_family_id", metadata.nuisance_family_id.as_str()),
        (
            "nuisance_instance_id",
            metadata.nuisance_instance_id.as_str(),
        ),
    ] {
        if value.trim().is_empty() {
            return Err(FactorialExperimentError::EmptyIdentifier(name));
        }
        if !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'-'))
        {
            return Err(FactorialExperimentError::InvalidIdentifier(name));
        }
    }
    if let Some(fraction) = metadata.target_component_variance_fraction {
        if !fraction.is_finite() || !(0.0..1.0).contains(&fraction) {
            return Err(FactorialExperimentError::InvalidTargetVarianceFraction);
        }
    }
    Ok(())
}

fn validate_matrix(
    matrix: &[Vec<f64>],
    name: &'static str,
) -> Result<(usize, usize), FactorialExperimentError> {
    if matrix.is_empty() || matrix[0].is_empty() {
        return Err(FactorialExperimentError::EmptyMatrix);
    }
    let width = matrix[0].len();
    if matrix.iter().any(|row| row.len() != width) {
        return Err(FactorialExperimentError::RaggedMatrix(name));
    }
    if matrix.len() < 3 || width < 2 {
        return Err(FactorialExperimentError::InsufficientMatrixShape);
    }
    if matrix.iter().flatten().any(|value| !value.is_finite()) {
        return Err(FactorialExperimentError::NonFiniteInput(name));
    }
    Ok((matrix.len(), width))
}

fn validate_matching_matrix(
    matrix: &[Vec<f64>],
    time_points: usize,
    roi_count: usize,
    name: &'static str,
) -> Result<(), FactorialExperimentError> {
    let (actual_time_points, actual_roi_count) = validate_matrix(matrix, name)?;
    if actual_time_points != time_points || actual_roi_count != roi_count {
        return Err(FactorialExperimentError::DimensionMismatch(name));
    }
    Ok(())
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
            let centered_clean = clean[time][roi] - clean_mean;
            let centered_nuisance = nuisance[time][roi] - nuisance_mean;
            clean_squares += centered_clean * centered_clean;
            nuisance_squares += centered_nuisance * centered_nuisance;
            cross_products += centered_clean * centered_nuisance;
        }
    }
    (
        clean_squares / divisor,
        nuisance_squares / divisor,
        cross_products / divisor,
    )
}

fn rms(matrix: &[Vec<f64>]) -> f64 {
    let count = matrix.iter().map(Vec::len).sum::<usize>();
    let squares = matrix
        .iter()
        .flatten()
        .map(|value| value * value)
        .sum::<f64>();
    (squares / count as f64).sqrt()
}

fn max_abs(matrix: &[Vec<f64>]) -> f64 {
    matrix
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(0.0, f64::max)
}

fn max_matched_delta(
    first_cell: &[Vec<f64>],
    first_base: &[Vec<f64>],
    second_cell: &[Vec<f64>],
    second_base: &[Vec<f64>],
) -> f64 {
    first_cell
        .iter()
        .zip(first_base.iter())
        .zip(second_cell.iter().zip(second_base.iter()))
        .flat_map(|((first, first_base), (second, second_base))| {
            first
                .iter()
                .zip(first_base.iter())
                .zip(second.iter().zip(second_base.iter()))
                .map(|((first, first_base), (second, second_base))| {
                    ((first - first_base) - (second - second_base)).abs()
                })
        })
        .fold(0.0, f64::max)
}

fn max_additive_interaction(
    clean: &[Vec<f64>],
    signal_only: &[Vec<f64>],
    nuisance_only: &[Vec<f64>],
    combined: &[Vec<f64>],
) -> f64 {
    clean
        .iter()
        .zip(signal_only.iter())
        .zip(nuisance_only.iter().zip(combined.iter()))
        .flat_map(|((clean, signal), (nuisance, combined))| {
            clean
                .iter()
                .zip(signal.iter())
                .zip(nuisance.iter().zip(combined.iter()))
                .map(|((clean, signal), (nuisance, combined))| {
                    (combined - signal - nuisance + clean).abs()
                })
        })
        .fold(0.0, f64::max)
}

fn summarize_cell(cell_id: &'static str, matrix: &[Vec<f64>]) -> FactorialCellSummary {
    FactorialCellSummary {
        cell_id,
        rms: rms(matrix),
        max_absolute_value: max_abs(matrix),
        fnv1a64_fingerprint: matrix_fingerprint(matrix),
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata() -> FactorialExperimentMetadata {
        FactorialExperimentMetadata {
            experiment_id: "fixture-01".to_string(),
            signal_id: "target-network-up".to_string(),
            nuisance_family_id: "graph_smooth_low_rank".to_string(),
            nuisance_instance_id: "field-seed-42".to_string(),
            target_component_variance_fraction: Some(0.10),
        }
    }

    fn clean() -> Vec<Vec<f64>> {
        (0..32)
            .map(|time| {
                (0..4)
                    .map(|roi| ((time as f64 + 1.0) * (roi as f64 + 1.0) * 0.07).sin())
                    .collect()
            })
            .collect()
    }

    fn signal() -> Vec<Vec<f64>> {
        (0..32)
            .map(|time| {
                (0..4)
                    .map(|roi| 0.15 * ((time as f64 + 2.0) * (roi as f64 + 1.0) * 0.13).sin())
                    .collect()
            })
            .collect()
    }

    fn nuisance() -> Vec<Vec<f64>> {
        (0..32)
            .map(|time| {
                (0..4)
                    .map(|roi| 0.08 * ((time as f64 + 3.0) * (roi as f64 + 1.0) * 0.11).cos())
                    .collect()
            })
            .collect()
    }

    #[test]
    fn builds_all_four_cells_and_reuses_the_same_nuisance_innovation() {
        let clean = clean();
        let signal = signal();
        let nuisance = nuisance();
        let experiment =
            run_factorial_counterfactual_experiment(&clean, &signal, &nuisance, &metadata())
                .unwrap();
        assert_eq!(experiment.clean_null, clean);
        assert_eq!(experiment.signal_only.len(), clean.len());
        assert_eq!(experiment.nuisance_only.len(), clean.len());
        assert_eq!(experiment.signal_plus_nuisance.len(), clean.len());
        assert!(experiment.report.invariants.all_invariants_within_tolerance);
        assert_eq!(experiment.report.cells.len(), 4);
        assert_eq!(experiment.report.cells[0].cell_id, "clean_null");
        assert_eq!(experiment.report.cells[1].cell_id, "signal_only");
        assert_eq!(experiment.report.cells[2].cell_id, "nuisance_only");
        assert_eq!(experiment.report.cells[3].cell_id, "signal_plus_nuisance");
    }

    #[test]
    fn repeated_invocation_has_identical_fingerprints_and_report() {
        let clean = clean();
        let signal = signal();
        let nuisance = nuisance();
        let first =
            run_factorial_counterfactual_experiment(&clean, &signal, &nuisance, &metadata())
                .unwrap();
        let second =
            run_factorial_counterfactual_experiment(&clean, &signal, &nuisance, &metadata())
                .unwrap();
        assert_eq!(first.report, second.report);
        assert_eq!(first.signal_plus_nuisance, second.signal_plus_nuisance);
    }

    #[test]
    fn component_variance_and_signed_covariance_are_reported() {
        let clean = clean();
        let nuisance = nuisance();
        let signal = signal();
        let experiment =
            run_factorial_counterfactual_experiment(&clean, &signal, &nuisance, &metadata())
                .unwrap();
        assert!(experiment.report.clean_component_variance > 0.0);
        assert!(experiment.report.nuisance_component_variance > 0.0);
        let aligned = clean
            .iter()
            .map(|row| row.iter().map(|value| *value * 0.1).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let aligned_result =
            run_factorial_counterfactual_experiment(&clean, &signal, &aligned, &metadata())
                .unwrap();
        assert!(aligned_result.report.clean_nuisance_covariance > 0.0);
        let anti_aligned = aligned
            .iter()
            .map(|row| row.iter().map(|value| -*value).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let anti_result =
            run_factorial_counterfactual_experiment(&clean, &signal, &anti_aligned, &metadata())
                .unwrap();
        assert!(anti_result.report.clean_nuisance_covariance < 0.0);
    }

    #[test]
    fn optional_calibrated_fraction_is_recorded_without_being_assumed() {
        let mut metadata = metadata();
        metadata.target_component_variance_fraction = None;
        let experiment =
            run_factorial_counterfactual_experiment(&clean(), &signal(), &nuisance(), &metadata)
                .unwrap();
        assert_eq!(experiment.report.target_component_variance_fraction, None);
        assert!(experiment.report.realized_component_variance_fraction >= 0.0);
    }

    #[test]
    fn insufficient_time_or_roi_dimensions_fail_closed() {
        let short_time = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let full = vec![vec![0.0, 1.0]; 3];
        assert!(matches!(
            run_factorial_counterfactual_experiment(
                &short_time,
                &short_time,
                &short_time,
                &metadata()
            ),
            Err(FactorialExperimentError::InsufficientMatrixShape)
        ));
        assert!(matches!(
            run_factorial_counterfactual_experiment(
                &full,
                &full,
                &full.iter().map(|_| vec![0.0]).collect::<Vec<_>>(),
                &metadata()
            ),
            Err(FactorialExperimentError::InsufficientMatrixShape)
        ));
    }

    #[test]
    fn mismatched_dimensions_and_ragged_matrices_fail_closed() {
        let clean = clean();
        let signal = signal();
        let nuisance = nuisance();
        assert!(matches!(
            run_factorial_counterfactual_experiment(&clean, &signal[..31], &nuisance, &metadata()),
            Err(FactorialExperimentError::DimensionMismatch(_))
        ));
        let ragged = vec![vec![0.0, 1.0], vec![0.0], vec![1.0, 0.0]];
        assert!(matches!(
            run_factorial_counterfactual_experiment(&clean, &signal, &ragged, &metadata()),
            Err(FactorialExperimentError::RaggedMatrix(_))
        ));
    }

    #[test]
    fn non_finite_inputs_and_overflowing_outputs_fail_closed() {
        let mut invalid = clean();
        invalid[0][0] = f64::NAN;
        assert!(matches!(
            run_factorial_counterfactual_experiment(&invalid, &signal(), &nuisance(), &metadata()),
            Err(FactorialExperimentError::NonFiniteInput(_))
        ));
        let mut huge_signal = signal();
        huge_signal[0][0] = f64::MAX;
        let mut huge_nuisance = nuisance();
        huge_nuisance[0][0] = f64::MAX;
        assert!(matches!(
            run_factorial_counterfactual_experiment(
                &clean(),
                &huge_signal,
                &huge_nuisance,
                &metadata()
            ),
            Err(FactorialExperimentError::NonFiniteOutput(_))
        ));
    }

    #[test]
    fn finite_inputs_with_overflowing_cell_rms_fail_closed() {
        let clean = clean();
        let mut huge_signal = signal();
        for row in &mut huge_signal {
            for value in row {
                *value = 1e200;
            }
        }
        let result =
            run_factorial_counterfactual_experiment(&clean, &huge_signal, &nuisance(), &metadata());
        assert!(matches!(
            result,
            Err(FactorialExperimentError::NonFiniteMoment)
        ));
    }

    #[test]
    fn joint_time_and_roi_permutation_preserves_cell_and_moment_contracts() {
        let clean = clean();
        let signal = signal();
        let nuisance = nuisance();
        let original =
            run_factorial_counterfactual_experiment(&clean, &signal, &nuisance, &metadata())
                .unwrap();
        let time_permutation = (0..clean.len())
            .map(|index| (index * 7) % clean.len())
            .collect::<Vec<_>>();
        let roi_permutation = [2usize, 0, 3, 1];
        let permute = |matrix: &[Vec<f64>]| {
            time_permutation
                .iter()
                .map(|&time| {
                    roi_permutation
                        .iter()
                        .map(|&roi| matrix[time][roi])
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        let permuted = run_factorial_counterfactual_experiment(
            &permute(&clean),
            &permute(&signal),
            &permute(&nuisance),
            &metadata(),
        )
        .unwrap();
        assert_eq!(permuted.clean_null, permute(&original.clean_null));
        assert_eq!(permuted.signal_only, permute(&original.signal_only));
        assert_eq!(permuted.nuisance_only, permute(&original.nuisance_only));
        assert_eq!(
            permuted.signal_plus_nuisance,
            permute(&original.signal_plus_nuisance)
        );
        assert!(
            (permuted.report.clean_component_variance - original.report.clean_component_variance)
                .abs()
                < 1e-12
        );
        assert!(
            (permuted.report.nuisance_component_variance
                - original.report.nuisance_component_variance)
                .abs()
                < 1e-12
        );
        assert!(
            (permuted.report.clean_nuisance_covariance - original.report.clean_nuisance_covariance)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn global_amplitude_scaling_scales_rms_linearly_and_variance_quadratically() {
        let clean = clean();
        let signal = signal();
        let nuisance = nuisance();
        let original =
            run_factorial_counterfactual_experiment(&clean, &signal, &nuisance, &metadata())
                .unwrap();
        let factor = 2.5;
        let scale = |matrix: &[Vec<f64>]| {
            matrix
                .iter()
                .map(|row| row.iter().map(|value| value * factor).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        };
        let scaled = run_factorial_counterfactual_experiment(
            &scale(&clean),
            &scale(&signal),
            &scale(&nuisance),
            &metadata(),
        )
        .unwrap();
        assert!((scaled.report.cells[0].rms - factor * original.report.cells[0].rms).abs() < 1e-12);
        assert!(
            (scaled.report.clean_component_variance
                - factor * factor * original.report.clean_component_variance)
                .abs()
                < 1e-10
        );
        assert!(
            (scaled.report.nuisance_component_variance
                - factor * factor * original.report.nuisance_component_variance)
                .abs()
                < 1e-10
        );
        assert!(
            (scaled.report.clean_nuisance_covariance
                - factor * factor * original.report.clean_nuisance_covariance)
                .abs()
                < 1e-10
        );
    }

    #[test]
    fn zero_nuisance_collapses_nuisance_cells_to_clean_and_signal_cells() {
        let clean = clean();
        let signal = signal();
        let zero = vec![vec![0.0; 4]; 32];
        let experiment =
            run_factorial_counterfactual_experiment(&clean, &signal, &zero, &metadata()).unwrap();
        assert_eq!(experiment.nuisance_only, experiment.clean_null);
        assert_eq!(experiment.signal_plus_nuisance, experiment.signal_only);
        assert_eq!(experiment.report.nuisance_component_variance, 0.0);
        assert_eq!(experiment.report.clean_nuisance_covariance, 0.0);
    }

    #[test]
    fn zero_signal_collapses_signal_cells_to_clean_and_nuisance_cells() {
        let clean = clean();
        let zero = vec![vec![0.0; 4]; 32];
        let nuisance = nuisance();
        let experiment =
            run_factorial_counterfactual_experiment(&clean, &zero, &nuisance, &metadata()).unwrap();
        assert_eq!(experiment.signal_only, experiment.clean_null);
        assert_eq!(experiment.signal_plus_nuisance, experiment.nuisance_only);
        assert_eq!(experiment.report.signal_component_rms, 0.0);
    }

    #[test]
    fn invalid_provenance_and_target_fraction_fail_closed() {
        let mut invalid = metadata();
        invalid.nuisance_instance_id = "field id with spaces".to_string();
        assert!(matches!(
            run_factorial_counterfactual_experiment(&clean(), &signal(), &nuisance(), &invalid),
            Err(FactorialExperimentError::InvalidIdentifier(_))
        ));
        let mut invalid = metadata();
        invalid.target_component_variance_fraction = Some(1.0);
        assert!(matches!(
            run_factorial_counterfactual_experiment(&clean(), &signal(), &nuisance(), &invalid),
            Err(FactorialExperimentError::InvalidTargetVarianceFraction)
        ));
    }

    #[test]
    fn cell_fingerprints_change_when_nuisance_changes() {
        let clean = clean();
        let signal = signal();
        let nuisance = nuisance();
        let original =
            run_factorial_counterfactual_experiment(&clean, &signal, &nuisance, &metadata())
                .unwrap();
        let mut changed_nuisance = nuisance.clone();
        changed_nuisance[0][0] += 0.01;
        let changed = run_factorial_counterfactual_experiment(
            &clean,
            &signal,
            &changed_nuisance,
            &metadata(),
        )
        .unwrap();
        assert_ne!(
            original.report.cells[2].fnv1a64_fingerprint,
            changed.report.cells[2].fnv1a64_fingerprint
        );
        assert_ne!(
            original.report.cells[3].fnv1a64_fingerprint,
            changed.report.cells[3].fnv1a64_fingerprint
        );
    }
}
