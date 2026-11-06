// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Joint covariance diagnostics for SPECTRA's five-metric score vectors.
//!
//! This module computes descriptive sample covariance geometry from five
//! metrics observed on the same replication. It does not estimate a confidence
//! ellipsoid, establish conditional sub-Gaussianity, prove a fixed conditional
//! mean, allocate alpha, or change an allocator/stopping decision.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const SCORE_METRIC_DIMENSION: usize = 5;
pub const SCORE_COVARIANCE_DIAGNOSTIC_METHOD_ID: &str =
    "spectra-five-metric-joint-covariance-diagnostic-v1";

pub const SCORE_METRIC_NAMES: [&str; SCORE_METRIC_DIMENSION] = [
    "target_information",
    "model_discrimination_information",
    "representativeness",
    "target_utility",
    "participant_burden_seconds",
];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetricRange {
    pub lower: f64,
    pub upper: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectionalProbe {
    pub direction_id: u16,
    /// Frozen coefficients in the five-metric coordinate system.
    pub weights: [f64; SCORE_METRIC_DIMENSION],
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScoreCovarianceContract {
    pub policy_version: String,
    pub source_digest: String,
    pub action_ids: Vec<u32>,
    pub scenario_ids: Vec<u16>,
    pub metric_ranges: [MetricRange; SCORE_METRIC_DIMENSION],
    pub directional_probes: Vec<DirectionalProbe>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoreVectorObservation {
    pub action_id: u32,
    pub scenario_id: u16,
    pub replication_index: u32,
    /// Five raw metric values from the same experiment/simulation replicate.
    pub metrics: [f64; SCORE_METRIC_DIMENSION],
}

#[derive(Debug, Clone, PartialEq)]
pub struct DirectionalVarianceDiagnostic {
    pub direction_id: u16,
    pub covariance_aware_variance: f64,
    /// Sum of diagonal contributions; equivalent to assuming zero
    /// cross-metric covariance for this frozen direction.
    pub diagonal_only_variance: f64,
    /// covariance_aware_variance / diagonal_only_variance when denominator > 0.
    pub covariance_to_diagonal_ratio: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScoreCellCovarianceDiagnostic {
    pub action_id: u32,
    pub scenario_id: u16,
    pub sample_count: usize,
    pub mean_vector: [f64; SCORE_METRIC_DIMENSION],
    pub sample_covariance: [[f64; SCORE_METRIC_DIMENSION]; SCORE_METRIC_DIMENSION],
    pub correlation_matrix: [[Option<f64>; SCORE_METRIC_DIMENSION]; SCORE_METRIC_DIMENSION],
    /// Ascending eigenvalues of the symmetric sample covariance matrix.
    pub covariance_eigenvalues: [f64; SCORE_METRIC_DIMENSION],
    pub effective_rank: f64,
    /// None means singular or numerically rank-deficient covariance.
    pub covariance_condition_number: Option<f64>,
    /// Frozen directional projections; these remain descriptive diagnostics.
    pub directional_variances: Vec<DirectionalVarianceDiagnostic>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScoreCovarianceDiagnosticReceipt {
    pub method_id: &'static str,
    pub policy_version: String,
    pub source_digest: String,
    pub action_ids: Vec<u32>,
    pub scenario_ids: Vec<u16>,
    pub metric_ranges: [MetricRange; SCORE_METRIC_DIMENSION],
    pub directional_probes: Vec<DirectionalProbe>,
    pub cells: Vec<ScoreCellCovarianceDiagnostic>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreCovarianceError {
    MissingPolicyVersion,
    MissingSourceDigest,
    EmptyActionSet,
    EmptyScenarioSet,
    DuplicateActionId,
    DuplicateScenarioId,
    DuplicateDirectionalProbe,
    InvalidMetricRange,
    InvalidDirectionalProbe,
    EmptyObservationSet,
    UnknownActionId,
    UnknownScenarioId,
    InvalidMetricObservation,
    DuplicateObservation,
    IncompleteActionScenarioGrid,
    NonContiguousReplicationIndices,
    InsufficientReplications,
    NumericFailure,
    ReplayMismatch,
}

impl fmt::Display for ScoreCovarianceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::MissingPolicyVersion => "score-covariance policy version must not be empty",
            Self::MissingSourceDigest => "score-covariance source digest must not be empty",
            Self::EmptyActionSet => "score covariance requires a nonempty frozen action set",
            Self::EmptyScenarioSet => "score covariance requires a nonempty frozen scenario set",
            Self::DuplicateActionId => "frozen action IDs must be unique",
            Self::DuplicateScenarioId => "frozen scenario IDs must be unique",
            Self::DuplicateDirectionalProbe => "directional probe IDs must be unique",
            Self::InvalidMetricRange => "all five metric ranges must be finite and ordered",
            Self::InvalidDirectionalProbe => "directional probes must be finite and nonzero",
            Self::EmptyObservationSet => "score covariance requires raw vector observations",
            Self::UnknownActionId => "observation references an action outside the frozen set",
            Self::UnknownScenarioId => "observation references a scenario outside the frozen set",
            Self::InvalidMetricObservation => "a metric observation is nonfinite or outside its frozen range",
            Self::DuplicateObservation => "duplicate action × scenario × replication vector",
            Self::IncompleteActionScenarioGrid => "every frozen action × scenario cell must be present",
            Self::NonContiguousReplicationIndices => "cell replication indices must be contiguous from zero",
            Self::InsufficientReplications => "each cell requires at least two complete vector observations",
            Self::NumericFailure => "covariance or eigensystem computation was not numerically representable",
            Self::ReplayMismatch => "score-covariance diagnostic receipt does not reproduce",
        };
        f.write_str(message)
    }
}
impl std::error::Error for ScoreCovarianceError {}

/// Compute five-dimensional sample covariance geometry for every action ×
/// scenario stream. The five coordinates must come from the same raw replicate.
///
/// The diagnostic preserves the sample covariance, correlation, spectrum,
/// effective rank and variance in any pre-registered direction. None of these
/// fields is an inferential confidence set or a substitute for alpha allocation.
pub fn diagnose_score_vector_covariance(
    contract: &ScoreCovarianceContract,
    observations: &[ScoreVectorObservation],
) -> Result<ScoreCovarianceDiagnosticReceipt, ScoreCovarianceError> {
    validate_contract(contract)?;
    if observations.is_empty() {
        return Err(ScoreCovarianceError::EmptyObservationSet);
    }

    let mut action_ids = contract.action_ids.clone();
    action_ids.sort_unstable();
    let mut scenario_ids = contract.scenario_ids.clone();
    scenario_ids.sort_unstable();

    let action_set: BTreeSet<u32> = action_ids.iter().copied().collect();
    let scenario_set: BTreeSet<u16> = scenario_ids.iter().copied().collect();
    let mut grouped =
        BTreeMap::<(u32, u16), BTreeMap<u32, [f64; SCORE_METRIC_DIMENSION]>>::new();

    for observation in observations {
        if !action_set.contains(&observation.action_id) {
            return Err(ScoreCovarianceError::UnknownActionId);
        }
        if !scenario_set.contains(&observation.scenario_id) {
            return Err(ScoreCovarianceError::UnknownScenarioId);
        }
        for (index, value) in observation.metrics.iter().enumerate() {
            let range = contract.metric_ranges[index];
            if !value.is_finite() || *value < range.lower || *value > range.upper {
                return Err(ScoreCovarianceError::InvalidMetricObservation);
            }
        }

        let cell = grouped
            .entry((observation.action_id, observation.scenario_id))
            .or_default();
        if cell
            .insert(observation.replication_index, observation.metrics)
            .is_some()
        {
            return Err(ScoreCovarianceError::DuplicateObservation);
        }
    }

    let expected_cells = action_ids
        .len()
        .checked_mul(scenario_ids.len())
        .ok_or(ScoreCovarianceError::IncompleteActionScenarioGrid)?;
    if grouped.len() != expected_cells {
        return Err(ScoreCovarianceError::IncompleteActionScenarioGrid);
    }

    let mut cells = Vec::with_capacity(expected_cells);
    for action_id in &action_ids {
        for scenario_id in &scenario_ids {
            let cell = grouped
                .get(&(*action_id, *scenario_id))
                .ok_or(ScoreCovarianceError::IncompleteActionScenarioGrid)?;
            let sample_count = cell.len();
            if sample_count < 2 {
                return Err(ScoreCovarianceError::InsufficientReplications);
            }
            if cell
                .keys()
                .enumerate()
                .any(|(expected, observed)| *observed as usize != expected)
            {
                return Err(ScoreCovarianceError::NonContiguousReplicationIndices);
            }

            let mut moments = VectorMoments::default();
            for metrics in cell.values() {
                moments.push(metrics)?;
            }
            let (mean_vector, sample_covariance) = moments.finish()?;
            let correlation_matrix = correlation_matrix(&sample_covariance)?;
            let covariance_eigenvalues = symmetric_eigenvalues(sample_covariance)?;
            let trace = covariance_eigenvalues.iter().sum::<f64>();
            let squared_spectrum = covariance_eigenvalues
                .iter()
                .map(|value| value * value)
                .sum::<f64>();
            if !trace.is_finite() || !squared_spectrum.is_finite() || squared_spectrum < 0.0 {
                return Err(ScoreCovarianceError::NumericFailure);
            }
            let effective_rank = if squared_spectrum > 0.0 {
                (trace * trace / squared_spectrum)
                    .clamp(0.0, SCORE_METRIC_DIMENSION as f64)
            } else {
                0.0
            };
            let eigen_scale = covariance_eigenvalues[0].abs().max(1.0);
            let eigen_tolerance = 1.0e-12 * eigen_scale;
            if covariance_eigenvalues
                .iter()
                .any(|value| *value < -eigen_tolerance)
            {
                return Err(ScoreCovarianceError::NumericFailure);
            }
            let positive_eigenvalues = covariance_eigenvalues
                .iter()
                .copied()
                .filter(|value| *value > eigen_tolerance)
                .collect::<Vec<_>>();
            let covariance_condition_number = if positive_eigenvalues.len()
                == SCORE_METRIC_DIMENSION
            {
                Some(
                    positive_eigenvalues[SCORE_METRIC_DIMENSION - 1]
                        / positive_eigenvalues[0],
                )
            } else {
                None
            };

            let mut directional_variances =
                Vec::with_capacity(contract.directional_probes.len());
            for probe in &contract.directional_probes {
                let mut covariance_aware_variance = 0.0;
                let mut diagonal_only_variance = 0.0;
                for row in 0..SCORE_METRIC_DIMENSION {
                    diagonal_only_variance +=
                        probe.weights[row] * probe.weights[row] * sample_covariance[row][row];
                    for column in 0..SCORE_METRIC_DIMENSION {
                        covariance_aware_variance += probe.weights[row]
                            * sample_covariance[row][column]
                            * probe.weights[column];
                    }
                }
                let tolerance = 1.0e-12
                    * diagonal_only_variance.abs().max(covariance_aware_variance.abs()).max(1.0);
                if !covariance_aware_variance.is_finite()
                    || !diagonal_only_variance.is_finite()
                    || covariance_aware_variance < -tolerance
                    || diagonal_only_variance < -tolerance
                {
                    return Err(ScoreCovarianceError::NumericFailure);
                }
                let covariance_aware_variance = covariance_aware_variance.max(0.0);
                let diagonal_only_variance = diagonal_only_variance.max(0.0);
                let covariance_to_diagonal_ratio = if diagonal_only_variance > 0.0 {
                    Some(covariance_aware_variance / diagonal_only_variance)
                } else {
                    None
                };
                directional_variances.push(DirectionalVarianceDiagnostic {
                    direction_id: probe.direction_id,
                    covariance_aware_variance,
                    diagonal_only_variance,
                    covariance_to_diagonal_ratio,
                });
            }

            cells.push(ScoreCellCovarianceDiagnostic {
                action_id: *action_id,
                scenario_id: *scenario_id,
                sample_count,
                mean_vector,
                sample_covariance,
                correlation_matrix,
                covariance_eigenvalues,
                effective_rank,
                covariance_condition_number,
                directional_variances,
            });
        }
    }

    Ok(ScoreCovarianceDiagnosticReceipt {
        method_id: SCORE_COVARIANCE_DIAGNOSTIC_METHOD_ID,
        policy_version: contract.policy_version.clone(),
        source_digest: contract.source_digest.clone(),
        action_ids,
        scenario_ids,
        metric_ranges: contract.metric_ranges,
        directional_probes: contract.directional_probes.clone(),
        cells,
    })
}

/// Recompute the complete descriptive covariance receipt from raw observations.
pub fn verify_score_covariance_diagnostic_replay(
    contract: &ScoreCovarianceContract,
    observations: &[ScoreVectorObservation],
    recorded: &ScoreCovarianceDiagnosticReceipt,
) -> Result<(), ScoreCovarianceError> {
    let expected = diagnose_score_vector_covariance(contract, observations)?;
    if &expected == recorded {
        Ok(())
    } else {
        Err(ScoreCovarianceError::ReplayMismatch)
    }
}

fn validate_contract(contract: &ScoreCovarianceContract) -> Result<(), ScoreCovarianceError> {
    if contract.policy_version.trim().is_empty() {
        return Err(ScoreCovarianceError::MissingPolicyVersion);
    }
    if contract.source_digest.trim().is_empty() {
        return Err(ScoreCovarianceError::MissingSourceDigest);
    }
    if contract.action_ids.is_empty() {
        return Err(ScoreCovarianceError::EmptyActionSet);
    }
    if contract.scenario_ids.is_empty() {
        return Err(ScoreCovarianceError::EmptyScenarioSet);
    }
    let mut actions = contract.action_ids.clone();
    actions.sort_unstable();
    if actions.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(ScoreCovarianceError::DuplicateActionId);
    }
    let mut scenarios = contract.scenario_ids.clone();
    scenarios.sort_unstable();
    if scenarios.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(ScoreCovarianceError::DuplicateScenarioId);
    }
    for range in contract.metric_ranges {
        if !range.lower.is_finite()
            || !range.upper.is_finite()
            || range.lower > range.upper
        {
            return Err(ScoreCovarianceError::InvalidMetricRange);
        }
    }
    let mut direction_ids = BTreeSet::new();
    for probe in &contract.directional_probes {
        if !direction_ids.insert(probe.direction_id)
            || probe.weights.iter().any(|value| !value.is_finite())
            || probe.weights.iter().all(|value| *value == 0.0)
        {
            return Err(ScoreCovarianceError::InvalidDirectionalProbe);
        }
    }
    Ok(())
}

#[derive(Debug, Default)]
struct VectorMoments {
    count: usize,
    mean: [f64; SCORE_METRIC_DIMENSION],
    co_moment: [[f64; SCORE_METRIC_DIMENSION]; SCORE_METRIC_DIMENSION],
}

impl VectorMoments {
    fn push(
        &mut self,
        values: &[f64; SCORE_METRIC_DIMENSION],
    ) -> Result<(), ScoreCovarianceError> {
        self.count = self
            .count
            .checked_add(1)
            .ok_or(ScoreCovarianceError::NumericFailure)?;
        let mut delta = [0.0; SCORE_METRIC_DIMENSION];
        for index in 0..SCORE_METRIC_DIMENSION {
            delta[index] = values[index] - self.mean[index];
            self.mean[index] += delta[index] / self.count as f64;
        }
        let mut delta_after = [0.0; SCORE_METRIC_DIMENSION];
        for index in 0..SCORE_METRIC_DIMENSION {
            delta_after[index] = values[index] - self.mean[index];
        }
        for row in 0..SCORE_METRIC_DIMENSION {
            for column in row..SCORE_METRIC_DIMENSION {
                let increment = 0.5
                    * (delta[row] * delta_after[column]
                        + delta[column] * delta_after[row]);
                self.co_moment[row][column] += increment;
                if !self.co_moment[row][column].is_finite() {
                    return Err(ScoreCovarianceError::NumericFailure);
                }
            }
        }
        if self.mean.iter().any(|value| !value.is_finite()) {
            return Err(ScoreCovarianceError::NumericFailure);
        }
        Ok(())
    }

    fn finish(
        &self,
    ) -> Result<
        (
            [f64; SCORE_METRIC_DIMENSION],
            [[f64; SCORE_METRIC_DIMENSION]; SCORE_METRIC_DIMENSION],
        ),
        ScoreCovarianceError,
    > {
        if self.count < 2 {
            return Err(ScoreCovarianceError::InsufficientReplications);
        }
        let denominator = (self.count - 1) as f64;
        let mut covariance = [[0.0; SCORE_METRIC_DIMENSION]; SCORE_METRIC_DIMENSION];
        for row in 0..SCORE_METRIC_DIMENSION {
            for column in row..SCORE_METRIC_DIMENSION {
                let value = self.co_moment[row][column] / denominator;
                if !value.is_finite() {
                    return Err(ScoreCovarianceError::NumericFailure);
                }
                covariance[row][column] = value;
                covariance[column][row] = value;
            }
        }
        for index in 0..SCORE_METRIC_DIMENSION {
            if covariance[index][index] < 0.0 {
                let tolerance = 1.0e-12 * self.mean[index].abs().max(1.0);
                if covariance[index][index] < -tolerance {
                    return Err(ScoreCovarianceError::NumericFailure);
                }
                covariance[index][index] = 0.0;
            }
        }
        Ok((self.mean, covariance))
    }
}

fn correlation_matrix(
    covariance: &[[f64; SCORE_METRIC_DIMENSION]; SCORE_METRIC_DIMENSION],
) -> Result<[[Option<f64>; SCORE_METRIC_DIMENSION]; SCORE_METRIC_DIMENSION], ScoreCovarianceError> {
    let mut correlation = [[None; SCORE_METRIC_DIMENSION]; SCORE_METRIC_DIMENSION];
    for row in 0..SCORE_METRIC_DIMENSION {
        for column in 0..SCORE_METRIC_DIMENSION {
            let left_variance = covariance[row][row];
            let right_variance = covariance[column][column];
            if left_variance > 0.0 && right_variance > 0.0 {
                let value = covariance[row][column] / (left_variance * right_variance).sqrt();
                if !value.is_finite() {
                    return Err(ScoreCovarianceError::NumericFailure);
                }
                correlation[row][column] = Some(value.clamp(-1.0, 1.0));
            } else if row == column && left_variance == 0.0 {
                correlation[row][column] = None;
            }
        }
    }
    Ok(correlation)
}

/// Deterministic Jacobi eigensolver for a small symmetric matrix. This is
/// descriptive only; no covariance inverse is formed.
fn symmetric_eigenvalues(
    matrix: [[f64; SCORE_METRIC_DIMENSION]; SCORE_METRIC_DIMENSION],
) -> Result<[f64; SCORE_METRIC_DIMENSION], ScoreCovarianceError> {
    let mut a = matrix;
    let max_iterations = SCORE_METRIC_DIMENSION * SCORE_METRIC_DIMENSION * 64;
    for _ in 0..max_iterations {
        let mut p = 0;
        let mut q = 1;
        let mut largest = 0.0_f64;
        for row in 0..SCORE_METRIC_DIMENSION {
            for column in (row + 1)..SCORE_METRIC_DIMENSION {
                if a[row][column].abs() > largest {
                    largest = a[row][column].abs();
                    p = row;
                    q = column;
                }
            }
        }
        let scale = (0..SCORE_METRIC_DIMENSION)
            .map(|index| a[index][index].abs())
            .fold(1.0_f64, f64::max);
        if largest <= 1.0e-14 * scale {
            break;
        }

        let angle = 0.5 * (2.0 * a[p][q]).atan2(a[q][q] - a[p][p]);
        let cosine = angle.cos();
        let sine = angle.sin();
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];

        for index in 0..SCORE_METRIC_DIMENSION {
            if index != p && index != q {
                let aip = a[index][p];
                let aiq = a[index][q];
                a[index][p] = cosine * aip - sine * aiq;
                a[p][index] = a[index][p];
                a[index][q] = sine * aip + cosine * aiq;
                a[q][index] = a[index][q];
            }
        }

        a[p][p] = cosine * cosine * app
            - 2.0 * sine * cosine * apq
            + sine * sine * aqq;
        a[q][q] = sine * sine * app
            + 2.0 * sine * cosine * apq
            + cosine * cosine * aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
    }

    if a.iter().flatten().any(|value| !value.is_finite()) {
        return Err(ScoreCovarianceError::NumericFailure);
    }
    let mut eigenvalues = std::array::from_fn(|index| a[index][index]);
    eigenvalues.sort_by(f64::total_cmp);
    Ok(eigenvalues)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ranges() -> [MetricRange; SCORE_METRIC_DIMENSION] {
        [MetricRange { lower: -10.0, upper: 10.0 }; SCORE_METRIC_DIMENSION]
    }

    fn contract() -> ScoreCovarianceContract {
        ScoreCovarianceContract {
            policy_version: "score-covariance-test-v1".to_owned(),
            source_digest: "test-source-digest".to_owned(),
            action_ids: vec![1],
            scenario_ids: vec![7],
            metric_ranges: ranges(),
            directional_probes: vec![
                DirectionalProbe {
                    direction_id: 1,
                    weights: [1.0, 1.0, 0.0, 0.0, 0.0],
                },
                DirectionalProbe {
                    direction_id: 2,
                    weights: [1.0, -1.0, 0.0, 0.0, 0.0],
                },
            ],
        }
    }

    fn observations() -> Vec<ScoreVectorObservation> {
        let common = [-2.0, -1.0, 0.0, 1.0, 2.0];
        common
            .iter()
            .enumerate()
            .map(|(index, value)| ScoreVectorObservation {
                action_id: 1,
                scenario_id: 7,
                replication_index: index as u32,
                metrics: [
                    *value,
                    *value,
                    2.0 * value,
                    -*value,
                    0.5 * value,
                ],
            })
            .collect()
    }

    #[test]
    fn common_mode_and_opposite_mode_directions_reflect_covariance() {
        let result = diagnose_score_vector_covariance(&contract(), &observations()).unwrap();
        let cell = &result.cells[0];
        assert_eq!(cell.sample_count, 5);
        assert!(cell.correlation_matrix[0][1].unwrap() > 0.999_999);
        assert!(cell.correlation_matrix[0][3].unwrap() < -0.999_999);
        let common_mode = &cell.directional_variances[0];
        let opposite_mode = &cell.directional_variances[1];
        assert!(
            common_mode.covariance_to_diagonal_ratio.unwrap() > 1.9
        );
        assert!(
            opposite_mode.covariance_aware_variance
                < opposite_mode.diagonal_only_variance * 1.0e-12
        );
        assert!(cell.covariance_condition_number.is_none());
        assert!(cell.effective_rank < 1.01);
    }

    #[test]
    fn input_order_does_not_change_covariance_receipt() {
        let rows = observations();
        let first = diagnose_score_vector_covariance(&contract(), &rows).unwrap();
        let mut reversed = rows;
        reversed.reverse();
        let second = diagnose_score_vector_covariance(&contract(), &reversed).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn replay_accepts_exact_receipt_and_rejects_covariance_mutation() {
        let rows = observations();
        let c = contract();
        let receipt = diagnose_score_vector_covariance(&c, &rows).unwrap();
        assert_eq!(verify_score_covariance_diagnostic_replay(&c, &rows, &receipt), Ok(()));
        let mut changed = receipt.clone();
        changed.cells[0].sample_covariance[0][1] += 1.0;
        assert_eq!(
            verify_score_covariance_diagnostic_replay(&c, &rows, &changed),
            Err(ScoreCovarianceError::ReplayMismatch)
        );
    }

    #[test]
    fn missing_duplicate_unknown_and_nonfinite_vector_rows_fail_closed() {
        let c = contract();
        let rows = observations();
        let missing = &rows[..rows.len() - 1];
        assert_eq!(
            diagnose_score_vector_covariance(&c, missing),
            Err(ScoreCovarianceError::InsufficientReplications)
        );
        let duplicate = vec![rows[0], rows[0]];
        assert_eq!(
            diagnose_score_vector_covariance(&c, &duplicate),
            Err(ScoreCovarianceError::DuplicateObservation)
        );
        let mut unknown = rows.clone();
        unknown[0].action_id = 99;
        assert_eq!(
            diagnose_score_vector_covariance(&c, &unknown),
            Err(ScoreCovarianceError::UnknownActionId)
        );
        let mut nonfinite = rows;
        nonfinite[0].metrics[2] = f64::NAN;
        assert_eq!(
            diagnose_score_vector_covariance(&c, &nonfinite),
            Err(ScoreCovarianceError::InvalidMetricObservation)
        );
    }

    #[test]
    fn rejects_invalid_ranges_and_duplicate_or_zero_probes() {
        let mut c = contract();
        c.metric_ranges[0] = MetricRange { lower: f64::NAN, upper: 1.0 };
        assert_eq!(
            diagnose_score_vector_covariance(&c, &observations()),
            Err(ScoreCovarianceError::InvalidMetricRange)
        );
        let mut c = contract();
        c.directional_probes.push(c.directional_probes[0]);
        assert_eq!(
            diagnose_score_vector_covariance(&c, &observations()),
            Err(ScoreCovarianceError::InvalidDirectionalProbe)
        );
        let mut c = contract();
        c.directional_probes[0].weights = [0.0; SCORE_METRIC_DIMENSION];
        assert_eq!(
            diagnose_score_vector_covariance(&c, &observations()),
            Err(ScoreCovarianceError::InvalidDirectionalProbe)
        );
    }

    #[test]
    fn recovers_diagonal_covariance_for_uncorrelated_vector_fixture() {
        let mut c = contract();
        c.directional_probes = vec![DirectionalProbe {
            direction_id: 5,
            weights: [1.0, 1.0, 1.0, 1.0, 1.0],
        }];
        let values = [
            [-2.0, 2.0, -1.0, 1.0, 0.5],
            [-1.0, -1.0, 2.0, -2.0, -0.5],
            [0.0, 1.0, -2.0, 0.0, 1.0],
            [1.0, -2.0, 1.0, 2.0, -1.0],
            [2.0, 0.0, 0.0, -1.0, 0.0],
        ];
        let rows = values
            .iter()
            .enumerate()
            .map(|(index, metrics)| ScoreVectorObservation {
                action_id: 1,
                scenario_id: 7,
                replication_index: index as u32,
                metrics: *metrics,
            })
            .collect::<Vec<_>>();
        let result = diagnose_score_vector_covariance(&c, &rows).unwrap();
        let diagnostic = &result.cells[0].directional_variances[0];
        assert!(
            (diagnostic.covariance_to_diagonal_ratio.unwrap() - 1.0).abs() < 0.5
        );
    }
}
