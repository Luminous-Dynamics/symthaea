//! Projection-composition assurance for RCAF.
//! Demonstrates nuisance re-entry under sequential finite-dimensional orthogonal projections.
//! This is a mathematical reference kernel, not a general fMRI filter or denoiser.

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct ProjectionOrderReport {
    pub schema_version: &'static str,
    pub time_points: usize,
    pub signal_columns: usize,
    pub nuisance_design_columns: usize,
    pub filter_design_columns: usize,
    pub nuisance_design_rank: usize,
    pub filter_design_rank: usize,
    pub joint_design_rank: usize,
    pub dependent_columns_dropped: usize,
    /// Frobenius norm of the projection-operator commutator, normalized by sqrt(time_points).
    /// This is an input-independent measure of structural order sensitivity.
    pub projection_operator_commutator_ratio: f64,
    pub nuisance_then_filter_nuisance_leakage: f64,
    pub filter_then_nuisance_nuisance_leakage: f64,
    pub nuisance_then_filter_filter_leakage: f64,
    pub filter_then_nuisance_filter_leakage: f64,
    pub joint_nuisance_leakage: f64,
    pub joint_filter_leakage: f64,
    pub order_difference_ratio: f64,
    pub nuisance_then_filter_vs_joint_ratio: f64,
    pub filter_then_nuisance_vs_joint_ratio: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProjectionCompositionError {
    InvalidObservation(&'static str),
    InvalidDesign(&'static str),
    DimensionMismatch(&'static str),
    ZeroRankDesign(&'static str),
}

impl fmt::Display for ProjectionCompositionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidObservation(reason) => {
                write!(f, "invalid projection observation: {reason}")
            }
            Self::InvalidDesign(reason) => write!(f, "invalid projection design: {reason}"),
            Self::DimensionMismatch(reason) => write!(f, "projection dimension mismatch: {reason}"),
            Self::ZeroRankDesign(reason) => write!(f, "projection design has zero rank: {reason}"),
        }
    }
}

impl std::error::Error for ProjectionCompositionError {}

#[derive(Debug, Clone)]
struct OrthonormalBasis {
    columns: Vec<Vec<f64>>,
    input_columns: usize,
    dropped_columns: usize,
}

impl OrthonormalBasis {
    fn rank(&self) -> usize {
        self.columns.len()
    }
}

/// Run nuisance-then-filter, filter-then-nuisance, and one-shot joint projections.
/// Matrices use time x signal layout; design matrices use time x regressor layout.
pub fn run_projection_order_challenge(
    observations: &[Vec<f64>],
    nuisance_design: &[Vec<f64>],
    removed_filter_design: &[Vec<f64>],
) -> Result<ProjectionOrderReport, ProjectionCompositionError> {
    validate_observations(observations)?;
    let time_points = observations.len();
    let signal_columns = observations[0].len();
    validate_design(nuisance_design, time_points)?;
    validate_design(removed_filter_design, time_points)?;

    let nuisance_basis = orthonormal_basis(nuisance_design)?;
    let filter_basis = orthonormal_basis(removed_filter_design)?;
    if nuisance_basis.rank() == 0 {
        return Err(ProjectionCompositionError::ZeroRankDesign("nuisance"));
    }
    if filter_basis.rank() == 0 {
        return Err(ProjectionCompositionError::ZeroRankDesign("filter"));
    }

    let nuisance_then_filter =
        residualize(&residualize(observations, &nuisance_basis), &filter_basis);
    let filter_then_nuisance =
        residualize(&residualize(observations, &filter_basis), &nuisance_basis);
    let joint_basis =
        orthonormal_basis(&concatenate_designs(nuisance_design, removed_filter_design))?;
    let joint_output = residualize(observations, &joint_basis);
    let original_norm = frobenius_norm(observations).max(1e-30);

    Ok(ProjectionOrderReport {
        schema_version: "rcaf.projection-order/0.1.0",
        time_points,
        signal_columns,
        nuisance_design_columns: nuisance_basis.input_columns,
        filter_design_columns: filter_basis.input_columns,
        nuisance_design_rank: nuisance_basis.rank(),
        filter_design_rank: filter_basis.rank(),
        joint_design_rank: joint_basis.rank(),
        dependent_columns_dropped: nuisance_basis.dropped_columns
            + filter_basis.dropped_columns
            + joint_basis.dropped_columns,
        projection_operator_commutator_ratio: projection_operator_commutator_ratio(
            &nuisance_basis,
            &filter_basis,
            time_points,
        ),
        nuisance_then_filter_nuisance_leakage: projected_energy_fraction(
            &nuisance_then_filter,
            &nuisance_basis,
        ),
        filter_then_nuisance_nuisance_leakage: projected_energy_fraction(
            &filter_then_nuisance,
            &nuisance_basis,
        ),
        nuisance_then_filter_filter_leakage: projected_energy_fraction(
            &nuisance_then_filter,
            &filter_basis,
        ),
        filter_then_nuisance_filter_leakage: projected_energy_fraction(
            &filter_then_nuisance,
            &filter_basis,
        ),
        joint_nuisance_leakage: projected_energy_fraction(&joint_output, &nuisance_basis),
        joint_filter_leakage: projected_energy_fraction(&joint_output, &filter_basis),
        order_difference_ratio: frobenius_difference(&nuisance_then_filter, &filter_then_nuisance)
            / original_norm,
        nuisance_then_filter_vs_joint_ratio: frobenius_difference(
            &nuisance_then_filter,
            &joint_output,
        ) / original_norm,
        filter_then_nuisance_vs_joint_ratio: frobenius_difference(
            &filter_then_nuisance,
            &joint_output,
        ) / original_norm,
    })
}

fn validate_observations(observations: &[Vec<f64>]) -> Result<(), ProjectionCompositionError> {
    if observations.len() < 3 {
        return Err(ProjectionCompositionError::InvalidObservation(
            "at least three time points are required",
        ));
    }
    let columns = observations[0].len();
    if columns == 0 || observations.iter().any(|row| row.len() != columns) {
        return Err(ProjectionCompositionError::InvalidObservation(
            "observations must be a non-empty rectangular matrix",
        ));
    }
    if observations
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(ProjectionCompositionError::InvalidObservation(
            "observations must be finite",
        ));
    }
    Ok(())
}

fn validate_design(
    design: &[Vec<f64>],
    time_points: usize,
) -> Result<(), ProjectionCompositionError> {
    if design.len() != time_points {
        return Err(ProjectionCompositionError::DimensionMismatch(
            "design rows must match observation time points",
        ));
    }
    let columns = design.first().map_or(0, Vec::len);
    if columns == 0 || design.iter().any(|row| row.len() != columns) {
        return Err(ProjectionCompositionError::InvalidDesign(
            "design must be a non-empty rectangular matrix",
        ));
    }
    if design.iter().flatten().any(|value| !value.is_finite()) {
        return Err(ProjectionCompositionError::InvalidDesign(
            "design values must be finite",
        ));
    }
    Ok(())
}

/// Modified Gram-Schmidt with re-orthogonalization. Dependent columns are counted and dropped.
fn orthonormal_basis(design: &[Vec<f64>]) -> Result<OrthonormalBasis, ProjectionCompositionError> {
    let rows = design.len();
    let columns = design.first().map_or(0, Vec::len);
    if rows == 0 || columns == 0 {
        return Err(ProjectionCompositionError::InvalidDesign(
            "design must have at least one row and column",
        ));
    }

    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(columns);
    let mut dropped_columns = 0usize;
    for column in 0..columns {
        let mut candidate: Vec<f64> = design.iter().map(|row| row[column]).collect();
        let original_norm = vector_norm(&candidate);
        if !original_norm.is_finite() || original_norm <= 1e-30 {
            dropped_columns += 1;
            continue;
        }
        for _ in 0..2 {
            for previous in &basis {
                let projection = dot(&candidate, previous);
                for (value, direction) in candidate.iter_mut().zip(previous.iter()) {
                    *value -= projection * direction;
                }
            }
        }
        let norm = vector_norm(&candidate);
        if !norm.is_finite() || norm <= original_norm * 1e-10 {
            dropped_columns += 1;
            continue;
        }
        for value in &mut candidate {
            *value /= norm;
        }
        basis.push(candidate);
    }
    Ok(OrthonormalBasis {
        columns: basis,
        input_columns: columns,
        dropped_columns,
    })
}

fn residualize(observations: &[Vec<f64>], basis: &OrthonormalBasis) -> Vec<Vec<f64>> {
    let mut residual = observations.to_vec();
    for direction in &basis.columns {
        let width = residual[0].len();
        let mut coefficients = vec![0.0; width];
        for (time, row) in residual.iter().enumerate() {
            for signal in 0..width {
                coefficients[signal] += direction[time] * row[signal];
            }
        }
        for (time, row) in residual.iter_mut().enumerate() {
            for signal in 0..width {
                row[signal] -= direction[time] * coefficients[signal];
            }
        }
    }
    residual
}

fn projected_energy_fraction(observations: &[Vec<f64>], basis: &OrthonormalBasis) -> f64 {
    let total = observations.iter().flatten().map(|x| x * x).sum::<f64>();
    if total <= 1e-30 {
        return 0.0;
    }
    let width = observations[0].len();
    let mut projected = 0.0;
    for direction in &basis.columns {
        for signal in 0..width {
            let coefficient = observations
                .iter()
                .enumerate()
                .map(|(time, row)| direction[time] * row[signal])
                .sum::<f64>();
            projected += coefficient * coefficient;
        }
    }
    (projected / total).clamp(0.0, 1.0)
}

/// Compute ||P_N P_H - P_H P_N||_F / sqrt(T) without materializing either
/// projector. The orthonormal bases are sufficient because P = Q Q^T.
fn projection_operator_commutator_ratio(
    nuisance_basis: &OrthonormalBasis,
    filter_basis: &OrthonormalBasis,
    time_points: usize,
) -> f64 {
    let mut commutator = vec![vec![0.0; time_points]; time_points];
    for nuisance_direction in &nuisance_basis.columns {
        for filter_direction in &filter_basis.columns {
            let overlap = dot(nuisance_direction, filter_direction);
            for (i, row) in commutator.iter_mut().enumerate() {
                for (j, value) in row.iter_mut().enumerate() {
                    *value += overlap
                        * (nuisance_direction[i] * filter_direction[j]
                            - filter_direction[i] * nuisance_direction[j]);
                }
            }
        }
    }
    frobenius_norm(&commutator) / (time_points as f64).sqrt()
}

fn concatenate_designs(left: &[Vec<f64>], right: &[Vec<f64>]) -> Vec<Vec<f64>> {
    left.iter()
        .zip(right.iter())
        .map(|(left_row, right_row)| {
            let mut row = Vec::with_capacity(left_row.len() + right_row.len());
            row.extend_from_slice(left_row);
            row.extend_from_slice(right_row);
            row
        })
        .collect()
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right.iter()).map(|(a, b)| a * b).sum()
}

fn vector_norm(values: &[f64]) -> f64 {
    dot(values, values).sqrt()
}

fn frobenius_norm(matrix: &[Vec<f64>]) -> f64 {
    matrix.iter().flatten().map(|x| x * x).sum::<f64>().sqrt()
}

fn frobenius_difference(left: &[Vec<f64>], right: &[Vec<f64>]) -> f64 {
    left.iter()
        .zip(right.iter())
        .flat_map(|(a, b)| a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn col(values: &[f64]) -> Vec<Vec<f64>> {
        values.iter().map(|value| vec![*value]).collect()
    }

    #[test]
    fn sequential_projection_can_reintroduce_removed_nuisance() {
        let observations = col(&[1.0, 1.0, 0.0, 0.0]);
        let nuisance = col(&[1.0, 0.0, 0.0, 0.0]);
        let filter = col(&[1.0, 1.0, 0.0, 0.0]);
        let report = run_projection_order_challenge(&observations, &nuisance, &filter).unwrap();
        assert!(report.nuisance_then_filter_nuisance_leakage > 0.1);
        assert!(report.filter_then_nuisance_nuisance_leakage < 1e-10);
        assert!(report.order_difference_ratio > 0.1);
        assert!(report.projection_operator_commutator_ratio > 0.0);
    }

    #[test]
    fn joint_projection_is_orthogonal_to_both_declared_subspaces() {
        let observations = vec![
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![0.0, 2.0],
            vec![0.0, -1.0],
        ];
        let nuisance = col(&[1.0, 0.0, 0.0, 0.0]);
        let filter = col(&[1.0, 1.0, 0.0, 0.0]);
        let report = run_projection_order_challenge(&observations, &nuisance, &filter).unwrap();
        assert!(report.joint_nuisance_leakage < 1e-10);
        assert!(report.joint_filter_leakage < 1e-10);
    }

    #[test]
    fn orthogonal_projection_subspaces_commute() {
        let observations = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![2.0, 1.0],
            vec![3.0, -1.0],
        ];
        let nuisance = col(&[1.0, 0.0, 0.0, 0.0]);
        let filter = col(&[0.0, 1.0, 0.0, 0.0]);
        let report = run_projection_order_challenge(&observations, &nuisance, &filter).unwrap();
        assert!(report.order_difference_ratio < 1e-10);
        assert!(report.projection_operator_commutator_ratio < 1e-12);
    }

    #[test]
    fn dependent_design_columns_are_counted_and_dropped() {
        let observations = vec![vec![1.0], vec![2.0], vec![0.0], vec![1.0]];
        let nuisance = vec![
            vec![1.0, 2.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
        ];
        let filter = col(&[0.0, 1.0, 0.0, 0.0]);
        let report = run_projection_order_challenge(&observations, &nuisance, &filter).unwrap();
        assert_eq!(report.nuisance_design_rank, 1);
        assert!(report.dependent_columns_dropped >= 1);
    }

    #[test]
    fn invalid_dimensions_and_non_finite_inputs_fail_closed() {
        let observations = col(&[1.0, 0.0, 2.0, 1.0]);
        let nuisance = col(&[1.0, 0.0, 0.0, 0.0]);
        assert!(matches!(
            run_projection_order_challenge(&observations, &nuisance, &col(&[1.0, 0.0, 0.0])),
            Err(ProjectionCompositionError::DimensionMismatch(_))
        ));
        assert!(matches!(
            run_projection_order_challenge(
                &observations,
                &col(&[1.0, f64::NAN, 0.0, 0.0]),
                &nuisance
            ),
            Err(ProjectionCompositionError::InvalidDesign(_))
        ));
        assert!(matches!(
            run_projection_order_challenge(
                &col(&[1.0, 0.0, f64::INFINITY, 1.0]),
                &nuisance,
                &nuisance
            ),
            Err(ProjectionCompositionError::InvalidObservation(_))
        ));
    }
}
