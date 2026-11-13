// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Data-only Jacobian observability diagnostics.
//!
//! This module reports numerical rank, singular values, condition number and
//! weak right-singular directions before priors are applied. It does not fix
//! coordinate/time gauges, estimate a trajectory, or claim that a full-rank
//! local Jacobian guarantees global identifiability.

use std::error::Error;
use std::fmt;

/// Maximum parameter dimension accepted by the dependency-free Jacobi kernel.
pub const MAX_OBSERVABILITY_PARAMETERS: usize = 64;

/// Invalid input or numerical failure while analyzing a measurement Jacobian.
#[derive(Clone, Debug, PartialEq)]
pub enum ObservabilityError {
    NoObservations,
    NoParameters,
    TooManyParameters {
        count: usize,
        maximum: usize,
    },
    ParameterScaleCount {
        expected: usize,
        actual: usize,
    },
    InvalidParameterScale {
        column: usize,
    },
    InvalidRelativeTolerance,
    RaggedJacobian {
        row: usize,
        expected: usize,
        actual: usize,
    },
    NonFiniteJacobian {
        row: usize,
        column: usize,
    },
    NumericalOverflow,
    JacobiSvdDidNotConverge,
}

impl fmt::Display for ObservabilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoObservations => write!(f, "Jacobian has no observation rows"),
            Self::NoParameters => write!(f, "Jacobian has no parameter columns"),
            Self::TooManyParameters { count, maximum } => {
                write!(
                    f,
                    "Jacobian has {count} parameters; maximum supported is {maximum}"
                )
            }
            Self::ParameterScaleCount { expected, actual } => {
                write!(f, "received {actual} parameter scales; expected {expected}")
            }
            Self::InvalidParameterScale { column } => {
                write!(
                    f,
                    "parameter scale at column {column} is not finite and positive"
                )
            }
            Self::InvalidRelativeTolerance => {
                write!(
                    f,
                    "relative tolerance must be finite and between zero and one"
                )
            }
            Self::RaggedJacobian {
                row,
                expected,
                actual,
            } => write!(
                f,
                "Jacobian row {row} has {actual} columns; expected {expected}"
            ),
            Self::NonFiniteJacobian { row, column } => {
                write!(
                    f,
                    "Jacobian entry at row {row}, column {column} is not finite"
                )
            }
            Self::NumericalOverflow => {
                write!(f, "Jacobian scaling or information accumulation overflowed")
            }
            Self::JacobiSvdDidNotConverge => {
                write!(f, "one-sided Jacobi SVD did not converge")
            }
        }
    }
}

impl Error for ObservabilityError {}

/// Local, data-only observability report.
///
/// Singular values are computed from the column-scaled Jacobian
/// `J_scaled = J * diag(parameter_scales)`. A scale is a characteristic
/// parameter displacement in the parameter's physical unit; this makes the
/// diagnostic dimensionless and avoids comparing metres, seconds and radians
/// without an explicit convention.
///
/// Weak directions are right-singular vectors in these scaled coordinates.
/// They identify parameter combinations that the supplied observations do not
/// constrain above `relative_tolerance * largest_singular_value`.
///
/// No prior precision is accepted by this API. A caller must report any prior
/// regularization separately rather than allowing it to disguise a data null
/// direction.
#[derive(Clone, Debug, PartialEq)]
pub struct ObservabilityReport {
    pub observation_count: usize,
    pub parameter_count: usize,
    pub numerical_rank: usize,
    pub nullity: usize,
    /// Singular values in descending order, for the scaled Jacobian.
    pub singular_values: Vec<f64>,
    /// Present only when every parameter column is numerically observable.
    pub condition_number: Option<f64>,
    /// Right-singular vectors whose singular values are at or below the
    /// declared rank threshold; vectors use scaled parameter coordinates.
    pub weak_directions: Vec<Vec<f64>>,
    pub parameter_scales: Vec<f64>,
    pub relative_tolerance: f64,
}

impl ObservabilityReport {
    /// Whether all supplied parameter directions pass the declared rank test.
    ///
    /// This is a local, gauge-dependent statement. The caller remains
    /// responsible for fixing and documenting time-origin and coordinate-frame
    /// gauges before interpreting full column rank as physical identifiability.
    pub fn has_full_column_rank(&self) -> bool {
        self.numerical_rank == self.parameter_count
    }

    /// Smallest singular value, or zero for a zero-information Jacobian.
    pub fn smallest_singular_value(&self) -> f64 {
        self.singular_values.last().copied().unwrap_or(0.0)
    }
}

/// Analyze a measurement Jacobian without adding priors.
///
/// `jacobian[row][column]` is the derivative of a whitened residual with
/// respect to one physical parameter. If residuals are not whitened, callers
/// must whiten them using the declared measurement covariance before calling
/// this function. Parameter scales must be finite and strictly positive.
///
/// The implementation applies a one-sided Jacobi SVD directly to the
/// scaled Jacobian, avoiding the condition-number squaring caused by explicitly
/// forming `J_scaled^T * J_scaled`. This dependency-free kernel is intended for
/// small calibration/state blocks (at most 64 parameters), not as a replacement
/// for a high-performance SVD or rank-revealing QR package.
pub fn analyze_jacobian(
    jacobian: &[Vec<f64>],
    parameter_scales: &[f64],
    relative_tolerance: f64,
) -> Result<ObservabilityReport, ObservabilityError> {
    if jacobian.is_empty() {
        return Err(ObservabilityError::NoObservations);
    }

    let parameter_count = jacobian[0].len();
    if parameter_count == 0 {
        return Err(ObservabilityError::NoParameters);
    }
    if parameter_count > MAX_OBSERVABILITY_PARAMETERS {
        return Err(ObservabilityError::TooManyParameters {
            count: parameter_count,
            maximum: MAX_OBSERVABILITY_PARAMETERS,
        });
    }
    if parameter_scales.len() != parameter_count {
        return Err(ObservabilityError::ParameterScaleCount {
            expected: parameter_count,
            actual: parameter_scales.len(),
        });
    }
    if !relative_tolerance.is_finite() || relative_tolerance <= 0.0 || relative_tolerance >= 1.0 {
        return Err(ObservabilityError::InvalidRelativeTolerance);
    }

    for (column, scale) in parameter_scales.iter().copied().enumerate() {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(ObservabilityError::InvalidParameterScale { column });
        }
    }

    let observation_count = jacobian.len();
    let mut scaled_jacobian = vec![vec![0.0; parameter_count]; observation_count];

    for (row_index, row) in jacobian.iter().enumerate() {
        if row.len() != parameter_count {
            return Err(ObservabilityError::RaggedJacobian {
                row: row_index,
                expected: parameter_count,
                actual: row.len(),
            });
        }

        for (column, value) in row.iter().copied().enumerate() {
            if !value.is_finite() {
                return Err(ObservabilityError::NonFiniteJacobian {
                    row: row_index,
                    column,
                });
            }
            let scaled = value * parameter_scales[column];
            if !scaled.is_finite() {
                return Err(ObservabilityError::NumericalOverflow);
            }
            scaled_jacobian[row_index][column] = scaled;
        }
    }

    // Work directly on J rather than forming J^T J. The latter squares the
    // condition number and can erase exactly the weak singular directions this
    // diagnostic exists to expose. A one-sided Jacobi SVD keeps the small,
    // dependency-free parameter-block kernel numerically better behaved.
    let (singular_values, right_singular_vectors) =
        one_sided_jacobi_svd(scaled_jacobian, parameter_count)?;

    let largest = singular_values.first().copied().unwrap_or(0.0);
    let rank_threshold = largest * relative_tolerance;
    let numerical_rank = singular_values
        .iter()
        .filter(|value| **value > rank_threshold)
        .count();
    let nullity = parameter_count - numerical_rank;

    let mut weak_directions = Vec::with_capacity(nullity);
    for (position, singular_value) in singular_values.iter().copied().enumerate() {
        if singular_value <= rank_threshold {
            let direction = (0..parameter_count)
                .map(|row| right_singular_vectors[row][position])
                .collect();
            weak_directions.push(direction);
        }
    }

    let condition_number = if numerical_rank == parameter_count {
        let smallest = singular_values[parameter_count - 1];
        if smallest > 0.0 {
            Some(largest / smallest)
        } else {
            None
        }
    } else {
        None
    };

    Ok(ObservabilityReport {
        observation_count,
        parameter_count,
        numerical_rank,
        nullity,
        singular_values,
        condition_number,
        weak_directions,
        parameter_scales: parameter_scales.to_vec(),
        relative_tolerance,
    })
}

/// One-sided Jacobi SVD for a tall or square matrix with a small column count.
///
/// The right singular vectors are returned as columns. Rotations orthogonalize
/// columns of the working matrix directly, avoiding the loss of small singular
/// values caused by explicitly forming the Gram matrix. This routine is
/// intentionally capped by MAX_OBSERVABILITY_PARAMETERS at its public boundary.
fn one_sided_jacobi_svd(
    mut matrix: Vec<Vec<f64>>,
    column_count: usize,
) -> Result<(Vec<f64>, Vec<Vec<f64>>), ObservabilityError> {
    let mut right_vectors = vec![vec![0.0; column_count]; column_count];
    for (index, row) in right_vectors.iter_mut().enumerate() {
        row[index] = 1.0;
    }

    if matrix
        .iter()
        .all(|row| row.iter().all(|value| *value == 0.0))
    {
        return Ok((vec![0.0; column_count], right_vectors));
    }

    const MAX_SWEEPS: usize = 128;
    const ORTHOGONALITY_TOLERANCE: f64 = 16.0 * f64::EPSILON;
    let mut converged = false;

    for _ in 0..MAX_SWEEPS {
        let mut rotated_any_pair = false;

        for left in 0..column_count {
            for right in (left + 1)..column_count {
                let mut alpha = 0.0;
                let mut beta = 0.0;
                let mut gamma = 0.0;

                for row in &matrix {
                    alpha += row[left] * row[left];
                    beta += row[right] * row[right];
                    gamma += row[left] * row[right];
                }

                if !alpha.is_finite() || !beta.is_finite() || !gamma.is_finite() {
                    return Err(ObservabilityError::NumericalOverflow);
                }
                if alpha == 0.0 || beta == 0.0 {
                    continue;
                }

                let scale = alpha.sqrt() * beta.sqrt();
                if !scale.is_finite() {
                    return Err(ObservabilityError::NumericalOverflow);
                }
                if gamma.abs() <= ORTHOGONALITY_TOLERANCE * scale {
                    continue;
                }

                let tau = (beta - alpha) / (2.0 * gamma);
                let tangent = if tau >= 0.0 {
                    1.0 / (tau + tau.hypot(1.0))
                } else {
                    -1.0 / (-tau + tau.hypot(1.0))
                };
                let cosine = 1.0 / (1.0 + tangent * tangent).sqrt();
                let sine = tangent * cosine;

                for row in &mut matrix {
                    let left_value = row[left];
                    let right_value = row[right];
                    row[left] = cosine * left_value - sine * right_value;
                    row[right] = sine * left_value + cosine * right_value;
                    if !row[left].is_finite() || !row[right].is_finite() {
                        return Err(ObservabilityError::NumericalOverflow);
                    }
                }

                for row in &mut right_vectors {
                    let left_value = row[left];
                    let right_value = row[right];
                    row[left] = cosine * left_value - sine * right_value;
                    row[right] = sine * left_value + cosine * right_value;
                }

                rotated_any_pair = true;
            }
        }

        if !rotated_any_pair {
            converged = true;
            break;
        }
    }

    if !converged {
        // Check the final maximum normalized column correlation before
        // reporting a solver failure; the last sweep can have made the matrix
        // converged even if it performed a final round of tiny rotations.
        let mut maximum_correlation = 0.0_f64;
        for left in 0..column_count {
            for right in (left + 1)..column_count {
                let mut alpha = 0.0;
                let mut beta = 0.0;
                let mut gamma = 0.0;
                for row in &matrix {
                    alpha += row[left] * row[left];
                    beta += row[right] * row[right];
                    gamma += row[left] * row[right];
                }
                if !alpha.is_finite() || !beta.is_finite() || !gamma.is_finite() {
                    return Err(ObservabilityError::NumericalOverflow);
                }
                let denominator = alpha.sqrt() * beta.sqrt();
                if denominator > 0.0 {
                    maximum_correlation = maximum_correlation.max(gamma.abs() / denominator);
                }
            }
        }
        if maximum_correlation > ORTHOGONALITY_TOLERANCE {
            return Err(ObservabilityError::JacobiSvdDidNotConverge);
        }
    }

    let mut indexed_singular_values = Vec::with_capacity(column_count);
    for column in 0..column_count {
        let norm = matrix
            .iter()
            .fold(0.0_f64, |accumulator, row| accumulator.hypot(row[column]));
        if !norm.is_finite() {
            return Err(ObservabilityError::NumericalOverflow);
        }
        indexed_singular_values.push((column, norm));
    }
    indexed_singular_values.sort_by(|left, right| right.1.total_cmp(&left.1));

    let singular_values = indexed_singular_values
        .iter()
        .map(|(_, singular_value)| *singular_value)
        .collect();
    let sorted_right_vectors = (0..column_count)
        .map(|row| {
            indexed_singular_values
                .iter()
                .map(|(column, _)| right_vectors[row][*column])
                .collect()
        })
        .collect();

    Ok((singular_values, sorted_right_vectors))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_jacobian_is_full_rank_with_unit_condition() {
        let report = analyze_jacobian(
            &[
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
            &[1.0, 1.0, 1.0],
            1.0e-10,
        )
        .unwrap();

        assert_eq!(report.numerical_rank, 3);
        assert_eq!(report.nullity, 0);
        assert!(report.has_full_column_rank());
        assert!((report.condition_number.unwrap() - 1.0).abs() < 1.0e-12);
        for singular_value in report.singular_values {
            assert!((singular_value - 1.0).abs() < 1.0e-12);
        }
    }

    #[test]
    fn missing_parameter_column_is_reported_as_a_null_direction() {
        let report = analyze_jacobian(
            &[
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![1.0, 1.0, 0.0],
            ],
            &[1.0, 1.0, 1.0],
            1.0e-8,
        )
        .unwrap();

        assert_eq!(report.numerical_rank, 2);
        assert_eq!(report.nullity, 1);
        assert_eq!(report.weak_directions.len(), 1);
        assert!(report.condition_number.is_none());
        assert!(report.weak_directions[0][2].abs() > 1.0 - 1.0e-10);
    }

    #[test]
    fn parameter_scales_remove_unit_driven_conditioning() {
        let jacobian = &[vec![1.0, 0.0], vec![0.0, 1.0e-6]];
        let unscaled = analyze_jacobian(jacobian, &[1.0, 1.0], 1.0e-10).unwrap();
        let scaled = analyze_jacobian(jacobian, &[1.0, 1.0e6], 1.0e-10).unwrap();

        assert_eq!(unscaled.numerical_rank, 2);
        assert!(unscaled.condition_number.unwrap() > 999_000.0);
        assert_eq!(scaled.numerical_rank, 2);
        assert!((scaled.condition_number.unwrap() - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn near_dependent_columns_are_rank_deficient_at_declared_tolerance() {
        let report =
            analyze_jacobian(&[vec![1.0, 1.0], vec![0.0, 1.0e-5]], &[1.0, 1.0], 1.0e-4).unwrap();

        assert_eq!(report.numerical_rank, 1);
        assert_eq!(report.nullity, 1);
        assert!(report.smallest_singular_value() < 1.0e-4);
    }

    #[test]
    fn direct_svd_preserves_a_small_singular_value_lost_by_gram_squaring() {
        // J has a real second singular value about 7.07e-9. Forming J^T J
        // subtracts nearly equal O(1) terms and loses that direction in f64.
        let report =
            analyze_jacobian(&[vec![1.0, 1.0], vec![0.0, 1.0e-8]], &[1.0, 1.0], 1.0e-10).unwrap();

        assert_eq!(report.numerical_rank, 2);
        assert!(report.singular_values[1] > 6.0e-9);
        assert!(report.singular_values[1] < 8.0e-9);
        assert!(report.condition_number.unwrap() > 1.0e8);
    }

    #[test]
    fn stationary_target_cannot_identify_a_clock_offset() {
        // With a static target and static sensors, changing the sensor clock
        // offset does not change any predicted bearing. The fourth Jacobian
        // column is therefore a true data-only null direction.
        let report = analyze_jacobian(
            &[
                vec![2.0, 0.1, 0.0, 0.0],
                vec![0.0, 1.5, 0.2, 0.0],
                vec![0.1, 0.2, 1.0, 0.0],
                vec![1.0, -0.4, 0.2, 0.0],
            ],
            &[1.0, 1.0, 1.0, 1.0],
            1.0e-8,
        )
        .unwrap();

        assert_eq!(report.numerical_rank, 3);
        assert_eq!(report.nullity, 1);
        assert!(report.weak_directions[0][3].abs() > 1.0 - 1.0e-10);
    }

    #[test]
    fn zero_information_jacobian_has_zero_rank() {
        let report =
            analyze_jacobian(&[vec![0.0, 0.0], vec![0.0, 0.0]], &[1.0, 1.0], 1.0e-8).unwrap();

        assert_eq!(report.numerical_rank, 0);
        assert_eq!(report.nullity, 2);
        assert_eq!(report.condition_number, None);
    }

    #[test]
    fn row_orthogonal_transform_preserves_singular_values() {
        let original = &[vec![1.0, 2.0], vec![3.0, -1.0], vec![0.5, 4.0]];
        // Apply a 2x2 orthogonal rotation to the first two observation rows.
        let rotated = vec![
            vec![
                (original[0][0] + original[1][0]) / 2.0_f64.sqrt(),
                (original[0][1] + original[1][1]) / 2.0_f64.sqrt(),
            ],
            vec![
                (original[0][0] - original[1][0]) / 2.0_f64.sqrt(),
                (original[0][1] - original[1][1]) / 2.0_f64.sqrt(),
            ],
            original[2].clone(),
        ];
        let baseline = analyze_jacobian(original, &[1.0, 1.0], 1.0e-10).unwrap();
        let rotated_report = analyze_jacobian(&rotated, &[1.0, 1.0], 1.0e-10).unwrap();
        for (left, right) in baseline
            .singular_values
            .iter()
            .zip(rotated_report.singular_values.iter())
        {
            assert!((left - right).abs() < 1.0e-10);
        }
    }

    #[test]
    fn invalid_inputs_fail_closed() {
        assert_eq!(
            analyze_jacobian(&[], &[1.0], 1.0e-8),
            Err(ObservabilityError::NoObservations)
        );
        assert_eq!(
            analyze_jacobian(&[vec![]], &[], 1.0e-8),
            Err(ObservabilityError::NoParameters)
        );
        assert!(matches!(
            analyze_jacobian(&[vec![1.0, 2.0]], &[1.0], 1.0e-8),
            Err(ObservabilityError::ParameterScaleCount { .. })
        ));
        assert!(matches!(
            analyze_jacobian(&[vec![1.0, 2.0]], &[1.0, 0.0], 1.0e-8),
            Err(ObservabilityError::InvalidParameterScale { column: 1 })
        ));
        assert!(matches!(
            analyze_jacobian(&[vec![1.0], vec![1.0, 2.0]], &[1.0], 1.0e-8),
            Err(ObservabilityError::RaggedJacobian { .. })
        ));
        assert!(matches!(
            analyze_jacobian(&[vec![f64::NAN]], &[1.0], 1.0e-8),
            Err(ObservabilityError::NonFiniteJacobian { .. })
        ));
        assert_eq!(
            analyze_jacobian(&[vec![1.0]], &[1.0], 0.0),
            Err(ObservabilityError::InvalidRelativeTolerance)
        );
    }
}
