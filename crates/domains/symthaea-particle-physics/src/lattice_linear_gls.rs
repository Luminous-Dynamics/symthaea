// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Generic covariance-aware linear-model fitting for lattice analyses.
//!
//! The caller supplies the full observable covariance and an explicit design
//! matrix. This module does not choose fit ranges, basis functions, covariance
//! regularization, or a physical model.

use crate::lattice_covariance::CovarianceError;

#[derive(Debug, Clone, PartialEq)]
pub struct CorrelatedLinearFit {
    pub parameters: Vec<f64>,
    pub parameter_covariance: Vec<Vec<f64>>,
    pub chi_square: f64,
    pub degrees_of_freedom: usize,
    pub chi_square_per_dof: f64,
}

fn validate_square_covariance(
    covariance: &[Vec<f64>],
    dimension: usize,
) -> Result<(), CovarianceError> {
    if covariance.len() != dimension || covariance.iter().any(|row| row.len() != dimension) {
        return Err(CovarianceError::DimensionMismatch);
    }
    if covariance.iter().flatten().any(|value| !value.is_finite()) {
        return Err(CovarianceError::NonFinite);
    }
    for i in 0..dimension {
        for j in (i + 1)..dimension {
            let scale = covariance[i][j]
                .abs()
                .max(covariance[j][i].abs())
                .max(1.0);
            if (covariance[i][j] - covariance[j][i]).abs() > 1.0e-12 * scale {
                return Err(CovarianceError::NonSymmetric);
            }
        }
    }
    Ok(())
}

fn cholesky(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, CovarianceError> {
    let n = matrix.len();
    validate_square_covariance(matrix, n)?;
    let mut lower = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut residual = matrix[i][j];
            for k in 0..j {
                residual -= lower[i][k] * lower[j][k];
            }
            if i == j {
                if !residual.is_finite() || residual <= 1.0e-14 {
                    return Err(CovarianceError::NotPositiveDefinite);
                }
                lower[i][j] = residual.sqrt();
            } else {
                lower[i][j] = residual / lower[j][j];
            }
        }
    }
    Ok(lower)
}

fn solve_spd(matrix: &[Vec<f64>], rhs: &[f64]) -> Result<Vec<f64>, CovarianceError> {
    if matrix.len() != rhs.len() {
        return Err(CovarianceError::DimensionMismatch);
    }
    if rhs.iter().any(|value| !value.is_finite()) {
        return Err(CovarianceError::NonFinite);
    }
    let lower = cholesky(matrix)?;
    let n = rhs.len();
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut residual = rhs[i];
        for k in 0..i {
            residual -= lower[i][k] * y[k];
        }
        y[i] = residual / lower[i][i];
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut residual = y[i];
        for k in (i + 1)..n {
            residual -= lower[k][i] * x[k];
        }
        x[i] = residual / lower[i][i];
    }
    if x.iter().any(|value| !value.is_finite()) {
        return Err(CovarianceError::NonFinite);
    }
    Ok(x)
}

fn invert_spd(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, CovarianceError> {
    let n = matrix.len();
    let mut inverse = vec![vec![0.0; n]; n];
    for column in 0..n {
        let mut unit = vec![0.0; n];
        unit[column] = 1.0;
        let solved = solve_spd(matrix, &unit)?;
        for row in 0..n {
            inverse[row][column] = solved[row];
        }
    }
    Ok(inverse)
}

/// Fit an explicitly supplied linear model `y = X beta` with full covariance.
///
/// `design[row][column]` is the value of one caller-declared basis function for
/// one observation. The function refuses underdetermined/rank-deficient normal
/// systems and performs no regularization or basis selection.
pub fn correlated_linear_fit(
    values: &[f64],
    covariance: &[Vec<f64>],
    design: &[Vec<f64>],
) -> Result<CorrelatedLinearFit, CovarianceError> {
    let observation_count = values.len();
    if observation_count == 0 {
        return Err(CovarianceError::EmptyVector);
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(CovarianceError::NonFinite);
    }
    validate_square_covariance(covariance, observation_count)?;
    if design.len() != observation_count || design.is_empty() {
        return Err(CovarianceError::DimensionMismatch);
    }
    let parameter_count = design[0].len();
    if parameter_count == 0 || design.iter().any(|row| row.len() != parameter_count) {
        return Err(CovarianceError::DimensionMismatch);
    }
    if design.iter().flatten().any(|value| !value.is_finite()) {
        return Err(CovarianceError::NonFinite);
    }
    if observation_count <= parameter_count {
        return Err(CovarianceError::TooFewSamples);
    }

    let cinv_values = solve_spd(covariance, values)?;
    let mut cinv_columns = Vec::with_capacity(parameter_count);
    for parameter in 0..parameter_count {
        let column = design
            .iter()
            .map(|row| row[parameter])
            .collect::<Vec<_>>();
        cinv_columns.push(solve_spd(covariance, &column)?);
    }

    let mut normal = vec![vec![0.0; parameter_count]; parameter_count];
    let mut rhs = vec![0.0; parameter_count];
    for a in 0..parameter_count {
        rhs[a] = design
            .iter()
            .zip(&cinv_values)
            .map(|(row, weighted)| row[a] * weighted)
            .sum();
        for b in 0..parameter_count {
            normal[a][b] = design
                .iter()
                .zip(&cinv_columns[b])
                .map(|(row, weighted)| row[a] * weighted)
                .sum();
        }
    }

    // Roundoff can make X^T C^-1 X microscopically asymmetric even though the
    // exact matrix is symmetric. Symmetrize only that arithmetic noise before
    // the fail-closed SPD factorization.
    for i in 0..parameter_count {
        for j in (i + 1)..parameter_count {
            let average = 0.5 * (normal[i][j] + normal[j][i]);
            normal[i][j] = average;
            normal[j][i] = average;
        }
    }

    let parameters = solve_spd(&normal, &rhs)?;
    let parameter_covariance = invert_spd(&normal)?;

    let mut residual = vec![0.0; observation_count];
    for (row_index, row) in design.iter().enumerate() {
        let prediction = row
            .iter()
            .zip(&parameters)
            .map(|(basis, parameter)| basis * parameter)
            .sum::<f64>();
        residual[row_index] = values[row_index] - prediction;
    }
    let weighted_residual = solve_spd(covariance, &residual)?;
    let chi_square = residual
        .iter()
        .zip(&weighted_residual)
        .map(|(a, b)| a * b)
        .sum::<f64>();
    if !chi_square.is_finite() || chi_square < -1.0e-12 {
        return Err(CovarianceError::NonFinite);
    }
    let chi_square = chi_square.max(0.0);
    let degrees_of_freedom = observation_count - parameter_count;

    Ok(CorrelatedLinearFit {
        parameters,
        parameter_covariance,
        chi_square,
        degrees_of_freedom,
        chi_square_per_dof: chi_square / degrees_of_freedom as f64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice_covariance::correlated_constant_fit;

    #[test]
    fn one_column_model_matches_existing_constant_gls() {
        let values = [1.0, 2.0, 3.0];
        let covariance = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 4.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let design = vec![vec![1.0], vec![1.0], vec![1.0]];
        let generic = correlated_linear_fit(&values, &covariance, &design).unwrap();
        let legacy = correlated_constant_fit(&values, &covariance).unwrap();
        assert!((generic.parameters[0] - legacy.value).abs() < 1.0e-14);
        assert!((generic.parameter_covariance[0][0].sqrt() - legacy.sigma).abs() < 1.0e-14);
        assert!((generic.chi_square - legacy.chi_square).abs() < 1.0e-14);
    }

    #[test]
    fn exact_two_parameter_line_is_recovered() {
        let values = [1.0, 3.0, 5.0, 7.0];
        let covariance = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let design = vec![
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![1.0, 2.0],
            vec![1.0, 3.0],
        ];
        let fit = correlated_linear_fit(&values, &covariance, &design).unwrap();
        assert!((fit.parameters[0] - 1.0).abs() < 1.0e-14);
        assert!((fit.parameters[1] - 2.0).abs() < 1.0e-14);
        assert!(fit.chi_square < 1.0e-24);
    }

    #[test]
    fn rank_deficient_design_fails_closed() {
        let values = [1.0, 2.0, 3.0];
        let covariance = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let design = vec![vec![1.0, 2.0], vec![1.0, 2.0], vec![1.0, 2.0]];
        assert_eq!(
            correlated_linear_fit(&values, &covariance, &design),
            Err(CovarianceError::NotPositiveDefinite)
        );
    }

    #[test]
    fn underdetermined_model_fails_closed() {
        let values = [1.0, 2.0];
        let covariance = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let design = vec![vec![1.0, 0.0], vec![1.0, 1.0]];
        assert_eq!(
            correlated_linear_fit(&values, &covariance, &design),
            Err(CovarianceError::TooFewSamples)
        );
    }
}
