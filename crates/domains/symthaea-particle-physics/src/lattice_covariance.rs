// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Covariance-aware statistics for lattice observables.
//!
//! Correlator times, fit-window points and derived observables are generally
//! correlated.  Treating them as independent can understate uncertainty.  This
//! module provides sample covariance estimation and a generalized least-squares
//! constant fit using an explicitly supplied positive-definite covariance
//! matrix.  It does not choose fit windows or manufacture a covariance model.

#[derive(Debug, Clone, PartialEq)]
pub struct CorrelatedConstantFit {
    pub value: f64,
    pub sigma: f64,
    pub chi_square: f64,
    pub degrees_of_freedom: usize,
    pub chi_square_per_dof: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CovarianceError {
    TooFewSamples,
    EmptyVector,
    DimensionMismatch,
    NonFinite,
    NonSymmetric,
    NotPositiveDefinite,
}

/// Estimate the unbiased covariance matrix across vector-valued samples.
///
/// Samples should already represent the statistically appropriate units
/// (for example independent blocks or bootstrap/jackknife replicates). This
/// helper does not assert that raw Markov-chain samples are independent.
pub fn sample_covariance(samples: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, CovarianceError> {
    if samples.len() < 2 {
        return Err(CovarianceError::TooFewSamples);
    }
    let width = samples[0].len();
    if width == 0 {
        return Err(CovarianceError::EmptyVector);
    }
    if samples.iter().any(|sample| sample.len() != width) {
        return Err(CovarianceError::DimensionMismatch);
    }
    if samples.iter().flatten().any(|value| !value.is_finite()) {
        return Err(CovarianceError::NonFinite);
    }

    let count = samples.len() as f64;
    let mut mean = vec![0.0; width];
    for sample in samples {
        for (index, value) in sample.iter().enumerate() {
            mean[index] += value;
        }
    }
    for value in &mut mean {
        *value /= count;
    }

    let mut covariance = vec![vec![0.0; width]; width];
    for sample in samples {
        for i in 0..width {
            let di = sample[i] - mean[i];
            for j in 0..width {
                covariance[i][j] += di * (sample[j] - mean[j]);
            }
        }
    }
    let denominator = (samples.len() - 1) as f64;
    for row in &mut covariance {
        for value in row {
            *value /= denominator;
        }
    }
    Ok(covariance)
}

fn validate_covariance(covariance: &[Vec<f64>], dimension: usize) -> Result<(), CovarianceError> {
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

fn cholesky(covariance: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, CovarianceError> {
    let n = covariance.len();
    let mut lower = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut residual = covariance[i][j];
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

fn solve_spd(covariance: &[Vec<f64>], rhs: &[f64]) -> Result<Vec<f64>, CovarianceError> {
    let n = rhs.len();
    validate_covariance(covariance, n)?;
    let lower = cholesky(covariance)?;

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
    Ok(x)
}

/// Generalized least-squares fit to one constant using a full covariance matrix.
///
/// `C` must be symmetric positive definite. Near-singular covariance matrices
/// are rejected rather than silently regularized; any shrinkage or SVD cutoff
/// must therefore be an explicit upstream analysis decision with provenance.
pub fn correlated_constant_fit(
    values: &[f64],
    covariance: &[Vec<f64>],
) -> Result<CorrelatedConstantFit, CovarianceError> {
    let n = values.len();
    if n == 0 {
        return Err(CovarianceError::EmptyVector);
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(CovarianceError::NonFinite);
    }
    validate_covariance(covariance, n)?;
    if n < 2 {
        return Err(CovarianceError::TooFewSamples);
    }

    let ones = vec![1.0; n];
    let cinv_ones = solve_spd(covariance, &ones)?;
    let cinv_values = solve_spd(covariance, values)?;
    let denominator: f64 = cinv_ones.iter().sum();
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(CovarianceError::NotPositiveDefinite);
    }
    let numerator: f64 = cinv_values.iter().sum();
    let value = numerator / denominator;
    let sigma = (1.0 / denominator).sqrt();

    let residual = values.iter().map(|sample| sample - value).collect::<Vec<_>>();
    let weighted_residual = solve_spd(covariance, &residual)?;
    let chi_square = residual
        .iter()
        .zip(&weighted_residual)
        .map(|(a, b)| a * b)
        .sum::<f64>();
    let degrees_of_freedom = n - 1;

    Ok(CorrelatedConstantFit {
        value,
        sigma,
        chi_square,
        degrees_of_freedom,
        chi_square_per_dof: chi_square / degrees_of_freedom as f64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_covariance_recovers_exact_linear_fixture() {
        let samples = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]];
        let covariance = sample_covariance(&samples).unwrap();
        assert!((covariance[0][0] - 1.0).abs() < 1.0e-14);
        assert!((covariance[0][1] - 2.0).abs() < 1.0e-14);
        assert!((covariance[1][1] - 4.0).abs() < 1.0e-14);
    }

    #[test]
    fn diagonal_covariance_matches_inverse_variance_constant_fit() {
        let values = [1.0, 2.0, 3.0];
        let covariance = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 4.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let fit = correlated_constant_fit(&values, &covariance).unwrap();
        assert!((fit.value - 2.0).abs() < 1.0e-14);
        assert!((fit.sigma - 2.0 / 3.0).abs() < 1.0e-14);
        assert!((fit.chi_square - 2.0).abs() < 1.0e-14);
        assert!((fit.chi_square_per_dof - 1.0).abs() < 1.0e-14);
    }

    #[test]
    fn positive_correlation_increases_constant_uncertainty() {
        let values = [2.0, 2.0];
        let independent = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let correlated = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
        let independent_fit = correlated_constant_fit(&values, &independent).unwrap();
        let correlated_fit = correlated_constant_fit(&values, &correlated).unwrap();
        assert!(correlated_fit.sigma > independent_fit.sigma);
        assert!((correlated_fit.value - 2.0).abs() < 1.0e-14);
    }

    #[test]
    fn singular_covariance_fails_closed() {
        let values = [1.0, 2.0];
        let singular = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        assert_eq!(
            correlated_constant_fit(&values, &singular),
            Err(CovarianceError::NotPositiveDefinite)
        );
    }
}
