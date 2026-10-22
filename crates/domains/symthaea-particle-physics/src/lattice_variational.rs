// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Variational spectroscopy primitives for lattice field theory.
//!
//! This module solves the symmetric generalized eigenvalue problem
//!
//! `C(t) v_n = lambda_n(t, t0) C(t0) v_n`
//!
//! for a positive-definite reference correlation matrix `C(t0)`.  It is an
//! analysis primitive only: it does not choose operators, select `t0`, identify
//! plateaus, generate gauge ensembles, or establish a physical state by itself.

#[derive(Debug, Clone, PartialEq)]
pub enum VariationalError {
    EmptyMatrix,
    NonSquare,
    DimensionMismatch,
    NonFinite,
    NonSymmetric,
    ReferenceNotPositiveDefinite,
    JacobiDidNotConverge,
    InvalidTimeStep,
    NonPositivePrincipalCorrelator,
}

fn validate_symmetric(matrix: &[Vec<f64>]) -> Result<usize, VariationalError> {
    let n = matrix.len();
    if n == 0 {
        return Err(VariationalError::EmptyMatrix);
    }
    if matrix.iter().any(|row| row.len() != n) {
        return Err(VariationalError::NonSquare);
    }
    if matrix.iter().flatten().any(|value| !value.is_finite()) {
        return Err(VariationalError::NonFinite);
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let scale = matrix[i][j].abs().max(matrix[j][i].abs()).max(1.0);
            if (matrix[i][j] - matrix[j][i]).abs() > 1.0e-12 * scale {
                return Err(VariationalError::NonSymmetric);
            }
        }
    }
    Ok(n)
}

fn transpose(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = matrix.len();
    let mut out = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            out[j][i] = matrix[i][j];
        }
    }
    out
}

fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut out = vec![vec![0.0; n]; n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i][k];
            for j in 0..n {
                out[i][j] += aik * b[k][j];
            }
        }
    }
    out
}

fn cholesky(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, VariationalError> {
    let n = matrix.len();
    let mut lower = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut residual = matrix[i][j];
            for k in 0..j {
                residual -= lower[i][k] * lower[j][k];
            }
            if i == j {
                if !residual.is_finite() || residual <= 1.0e-14 {
                    return Err(VariationalError::ReferenceNotPositiveDefinite);
                }
                lower[i][j] = residual.sqrt();
            } else {
                lower[i][j] = residual / lower[j][j];
            }
        }
    }
    Ok(lower)
}

fn inverse_lower(lower: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = lower.len();
    let mut inverse = vec![vec![0.0; n]; n];
    for column in 0..n {
        let mut solution = vec![0.0; n];
        for i in 0..n {
            let mut rhs = if i == column { 1.0 } else { 0.0 };
            for k in 0..i {
                rhs -= lower[i][k] * solution[k];
            }
            solution[i] = rhs / lower[i][i];
        }
        for i in 0..n {
            inverse[i][column] = solution[i];
        }
    }
    inverse
}

fn jacobi_eigenvalues(mut matrix: Vec<Vec<f64>>) -> Result<Vec<f64>, VariationalError> {
    let n = matrix.len();
    if n == 1 {
        return Ok(vec![matrix[0][0]]);
    }

    let max_iterations = 100 * n * n;
    let tolerance = 1.0e-13;

    for _ in 0..max_iterations {
        let mut p = 0;
        let mut q = 1;
        let mut max_offdiag = matrix[p][q].abs();
        for i in 0..n {
            for j in (i + 1)..n {
                let candidate = matrix[i][j].abs();
                if candidate > max_offdiag {
                    max_offdiag = candidate;
                    p = i;
                    q = j;
                }
            }
        }

        if max_offdiag < tolerance {
            let mut eigenvalues = (0..n).map(|i| matrix[i][i]).collect::<Vec<_>>();
            eigenvalues.sort_by(|a, b| b.total_cmp(a));
            return Ok(eigenvalues);
        }

        let app = matrix[p][p];
        let aqq = matrix[q][q];
        let apq = matrix[p][q];
        let theta = 0.5 * (2.0 * apq).atan2(aqq - app);
        let c = theta.cos();
        let s = theta.sin();

        for k in 0..n {
            if k == p || k == q {
                continue;
            }
            let akp = matrix[k][p];
            let akq = matrix[k][q];
            let new_kp = c * akp - s * akq;
            let new_kq = s * akp + c * akq;
            matrix[k][p] = new_kp;
            matrix[p][k] = new_kp;
            matrix[k][q] = new_kq;
            matrix[q][k] = new_kq;
        }

        matrix[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        matrix[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        matrix[p][q] = 0.0;
        matrix[q][p] = 0.0;
    }

    Err(VariationalError::JacobiDidNotConverge)
}

/// Solve the symmetric GEVP and return principal correlators in descending order.
///
/// The reference matrix must be symmetric positive definite.  The transform
/// `L^-1 C(t) L^-T`, where `C(t0) = L L^T`, reduces the problem to an ordinary
/// real-symmetric eigenproblem.  No state identity is inferred from ordering.
pub fn principal_correlators(
    correlation_t: &[Vec<f64>],
    correlation_t0: &[Vec<f64>],
) -> Result<Vec<f64>, VariationalError> {
    let n = validate_symmetric(correlation_t)?;
    let n0 = validate_symmetric(correlation_t0)?;
    if n != n0 {
        return Err(VariationalError::DimensionMismatch);
    }

    let lower = cholesky(correlation_t0)?;
    let lower_inverse = inverse_lower(&lower);
    let transformed = mat_mul(
        &mat_mul(&lower_inverse, correlation_t),
        &transpose(&lower_inverse),
    );

    // Remove only roundoff-level asymmetry introduced by floating point matrix
    // multiplication. A materially non-symmetric input was already rejected.
    let mut symmetric = transformed;
    for i in 0..n {
        for j in (i + 1)..n {
            let mean = 0.5 * (symmetric[i][j] + symmetric[j][i]);
            symmetric[i][j] = mean;
            symmetric[j][i] = mean;
        }
    }

    jacobi_eigenvalues(symmetric)
}

/// Convert two matched principal-correlator spectra into effective energies.
///
/// This assumes the caller has already established state matching between the
/// two times. Sorting eigenvalues independently is not, by itself, a state-
/// tracking theorem near crossings or degeneracies.
pub fn principal_effective_energies(
    lambda_t: &[f64],
    lambda_t_plus_dt: &[f64],
    delta_t: f64,
) -> Result<Vec<f64>, VariationalError> {
    if !delta_t.is_finite() || delta_t <= 0.0 {
        return Err(VariationalError::InvalidTimeStep);
    }
    if lambda_t.len() != lambda_t_plus_dt.len() {
        return Err(VariationalError::DimensionMismatch);
    }

    lambda_t
        .iter()
        .zip(lambda_t_plus_dt)
        .map(|(&current, &next)| {
            if !current.is_finite() || !next.is_finite() {
                return Err(VariationalError::NonFinite);
            }
            if current <= 0.0 || next <= 0.0 {
                return Err(VariationalError::NonPositivePrincipalCorrelator);
            }
            Ok((current / next).ln() / delta_t)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_state_correlator(t: f64, energies: [f64; 2]) -> Vec<Vec<f64>> {
        let z = [[1.0, 0.35], [0.2, 1.1]];
        let exponentials = [(-energies[0] * t).exp(), (-energies[1] * t).exp()];
        let mut out = vec![vec![0.0; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                for state in 0..2 {
                    out[i][j] += z[i][state] * exponentials[state] * z[j][state];
                }
            }
        }
        out
    }

    #[test]
    fn gevp_recovers_two_exact_principal_correlators() {
        let energies = [0.4, 1.0];
        let t0 = 1.0;
        let t = 3.0;
        let values = principal_correlators(
            &two_state_correlator(t, energies),
            &two_state_correlator(t0, energies),
        )
        .unwrap();

        let expected = [
            (-energies[0] * (t - t0)).exp(),
            (-energies[1] * (t - t0)).exp(),
        ];
        assert!((values[0] - expected[0]).abs() < 1.0e-10);
        assert!((values[1] - expected[1]).abs() < 1.0e-10);
    }

    #[test]
    fn principal_effective_energy_recovers_input_energies() {
        let energies = [0.4, 1.0];
        let t0 = 1.0;
        let t = 2.0;
        let dt = 1.0;
        let lambda_t = principal_correlators(
            &two_state_correlator(t, energies),
            &two_state_correlator(t0, energies),
        )
        .unwrap();
        let lambda_next = principal_correlators(
            &two_state_correlator(t + dt, energies),
            &two_state_correlator(t0, energies),
        )
        .unwrap();
        let extracted = principal_effective_energies(&lambda_t, &lambda_next, dt).unwrap();
        assert!((extracted[0] - energies[0]).abs() < 1.0e-10);
        assert!((extracted[1] - energies[1]).abs() < 1.0e-10);
    }

    #[test]
    fn singular_reference_is_rejected() {
        let singular = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        let result = principal_correlators(&singular, &singular);
        assert_eq!(result, Err(VariationalError::ReferenceNotPositiveDefinite));
    }

    #[test]
    fn non_symmetric_input_is_rejected() {
        let bad = vec![vec![1.0, 0.1], vec![0.2, 1.0]];
        let identity = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert_eq!(
            principal_correlators(&bad, &identity),
            Err(VariationalError::NonSymmetric)
        );
    }
}
