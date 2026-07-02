// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Generalized eigenvalue solver for the Roothaan-Hall equation: FC = SCε.
//!
//! Uses **canonical orthogonalization** (not Cholesky) to transform to a standard
//! eigenvalue problem. This is critical because:
//!
//! 1. Cholesky requires S to be strictly positive definite, but basis sets with
//!    near-linear dependence can have eigenvalues of S approaching zero.
//! 2. Canonical orthogonalization diagonalizes S, discards eigenvectors below a
//!    threshold (default 1e-6), and builds the transformation matrix X from the
//!    survivors. This is numerically stable even with large basis sets.
//!
//! The procedure:
//! 1. Diagonalize S: S = UsU^T (eigendecompose)
//! 2. Discard eigenvalues s_i < threshold
//! 3. Form X = U * s^{-1/2} (surviving columns only)
//! 4. Transform: F' = X^T F X
//! 5. Solve standard eigenproblem: F'C' = C'ε
//! 6. Back-transform: C = X C'
//!
//! Reference: Szabo & Ostlund, "Modern Quantum Chemistry", Section 3.4.5.

use crate::constants::CANONICAL_ORTH_THRESHOLD;

/// Result of the generalized eigenvalue problem FC = SCε.
#[derive(Debug, Clone)]
pub struct GeneralizedEigenResult {
    /// Eigenvalues (orbital energies) in ascending order.
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors (MO coefficients) as column-major: C[μ, i] = coefficients[μ * n_mo + i]
    /// where μ indexes AOs and i indexes MOs.
    pub coefficients: Vec<f64>,
    /// Number of basis functions retained after linear dependence removal.
    pub n_independent: usize,
    /// Number of basis functions discarded.
    pub n_discarded: usize,
    /// The orthogonalization matrix X (n_basis × n_independent).
    pub orthogonalization_matrix: Vec<f64>,
}

/// Compute the canonical orthogonalization matrix X from overlap matrix S.
///
/// Returns (X, n_independent, n_discarded) where X is n_basis × n_independent.
pub fn canonical_orthogonalization(s_matrix: &[f64], n: usize) -> (Vec<f64>, usize, usize) {
    canonical_orthogonalization_with_threshold(s_matrix, n, CANONICAL_ORTH_THRESHOLD)
}

/// Compute canonical orthogonalization with custom threshold.
pub fn canonical_orthogonalization_with_threshold(
    s_matrix: &[f64],
    n: usize,
    threshold: f64,
) -> (Vec<f64>, usize, usize) {
    // Step 1: Diagonalize S using Jacobi eigendecomposition
    let (eigenvalues, eigenvectors) = symmetric_eigen(s_matrix, n);

    // Step 2: Find surviving eigenvectors (eigenvalue >= threshold)
    let mut survivors: Vec<usize> = Vec::new();
    for (i, &ev) in eigenvalues.iter().enumerate() {
        if ev >= threshold {
            survivors.push(i);
        }
    }

    let n_independent = survivors.len();
    let n_discarded = n - n_independent;

    // Step 3: Build X = U * s^{-1/2} for surviving columns
    // X[μ, k] = U[μ, survivors[k]] / sqrt(eigenvalues[survivors[k]])
    let mut x = vec![0.0; n * n_independent];
    for (k, &idx) in survivors.iter().enumerate() {
        let inv_sqrt = 1.0 / eigenvalues[idx].sqrt();
        for mu in 0..n {
            x[mu * n_independent + k] = eigenvectors[mu * n + idx] * inv_sqrt;
        }
    }

    (x, n_independent, n_discarded)
}

/// Solve the generalized eigenvalue problem FC = SCε.
///
/// Takes the Fock matrix F and the pre-computed orthogonalization matrix X.
/// Returns eigenvalues and eigenvectors in the original AO basis.
pub fn solve_generalized_eigen(
    f_matrix: &[f64],
    x_matrix: &[f64],
    n_basis: usize,
    n_independent: usize,
) -> GeneralizedEigenResult {
    // Step 1: F' = X^T F X (n_independent × n_independent)
    let f_prime = xtax(x_matrix, f_matrix, n_basis, n_independent);

    // Step 2: Diagonalize F'
    let (eigenvalues, eigvecs_prime) = symmetric_eigen(&f_prime, n_independent);

    // Step 3: Sort by eigenvalue (ascending)
    let mut indices: Vec<usize> = (0..n_independent).collect();
    indices.sort_by(|&a, &b| eigenvalues[a].total_cmp(&eigenvalues[b]));

    let sorted_eigenvalues: Vec<f64> = indices.iter().map(|&i| eigenvalues[i]).collect();

    // Step 4: Back-transform C = X * C'
    // C[μ, i] = Σ_k X[μ, k] * C'[k, sorted_i]
    let mut coefficients = vec![0.0; n_basis * n_independent];
    for (i, &orig_i) in indices.iter().enumerate() {
        for mu in 0..n_basis {
            let mut sum = 0.0;
            for k in 0..n_independent {
                sum += x_matrix[mu * n_independent + k] * eigvecs_prime[k * n_independent + orig_i];
            }
            coefficients[mu * n_independent + i] = sum;
        }
    }

    GeneralizedEigenResult {
        eigenvalues: sorted_eigenvalues,
        coefficients,
        n_independent,
        n_discarded: n_basis - n_independent,
        orthogonalization_matrix: x_matrix.to_vec(),
    }
}

// ── Internal linear algebra (pure f64, no HDC overhead for inner loops) ─────

/// Symmetric eigendecomposition via Jacobi rotations.
/// Returns (eigenvalues, eigenvectors) where eigenvectors are column-major.
fn symmetric_eigen(matrix: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = matrix.to_vec();
    let mut v = vec![0.0; n * n];
    // Initialize V = I
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    let max_iter = 100 * n * n;
    let tol = 1e-14;

    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                if a[i * n + j].abs() > max_val {
                    max_val = a[i * n + j].abs();
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < tol {
            break;
        }

        // Compute rotation angle
        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];

        let theta = if (app - aqq).abs() < 1e-30 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Apply Jacobi rotation to A
        let mut new_a = a.clone();
        for i in 0..n {
            if i != p && i != q {
                let aip = a[i * n + p];
                let aiq = a[i * n + q];
                new_a[i * n + p] = c * aip + s * aiq;
                new_a[p * n + i] = new_a[i * n + p];
                new_a[i * n + q] = -s * aip + c * aiq;
                new_a[q * n + i] = new_a[i * n + q];
            }
        }
        new_a[p * n + p] = c * c * app + 2.0 * c * s * apq + s * s * aqq;
        new_a[q * n + q] = s * s * app - 2.0 * c * s * apq + c * c * aqq;
        new_a[p * n + q] = 0.0;
        new_a[q * n + p] = 0.0;
        a = new_a;

        // Apply rotation to V
        for i in 0..n {
            let vip = v[i * n + p];
            let viq = v[i * n + q];
            v[i * n + p] = c * vip + s * viq;
            v[i * n + q] = -s * vip + c * viq;
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    (eigenvalues, v)
}

/// Compute X^T * A * X where X is n×m and A is n×n, result is m×m.
fn xtax(x: &[f64], a: &[f64], n: usize, m: usize) -> Vec<f64> {
    // First: AX = A * X (n×m)
    let mut ax = vec![0.0; n * m];
    for i in 0..n {
        for j in 0..m {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[i * n + k] * x[k * m + j];
            }
            ax[i * m + j] = sum;
        }
    }

    // Then: X^T * AX (m×m)
    let mut result = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..m {
            let mut sum = 0.0;
            for k in 0..n {
                sum += x[k * m + i] * ax[k * m + j];
            }
            result[i * m + j] = sum;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symmetric_eigen_identity() {
        // Eigenvalues of identity should be all 1.0
        let n = 3;
        let mut eye = vec![0.0; n * n];
        for i in 0..n {
            eye[i * n + i] = 1.0;
        }
        let (eigenvalues, _) = symmetric_eigen(&eye, n);
        for &ev in &eigenvalues {
            assert!((ev - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_symmetric_eigen_diagonal() {
        let n = 3;
        let mut diag = vec![0.0; n * n];
        diag[0] = 3.0;
        diag[4] = 1.0;
        diag[8] = 2.0;
        let (eigenvalues, _) = symmetric_eigen(&diag, n);

        let mut sorted = eigenvalues.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        assert!((sorted[0] - 1.0).abs() < 1e-10);
        assert!((sorted[1] - 2.0).abs() < 1e-10);
        assert!((sorted[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_canonical_orthogonalization_identity() {
        // For S = I, X should be I and nothing discarded
        let n = 3;
        let mut eye = vec![0.0; n * n];
        for i in 0..n {
            eye[i * n + i] = 1.0;
        }
        let (x, n_ind, n_disc) = canonical_orthogonalization(&eye, n);
        assert_eq!(n_ind, 3);
        assert_eq!(n_disc, 0);
        assert_eq!(x.len(), 9);
    }

    #[test]
    fn test_canonical_orthogonalization_discards_linear_dependence() {
        // S with one near-zero eigenvalue
        let n = 2;
        // S = [[1, 0.9999999], [0.9999999, 1]] → eigenvalues ~2.0 and ~1e-7
        let s = vec![1.0, 0.9999999, 0.9999999, 1.0];
        let (x, n_ind, n_disc) = canonical_orthogonalization(&s, n);

        // Should discard the near-zero eigenvector
        assert_eq!(n_ind, 1, "Should keep 1 of 2 functions");
        assert_eq!(n_disc, 1, "Should discard 1 function");
        assert_eq!(x.len(), 2); // n * n_independent = 2 * 1
    }

    #[test]
    fn test_generalized_eigen_known_problem() {
        // Simple 2x2: F = [[2, 1], [1, 2]], S = [[1, 0], [0, 1]]
        // Standard eigenvalues: 1 and 3
        let n = 2;
        let f = vec![2.0, 1.0, 1.0, 2.0];
        let s = vec![1.0, 0.0, 0.0, 1.0];

        let (x, n_ind, _) = canonical_orthogonalization(&s, n);
        let result = solve_generalized_eigen(&f, &x, n, n_ind);

        assert_eq!(result.eigenvalues.len(), 2);
        let mut evals = result.eigenvalues.clone();
        evals.sort_by(|a, b| a.total_cmp(b));
        assert!(
            (evals[0] - 1.0).abs() < 1e-8,
            "First eigenvalue: {}, expected 1.0",
            evals[0]
        );
        assert!(
            (evals[1] - 3.0).abs() < 1e-8,
            "Second eigenvalue: {}, expected 3.0",
            evals[1]
        );
    }

    #[test]
    fn test_generalized_eigen_with_overlap() {
        // F = [[1, 0.5], [0.5, 2]], S = [[1, 0.3], [0.3, 1]]
        // Reference: eigenvalues from Wolfram Alpha
        let n = 2;
        let f = vec![1.0, 0.5, 0.5, 2.0];
        let s = vec![1.0, 0.3, 0.3, 1.0];

        let (x, n_ind, _) = canonical_orthogonalization(&s, n);
        let result = solve_generalized_eigen(&f, &x, n, n_ind);

        assert_eq!(result.n_independent, 2);
        assert_eq!(result.n_discarded, 0);

        // Verify FC = SCε by checking residuals
        for i in 0..n_ind {
            let eps = result.eigenvalues[i];
            for mu in 0..n {
                let mut fc = 0.0;
                let mut sc = 0.0;
                for nu in 0..n {
                    fc += f[mu * n + nu] * result.coefficients[nu * n_ind + i];
                    sc += s[mu * n + nu] * result.coefficients[nu * n_ind + i];
                }
                let residual = (fc - eps * sc).abs();
                assert!(
                    residual < 1e-8,
                    "Residual at ({}, {}) = {} (eigenvalue={})",
                    mu,
                    i,
                    residual,
                    eps,
                );
            }
        }
    }
}
