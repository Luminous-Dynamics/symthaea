// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! DIIS (Direct Inversion in the Iterative Subspace) convergence accelerator.
//!
//! Stores recent Fock matrices and error vectors, then extrapolates an
//! optimal Fock matrix by solving a small linear system. Typically cuts
//! SCF iterations from 50-100 down to 10-20.
//!
//! Error vector: e = FPS - SPF (in the AO basis).
//!
//! Reference: Pulay, P. (1980). Chem. Phys. Lett. 73, 393.

use crate::constants::DIIS_SUBSPACE_SIZE;

/// DIIS convergence accelerator.
pub struct Diis {
    /// Stored Fock matrices
    fock_list: Vec<Vec<f64>>,
    /// Stored error vectors (flattened)
    error_list: Vec<Vec<f64>>,
    /// Maximum subspace size
    max_size: usize,
    /// Basis size
    n: usize,
}

impl Diis {
    /// Create a new DIIS with given basis size.
    pub fn new(n: usize) -> Self {
        Self {
            fock_list: Vec::new(),
            error_list: Vec::new(),
            max_size: DIIS_SUBSPACE_SIZE,
            n,
        }
    }

    /// Compute the error vector: e = FPS - SPF.
    fn compute_error(fock: &[f64], density: &[f64], overlap: &[f64], n: usize) -> Vec<f64> {
        // FP = F * P
        let fp = mat_mul(fock, density, n);
        // SP = S * P
        let sp = mat_mul(overlap, density, n);
        // FPS = FP * S
        let fps = mat_mul(&fp, overlap, n);
        // SPF = SP * F
        let spf = mat_mul(&sp, fock, n);

        // e = FPS - SPF
        fps.iter().zip(spf.iter()).map(|(&a, &b)| a - b).collect()
    }

    /// Add a new Fock matrix and compute/store the error.
    /// Returns the extrapolated Fock matrix.
    pub fn extrapolate(&mut self, fock: &[f64], density: &[f64], overlap: &[f64]) -> Vec<f64> {
        let error = Self::compute_error(fock, density, overlap, self.n);

        self.fock_list.push(fock.to_vec());
        self.error_list.push(error);

        // Trim to max size
        while self.fock_list.len() > self.max_size {
            self.fock_list.remove(0);
            self.error_list.remove(0);
        }

        let m = self.fock_list.len();
        if m < 2 {
            return fock.to_vec();
        }

        // Build the B matrix (m+1 × m+1):
        // B[i][j] = <e_i | e_j>  for i,j < m
        // B[m][j] = B[i][m] = -1
        // B[m][m] = 0
        let dim = m + 1;
        let mut b = vec![0.0; dim * dim];

        for i in 0..m {
            for j in i..m {
                let dot: f64 = self.error_list[i]
                    .iter()
                    .zip(self.error_list[j].iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                b[i * dim + j] = dot;
                b[j * dim + i] = dot;
            }
        }

        // Lagrange constraint row/column
        for i in 0..m {
            b[m * dim + i] = -1.0;
            b[i * dim + m] = -1.0;
        }
        b[m * dim + m] = 0.0;

        // RHS: [0, 0, ..., 0, -1]
        let mut rhs = vec![0.0; dim];
        rhs[m] = -1.0;

        // Solve B * c = rhs via Gaussian elimination
        let coeffs = match solve_linear_system(&b, &rhs, dim) {
            Some(c) => c,
            None => return fock.to_vec(), // Fallback: no extrapolation
        };

        // Extrapolated Fock: F_new = Σ_i c_i * F_i
        let nn = self.n * self.n;
        let mut f_new = vec![0.0; nn];
        for i in 0..m {
            let ci = coeffs[i];
            for k in 0..nn {
                f_new[k] += ci * self.fock_list[i][k];
            }
        }

        f_new
    }
}

/// Simple n×n matrix multiplication.
fn mat_mul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i * n + k];
            if aik.abs() < 1e-15 {
                continue;
            }
            for j in 0..n {
                c[i * n + j] += aik * b[k * n + j];
            }
        }
    }
    c
}

/// Solve Ax = b via Gaussian elimination with partial pivoting.
fn solve_linear_system(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut aug = vec![0.0; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = aug[col * (n + 1) + col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = aug[row * (n + 1) + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < 1e-14 {
            return None; // Singular
        }

        // Swap rows
        if max_row != col {
            for j in 0..=n {
                let tmp = aug[col * (n + 1) + j];
                aug[col * (n + 1) + j] = aug[max_row * (n + 1) + j];
                aug[max_row * (n + 1) + j] = tmp;
            }
        }

        // Eliminate below
        let pivot = aug[col * (n + 1) + col];
        for row in (col + 1)..n {
            let factor = aug[row * (n + 1) + col] / pivot;
            for j in col..=n {
                aug[row * (n + 1) + j] -= factor * aug[col * (n + 1) + j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[i * (n + 1) + n];
        for j in (i + 1)..n {
            sum -= aug[i * (n + 1) + j] * x[j];
        }
        x[i] = sum / aug[i * (n + 1) + i];
    }

    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_linear_system_2x2() {
        // [2, 1] [x]   [5]     x = 2, y = 1
        // [1, 3] [y] = [5]
        let a = vec![2.0, 1.0, 1.0, 3.0];
        let b = vec![5.0, 5.0];
        let x = solve_linear_system(&a, &b, 2).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-10);
        assert!((x[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_diis_returns_input_with_one_entry() {
        let n = 2;
        let mut diis = Diis::new(n);
        let fock = vec![1.0, 0.5, 0.5, 2.0];
        let density = vec![1.0, 0.0, 0.0, 1.0];
        let overlap = vec![1.0, 0.0, 0.0, 1.0];

        let result = diis.extrapolate(&fock, &density, &overlap);
        // With only one entry, should return the input
        for i in 0..4 {
            assert!((result[i] - fock[i]).abs() < 1e-14);
        }
    }
}
