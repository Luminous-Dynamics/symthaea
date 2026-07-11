// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Eigenvalues: power iteration for the dominant pair of a general matrix, and
//! the Jacobi algorithm for all eigenvalues of a symmetric matrix.

use crate::matrix::Matrix;

/// The dominant eigenvalue and its (unit) eigenvector via power iteration.
/// `None` if not square or the iteration collapses to the zero vector.
pub fn power_iteration(m: &Matrix, iterations: usize) -> Option<(f64, Vec<f64>)> {
    if !m.is_square() {
        return None;
    }
    let n = m.rows;
    let mut v = vec![1.0 / (n as f64).sqrt(); n];
    let mut eigenvalue = 0.0;
    for _ in 0..iterations {
        let mut w = m.mul_vec(&v)?;
        let norm = (w.iter().map(|x| x * x).sum::<f64>()).sqrt();
        if norm < 1e-300 {
            return None;
        }
        for x in w.iter_mut() {
            *x /= norm;
        }
        // Rayleigh quotient vᵀ M v for the eigenvalue estimate.
        let mv = m.mul_vec(&w)?;
        eigenvalue = w.iter().zip(&mv).map(|(a, b)| a * b).sum();
        v = w;
    }
    Some((eigenvalue, v))
}

/// All eigenvalues of a **symmetric** matrix via the cyclic Jacobi algorithm,
/// returned in ascending order. `None` if not square. (The matrix is assumed
/// symmetric; the lower triangle is ignored.)
pub fn jacobi_eigenvalues(m: &Matrix, sweeps: usize) -> Option<Vec<f64>> {
    if !m.is_square() {
        return None;
    }
    let n = m.rows;
    let mut a = m.to_rows();
    for _ in 0..sweeps {
        // Largest off-diagonal magnitude; stop when negligible.
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[p][q] * a[p][q];
            }
        }
        if off.sqrt() < 1e-14 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                // Jacobi rotation to zero a[p][q].
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[k][p];
                    let akq = a[k][q];
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p][k];
                    let aqk = a[q][k];
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
            }
        }
    }
    let mut eig: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    eig.sort_by(|x, y| x.partial_cmp(y).unwrap());
    Some(eig)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_iteration_dominant() {
        // [[2,1],[1,2]] has eigenvalues 3 (vector [1,1]) and 1.
        let a = Matrix::from_rows(vec![vec![2.0, 1.0], vec![1.0, 2.0]]).unwrap();
        let (lambda, v) = power_iteration(&a, 200).unwrap();
        assert!((lambda - 3.0).abs() < 1e-9, "λ={lambda}");
        // Eigenvector proportional to [1,1] → components equal in magnitude.
        assert!((v[0].abs() - v[1].abs()).abs() < 1e-9);
    }

    #[test]
    fn jacobi_all_eigenvalues() {
        // Symmetric [[2,1],[1,2]] → {1, 3}.
        let a = Matrix::from_rows(vec![vec![2.0, 1.0], vec![1.0, 2.0]]).unwrap();
        let eig = jacobi_eigenvalues(&a, 100).unwrap();
        assert!((eig[0] - 1.0).abs() < 1e-9, "{eig:?}");
        assert!((eig[1] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn jacobi_diagonal_and_trace() {
        // Eigenvalues of a diagonal matrix are its diagonal; trace is preserved.
        let a = Matrix::from_rows(vec![
            vec![4.0, 0.0, 0.0],
            vec![0.0, -1.0, 0.0],
            vec![0.0, 0.0, 2.0],
        ])
        .unwrap();
        let eig = jacobi_eigenvalues(&a, 50).unwrap();
        assert_eq!(eig.len(), 3);
        assert!((eig.iter().sum::<f64>() - 5.0).abs() < 1e-12); // trace
        assert!((eig[0] - (-1.0)).abs() < 1e-12);
    }
}
