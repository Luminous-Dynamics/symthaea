// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! LU decomposition (Doolittle, partial pivoting) and the operations built on
//! it: determinant, linear solve, and inverse. This is the shared home for the
//! Gaussian elimination that domain crates previously reinvented.

use crate::matrix::Matrix;

/// An LU decomposition `PA = LU` of a square matrix: `lu` packs `L` (unit
/// diagonal, below) and `U` (on and above), `piv` is the row permutation, and
/// `sign` is the permutation's sign (for the determinant).
pub struct Lu {
    lu: Vec<Vec<f64>>,
    piv: Vec<usize>,
    sign: f64,
    n: usize,
}

/// Decompose a square matrix with partial pivoting. `None` if not square or
/// singular.
pub fn lu_decompose(m: &Matrix) -> Option<Lu> {
    if !m.is_square() {
        return None;
    }
    let n = m.rows;
    let mut a = m.to_rows();
    let mut piv: Vec<usize> = (0..n).collect();
    let mut sign = 1.0;
    for col in 0..n {
        // Partial pivot: largest magnitude in this column, on/below the diagonal.
        let mut p = col;
        for r in (col + 1)..n {
            if a[r][col].abs() > a[p][col].abs() {
                p = r;
            }
        }
        if a[p][col].abs() < 1e-14 {
            return None; // singular
        }
        if p != col {
            a.swap(p, col);
            piv.swap(p, col);
            sign = -sign;
        }
        for r in (col + 1)..n {
            let factor = a[r][col] / a[col][col];
            a[r][col] = factor; // store the multiplier (L)
            for c in (col + 1)..n {
                a[r][c] -= factor * a[col][c];
            }
        }
    }
    Some(Lu {
        lu: a,
        piv,
        sign,
        n,
    })
}

impl Lu {
    /// Solve `A x = b` using the decomposition.
    pub fn solve(&self, b: &[f64]) -> Option<Vec<f64>> {
        if b.len() != self.n {
            return None;
        }
        // Apply the row permutation.
        let mut y: Vec<f64> = self.piv.iter().map(|&p| b[p]).collect();
        // Forward substitution (L, unit diagonal).
        for i in 0..self.n {
            for j in 0..i {
                let v = y[j];
                y[i] -= self.lu[i][j] * v;
            }
        }
        // Back substitution (U).
        for i in (0..self.n).rev() {
            for j in (i + 1)..self.n {
                let v = y[j];
                y[i] -= self.lu[i][j] * v;
            }
            y[i] /= self.lu[i][i];
        }
        Some(y)
    }

    /// The determinant: `sign · ∏ U_ii`.
    pub fn determinant(&self) -> f64 {
        (0..self.n).fold(self.sign, |acc, i| acc * self.lu[i][i])
    }
}

/// Determinant of a square matrix (`0.0` if singular, `None` if not square).
pub fn determinant(m: &Matrix) -> Option<f64> {
    if !m.is_square() {
        return None;
    }
    Some(lu_decompose(m).map(|lu| lu.determinant()).unwrap_or(0.0))
}

/// Solve `A x = b`. `None` if `A` is not square, singular, or `b` mismatches.
pub fn solve(m: &Matrix, b: &[f64]) -> Option<Vec<f64>> {
    lu_decompose(m)?.solve(b)
}

/// The inverse of a square matrix (`None` if not square or singular).
pub fn inverse(m: &Matrix) -> Option<Matrix> {
    let lu = lu_decompose(m)?;
    let n = m.rows;
    let mut inv = Matrix::zeros(n, n);
    for col in 0..n {
        let mut e = vec![0.0; n];
        e[col] = 1.0;
        let x = lu.solve(&e)?;
        for (row, &val) in x.iter().enumerate() {
            inv.set(row, col, val);
        }
    }
    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn determinant_values() {
        let a = Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        assert!((determinant(&a).unwrap() - (-2.0)).abs() < 1e-12);
        let d = Matrix::from_rows(vec![
            vec![2.0, 0.0, 0.0],
            vec![0.0, 3.0, 0.0],
            vec![0.0, 0.0, 4.0],
        ])
        .unwrap();
        assert!((determinant(&d).unwrap() - 24.0).abs() < 1e-12);
    }

    #[test]
    fn singular_has_zero_determinant() {
        let s = Matrix::from_rows(vec![vec![1.0, 2.0], vec![2.0, 4.0]]).unwrap();
        assert!(determinant(&s).unwrap().abs() < 1e-12);
        assert!(solve(&s, &[1.0, 2.0]).is_none());
        assert!(inverse(&s).is_none());
    }

    #[test]
    fn solve_linear_system() {
        // 2x + y = 3 ; x + 3y = 5  →  x = 0.8, y = 1.4.
        let a = Matrix::from_rows(vec![vec![2.0, 1.0], vec![1.0, 3.0]]).unwrap();
        let x = solve(&a, &[3.0, 5.0]).unwrap();
        assert!((x[0] - 0.8).abs() < 1e-12, "{x:?}");
        assert!((x[1] - 1.4).abs() < 1e-12);
    }

    #[test]
    fn inverse_times_original_is_identity() {
        let a = Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let inv = inverse(&a).unwrap();
        let prod = a.mul(&inv).unwrap();
        assert!(prod.max_abs_diff(&Matrix::identity(2)) < 1e-12);
    }
}
