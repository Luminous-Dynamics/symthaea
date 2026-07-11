// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! A dense, row-major `f64` matrix with the core algebraic operations.

/// A dense matrix stored row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    /// A `rows × cols` zero matrix.
    pub fn zeros(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Build from row vectors. `Err` if rows differ in length or it is empty.
    pub fn from_rows(rows: Vec<Vec<f64>>) -> Result<Matrix, String> {
        if rows.is_empty() || rows[0].is_empty() {
            return Err("empty matrix".to_string());
        }
        let cols = rows[0].len();
        if rows.iter().any(|r| r.len() != cols) {
            return Err("ragged rows".to_string());
        }
        let r = rows.len();
        let data = rows.into_iter().flatten().collect();
        Ok(Matrix {
            rows: r,
            cols,
            data,
        })
    }

    /// The `n × n` identity.
    pub fn identity(n: usize) -> Matrix {
        let mut m = Matrix::zeros(n, n);
        for i in 0..n {
            m.set(i, i, 1.0);
        }
        m
    }

    /// Element `(i, j)`.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    /// Set element `(i, j)`.
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.cols + j] = v;
    }

    /// Whether the matrix is square.
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// The transpose.
    pub fn transpose(&self) -> Matrix {
        let mut t = Matrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                t.set(j, i, self.get(i, j));
            }
        }
        t
    }

    /// Matrix product `self · other`; `None` on a dimension mismatch.
    pub fn mul(&self, other: &Matrix) -> Option<Matrix> {
        if self.cols != other.rows {
            return None;
        }
        let mut out = Matrix::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.get(i, k);
                if a == 0.0 {
                    continue;
                }
                for j in 0..other.cols {
                    let v = out.get(i, j) + a * other.get(k, j);
                    out.set(i, j, v);
                }
            }
        }
        Some(out)
    }

    /// Matrix–vector product `self · v`; `None` on a dimension mismatch.
    pub fn mul_vec(&self, v: &[f64]) -> Option<Vec<f64>> {
        if self.cols != v.len() {
            return None;
        }
        Some(
            (0..self.rows)
                .map(|i| (0..self.cols).map(|j| self.get(i, j) * v[j]).sum())
                .collect(),
        )
    }

    /// Copy out as row vectors.
    pub fn to_rows(&self) -> Vec<Vec<f64>> {
        (0..self.rows)
            .map(|i| (0..self.cols).map(|j| self.get(i, j)).collect())
            .collect()
    }

    /// Max absolute element-wise difference from `other` (for approximate
    /// comparison); `f64::INFINITY` on a shape mismatch.
    pub fn max_abs_diff(&self, other: &Matrix) -> f64 {
        if self.rows != other.rows || self.cols != other.cols {
            return f64::INFINITY;
        }
        self.data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiply() {
        let a = Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        let b = Matrix::from_rows(vec![vec![5.0, 6.0], vec![7.0, 8.0]]).unwrap();
        let c = a.mul(&b).unwrap();
        assert_eq!(c.to_rows(), vec![vec![19.0, 22.0], vec![43.0, 50.0]]);
    }

    #[test]
    fn transpose_and_identity() {
        let a = Matrix::from_rows(vec![vec![1.0, 2.0, 3.0]]).unwrap();
        assert_eq!(
            a.transpose().to_rows(),
            vec![vec![1.0], vec![2.0], vec![3.0]]
        );
        let i = Matrix::identity(3);
        assert_eq!(a.mul(&i).unwrap(), a);
    }

    #[test]
    fn mul_vec_works() {
        let a = Matrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();
        assert_eq!(a.mul_vec(&[1.0, 1.0]).unwrap(), vec![3.0, 7.0]);
        assert!(a.mul_vec(&[1.0]).is_none());
    }
}
