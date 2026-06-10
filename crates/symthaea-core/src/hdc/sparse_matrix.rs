// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Compressed Sparse Row (CSR) matrix for PDE solvers and graph Laplacians.

/// CSR sparse matrix: stores only non-zero entries.
#[derive(Debug, Clone)]
pub struct CsrMatrix {
    pub nrows: usize,
    pub ncols: usize,
    pub values: Vec<f64>,
    pub col_indices: Vec<usize>,
    pub row_ptr: Vec<usize>,
}

impl CsrMatrix {
    pub fn from_triplets(nrows: usize, ncols: usize, triplets: &[(usize, usize, f64)]) -> Self {
        let mut row_counts = vec![0usize; nrows];
        for &(r, _, _) in triplets {
            row_counts[r] += 1;
        }
        let mut row_ptr = vec![0usize; nrows + 1];
        for i in 0..nrows {
            row_ptr[i + 1] = row_ptr[i] + row_counts[i];
        }
        let nnz = row_ptr[nrows];
        let mut values = vec![0.0; nnz];
        let mut col_indices = vec![0usize; nnz];
        let mut offsets = vec![0usize; nrows];
        for &(r, c, v) in triplets {
            let pos = row_ptr[r] + offsets[r];
            values[pos] = v;
            col_indices[pos] = c;
            offsets[r] += 1;
        }
        // Sort columns within each row
        for i in 0..nrows {
            let (start, end) = (row_ptr[i], row_ptr[i + 1]);
            let sv = &mut values[start..end];
            let sc = &mut col_indices[start..end];
            for j in 1..sv.len() {
                let mut k = j;
                while k > 0 && sc[k - 1] > sc[k] {
                    sc.swap(k - 1, k);
                    sv.swap(k - 1, k);
                    k -= 1;
                }
            }
        }
        CsrMatrix {
            nrows,
            ncols,
            values,
            col_indices,
            row_ptr,
        }
    }

    pub fn identity(n: usize) -> Self {
        Self::from_triplets(n, n, &(0..n).map(|i| (i, i, 1.0)).collect::<Vec<_>>())
    }

    pub fn tridiagonal(n: usize, sub: f64, diag: f64, sup: f64) -> Self {
        let mut t = Vec::with_capacity(3 * n);
        for i in 0..n {
            t.push((i, i, diag));
            if i > 0 {
                t.push((i, i - 1, sub));
            }
            if i + 1 < n {
                t.push((i, i + 1, sup));
            }
        }
        Self::from_triplets(n, n, &t)
    }

    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        for k in self.row_ptr[row]..self.row_ptr[row + 1] {
            if self.col_indices[k] == col {
                return self.values[k];
            }
            if self.col_indices[k] > col {
                break;
            }
        }
        0.0
    }

    pub fn matvec(&self, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; self.nrows];
        for i in 0..self.nrows {
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                y[i] += self.values[k] * x[self.col_indices[k]];
            }
        }
        y
    }

    pub fn cg_solve(&self, b: &[f64], tol: f64, max_iter: usize) -> (Vec<f64>, usize, f64) {
        let n = self.nrows;
        let mut x = vec![0.0; n];
        let mut r = b.to_vec();
        let mut p = r.clone();
        let mut rs_old: f64 = r.iter().map(|ri| ri * ri).sum();
        if rs_old.sqrt() < tol {
            return (x, 0, rs_old.sqrt());
        }
        for iter in 0..max_iter {
            let ap = self.matvec(&p);
            let pap: f64 = p.iter().zip(&ap).map(|(pi, api)| pi * api).sum();
            if pap.abs() < 1e-30 {
                break;
            }
            let alpha = rs_old / pap;
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            let rs_new: f64 = r.iter().map(|ri| ri * ri).sum();
            if rs_new.sqrt() < tol {
                return (x, iter + 1, rs_new.sqrt());
            }
            let beta = rs_new / rs_old;
            for i in 0..n {
                p[i] = r[i] + beta * p[i];
            }
            rs_old = rs_new;
        }
        let fr: f64 = r.iter().map(|ri| ri * ri).sum::<f64>().sqrt();
        (x, max_iter, fr)
    }

    pub fn transpose(&self) -> CsrMatrix {
        let t: Vec<_> = (0..self.nrows)
            .flat_map(|i| {
                (self.row_ptr[i]..self.row_ptr[i + 1])
                    .map(move |k| (self.col_indices[k], i, self.values[k]))
            })
            .collect();
        CsrMatrix::from_triplets(self.ncols, self.nrows, &t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csr_identity() {
        let eye = CsrMatrix::identity(5);
        assert_eq!(eye.nnz(), 5);
        assert_eq!(eye.get(2, 2), 1.0);
        assert_eq!(eye.get(2, 3), 0.0);
    }

    #[test]
    fn test_csr_matvec() {
        let a = CsrMatrix::from_triplets(
            3,
            3,
            &[
                (0, 0, 2.0),
                (0, 2, 1.0),
                (1, 1, 3.0),
                (2, 0, 1.0),
                (2, 2, 4.0),
            ],
        );
        let y = a.matvec(&[1.0, 2.0, 3.0]);
        assert!((y[0] - 5.0).abs() < 1e-12);
        assert!((y[1] - 6.0).abs() < 1e-12);
        assert!((y[2] - 13.0).abs() < 1e-12);
    }

    #[test]
    fn test_csr_cg_poisson_1d() {
        let n = 50;
        let h = 1.0 / (n + 1) as f64;
        let lap = CsrMatrix::tridiagonal(n, -1.0 / (h * h), 2.0 / (h * h), -1.0 / (h * h));
        let b = vec![1.0; n];
        let (sol, iters, res) = lap.cg_solve(&b, 1e-10, 1000);
        assert!(res < 1e-8, "CG should converge: res={}", res);
        assert!(iters < 200);
        let mid = n / 2;
        let x_mid = (mid + 1) as f64 * h;
        let exact = x_mid * (1.0 - x_mid) / 2.0;
        assert!((sol[mid] - exact).abs() < 0.01);
    }

    #[test]
    fn test_csr_large_sparse() {
        let n = 1000;
        let mut t = Vec::new();
        for i in 0..n {
            t.push((i, i, 4.0));
            if i > 0 {
                t.push((i, i - 1, -1.0));
            }
            if i + 1 < n {
                t.push((i, i + 1, -1.0));
            }
        }
        let a = CsrMatrix::from_triplets(n, n, &t);
        assert!(a.nnz() < n * n / 10);
        let x: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let y = a.matvec(&x);
        assert!(y.iter().all(|v| v.is_finite()));
    }
}
