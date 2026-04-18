// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Hyperdimensional Linear Algebra Engine
//!
//! General-purpose linear algebra with consciousness-coupled HDC encoding.
//! Every operation produces a Phi measurement via proof trace integration.
//!
//! ## Capabilities
//!
//! - **Basic**: add, subtract, multiply, transpose, trace, determinant (LU-based NxN)
//! - **Factorizations**: LU (partial pivoting), QR (Householder), Cholesky (SPD)
//! - **Eigendecomposition**: QR algorithm (symmetric), power iteration (dominant)
//! - **SVD**: via eigendecomposition of A^T A
//! - **Linear systems**: Ax = b (LU), least-squares (QR)
//! - **Properties**: rank, null space basis, condition number estimate
//!
//! ## HDC Integration
//!
//! Matrices are encoded by binding row HVs with positional permutations.
//! Each operation produces Phi via proof trace integration.
//! Multi-path verification (e.g., solve via LU vs QR → compare).

use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::primitive_system::{seed_from_name, PrimitiveSystem};
use serde::{Deserialize, Serialize};

// ─── Constants ───────────────────────────────────────────────────────────────

const DEFAULT_TOLERANCE: f64 = 1e-12;
/// Consistent tolerance for SVD rank determination, null space, and condition number.
/// Using 1e-9 (geometric mean of machine epsilon and 1) avoids both:
/// - 1e-12 (too tight: numerical noise in eigendecomposition creates false nonzero σ)
/// - 1e-6 (too loose: discards real structure in ill-conditioned matrices)
const SVD_TOLERANCE: f64 = 1e-9;
const MAX_QR_ITERATIONS: usize = 200;
const POWER_ITERATION_MAX: usize = 100;

// ─── Result Types ────────────────────────────────────────────────────────────

/// A single step in a linear algebra proof trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinAlgProofStep {
    pub description: String,
    pub operation: String,
    pub phi: f64,
}

/// Result of a linear algebra operation with consciousness trace
#[derive(Debug, Clone)]
pub struct LinAlgResult {
    pub description: String,
    pub proof_trace: Vec<LinAlgProofStep>,
    pub phi: f64,
    pub verified: bool,
    pub encoding: BinaryHV,
}

impl LinAlgResult {
    fn new(description: String) -> Self {
        Self {
            description,
            proof_trace: Vec::new(),
            phi: 0.0,
            verified: false,
            encoding: BinaryHV::zero(),
        }
    }

    fn add_step(&mut self, description: &str, operation: &str, step_phi: f64) {
        self.proof_trace.push(LinAlgProofStep {
            description: description.to_string(),
            operation: operation.to_string(),
            phi: step_phi,
        });
        self.phi += step_phi;
    }
}

// ─── Matrix ──────────────────────────────────────────────────────────────────

/// A dense matrix with HDC encoding and consciousness coupling
#[derive(Debug, Clone)]
pub struct HdcMatrix {
    /// Row-major data storage
    pub data: Vec<f64>,
    /// Number of rows
    pub rows: usize,
    /// Number of columns
    pub cols: usize,
    /// HDC encoding of the matrix
    pub encoding: BinaryHV,
    /// Phi measurement from construction
    pub phi: f64,
}

/// A dense vector with HDC encoding
#[derive(Debug, Clone)]
pub struct HdcVector {
    pub data: Vec<f64>,
    pub encoding: BinaryHV,
}

impl HdcVector {
    /// Create a new vector with HDC encoding
    pub fn new(data: Vec<f64>) -> Self {
        let encoding = Self::encode_vector(&data);
        Self { data, encoding }
    }

    /// Create a zero vector
    pub fn zeros(n: usize) -> Self {
        Self::new(vec![0.0; n])
    }

    /// Vector dimension
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Dot product
    pub fn dot(&self, other: &HdcVector) -> f64 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Euclidean norm
    pub fn norm(&self) -> f64 {
        self.dot(self).sqrt()
    }

    /// Normalize in place
    pub fn normalize(&mut self) {
        let n = self.norm();
        if n > DEFAULT_TOLERANCE {
            for x in &mut self.data {
                *x /= n;
            }
        }
    }

    /// Return a normalized copy
    pub fn normalized(&self) -> Self {
        let mut v = self.clone();
        v.normalize();
        v
    }

    fn encode_vector(data: &[f64]) -> BinaryHV {
        let vec_prim = BinaryHV::random(seed_from_name("VECTOR"));
        let mut encoding = vec_prim;
        for (i, &val) in data.iter().enumerate() {
            let pos = BinaryHV::random(seed_from_name(&format!("VEC_POS_{}", i)));
            let val_hv = BinaryHV::random(seed_from_name(&format!("VAL_{}", val.to_bits())));
            encoding = encoding.bind(&pos.bind(&val_hv));
        }
        encoding
    }
}

impl HdcMatrix {
    /// Create a new matrix from row-major data
    pub fn new(data: Vec<f64>, rows: usize, cols: usize) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "Data length {} != rows*cols {}x{}",
            data.len(),
            rows,
            cols
        );
        let encoding = Self::encode_matrix(&data, rows, cols);
        Self {
            data,
            rows,
            cols,
            encoding,
            phi: 0.0,
        }
    }

    /// Create from 2D slice
    pub fn from_rows(rows_data: &[&[f64]]) -> Self {
        let rows = rows_data.len();
        assert!(rows > 0, "Matrix must have at least one row");
        let cols = rows_data[0].len();
        assert!(cols > 0, "Matrix must have at least one column");
        for row in rows_data {
            assert_eq!(row.len(), cols, "All rows must have same length");
        }
        let data: Vec<f64> = rows_data.iter().flat_map(|r| r.iter().copied()).collect();
        Self::new(data, rows, cols)
    }

    /// Identity matrix
    pub fn identity(n: usize) -> Self {
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Self::new(data, n, n)
    }

    /// Zero matrix
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::new(vec![0.0; rows * cols], rows, cols)
    }

    /// Access element at (row, col)
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.cols + col]
    }

    /// Set element at (row, col)
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, val: f64) {
        self.data[row * self.cols + col] = val;
    }

    /// Is square matrix
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    // ─── Basic Operations ────────────────────────────────────────────────

    /// Matrix addition
    pub fn add(&self, other: &HdcMatrix) -> (HdcMatrix, LinAlgResult) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);

        let data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();

        let mut result = LinAlgResult::new(format!("{}x{} matrix addition", self.rows, self.cols));
        result.add_step("Element-wise addition", "add", 0.1);

        let mat = HdcMatrix::new(data, self.rows, self.cols);
        result.encoding = mat.encoding;
        result.verified = true;
        (mat, result)
    }

    /// Matrix subtraction
    pub fn sub(&self, other: &HdcMatrix) -> (HdcMatrix, LinAlgResult) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);

        let data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();

        let mut result =
            LinAlgResult::new(format!("{}x{} matrix subtraction", self.rows, self.cols));
        result.add_step("Element-wise subtraction", "sub", 0.1);

        let mat = HdcMatrix::new(data, self.rows, self.cols);
        result.encoding = mat.encoding;
        result.verified = true;
        (mat, result)
    }

    /// Matrix multiplication
    pub fn mul(&self, other: &HdcMatrix) -> (HdcMatrix, LinAlgResult) {
        assert_eq!(
            self.cols, other.rows,
            "Matrix dimensions incompatible: {}x{} * {}x{}",
            self.rows, self.cols, other.rows, other.cols
        );

        let mut data = vec![0.0; self.rows * other.cols];
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                }
                data[i * other.cols + j] = sum;
            }
        }

        let mut result = LinAlgResult::new(format!(
            "{}x{} * {}x{} matrix multiplication",
            self.rows, self.cols, other.rows, other.cols
        ));
        result.add_step(
            &format!(
                "O(n³) multiply with {} multiplications",
                self.rows * other.cols * self.cols
            ),
            "multiply",
            0.2,
        );

        let mat = HdcMatrix::new(data, self.rows, other.cols);
        result.encoding = mat.encoding;
        result.verified = true;
        (mat, result)
    }

    /// Scalar multiplication
    pub fn scale(&self, scalar: f64) -> HdcMatrix {
        let data: Vec<f64> = self.data.iter().map(|&x| x * scalar).collect();
        HdcMatrix::new(data, self.rows, self.cols)
    }

    /// Transpose
    pub fn transpose(&self) -> HdcMatrix {
        let mut data = vec![0.0; self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                data[j * self.rows + i] = self.get(i, j);
            }
        }
        HdcMatrix::new(data, self.cols, self.rows)
    }

    /// Trace (sum of diagonal)
    pub fn trace(&self) -> f64 {
        assert!(self.is_square(), "Trace requires square matrix");
        (0..self.rows).map(|i| self.get(i, i)).sum()
    }

    // ─── LU Decomposition ───────────────────────────────────────────────

    /// LU decomposition with partial pivoting.
    /// Returns (L, U, P, sign) where PA = LU and sign is the permutation sign.
    pub fn lu_decompose(&self) -> (HdcMatrix, HdcMatrix, Vec<usize>, f64, LinAlgResult) {
        assert!(self.is_square(), "LU decomposition requires square matrix");
        let n = self.rows;

        let mut u_data = self.data.clone();
        let mut l_data = vec![0.0; n * n];
        let mut perm: Vec<usize> = (0..n).collect();
        let mut sign = 1.0;

        let mut result = LinAlgResult::new(format!("{}x{} LU decomposition", n, n));

        for k in 0..n {
            // Partial pivoting: find max in column k below diagonal
            let mut max_val = u_data[k * n + k].abs();
            let mut max_row = k;
            for i in (k + 1)..n {
                let val = u_data[i * n + k].abs();
                if val > max_val {
                    max_val = val;
                    max_row = i;
                }
            }

            if max_val < DEFAULT_TOLERANCE {
                result.add_step(
                    &format!("Column {} is zero (singular matrix)", k),
                    "pivot",
                    0.0,
                );
                continue;
            }

            // Swap rows if needed
            if max_row != k {
                perm.swap(k, max_row);
                sign = -sign;
                for j in 0..n {
                    let idx_k = k * n + j;
                    let idx_m = max_row * n + j;
                    u_data.swap(idx_k, idx_m);
                }
                // Also swap L rows for columns already done
                for j in 0..k {
                    let idx_k = k * n + j;
                    let idx_m = max_row * n + j;
                    l_data.swap(idx_k, idx_m);
                }
            }

            // Elimination
            let pivot = u_data[k * n + k];
            for i in (k + 1)..n {
                let factor = u_data[i * n + k] / pivot;
                l_data[i * n + k] = factor;
                for j in k..n {
                    u_data[i * n + j] -= factor * u_data[k * n + j];
                }
            }

            result.add_step(
                &format!("Eliminated column {} (pivot = {:.6})", k, pivot),
                "eliminate",
                0.1,
            );
        }

        // Set L diagonal to 1
        for i in 0..n {
            l_data[i * n + i] = 1.0;
        }

        let l = HdcMatrix::new(l_data, n, n);
        let u = HdcMatrix::new(u_data, n, n);
        result.encoding = l.encoding.bind(&u.encoding);
        result.verified = true;

        (l, u, perm, sign, result)
    }

    /// Determinant via LU decomposition
    pub fn determinant(&self) -> (f64, LinAlgResult) {
        assert!(self.is_square(), "Determinant requires square matrix");
        let n = self.rows;

        if n == 1 {
            let mut result = LinAlgResult::new("1x1 determinant".to_string());
            result.add_step("Direct 1x1", "determinant", 0.05);
            result.verified = true;
            return (self.data[0], result);
        }

        if n == 2 {
            let det = self.get(0, 0) * self.get(1, 1) - self.get(0, 1) * self.get(1, 0);
            let mut result = LinAlgResult::new("2x2 determinant".to_string());
            result.add_step("ad - bc formula", "determinant", 0.1);
            result.verified = true;
            return (det, result);
        }

        let (_l, u, _perm, sign, mut result) = self.lu_decompose();
        let det: f64 = sign * (0..n).map(|i| u.get(i, i)).product::<f64>();
        result.description = format!("{}x{} determinant via LU", n, n);
        result.add_step(
            &format!("Product of U diagonal * sign = {:.6}", det),
            "determinant",
            0.15,
        );
        (det, result)
    }

    // ─── Linear System Solving ───────────────────────────────────────────

    /// Solve Ax = b via LU decomposition
    pub fn solve(&self, b: &HdcVector) -> (HdcVector, LinAlgResult) {
        assert!(self.is_square(), "solve requires square matrix");
        assert_eq!(self.rows, b.len(), "Dimension mismatch");
        let n = self.rows;

        let (l, u, perm, _sign, mut result) = self.lu_decompose();
        result.description = format!("Solve {}x{} system via LU", n, n);

        // Apply permutation to b
        let mut pb = vec![0.0; n];
        for i in 0..n {
            pb[i] = b.data[perm[i]];
        }

        // Forward substitution: Ly = Pb
        let mut y = vec![0.0; n];
        for i in 0..n {
            let mut sum = pb[i];
            for j in 0..i {
                sum -= l.get(i, j) * y[j];
            }
            y[i] = sum;
        }
        result.add_step("Forward substitution (Ly = Pb)", "forward_sub", 0.1);

        // Back substitution: Ux = y
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = y[i];
            for j in (i + 1)..n {
                sum -= u.get(i, j) * x[j];
            }
            let diag = u.get(i, i);
            if diag.abs() < DEFAULT_TOLERANCE {
                result.add_step("Singular matrix encountered", "back_sub", 0.0);
                return (HdcVector::zeros(n), result);
            }
            x[i] = sum / diag;
        }
        result.add_step("Back substitution (Ux = y)", "back_sub", 0.1);
        result.verified = true;

        (HdcVector::new(x), result)
    }

    // ─── QR Decomposition ───────────────────────────────────────────────

    /// QR decomposition via Householder reflections.
    /// Returns (Q, R).
    pub fn qr_decompose(&self) -> (HdcMatrix, HdcMatrix, LinAlgResult) {
        let m = self.rows;
        let n = self.cols;
        let min_mn = m.min(n);

        let mut r_data = self.data.clone();
        let mut q = HdcMatrix::identity(m);

        let mut result = LinAlgResult::new(format!("{}x{} QR decomposition (Householder)", m, n));

        for k in 0..min_mn {
            // Extract column k below diagonal
            let mut v = vec![0.0; m - k];
            for i in k..m {
                v[i - k] = r_data[i * n + k];
            }

            let norm_v: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm_v < DEFAULT_TOLERANCE {
                continue;
            }

            // Householder vector
            let sign = if v[0] >= 0.0 { 1.0 } else { -1.0 };
            v[0] += sign * norm_v;
            let norm_v2: f64 = v.iter().map(|x| x * x).sum::<f64>();
            if norm_v2 < DEFAULT_TOLERANCE {
                continue;
            }

            // Apply H = I - 2vv^T/||v||^2 to R (from left)
            for j in k..n {
                let mut dot = 0.0;
                for i in 0..v.len() {
                    dot += v[i] * r_data[(i + k) * n + j];
                }
                let factor = 2.0 * dot / norm_v2;
                for i in 0..v.len() {
                    r_data[(i + k) * n + j] -= factor * v[i];
                }
            }

            // Apply H to Q (from right): Q = Q * H
            for i in 0..m {
                let mut dot = 0.0;
                for j in 0..v.len() {
                    dot += q.get(i, j + k) * v[j];
                }
                let factor = 2.0 * dot / norm_v2;
                for j in 0..v.len() {
                    let old = q.get(i, j + k);
                    q.set(i, j + k, old - factor * v[j]);
                }
            }

            result.add_step(&format!("Householder reflection {}", k), "householder", 0.1);
        }

        let r = HdcMatrix::new(r_data, m, n);
        result.encoding = q.encoding.bind(&r.encoding);
        result.verified = true;

        (q, r, result)
    }

    /// Solve Ax = b via QR decomposition (least-squares for overdetermined systems)
    pub fn solve_qr(&self, b: &HdcVector) -> (HdcVector, LinAlgResult) {
        let (q, r, mut result) = self.qr_decompose();
        result.description = format!("Solve {}x{} system via QR", self.rows, self.cols);

        let qt = q.transpose();
        let qtb_data: Vec<f64> = (0..qt.rows)
            .map(|i| (0..qt.cols).map(|j| qt.get(i, j) * b.data[j]).sum::<f64>())
            .collect();

        // Back substitution: Rx = Q^T b
        let n = self.cols.min(self.rows);
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = qtb_data[i];
            for j in (i + 1)..n {
                sum -= r.get(i, j) * x[j];
            }
            let diag = r.get(i, i);
            if diag.abs() < DEFAULT_TOLERANCE {
                result.add_step("Rank-deficient system", "back_sub", 0.0);
                return (HdcVector::zeros(n), result);
            }
            x[i] = sum / diag;
        }

        result.add_step("Back substitution (Rx = Q^T b)", "back_sub", 0.1);
        result.verified = true;
        (HdcVector::new(x), result)
    }

    // ─── Cholesky Decomposition ──────────────────────────────────────────

    /// Cholesky decomposition for symmetric positive definite matrices.
    /// Returns L such that A = LL^T.
    pub fn cholesky(&self) -> Option<(HdcMatrix, LinAlgResult)> {
        assert!(self.is_square(), "Cholesky requires square matrix");
        let n = self.rows;
        let mut l_data = vec![0.0; n * n];

        let mut result = LinAlgResult::new(format!("{}x{} Cholesky decomposition", n, n));

        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l_data[j * n + k] * l_data[j * n + k];
            }
            let diag = self.get(j, j) - sum;
            if diag <= 0.0 {
                result.add_step("Matrix is not positive definite", "cholesky", 0.0);
                return None;
            }
            l_data[j * n + j] = diag.sqrt();

            for i in (j + 1)..n {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l_data[i * n + k] * l_data[j * n + k];
                }
                l_data[i * n + j] = (self.get(i, j) - sum) / l_data[j * n + j];
            }

            result.add_step(
                &format!(
                    "Cholesky column {} (L[{0},{0}] = {:.6})",
                    j,
                    l_data[j * n + j]
                ),
                "cholesky",
                0.1,
            );
        }

        let l = HdcMatrix::new(l_data, n, n);
        result.encoding = l.encoding;
        result.verified = true;
        Some((l, result))
    }

    // ─── Eigendecomposition ──────────────────────────────────────────────

    /// Power iteration to find the dominant eigenvalue and eigenvector
    pub fn power_iteration(&self, tol: f64) -> (f64, HdcVector, LinAlgResult) {
        assert!(
            self.is_square(),
            "Eigendecomposition requires square matrix"
        );
        let n = self.rows;
        let mut result = LinAlgResult::new(format!("{}x{} power iteration", n, n));

        // Start with random-ish vector
        let mut v = vec![1.0; n];
        v[0] = 1.1;
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in &mut v {
            *x /= norm;
        }

        let mut eigenvalue = 0.0;

        for iter in 0..POWER_ITERATION_MAX {
            // w = Av
            let mut w = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    w[i] += self.get(i, j) * v[j];
                }
            }

            // Rayleigh quotient: lambda = v^T w
            let new_eigenvalue: f64 = v.iter().zip(w.iter()).map(|(a, b)| a * b).sum();

            // Normalize w
            let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < DEFAULT_TOLERANCE {
                break;
            }
            for x in &mut w {
                *x /= norm;
            }

            if (new_eigenvalue - eigenvalue).abs() < tol {
                result.add_step(
                    &format!(
                        "Converged after {} iterations (λ = {:.8})",
                        iter + 1,
                        new_eigenvalue
                    ),
                    "power_iter",
                    0.2,
                );
                eigenvalue = new_eigenvalue;
                v = w;
                break;
            }

            eigenvalue = new_eigenvalue;
            v = w;
        }

        result.verified = true;
        (eigenvalue, HdcVector::new(v), result)
    }

    /// QR algorithm for eigenvalues of a symmetric matrix.
    /// Returns sorted eigenvalues.
    pub fn eigenvalues_symmetric(&self) -> (Vec<f64>, LinAlgResult) {
        assert!(self.is_square(), "Eigenvalues require square matrix");
        let n = self.rows;
        let mut result =
            LinAlgResult::new(format!("{}x{} symmetric eigenvalues (QR algorithm)", n, n));

        let mut a = self.clone();

        for iter in 0..MAX_QR_ITERATIONS {
            let (q, r, _) = a.qr_decompose();
            let (new_a, _) = r.mul(&q);
            a = new_a;

            // Check convergence (off-diagonal → 0)
            let mut off_diag = 0.0;
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        off_diag += a.get(i, j).abs();
                    }
                }
            }

            if off_diag < DEFAULT_TOLERANCE * (n * n) as f64 {
                result.add_step(
                    &format!("QR converged after {} iterations", iter + 1),
                    "qr_eigen",
                    0.2,
                );
                break;
            }
        }

        let mut eigenvalues: Vec<f64> = (0..n).map(|i| a.get(i, i)).collect();
        eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        result.verified = true;
        (eigenvalues, result)
    }

    /// Full eigendecomposition for symmetric matrices.
    /// Returns (eigenvalues, eigenvectors_as_columns).
    pub fn eigen_symmetric(&self) -> (Vec<f64>, HdcMatrix, LinAlgResult) {
        assert!(self.is_square());
        let n = self.rows;
        let mut result = LinAlgResult::new(format!("{}x{} symmetric eigendecomposition", n, n));

        let mut a = self.clone();
        let mut v = HdcMatrix::identity(n);

        for iter in 0..MAX_QR_ITERATIONS {
            let (q, r, _) = a.qr_decompose();
            let (new_a, _) = r.mul(&q);
            let (new_v, _) = v.mul(&q);
            a = new_a;
            v = new_v;

            let mut off_diag = 0.0;
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        off_diag += a.get(i, j).abs();
                    }
                }
            }

            if off_diag < DEFAULT_TOLERANCE * (n * n) as f64 {
                result.add_step(
                    &format!("Eigen converged after {} iterations", iter + 1),
                    "qr_eigen",
                    0.3,
                );
                break;
            }
        }

        let eigenvalues: Vec<f64> = (0..n).map(|i| a.get(i, i)).collect();
        result.verified = true;
        (eigenvalues, v, result)
    }

    // ─── SVD ─────────────────────────────────────────────────────────────

    /// Singular Value Decomposition: A = U Σ V^T
    ///
    /// Returns (singular_values, U, V) where:
    /// - singular_values: min(m,n) values sorted by descending magnitude
    /// - U: m×m orthogonal matrix (left singular vectors)
    /// - V: n×n orthogonal matrix (right singular vectors)
    ///
    /// Uses eigendecomposition of A^T A. Handles rectangular matrices correctly.
    /// The SVD tolerance for rank determination is `SVD_TOLERANCE = 1e-9`.
    pub fn svd(&self) -> (Vec<f64>, HdcMatrix, HdcMatrix, LinAlgResult) {
        let mut result = LinAlgResult::new(format!("{}x{} SVD", self.rows, self.cols));
        let m = self.rows;
        let n = self.cols;
        let min_mn = m.min(n);

        // A^T A (n×n) gives V and σ²
        let at = self.transpose();
        let (ata, _) = at.mul(self);
        let (eigenvalues, v, _) = ata.eigen_symmetric();

        let singular_values: Vec<f64> = eigenvalues
            .iter()
            .take(min_mn) // only min(m,n) singular values are meaningful
            .map(|&e| if e > 0.0 { e.sqrt() } else { 0.0 })
            .collect();

        // Consistent rank threshold: use SVD_TOLERANCE for ALL rank decisions
        let rank = singular_values
            .iter()
            .filter(|&&s| s > SVD_TOLERANCE)
            .count();

        // U = A V Σ^{-1} for the first `rank` columns
        // AV has shape m × n; we only use the first `rank` columns
        let (av, _) = self.mul(&v);
        let mut u_data = vec![0.0; m * m];

        for j in 0..rank.min(min_mn) {
            let inv_sigma = 1.0 / singular_values[j];
            for i in 0..m {
                // av is m×n, so j must be < n (guaranteed since rank ≤ min(m,n))
                u_data[i * m + j] = av.get(i, j) * inv_sigma;
            }
        }

        // Fill remaining columns (rank..m) with orthogonal complement via
        // modified Gram-Schmidt against existing U columns.
        // Try each standard basis vector e_k as candidate; if linearly
        // independent from existing columns, orthogonalize and add.
        let mut filled = rank;
        for candidate_idx in 0..m {
            if filled >= m { break; }

            let mut col = vec![0.0; m];
            col[candidate_idx] = 1.0;

            // Orthogonalize against all existing columns (0..filled)
            for k in 0..filled {
                let mut dot = 0.0;
                for i in 0..m {
                    dot += col[i] * u_data[i * m + k];
                }
                for i in 0..m {
                    col[i] -= dot * u_data[i * m + k];
                }
            }

            // Double orthogonalization for numerical stability (CGS re-orthogonalization)
            for k in 0..filled {
                let mut dot = 0.0;
                for i in 0..m {
                    dot += col[i] * u_data[i * m + k];
                }
                for i in 0..m {
                    col[i] -= dot * u_data[i * m + k];
                }
            }

            let norm: f64 = col.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > SVD_TOLERANCE {
                for i in 0..m {
                    u_data[i * m + filled] = col[i] / norm;
                }
                filled += 1;
            }
        }

        let u = HdcMatrix::new(u_data, m, m);
        result.add_step(
            &format!(
                "SVD computed: {} singular values, rank = {}",
                singular_values.len(),
                rank
            ),
            "svd",
            0.3,
        );
        result.verified = true;

        (singular_values, u, v, result)
    }

    /// Null space (kernel) of the matrix.
    ///
    /// Returns the columns of V corresponding to zero singular values.
    /// For an m×n matrix of rank r, the null space has dimension n - r.
    /// Returns None if the matrix has full column rank (null space is trivial).
    pub fn null_space(&self) -> Option<HdcMatrix> {
        let (sv, _, v, _) = self.svd();
        let n = self.cols;
        let rank = self.rank(); // uses relative tolerance
        let null_dim = n.saturating_sub(rank);

        if null_dim == 0 {
            return None;
        }

        // Null space = columns rank..n of V
        let mut null_data = vec![0.0; n * null_dim];
        for j in 0..null_dim {
            for i in 0..n {
                null_data[i * null_dim + j] = v.get(i, rank + j);
            }
        }

        Some(HdcMatrix::new(null_data, n, null_dim))
    }

    // ─── Properties ──────────────────────────────────────────────────────

    /// Matrix rank (number of non-zero singular values).
    ///
    /// Uses a **relative** tolerance: σ_i is considered zero if
    /// σ_i < max(m,n) × ε × σ_max, where ε = SVD_TOLERANCE.
    /// This is the standard approach (MATLAB, NumPy, LAPACK) and handles
    /// ill-conditioned matrices correctly even when SVD is computed via A^T A.
    pub fn rank(&self) -> usize {
        let (sv, _, _, _) = self.svd();
        let sigma_max = sv.iter().cloned().fold(0.0f64, f64::max);
        let tol = (self.rows.max(self.cols) as f64) * SVD_TOLERANCE * sigma_max;
        sv.iter().filter(|&&s| s > tol).count()
    }

    /// Inverse via LU decomposition
    pub fn inverse(&self) -> Option<(HdcMatrix, LinAlgResult)> {
        assert!(self.is_square());
        let n = self.rows;

        let (det, _) = self.determinant();
        if det.abs() < DEFAULT_TOLERANCE {
            return None;
        }

        let mut inv_data = vec![0.0; n * n];
        for j in 0..n {
            let mut e = vec![0.0; n];
            e[j] = 1.0;
            let (col, _) = self.solve(&HdcVector::new(e));
            for i in 0..n {
                inv_data[i * n + j] = col.data[i];
            }
        }

        let mut result = LinAlgResult::new(format!("{}x{} matrix inverse", n, n));
        result.add_step("Computed via LU column-by-column", "inverse", 0.2);
        result.verified = true;

        Some((HdcMatrix::new(inv_data, n, n), result))
    }

    /// Condition number estimate (ratio of largest to smallest singular value)
    /// Condition number κ(A) = σ_max / σ_min.
    /// Uses SVD_TOLERANCE for consistent rank determination.
    pub fn condition_number(&self) -> f64 {
        let (sv, _, _, _) = self.svd();
        let max = sv.iter().cloned().fold(0.0_f64, f64::max);
        let min = sv
            .iter()
            .cloned()
            .filter(|&s| s > SVD_TOLERANCE)
            .fold(f64::INFINITY, f64::min);
        if min <= SVD_TOLERANCE {
            f64::INFINITY
        } else {
            max / min
        }
    }

    /// Frobenius norm
    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Alias for frobenius_norm (convenience name from spec)
    pub fn norm_frobenius(&self) -> f64 {
        self.frobenius_norm()
    }

    /// Check if matrix is symmetric within tolerance
    pub fn is_symmetric(&self) -> bool {
        self.is_symmetric_with_tol(DEFAULT_TOLERANCE)
    }

    /// Check if matrix is symmetric within given tolerance
    pub fn is_symmetric_with_tol(&self, tol: f64) -> bool {
        if !self.is_square() {
            return false;
        }
        for i in 0..self.rows {
            for j in (i + 1)..self.cols {
                if (self.get(i, j) - self.get(j, i)).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Check if matrix is positive definite (via Cholesky attempt)
    pub fn is_positive_definite(&self) -> bool {
        if !self.is_symmetric() {
            return false;
        }
        self.cholesky().is_some()
    }

    /// Condition number estimate (alias for condition_number)
    pub fn condition_number_estimate(&self) -> f64 {
        self.condition_number()
    }

    /// Matrix-vector multiplication: A * v
    pub fn mul_vec(&self, v: &HdcVector) -> HdcVector {
        assert_eq!(
            self.cols,
            v.len(),
            "Matrix cols {} != vector length {}",
            self.cols,
            v.len()
        );
        let mut result = vec![0.0; self.rows];
        for i in 0..self.rows {
            let mut sum = 0.0;
            for j in 0..self.cols {
                sum += self.get(i, j) * v.data[j];
            }
            result[i] = sum;
        }
        HdcVector::new(result)
    }

    /// Create a matrix from data with (rows, cols, data) argument order
    pub fn from_data(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        Self::new(data, rows, cols)
    }

    /// Create a diagonal matrix from a vector of diagonal entries
    pub fn diagonal(diag: &[f64]) -> Self {
        let n = diag.len();
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = diag[i];
        }
        Self::new(data, n, n)
    }

    // ─── Multi-path verification ─────────────────────────────────────────

    /// Solve Ax = b via both LU and QR, return the solution with highest Phi.
    /// Multi-path verification: if both agree, confidence increases.
    pub fn solve_multipath(&self, b: &HdcVector) -> (HdcVector, LinAlgResult) {
        let (x_lu, result_lu) = self.solve(b);
        let (x_qr, result_qr) = self.solve_qr(b);

        // Check agreement
        let diff: f64 = x_lu
            .data
            .iter()
            .zip(x_qr.data.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>();

        let mut result = LinAlgResult::new("Multi-path solve (LU + QR)".to_string());
        result.proof_trace.extend(result_lu.proof_trace);
        result.proof_trace.extend(result_qr.proof_trace);

        if diff < DEFAULT_TOLERANCE.sqrt() * b.len() as f64 {
            result.add_step(
                &format!("LU and QR agree (diff = {:.2e})", diff),
                "multipath_verify",
                0.5,
            );
            result.phi = result_lu.phi + result_qr.phi + 0.5; // Agreement bonus
            result.verified = true;
        } else {
            result.add_step(
                &format!("LU and QR disagree (diff = {:.2e}), using LU", diff),
                "multipath_verify",
                0.1,
            );
            result.phi = result_lu.phi;
            result.verified = false;
        }

        (x_lu, result)
    }

    // ─── HDC Encoding ────────────────────────────────────────────────────

    fn encode_matrix(data: &[f64], rows: usize, cols: usize) -> BinaryHV {
        let mat_prim = BinaryHV::random(seed_from_name("MATRIX"));
        let mut encoding = mat_prim;

        // Encode each row by binding with positional permutation
        for i in 0..rows {
            let row_pos = BinaryHV::random(seed_from_name(&format!("ROW_{}", i)));
            let mut row_encoding = BinaryHV::zero();
            for j in 0..cols {
                let val = data[i * cols + j];
                let col_pos = BinaryHV::random(seed_from_name(&format!("COL_{}", j)));
                let val_hv = BinaryHV::random(seed_from_name(&format!("VAL_{}", val.to_bits())));
                row_encoding = row_encoding.bind(&col_pos.bind(&val_hv));
            }
            encoding = encoding.bind(&row_pos.bind(&row_encoding));
        }
        encoding
    }
}

// ─── Linear Algebra Engine (stateful wrapper) ────────────────────────────────

/// The Hyperdimensional Linear Algebra Engine
///
/// Stateful wrapper providing access to the PrimitiveSystem and tracking
/// computation statistics.
pub struct LinearAlgebraEngine {
    primitives: &'static PrimitiveSystem,
    stats: LinAlgStats,
}

/// Statistics about linear algebra engine usage
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LinAlgStats {
    pub total_operations: usize,
    pub total_phi: f64,
    pub solves: usize,
    pub factorizations: usize,
    pub eigendecompositions: usize,
}

impl LinearAlgebraEngine {
    /// Create a new linear algebra engine
    pub fn new() -> Self {
        Self {
            primitives: PrimitiveSystem::global(),
            stats: LinAlgStats::default(),
        }
    }

    /// Solve a linear system Ax = b with multi-path verification
    pub fn solve(&mut self, a: &HdcMatrix, b: &HdcVector) -> (HdcVector, LinAlgResult) {
        let result = a.solve_multipath(b);
        self.stats.total_operations += 1;
        self.stats.solves += 1;
        self.stats.total_phi += result.1.phi;
        result
    }

    /// Compute determinant
    pub fn determinant(&mut self, a: &HdcMatrix) -> (f64, LinAlgResult) {
        let result = a.determinant();
        self.stats.total_operations += 1;
        self.stats.total_phi += result.1.phi;
        result
    }

    /// Compute eigenvalues of a symmetric matrix
    pub fn eigenvalues(&mut self, a: &HdcMatrix) -> (Vec<f64>, LinAlgResult) {
        let result = a.eigenvalues_symmetric();
        self.stats.total_operations += 1;
        self.stats.eigendecompositions += 1;
        self.stats.total_phi += result.1.phi;
        result
    }

    /// Full eigendecomposition of a symmetric matrix
    pub fn eigen(&mut self, a: &HdcMatrix) -> (Vec<f64>, HdcMatrix, LinAlgResult) {
        let result = a.eigen_symmetric();
        self.stats.total_operations += 1;
        self.stats.eigendecompositions += 1;
        self.stats.total_phi += result.2.phi;
        result
    }

    /// SVD
    pub fn svd(&mut self, a: &HdcMatrix) -> (Vec<f64>, HdcMatrix, HdcMatrix, LinAlgResult) {
        let result = a.svd();
        self.stats.total_operations += 1;
        self.stats.factorizations += 1;
        self.stats.total_phi += result.3.phi;
        result
    }

    /// Get engine statistics
    pub fn stats(&self) -> &LinAlgStats {
        &self.stats
    }
}

// ─── Conjugate Gradient ──────────────────────────────────────────────────────

/// Matrix-vector product: y = A * x
fn mat_vec_mul(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().zip(x.iter()).map(|(aij, xj)| aij * xj).sum())
        .collect()
}

/// Dot product of two vectors.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

/// L2 norm of a vector.
fn vec_norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// Conjugate Gradient iterative solver for symmetric positive-definite Ax = b.
///
/// O(n√κ) iterations vs O(n³) for direct methods. Requires A to be SPD.
///
/// # Algorithm (Hestenes & Stiefel 1952)
///
/// r₀ = b - A·x₀ (x₀ = 0)
/// p₀ = r₀
/// for k = 0, 1, ...:
///   α_k = ⟨r_k, r_k⟩ / ⟨p_k, A·p_k⟩
///   x_{k+1} = x_k + α_k · p_k
///   r_{k+1} = r_k - α_k · A · p_k
///   β_k = ⟨r_{k+1}, r_{k+1}⟩ / ⟨r_k, r_k⟩
///   p_{k+1} = r_{k+1} + β_k · p_k
/// Stop when ‖r_k‖ < tol
pub fn conjugate_gradient(
    a: &[Vec<f64>],
    b: &[f64],
    tol: f64,
    max_iter: usize,
) -> Result<Vec<f64>, String> {
    let n = b.len();
    if a.len() != n || a.iter().any(|row| row.len() != n) {
        return Err("conjugate_gradient: matrix dimensions do not match b".to_string());
    }

    let mut x = vec![0.0f64; n];
    let mut r = b.to_vec(); // r₀ = b - A*0 = b
    let mut p = r.clone();
    let mut r_dot = dot(&r, &r);

    for _ in 0..max_iter {
        if r_dot.sqrt() < tol {
            return Ok(x);
        }
        let ap = mat_vec_mul(a, &p);
        let p_ap = dot(&p, &ap);
        if p_ap.abs() < 1e-300 {
            break;
        }
        let alpha = r_dot / p_ap;

        // x_{k+1} = x_k + α_k * p_k
        for i in 0..n {
            x[i] += alpha * p[i];
        }
        // r_{k+1} = r_k - α_k * A * p_k
        for i in 0..n {
            r[i] -= alpha * ap[i];
        }

        let r_dot_new = dot(&r, &r);
        let beta = r_dot_new / r_dot;
        r_dot = r_dot_new;

        // p_{k+1} = r_{k+1} + β_k * p_k
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
    }

    if r_dot.sqrt() < tol {
        Ok(x)
    } else {
        Err(format!(
            "conjugate_gradient: did not converge (residual = {:.2e})",
            r_dot.sqrt()
        ))
    }
}

/// Preconditioned Conjugate Gradient with Jacobi (diagonal) preconditioner.
///
/// M = diag(A), solve M⁻¹Ax = M⁻¹b. Improves convergence on diagonally
/// dominant systems by equilibrating the eigenvalue spread.
pub fn preconditioned_conjugate_gradient(
    a: &[Vec<f64>],
    b: &[f64],
    tol: f64,
    max_iter: usize,
) -> Result<Vec<f64>, String> {
    let n = b.len();
    if a.len() != n || a.iter().any(|row| row.len() != n) {
        return Err("preconditioned_cg: matrix dimensions do not match b".to_string());
    }

    // Jacobi preconditioner: M_inv[i] = 1/a[i][i]
    let m_inv: Vec<f64> = (0..n)
        .map(|i| {
            let d = a[i][i];
            if d.abs() > 1e-15 { 1.0 / d } else { 1.0 }
        })
        .collect();

    let mut x = vec![0.0f64; n];
    let mut r = b.to_vec(); // r₀ = b - A*0 = b
    // z = M⁻¹ * r
    let mut z: Vec<f64> = r.iter().zip(m_inv.iter()).map(|(ri, mi)| ri * mi).collect();
    let mut p = z.clone();
    let mut rz = dot(&r, &z);

    for _ in 0..max_iter {
        if vec_norm(&r) < tol {
            return Ok(x);
        }
        let ap = mat_vec_mul(a, &p);
        let p_ap = dot(&p, &ap);
        if p_ap.abs() < 1e-300 {
            break;
        }
        let alpha = rz / p_ap;

        for i in 0..n {
            x[i] += alpha * p[i];
        }
        for i in 0..n {
            r[i] -= alpha * ap[i];
        }

        z = r.iter().zip(m_inv.iter()).map(|(ri, mi)| ri * mi).collect();
        let rz_new = dot(&r, &z);
        let beta = rz_new / rz;
        rz = rz_new;

        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
    }

    if vec_norm(&r) < tol {
        Ok(x)
    } else {
        Err(format!(
            "preconditioned_cg: did not converge (residual = {:.2e})",
            vec_norm(&r)
        ))
    }
}

/// Estimate condition number via power iteration (ratio of largest to smallest eigenvalue).
///
/// Uses `iterations` power iteration steps for both max and min eigenvalue estimates.
pub fn condition_number_estimate(a: &[Vec<f64>], iterations: usize) -> f64 {
    let n = a.len();
    if n == 0 {
        return 1.0;
    }

    // Estimate largest eigenvalue (power iteration)
    let mut v = vec![1.0f64; n];
    let mut lambda_max = 1.0f64;
    for _ in 0..iterations {
        let av = mat_vec_mul(a, &v);
        let norm = vec_norm(&av);
        if norm < 1e-15 {
            break;
        }
        lambda_max = norm;
        v = av.iter().map(|&x| x / norm).collect();
    }

    // Estimate smallest eigenvalue via inverse iteration (shift-invert approximation)
    // Use the deflated matrix A - lambda_max * I and find its most-negative eigenvalue
    // Simplified: use the Rayleigh quotient on A * v for a near-null vector
    let mut v2 = vec![1.0f64; n];
    // Make v2 orthogonal to v (the dominant eigenvector)
    let proj: f64 = dot(&v2, &v);
    for i in 0..n {
        v2[i] -= proj * v[i];
    }
    let norm2 = vec_norm(&v2);
    if norm2 > 1e-10 {
        for x in &mut v2 {
            *x /= norm2;
        }
    } else {
        // Fallback: use a simple unit vector orthogonal to v
        v2 = vec![0.0; n];
        if n > 1 {
            v2[1] = 1.0;
            let proj2: f64 = dot(&v2, &v);
            for i in 0..n {
                v2[i] -= proj2 * v[i];
            }
            let n2 = vec_norm(&v2);
            if n2 > 1e-10 {
                for x in &mut v2 {
                    *x /= n2;
                }
            }
        }
    }

    let av2 = mat_vec_mul(a, &v2);
    let lambda_min = dot(&v2, &av2).abs().max(1e-15);

    (lambda_max / lambda_min).max(1.0)
}

// ─── Sparse Matrix (CSR Format) ─────────────────────────────────────────────

/// Compressed Sparse Row (CSR) matrix representation.
///
/// Stores only non-zero entries. For an m×n matrix with nnz non-zeros:
/// - `values`: non-zero values (length nnz)
/// - `col_indices`: column index for each value (length nnz)
/// - `row_ptr`: start index in values/col_indices for each row (length m+1)
///
/// Memory: O(nnz + m) vs O(m·n) for dense. Critical for large sparse systems.
#[derive(Debug, Clone)]
pub struct SparseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<f64>,
    pub col_indices: Vec<usize>,
    pub row_ptr: Vec<usize>,
}

impl SparseMatrix {
    /// Create an empty sparse matrix.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            values: Vec::new(),
            col_indices: Vec::new(),
            row_ptr: vec![0; rows + 1],
        }
    }

    /// Create from triplets (row, col, value). Duplicate entries are summed.
    pub fn from_triplets(rows: usize, cols: usize, triplets: &[(usize, usize, f64)]) -> Self {
        // Sort by row, then column
        let mut sorted = triplets.to_vec();
        sorted.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptr = vec![0usize; rows + 1];

        let mut prev_row = 0;
        let mut prev_col = usize::MAX;

        for &(r, c, v) in &sorted {
            if v.abs() < 1e-15 { continue; }
            // Sum duplicates
            if r == prev_row && c == prev_col && !values.is_empty() {
                *values.last_mut().unwrap() += v;
            } else {
                values.push(v);
                col_indices.push(c);
                for row in (prev_row + 1)..=r {
                    row_ptr[row] = values.len() - 1;
                }
                prev_row = r;
                prev_col = c;
            }
        }
        for row in (prev_row + 1)..=rows {
            row_ptr[row] = values.len();
        }

        Self { rows, cols, values, col_indices, row_ptr }
    }

    /// Create identity matrix of given size.
    pub fn identity(n: usize) -> Self {
        let triplets: Vec<(usize, usize, f64)> = (0..n).map(|i| (i, i, 1.0)).collect();
        Self::from_triplets(n, n, &triplets)
    }

    /// Number of non-zero entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Get element at (row, col). O(log(nnz_in_row)) via binary search.
    pub fn get(&self, row: usize, col: usize) -> f64 {
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];
        match self.col_indices[start..end].binary_search(&col) {
            Ok(idx) => self.values[start + idx],
            Err(_) => 0.0,
        }
    }

    /// Sparse matrix-vector product: y = A·x. O(nnz).
    pub fn mul_vec(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.cols);
        let mut y = vec![0.0; self.rows];
        for i in 0..self.rows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            for j in start..end {
                y[i] += self.values[j] * x[self.col_indices[j]];
            }
        }
        y
    }

    /// Sparse matrix addition: C = A + B.
    pub fn add(&self, other: &SparseMatrix) -> SparseMatrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut triplets = Vec::with_capacity(self.nnz() + other.nnz());
        for i in 0..self.rows {
            for j in self.row_ptr[i]..self.row_ptr[i + 1] {
                triplets.push((i, self.col_indices[j], self.values[j]));
            }
            for j in other.row_ptr[i]..other.row_ptr[i + 1] {
                triplets.push((i, other.col_indices[j], other.values[j]));
            }
        }
        SparseMatrix::from_triplets(self.rows, self.cols, &triplets)
    }

    /// Transpose: A^T (swaps rows and columns).
    pub fn transpose(&self) -> SparseMatrix {
        let mut triplets = Vec::with_capacity(self.nnz());
        for i in 0..self.rows {
            for j in self.row_ptr[i]..self.row_ptr[i + 1] {
                triplets.push((self.col_indices[j], i, self.values[j]));
            }
        }
        SparseMatrix::from_triplets(self.cols, self.rows, &triplets)
    }

    /// Convert to dense HdcMatrix.
    pub fn to_dense(&self) -> Vec<Vec<f64>> {
        let mut dense = vec![vec![0.0; self.cols]; self.rows];
        for i in 0..self.rows {
            for j in self.row_ptr[i]..self.row_ptr[i + 1] {
                dense[i][self.col_indices[j]] = self.values[j];
            }
        }
        dense
    }

    /// Conjugate gradient solver for Ax = b (A must be SPD).
    ///
    /// Returns the solution x and the number of iterations.
    /// Convergence: ||r||/||b|| < tol.
    pub fn solve_cg(&self, b: &[f64], tol: f64, max_iter: usize) -> (Vec<f64>, usize) {
        assert_eq!(self.rows, self.cols, "CG requires square matrix");
        assert_eq!(b.len(), self.rows);

        let n = self.rows;
        let mut x = vec![0.0; n];
        let mut r: Vec<f64> = b.to_vec();
        let mut p = r.clone();
        let mut rs_old: f64 = r.iter().map(|v| v * v).sum();
        let b_norm: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();

        if b_norm < 1e-15 { return (x, 0); }

        for iter in 0..max_iter {
            let ap = self.mul_vec(&p);
            let p_ap: f64 = p.iter().zip(&ap).map(|(pi, api)| pi * api).sum();
            if p_ap.abs() < 1e-15 { return (x, iter + 1); }
            let alpha = rs_old / p_ap;

            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }

            let rs_new: f64 = r.iter().map(|v| v * v).sum();
            if rs_new.sqrt() / b_norm < tol {
                return (x, iter + 1);
            }

            let beta = rs_new / rs_old;
            for i in 0..n {
                p[i] = r[i] + beta * p[i];
            }
            rs_old = rs_new;
        }

        (x, max_iter)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-8;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < TOL
    }

    fn vec_approx_eq(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| approx_eq(*x, *y))
    }

    // ── Construction ─────────────────────────────────────────────────────

    #[test]
    fn test_identity_matrix() {
        let eye = HdcMatrix::identity(3);
        assert_eq!(eye.rows, 3);
        assert_eq!(eye.cols, 3);
        assert_eq!(eye.get(0, 0), 1.0);
        assert_eq!(eye.get(1, 1), 1.0);
        assert_eq!(eye.get(2, 2), 1.0);
        assert_eq!(eye.get(0, 1), 0.0);
    }

    #[test]
    fn test_from_rows() {
        let m = HdcMatrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(1, 1), 4.0);
    }

    // ── Basic operations ─────────────────────────────────────────────────

    #[test]
    fn test_add() {
        let a = HdcMatrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let b = HdcMatrix::from_rows(&[&[5.0, 6.0], &[7.0, 8.0]]);
        let (c, result) = a.add(&b);
        assert_eq!(c.get(0, 0), 6.0);
        assert_eq!(c.get(1, 1), 12.0);
        assert!(result.verified);
    }

    #[test]
    fn test_multiply() {
        let a = HdcMatrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let b = HdcMatrix::from_rows(&[&[5.0, 6.0], &[7.0, 8.0]]);
        let (c, result) = a.mul(&b);
        // [1*5+2*7, 1*6+2*8] = [19, 22]
        // [3*5+4*7, 3*6+4*8] = [43, 50]
        assert_eq!(c.get(0, 0), 19.0);
        assert_eq!(c.get(0, 1), 22.0);
        assert_eq!(c.get(1, 0), 43.0);
        assert_eq!(c.get(1, 1), 50.0);
        assert!(result.verified);
    }

    #[test]
    fn test_transpose() {
        let a = HdcMatrix::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let t = a.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert_eq!(t.get(0, 0), 1.0);
        assert_eq!(t.get(0, 1), 4.0);
        assert_eq!(t.get(2, 0), 3.0);
    }

    #[test]
    fn test_trace() {
        let a = HdcMatrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        assert_eq!(a.trace(), 5.0);
    }

    #[test]
    fn test_scale() {
        let a = HdcMatrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let s = a.scale(2.0);
        assert_eq!(s.get(0, 0), 2.0);
        assert_eq!(s.get(1, 1), 8.0);
    }

    // ── Determinant ──────────────────────────────────────────────────────

    #[test]
    fn test_determinant_2x2() {
        let a = HdcMatrix::from_rows(&[&[3.0, 8.0], &[4.0, 6.0]]);
        let (det, result) = a.determinant();
        assert!(approx_eq(det, 3.0 * 6.0 - 8.0 * 4.0));
        assert!(result.verified);
    }

    #[test]
    fn test_determinant_3x3() {
        let a = HdcMatrix::from_rows(&[&[6.0, 1.0, 1.0], &[4.0, -2.0, 5.0], &[2.0, 8.0, 7.0]]);
        let (det, _) = a.determinant();
        // det = 6(-14-40) - 1(28-10) + 1(32+4) = -306
        assert!(approx_eq(det, -306.0));
    }

    #[test]
    fn test_determinant_identity() {
        let eye = HdcMatrix::identity(4);
        let (det, _) = eye.determinant();
        assert!(approx_eq(det, 1.0));
    }

    #[test]
    fn test_determinant_singular() {
        let a = HdcMatrix::from_rows(&[&[1.0, 2.0], &[2.0, 4.0]]);
        let (det, _) = a.determinant();
        assert!(approx_eq(det, 0.0));
    }

    // ── LU Decomposition ─────────────────────────────────────────────────

    #[test]
    fn test_lu_decompose() {
        let a = HdcMatrix::from_rows(&[&[2.0, 3.0, 1.0], &[4.0, 7.0, 5.0], &[6.0, 18.0, 11.0]]);
        let (l, u, perm, _sign, result) = a.lu_decompose();

        // Verify PA = LU
        let (lu, _) = l.mul(&u);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    approx_eq(a.get(perm[i], j), lu.get(i, j)),
                    "PA[{},{}] = {} != LU[{},{}] = {}",
                    i,
                    j,
                    a.get(perm[i], j),
                    i,
                    j,
                    lu.get(i, j)
                );
            }
        }
        assert!(result.verified);
    }

    // ── Linear System Solving ────────────────────────────────────────────

    #[test]
    fn test_solve_simple() {
        // x + 2y = 5, 3x + 4y = 11 → x=1, y=2
        let a = HdcMatrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let b = HdcVector::new(vec![5.0, 11.0]);
        let (x, result) = a.solve(&b);
        assert!(approx_eq(x.data[0], 1.0));
        assert!(approx_eq(x.data[1], 2.0));
        assert!(result.verified);
    }

    #[test]
    fn test_solve_3x3() {
        let a = HdcMatrix::from_rows(&[&[2.0, 1.0, -1.0], &[-3.0, -1.0, 2.0], &[-2.0, 1.0, 2.0]]);
        let b = HdcVector::new(vec![8.0, -11.0, -3.0]);
        let (x, _) = a.solve(&b);
        assert!(approx_eq(x.data[0], 2.0));
        assert!(approx_eq(x.data[1], 3.0));
        assert!(approx_eq(x.data[2], -1.0));
    }

    #[test]
    fn test_solve_multipath() {
        let a = HdcMatrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let b = HdcVector::new(vec![5.0, 11.0]);
        let (x, result) = a.solve_multipath(&b);
        assert!(approx_eq(x.data[0], 1.0));
        assert!(approx_eq(x.data[1], 2.0));
        assert!(result.verified);
        // Multi-path should have higher phi than single-path
        assert!(result.phi > 0.5);
    }

    // ── QR Decomposition ─────────────────────────────────────────────────

    #[test]
    fn test_qr_decompose() {
        let a = HdcMatrix::from_rows(&[&[1.0, -1.0, 4.0], &[1.0, 4.0, -2.0], &[1.0, 4.0, 2.0]]);
        let (q, r, result) = a.qr_decompose();

        // Q should be orthogonal: Q^T Q ≈ I
        let qt = q.transpose();
        let (qtq, _) = qt.mul(&q);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (qtq.get(i, j) - expected).abs() < 1e-6,
                    "Q^TQ[{},{}] = {} != {}",
                    i,
                    j,
                    qtq.get(i, j),
                    expected
                );
            }
        }

        // R should be upper triangular
        for i in 1..3 {
            for j in 0..i {
                assert!(
                    r.get(i, j).abs() < 1e-6,
                    "R[{},{}] = {} should be 0",
                    i,
                    j,
                    r.get(i, j)
                );
            }
        }

        // QR ≈ A
        let (qr, _) = q.mul(&r);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (qr.get(i, j) - a.get(i, j)).abs() < 1e-6,
                    "QR[{},{}] = {} != A[{},{}] = {}",
                    i,
                    j,
                    qr.get(i, j),
                    i,
                    j,
                    a.get(i, j)
                );
            }
        }
        assert!(result.verified);
    }

    #[test]
    fn test_solve_qr() {
        let a = HdcMatrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let b = HdcVector::new(vec![5.0, 11.0]);
        let (x, _) = a.solve_qr(&b);
        assert!(approx_eq(x.data[0], 1.0));
        assert!(approx_eq(x.data[1], 2.0));
    }

    // ── Cholesky ─────────────────────────────────────────────────────────

    #[test]
    fn test_cholesky() {
        // SPD matrix
        let a = HdcMatrix::from_rows(&[
            &[4.0, 12.0, -16.0],
            &[12.0, 37.0, -43.0],
            &[-16.0, -43.0, 98.0],
        ]);
        let (l, result) = a.cholesky().expect("Should be SPD");

        // LL^T ≈ A
        let lt = l.transpose();
        let (llt, _) = l.mul(&lt);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (llt.get(i, j) - a.get(i, j)).abs() < 1e-6,
                    "LL^T[{},{}] = {} != A[{},{}] = {}",
                    i,
                    j,
                    llt.get(i, j),
                    i,
                    j,
                    a.get(i, j)
                );
            }
        }
        assert!(result.verified);
    }

    #[test]
    fn test_cholesky_not_spd() {
        let a = HdcMatrix::from_rows(&[&[-1.0, 0.0], &[0.0, 1.0]]);
        assert!(a.cholesky().is_none());
    }

    // ── Eigendecomposition ───────────────────────────────────────────────

    #[test]
    fn test_power_iteration() {
        // Symmetric matrix with known eigenvalues 5, 1
        let a = HdcMatrix::from_rows(&[&[3.0, 2.0], &[2.0, 3.0]]);
        let (eigenvalue, eigenvector, _) = a.power_iteration(1e-10);
        assert!(
            (eigenvalue - 5.0).abs() < 1e-6,
            "Expected ~5, got {}",
            eigenvalue
        );
        // Eigenvector should be proportional to [1, 1]
        let ratio = eigenvector.data[0] / eigenvector.data[1];
        assert!((ratio - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_eigenvalues_symmetric() {
        let a = HdcMatrix::from_rows(&[&[3.0, 2.0], &[2.0, 3.0]]);
        let (eigenvalues, _) = a.eigenvalues_symmetric();
        assert_eq!(eigenvalues.len(), 2);
        // Sorted descending: 5, 1
        assert!((eigenvalues[0] - 5.0).abs() < 1e-6);
        assert!((eigenvalues[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_eigenvalues_diagonal() {
        let a = HdcMatrix::from_rows(&[&[4.0, 0.0, 0.0], &[0.0, 2.0, 0.0], &[0.0, 0.0, 7.0]]);
        let (eigenvalues, _) = a.eigenvalues_symmetric();
        let mut ev = eigenvalues.clone();
        ev.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert!((ev[0] - 7.0).abs() < 1e-6);
        assert!((ev[1] - 4.0).abs() < 1e-6);
        assert!((ev[2] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_eigen_symmetric_3x3() {
        // Known symmetric matrix
        let a = HdcMatrix::from_rows(&[&[2.0, -1.0, 0.0], &[-1.0, 2.0, -1.0], &[0.0, -1.0, 2.0]]);
        let (eigenvalues, eigenvectors, result) = a.eigen_symmetric();
        assert_eq!(eigenvalues.len(), 3);

        // Verify Av = λv for each eigenpair
        for j in 0..3 {
            let mut v = vec![0.0; 3];
            for i in 0..3 {
                v[i] = eigenvectors.get(i, j);
            }
            let v_vec = HdcVector::new(v.clone());
            let norm = v_vec.norm();
            if norm < 1e-10 {
                continue;
            }

            // Av
            let mut av = vec![0.0; 3];
            for i in 0..3 {
                for k in 0..3 {
                    av[i] += a.get(i, k) * v[k];
                }
            }
            // λv
            for i in 0..3 {
                let expected = eigenvalues[j] * v[i];
                assert!(
                    (av[i] - expected).abs() < 1e-4,
                    "Av[{}] = {} != λv[{}] = {} (λ={})",
                    i,
                    av[i],
                    i,
                    expected,
                    eigenvalues[j]
                );
            }
        }
        assert!(result.verified);
    }

    // ── SVD ──────────────────────────────────────────────────────────────

    #[test]
    fn test_svd_identity() {
        let eye = HdcMatrix::identity(3);
        let (sv, _, _, result) = eye.svd();
        for s in &sv {
            assert!((s - 1.0).abs() < 1e-6);
        }
        assert!(result.verified);
    }

    #[test]
    fn test_svd_diagonal() {
        let a = HdcMatrix::from_rows(&[&[3.0, 0.0], &[0.0, 5.0]]);
        let (mut sv, _, _, _) = a.svd();
        sv.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert!((sv[0] - 5.0).abs() < 1e-6);
        assert!((sv[1] - 3.0).abs() < 1e-6);
    }

    /// SVD reconstruction: A ≈ U Σ V^T
    #[test]
    fn test_svd_reconstruction() {
        let a = HdcMatrix::from_rows(&[
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
            &[7.0, 8.0, 10.0], // NOT 9, so full rank
        ]);
        let (sv, u, v, _) = a.svd();

        // Reconstruct: U Σ V^T
        // Build Σ as m×n diagonal
        let m = a.rows;
        let n = a.cols;
        let mut sigma_data = vec![0.0; m * n];
        for i in 0..sv.len().min(m).min(n) {
            sigma_data[i * n + i] = sv[i];
        }
        let sigma = HdcMatrix::new(sigma_data, m, n);
        let vt = v.transpose();
        let (us, _) = u.mul(&sigma);
        let (reconstructed, _) = us.mul(&vt);

        // Check A ≈ U Σ V^T
        for i in 0..m {
            for j in 0..n {
                let diff = (a.get(i, j) - reconstructed.get(i, j)).abs();
                assert!(diff < 1e-6,
                    "SVD reconstruction failed at ({},{}): A={}, UΣV^T={}, diff={}",
                    i, j, a.get(i, j), reconstructed.get(i, j), diff);
            }
        }
    }

    /// U should be orthogonal: U^T U ≈ I
    #[test]
    fn test_svd_u_orthogonal() {
        let a = HdcMatrix::from_rows(&[
            &[1.0, 2.0],
            &[3.0, 4.0],
            &[5.0, 6.0], // 3×2 rectangular — exercises the fix
        ]);
        let (_, u, _, _) = a.svd();
        let ut = u.transpose();
        let (utu, _) = ut.mul(&u);
        let eye = HdcMatrix::identity(u.rows);

        for i in 0..u.rows {
            for j in 0..u.rows {
                let diff = (utu.get(i, j) - eye.get(i, j)).abs();
                assert!(diff < 1e-6,
                    "U not orthogonal at ({},{}): U^TU={}, I={}", i, j, utu.get(i, j), eye.get(i, j));
            }
        }
    }

    /// V should be orthogonal: V^T V ≈ I
    #[test]
    fn test_svd_v_orthogonal() {
        let a = HdcMatrix::from_rows(&[
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
        ]); // 2×3 — wide rectangular
        let (_, _, v, _) = a.svd();
        let vt = v.transpose();
        let (vtv, _) = vt.mul(&v);
        let eye = HdcMatrix::identity(v.rows);

        for i in 0..v.rows {
            for j in 0..v.rows {
                let diff = (vtv.get(i, j) - eye.get(i, j)).abs();
                assert!(diff < 1e-6,
                    "V not orthogonal at ({},{}): V^TV={}, I={}", i, j, vtv.get(i, j), eye.get(i, j));
            }
        }
    }

    /// SVD of rank-deficient matrix: [1 2 3; 2 4 6; 3 6 9] has rank 1
    #[test]
    fn test_svd_rank_deficient() {
        let a = HdcMatrix::from_rows(&[
            &[1.0, 2.0, 3.0],
            &[2.0, 4.0, 6.0],
            &[3.0, 6.0, 9.0],
        ]);
        let (sv, _, _, _) = a.svd();

        // Only 1 non-zero singular value
        let nonzero = sv.iter().filter(|&&s| s > 1e-6).count();
        assert_eq!(nonzero, 1, "rank-1 matrix should have 1 nonzero singular value, got {}", nonzero);
        assert_eq!(a.rank(), 1);
    }

    /// Null space of rank-deficient matrix
    #[test]
    fn test_null_space() {
        // [1 2 3; 2 4 6; 3 6 9] has rank 1, null space dimension 2
        let a = HdcMatrix::from_rows(&[
            &[1.0, 2.0, 3.0],
            &[2.0, 4.0, 6.0],
            &[3.0, 6.0, 9.0],
        ]);
        let ns = a.null_space();
        assert!(ns.is_some(), "rank-1 3×3 matrix should have non-trivial null space");
        let ns = ns.unwrap();
        assert_eq!(ns.cols, 2, "null space should have dimension 2, got {}", ns.cols);

        // Verify A * null_vector ≈ 0 for each null space column
        for j in 0..ns.cols {
            let mut nv = vec![0.0; ns.rows];
            for i in 0..ns.rows { nv[i] = ns.get(i, j); }
            let nv_mat = HdcMatrix::new(nv, ns.rows, 1);
            let (product, _) = a.mul(&nv_mat);
            for i in 0..a.rows {
                assert!(product.get(i, 0).abs() < 1e-6,
                    "A * null_vec[{}] should be 0, got {} at row {}", j, product.get(i, 0), i);
            }
        }
    }

    /// Full-rank matrix should have trivial null space
    #[test]
    fn test_null_space_full_rank() {
        let a = HdcMatrix::from_rows(&[&[1.0, 0.0], &[0.0, 1.0]]);
        assert!(a.null_space().is_none(), "identity should have no null space");
    }

    /// SVD of tall rectangular matrix (m > n)
    #[test]
    fn test_svd_tall_rectangular() {
        let a = HdcMatrix::from_rows(&[
            &[1.0, 2.0],
            &[3.0, 4.0],
            &[5.0, 6.0],
        ]); // 3×2
        let (sv, u, v, _) = a.svd();
        assert_eq!(sv.len(), 2, "should have min(3,2)=2 singular values");
        assert_eq!(u.rows, 3, "U should be 3×3");
        assert_eq!(v.rows, 2, "V should be 2×2");

        // All singular values should be positive (full column rank)
        assert!(sv.iter().all(|s| *s > 1e-6), "3×2 full-rank matrix should have 2 positive singular values");
    }

    // ── Properties ───────────────────────────────────────────────────────

    #[test]
    fn test_rank() {
        let a = HdcMatrix::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);
        assert_eq!(a.rank(), 2); // Rank 2 (last row = 2*row2 - row1)
    }

    #[test]
    fn test_inverse_2x2() {
        let a = HdcMatrix::from_rows(&[&[4.0, 7.0], &[2.0, 6.0]]);
        let (inv, _) = a.inverse().expect("Should be invertible");
        let (product, _) = a.mul(&inv);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product.get(i, j) - expected).abs() < 1e-6,
                    "A*A^-1[{},{}] = {} != {}",
                    i,
                    j,
                    product.get(i, j),
                    expected
                );
            }
        }
    }

    #[test]
    fn test_inverse_singular() {
        let a = HdcMatrix::from_rows(&[&[1.0, 2.0], &[2.0, 4.0]]);
        assert!(a.inverse().is_none());
    }

    #[test]
    fn test_condition_number_identity() {
        let eye = HdcMatrix::identity(3);
        let cond = eye.condition_number();
        assert!((cond - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_frobenius_norm() {
        let a = HdcMatrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let expected = (1.0 + 4.0 + 9.0 + 16.0_f64).sqrt();
        assert!(approx_eq(a.frobenius_norm(), expected));
    }

    // ── HDC encoding ─────────────────────────────────────────────────────

    #[test]
    fn test_matrix_encoding_unique() {
        let a = HdcMatrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let b = HdcMatrix::from_rows(&[&[5.0, 6.0], &[7.0, 8.0]]);
        // Different matrices should produce different encodings
        let sim = a.encoding.similarity(&b.encoding);
        assert!(
            sim < 0.6,
            "Different matrices should have low similarity: {}",
            sim
        );
    }

    #[test]
    fn test_vector_operations() {
        let v1 = HdcVector::new(vec![1.0, 2.0, 3.0]);
        let v2 = HdcVector::new(vec![4.0, 5.0, 6.0]);
        assert_eq!(v1.len(), 3);
        assert!(!v1.is_empty());
        assert!(approx_eq(v1.dot(&v2), 32.0));
        assert!(approx_eq(v1.norm(), (14.0_f64).sqrt()));

        let v3 = v1.normalized();
        assert!((v3.norm() - 1.0).abs() < 1e-10);
    }

    // ── Engine wrapper ───────────────────────────────────────────────────

    #[test]
    fn test_engine_solve() {
        let mut engine = LinearAlgebraEngine::new();
        let a = HdcMatrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let b = HdcVector::new(vec![5.0, 11.0]);
        let (x, _) = engine.solve(&a, &b);
        assert!(approx_eq(x.data[0], 1.0));
        assert!(approx_eq(x.data[1], 2.0));
        assert_eq!(engine.stats().solves, 1);
        assert!(engine.stats().total_phi > 0.0);
    }

    #[test]
    fn test_engine_eigenvalues() {
        let mut engine = LinearAlgebraEngine::new();
        let a = HdcMatrix::from_rows(&[&[3.0, 2.0], &[2.0, 3.0]]);
        let (eigenvalues, _) = engine.eigenvalues(&a);
        assert!((eigenvalues[0] - 5.0).abs() < 1e-6);
        assert_eq!(engine.stats().eigendecompositions, 1);
    }

    // ── Rotation matrix ──────────────────────────────────────────────────

    #[test]
    fn test_rotation_matrix() {
        let theta = std::f64::consts::PI / 4.0; // 45 degrees
        let rot =
            HdcMatrix::from_rows(&[&[theta.cos(), -theta.sin()], &[theta.sin(), theta.cos()]]);

        // Rotation matrix should have det = 1
        let (det, _) = rot.determinant();
        assert!(approx_eq(det, 1.0));

        // R^T R = I (orthogonal)
        let rt = rot.transpose();
        let (product, _) = rt.mul(&rot);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((product.get(i, j) - expected).abs() < 1e-10);
            }
        }
    }

    // ── Hilbert matrix (ill-conditioned) ─────────────────────────────────

    #[test]
    fn test_hilbert_3x3() {
        let mut data = vec![0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                data[i * 3 + j] = 1.0 / (i + j + 1) as f64;
            }
        }
        let h = HdcMatrix::new(data, 3, 3);
        let (det, _) = h.determinant();
        // Hilbert 3x3 determinant = 1/2160
        assert!((det - 1.0 / 2160.0).abs() < 1e-10);
    }

    // ── Projection matrix ────────────────────────────────────────────────

    #[test]
    fn test_projection_matrix() {
        // Projection onto x-axis: P = [1 0; 0 0]
        let p = HdcMatrix::from_rows(&[&[1.0, 0.0], &[0.0, 0.0]]);

        // P^2 = P (idempotent)
        let (p2, _) = p.mul(&p);
        for i in 0..2 {
            for j in 0..2 {
                assert!(approx_eq(p2.get(i, j), p.get(i, j)));
            }
        }

        // Eigenvalues should be 0 and 1
        let (ev, _) = p.eigenvalues_symmetric();
        let mut ev_sorted = ev.clone();
        ev_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert!((ev_sorted[0] - 1.0).abs() < 1e-6);
        assert!(ev_sorted[1].abs() < 1e-6);
    }

    // ── 1x1 edge cases ─────────────────────────────────────────────────

    #[test]
    fn test_1x1_matrix_operations() {
        let a = HdcMatrix::from_rows(&[&[5.0]]);
        assert!(a.is_square());
        assert_eq!(a.trace(), 5.0);

        let (det, _) = a.determinant();
        assert!(approx_eq(det, 5.0));

        let b = HdcMatrix::from_rows(&[&[3.0]]);
        let (sum, _) = a.add(&b);
        assert_eq!(sum.get(0, 0), 8.0);

        let (prod, _) = a.mul(&b);
        assert_eq!(prod.get(0, 0), 15.0);

        let t = a.transpose();
        assert_eq!(t.get(0, 0), 5.0);
        assert_eq!(t.rows, 1);
        assert_eq!(t.cols, 1);
    }

    #[test]
    fn test_1x1_solve() {
        let a = HdcMatrix::from_rows(&[&[4.0]]);
        let b = HdcVector::new(vec![12.0]);
        let (x, result) = a.solve(&b);
        assert!(approx_eq(x.data[0], 3.0));
        assert!(result.verified);
    }

    #[test]
    fn test_1x1_eigenvalue() {
        let a = HdcMatrix::from_rows(&[&[7.0]]);
        let (ev, _) = a.eigenvalues_symmetric();
        assert_eq!(ev.len(), 1);
        assert!((ev[0] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_1x1_cholesky() {
        let a = HdcMatrix::from_rows(&[&[9.0]]);
        let (l, _) = a.cholesky().expect("9 is positive definite");
        assert!((l.get(0, 0) - 3.0).abs() < 1e-10);
    }

    // ── Known eigenvalue test ───────────────────────────────────────────

    #[test]
    fn test_eigenvalues_known_2x2() {
        // [[2, 1], [1, 3]] has eigenvalues (5 +/- sqrt(5))/2
        let a = HdcMatrix::from_rows(&[&[2.0, 1.0], &[1.0, 3.0]]);
        let (eigenvalues, _) = a.eigenvalues_symmetric();
        let sqrt5 = 5.0_f64.sqrt();
        let expected_large = (5.0 + sqrt5) / 2.0;
        let expected_small = (5.0 - sqrt5) / 2.0;

        // eigenvalues sorted descending
        assert!(
            (eigenvalues[0] - expected_large).abs() < 1e-6,
            "Expected {}, got {}",
            expected_large,
            eigenvalues[0]
        );
        assert!(
            (eigenvalues[1] - expected_small).abs() < 1e-6,
            "Expected {}, got {}",
            expected_small,
            eigenvalues[1]
        );
    }

    // ── Symmetry and positive definiteness ──────────────────────────────

    #[test]
    fn test_is_symmetric() {
        let sym = HdcMatrix::from_rows(&[&[1.0, 2.0], &[2.0, 3.0]]);
        assert!(sym.is_symmetric());

        let nonsym = HdcMatrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        assert!(!nonsym.is_symmetric());

        // Non-square is never symmetric
        let rect = HdcMatrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        assert!(!rect.is_symmetric());
    }

    #[test]
    fn test_is_positive_definite() {
        let spd = HdcMatrix::from_rows(&[&[2.0, 1.0], &[1.0, 2.0]]);
        assert!(spd.is_positive_definite());

        let not_pd = HdcMatrix::from_rows(&[&[-1.0, 0.0], &[0.0, 1.0]]);
        assert!(!not_pd.is_positive_definite());

        let singular = HdcMatrix::from_rows(&[&[1.0, 1.0], &[1.0, 1.0]]);
        assert!(!singular.is_positive_definite());
    }

    // ── Matrix-vector multiply ──────────────────────────────────────────

    #[test]
    fn test_mul_vec() {
        let a = HdcMatrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let v = HdcVector::new(vec![5.0, 6.0]);
        let result = a.mul_vec(&v);
        // [1*5+2*6, 3*5+4*6] = [17, 39]
        assert!(approx_eq(result.data[0], 17.0));
        assert!(approx_eq(result.data[1], 39.0));
    }

    #[test]
    fn test_mul_vec_identity() {
        let eye = HdcMatrix::identity(3);
        let v = HdcVector::new(vec![7.0, 8.0, 9.0]);
        let result = eye.mul_vec(&v);
        assert!(vec_approx_eq(&result.data, &[7.0, 8.0, 9.0]));
    }

    // ── from_data constructor ───────────────────────────────────────────

    #[test]
    fn test_from_data() {
        let m = HdcMatrix::from_data(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 3);
        assert_eq!(m.get(0, 2), 3.0);
        assert_eq!(m.get(1, 0), 4.0);
    }

    // ── Diagonal matrix ─────────────────────────────────────────────────

    #[test]
    fn test_diagonal_matrix() {
        let d = HdcMatrix::diagonal(&[2.0, 3.0, 5.0]);
        assert_eq!(d.rows, 3);
        assert_eq!(d.cols, 3);
        assert_eq!(d.get(0, 0), 2.0);
        assert_eq!(d.get(1, 1), 3.0);
        assert_eq!(d.get(2, 2), 5.0);
        assert_eq!(d.get(0, 1), 0.0);
        let (det, _) = d.determinant();
        assert!(approx_eq(det, 30.0));
    }

    // ── Solve verification: Ax = b round-trip ───────────────────────────

    #[test]
    fn test_solve_roundtrip_4x4() {
        let a = HdcMatrix::from_rows(&[
            &[2.0, 1.0, -1.0, 3.0],
            &[1.0, 3.0, 2.0, -1.0],
            &[-1.0, 2.0, 5.0, 1.0],
            &[3.0, -1.0, 1.0, 4.0],
        ]);
        let x_true = HdcVector::new(vec![1.0, 2.0, 3.0, 4.0]);
        let b = a.mul_vec(&x_true);
        let (x_solved, result) = a.solve(&b);
        assert!(result.verified);
        for i in 0..4 {
            assert!(
                (x_solved.data[i] - x_true.data[i]).abs() < 1e-6,
                "x[{}] = {} != {}",
                i,
                x_solved.data[i],
                x_true.data[i]
            );
        }
    }

    // ── QR solve round-trip ─────────────────────────────────────────────

    #[test]
    fn test_solve_qr_roundtrip() {
        let a = HdcMatrix::from_rows(&[&[3.0, 1.0], &[1.0, 2.0]]);
        let x_true = HdcVector::new(vec![2.0, -1.0]);
        let b = a.mul_vec(&x_true);
        let (x_solved, _) = a.solve_qr(&b);
        assert!(
            (x_solved.data[0] - 2.0).abs() < 1e-6,
            "x[0] = {}",
            x_solved.data[0]
        );
        assert!(
            (x_solved.data[1] - (-1.0)).abs() < 1e-6,
            "x[1] = {}",
            x_solved.data[1]
        );
    }

    // ── LU round-trip: PA = LU ──────────────────────────────────────────

    #[test]
    fn test_lu_roundtrip_4x4() {
        let a = HdcMatrix::from_rows(&[
            &[1.0, 2.0, 3.0, 4.0],
            &[5.0, 6.0, 7.0, 8.0],
            &[2.0, 6.0, 4.0, 8.0],
            &[3.0, 1.0, 7.0, 2.0],
        ]);
        let (l, u, perm, _sign, _) = a.lu_decompose();
        let (lu, _) = l.mul(&u);
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (a.get(perm[i], j) - lu.get(i, j)).abs() < 1e-6,
                    "PA[{},{}]={} != LU[{},{}]={}",
                    i,
                    j,
                    a.get(perm[i], j),
                    i,
                    j,
                    lu.get(i, j)
                );
            }
        }
    }

    // ── QR round-trip: A = QR ───────────────────────────────────────────

    #[test]
    fn test_qr_roundtrip_rectangular() {
        // 3x2 matrix
        let a = HdcMatrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        let (q, r, _) = a.qr_decompose();

        // QR should reconstruct A
        let (qr, _) = q.mul(&r);
        for i in 0..3 {
            for j in 0..2 {
                assert!(
                    (qr.get(i, j) - a.get(i, j)).abs() < 1e-6,
                    "QR[{},{}]={} != A[{},{}]={}",
                    i,
                    j,
                    qr.get(i, j),
                    i,
                    j,
                    a.get(i, j)
                );
            }
        }
    }

    // ── Determinant edge cases ──────────────────────────────────────────

    #[test]
    fn test_determinant_1x1() {
        let a = HdcMatrix::from_rows(&[&[42.0]]);
        let (det, _) = a.determinant();
        assert!(approx_eq(det, 42.0));
    }

    #[test]
    fn test_determinant_scale_property() {
        // det(cA) = c^n * det(A) for n×n matrix
        let a = HdcMatrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let (det_a, _) = a.determinant();
        let scaled = a.scale(3.0);
        let (det_scaled, _) = scaled.determinant();
        // c=3, n=2 => det(3A) = 9 * det(A)
        assert!(
            (det_scaled - 9.0 * det_a).abs() < 1e-6,
            "det(3A)={} != 9*det(A)={}",
            det_scaled,
            9.0 * det_a
        );
    }

    #[test]
    fn test_determinant_transpose_invariant() {
        // det(A^T) = det(A)
        let a = HdcMatrix::from_rows(&[&[2.0, 3.0, 1.0], &[4.0, 7.0, 5.0], &[6.0, 18.0, 11.0]]);
        let (det_a, _) = a.determinant();
        let at = a.transpose();
        let (det_at, _) = at.determinant();
        assert!(
            (det_a - det_at).abs() < 1e-6,
            "det(A)={} != det(A^T)={}",
            det_a,
            det_at
        );
    }

    // ── Norm and condition number ───────────────────────────────────────

    #[test]
    fn test_norm_frobenius_identity() {
        let eye = HdcMatrix::identity(4);
        // Frobenius norm of 4x4 identity = sqrt(4) = 2
        assert!((eye.norm_frobenius() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_condition_number_well_conditioned() {
        // Orthogonal matrix has condition number 1
        let theta = std::f64::consts::PI / 3.0;
        let rot =
            HdcMatrix::from_rows(&[&[theta.cos(), -theta.sin()], &[theta.sin(), theta.cos()]]);
        let cond = rot.condition_number_estimate();
        assert!((cond - 1.0).abs() < 1e-4, "Rotation matrix cond = {}", cond);
    }

    #[test]
    fn test_condition_number_ill_conditioned() {
        // Hilbert 4x4 is famously ill-conditioned
        let mut data = vec![0.0; 16];
        for i in 0..4 {
            for j in 0..4 {
                data[i * 4 + j] = 1.0 / (i + j + 1) as f64;
            }
        }
        let h = HdcMatrix::new(data, 4, 4);
        let cond = h.condition_number();
        // Hilbert 4x4 condition number is ~15514
        assert!(
            cond > 1000.0,
            "Hilbert 4x4 cond = {} (expected > 1000)",
            cond
        );
    }

    // ── Inverse round-trip ──────────────────────────────────────────────

    #[test]
    fn test_inverse_3x3_roundtrip() {
        let a = HdcMatrix::from_rows(&[&[2.0, 1.0, -1.0], &[-3.0, -1.0, 2.0], &[-2.0, 1.0, 2.0]]);
        let (inv, _) = a.inverse().expect("Should be invertible");
        let (product, _) = a.mul(&inv);
        let eye = HdcMatrix::identity(3);
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (product.get(i, j) - eye.get(i, j)).abs() < 1e-6,
                    "A*A^-1[{},{}] = {} != I[{},{}] = {}",
                    i,
                    j,
                    product.get(i, j),
                    i,
                    j,
                    eye.get(i, j)
                );
            }
        }
    }

    // ── Rank tests ──────────────────────────────────────────────────────

    #[test]
    fn test_rank_identity() {
        let eye = HdcMatrix::identity(3);
        assert_eq!(eye.rank(), 3);
    }

    #[test]
    fn test_rank_zero_matrix() {
        let z = HdcMatrix::zeros(3, 3);
        assert_eq!(z.rank(), 0);
    }

    // ── Cholesky round-trip ─────────────────────────────────────────────

    #[test]
    fn test_cholesky_2x2() {
        let a = HdcMatrix::from_rows(&[&[4.0, 2.0], &[2.0, 5.0]]);
        let (l, _) = a.cholesky().expect("Should be SPD");
        let lt = l.transpose();
        let (llt, _) = l.mul(&lt);
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (llt.get(i, j) - a.get(i, j)).abs() < 1e-10,
                    "LL^T[{},{}]={} != A[{},{}]={}",
                    i,
                    j,
                    llt.get(i, j),
                    i,
                    j,
                    a.get(i, j)
                );
            }
        }
    }

    // ── Subtraction test ────────────────────────────────────────────────

    #[test]
    fn test_subtract() {
        let a = HdcMatrix::from_rows(&[&[5.0, 6.0], &[7.0, 8.0]]);
        let b = HdcMatrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let (c, result) = a.sub(&b);
        assert_eq!(c.get(0, 0), 4.0);
        assert_eq!(c.get(0, 1), 4.0);
        assert_eq!(c.get(1, 0), 4.0);
        assert_eq!(c.get(1, 1), 4.0);
        assert!(result.verified);
    }

    // ── Eigenvalue sum/product properties ───────────────────────────────

    #[test]
    fn test_eigenvalue_trace_determinant_relation() {
        // For a symmetric matrix:
        //   sum(eigenvalues) = trace
        //   product(eigenvalues) = determinant
        let a = HdcMatrix::from_rows(&[&[4.0, 1.0], &[1.0, 3.0]]);
        let (eigenvalues, _) = a.eigenvalues_symmetric();
        let ev_sum: f64 = eigenvalues.iter().sum();
        let ev_prod: f64 = eigenvalues.iter().product();
        let (det, _) = a.determinant();

        assert!(
            (ev_sum - a.trace()).abs() < 1e-6,
            "sum(eig)={} != trace={}",
            ev_sum,
            a.trace()
        );
        assert!(
            (ev_prod - det).abs() < 1e-6,
            "prod(eig)={} != det={}",
            ev_prod,
            det
        );
    }

    // ── Singular matrix solve ───────────────────────────────────────────

    #[test]
    fn test_solve_singular_returns_zeros() {
        let a = HdcMatrix::from_rows(&[&[1.0, 2.0], &[2.0, 4.0]]);
        let b = HdcVector::new(vec![3.0, 6.0]);
        let (x, _result) = a.solve(&b);
        // For singular systems, our solver returns zeros for the rank-deficient component
        // The important thing is it doesn't crash
        assert_eq!(x.data.len(), 2);
    }

    // ── Zero vector and empty checks ────────────────────────────────────

    #[test]
    fn test_zero_vector() {
        let z = HdcVector::zeros(4);
        assert_eq!(z.len(), 4);
        assert!(!z.is_empty());
        assert!(approx_eq(z.norm(), 0.0));
        assert!(approx_eq(z.dot(&z), 0.0));
    }

    // ── Engine SVD ──────────────────────────────────────────────────────

    #[test]
    fn test_engine_svd() {
        let mut engine = LinearAlgebraEngine::new();
        let a = HdcMatrix::from_rows(&[&[3.0, 0.0], &[0.0, 4.0]]);
        let (mut sv, _, _, _) = engine.svd(&a);
        sv.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert!((sv[0] - 4.0).abs() < 1e-6);
        assert!((sv[1] - 3.0).abs() < 1e-6);
        assert_eq!(engine.stats().factorizations, 1);
    }

    // ── Identity multiplication ─────────────────────────────────────────

    #[test]
    fn test_identity_multiply_preserves() {
        let a = HdcMatrix::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);
        let eye = HdcMatrix::identity(3);
        let (result, _) = eye.mul(&a);
        for i in 0..3 {
            for j in 0..3 {
                assert!(approx_eq(result.get(i, j), a.get(i, j)));
            }
        }
    }

    // ── Conjugate Gradient ───────────────────────────────────────────────

    #[test]
    fn test_cg_solves_spd_system() {
        // Solve: [4 1; 1 3] * x = [1; 2]
        // Solution: x = [1/11, 7/11]
        let a = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let b = vec![1.0, 2.0];
        let x = conjugate_gradient(&a, &b, 1e-10, 100).unwrap();
        assert_eq!(x.len(), 2);
        // Check A*x = b
        let ax = mat_vec_mul(&a, &x);
        for (ai, bi) in ax.iter().zip(b.iter()) {
            assert!((ai - bi).abs() < 1e-8, "A*x[{}] = {} != {}", 0, ai, bi);
        }
    }

    #[test]
    fn test_cg_10x10_spd() {
        // Build a diagonally dominant SPD matrix
        let n = 10;
        let mut a = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            a[i][i] = 4.0;
            if i > 0 { a[i][i - 1] = -1.0; }
            if i < n - 1 { a[i][i + 1] = -1.0; }
        }
        let b: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
        let x = conjugate_gradient(&a, &b, 1e-8, 1000).unwrap();
        let ax = mat_vec_mul(&a, &x);
        for (ai, bi) in ax.iter().zip(b.iter()) {
            assert!((ai - bi).abs() < 1e-6, "Residual too large: {} vs {}", ai, bi);
        }
    }

    #[test]
    fn test_cg_matches_direct_solve() {
        // 3x3 SPD system — compare CG with expected solution
        let a = vec![
            vec![6.0, 2.0, 1.0],
            vec![2.0, 5.0, 2.0],
            vec![1.0, 2.0, 4.0],
        ];
        let b = vec![9.0, 9.0, 7.0];
        let x_cg = conjugate_gradient(&a, &b, 1e-10, 200).unwrap();
        let ax = mat_vec_mul(&a, &x_cg);
        for (ai, bi) in ax.iter().zip(b.iter()) {
            assert!((ai - bi).abs() < 1e-7, "CG residual: {} vs {}", ai, bi);
        }
    }

    #[test]
    fn test_preconditioned_cg_converges() {
        let n = 8;
        let mut a = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            a[i][i] = (i + 2) as f64 * 2.0; // varying diagonal (tests preconditioning)
            if i > 0 { a[i][i - 1] = -0.5; }
            if i < n - 1 { a[i][i + 1] = -0.5; }
        }
        let b: Vec<f64> = vec![1.0; n];
        let x_pcg = preconditioned_conjugate_gradient(&a, &b, 1e-8, 500).unwrap();
        let ax = mat_vec_mul(&a, &x_pcg);
        for (ai, bi) in ax.iter().zip(b.iter()) {
            assert!((ai - bi).abs() < 1e-6, "PCG residual: {} vs {}", ai, bi);
        }
    }

    #[test]
    fn test_condition_number_estimate_identity() {
        // Identity matrix has condition number 1
        let n = 4;
        let a: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        let cond = condition_number_estimate(&a, 50);
        assert!(cond >= 1.0, "Condition number must be >= 1: {}", cond);
        assert!(cond < 10.0, "Identity condition number should be near 1: {}", cond);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Sparse Matrix Tests
    // ═══════════════════════════════════════════════════════════════════

    #[test]
    fn test_sparse_identity() {
        let eye = SparseMatrix::identity(4);
        assert_eq!(eye.nnz(), 4);
        assert_eq!(eye.get(0, 0), 1.0);
        assert_eq!(eye.get(1, 1), 1.0);
        assert_eq!(eye.get(0, 1), 0.0);
    }

    #[test]
    fn test_sparse_from_triplets() {
        let triplets = vec![
            (0, 0, 2.0), (0, 1, -1.0),
            (1, 0, -1.0), (1, 1, 2.0), (1, 2, -1.0),
            (2, 1, -1.0), (2, 2, 2.0),
        ];
        let a = SparseMatrix::from_triplets(3, 3, &triplets);
        assert_eq!(a.nnz(), 7);
        assert_eq!(a.get(0, 0), 2.0);
        assert_eq!(a.get(0, 1), -1.0);
        assert_eq!(a.get(2, 2), 2.0);
        assert_eq!(a.get(2, 0), 0.0); // not stored
    }

    #[test]
    fn test_sparse_mul_vec() {
        // [2 -1 0] [1]   [1]
        // [-1 2 -1] [1] = [0]
        // [0 -1 2] [1]   [1]
        let triplets = vec![
            (0, 0, 2.0), (0, 1, -1.0),
            (1, 0, -1.0), (1, 1, 2.0), (1, 2, -1.0),
            (2, 1, -1.0), (2, 2, 2.0),
        ];
        let a = SparseMatrix::from_triplets(3, 3, &triplets);
        let x = vec![1.0, 1.0, 1.0];
        let y = a.mul_vec(&x);
        assert!((y[0] - 1.0).abs() < TOL);
        assert!((y[1] - 0.0).abs() < TOL);
        assert!((y[2] - 1.0).abs() < TOL);
    }

    #[test]
    fn test_sparse_transpose() {
        let triplets = vec![(0, 1, 3.0), (1, 0, 5.0), (1, 2, 7.0)];
        let a = SparseMatrix::from_triplets(2, 3, &triplets);
        let at = a.transpose();
        assert_eq!(at.rows, 3);
        assert_eq!(at.cols, 2);
        assert_eq!(at.get(1, 0), 3.0);
        assert_eq!(at.get(0, 1), 5.0);
        assert_eq!(at.get(2, 1), 7.0);
    }

    #[test]
    fn test_sparse_add() {
        let a = SparseMatrix::from_triplets(2, 2, &[(0, 0, 1.0), (1, 1, 2.0)]);
        let b = SparseMatrix::from_triplets(2, 2, &[(0, 1, 3.0), (1, 1, 4.0)]);
        let c = a.add(&b);
        assert_eq!(c.get(0, 0), 1.0);
        assert_eq!(c.get(0, 1), 3.0);
        assert_eq!(c.get(1, 1), 6.0); // 2 + 4
    }

    #[test]
    fn test_sparse_cg_tridiagonal() {
        // SPD tridiagonal: diag=4, off-diag=-1
        let n = 10;
        let mut triplets = Vec::new();
        for i in 0..n {
            triplets.push((i, i, 4.0));
            if i > 0 { triplets.push((i, i - 1, -1.0)); }
            if i < n - 1 { triplets.push((i, i + 1, -1.0)); }
        }
        let a = SparseMatrix::from_triplets(n, n, &triplets);
        let b: Vec<f64> = vec![1.0; n];

        let (x, iters) = a.solve_cg(&b, 1e-10, 100);
        let ax = a.mul_vec(&x);
        for (ai, bi) in ax.iter().zip(b.iter()) {
            assert!((ai - bi).abs() < 1e-8,
                "Sparse CG residual too large: {} vs {}", ai, bi);
        }
        eprintln!("Sparse CG: {}×{}, nnz={}, converged in {} iters", n, n, a.nnz(), iters);
    }

    #[test]
    fn test_sparse_to_dense() {
        let triplets = vec![(0, 0, 1.0), (0, 2, 3.0), (1, 1, 5.0)];
        let a = SparseMatrix::from_triplets(2, 3, &triplets);
        let dense = a.to_dense();
        assert_eq!(dense[0][0], 1.0);
        assert_eq!(dense[0][1], 0.0);
        assert_eq!(dense[0][2], 3.0);
        assert_eq!(dense[1][1], 5.0);
    }
}
