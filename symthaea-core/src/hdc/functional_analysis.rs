// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Functional Analysis
//!
//! Normed spaces, bounded linear operators, spectral theory, Fourier series (L² theory),
//! Fredholm alternative, and Sobolev space norms — all in pure Rust (no external crates).
//!
//! ## Capabilities
//!
//! - **L2Vec** — finite-dimensional ℓ² (Hilbert) space with Gram-Schmidt
//! - **BoundedOperator** — matrix-represented linear maps with adjoint, norm, spectral test
//! - **SpectralDecomposition** — Jacobi eigenvalue algorithm for symmetric matrices
//! - **FourierSeries** — L² Fourier coefficients, Parseval's theorem, convergence rate
//! - **Fredholm alternative** — solvability check + Neumann series solver
//! - **Sobolev norms** H¹, H²
//!
//! ## References
//!
//! - Reed & Simon (1980) — *Methods of Modern Mathematical Physics*, Vol. I
//! - Kreyszig (1978) — *Introductory Functional Analysis with Applications*
//! - Atkinson & Han (2009) — *Theoretical Numerical Analysis*, Springer

// ═══════════════════════════════════════════════════════════════════════════
// NORMED / HILBERT SPACE  (finite-dimensional ℓ²)
// ═══════════════════════════════════════════════════════════════════════════

/// Finite-dimensional ℓ² space element (Hilbert space).
#[derive(Debug, Clone, PartialEq)]
pub struct L2Vec {
    pub components: Vec<f64>,
}

impl L2Vec {
    /// Create from a slice.
    pub fn new(components: Vec<f64>) -> Self {
        Self { components }
    }

    /// Dimension of the space.
    pub fn dim(&self) -> usize {
        self.components.len()
    }

    /// ‖x‖ = √(Σ xᵢ²)
    pub fn norm(&self) -> f64 {
        self.components.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    /// ⟨x, y⟩ = Σ xᵢ yᵢ
    pub fn inner_product(&self, other: &Self) -> f64 {
        assert_eq!(
            self.components.len(),
            other.components.len(),
            "dimension mismatch"
        );
        self.components
            .iter()
            .zip(other.components.iter())
            .map(|(&a, &b)| a * b)
            .sum()
    }

    /// Return unit vector x / ‖x‖. Panics if zero vector.
    pub fn normalize(&self) -> Self {
        let n = self.norm();
        assert!(n > 0.0, "cannot normalize zero vector");
        Self {
            components: self.components.iter().map(|&x| x / n).collect(),
        }
    }

    /// ‖x - y‖
    pub fn distance(&self, other: &Self) -> f64 {
        assert_eq!(self.components.len(), other.components.len());
        self.components
            .iter()
            .zip(other.components.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
    }

    /// Element-wise sum.
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.components.len(), other.components.len());
        Self {
            components: self
                .components
                .iter()
                .zip(other.components.iter())
                .map(|(&a, &b)| a + b)
                .collect(),
        }
    }

    /// Element-wise difference.
    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.components.len(), other.components.len());
        Self {
            components: self
                .components
                .iter()
                .zip(other.components.iter())
                .map(|(&a, &b)| a - b)
                .collect(),
        }
    }

    /// Scalar multiple.
    pub fn scale(&self, s: f64) -> Self {
        Self {
            components: self.components.iter().map(|&x| x * s).collect(),
        }
    }

    /// Project out all provided orthonormal basis directions (one Gram-Schmidt step).
    ///
    /// Returns the component of `self` orthogonal to span(basis).
    pub fn orthogonal_complement_component(&self, basis: &[L2Vec]) -> L2Vec {
        let mut result = self.clone();
        for b in basis {
            let coeff = result.inner_product(b);
            let proj = b.scale(coeff);
            result = result.sub(&proj);
        }
        result
    }
}

/// Gram-Schmidt orthonormalization of a set of vectors.
///
/// Returns an orthonormal basis for the span of `vecs`.
/// Linearly dependent vectors (those whose residual norm < 1e-12) are dropped.
pub fn gram_schmidt(vecs: &[L2Vec]) -> Vec<L2Vec> {
    let mut basis: Vec<L2Vec> = Vec::new();
    for v in vecs {
        let residual = v.orthogonal_complement_component(&basis);
        let n = residual.norm();
        if n > 1e-12 {
            basis.push(residual.scale(1.0 / n));
        }
    }
    basis
}

// ═══════════════════════════════════════════════════════════════════════════
// BOUNDED LINEAR OPERATORS
// ═══════════════════════════════════════════════════════════════════════════

/// A bounded linear operator T: ℝᵐ → ℝⁿ represented by an n×m matrix.
#[derive(Debug, Clone)]
pub struct BoundedOperator {
    /// n×m matrix; `matrix[i][j]` is row i, column j.
    pub matrix: Vec<Vec<f64>>,
}

impl BoundedOperator {
    /// Create from a row-major matrix. Each row must have the same length.
    pub fn new(matrix: Vec<Vec<f64>>) -> Self {
        if matrix.len() > 1 {
            let cols = matrix[0].len();
            for row in &matrix {
                assert_eq!(row.len(), cols, "all rows must have equal length");
            }
        }
        Self { matrix }
    }

    /// Number of rows (output dimension).
    pub fn rows(&self) -> usize {
        self.matrix.len()
    }

    /// Number of columns (input dimension).
    pub fn cols(&self) -> usize {
        if self.matrix.is_empty() {
            0
        } else {
            self.matrix[0].len()
        }
    }

    /// Apply operator: y = Tx.
    pub fn apply(&self, x: &L2Vec) -> L2Vec {
        assert_eq!(x.dim(), self.cols(), "dimension mismatch");
        let components = self
            .matrix
            .iter()
            .map(|row| row.iter().zip(x.components.iter()).map(|(&a, &b)| a * b).sum())
            .collect();
        L2Vec { components }
    }

    /// Compose operators: (self ∘ other)(x) = self(other(x)).
    pub fn compose(&self, other: &Self) -> Self {
        assert_eq!(self.cols(), other.rows(), "incompatible dimensions for composition");
        let n = self.rows();
        let m = other.cols();
        let k = self.cols();
        let mut result = vec![vec![0.0; m]; n];
        for i in 0..n {
            for j in 0..m {
                let mut s = 0.0;
                for l in 0..k {
                    s += self.matrix[i][l] * other.matrix[l][j];
                }
                result[i][j] = s;
            }
        }
        Self { matrix: result }
    }

    /// Adjoint (transpose for real operators).
    pub fn adjoint(&self) -> Self {
        let n = self.rows();
        let m = self.cols();
        let mut result = vec![vec![0.0; n]; m];
        for i in 0..n {
            for j in 0..m {
                result[j][i] = self.matrix[i][j];
            }
        }
        Self { matrix: result }
    }

    /// Estimate the operator norm (spectral radius for symmetric ops) via power iteration.
    pub fn operator_norm_estimate(&self, iterations: usize) -> f64 {
        assert_eq!(self.rows(), self.cols(), "power iteration requires square operator");
        let n = self.cols();
        if n == 0 {
            return 0.0;
        }
        // Start from all-ones vector
        let mut v = L2Vec {
            components: vec![1.0 / (n as f64).sqrt(); n],
        };
        let mut sigma = 0.0;
        for _ in 0..iterations {
            let w = self.apply(&v);
            sigma = w.norm();
            if sigma < 1e-15 {
                return 0.0;
            }
            v = w.scale(1.0 / sigma);
        }
        sigma
    }

    /// Is the operator self-adjoint (symmetric)?
    pub fn is_self_adjoint(&self) -> bool {
        if self.rows() != self.cols() {
            return false;
        }
        let n = self.rows();
        for i in 0..n {
            for j in 0..n {
                if (self.matrix[i][j] - self.matrix[j][i]).abs() > 1e-10 {
                    return false;
                }
            }
        }
        true
    }

    /// In finite dimensions every bounded operator is compact.
    pub fn is_compact(&self) -> bool {
        true
    }

    /// Check positive definiteness via Sylvester's criterion (all leading minors > 0).
    pub fn is_positive_definite(&self) -> bool {
        if !self.is_self_adjoint() {
            return false;
        }
        let n = self.rows();
        for k in 1..=n {
            if leading_minor_det(&self.matrix, k) <= 0.0 {
                return false;
            }
        }
        true
    }
}

/// Compute determinant of the k×k leading principal minor using Gaussian elimination.
fn leading_minor_det(matrix: &[Vec<f64>], k: usize) -> f64 {
    let mut m: Vec<Vec<f64>> = (0..k).map(|i| matrix[i][..k].to_vec()).collect();
    let mut det = 1.0;
    for col in 0..k {
        // find pivot
        let pivot_row = (col..k).max_by(|&a, &b| m[a][col].abs().partial_cmp(&m[b][col].abs()).unwrap());
        if let Some(pr) = pivot_row {
            if m[pr][col].abs() < 1e-15 {
                return 0.0;
            }
            if pr != col {
                m.swap(pr, col);
                det = -det;
            }
        }
        det *= m[col][col];
        let pivot = m[col][col];
        for row in (col + 1)..k {
            let factor = m[row][col] / pivot;
            for j in col..k {
                let val = m[col][j] * factor;
                m[row][j] -= val;
            }
        }
    }
    det
}

// ═══════════════════════════════════════════════════════════════════════════
// SPECTRAL THEORY
// ═══════════════════════════════════════════════════════════════════════════

/// Spectral decomposition of a symmetric operator.
pub struct SpectralDecomposition {
    pub eigenvalues: Vec<f64>,
    pub eigenvectors: Vec<L2Vec>, // orthonormal
}

/// Compute spectral decomposition of a symmetric operator via the Jacobi eigenvalue algorithm.
///
/// Returns `Err` if the operator is not symmetric.
pub fn spectral_decompose(op: &BoundedOperator) -> Result<SpectralDecomposition, String> {
    if !op.is_self_adjoint() {
        return Err("spectral_decompose requires a symmetric (self-adjoint) operator".into());
    }
    let n = op.rows();
    if n == 0 {
        return Ok(SpectralDecomposition {
            eigenvalues: vec![],
            eigenvectors: vec![],
        });
    }
    // Work with a mutable copy
    let mut a: Vec<Vec<f64>> = op.matrix.clone();
    // Accumulate eigenvectors in identity matrix
    let mut v: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect();

    let max_iter = 100 * n * n;
    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                if a[i][j].abs() > max_val {
                    max_val = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-12 {
            break;
        }
        // Compute Jacobi rotation angle
        let theta = if (a[q][q] - a[p][p]).abs() < 1e-15 {
            std::f64::consts::PI / 4.0
        } else {
            0.5 * ((2.0 * a[p][q]) / (a[q][q] - a[p][p])).atan()
        };
        let c = theta.cos();
        let s = theta.sin();
        // Apply rotation
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        for r in 0..n {
            if r != p && r != q {
                let arp = a[r][p];
                let arq = a[r][q];
                a[r][p] = c * arp - s * arq;
                a[p][r] = a[r][p];
                a[r][q] = s * arp + c * arq;
                a[q][r] = a[r][q];
            }
        }
        // Update eigenvector matrix
        for r in 0..n {
            let vrp = v[r][p];
            let vrq = v[r][q];
            v[r][p] = c * vrp - s * vrq;
            v[r][q] = s * vrp + c * vrq;
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    let eigenvectors: Vec<L2Vec> = (0..n)
        .map(|j| L2Vec {
            components: (0..n).map(|i| v[i][j]).collect(),
        })
        .collect();

    // Sort by eigenvalue ascending
    let mut pairs: Vec<(f64, L2Vec)> = eigenvalues.into_iter().zip(eigenvectors).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let (eigenvalues, eigenvectors) = pairs.into_iter().unzip();

    Ok(SpectralDecomposition {
        eigenvalues,
        eigenvectors,
    })
}

/// Compute f(A)x = Σᵢ f(λᵢ) ⟨x, eᵢ⟩ eᵢ via the spectral decomposition.
pub fn functional_calculus_apply(
    decomp: &SpectralDecomposition,
    f: &dyn Fn(f64) -> f64,
    x: &L2Vec,
) -> L2Vec {
    let n = x.dim();
    let mut result = L2Vec {
        components: vec![0.0; n],
    };
    for (lambda, ev) in decomp.eigenvalues.iter().zip(decomp.eigenvectors.iter()) {
        let coeff = f(*lambda) * ev.inner_product(x);
        let term = ev.scale(coeff);
        result = result.add(&term);
    }
    result
}

/// Compute exp(tA) for a symmetric operator via its spectral decomposition.
///
/// For general (non-symmetric) operators falls back to Taylor series (20 terms).
pub fn matrix_exp(op: &BoundedOperator, t: f64) -> BoundedOperator {
    let n = op.rows();
    assert_eq!(n, op.cols(), "matrix_exp requires square operator");
    if op.is_self_adjoint() {
        if let Ok(decomp) = spectral_decompose(op) {
            return spectral_to_operator(&decomp, |lambda| (t * lambda).exp(), n);
        }
    }
    // Taylor series fallback: exp(tA) = Σ (tA)ⁿ/n!
    let mut result = identity_matrix(n);
    let mut term = identity_matrix(n);
    let ta = scale_matrix(&op.matrix, t);
    let mut factorial = 1.0;
    for k in 1..=20usize {
        factorial *= k as f64;
        term = mat_mul(&term, &ta);
        let scaled = scale_matrix(&term, 1.0 / factorial);
        result = mat_add(&result, &scaled);
    }
    BoundedOperator { matrix: result }
}

/// Compute √A for a positive semi-definite symmetric operator via spectral decomp.
pub fn matrix_sqrt(op: &BoundedOperator) -> Result<BoundedOperator, String> {
    if !op.is_self_adjoint() {
        return Err("matrix_sqrt requires a symmetric operator".into());
    }
    let decomp = spectral_decompose(op)?;
    if decomp.eigenvalues.iter().any(|&lambda| lambda < -1e-10) {
        return Err("matrix_sqrt requires positive semi-definite operator".into());
    }
    let n = op.rows();
    Ok(spectral_to_operator(&decomp, |lambda| lambda.max(0.0).sqrt(), n))
}

// Helper: reconstruct operator from spectral decomp using f(λ).
fn spectral_to_operator(decomp: &SpectralDecomposition, f: impl Fn(f64) -> f64, n: usize) -> BoundedOperator {
    let mut matrix = vec![vec![0.0; n]; n];
    for (lambda, ev) in decomp.eigenvalues.iter().zip(decomp.eigenvectors.iter()) {
        let fl = f(*lambda);
        for i in 0..n {
            for j in 0..n {
                matrix[i][j] += fl * ev.components[i] * ev.components[j];
            }
        }
    }
    BoundedOperator { matrix }
}

// ─── matrix helpers ──────────────────────────────────────────────────────────

fn identity_matrix(n: usize) -> Vec<Vec<f64>> {
    (0..n).map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect()
}

fn scale_matrix(m: &[Vec<f64>], s: f64) -> Vec<Vec<f64>> {
    m.iter().map(|row| row.iter().map(|&x| x * s).collect()).collect()
}

fn mat_add(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter().zip(b.iter()).map(|(ra, rb)| ra.iter().zip(rb.iter()).map(|(&x, &y)| x + y).collect()).collect()
}

fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = b[0].len();
    let k = b.len();
    let mut c = vec![vec![0.0; m]; n];
    for i in 0..n {
        for l in 0..k {
            if a[i][l].abs() < 1e-300 { continue; }
            for j in 0..m {
                c[i][j] += a[i][l] * b[l][j];
            }
        }
    }
    c
}

// ═══════════════════════════════════════════════════════════════════════════
// FOURIER SERIES (L² theory)
// ═══════════════════════════════════════════════════════════════════════════

/// Fourier series of a function on [-π, π].
/// Coefficients: (aₙ, bₙ) for n = 0, 1, 2, …
pub struct FourierSeries {
    pub coefficients: Vec<(f64, f64)>,
}

impl FourierSeries {
    /// Compute Fourier coefficients of f on [-π, π] via n_quad-point trapezoidal rule.
    ///
    /// `n_terms` controls how many (a_n, b_n) pairs are computed (n = 0..n_terms-1).
    pub fn from_function<F: Fn(f64) -> f64>(f: F, n_terms: usize, n_quad: usize) -> Self {
        let pi = std::f64::consts::PI;
        let dx = 2.0 * pi / (n_quad as f64);
        // Pre-evaluate f
        let xs: Vec<f64> = (0..n_quad).map(|k| -pi + (k as f64) * dx).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| f(x)).collect();

        let mut coefficients = Vec::with_capacity(n_terms);
        for n in 0..n_terms {
            let n_f = n as f64;
            let a_n: f64 = ys.iter().zip(xs.iter()).map(|(&y, &x)| y * (n_f * x).cos()).sum::<f64>() * dx / pi;
            let b_n: f64 = ys.iter().zip(xs.iter()).map(|(&y, &x)| y * (n_f * x).sin()).sum::<f64>() * dx / pi;
            coefficients.push((a_n, b_n));
        }
        // Correct a_0 (should be (1/π)∫f dx, but b_0 is always 0)
        // The a_0 computed above is already (1/π)∫ f·cos(0)dx = (1/π)∫ f dx which is correct.
        Self { coefficients }
    }

    /// Evaluate the Fourier series at x.
    /// f(x) ≈ a₀/2 + Σₙ≥₁ (aₙ cos(nx) + bₙ sin(nx))
    pub fn evaluate(&self, x: f64) -> f64 {
        let mut sum = self.coefficients[0].0 / 2.0;
        for (n, &(a_n, b_n)) in self.coefficients.iter().enumerate().skip(1) {
            let nf = n as f64;
            sum += a_n * (nf * x).cos() + b_n * (nf * x).sin();
        }
        sum
    }

    /// ‖f‖² via Parseval's theorem: π(a₀²/2 + Σₙ≥₁ (aₙ² + bₙ²))
    pub fn l2_norm_squared(&self) -> f64 {
        let pi = std::f64::consts::PI;
        let a0 = self.coefficients[0].0;
        let mut s = a0 * a0 / 2.0;
        for &(a_n, b_n) in self.coefficients.iter().skip(1) {
            s += a_n * a_n + b_n * b_n;
        }
        pi * s
    }

    /// Rough truncation error bound: Σ_{k > n} (aₖ² + bₖ²)
    pub fn truncation_error_bound(&self, n: usize) -> f64 {
        self.coefficients
            .iter()
            .skip(n + 1)
            .map(|&(a, b)| a * a + b * b)
            .sum()
    }
}

/// Estimate the decay rate of |aₙ| + |bₙ| via log-linear regression on log(n) vs log(|coeff|).
pub fn fourier_convergence_rate(coefficients: &[(f64, f64)]) -> f64 {
    // log(|aₙ| + |bₙ|) ≈ -rate * log(n) + const
    let pairs: Vec<(f64, f64)> = coefficients
        .iter()
        .enumerate()
        .skip(1)
        .filter_map(|(n, &(a, b))| {
            let mag = a.abs() + b.abs();
            if mag > 1e-15 {
                Some(((n as f64).ln(), mag.ln()))
            } else {
                None
            }
        })
        .collect();
    if pairs.len() < 2 {
        return 0.0;
    }
    // Simple OLS slope
    let n = pairs.len() as f64;
    let mx: f64 = pairs.iter().map(|(x, _)| x).sum::<f64>() / n;
    let my: f64 = pairs.iter().map(|(_, y)| y).sum::<f64>() / n;
    let num: f64 = pairs.iter().map(|(x, y)| (x - mx) * (y - my)).sum();
    let den: f64 = pairs.iter().map(|(x, _)| (x - mx) * (x - mx)).sum();
    if den.abs() < 1e-15 {
        0.0
    } else {
        -num / den // negate so that positive = decay
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// FREDHOLM ALTERNATIVE
// ═══════════════════════════════════════════════════════════════════════════

/// Data for a Fredholm integral equation of the second kind: (I - λK)x = y.
pub struct FredholmData {
    /// Discretized kernel matrix K (n×n).
    pub kernel_matrix: Vec<Vec<f64>>,
    pub lambda: f64,
    /// Right-hand side y (length n).
    pub rhs: Vec<f64>,
}

/// Check whether y lies in the range of (I - λK), i.e. y ⊥ ker(I - λKᵀ).
///
/// Returns `true` if the equation is solvable (Fredholm alternative).
pub fn fredholm_check_solvability(data: &FredholmData) -> bool {
    let n = data.rhs.len();
    let kt = transpose(&data.kernel_matrix);
    // Build I - λKᵀ
    let mut a = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            a[i][j] = if i == j { 1.0 } else { 0.0 } - data.lambda * kt[i][j];
        }
    }
    // Find nullspace of A via Gaussian elimination; check rhs ⊥ nullspace
    // For finite-dimensional operators the nullspace can be found by row reduction.
    // We use a simple approach: solve Aᵀ z = 0 by finding small-singular-value directions.
    // Practical approach: check if |det(A)| > tol (full rank → always solvable for given y).
    let det = mat_det(&a);
    if det.abs() > 1e-10 {
        return true; // A full rank → unique solution for any y
    }
    // A is singular; y must be orthogonal to null(Aᵀ) = null(A)ᵀ.
    // Approximate null(Aᵀ) by finding a vector z with Az ≈ 0 via power iteration on AAᵀ.
    let aat = mat_mul(&a, &transpose(&a));
    let (smallest_ev_vec, smallest_ev) = smallest_eigenvector(&aat, 50);
    if smallest_ev.abs() > 1e-6 {
        return true; // not actually singular
    }
    // Check y ⊥ z
    let dot: f64 = data.rhs.iter().zip(smallest_ev_vec.iter()).map(|(&yi, &zi)| yi * zi).sum();
    dot.abs() < 1e-8
}

/// Solve (I - λK)x = y via Neumann series: x = Σₖ (λK)ᵏ y.
///
/// Converges when |λ| ‖K‖ < 1.
pub fn fredholm_solve(data: &FredholmData, tol: f64) -> Result<Vec<f64>, String> {
    let n = data.rhs.len();
    // Estimate ‖K‖ via Frobenius norm as upper bound on operator norm
    let k_norm: f64 = data.kernel_matrix.iter().flat_map(|r| r.iter()).map(|&x| x * x).sum::<f64>().sqrt();
    if data.lambda.abs() * k_norm >= 1.0 {
        return Err(format!(
            "Neumann series does not converge: |λ|‖K‖ = {:.4} ≥ 1",
            data.lambda.abs() * k_norm
        ));
    }
    let mut x = data.rhs.clone();
    let mut term = data.rhs.clone();
    for _iter in 0..200 {
        // term ← λK * term
        let new_term: Vec<f64> = (0..n)
            .map(|i| data.lambda * data.kernel_matrix[i].iter().zip(term.iter()).map(|(&k, &t)| k * t).sum::<f64>())
            .collect();
        let norm: f64 = new_term.iter().map(|&x| x * x).sum::<f64>().sqrt();
        for i in 0..n {
            x[i] += new_term[i];
        }
        term = new_term;
        if norm < tol {
            break;
        }
    }
    Ok(x)
}

// ─── matrix helpers for Fredholm ─────────────────────────────────────────────

fn transpose(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if m.is_empty() { return vec![]; }
    let rows = m.len();
    let cols = m[0].len();
    let mut t = vec![vec![0.0; rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            t[j][i] = m[i][j];
        }
    }
    t
}

fn mat_det(m: &[Vec<f64>]) -> f64 {
    let n = m.len();
    leading_minor_det(m, n)
}

/// Find the eigenvector corresponding to the smallest eigenvalue via inverse-free power iteration.
/// Returns (eigenvector, eigenvalue_estimate).
fn smallest_eigenvector(m: &[Vec<f64>], iterations: usize) -> (Vec<f64>, f64) {
    let n = m.len();
    // Shift: work with M - max_eig * I; smallest eigenvalue of M becomes most negative.
    // Simple: just do inverse iteration approximation via (M + I)^{-k} v or deflation.
    // Easier: find all eigenvalues for small n via characteristic polynomial or just use
    // the Jacobi approach on the matrix wrapped as BoundedOperator.
    let op = BoundedOperator { matrix: m.to_vec() };
    if let Ok(decomp) = spectral_decompose(&op) {
        let (min_idx, &min_ev) = decomp
            .eigenvalues
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .unwrap_or((0, &0.0));
        return (decomp.eigenvectors[min_idx].components.clone(), min_ev);
    }
    // Fallback: return uniform vector
    let norm = (n as f64).sqrt();
    (vec![1.0 / norm; n], 0.0)
}

// ═══════════════════════════════════════════════════════════════════════════
// SOBOLEV SPACE NORMS
// ═══════════════════════════════════════════════════════════════════════════

/// ‖f‖_{H¹([a,b])} = √(‖f‖_{L²}² + ‖f'‖_{L²}²) computed via n-point trapezoidal rule.
pub fn sobolev_norm_h1<F, DF>(f: F, df: DF, a: f64, b: f64, n: usize) -> f64
where
    F: Fn(f64) -> f64,
    DF: Fn(f64) -> f64,
{
    let h = (b - a) / (n as f64);
    let mut l2_f = 0.0;
    let mut l2_df = 0.0;
    for k in 0..=n {
        let x = a + k as f64 * h;
        let w = if k == 0 || k == n { 0.5 } else { 1.0 };
        let fv = f(x);
        let dfv = df(x);
        l2_f += w * fv * fv;
        l2_df += w * dfv * dfv;
    }
    (h * (l2_f + l2_df)).sqrt()
}

/// ‖f‖_{H²([a,b])} = √(‖f‖² + ‖f'‖² + ‖f''‖²) via n-point trapezoidal rule.
pub fn sobolev_norm_h2<F, DF, DDF>(f: F, df: DF, ddf: DDF, a: f64, b: f64, n: usize) -> f64
where
    F: Fn(f64) -> f64,
    DF: Fn(f64) -> f64,
    DDF: Fn(f64) -> f64,
{
    let h = (b - a) / (n as f64);
    let mut s = 0.0;
    for k in 0..=n {
        let x = a + k as f64 * h;
        let w = if k == 0 || k == n { 0.5 } else { 1.0 };
        let fv = f(x);
        let dfv = df(x);
        let ddfv = ddf(x);
        s += w * (fv * fv + dfv * dfv + ddfv * ddfv);
    }
    (h * s).sqrt()
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ─── Gram-Schmidt ────────────────────────────────────────────────────────

    #[test]
    fn test_gram_schmidt_orthogonality() {
        let vecs = vec![
            L2Vec::new(vec![1.0, 1.0, 0.0]),
            L2Vec::new(vec![1.0, 0.0, 1.0]),
            L2Vec::new(vec![0.0, 1.0, 1.0]),
        ];
        let basis = gram_schmidt(&vecs);
        assert_eq!(basis.len(), 3);
        for b in &basis {
            assert!(close(b.norm(), 1.0, 1e-12), "basis vector not unit");
        }
        for i in 0..basis.len() {
            for j in (i + 1)..basis.len() {
                assert!(
                    close(basis[i].inner_product(&basis[j]), 0.0, 1e-11),
                    "basis not orthogonal: {} {} ip={}",
                    i,
                    j,
                    basis[i].inner_product(&basis[j])
                );
            }
        }
    }

    #[test]
    fn test_gram_schmidt_two_same_vectors() {
        let vecs = vec![
            L2Vec::new(vec![1.0, 0.0]),
            L2Vec::new(vec![1.0, 0.0]),
        ];
        let basis = gram_schmidt(&vecs);
        assert_eq!(basis.len(), 1, "dependent vector should be dropped");
    }

    // ─── BoundedOperator ─────────────────────────────────────────────────────

    #[test]
    fn test_operator_norm_identity() {
        let id = BoundedOperator::new(vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ]);
        let norm = id.operator_norm_estimate(100);
        assert!(close(norm, 1.0, 1e-6), "identity norm = {}", norm);
    }

    #[test]
    fn test_adjoint_transpose() {
        let op = BoundedOperator::new(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
        let adj = op.adjoint();
        assert_eq!(adj.rows(), 2);
        assert_eq!(adj.cols(), 3);
        assert!(close(adj.matrix[0][1], 3.0, 1e-12));
        assert!(close(adj.matrix[1][2], 6.0, 1e-12));
    }

    #[test]
    fn test_is_self_adjoint() {
        let sym = BoundedOperator::new(vec![vec![2.0, 1.0], vec![1.0, 3.0]]);
        assert!(sym.is_self_adjoint());
        let asym = BoundedOperator::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert!(!asym.is_self_adjoint());
    }

    #[test]
    fn test_is_positive_definite() {
        // [[2,1],[1,2]]: eigenvalues 1 and 3, both positive
        let pd = BoundedOperator::new(vec![vec![2.0, 1.0], vec![1.0, 2.0]]);
        assert!(pd.is_positive_definite());
        // [[1,2],[2,1]]: eigenvalues -1 and 3 → not PD
        let npd = BoundedOperator::new(vec![vec![1.0, 2.0], vec![2.0, 1.0]]);
        assert!(!npd.is_positive_definite());
    }

    // ─── Spectral decomposition ───────────────────────────────────────────────

    #[test]
    fn test_spectral_decompose_diagonal() {
        let diag = BoundedOperator::new(vec![
            vec![3.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 2.0],
        ]);
        let decomp = spectral_decompose(&diag).unwrap();
        let mut evs = decomp.eigenvalues.clone();
        evs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(close(evs[0], 1.0, 1e-10));
        assert!(close(evs[1], 2.0, 1e-10));
        assert!(close(evs[2], 3.0, 1e-10));
    }

    #[test]
    fn test_matrix_exp_zero_is_identity() {
        let zero = BoundedOperator::new(vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
        let exp_zero = matrix_exp(&zero, 1.0);
        assert!(close(exp_zero.matrix[0][0], 1.0, 1e-10));
        assert!(close(exp_zero.matrix[0][1], 0.0, 1e-10));
        assert!(close(exp_zero.matrix[1][0], 0.0, 1e-10));
        assert!(close(exp_zero.matrix[1][1], 1.0, 1e-10));
    }

    #[test]
    fn test_matrix_sqrt_of_identity() {
        let id = BoundedOperator::new(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        let sq = matrix_sqrt(&id).unwrap();
        assert!(close(sq.matrix[0][0], 1.0, 1e-10));
        assert!(close(sq.matrix[1][1], 1.0, 1e-10));
    }

    #[test]
    fn test_spectral_symmetric_2x2() {
        // [[4,2],[2,1]] eigenvalues: (5 ± sqrt(5))/2... use a simpler example
        let op = BoundedOperator::new(vec![vec![3.0, 1.0], vec![1.0, 3.0]]);
        let decomp = spectral_decompose(&op).unwrap();
        // eigenvalues should be 2 and 4
        let mut evs = decomp.eigenvalues.clone();
        evs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(close(evs[0], 2.0, 1e-8));
        assert!(close(evs[1], 4.0, 1e-8));
    }

    // ─── Fourier series ───────────────────────────────────────────────────────

    #[test]
    fn test_fourier_series_constant() {
        // f(x) = c → a₀ = 2c, all other coefficients ≈ 0
        let c = 3.0_f64;
        let series = FourierSeries::from_function(|_| c, 5, 1000);
        // a₀ = (1/π)∫_{-π}^{π} c dx = 2c
        assert!(close(series.coefficients[0].0, 2.0 * c, 0.01), "a0 = {}", series.coefficients[0].0);
        for &(a, b) in series.coefficients.iter().skip(1) {
            assert!(close(a, 0.0, 0.02), "aₙ should be 0, got {}", a);
            assert!(close(b, 0.0, 0.02), "bₙ should be 0, got {}", b);
        }
    }

    #[test]
    fn test_fourier_evaluate_constant() {
        let series = FourierSeries::from_function(|_| 5.0, 10, 2000);
        for x in [-2.0, -1.0, 0.0, 1.0, 2.0] {
            assert!(close(series.evaluate(x), 5.0, 0.05), "evaluate at {}", x);
        }
    }

    #[test]
    fn test_parseval_theorem() {
        // f(x) = sin(x): exact L² norm on [-π,π] = π, so Parseval gives π
        let series = FourierSeries::from_function(|x: f64| x.sin(), 20, 10000);
        let parseval = series.l2_norm_squared();
        // ∫_{-π}^{π} sin²(x) dx = π
        assert!(close(parseval, std::f64::consts::PI, 0.05), "Parseval = {}", parseval);
    }

    // ─── Fredholm ─────────────────────────────────────────────────────────────

    #[test]
    fn test_fredholm_neumann_series_converges() {
        // Small λK should converge
        let data = FredholmData {
            kernel_matrix: vec![vec![0.1, 0.05], vec![0.05, 0.1]],
            lambda: 0.5,
            rhs: vec![1.0, 0.0],
        };
        let result = fredholm_solve(&data, 1e-10);
        assert!(result.is_ok(), "Neumann series should converge");
        let x = result.unwrap();
        // Verify (I - λK)x ≈ rhs
        let n = x.len();
        for i in 0..n {
            let lhs: f64 = x[i] - data.lambda * data.kernel_matrix[i].iter().zip(x.iter()).map(|(&k, &xi)| k * xi).sum::<f64>();
            assert!(close(lhs, data.rhs[i], 1e-8), "residual at {}: {}", i, lhs - data.rhs[i]);
        }
    }

    #[test]
    fn test_fredholm_does_not_converge_large_lambda() {
        let data = FredholmData {
            kernel_matrix: vec![vec![1.0, 1.0], vec![1.0, 1.0]],
            lambda: 2.0,
            rhs: vec![1.0, 1.0],
        };
        let result = fredholm_solve(&data, 1e-10);
        assert!(result.is_err());
    }

    // ─── Sobolev norms ────────────────────────────────────────────────────────

    #[test]
    fn test_sobolev_h1_constant_function() {
        // f = 1, f' = 0 on [0,1]: H¹ norm = sqrt(∫1 dx) = 1
        let norm = sobolev_norm_h1(|_| 1.0, |_| 0.0, 0.0, 1.0, 1000);
        assert!(close(norm, 1.0, 1e-6));
    }

    #[test]
    fn test_sobolev_h2_linear() {
        // f = x, f' = 1, f'' = 0 on [0,1]: H² = sqrt(∫x² dx + ∫1 dx) = sqrt(1/3 + 1) = sqrt(4/3)
        let norm = sobolev_norm_h2(|x| x, |_| 1.0, |_| 0.0, 0.0, 1.0, 1000);
        let expected = (4.0_f64 / 3.0).sqrt();
        assert!(close(norm, expected, 1e-4));
    }
}
