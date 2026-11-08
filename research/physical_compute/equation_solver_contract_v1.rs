//! Dependency-free numerical oracle for condition-aware equation-solving workloads.
//!
//! This is a software reference, not an analog-device model. It creates exact
//! known-answer SPD systems and compares direct Cholesky with unpreconditioned
//! conjugate gradients. Residual, backward error and forward error are kept
//! separate because a small residual does not imply a small forward error for
//! an ill-conditioned system.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixError {
    Empty,
    DimensionMismatch,
    InvalidParameter,
    NonFinite,
    NotSymmetric,
    NotPositiveDefinite,
    Breakdown,
    InvalidTolerance,
    TooLarge,
    AllocationFailed,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DenseMatrix {
    n: usize,
    values: Vec<f64>,
}

impl DenseMatrix {
    /// Maximum dense reference size; larger systems need an explicitly sparse solver.
    pub const MAX_ELEMENTS: usize = 16 * 1024 * 1024;

    pub fn new(n: usize, values: Vec<f64>) -> Result<Self, MatrixError> {
        if n == 0 {
            return Err(MatrixError::Empty);
        }
        let expected = n.checked_mul(n).ok_or(MatrixError::InvalidParameter)?;
        if expected > Self::MAX_ELEMENTS {
            return Err(MatrixError::TooLarge);
        }
        if values.len() != expected {
            return Err(MatrixError::DimensionMismatch);
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(MatrixError::NonFinite);
        }
        Ok(Self { n, values })
    }

    pub fn dimension(&self) -> usize {
        self.n
    }

    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        if row >= self.n || col >= self.n {
            return None;
        }
        self.values.get(row * self.n + col).copied()
    }

    pub fn mat_vec(&self, x: &[f64]) -> Result<Vec<f64>, MatrixError> {
        if x.len() != self.n {
            return Err(MatrixError::DimensionMismatch);
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(MatrixError::NonFinite);
        }
        let mut out = vec![0.0; self.n];
        for (row, slot) in out.iter_mut().enumerate() {
            let mut sum = 0.0;
            for (col, value) in x.iter().enumerate() {
                sum += self.values[row * self.n + col] * value;
            }
            *slot = sum;
        }
        if out.iter().any(|v| !v.is_finite()) {
            return Err(MatrixError::NonFinite);
        }
        Ok(out)
    }

    pub fn frobenius_norm(&self) -> f64 {
        self.values.iter().fold(0.0_f64, |norm, value| norm.hypot(*value))
    }

    pub fn is_symmetric(&self, relative_tolerance: f64) -> bool {
        if !relative_tolerance.is_finite() || relative_tolerance < 0.0 {
            return false;
        }
        for row in 0..self.n {
            for col in 0..row {
                let a = self.values[row * self.n + col];
                let b = self.values[col * self.n + row];
                let scale = 1.0_f64.max(a.abs()).max(b.abs());
                if (a - b).abs() > relative_tolerance * scale {
                    return false;
                }
            }
        }
        true
    }
}

/// Compensated dot product. This reduces avoidable summation error in the
/// digital reference; it does not make the reference exact arithmetic.
pub fn dot(a: &[f64], b: &[f64]) -> Result<f64, MatrixError> {
    if a.len() != b.len() {
        return Err(MatrixError::DimensionMismatch);
    }
    if a.iter().chain(b.iter()).any(|v| !v.is_finite()) {
        return Err(MatrixError::NonFinite);
    }
    let mut sum = 0.0_f64;
    let mut correction = 0.0_f64;
    for (left, right) in a.iter().zip(b) {
        let product = left * right;
        if !product.is_finite() {
            return Err(MatrixError::NonFinite);
        }
        let next = sum + product;
        if !next.is_finite() {
            return Err(MatrixError::NonFinite);
        }
        if sum.abs() >= product.abs() {
            correction += (sum - next) + product;
        } else {
            correction += (product - next) + sum;
        }
        sum = next;
    }
    let value = sum + correction;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(MatrixError::NonFinite)
    }
}

pub fn norm2(x: &[f64]) -> f64 {
    x.iter().fold(0.0_f64, |norm, value| norm.hypot(*value))
}

fn try_zeroed_vector(length: usize) -> Result<Vec<f64>, MatrixError> {
    let mut values = Vec::new();
    values.try_reserve_exact(length).map_err(|_| MatrixError::AllocationFailed)?;
    values.resize(length, 0.0);
    Ok(values)
}

fn try_copy_vector(source: &[f64]) -> Result<Vec<f64>, MatrixError> {
    let mut values = Vec::new();
    values.try_reserve_exact(source.len()).map_err(|_| MatrixError::AllocationFailed)?;
    values.extend_from_slice(source);
    Ok(values)
}

/// A reproducible SPD family with a known spectrum:
///
///     A = lambda * I + alpha * 1 * 1^T
///
/// For lambda > 0 and alpha >= 0, its eigenvalues are lambda (multiplicity
/// n-1) and lambda + n*alpha. Thus kappa_2(A) is known exactly.
pub fn rank_one_spd_known_solution(
    lambda: f64,
    alpha: f64,
    x_truth: &[f64],
) -> Result<(DenseMatrix, Vec<f64>, f64), MatrixError> {
    let n = x_truth.len();
    if n == 0 {
        return Err(MatrixError::Empty);
    }
    if !lambda.is_finite() || !alpha.is_finite() || lambda <= 0.0 || alpha < 0.0 {
        return Err(MatrixError::InvalidParameter);
    }
    if x_truth.iter().any(|v| !v.is_finite()) {
        return Err(MatrixError::NonFinite);
    }

    let elements = n.checked_mul(n).ok_or(MatrixError::InvalidParameter)?;
    if elements > DenseMatrix::MAX_ELEMENTS {
        return Err(MatrixError::TooLarge);
    }
    let mut values = Vec::new();
    values.try_reserve_exact(elements).map_err(|_| MatrixError::AllocationFailed)?;
    values.resize(elements, alpha);
    for i in 0..n {
        values[i * n + i] += lambda;
    }
    let matrix = DenseMatrix::new(n, values)?;
    let rhs = matrix.mat_vec(x_truth)?;
    let condition_number = (lambda + (n as f64) * alpha) / lambda;
    Ok((matrix, rhs, condition_number))
}

pub fn cholesky_solve(a: &DenseMatrix, b: &[f64]) -> Result<Vec<f64>, MatrixError> {
    let n = a.dimension();
    if b.len() != n {
        return Err(MatrixError::DimensionMismatch);
    }
    if b.iter().any(|v| !v.is_finite()) {
        return Err(MatrixError::NonFinite);
    }
    if !a.is_symmetric(1.0e-12) {
        return Err(MatrixError::NotSymmetric);
    }

    let elements = n.checked_mul(n).ok_or(MatrixError::InvalidParameter)?;
    let mut lower = Vec::new();
    lower.try_reserve_exact(elements).map_err(|_| MatrixError::AllocationFailed)?;
    lower.resize(elements, 0.0);
    for row in 0..n {
        for col in 0..=row {
            let mut sum = a.values[row * n + col];
            for k in 0..col {
                sum -= lower[row * n + k] * lower[col * n + k];
            }
            if row == col {
                if !sum.is_finite() || sum <= 0.0 {
                    return Err(MatrixError::NotPositiveDefinite);
                }
                lower[row * n + col] = sum.sqrt();
            } else {
                let pivot = lower[col * n + col];
                if pivot <= 0.0 || !pivot.is_finite() {
                    return Err(MatrixError::NotPositiveDefinite);
                }
                lower[row * n + col] = sum / pivot;
            }
        }
    }

    // Forward substitution: L y = b.
    let mut y = try_zeroed_vector(n)?;
    for row in 0..n {
        let mut sum = b[row];
        for col in 0..row {
            sum -= lower[row * n + col] * y[col];
        }
        y[row] = sum / lower[row * n + row];
    }

    // Back substitution: L^T x = y.
    let mut x = try_zeroed_vector(n)?;
    for row in (0..n).rev() {
        let mut sum = y[row];
        for col in (row + 1)..n {
            sum -= lower[col * n + row] * x[col];
        }
        x[row] = sum / lower[row * n + row];
    }

    if x.iter().any(|v| !v.is_finite()) {
        return Err(MatrixError::NonFinite);
    }
    Ok(x)
}

/// A matrix-free linear operator interface for solver comparisons.
///
/// Implementations must return deterministic results for a given input.
/// Conjugate gradient additionally requires a symmetric positive-definite
/// operator; the generic type system cannot prove that property.
pub trait LinearOperator {
    fn dimension(&self) -> usize;
    fn mat_vec(&self, x: &[f64]) -> Result<Vec<f64>, MatrixError>;
    fn frobenius_norm(&self) -> f64;
    fn is_symmetric(&self, relative_tolerance: f64) -> bool;
    fn diagonal_value(&self, index: usize) -> Option<f64>;
}

impl LinearOperator for DenseMatrix {
    fn dimension(&self) -> usize {
        self.n
    }

    fn mat_vec(&self, x: &[f64]) -> Result<Vec<f64>, MatrixError> {
        DenseMatrix::mat_vec(self, x)
    }

    fn frobenius_norm(&self) -> f64 {
        DenseMatrix::frobenius_norm(self)
    }

    fn is_symmetric(&self, relative_tolerance: f64) -> bool {
        DenseMatrix::is_symmetric(self, relative_tolerance)
    }

    fn diagonal_value(&self, index: usize) -> Option<f64> {
        self.get(index, index)
    }
}

/// Matrix-free 1-D Poisson operator (the standard second-difference stencil):
/// diagonal = 2, adjacent off-diagonal = -1.
///
/// This is symmetric positive definite, tridiagonal, and costs O(n) storage
/// and O(n) per matrix-vector product instead of O(n^2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Poisson1d {
    n: usize,
}

impl Poisson1d {
    pub const MAX_DIMENSION: usize = 1_000_000;

    pub fn new(n: usize) -> Result<Self, MatrixError> {
        if n == 0 {
            return Err(MatrixError::Empty);
        }
        if n > Self::MAX_DIMENSION {
            return Err(MatrixError::TooLarge);
        }
        Ok(Self { n })
    }

    pub fn known_solution(
        &self,
        x_truth: &[f64],
    ) -> Result<(Vec<f64>, f64), MatrixError> {
        if x_truth.len() != self.n {
            return Err(MatrixError::DimensionMismatch);
        }
        if x_truth.iter().any(|value| !value.is_finite()) {
            return Err(MatrixError::NonFinite);
        }
        let rhs = self.mat_vec(x_truth)?;
        let angle = std::f64::consts::PI / (2.0 * (self.n as f64 + 1.0));
        let sine = angle.sin();
        let cosine = angle.cos();
        let condition_number = (cosine * cosine) / (sine * sine);
        if !condition_number.is_finite() {
            return Err(MatrixError::NonFinite);
        }
        Ok((rhs, condition_number))
    }
}

impl LinearOperator for Poisson1d {
    fn dimension(&self) -> usize {
        self.n
    }

    fn mat_vec(&self, x: &[f64]) -> Result<Vec<f64>, MatrixError> {
        if x.len() != self.n {
            return Err(MatrixError::DimensionMismatch);
        }
        if x.iter().any(|value| !value.is_finite()) {
            return Err(MatrixError::NonFinite);
        }
        let mut out = Vec::new();
        out.try_reserve_exact(self.n).map_err(|_| MatrixError::AllocationFailed)?;
        out.resize(self.n, 0.0);
        for i in 0..self.n {
            let mut value = 2.0 * x[i];
            if i > 0 {
                value -= x[i - 1];
            }
            if i + 1 < self.n {
                value -= x[i + 1];
            }
            if !value.is_finite() {
                return Err(MatrixError::NonFinite);
            }
            out[i] = value;
        }
        Ok(out)
    }

    fn frobenius_norm(&self) -> f64 {
        // There are n diagonal entries of magnitude 2 and 2(n-1)
        // off-diagonal entries of magnitude 1.
        ((4.0 * self.n as f64) + (2.0 * self.n.saturating_sub(1) as f64)).sqrt()
    }

    fn is_symmetric(&self, relative_tolerance: f64) -> bool {
        relative_tolerance.is_finite() && relative_tolerance >= 0.0
    }

    fn diagonal_value(&self, index: usize) -> Option<f64> {
        if index < self.n { Some(2.0) } else { None }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct IterativeSolution {
    pub x: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
    pub residual_norm: f64,
}

/// Unpreconditioned conjugate gradients for symmetric positive-definite A.
/// The stopping threshold is relative to ||b||_2 (with a defined zero-b case).
pub fn conjugate_gradient<A: LinearOperator>(
    a: &A,
    b: &[f64],
    relative_tolerance: f64,
    max_iterations: usize,
) -> Result<IterativeSolution, MatrixError> {
    let n = a.dimension();
    if b.len() != n {
        return Err(MatrixError::DimensionMismatch);
    }
    if !relative_tolerance.is_finite()
        || relative_tolerance <= 0.0
        || relative_tolerance >= 1.0
        || max_iterations == 0
    {
        return Err(MatrixError::InvalidTolerance);
    }
    if !a.is_symmetric(1.0e-12) {
        return Err(MatrixError::NotSymmetric);
    }

    let mut x = try_zeroed_vector(n)?;
    let mut r = try_copy_vector(b)?;
    let mut p = try_copy_vector(&r)?;
    let mut rs_old = dot(&r, &r)?;
    let b_norm = norm2(b);
    let threshold = relative_tolerance * b_norm;

    if b_norm == 0.0 {
        return Ok(IterativeSolution {
            x,
            iterations: 0,
            converged: true,
            residual_norm: 0.0,
        });
    }

    let initial_norm = rs_old.sqrt();
    if initial_norm <= threshold {
        return Ok(IterativeSolution {
            x,
            iterations: 0,
            converged: true,
            residual_norm: initial_norm,
        });
    }

    for iteration in 1..=max_iterations {
        let ap = a.mat_vec(&p)?;
        let denominator = dot(&p, &ap)?;
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(MatrixError::Breakdown);
        }
        let step = rs_old / denominator;
        if !step.is_finite() {
            return Err(MatrixError::NonFinite);
        }

        for i in 0..n {
            x[i] += step * p[i];
            r[i] -= step * ap[i];
        }
        let rs_new = dot(&r, &r)?;
        let residual_norm = rs_new.sqrt();

        if residual_norm <= threshold {
            // Recompute b - A*x from the matrix and candidate, rather than
            // trusting the recursively updated CG residual at acceptance.
            let ax = a.mat_vec(&x)?;
            let mut true_residual = try_zeroed_vector(n)?;
            for (out, (rhs, lhs)) in true_residual.iter_mut().zip(b.iter().zip(ax)) {
                *out = rhs - lhs;
            }
            let true_residual_norm = norm2(&true_residual);
            if true_residual_norm <= threshold {
                return Ok(IterativeSolution {
                    x,
                    iterations: iteration,
                    converged: true,
                    residual_norm: true_residual_norm,
                });
            }

            // Residual replacement: restart the Krylov direction from the
            // independently recomputed residual; do not accept recurrence drift.
            r = true_residual;
            p = try_copy_vector(&r)?;
            rs_old = dot(&r, &r)?;
            continue;
        }

        if rs_old <= 0.0 || !rs_new.is_finite() {
            return Err(MatrixError::Breakdown);
        }
        let beta = rs_new / rs_old;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rs_old = rs_new;
    }

    Ok(IterativeSolution {
        x,
        iterations: max_iterations,
        converged: false,
        residual_norm: norm2(&r),
    })
}

/// Jacobi (diagonal) preconditioner. This only changes solver convergence;
/// its setup and application costs must be included in architecture benchmarks.
#[derive(Clone, Debug, PartialEq)]
pub struct JacobiPreconditioner {
    inverse_diagonal: Vec<f64>,
}

impl JacobiPreconditioner {
    pub fn from_operator<A: LinearOperator>(a: &A) -> Result<Self, MatrixError> {
        let n = a.dimension();
        let mut inverse_diagonal = Vec::new();
        inverse_diagonal.try_reserve_exact(n).map_err(|_| MatrixError::AllocationFailed)?;
        for index in 0..n {
            let diagonal = a.diagonal_value(index).ok_or(MatrixError::DimensionMismatch)?;
            if !diagonal.is_finite() || diagonal <= 0.0 {
                return Err(MatrixError::NotPositiveDefinite);
            }
            inverse_diagonal.push(1.0 / diagonal);
        }
        Ok(Self { inverse_diagonal })
    }

    pub fn apply(&self, residual: &[f64]) -> Result<Vec<f64>, MatrixError> {
        if residual.len() != self.inverse_diagonal.len() {
            return Err(MatrixError::DimensionMismatch);
        }
        if residual.iter().any(|value| !value.is_finite()) {
            return Err(MatrixError::NonFinite);
        }
        let mut result = try_zeroed_vector(residual.len())?;
        for (out, (r, inverse)) in result.iter_mut().zip(residual.iter().zip(&self.inverse_diagonal)) {
            *out = r * inverse;
        }
        if result.iter().any(|value| !value.is_finite()) {
            return Err(MatrixError::NonFinite);
        }
        Ok(result)
    }
}

/// Preconditioned conjugate gradients (PCG) with a fixed Jacobi preconditioner.
/// Acceptance uses a recomputed true residual, just like the unpreconditioned
/// reference. The preconditioner must be fixed during one solve.
pub fn preconditioned_conjugate_gradient<A: LinearOperator>(
    a: &A,
    preconditioner: &JacobiPreconditioner,
    b: &[f64],
    relative_tolerance: f64,
    max_iterations: usize,
) -> Result<IterativeSolution, MatrixError> {
    let n = a.dimension();
    if b.len() != n {
        return Err(MatrixError::DimensionMismatch);
    }
    if !relative_tolerance.is_finite()
        || relative_tolerance <= 0.0
        || relative_tolerance >= 1.0
        || max_iterations == 0
    {
        return Err(MatrixError::InvalidTolerance);
    }
    if !a.is_symmetric(1.0e-12) {
        return Err(MatrixError::NotSymmetric);
    }

    let mut x = try_zeroed_vector(n)?;
    let mut r = try_copy_vector(b)?;
    let mut z = preconditioner.apply(&r)?;
    let mut p = try_copy_vector(&z)?;
    let mut rho = dot(&r, &z)?;
    let b_norm = norm2(b);
    let threshold = relative_tolerance * b_norm;

    if b_norm == 0.0 {
        return Ok(IterativeSolution {
            x,
            iterations: 0,
            converged: true,
            residual_norm: 0.0,
        });
    }
    if !rho.is_finite() || rho <= 0.0 {
        return Err(MatrixError::Breakdown);
    }

    for iteration in 1..=max_iterations {
        let ap = a.mat_vec(&p)?;
        let denominator = dot(&p, &ap)?;
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(MatrixError::Breakdown);
        }
        let alpha = rho / denominator;
        if !alpha.is_finite() {
            return Err(MatrixError::NonFinite);
        }
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }

        if norm2(&r) <= threshold {
            let ax = a.mat_vec(&x)?;
            let mut true_residual = try_zeroed_vector(n)?;
            for (out, (rhs, lhs)) in true_residual.iter_mut().zip(b.iter().zip(ax)) {
                *out = rhs - lhs;
            }
            let true_norm = norm2(&true_residual);
            if true_norm <= threshold {
                return Ok(IterativeSolution {
                    x,
                    iterations: iteration,
                    converged: true,
                    residual_norm: true_norm,
                });
            }
            r = true_residual;
            z = preconditioner.apply(&r)?;
            p = try_copy_vector(&z)?;
            rho = dot(&r, &z)?;
            if !rho.is_finite() || rho <= 0.0 {
                return Err(MatrixError::Breakdown);
            }
            continue;
        }

        z = preconditioner.apply(&r)?;
        let next_rho = dot(&r, &z)?;
        if !next_rho.is_finite() || next_rho <= 0.0 {
            return Err(MatrixError::Breakdown);
        }
        let beta = next_rho / rho;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rho = next_rho;
    }

    Ok(IterativeSolution {
        x,
        iterations: max_iterations,
        converged: false,
        residual_norm: norm2(&r),
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ErrorMetrics {
    pub residual_norm: f64,
    pub relative_residual: f64,
    pub relative_backward_error: f64,
    pub absolute_forward_error: f64,
    pub relative_forward_error: f64,
}

/// Evaluate a candidate against b and an evaluator-only reference solution.
/// The reference must not be exposed to the candidate or its online verifier.
pub fn error_metrics<A: LinearOperator>(
    a: &A,
    b: &[f64],
    x: &[f64],
    x_truth: &[f64],
) -> Result<ErrorMetrics, MatrixError> {
    if b.len() != a.dimension() || x.len() != a.dimension() || x_truth.len() != a.dimension() {
        return Err(MatrixError::DimensionMismatch);
    }
    if b.iter().chain(x.iter()).chain(x_truth.iter()).any(|v| !v.is_finite()) {
        return Err(MatrixError::NonFinite);
    }
    let ax = a.mat_vec(x)?;
    let mut residual = try_zeroed_vector(b.len())?;
    for (out, (lhs, rhs)) in residual.iter_mut().zip(ax.iter().zip(b)) {
        *out = lhs - rhs;
    }
    let residual_norm = norm2(&residual);
    let b_norm = norm2(b);
    let relative_residual = if b_norm == 0.0 {
        if residual_norm == 0.0 { 0.0 } else { f64::INFINITY }
    } else {
        residual_norm / b_norm
    };

    let denominator = a.frobenius_norm() * norm2(x) + b_norm;
    let relative_backward_error = if denominator == 0.0 {
        if residual_norm == 0.0 { 0.0 } else { f64::INFINITY }
    } else {
        residual_norm / denominator
    };

    let mut difference = try_zeroed_vector(x.len())?;
    for (out, (value, truth)) in difference.iter_mut().zip(x.iter().zip(x_truth)) {
        *out = value - truth;
    }
    let absolute_forward_error = norm2(&difference);
    let truth_norm = norm2(x_truth);
    let relative_forward_error = if truth_norm == 0.0 {
        absolute_forward_error
    } else {
        absolute_forward_error / truth_norm
    };

    Ok(ErrorMetrics {
        residual_norm,
        relative_residual,
        relative_backward_error,
        absolute_forward_error,
        relative_forward_error,
    })
}

/// Stable diagonal linear ODE: dx_i/dt = -rate_i * x_i.
///
/// This is a small analytic trajectory oracle for the differential-equation
/// benchmark lane. It is not a model of a particular analog circuit.
#[derive(Clone, Debug, PartialEq)]
pub struct DiagonalDecayOde {
    rates: Vec<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrajectoryMetrics {
    pub final_absolute_error: f64,
    pub final_relative_error: f64,
    pub maximum_absolute_error: f64,
    pub maximum_relative_error: f64,
}

impl DiagonalDecayOde {
    pub const MAX_DIMENSION: usize = 1_000_000;
    pub const MAX_TRAJECTORY_SCALARS: usize = 16 * 1024 * 1024;

    pub fn new(rates: Vec<f64>) -> Result<Self, MatrixError> {
        if rates.is_empty() {
            return Err(MatrixError::Empty);
        }
        if rates.len() > Self::MAX_DIMENSION {
            return Err(MatrixError::TooLarge);
        }
        if rates.iter().any(|rate| !rate.is_finite() || *rate < 0.0) {
            return Err(MatrixError::InvalidParameter);
        }
        Ok(Self { rates })
    }

    pub fn dimension(&self) -> usize {
        self.rates.len()
    }

    pub fn analytic_state(
        &self,
        initial: &[f64],
        time: f64,
    ) -> Result<Vec<f64>, MatrixError> {
        if initial.len() != self.dimension() {
            return Err(MatrixError::DimensionMismatch);
        }
        if !time.is_finite() || time < 0.0 {
            return Err(MatrixError::InvalidParameter);
        }
        if initial.iter().any(|value| !value.is_finite()) {
            return Err(MatrixError::NonFinite);
        }
        let mut result = try_zeroed_vector(initial.len())?;
        for (out, (x0, rate)) in result.iter_mut().zip(initial.iter().zip(&self.rates)) {
            *out = x0 * (-rate * time).exp();
        }
        if result.iter().any(|value| !value.is_finite()) {
            return Err(MatrixError::NonFinite);
        }
        Ok(result)
    }

    /// Classical explicit RK4 trajectory. The returned vector includes the
    /// initial state followed by one state per integration step.
    pub fn rk4_trajectory(
        &self,
        initial: &[f64],
        step_size: f64,
        steps: usize,
    ) -> Result<Vec<Vec<f64>>, MatrixError> {
        if initial.len() != self.dimension() {
            return Err(MatrixError::DimensionMismatch);
        }
        if !step_size.is_finite() || step_size <= 0.0 || steps == 0 {
            return Err(MatrixError::InvalidTolerance);
        }
        if initial.iter().any(|value| !value.is_finite()) {
            return Err(MatrixError::NonFinite);
        }
        let points = steps.checked_add(1).ok_or(MatrixError::TooLarge)?;
        let total_scalars = points.checked_mul(self.dimension()).ok_or(MatrixError::TooLarge)?;
        if total_scalars > Self::MAX_TRAJECTORY_SCALARS {
            return Err(MatrixError::TooLarge);
        }
        if !(step_size * steps as f64).is_finite() {
            return Err(MatrixError::NonFinite);
        }

        let mut trajectory = Vec::new();
        trajectory.try_reserve_exact(points).map_err(|_| MatrixError::AllocationFailed)?;
        let mut state = Vec::new();
        state.try_reserve_exact(self.dimension()).map_err(|_| MatrixError::AllocationFailed)?;
        state.extend_from_slice(initial);
        let mut initial_saved = Vec::new();
        initial_saved.try_reserve_exact(state.len()).map_err(|_| MatrixError::AllocationFailed)?;
        initial_saved.extend_from_slice(&state);
        trajectory.push(initial_saved);

        for _ in 0..steps {
            let mut next = Vec::new();
            next.try_reserve_exact(self.dimension()).map_err(|_| MatrixError::AllocationFailed)?;
            for (value, rate) in state.iter().zip(&self.rates) {
                let k1 = -rate * value;
                let k2 = -rate * (value + 0.5 * step_size * k1);
                let k3 = -rate * (value + 0.5 * step_size * k2);
                let k4 = -rate * (value + step_size * k3);
                let updated = value + (step_size / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
                if !updated.is_finite() {
                    return Err(MatrixError::NonFinite);
                }
                next.push(updated);
            }
            state = next;
            let mut saved = Vec::new();
            saved.try_reserve_exact(state.len()).map_err(|_| MatrixError::AllocationFailed)?;
            saved.extend_from_slice(&state);
            trajectory.push(saved);
        }
        Ok(trajectory)
    }

    /// Evaluator-side comparison against the analytic trajectory. Keep these
    /// metrics outside the candidate/online-verifier API.
    pub fn trajectory_error(
        &self,
        initial: &[f64],
        step_size: f64,
        trajectory: &[Vec<f64>],
    ) -> Result<TrajectoryMetrics, MatrixError> {
        if initial.len() != self.dimension() || trajectory.is_empty() {
            return Err(MatrixError::DimensionMismatch);
        }
        if !step_size.is_finite() || step_size <= 0.0 {
            return Err(MatrixError::InvalidTolerance);
        }
        let expected_len = trajectory.len();
        let mut maximum_absolute_error = 0.0_f64;
        let mut maximum_relative_error = 0.0_f64;
        let mut final_absolute_error = 0.0_f64;
        let mut final_relative_error = 0.0_f64;

        for (index, candidate) in trajectory.iter().enumerate() {
            if candidate.len() != self.dimension() {
                return Err(MatrixError::DimensionMismatch);
            }
            if candidate.iter().any(|value| !value.is_finite()) {
                return Err(MatrixError::NonFinite);
            }
            let reference = self.analytic_state(initial, step_size * index as f64)?;
            let mut absolute = 0.0_f64;
            for (actual, expected) in candidate.iter().zip(&reference) {
                absolute = absolute.hypot(actual - expected);
            }
            let reference_norm = norm2(&reference);
            let relative = if reference_norm == 0.0 { absolute } else { absolute / reference_norm };
            maximum_absolute_error = maximum_absolute_error.max(absolute);
            maximum_relative_error = maximum_relative_error.max(relative);
            if index + 1 == expected_len {
                final_absolute_error = absolute;
                final_relative_error = relative;
            }
        }

        Ok(TrajectoryMetrics {
            final_absolute_error,
            final_relative_error,
            maximum_absolute_error,
            maximum_relative_error,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_one_spd_generator_has_known_condition_and_rhs() {
        let truth = [1.0, -2.0, 0.5, 3.0];
        let (a, b, kappa) = rank_one_spd_known_solution(2.0, 0.25, &truth).unwrap();
        assert_eq!(a.dimension(), 4);
        assert!((kappa - 1.5).abs() < 1.0e-12);
        let recovered_rhs = a.mat_vec(&truth).unwrap();
        for (actual, expected) in b.iter().zip(recovered_rhs) {
            assert!((actual - expected).abs() < 1.0e-12);
        }
    }

    #[test]
    fn cholesky_recovers_known_spd_solution() {
        let truth = [0.5, -1.25, 2.0, 0.75, -0.25];
        let (a, b, _) = rank_one_spd_known_solution(1.5, 0.1, &truth).unwrap();
        let x = cholesky_solve(&a, &b).unwrap();
        let metrics = error_metrics(&a, &b, &x, &truth).unwrap();
        assert!(metrics.relative_residual < 1.0e-12);
        assert!(metrics.relative_forward_error < 1.0e-12);
    }

    #[test]
    fn conjugate_gradient_matches_direct_reference() {
        let truth = [1.0, 0.5, -1.0, 2.0, -0.5, 0.25];
        let (a, b, _) = rank_one_spd_known_solution(2.0, 0.2, &truth).unwrap();
        let iterative = conjugate_gradient(&a, &b, 1.0e-12, 64).unwrap();
        let direct = cholesky_solve(&a, &b).unwrap();
        assert!(iterative.converged);
        assert!(iterative.iterations <= 6);
        for (actual, expected) in iterative.x.iter().zip(direct) {
            assert!((actual - expected).abs() < 1.0e-10);
        }
    }

    #[test]
    fn ill_conditioned_negative_control_separates_residual_from_forward_error() {
        let a = DenseMatrix::new(2, vec![1.0, 0.0, 0.0, 1.0e-10]).unwrap();
        let truth = [1.0, 1.0];
        let b = [1.0, 1.0e-10];
        let bad_candidate = [1.0, 0.0];
        let metrics = error_metrics(&a, &b, &bad_candidate, &truth).unwrap();
        assert!(metrics.relative_residual < 1.0e-9);
        assert!(metrics.relative_forward_error > 0.7);
    }

    #[test]
    fn conjugate_gradient_breaks_down_on_indefinite_search_direction() {
        let a = DenseMatrix::new(2, vec![1.0, 0.0, 0.0, -1.0]).unwrap();
        let result = conjugate_gradient(&a, &[0.0, 1.0], 1.0e-10, 8);
        assert_eq!(result, Err(MatrixError::Breakdown));
    }

    #[test]
    fn cholesky_rejects_singular_and_nonsymmetric_inputs() {
        let singular = DenseMatrix::new(2, vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(cholesky_solve(&singular, &[1.0, 0.0]), Err(MatrixError::NotPositiveDefinite));

        let nonsymmetric = DenseMatrix::new(2, vec![1.0, 2.0, 0.0, 1.0]).unwrap();
        assert_eq!(cholesky_solve(&nonsymmetric, &[1.0, 1.0]), Err(MatrixError::NotSymmetric));
    }

    #[test]
    fn jacobi_pcg_solves_diagonally_scaled_spd_system() {
        let scales = [0.25, 0.5, 2.0, 8.0];
        let beta = 0.1;
        let n = scales.len();
        let mut values = vec![0.0; n * n];
        for row in 0..n {
            for col in 0..n {
                let core = if row == col { 1.0 + beta } else { beta };
                values[row * n + col] = scales[row] * core * scales[col];
            }
        }
        let matrix = DenseMatrix::new(n, values).unwrap();
        let truth = [1.0, -0.5, 2.0, 0.25];
        let rhs = matrix.mat_vec(&truth).unwrap();
        let preconditioner = JacobiPreconditioner::from_operator(&matrix).unwrap();
        let result = preconditioned_conjugate_gradient(
            &matrix,
            &preconditioner,
            &rhs,
            1.0e-12,
            16,
        ).unwrap();
        assert!(result.converged);
        let metrics = error_metrics(&matrix, &rhs, &result.x, &truth).unwrap();
        assert!(metrics.relative_residual < 1.0e-12);
        assert!(metrics.relative_forward_error < 1.0e-10);
    }

    #[test]
    fn jacobi_preconditioner_rejects_nonpositive_diagonal() {
        let matrix = DenseMatrix::new(2, vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(JacobiPreconditioner::from_operator(&matrix), Err(MatrixError::NotPositiveDefinite));
    }

    #[test]
    fn diagonal_decay_ode_rk4_converges_with_fourth_order_step_refinement() {
        let ode = DiagonalDecayOde::new(vec![1.0]).unwrap();
        let initial = [1.0];
        let coarse = ode.rk4_trajectory(&initial, 0.1, 10).unwrap();
        let fine = ode.rk4_trajectory(&initial, 0.05, 20).unwrap();
        let coarse_error = ode.trajectory_error(&initial, 0.1, &coarse).unwrap();
        let fine_error = ode.trajectory_error(&initial, 0.05, &fine).unwrap();
        assert!(coarse_error.final_absolute_error > fine_error.final_absolute_error * 10.0);
        assert!(fine_error.final_relative_error < 1.0e-7);
    }

    #[test]
    fn diagonal_decay_ode_handles_multiple_timescales() {
        let ode = DiagonalDecayOde::new(vec![0.5, 2.0, 10.0]).unwrap();
        let initial = [1.0, -0.5, 0.25];
        let trajectory = ode.rk4_trajectory(&initial, 0.005, 200).unwrap();
        let metrics = ode.trajectory_error(&initial, 0.005, &trajectory).unwrap();
        assert!(metrics.maximum_relative_error < 1.0e-7);
        assert!(metrics.final_absolute_error < 1.0e-8);
    }

    #[test]
    fn diagonal_decay_ode_rejects_bad_inputs_and_oversized_trajectories() {
        assert_eq!(DiagonalDecayOde::new(vec![]), Err(MatrixError::Empty));
        assert_eq!(DiagonalDecayOde::new(vec![f64::NAN]), Err(MatrixError::InvalidParameter));
        let ode = DiagonalDecayOde::new(vec![1.0, 2.0]).unwrap();
        assert_eq!(ode.analytic_state(&[1.0], 1.0), Err(MatrixError::DimensionMismatch));
        assert_eq!(ode.rk4_trajectory(&[1.0, 2.0], 0.0, 10), Err(MatrixError::InvalidTolerance));
        assert_eq!(
            ode.rk4_trajectory(&[1.0, 2.0], 1.0e-3, DiagonalDecayOde::MAX_TRAJECTORY_SCALARS),
            Err(MatrixError::TooLarge)
        );
    }

    #[test]
    fn matrix_free_poisson_operator_matches_known_spectrum_and_solution() {
        let n = 32;
        let operator = Poisson1d::new(n).unwrap();
        let truth: Vec<f64> = (0..n).map(|i| ((i + 1) as f64 * 0.17).sin()).collect();
        let (rhs, condition) = operator.known_solution(&truth).unwrap();
        let theta = std::f64::consts::PI / (2.0 * (n as f64 + 1.0));
        let expected = theta.cos().powi(2) / theta.sin().powi(2);
        assert!((condition - expected).abs() / expected < 1.0e-12);
        let reconstructed = operator.mat_vec(&truth).unwrap();
        assert_eq!(rhs, reconstructed);
        assert!(operator.frobenius_norm().is_finite());
    }

    #[test]
    fn conjugate_gradient_solves_matrix_free_poisson_system() {
        let n = 64;
        let operator = Poisson1d::new(n).unwrap();
        let truth: Vec<f64> = (0..n).map(|i| (0.11 * (i as f64 + 1.0)).sin()).collect();
        let (rhs, _) = operator.known_solution(&truth).unwrap();
        let solution = conjugate_gradient(&operator, &rhs, 1.0e-11, 20 * n).unwrap();
        assert!(solution.converged);
        assert!(solution.iterations <= n);
        let metrics = error_metrics(&operator, &rhs, &solution.x, &truth).unwrap();
        assert!(metrics.relative_residual < 1.0e-10);
        assert!(metrics.relative_forward_error < 1.0e-8);
    }

    #[test]
    fn poisson_operator_rejects_invalid_dimensions_and_non_finite_input() {
        assert_eq!(Poisson1d::new(0), Err(MatrixError::Empty));
        assert_eq!(Poisson1d::new(Poisson1d::MAX_DIMENSION + 1), Err(MatrixError::TooLarge));
        let operator = Poisson1d::new(4).unwrap();
        assert_eq!(operator.mat_vec(&[1.0, f64::NAN, 2.0, 3.0]), Err(MatrixError::NonFinite));
        assert_eq!(operator.known_solution(&[1.0]), Err(MatrixError::DimensionMismatch));
    }

    #[test]
    fn compensated_dot_handles_cancellation_better_than_naive_order() {
        let a = [1.0e16, 1.0, -1.0e16];
        let b = [1.0, 1.0, 1.0];
        assert_eq!(dot(&a, &b).unwrap(), 1.0);
    }

    #[test]
    fn dense_allocation_limit_is_checked_before_solver_work() {
        assert_eq!(DenseMatrix::new(4097, vec![]), Err(MatrixError::TooLarge));
        assert_eq!(rank_one_spd_known_solution(1.0, 0.0, &vec![1.0; 4097]), Err(MatrixError::TooLarge));
    }

    #[test]
    fn non_finite_evaluator_inputs_are_rejected() {
        let a = DenseMatrix::new(1, vec![1.0]).unwrap();
        assert_eq!(error_metrics(&a, &[f64::NAN], &[1.0], &[1.0]), Err(MatrixError::NonFinite));
        assert_eq!(error_metrics(&a, &[1.0], &[1.0], &[f64::INFINITY]), Err(MatrixError::NonFinite));
    }

    #[test]
    fn invalid_dimensions_and_tolerances_fail_closed() {
        assert_eq!(DenseMatrix::new(0, vec![]), Err(MatrixError::Empty));
        assert_eq!(DenseMatrix::new(2, vec![1.0]), Err(MatrixError::DimensionMismatch));
        let a = DenseMatrix::new(1, vec![2.0]).unwrap();
        assert_eq!(conjugate_gradient(&a, &[1.0], 0.0, 10), Err(MatrixError::InvalidTolerance));
        assert_eq!(a.mat_vec(&[1.0, 2.0]), Err(MatrixError::DimensionMismatch));
    }

    #[test]
    fn zero_rhs_is_a_converged_zero_solution() {
        let a = DenseMatrix::new(2, vec![2.0, 0.0, 0.0, 3.0]).unwrap();
        let result = conjugate_gradient(&a, &[0.0, 0.0], 1.0e-8, 10).unwrap();
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
        assert_eq!(result.x, vec![0.0, 0.0]);
    }
}
