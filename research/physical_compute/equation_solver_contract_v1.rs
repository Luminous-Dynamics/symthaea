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

pub fn dot(a: &[f64], b: &[f64]) -> Result<f64, MatrixError> {
    if a.len() != b.len() {
        return Err(MatrixError::DimensionMismatch);
    }
    if a.iter().chain(b.iter()).any(|v| !v.is_finite()) {
        return Err(MatrixError::NonFinite);
    }
    let value = a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    if value.is_finite() {
        Ok(value)
    } else {
        Err(MatrixError::NonFinite)
    }
}

pub fn norm2(x: &[f64]) -> f64 {
    x.iter().fold(0.0_f64, |norm, value| norm.hypot(*value))
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
    let n = a.n;
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
    let mut y = vec![0.0; n];
    for row in 0..n {
        let mut sum = b[row];
        for col in 0..row {
            sum -= lower[row * n + col] * y[col];
        }
        y[row] = sum / lower[row * n + row];
    }

    // Back substitution: L^T x = y.
    let mut x = vec![0.0; n];
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

#[derive(Clone, Debug, PartialEq)]
pub struct IterativeSolution {
    pub x: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
    pub residual_norm: f64,
}

/// Unpreconditioned conjugate gradients for symmetric positive-definite A.
/// The stopping threshold is relative to ||b||_2 (with a defined zero-b case).
pub fn conjugate_gradient(
    a: &DenseMatrix,
    b: &[f64],
    relative_tolerance: f64,
    max_iterations: usize,
) -> Result<IterativeSolution, MatrixError> {
    let n = a.n;
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

    let mut x = vec![0.0; n];
    let mut r = b.to_vec();
    let mut p = r.clone();
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
            let true_residual: Vec<f64> = b.iter().zip(ax).map(|(rhs, lhs)| rhs - lhs).collect();
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
            p = r.clone();
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
pub fn error_metrics(
    a: &DenseMatrix,
    b: &[f64],
    x: &[f64],
    x_truth: &[f64],
) -> Result<ErrorMetrics, MatrixError> {
    if b.len() != a.n || x.len() != a.n || x_truth.len() != a.n {
        return Err(MatrixError::DimensionMismatch);
    }
    if b.iter().chain(x.iter()).chain(x_truth.iter()).any(|v| !v.is_finite()) {
        return Err(MatrixError::NonFinite);
    }
    let ax = a.mat_vec(x)?;
    let residual: Vec<f64> = ax.iter().zip(b).map(|(lhs, rhs)| lhs - rhs).collect();
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

    let difference: Vec<f64> = x.iter().zip(x_truth).map(|(v, truth)| v - truth).collect();
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
