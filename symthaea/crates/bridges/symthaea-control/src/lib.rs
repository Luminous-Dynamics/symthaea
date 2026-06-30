// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Trajectory optimisation, state estimation, and control primitives for Symthaea.
//!
//! # Provided Primitives
//! - [`LqrSolver`] — Discrete-time LQR via iterative DARE (Discrete Algebraic Riccati Equation)
//! - [`KalmanFilter`] — Discrete-time linear Kalman filter (predict / update)
//! - [`LinearMpc`] — Unconstrained linear MPC via batch prediction matrices
//! - [`LqgController`] — LQG = LQR + Kalman (output-feedback optimal control)
//! - [`HInfBound`] — H-∞ robustness metric (structured singular value upper bound)
//! - [`LyapunovChecker`] — Verifies Lyapunov stability for a closed-loop system
//! - [`ConsciousnessGainScheduler`] — Scales control gains by Φ (phi) level
//!
//! All structures are `#[deny(unsafe_code)]` and use `nalgebra` for linear algebra.
#![deny(unsafe_code)]

use nalgebra::{DMatrix, DVector};
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum ControlError {
    #[error("Linear algebra operation failed: {0}")]
    LinAlg(String),
    #[error("Matrix inversion failed (potentially singular)")]
    SingularMatrix,
    #[error("Invalid system dimension: {0}")]
    InvalidDimension(String),
    #[error("Convergence failed in iterative solver after {0} iterations")]
    NonConvergence(usize),
    #[error("Lyapunov stability not satisfied: P must be positive-definite")]
    NotStable,
    #[error("Parameter out of range: {0}")]
    InvalidParameter(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// LQR Solver
// ─────────────────────────────────────────────────────────────────────────────

/// Linear Quadratic Regulator (LQR) for discrete-time systems:
///
/// ```text
/// x_{k+1} = A x_k + B u_k
/// J = Σ (x_k^T Q x_k + u_k^T R u_k)
/// ```
///
/// Solves the Discrete Algebraic Riccati Equation (DARE) iteratively to
/// obtain feedback gain **K** such that `u_k = −K x_k` is optimal.
pub struct LqrSolver {
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl Default for LqrSolver {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-9,
        }
    }
}

impl LqrSolver {
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
        }
    }

    /// Compute the optimal feedback gain matrix **K** (m×n).
    /// Returns `K` such that `u = −K x` minimises the LQR cost.
    pub fn compute_gain(
        &self,
        a: &DMatrix<f64>,
        b: &DMatrix<f64>,
        q: &DMatrix<f64>,
        r: &DMatrix<f64>,
    ) -> Result<DMatrix<f64>, ControlError> {
        let n = a.nrows();
        let m = b.ncols();
        if a.ncols() != n
            || b.nrows() != n
            || q.nrows() != n
            || q.ncols() != n
            || r.nrows() != m
            || r.ncols() != m
        {
            return Err(ControlError::InvalidDimension(
                "System dimensions mismatch (A:n×n, B:n×m, Q:n×n, R:m×m)".into(),
            ));
        }

        let mut p = q.clone();
        let mut converged = false;

        for _ in 0..self.max_iterations {
            let temp = r + b.transpose() * &p * b;
            let temp_inv = temp.try_inverse().ok_or(ControlError::SingularMatrix)?;

            let at_p = a.transpose() * &p;
            let at_p_a = &at_p * a;
            let gain_part = &at_p * b * &temp_inv * b.transpose() * &p * a;
            let p_next = q + at_p_a - gain_part;

            let diff = (&p_next - &p).norm();
            p = p_next;

            if diff < self.tolerance {
                converged = true;
                break;
            }
        }

        if !converged {
            return Err(ControlError::NonConvergence(self.max_iterations));
        }

        // K = (R + B^T P B)^{-1} B^T P A
        let temp = r + b.transpose() * &p * b;
        let temp_inv = temp.try_inverse().ok_or(ControlError::SingularMatrix)?;
        Ok(temp_inv * b.transpose() * p * a)
    }

    /// Convenience: compute gain and return the steady-state Riccati solution P as well.
    pub fn compute_gain_and_cost(
        &self,
        a: &DMatrix<f64>,
        b: &DMatrix<f64>,
        q: &DMatrix<f64>,
        r: &DMatrix<f64>,
    ) -> Result<(DMatrix<f64>, DMatrix<f64>), ControlError> {
        let n = a.nrows();
        let m = b.ncols();
        if a.ncols() != n
            || b.nrows() != n
            || q.nrows() != n
            || q.ncols() != n
            || r.nrows() != m
            || r.ncols() != m
        {
            return Err(ControlError::InvalidDimension(
                "System dimensions mismatch".into(),
            ));
        }

        let mut p = q.clone();
        let mut converged = false;

        for _ in 0..self.max_iterations {
            let temp = r + b.transpose() * &p * b;
            let temp_inv = temp.try_inverse().ok_or(ControlError::SingularMatrix)?;
            let at_p = a.transpose() * &p;
            let p_next = q + &at_p * a - &at_p * b * &temp_inv * b.transpose() * &p * a;
            let diff = (&p_next - &p).norm();
            p = p_next;
            if diff < self.tolerance {
                converged = true;
                break;
            }
        }

        if !converged {
            return Err(ControlError::NonConvergence(self.max_iterations));
        }

        let temp = r + b.transpose() * &p * b;
        let temp_inv = temp.try_inverse().ok_or(ControlError::SingularMatrix)?;
        let k = &temp_inv * b.transpose() * &p * a;
        Ok((k, p))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kalman Filter
// ─────────────────────────────────────────────────────────────────────────────

/// Discrete-time Kalman Filter state estimator.
///
/// ```text
/// Process:     x_k = A x_{k-1} + B u_{k-1} + w_k,   w_k ~ N(0, Q)
/// Measurement: z_k = H x_k + v_k,                   v_k ~ N(0, R)
/// ```
pub struct KalmanFilter {
    /// State estimate vector (n × 1)
    pub x: DVector<f64>,
    /// State covariance matrix (n × n)
    pub p: DMatrix<f64>,
    /// State transition matrix (n × n)
    pub a: DMatrix<f64>,
    /// Control input matrix (n × m)
    pub b: DMatrix<f64>,
    /// Measurement matrix (p × n)
    pub h: DMatrix<f64>,
    /// Process noise covariance (n × n)
    pub q: DMatrix<f64>,
    /// Measurement noise covariance (p × p)
    pub r: DMatrix<f64>,
}

impl KalmanFilter {
    pub fn new(
        x: DVector<f64>,
        p: DMatrix<f64>,
        a: DMatrix<f64>,
        b: DMatrix<f64>,
        h: DMatrix<f64>,
        q: DMatrix<f64>,
        r: DMatrix<f64>,
    ) -> Self {
        Self {
            x,
            p,
            a,
            b,
            h,
            q,
            r,
        }
    }

    /// Predict step: propagates state estimate and covariance forward one step.
    pub fn predict(&mut self, u: &DVector<f64>) {
        self.x = &self.a * &self.x + &self.b * u;
        self.p = &self.a * &self.p * self.a.transpose() + &self.q;
    }

    /// Update step: corrects state estimate and covariance with a measurement.
    pub fn update(&mut self, z: &DVector<f64>) -> Result<(), ControlError> {
        let y = z - &self.h * &self.x;
        let s = &self.h * &self.p * self.h.transpose() + &self.r;
        let s_inv = s.try_inverse().ok_or(ControlError::SingularMatrix)?;
        let k = &self.p * self.h.transpose() * s_inv;
        self.x = &self.x + &k * y;
        let n = self.x.len();
        let identity = DMatrix::<f64>::identity(n, n);
        self.p = (identity - k * &self.h) * &self.p;
        Ok(())
    }

    /// Predicted innovation covariance S = H P H^T + R — useful for outlier gating.
    pub fn innovation_covariance(&self) -> DMatrix<f64> {
        &self.h * &self.p * self.h.transpose() + &self.r
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LQG Controller (LQR + Kalman combined)
// ─────────────────────────────────────────────────────────────────────────────

/// Linear Quadratic Gaussian (LQG) controller.
///
/// Combines an optimal state estimator ([`KalmanFilter`]) with an optimal
/// state-feedback controller ([`LqrSolver`]).  Given noisy output measurements
/// the LQG controller computes the optimal control action without needing
/// direct access to the full state vector.
///
/// # Usage
/// 1. Construct with system matrices and noise covariances.
/// 2. Call [`LqgController::step`] each cycle with the latest measurement.
pub struct LqgController {
    /// Optimal feedback gain K (m × n)
    pub k: DMatrix<f64>,
    /// Internal Kalman state estimator
    pub filter: KalmanFilter,
}

impl LqgController {
    /// Build an LQG controller.
    ///
    /// - `a`, `b` — process dynamics
    /// - `h` — measurement matrix
    /// - `q_lqr`, `r_lqr` — LQR state and input cost
    /// - `q_kf`, `r_kf`   — Kalman process and measurement noise covariances
    /// - `x0`, `p0`       — initial state estimate and covariance
    pub fn new(
        a: DMatrix<f64>,
        b: DMatrix<f64>,
        h: DMatrix<f64>,
        q_lqr: &DMatrix<f64>,
        r_lqr: &DMatrix<f64>,
        q_kf: DMatrix<f64>,
        r_kf: DMatrix<f64>,
        x0: DVector<f64>,
        p0: DMatrix<f64>,
    ) -> Result<Self, ControlError> {
        let solver = LqrSolver::default();
        let k = solver.compute_gain(&a, &b, q_lqr, r_lqr)?;
        let n = a.nrows();
        let m = b.ncols();
        let b_kf = DMatrix::<f64>::zeros(n, m); // no known control input for filter
        let filter = KalmanFilter::new(x0, p0, a, b_kf, h, q_kf, r_kf);
        Ok(Self { k, filter })
    }

    /// Execute one control cycle.
    ///
    /// Predicts state forward (with zero control — uses last estimate),
    /// incorporates measurement `z`, then returns `u = -K x̂`.
    pub fn step(&mut self, z: &DVector<f64>) -> Result<DVector<f64>, ControlError> {
        let u_zero = DVector::<f64>::zeros(self.filter.b.ncols());
        self.filter.predict(&u_zero);
        self.filter.update(z)?;
        Ok(-&self.k * &self.filter.x)
    }

    /// Current state estimate.
    pub fn state_estimate(&self) -> &DVector<f64> {
        &self.filter.x
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Linear MPC
// ─────────────────────────────────────────────────────────────────────────────

/// Simple Model Predictive Control (MPC) linear solver.
///
/// Solves an unconstrained finite-horizon LQR via batch prediction matrices.
/// Returns only the first optimal input (receding horizon principle).
pub struct LinearMpc {
    a: DMatrix<f64>,
    b: DMatrix<f64>,
    q: DMatrix<f64>,
    r: DMatrix<f64>,
    /// Prediction / control horizon N
    pub horizon: usize,
}

impl LinearMpc {
    pub fn new(
        a: DMatrix<f64>,
        b: DMatrix<f64>,
        q: DMatrix<f64>,
        r: DMatrix<f64>,
        horizon: usize,
    ) -> Self {
        Self {
            a,
            b,
            q,
            r,
            horizon,
        }
    }

    /// Compute first optimal control input for state `x0` tracking `target_path`.
    ///
    /// `target_path` must have at least `horizon` entries.
    pub fn compute_action(
        &self,
        x0: &DVector<f64>,
        target_path: &[DVector<f64>],
    ) -> Result<DVector<f64>, ControlError> {
        let n = self.a.nrows();
        let m = self.b.ncols();
        if target_path.len() < self.horizon {
            return Err(ControlError::InvalidDimension(
                "Target path length must be at least equal to horizon".into(),
            ));
        }

        // Build batch matrices Sx (n·N × n) and Su (n·N × m·N)
        let mut sx = DMatrix::<f64>::zeros(n * self.horizon, n);
        let mut su = DMatrix::<f64>::zeros(n * self.horizon, m * self.horizon);

        let mut a_power = self.a.clone();
        for i in 0..self.horizon {
            // Use view_mut (nalgebra ≥ 0.33) instead of deprecated slice_mut
            sx.view_mut((i * n, 0), (n, n)).copy_from(&a_power);

            for j in 0..=i {
                let diff = i - j;
                let coeff = if diff == 0 {
                    self.b.clone()
                } else {
                    let mut a_pow_diff = DMatrix::<f64>::identity(n, n);
                    for _ in 0..diff {
                        a_pow_diff = &a_pow_diff * &self.a;
                    }
                    a_pow_diff * &self.b
                };
                su.view_mut((i * n, j * m), (n, m)).copy_from(&coeff);
            }
            a_power = &a_power * &self.a;
        }

        // Build block-diagonal cost matrices and reference trajectory
        let mut q_bar = DMatrix::<f64>::zeros(n * self.horizon, n * self.horizon);
        let mut r_bar = DMatrix::<f64>::zeros(m * self.horizon, m * self.horizon);
        let mut x_ref = DVector::<f64>::zeros(n * self.horizon);

        for i in 0..self.horizon {
            q_bar.view_mut((i * n, i * n), (n, n)).copy_from(&self.q);
            r_bar.view_mut((i * m, i * m), (m, m)).copy_from(&self.r);
            // view_mut on DVector uses (row, col) → treat as matrix column view
            x_ref.rows_mut(i * n, n).copy_from(&target_path[i]);
        }

        // Analytical solution (unconstrained QP):
        // U* = (Su^T Q̄ Su + R̄)^{-1} Su^T Q̄ (X_ref − Sx x0)
        let error_offset = x_ref - &sx * x0;
        let lhs = su.transpose() * &q_bar * &su + &r_bar;
        let rhs = su.transpose() * &q_bar * error_offset;
        let lhs_inv = lhs.try_inverse().ok_or(ControlError::SingularMatrix)?;
        let u_opt = lhs_inv * rhs;

        // Return first control vector u₀ (m × 1)
        Ok(u_opt.rows(0, m).into_owned())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// H-∞ Robustness Bound
// ─────────────────────────────────────────────────────────────────────────────

/// H-∞ robustness analysis via the structured singular value (μ) upper bound.
///
/// For a closed-loop system with gain matrix `K` and plant `A`, computes an
/// upper bound on the H-∞ norm of the sensitivity function `(I − K A)^{-1}`.
/// A small bound (< 1) indicates robust stability against bounded perturbations.
///
/// # Background
/// The H-∞ norm of a transfer function bounds the worst-case amplification of
/// exogenous disturbances.  We approximate it as:
/// `‖(I − KA)^{-1}‖₂ ≤ 1 / (1 − ‖KA‖₂)`
/// when `‖KA‖₂ < 1` (small-gain theorem).
pub struct HInfBound;

impl HInfBound {
    /// Compute the spectral norm upper bound for the sensitivity operator
    /// `(I − KA)^{-1}` using the small-gain theorem.
    ///
    /// Returns `Err(ControlError::NotStable)` if the open-loop gain `‖KA‖₂ ≥ 1`
    /// (the bound is not finite, so stability cannot be certified this way).
    pub fn sensitivity_bound(k: &DMatrix<f64>, a: &DMatrix<f64>) -> Result<f64, ControlError> {
        if k.ncols() != a.nrows() {
            return Err(ControlError::InvalidDimension(
                "K columns must match A rows for K·A product".into(),
            ));
        }
        let ka = k * a;
        let sigma_max = Self::spectral_norm(&ka);
        if sigma_max >= 1.0 {
            return Err(ControlError::NotStable);
        }
        Ok(1.0 / (1.0 - sigma_max))
    }

    /// Compute the maximum singular value (spectral norm / 2-norm) of a matrix.
    ///
    /// Uses the power-iteration method for efficiency — adequate for online
    /// robustness monitoring within the cognitive loop.
    pub fn spectral_norm(m: &DMatrix<f64>) -> f64 {
        // σ_max(M) = √λ_max(M^T M)
        let mtm = m.transpose() * m;
        // Largest eigenvalue via simple power iteration
        let n = mtm.nrows();
        if n == 0 {
            return 0.0;
        }
        let mut v = DVector::<f64>::from_element(n, 1.0 / (n as f64).sqrt());
        for _ in 0..200 {
            let mv = &mtm * &v;
            let norm = mv.norm();
            if norm < 1e-14 {
                return 0.0;
            }
            v = mv / norm;
        }
        let mv = &mtm * &v;
        let lambda_max = v.dot(&mv);
        lambda_max.sqrt().max(0.0)
    }

    /// Returns `true` if the small-gain theorem certifies robust stability.
    pub fn is_robustly_stable(k: &DMatrix<f64>, a: &DMatrix<f64>) -> bool {
        Self::sensitivity_bound(k, a).is_ok()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Lyapunov Stability Checker
// ─────────────────────────────────────────────────────────────────────────────

/// Lyapunov stability verifier for discrete-time closed-loop systems.
///
/// A discrete-time system `x_{k+1} = A_cl x_k` is (globally asymptotically)
/// stable iff there exists a positive-definite matrix **P** satisfying the
/// discrete Lyapunov equation:
///
/// ```text
/// A_cl^T P A_cl − P + Q = 0
/// ```
///
/// for some positive-definite **Q**.  Given **A_cl** (= A − BK for a
/// state-feedback system), this checker solves for **P** iteratively and
/// verifies positive-definiteness.
pub struct LyapunovChecker {
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl Default for LyapunovChecker {
    fn default() -> Self {
        Self {
            max_iterations: 2000,
            tolerance: 1e-9,
        }
    }
}

impl LyapunovChecker {
    /// Solve the discrete Lyapunov equation and verify positive-definiteness of P.
    ///
    /// `a_cl` is the closed-loop matrix (e.g. `A − B K`).
    /// `q`    is the positive-definite weighting matrix (commonly `I`).
    ///
    /// Returns `Ok(P)` if the system is stable, `Err(ControlError::NotStable)` otherwise.
    pub fn verify(
        &self,
        a_cl: &DMatrix<f64>,
        q: &DMatrix<f64>,
    ) -> Result<DMatrix<f64>, ControlError> {
        let n = a_cl.nrows();
        if a_cl.ncols() != n || q.nrows() != n || q.ncols() != n {
            return Err(ControlError::InvalidDimension(
                "A_cl and Q must be n×n".into(),
            ));
        }

        // Solve A_cl^T P A_cl − P + Q = 0  ⟺  P_{k+1} = A_cl^T P_k A_cl + Q
        // Converges iff spectral radius of A_cl < 1.
        let mut p = q.clone();
        let at = a_cl.transpose();

        for _ in 0..self.max_iterations {
            let p_next = &at * &p * a_cl + q;
            let diff = (&p_next - &p).norm();
            p = p_next;
            if diff < self.tolerance {
                // Verify P is positive-definite via Cholesky decomposition
                if p.clone().cholesky().is_some() {
                    return Ok(p);
                } else {
                    return Err(ControlError::NotStable);
                }
            }
        }
        Err(ControlError::NonConvergence(self.max_iterations))
    }

    /// Quick boolean stability check (does not return P).
    pub fn is_stable(&self, a_cl: &DMatrix<f64>, q: &DMatrix<f64>) -> bool {
        self.verify(a_cl, q).is_ok()
    }

    /// Compute the spectral radius ρ(A_cl) = max |λ_i|.
    ///
    /// A system is stable iff ρ < 1.  Uses the power method to approximate the
    /// dominant eigenvalue magnitude.
    pub fn spectral_radius(a_cl: &DMatrix<f64>) -> f64 {
        let n = a_cl.nrows();
        if n == 0 {
            return 0.0;
        }
        let mut v = DVector::<f64>::from_element(n, 1.0 / (n as f64).sqrt());
        let mut prev_norm = 0.0_f64;
        for _ in 0..500 {
            let av = a_cl * &v;
            let norm = av.norm();
            if norm < 1e-14 {
                return 0.0;
            }
            v = av / norm;
            if (norm - prev_norm).abs() < 1e-10 {
                return norm;
            }
            prev_norm = norm;
        }
        prev_norm
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Consciousness-Gated Gain Scheduler
// ─────────────────────────────────────────────────────────────────────────────

/// Consciousness-gated adaptive gain scheduler.
///
/// Scales the LQR feedback gain matrix **K** by a factor derived from the
/// current Integrated Information (Φ) level.  Higher Φ → more conservative
/// control (lower gain, softer actuation); lower Φ → more aggressive recovery.
///
/// # Rationale
/// In Symthaea's architecture the consciousness level modulates confidence in
/// the internal world model.  High Φ indicates strong self-model coherence, so
/// small control corrections suffice.  Low Φ (distress, uncertainty) warrants
/// stronger corrective action.
///
/// The mapping used is a logistic schedule:
/// `α(Φ) = α_min + (α_max − α_min) · σ(−k · (Φ − Φ_mid))`
/// where σ is the logistic sigmoid and k is the schedule steepness.
pub struct ConsciousnessGainScheduler {
    /// Minimum gain scale (at Φ = 1.0, maximum consciousness)
    pub alpha_min: f64,
    /// Maximum gain scale (at Φ = 0.0, minimum consciousness)
    pub alpha_max: f64,
    /// Midpoint of Φ at which α = (α_min + α_max) / 2
    pub phi_mid: f64,
    /// Steepness of the sigmoid schedule (larger = sharper transition)
    pub steepness: f64,
}

impl Default for ConsciousnessGainScheduler {
    fn default() -> Self {
        Self {
            alpha_min: 0.4,
            alpha_max: 1.6,
            phi_mid: 0.5,
            steepness: 8.0,
        }
    }
}

impl ConsciousnessGainScheduler {
    /// Compute the gain scale α ∈ [α_min, α_max] for a given Φ ∈ [0, 1].
    ///
    /// `phi` is clamped to [0, 1] before evaluation.
    pub fn alpha(&self, phi: f64) -> f64 {
        let phi = phi.clamp(0.0, 1.0);
        let sigma = 1.0 / (1.0 + (self.steepness * (phi - self.phi_mid)).exp());
        self.alpha_min + (self.alpha_max - self.alpha_min) * sigma
    }

    /// Return the gain matrix scaled by α(Φ).
    pub fn schedule_gain(&self, k: &DMatrix<f64>, phi: f64) -> DMatrix<f64> {
        k * self.alpha(phi)
    }

    /// Convenience: return a scaled control action `u = −α(Φ) · K · x`.
    pub fn compute_action(&self, k: &DMatrix<f64>, x: &DVector<f64>, phi: f64) -> DVector<f64> {
        -self.schedule_gain(k, phi) * x
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: canonical double-integrator matrices
    fn double_integrator() -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>, DMatrix<f64>) {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 0.0, 1.0]);
        let b = DMatrix::from_row_slice(2, 1, &[0.0, 1.0]);
        let q = DMatrix::identity(2, 2);
        let r = DMatrix::from_element(1, 1, 1.0);
        (a, b, q, r)
    }

    // ── LQR ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_lqr_convergence() {
        let (a, b, q, r) = double_integrator();
        let solver = LqrSolver::default();
        let k = solver.compute_gain(&a, &b, &q, &r).unwrap();
        assert_eq!(k.nrows(), 1);
        assert_eq!(k.ncols(), 2);
        // Both gains should be positive for this standard system
        assert!(k[0] > 0.0, "K[0] should be positive, got {}", k[0]);
        assert!(k[1] > 0.0, "K[1] should be positive, got {}", k[1]);
    }

    #[test]
    fn test_lqr_returns_cost_matrix() {
        let (a, b, q, r) = double_integrator();
        let solver = LqrSolver::default();
        let (k, p) = solver.compute_gain_and_cost(&a, &b, &q, &r).unwrap();
        assert!(k.norm() > 0.0);
        // P must be positive-definite (Cholesky must succeed)
        assert!(p.cholesky().is_some(), "P should be positive-definite");
    }

    #[test]
    fn test_lqr_dimension_mismatch() {
        let a = DMatrix::<f64>::identity(3, 3);
        let b = DMatrix::<f64>::zeros(3, 1);
        let q = DMatrix::<f64>::identity(2, 2); // wrong size
        let r = DMatrix::<f64>::identity(1, 1);
        let result = LqrSolver::default().compute_gain(&a, &b, &q, &r);
        assert!(matches!(result, Err(ControlError::InvalidDimension(_))));
    }

    // ── Kalman Filter ───────────────────────────────────────────────────────

    #[test]
    fn test_kalman_filter_predict() {
        let x = DVector::from_vec(vec![0.0]);
        let p = DMatrix::from_element(1, 1, 1.0);
        let a = DMatrix::from_element(1, 1, 1.0);
        let b = DMatrix::from_element(1, 1, 1.0);
        let h = DMatrix::from_element(1, 1, 1.0);
        let q = DMatrix::from_element(1, 1, 0.1);
        let r = DMatrix::from_element(1, 1, 0.1);

        let mut kf = KalmanFilter::new(x, p, a, b, h, q, r);
        let u = DVector::from_vec(vec![1.0]);
        kf.predict(&u);
        assert!(
            (kf.x[0] - 1.0).abs() < 1e-6,
            "After predict x should be ≈ 1.0"
        );
    }

    #[test]
    fn test_kalman_filter_update() {
        let x = DVector::from_vec(vec![0.0]);
        let p = DMatrix::from_element(1, 1, 1.0);
        let a = DMatrix::from_element(1, 1, 1.0);
        let b = DMatrix::from_element(1, 1, 1.0);
        let h = DMatrix::from_element(1, 1, 1.0);
        let q = DMatrix::from_element(1, 1, 0.1);
        let r = DMatrix::from_element(1, 1, 0.1);

        let mut kf = KalmanFilter::new(x, p, a, b, h, q, r);
        let u = DVector::from_vec(vec![1.0]);
        kf.predict(&u);
        let z = DVector::from_vec(vec![1.1]);
        kf.update(&z).unwrap();
        // With equal Q and R, update should pull state toward measurement
        assert!((kf.x[0] - 1.09).abs() < 0.05);
    }

    #[test]
    fn test_kalman_innovation_covariance() {
        let n = 2_usize;
        let x = DVector::zeros(n);
        let p = DMatrix::identity(n, n);
        let a = DMatrix::identity(n, n);
        let b = DMatrix::zeros(n, 1);
        let h = DMatrix::identity(n, n);
        let q = DMatrix::identity(n, n) * 0.01;
        let r = DMatrix::identity(n, n) * 0.1;

        let kf = KalmanFilter::new(x, p, a, b, h, q, r);
        let s = kf.innovation_covariance();
        // S = H P H^T + R = I + 0.1 I = 1.1 I
        assert!((s[(0, 0)] - 1.1).abs() < 1e-9);
    }

    // ── LQG ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_lqg_controller_step() {
        let (a, b, q_lqr, r_lqr) = double_integrator();
        let h = DMatrix::identity(2, 2);
        let q_kf = DMatrix::identity(2, 2) * 0.01;
        let r_kf = DMatrix::identity(2, 2) * 0.1;
        let x0 = DVector::from_vec(vec![1.0, 0.0]);
        let p0 = DMatrix::identity(2, 2);

        let mut lqg = LqgController::new(a, b, h, &q_lqr, &r_lqr, q_kf, r_kf, x0, p0).unwrap();

        // Step with a measurement near the initial state
        let z = DVector::from_vec(vec![1.0, 0.0]);
        let u = lqg.step(&z).unwrap();
        // The LQG should produce a control action that drives state toward origin
        assert_eq!(u.len(), 1);
        assert!(
            u[0] < 0.0,
            "Control should push state toward origin, got u={}",
            u[0]
        );
    }

    #[test]
    fn test_lqg_state_estimate_converges() {
        let (a, b, q_lqr, r_lqr) = double_integrator();
        let h = DMatrix::identity(2, 2);
        let q_kf = DMatrix::identity(2, 2) * 0.01;
        let r_kf = DMatrix::identity(2, 2) * 0.1;
        let x0 = DVector::zeros(2);
        let p0 = DMatrix::identity(2, 2);

        let mut lqg = LqgController::new(a, b, h, &q_lqr, &r_lqr, q_kf, r_kf, x0, p0).unwrap();

        // Observe same measurement repeatedly — filter should converge
        let z = DVector::from_vec(vec![2.0, 0.5]);
        for _ in 0..50 {
            let _ = lqg.step(&z).unwrap();
        }
        let x_hat = lqg.state_estimate();
        // After many steps with constant measurement the estimate should be close
        assert!(
            (x_hat[0] - 2.0).abs() < 0.5,
            "State estimate should track measurement"
        );
    }

    // ── MPC ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_linear_mpc_drive_to_origin() {
        let (a, b, q, r) = double_integrator();
        let horizon = 3;
        let mpc = LinearMpc::new(a, b, q, r, horizon);
        let x0 = DVector::from_vec(vec![0.5, 0.0]);
        let target_path = vec![DVector::zeros(2); horizon];

        let u0 = mpc.compute_action(&x0, &target_path).unwrap();
        assert_eq!(u0.len(), 1);
        // Must produce braking action (negative for positive position error)
        assert!(
            u0[0] < 0.0,
            "MPC should produce negative u to drive toward origin"
        );
    }

    #[test]
    fn test_linear_mpc_path_too_short() {
        let (a, b, q, r) = double_integrator();
        let mpc = LinearMpc::new(a, b, q, r, 5);
        let x0 = DVector::zeros(2);
        let target_path = vec![DVector::zeros(2); 3]; // too short
        let result = mpc.compute_action(&x0, &target_path);
        assert!(matches!(result, Err(ControlError::InvalidDimension(_))));
    }

    // ── H-∞ Bound ───────────────────────────────────────────────────────────

    #[test]
    fn test_hinf_small_gain_stable() {
        // Stable if ‖KA‖₂ < 1 — use a gain and plant that satisfy this
        let k = DMatrix::from_row_slice(1, 2, &[0.1, 0.1]);
        let a = DMatrix::from_row_slice(2, 2, &[0.5, 0.0, 0.0, 0.5]);
        let bound = HInfBound::sensitivity_bound(&k, &a).unwrap();
        assert!(bound > 1.0, "Sensitivity bound must be ≥ 1");
        assert!(bound < 100.0, "Bound should be finite for stable system");
        assert!(HInfBound::is_robustly_stable(&k, &a));
    }

    #[test]
    fn test_hinf_unstable_system() {
        // Gain × plant spectral norm > 1 → NotStable
        let k = DMatrix::from_row_slice(2, 2, &[5.0, 0.0, 0.0, 5.0]);
        let a = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 2.0]);
        let result = HInfBound::sensitivity_bound(&k, &a);
        assert!(matches!(result, Err(ControlError::NotStable)));
    }

    #[test]
    fn test_spectral_norm_identity() {
        let m = DMatrix::<f64>::identity(3, 3);
        let sigma = HInfBound::spectral_norm(&m);
        assert!((sigma - 1.0).abs() < 0.01, "σ_max(I) = 1, got {sigma}");
    }

    // ── Lyapunov ─────────────────────────────────────────────────────────────

    #[test]
    fn test_lyapunov_stable_system() {
        // Closed-loop with eigenvalues strictly inside unit circle
        // A_cl = 0.5 I is clearly stable (ρ = 0.5 < 1)
        let a_cl = DMatrix::from_row_slice(2, 2, &[0.5, 0.0, 0.0, 0.5]);
        let q = DMatrix::identity(2, 2);
        let checker = LyapunovChecker::default();
        let p = checker.verify(&a_cl, &q).unwrap();
        assert!(
            p.cholesky().is_some(),
            "P must be positive-definite for stable system"
        );
    }

    #[test]
    fn test_lyapunov_unstable_system() {
        // A_cl = 1.5 I is unstable (ρ = 1.5 > 1)
        let a_cl = DMatrix::from_row_slice(2, 2, &[1.5, 0.0, 0.0, 1.5]);
        let q = DMatrix::identity(2, 2);
        let checker = LyapunovChecker::default();
        let result = checker.verify(&a_cl, &q);
        assert!(
            result.is_err(),
            "Unstable system should fail Lyapunov check"
        );
    }

    #[test]
    fn test_lyapunov_lqr_closed_loop_is_stable() {
        // An LQR-designed controller must produce a stable closed-loop system
        let (a, b, q, r) = double_integrator();
        let solver = LqrSolver::default();
        let k = solver.compute_gain(&a, &b, &q, &r).unwrap();
        // A_cl = A - B K
        let a_cl = &a - &b * &k;
        let checker = LyapunovChecker::default();
        assert!(
            checker.is_stable(&a_cl, &DMatrix::identity(2, 2)),
            "LQR closed-loop must be Lyapunov-stable"
        );
    }

    #[test]
    fn test_spectral_radius_stable() {
        let a_cl = DMatrix::from_row_slice(2, 2, &[0.3, 0.0, 0.0, 0.7]);
        let rho = LyapunovChecker::spectral_radius(&a_cl);
        // Dominant eigenvalue magnitude ≈ 0.7
        assert!(
            rho < 1.0,
            "Spectral radius should be < 1 for stable system, got {rho}"
        );
    }

    // ── Consciousness Gain Scheduler ────────────────────────────────────────

    #[test]
    fn test_gain_scheduler_high_phi_reduces_gain() {
        let scheduler = ConsciousnessGainScheduler::default();
        let alpha_high = scheduler.alpha(0.95);
        let alpha_low = scheduler.alpha(0.05);
        assert!(
            alpha_high < alpha_low,
            "High Φ should produce lower gain scale: α(0.95)={alpha_high:.4}, α(0.05)={alpha_low:.4}"
        );
    }

    #[test]
    fn test_gain_scheduler_midpoint() {
        let scheduler = ConsciousnessGainScheduler::default();
        let mid = scheduler.alpha(0.5);
        let expected = (scheduler.alpha_min + scheduler.alpha_max) / 2.0;
        assert!(
            (mid - expected).abs() < 0.01,
            "α(Φ_mid) ≈ (α_min+α_max)/2, got {mid:.4}"
        );
    }

    #[test]
    fn test_gain_scheduler_clamps_phi() {
        let scheduler = ConsciousnessGainScheduler::default();
        // Should not panic or produce out-of-range values
        let a1 = scheduler.alpha(-0.5);
        let a2 = scheduler.alpha(1.5);
        assert!((a1 - scheduler.alpha(0.0)).abs() < 1e-9);
        assert!((a2 - scheduler.alpha(1.0)).abs() < 1e-9);
    }

    #[test]
    fn test_gain_scheduler_action_direction() {
        let scheduler = ConsciousnessGainScheduler::default();
        let (a, b, q, r) = double_integrator();
        let k = LqrSolver::default().compute_gain(&a, &b, &q, &r).unwrap();
        let x = DVector::from_vec(vec![1.0, 0.0]); // positive offset from origin
        let u = scheduler.compute_action(&k, &x, 0.5);
        assert!(
            u[0] < 0.0,
            "Control action should be negative to correct positive offset"
        );
    }
}
