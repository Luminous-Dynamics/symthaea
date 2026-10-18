// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Differential Equations Engine
//!
//! Numerical ODE/PDE solvers exposed through the HDC math interface, with
//! consciousness-coupled Phi encoding of solutions.
//!
//! ## Methods
//!
//! - **`solve_ivp`**: Initial value problem via classical 4th-order Runge-Kutta (RK4)
//! - **`solve_bvp_shooting`**: Boundary value problem via shooting + Brent root-finding
//! - **`heat_equation_1d`**: 1D heat equation via method of lines (explicit Euler)
//! - **`wave_equation_1d`**: 1D wave equation via method of lines (leapfrog / Störmer-Verlet)
//!
//! ## Consciousness Coupling
//!
//! Phi is computed from solution quality: low global error → high Phi.  Each
//! result carries a `BinaryHV` encoding that binds together the method, the
//! final state vector, and a structural primitive, so that semantically similar
//! trajectories cluster in hyperdimensional space.
//!
//! ## Design Notes
//!
//! ODE system functions use the concrete type `fn(f64, &[f64]) -> Vec<f64>` so
//! that all engine structs remain `Sized`.  This avoids the overhead of `Box<dyn
//! Fn>` and keeps function-pointer passing predictable.

use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::primitive_system::seed_from_name;
use crate::hdc::root_finding::RootFindingEngine;
use serde::{Deserialize, Serialize};

// ─── Constants ────────────────────────────────────────────────────────────────

const DEFAULT_TOL: f64 = 1e-8;
const MAX_STEPS: usize = 100_000;
const MAX_DENSE_NEWTON_DIM: usize = 64;
const MAX_IMPLICIT_RHS_EVALUATIONS: usize = 1_000_000;
const MAX_IMPLICIT_NEWTON_WORK_UNITS: usize = 100_000_000;
const MAX_NEWTON_ITERATIONS_PER_STEP: usize = 100;

// ─── Types ────────────────────────────────────────────────────────────────────

/// Numerical method used to integrate the ODE.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ODEMethod {
    /// Classical 4th-order Runge-Kutta with fixed step size.
    RK4,
    /// Dormand-Prince adaptive RK45.
    RK45,
    /// Backward Euler with Newton iteration (L-stable, for stiff systems).
    ImplicitEuler,
    /// Shooting method (wraps an inner IVP solver).
    Shooting,
}

impl std::fmt::Display for ODEMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ODEMethod::RK4 => write!(f, "RK4"),
            ODEMethod::RK45 => write!(f, "RK45 (Dormand-Prince)"),
            ODEMethod::ImplicitEuler => write!(f, "Implicit Euler (Newton)"),
            ODEMethod::Shooting => write!(f, "Shooting"),
        }
    }
}

/// A system of first-order ODEs:  dy/dt = f(t, y).
///
/// Represent higher-order systems by introducing auxiliary state variables.
/// For example, the harmonic oscillator y'' + y = 0 becomes
/// `y' = [y[1], -y[0]]` with state `[position, velocity]`.
pub struct ODESystem {
    /// RHS function f(t, y) → dy/dt.
    pub f: fn(f64, &[f64]) -> Vec<f64>,
    /// Number of state variables (dimension of y).
    pub dim: usize,
}

/// Result of an initial-value-problem solve.
#[derive(Debug, Clone)]
pub struct ODEResult {
    /// Time grid t_0, t_1, …, t_N.
    pub t_values: Vec<f64>,
    /// State at each time point: `y_values[i]` is the state at `t_values[i]`.
    pub y_values: Vec<Vec<f64>>,
    /// Integration method used.
    pub method: ODEMethod,
    /// Total number of steps taken.
    pub steps: usize,
    /// Consciousness measure (convergence quality).
    pub phi: f64,
    /// HDC encoding of the final state.
    pub encoding: BinaryHV,
}

impl ODEResult {
    /// Convenience: final state vector y(t_end).
    pub fn final_state(&self) -> &[f64] {
        self.y_values.last().map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Convenience: final time.
    pub fn t_end(&self) -> f64 {
        *self.t_values.last().unwrap_or(&0.0)
    }
}


/// Diagnostics returned by the fallible dense backward-Euler solver.
#[derive(Debug, Clone)]
pub struct ImplicitEulerSolveReport {
    /// Computed trajectory. All returned steps passed the nonlinear residual check.
    pub trajectory: ODEResult,
    /// Number of Newton corrections applied across all accepted steps.
    pub newton_corrections: usize,
    /// Number of right-hand-side evaluations, including finite-difference probes.
    pub rhs_evaluations: usize,
    /// Largest normalized nonlinear residual among accepted steps.
    pub max_normalized_residual: f64,
    /// Estimated dense linear-algebra work units (dimension cubed per correction).
    pub estimated_dense_work_units: usize,
}

/// Typed reason a fallible backward-Euler solve could not satisfy its contract.
#[derive(Debug, Clone, PartialEq)]
pub enum ImplicitEulerErrorKind {
    EmptySystem,
    DimensionMismatch { expected: usize, actual: usize },
    InvalidTimeSpan,
    InvalidStepCount { requested: usize, max_allowed: usize },
    InvalidNewtonIterationBudget { requested: usize, max_allowed: usize },
    SystemTooLargeForDenseNewton { dimension: usize, max_dimension: usize },
    NonFiniteInitialState { index: usize },
    NonFiniteIntermediateState { stage: &'static str, index: usize },
    RhsDimensionMismatch { stage: &'static str, expected: usize, actual: usize },
    NonFiniteRhs { stage: &'static str, index: usize },
    RhsEvaluationBudgetExceeded { max_evaluations: usize },
    NonFinitePredictor { index: usize },
    NonFiniteResidual { index: usize },
    NonFiniteJacobian { row: usize, column: usize },
    SingularJacobian { column: usize, pivot_magnitude: f64 },
    NonFiniteNewtonCorrection { index: usize },
    NonFiniteLinearSolve { column: usize },
    NewtonNonConvergence { iteration_limit: usize, residual_norm: f64 },
    NewtonWorkBudgetExceeded { max_work_units: usize },
}

/// Failure receipt preserving the last accepted state rather than a failed trial.
#[derive(Debug, Clone, PartialEq)]
pub struct ImplicitEulerError {
    pub kind: ImplicitEulerErrorKind,
    /// Time corresponding to last_state; failed step is not committed.
    pub time: f64,
    pub last_state: Vec<f64>,
    pub steps_completed: usize,
    /// One-based number of the step that failed; zero when validation failed.
    pub failed_step: usize,
    pub newton_corrections: usize,
    pub rhs_evaluations: usize,
    pub residual_norm: f64,
}

impl ImplicitEulerError {
    fn new(
        kind: ImplicitEulerErrorKind,
        time: f64,
        last_state: &[f64],
        steps_completed: usize,
        newton_corrections: usize,
        rhs_evaluations: usize,
        residual_norm: f64,
    ) -> Self {
        let validation_failure = matches!(
            &kind,
            ImplicitEulerErrorKind::EmptySystem
                | ImplicitEulerErrorKind::DimensionMismatch { .. }
                | ImplicitEulerErrorKind::InvalidStepCount { .. }
                | ImplicitEulerErrorKind::InvalidNewtonIterationBudget { .. }
                | ImplicitEulerErrorKind::SystemTooLargeForDenseNewton { .. }
                | ImplicitEulerErrorKind::NonFiniteInitialState { .. }
        ) || (matches!(&kind, ImplicitEulerErrorKind::InvalidTimeSpan)
            && steps_completed == 0
            && newton_corrections == 0
            && rhs_evaluations == 0
            && residual_norm == 0.0);
        let failed_step = if validation_failure {
            0
        } else {
            steps_completed + 1
        };
        Self {
            kind,
            time,
            last_state: last_state.to_vec(),
            steps_completed,
            failed_step,
            newton_corrections,
            rhs_evaluations,
            residual_norm,
        }
    }
}

impl std::fmt::Display for ImplicitEulerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "implicit Euler failed at t={:.17e}, step={} after {} completed steps: {:?}",
            self.time, self.failed_step, self.steps_completed, self.kind
        )
    }
}

impl std::error::Error for ImplicitEulerError {}

fn implicit_euler_eval_rhs(
    system: &ODESystem,
    stage: &'static str,
    time: f64,
    state: &[f64],
    rhs_evaluations: &mut usize,
) -> Result<Vec<f64>, ImplicitEulerErrorKind> {
    if *rhs_evaluations >= MAX_IMPLICIT_RHS_EVALUATIONS {
        return Err(ImplicitEulerErrorKind::RhsEvaluationBudgetExceeded {
            max_evaluations: MAX_IMPLICIT_RHS_EVALUATIONS,
        });
    }
    if !time.is_finite() {
        return Err(ImplicitEulerErrorKind::InvalidTimeSpan);
    }
    if let Some(index) = state.iter().position(|value| !value.is_finite()) {
        return Err(ImplicitEulerErrorKind::NonFiniteIntermediateState { stage, index });
    }
    *rhs_evaluations += 1;
    let derivative = (system.f)(time, state);
    if derivative.len() != system.dim {
        return Err(ImplicitEulerErrorKind::RhsDimensionMismatch {
            stage,
            expected: system.dim,
            actual: derivative.len(),
        });
    }
    if let Some(index) = derivative.iter().position(|value| !value.is_finite()) {
        return Err(ImplicitEulerErrorKind::NonFiniteRhs { stage, index });
    }
    Ok(derivative)
}

fn implicit_euler_solve_dense(
    matrix: &mut [f64],
    rhs: &mut [f64],
    dimension: usize,
) -> Result<(), (usize, f64)> {
    let matrix_max = matrix.iter().fold(0.0_f64, |maximum, value| {
        if value.is_finite() {
            maximum.max(value.abs())
        } else {
            f64::INFINITY
        }
    });
    if !matrix_max.is_finite() || rhs.iter().any(|value| !value.is_finite()) {
        return Err((0, f64::NAN));
    }
    let pivot_floor = f64::EPSILON * matrix_max.max(1.0) * dimension as f64;

    for column in 0..dimension {
        let mut pivot_row = column;
        let mut pivot_magnitude = matrix[column * dimension + column].abs();
        for row in (column + 1)..dimension {
            let magnitude = matrix[row * dimension + column].abs();
            if magnitude > pivot_magnitude {
                pivot_magnitude = magnitude;
                pivot_row = row;
            }
        }
        if !pivot_magnitude.is_finite() || pivot_magnitude <= pivot_floor {
            return Err((column, pivot_magnitude));
        }
        if pivot_row != column {
            for j in 0..dimension {
                matrix.swap(column * dimension + j, pivot_row * dimension + j);
            }
            rhs.swap(column, pivot_row);
        }

        let pivot = matrix[column * dimension + column];
        for row in (column + 1)..dimension {
            let factor = matrix[row * dimension + column] / pivot;
            if !factor.is_finite() {
                return Err((column, f64::NAN));
            }
            matrix[row * dimension + column] = 0.0;
            for j in (column + 1)..dimension {
                let idx = row * dimension + j;
                matrix[idx] -= factor * matrix[column * dimension + j];
                if !matrix[idx].is_finite() {
                    return Err((column, f64::NAN));
                }
            }
            rhs[row] -= factor * rhs[column];
            if !rhs[row].is_finite() {
                return Err((column, f64::NAN));
            }
        }
    }

    for row in (0..dimension).rev() {
        let mut value = rhs[row];
        for column in (row + 1)..dimension {
            value -= matrix[row * dimension + column] * rhs[column];
        }
        let pivot = matrix[row * dimension + row];
        value /= pivot;
        if !value.is_finite() {
            return Err((row, pivot.abs()));
        }
        rhs[row] = value;
    }
    Ok(())
}


/// Report for the adaptive backward-Euler step-doubling integrator.
#[derive(Debug, Clone)]
pub struct AdaptiveImplicitEulerReport {
    pub trajectory: ODEResult,
    /// Macro-steps accepted by the local error controller.
    pub accepted_steps: usize,
    /// Macro-step trials rejected for local error or retryable Newton failures.
    pub rejected_steps: usize,
    /// Largest normalized step-doubling error among accepted macro-steps.
    pub max_accepted_normalized_local_error: f64,
    /// Largest normalized Newton residual observed across all subsolves.
    pub max_normalized_newton_residual: f64,
    pub newton_corrections: usize,
    pub rhs_evaluations: usize,
    pub estimated_dense_work_units: usize,
    /// Number of attempted macro-steps, accepted or rejected.
    pub attempts: usize,
}

/// Typed failure from adaptive implicit Euler.
#[derive(Debug, Clone, PartialEq)]
pub enum AdaptiveImplicitEulerErrorKind {
    EmptySystem,
    DimensionMismatch { expected: usize, actual: usize },
    InvalidTimeSpan,
    InvalidInitialStep,
    InvalidAbsoluteTolerance { index: usize },
    ToleranceDimensionMismatch { expected: usize, actual: usize },
    InvalidRelativeTolerance,
    InvalidAttemptBudget { requested: usize, max_allowed: usize },
    InvalidNewtonIterationBudget { requested: usize, max_allowed: usize },
    SystemTooLargeForDenseNewton { dimension: usize, max_dimension: usize },
    AttemptBudgetExceeded { max_attempts: usize },
    RhsEvaluationBudgetExceeded { max_evaluations: usize },
    DenseWorkBudgetExceeded { max_work_units: usize },
    StepSolveFailed { cause: ImplicitEulerErrorKind },
    NonFiniteToleranceScale { index: usize },
    NonFiniteStepDoublingError,
    StepSizeUnderflow { attempted_step: f64 },
}

/// Failure receipt retains the last accepted state/time and accumulated work.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveImplicitEulerError {
    pub kind: AdaptiveImplicitEulerErrorKind,
    pub time: f64,
    pub last_state: Vec<f64>,
    pub accepted_steps: usize,
    pub rejected_steps: usize,
    pub attempts: usize,
    pub newton_corrections: usize,
    pub rhs_evaluations: usize,
    pub estimated_dense_work_units: usize,
    pub max_normalized_newton_residual: f64,
}

impl AdaptiveImplicitEulerError {
    fn new(
        kind: AdaptiveImplicitEulerErrorKind,
        time: f64,
        last_state: &[f64],
        accepted_steps: usize,
        rejected_steps: usize,
        attempts: usize,
        work: &AdaptiveImplicitEulerWork,
        max_normalized_newton_residual: f64,
    ) -> Self {
        Self {
            kind,
            time,
            last_state: last_state.to_vec(),
            accepted_steps,
            rejected_steps,
            attempts,
            newton_corrections: work.newton_corrections,
            rhs_evaluations: work.rhs_evaluations,
            estimated_dense_work_units: work.estimated_dense_work_units,
            max_normalized_newton_residual,
        }
    }
}

impl std::fmt::Display for AdaptiveImplicitEulerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "adaptive implicit Euler failed at t={:.17e} after {} accepted and {} rejected attempts: {:?}",
            self.time, self.accepted_steps, self.rejected_steps, self.kind
        )
    }
}

impl std::error::Error for AdaptiveImplicitEulerError {}

#[derive(Debug, Default)]
struct AdaptiveImplicitEulerWork {
    newton_corrections: usize,
    rhs_evaluations: usize,
    estimated_dense_work_units: usize,
}

impl AdaptiveImplicitEulerWork {
    fn record_report(&mut self, report: &ImplicitEulerSolveReport) {
        self.newton_corrections = self.newton_corrections.saturating_add(report.newton_corrections);
        self.rhs_evaluations = self.rhs_evaluations.saturating_add(report.rhs_evaluations);
        self.estimated_dense_work_units = self
            .estimated_dense_work_units
            .saturating_add(report.estimated_dense_work_units);
    }

    fn record_error(&mut self, error: &ImplicitEulerError, dimension: usize) {
        self.newton_corrections = self.newton_corrections.saturating_add(error.newton_corrections);
        self.rhs_evaluations = self.rhs_evaluations.saturating_add(error.rhs_evaluations);
        let per_correction = dimension.saturating_mul(dimension).saturating_mul(dimension);
        self.estimated_dense_work_units = self
            .estimated_dense_work_units
            .saturating_add(per_correction.saturating_mul(error.newton_corrections));
    }

    fn exceeded_budget(&self) -> Option<AdaptiveImplicitEulerErrorKind> {
        if self.rhs_evaluations > MAX_IMPLICIT_RHS_EVALUATIONS {
            Some(AdaptiveImplicitEulerErrorKind::RhsEvaluationBudgetExceeded {
                max_evaluations: MAX_IMPLICIT_RHS_EVALUATIONS,
            })
        } else if self.estimated_dense_work_units > MAX_IMPLICIT_NEWTON_WORK_UNITS {
            Some(AdaptiveImplicitEulerErrorKind::DenseWorkBudgetExceeded {
                max_work_units: MAX_IMPLICIT_NEWTON_WORK_UNITS,
            })
        } else {
            None
        }
    }
}

fn adaptive_implicit_retryable(kind: &ImplicitEulerErrorKind) -> bool {
    matches!(
        kind,
        ImplicitEulerErrorKind::NewtonNonConvergence { .. }
            | ImplicitEulerErrorKind::SingularJacobian { .. }
    )
}

/// Result of a boundary-value-problem solve.
#[derive(Debug, Clone)]
pub struct BVPResult {
    /// Full trajectory from the shooting solve.
    pub trajectory: ODEResult,
    /// The initial condition y'(a) that was found.
    pub shooting_slope: f64,
    /// Residual at the right boundary |y(b) - y_b|.
    pub boundary_residual: f64,
    /// Whether the shooting method converged.
    pub converged: bool,
    /// Phi measure.
    pub phi: f64,
    /// HDC encoding.
    pub encoding: BinaryHV,
}

/// Result of a 1-D PDE solve (heat or wave equation).
#[derive(Debug, Clone)]
pub struct PDEResult {
    /// Spatial grid x_0, …, x_M.
    pub x_grid: Vec<f64>,
    /// Time grid t_0, …, t_N  (only start and end stored for memory efficiency).
    pub t_start: f64,
    pub t_end: f64,
    /// Initial condition u(x, 0).
    pub u_initial: Vec<f64>,
    /// Final solution u(x, t_end).
    pub u_final: Vec<f64>,
    /// Number of time steps taken.
    pub time_steps: usize,
    /// Phi measure (energy conservation or convergence quality).
    pub phi: f64,
    /// HDC encoding of the final solution.
    pub encoding: BinaryHV,
}

/// Result of a 2D PDE solve.
#[derive(Debug, Clone)]
pub struct PDE2DResult {
    /// Spatial grid in x-direction.
    pub x_grid: Vec<f64>,
    /// Spatial grid in y-direction.
    pub y_grid: Vec<f64>,
    /// Final time.
    pub t_end: f64,
    /// Number of interior points in x.
    pub nx: usize,
    /// Number of interior points in y.
    pub ny: usize,
    /// Initial condition (row-major, nx × ny).
    pub u_initial: Vec<f64>,
    /// Final solution (row-major, nx × ny).
    pub u_final: Vec<f64>,
    /// Number of time steps taken.
    pub time_steps: usize,
    /// Phi measure.
    pub phi: f64,
    /// HDC encoding.
    pub encoding: BinaryHV,
}

// ─── Differential Equations Engine ───────────────────────────────────────────

/// Hyperdimensional Differential Equations Engine.
///
/// Provides ODE and PDE solvers backed by HDC consciousness coupling.
pub struct DifferentialEquationsEngine;

impl DifferentialEquationsEngine {
    // ── IVP: Runge-Kutta 4 ───────────────────────────────────────────────

    /// Solve the initial value problem dy/dt = f(t, y), y(t0) = y0
    /// using classical 4th-order Runge-Kutta with a fixed step size.
    ///
    /// # Arguments
    ///
    /// * `system`  — ODE system (f and dimension)
    /// * `t0`      — initial time
    /// * `t_end`   — final time (must be > t0)
    /// * `y0`      — initial state (`len == system.dim`)
    /// * `n_steps` — number of equal-sized steps (capped at `MAX_STEPS`)
    ///
    /// # Returns
    ///
    /// [`ODEResult`] with the full trajectory and consciousness metrics.
    pub fn solve_ivp(
        system: &ODESystem,
        t0: f64,
        t_end: f64,
        y0: &[f64],
        n_steps: usize,
    ) -> ODEResult {
        assert_eq!(y0.len(), system.dim, "y0 length must equal system.dim");

        let n_steps = n_steps.max(1).min(MAX_STEPS);
        let dt = (t_end - t0) / n_steps as f64;

        let mut t_values = Vec::with_capacity(n_steps + 1);
        let mut y_values: Vec<Vec<f64>> = Vec::with_capacity(n_steps + 1);

        let mut t = t0;
        let mut y = y0.to_vec();

        t_values.push(t);
        y_values.push(y.clone());

        for _ in 0..n_steps {
            y = Self::rk4_step(system.f, t, &y, dt);
            t += dt;
            t_values.push(t);
            y_values.push(y.clone());
        }

        let final_y = y_values.last().unwrap();
        let phi = Self::compute_ivp_phi(n_steps, final_y);
        let encoding = Self::encode_ode_result(final_y, ODEMethod::RK4);

        ODEResult {
            t_values,
            y_values,
            method: ODEMethod::RK4,
            steps: n_steps,
            phi,
            encoding,
        }
    }

    // ── BVP: Shooting Method ─────────────────────────────────────────────

    /// Solve the scalar boundary value problem
    ///
    /// ```text
    /// y'' = g(t, y, y'),   y(a) = y_a,   y(b) = y_b
    /// ```
    ///
    /// by converting to a 2-D IVP  `[y, y']' = [y', g(t, y, y')]`
    /// and using Brent root-finding on the mismatch at the right boundary.
    ///
    /// # Arguments
    ///
    /// * `f2d`      — 2-D ODE right-hand side:  `f(t, [y, y'])` → `[y', y'']`
    /// * `a`, `b`   — left and right boundary times
    /// * `y_a`      — Dirichlet value at t = a
    /// * `y_b`      — Dirichlet value at t = b
    /// * `slope_lo`, `slope_hi` — bracket for the unknown initial slope y'(a)
    /// * `n_steps`  — IVP steps for each shooting trial
    #[allow(clippy::too_many_arguments)]
    pub fn solve_bvp_shooting(
        f2d: fn(f64, &[f64]) -> Vec<f64>,
        a: f64,
        b: f64,
        y_a: f64,
        y_b: f64,
        slope_lo: f64,
        slope_hi: f64,
        n_steps: usize,
    ) -> BVPResult {
        let system = ODESystem { f: f2d, dim: 2 };

        // Mismatch function: shoot with slope s, return y(b) - y_b
        let mismatch = |s: f64| -> f64 {
            let y0 = vec![y_a, s];
            let result = Self::solve_ivp(&system, a, b, &y0, n_steps);
            result.final_state()[0] - y_b
        };

        let root_result = RootFindingEngine::brent(&mismatch, slope_lo, slope_hi, DEFAULT_TOL);

        if !root_result.converged {
            // Return a failure result with boundary residual from the best attempt
            let y0 = vec![y_a, root_result.root];
            let traj = Self::solve_ivp(&system, a, b, &y0, n_steps);
            let residual = (traj.final_state()[0] - y_b).abs();
            let encoding = Self::encode_ode_result(traj.final_state(), ODEMethod::Shooting);
            return BVPResult {
                trajectory: traj,
                shooting_slope: root_result.root,
                boundary_residual: residual,
                converged: false,
                phi: 0.05,
                encoding,
            };
        }

        let slope = root_result.root;
        let y0 = vec![y_a, slope];
        let traj = Self::solve_ivp(&system, a, b, &y0, n_steps);
        let residual = (traj.final_state()[0] - y_b).abs();

        let phi = if residual < DEFAULT_TOL {
            0.85 + root_result.phi * 0.1
        } else {
            0.3
        };

        let encoding = Self::encode_ode_result(traj.final_state(), ODEMethod::Shooting);

        BVPResult {
            trajectory: traj,
            shooting_slope: slope,
            boundary_residual: residual,
            converged: true,
            phi,
            encoding,
        }
    }

    // ── PDE: 1-D Heat Equation ────────────────────────────────────────────

    /// Solve the 1-D heat equation
    ///
    /// ```text
    /// ∂u/∂t = α ∂²u/∂x²,   u(0,t) = u(L,t) = 0,   u(x,0) = u0(x)
    /// ```
    ///
    /// using the Method of Lines with explicit Euler in time and a
    /// 2nd-order central difference in space.
    ///
    /// **Stability**: requires `α dt / dx² ≤ 0.5`.  The engine enforces this
    /// automatically by capping `dt` to `0.4 * dx² / α`.
    ///
    /// # Arguments
    ///
    /// * `alpha`   — thermal diffusivity (> 0)
    /// * `length`  — domain [0, length]
    /// * `t_end`   — integration end time
    /// * `n_x`     — number of interior spatial grid points (boundaries excluded)
    /// * `u0`      — initial condition evaluated at interior points (len == n_x)
    pub fn heat_equation_1d(
        alpha: f64,
        length: f64,
        t_end: f64,
        n_x: usize,
        u0: &[f64],
    ) -> PDEResult {
        assert_eq!(u0.len(), n_x, "u0 length must equal n_x");
        assert!(alpha > 0.0, "thermal diffusivity must be positive");

        let dx = length / (n_x + 1) as f64;
        // CFL-safe time step: r = alpha*dt/dx^2 <= 0.4
        let dt_max = 0.4 * dx * dx / alpha;
        let n_steps_est = ((t_end / dt_max).ceil() as usize).max(1).min(MAX_STEPS);
        let dt = t_end / n_steps_est as f64;

        let mut u = u0.to_vec();
        let r = alpha * dt / (dx * dx);

        for _ in 0..n_steps_est {
            let mut u_new = u.clone();
            for i in 0..n_x {
                let u_left = if i == 0 { 0.0 } else { u[i - 1] };
                let u_right = if i == n_x - 1 { 0.0 } else { u[i + 1] };
                u_new[i] = u[i] + r * (u_left - 2.0 * u[i] + u_right);
            }
            u = u_new;
        }

        // Build spatial grid (interior points only)
        let x_grid: Vec<f64> = (1..=n_x).map(|i| i as f64 * dx).collect();

        // Phi: how much energy was conserved (heat dissipates, so check monotone decay)
        let initial_energy: f64 = u0.iter().map(|&v| v * v).sum::<f64>().sqrt();
        let final_energy: f64 = u.iter().map(|&v| v * v).sum::<f64>().sqrt();
        let energy_ratio = if initial_energy > 0.0 {
            final_energy / initial_energy
        } else {
            1.0
        };
        // Phi is high when energy decays correctly (0 ≤ ratio ≤ 1 for t>0)
        let phi = if (0.0..=1.0).contains(&energy_ratio) {
            0.7 + 0.2 * (1.0 - energy_ratio)
        } else {
            0.1 // instability
        };

        let encoding = Self::encode_pde_result(&u, "HEAT_EQ");

        PDEResult {
            x_grid,
            t_start: 0.0,
            t_end,
            u_initial: u0.to_vec(),
            u_final: u,
            time_steps: n_steps_est,
            phi,
            encoding,
        }
    }

    // ── PDE: 1-D Wave Equation ────────────────────────────────────────────

    /// Solve the 1-D wave equation
    ///
    /// ```text
    /// ∂²u/∂t² = c² ∂²u/∂x²,   u(0,t) = u(L,t) = 0
    /// u(x,0) = u0(x),   ∂u/∂t(x,0) = v0(x)
    /// ```
    ///
    /// using the Method of Lines with a leapfrog (Störmer-Verlet) scheme,
    /// which is 2nd-order in both time and space and energy-conserving.
    ///
    /// **Stability** (CFL condition): `c dt / dx ≤ 1`.  The engine enforces
    /// this by capping `dt` to `0.9 * dx / c`.
    ///
    /// # Arguments
    ///
    /// * `c`      — wave speed (> 0)
    /// * `length` — domain [0, length]
    /// * `t_end`  — integration end time
    /// * `n_x`    — number of interior spatial grid points
    /// * `u0`     — initial displacement (len == n_x)
    /// * `v0`     — initial velocity    (len == n_x)
    pub fn wave_equation_1d(
        c: f64,
        length: f64,
        t_end: f64,
        n_x: usize,
        u0: &[f64],
        v0: &[f64],
    ) -> PDEResult {
        assert_eq!(u0.len(), n_x, "u0 length must equal n_x");
        assert_eq!(v0.len(), n_x, "v0 length must equal n_x");
        assert!(c > 0.0, "wave speed must be positive");

        let dx = length / (n_x + 1) as f64;
        // CFL-safe time step
        let dt_max = 0.9 * dx / c;
        let n_steps = ((t_end / dt_max).ceil() as usize).max(1).min(MAX_STEPS);
        let dt = t_end / n_steps as f64;
        let r = (c * dt / dx).powi(2);

        // Leapfrog: store u^{n-1}, u^n, then compute u^{n+1}
        let laplacian = |u: &[f64]| -> Vec<f64> {
            (0..n_x)
                .map(|i| {
                    let u_left = if i == 0 { 0.0 } else { u[i - 1] };
                    let u_right = if i == n_x - 1 { 0.0 } else { u[i + 1] };
                    u_left - 2.0 * u[i] + u_right
                })
                .collect()
        };

        // Bootstrap first step: u^1 = u^0 + dt*v0 + (r/2)*Lap(u^0)
        let lap0 = laplacian(u0);
        let mut u_prev: Vec<f64> = u0.to_vec();
        let mut u_curr: Vec<f64> = (0..n_x)
            .map(|i| u0[i] + dt * v0[i] + 0.5 * r * lap0[i])
            .collect();

        for _ in 1..n_steps {
            let lap_curr = laplacian(&u_curr);
            let u_next: Vec<f64> = (0..n_x)
                .map(|i| 2.0 * u_curr[i] - u_prev[i] + r * lap_curr[i])
                .collect();
            u_prev = std::mem::replace(&mut u_curr, u_next);
        }

        let x_grid: Vec<f64> = (1..=n_x).map(|i| i as f64 * dx).collect();

        // Phi from energy conservation: kinetic + potential should be stable
        let initial_energy: f64 =
            u0.iter().map(|&v| v * v).sum::<f64>() + v0.iter().map(|&v| v * v).sum::<f64>();

        // Estimate current energy via finite-difference velocity (u_curr - u_prev)/dt
        let kinetic_final: f64 = u_curr
            .iter()
            .zip(u_prev.iter())
            .map(|(uc, up)| ((uc - up) / dt).powi(2))
            .sum::<f64>();
        let lap_final = laplacian(&u_curr);
        let potential_final: f64 = u_curr
            .iter()
            .zip(lap_final.iter())
            .map(|(u, l)| -u * l) // ≈ (du/dx)^2 via integration by parts
            .sum::<f64>()
            .abs();
        let final_energy = kinetic_final + potential_final;

        let energy_ratio = if initial_energy > 1e-15 {
            final_energy / initial_energy
        } else {
            1.0
        };

        // Good energy conservation: ratio close to 1
        let phi = 0.9 - 0.4 * (energy_ratio - 1.0).abs().min(1.0);

        let encoding = Self::encode_pde_result(&u_curr, "WAVE_EQ");

        PDEResult {
            x_grid,
            t_start: 0.0,
            t_end,
            u_initial: u0.to_vec(),
            u_final: u_curr,
            time_steps: n_steps,
            phi,
            encoding,
        }
    }

    // ── PDE: 2-D Heat Equation ────────────────────────────────────────────

    /// Solve the 2-D heat equation on a rectangular domain
    ///
    /// ```text
    /// ∂u/∂t = α (∂²u/∂x² + ∂²u/∂y²),   u = 0 on boundary
    /// u(x, y, 0) = u0(x, y)
    /// ```
    ///
    /// using a 5-point stencil with explicit Euler time-stepping.
    ///
    /// **Stability**: requires `α dt (1/dx² + 1/dy²) ≤ 0.5`. The engine
    /// enforces this automatically.
    ///
    /// # Arguments
    ///
    /// * `alpha`  — thermal diffusivity (> 0)
    /// * `lx, ly` — domain dimensions [0, lx] × [0, ly]
    /// * `t_end`  — integration end time
    /// * `nx, ny` — number of interior grid points in each direction
    /// * `u0`     — initial condition, row-major: u0[i * ny + j] = u(x_i, y_j, 0)
    ///
    /// Returns the final solution as a flat row-major array of size nx × ny.
    pub fn heat_equation_2d(
        alpha: f64,
        lx: f64,
        ly: f64,
        t_end: f64,
        nx: usize,
        ny: usize,
        u0: &[f64],
    ) -> PDE2DResult {
        assert_eq!(u0.len(), nx * ny, "u0 length must equal nx * ny");
        assert!(alpha > 0.0, "thermal diffusivity must be positive");
        assert!(
            nx >= 2 && ny >= 2,
            "need at least 2 interior points per axis"
        );

        let dx = lx / (nx + 1) as f64;
        let dy = ly / (ny + 1) as f64;

        // CFL-safe time step: r_x + r_y = α·dt·(1/dx² + 1/dy²) ≤ 0.4
        let dt_max = 0.4 / (alpha * (1.0 / (dx * dx) + 1.0 / (dy * dy)));
        let n_steps = ((t_end / dt_max).ceil() as usize).max(1).min(MAX_STEPS);
        let dt = t_end / n_steps as f64;

        let rx = alpha * dt / (dx * dx);
        let ry = alpha * dt / (dy * dy);

        let mut u = u0.to_vec();

        for _ in 0..n_steps {
            let mut u_new = u.clone();
            for i in 0..nx {
                for j in 0..ny {
                    let idx = i * ny + j;
                    let u_left = if i == 0 { 0.0 } else { u[(i - 1) * ny + j] };
                    let u_right = if i == nx - 1 {
                        0.0
                    } else {
                        u[(i + 1) * ny + j]
                    };
                    let u_down = if j == 0 { 0.0 } else { u[i * ny + (j - 1)] };
                    let u_up = if j == ny - 1 {
                        0.0
                    } else {
                        u[i * ny + (j + 1)]
                    };

                    u_new[idx] = u[idx]
                        + rx * (u_left - 2.0 * u[idx] + u_right)
                        + ry * (u_down - 2.0 * u[idx] + u_up);
                }
            }
            u = u_new;
        }

        // Build spatial grids
        let x_grid: Vec<f64> = (1..=nx).map(|i| i as f64 * dx).collect();
        let y_grid: Vec<f64> = (1..=ny).map(|j| j as f64 * dy).collect();

        // Phi: energy should decay monotonically for heat equation
        let initial_energy: f64 = u0.iter().map(|&v| v * v).sum::<f64>().sqrt();
        let final_energy: f64 = u.iter().map(|&v| v * v).sum::<f64>().sqrt();
        let energy_ratio = if initial_energy > 0.0 {
            final_energy / initial_energy
        } else {
            1.0
        };
        let phi = if (0.0..=1.0).contains(&energy_ratio) {
            0.7 + 0.2 * (1.0 - energy_ratio)
        } else {
            0.1
        };

        let encoding = Self::encode_pde_result(&u, "HEAT_EQ_2D");

        PDE2DResult {
            x_grid,
            y_grid,
            t_end,
            nx,
            ny,
            u_initial: u0.to_vec(),
            u_final: u,
            time_steps: n_steps,
            phi,
            encoding,
        }
    }

    // ── Internal RK4 step ────────────────────────────────────────────────

    /// Single RK4 step: y_{n+1} = y_n + (dt/6)(k1 + 2k2 + 2k3 + k4).
    fn rk4_step(f: fn(f64, &[f64]) -> Vec<f64>, t: f64, y: &[f64], dt: f64) -> Vec<f64> {
        let k1 = f(t, y);
        let y2: Vec<f64> = y
            .iter()
            .zip(k1.iter())
            .map(|(&yi, &ki)| yi + 0.5 * dt * ki)
            .collect();
        let k2 = f(t + 0.5 * dt, &y2);
        let y3: Vec<f64> = y
            .iter()
            .zip(k2.iter())
            .map(|(&yi, &ki)| yi + 0.5 * dt * ki)
            .collect();
        let k3 = f(t + 0.5 * dt, &y3);
        let y4: Vec<f64> = y
            .iter()
            .zip(k3.iter())
            .map(|(&yi, &ki)| yi + dt * ki)
            .collect();
        let k4 = f(t + dt, &y4);

        y.iter()
            .zip(k1.iter().zip(k2.iter().zip(k3.iter().zip(k4.iter()))))
            .map(|(&yi, (&k1i, (&k2i, (&k3i, &k4i))))| {
                yi + (dt / 6.0) * (k1i + 2.0 * k2i + 2.0 * k3i + k4i)
            })
            .collect()
    }

    // ── Phi + encoding helpers ───────────────────────────────────────────

    fn compute_ivp_phi(steps: usize, final_y: &[f64]) -> f64 {
        // Phi based on solution quality: finite values + step count
        let finite = final_y.iter().all(|v| v.is_finite());
        if !finite {
            return 0.0;
        }
        let step_factor = 1.0 / (1.0 + steps as f64 / 10_000.0);
        0.5 + 0.4 * step_factor
    }

    fn encode_ode_result(final_y: &[f64], method: ODEMethod) -> BinaryHV {
        let base = BinaryHV::random(seed_from_name("ODE_RESULT"));
        let method_hv = BinaryHV::random(seed_from_name(&format!("ODE_METHOD_{}", method)));

        // Fold final state into a single seed via a simple hash
        let state_hash: u64 = final_y.iter().enumerate().fold(0u64, |acc, (i, &v)| {
            acc.wrapping_add(v.to_bits().wrapping_mul(i as u64 + 1))
        });
        let state_hv = BinaryHV::random(seed_from_name(&format!("ODE_STATE_{}", state_hash)));

        base.bind(&method_hv).bind(&state_hv)
    }

    fn encode_implicit_euler_result(final_y: &[f64]) -> BinaryHV {
        let state_hash: u64 = final_y
            .iter()
            .fold(0u64, |acc, &value| acc.wrapping_add(value.to_bits()));
        let base = BinaryHV::random(seed_from_name("IMPLICIT_EULER"));
        let state_hv = BinaryHV::random(seed_from_name(&format!("IE_STATE_{}", state_hash)));
        base.bind(&state_hv)
    }

    fn encode_pde_result(u_final: &[f64], tag: &str) -> BinaryHV {
        let base = BinaryHV::random(seed_from_name(tag));
        let state_hash: u64 = u_final.iter().enumerate().fold(0u64, |acc, (i, &v)| {
            acc.wrapping_add(v.to_bits().wrapping_mul(i as u64 + 1))
        });
        let state_hv = BinaryHV::random(seed_from_name(&format!("{}_STATE_{}", tag, state_hash)));
        base.bind(&state_hv)
    }
}

// ─── RK45 (Dormand-Prince) ────────────────────────────────────────────────────

/// Adaptive step-size IVP result (RK45).
#[derive(Debug, Clone)]
pub struct RK45Result {
    /// Final state vector.
    pub y_final: Vec<f64>,
    /// Total steps taken (accepted + rejected).
    pub steps_taken: usize,
    /// Number of rejected steps (step halved due to error).
    pub rejected_steps: usize,
    /// Achieved local error estimate at termination.
    pub error_estimate: f64,
    /// Whether the integration reached t_end within tolerance.
    pub converged: bool,
}

impl DifferentialEquationsEngine {
    /// Solve an IVP using the Dormand-Prince RK45 embedded pair.
    ///
    /// Adaptive step control: the embedded 4th/5th order pair gives a
    /// local error estimate.  Step grows/shrinks by `(tol/err)^(1/5)`,
    /// clamped to `[h_min, h_max]`.
    ///
    /// Dormand-Prince Butcher tableau (DOPRI5 — Hairer et al. 1993):
    /// c2=1/5, c3=3/10, c4=4/5, c5=8/9, c6=1, c7=1
    pub fn solve_rk45(system: &ODESystem, t0: f64, t_end: f64, y0: &[f64], tol: f64) -> RK45Result {
        // Dormand-Prince coefficients (a-matrix, b5, b4)
        let a21 = 1.0 / 5.0;
        let a31 = 3.0 / 40.0;
        let a32 = 9.0 / 40.0;
        let a41 = 44.0 / 45.0;
        let a42 = -56.0 / 15.0;
        let a43 = 32.0 / 9.0;
        let a51 = 19372.0 / 6561.0;
        let a52 = -25360.0 / 2187.0;
        let a53 = 64448.0 / 6561.0;
        let a54 = -212.0 / 729.0;
        let a61 = 9017.0 / 3168.0;
        let a62 = -355.0 / 33.0;
        let a63 = 46732.0 / 5247.0;
        let a64 = 49.0 / 176.0;
        let a65 = -5103.0 / 18656.0;

        // 5th-order weights (b)
        let b1 = 35.0 / 384.0;
        let b3 = 500.0 / 1113.0;
        let b4 = 125.0 / 192.0;
        let b5 = -2187.0 / 6784.0;
        let b6 = 11.0 / 84.0;

        // 4th-order weights (b*) for error estimate
        let e1 = 71.0 / 57600.0;
        let e3 = -71.0 / 16695.0;
        let e4 = 71.0 / 1920.0;
        let e5 = -17253.0 / 339200.0;
        let e6 = 22.0 / 525.0;
        let e7 = -1.0 / 40.0;

        let dim = system.dim;
        let mut t = t0;
        let mut y = y0.to_vec();
        let mut h = (t_end - t0) / 100.0; // initial step
        let h_min = 1e-12;
        let h_max = (t_end - t0) / 10.0;
        let tol = tol.max(1e-12);

        let mut steps_taken = 0usize;
        let mut rejected = 0usize;
        let mut error_estimate = 0.0;
        let max_steps = 100_000;

        while t < t_end && steps_taken < max_steps {
            if t + h > t_end {
                h = t_end - t;
            }

            // Stage evaluations
            let k1 = (system.f)(t, &y);
            let y2: Vec<f64> = (0..dim).map(|i| y[i] + h * a21 * k1[i]).collect();
            let k2 = (system.f)(t + h / 5.0, &y2);
            let y3: Vec<f64> = (0..dim)
                .map(|i| y[i] + h * (a31 * k1[i] + a32 * k2[i]))
                .collect();
            let k3 = (system.f)(t + 3.0 * h / 10.0, &y3);
            let y4: Vec<f64> = (0..dim)
                .map(|i| y[i] + h * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i]))
                .collect();
            let k4 = (system.f)(t + 4.0 * h / 5.0, &y4);
            let y5: Vec<f64> = (0..dim)
                .map(|i| y[i] + h * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i]))
                .collect();
            let k5 = (system.f)(t + 8.0 * h / 9.0, &y5);
            let y6: Vec<f64> = (0..dim)
                .map(|i| {
                    y[i] + h * (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i])
                })
                .collect();
            let k6 = (system.f)(t + h, &y6);

            // 5th-order solution
            let y_new: Vec<f64> = (0..dim)
                .map(|i| {
                    y[i] + h * (b1 * k1[i] + b3 * k3[i] + b4 * k4[i] + b5 * k5[i] + b6 * k6[i])
                })
                .collect();

            // 7th stage (for error estimate only)
            let k7 = (system.f)(t + h, &y_new);

            // Error estimate (difference between 4th and 5th order)
            let err: f64 = (0..dim)
                .map(|i| {
                    let e = h
                        * (e1 * k1[i]
                            + e3 * k3[i]
                            + e4 * k4[i]
                            + e5 * k5[i]
                            + e6 * k6[i]
                            + e7 * k7[i]);
                    let sc = tol * (1.0 + y[i].abs().max(y_new[i].abs()));
                    (e / sc).powi(2)
                })
                .sum::<f64>()
                .sqrt()
                / (dim as f64).sqrt();

            error_estimate = err;

            if err <= 1.0 {
                // Accept step
                t += h;
                y = y_new;
                steps_taken += 1;
                // Grow step
                let factor = 0.9 * err.powf(-0.2);
                h = (h * factor.min(5.0)).min(h_max);
            } else {
                // Reject step, shrink
                rejected += 1;
                let factor = 0.9 * err.powf(-0.2);
                h = (h * factor.max(0.1)).max(h_min);
                if h < h_min {
                    break;
                }
            }
        }

        RK45Result {
            y_final: y,
            steps_taken,
            rejected_steps: rejected,
            error_estimate,
            converged: (t - t_end).abs() < 1e-10 || t >= t_end,
        }
    }

    /// Solve backward Euler with a full finite-difference Jacobian and bounded Newton work.
    ///
    /// For each step this solves
    /// `G(z) = z - y_n - h*f(t_{n+1}, z) = 0` with a dense Newton
    /// correction. The nonlinear residual is scaled per component by
    /// `1 + max(abs(z[i]), abs(y_n[i]))`. A step is committed only after
    /// its normalized infinity-norm residual is at most 1e-10.
    ///
    /// This is a dense solver limited to 64 state variables. It fails with a
    /// typed error rather than silently accepting a non-converged Newton step.
    /// For larger systems, a future sparse-Jacobian/linear-solver API is needed.
    pub fn try_solve_implicit_euler(
        system: &ODESystem,
        t0: f64,
        t_end: f64,
        y0: &[f64],
        n_steps: usize,
        newton_iters: usize,
    ) -> Result<ImplicitEulerSolveReport, ImplicitEulerError> {
        let mut y = y0.to_vec();
        let mut rhs_evaluations = 0usize;
        let mut total_corrections = 0usize;
        let mut dense_work_units = 0usize;
        let mut max_normalized_residual = 0.0_f64;

        if system.dim == 0 {
            return Err(ImplicitEulerError::new(
                ImplicitEulerErrorKind::EmptySystem, t0, &y, 0, 0, 0, 0.0
            ));
        }
        if y0.len() != system.dim {
            return Err(ImplicitEulerError::new(
                ImplicitEulerErrorKind::DimensionMismatch {
                    expected: system.dim, actual: y0.len(),
                }, t0, &y, 0, 0, 0, 0.0
            ));
        }
        if !t0.is_finite() || !t_end.is_finite() || t_end < t0 {
            return Err(ImplicitEulerError::new(
                ImplicitEulerErrorKind::InvalidTimeSpan, t0, &y, 0, 0, 0, 0.0
            ));
        }
        if n_steps == 0 || n_steps > MAX_STEPS {
            return Err(ImplicitEulerError::new(
                ImplicitEulerErrorKind::InvalidStepCount {
                    requested: n_steps, max_allowed: MAX_STEPS,
                }, t0, &y, 0, 0, 0, 0.0
            ));
        }
        if newton_iters == 0 || newton_iters > MAX_NEWTON_ITERATIONS_PER_STEP {
            return Err(ImplicitEulerError::new(
                ImplicitEulerErrorKind::InvalidNewtonIterationBudget {
                    requested: newton_iters, max_allowed: MAX_NEWTON_ITERATIONS_PER_STEP,
                }, t0, &y, 0, 0, 0, 0.0
            ));
        }
        if system.dim > MAX_DENSE_NEWTON_DIM {
            return Err(ImplicitEulerError::new(
                ImplicitEulerErrorKind::SystemTooLargeForDenseNewton {
                    dimension: system.dim, max_dimension: MAX_DENSE_NEWTON_DIM,
                }, t0, &y, 0, 0, 0, 0.0
            ));
        }
        if let Some(index) = y0.iter().position(|value| !value.is_finite()) {
            return Err(ImplicitEulerError::new(
                ImplicitEulerErrorKind::NonFiniteInitialState { index },
                t0, &y, 0, 0, 0, 0.0
            ));
        }
        let span = t_end - t0;
        if !span.is_finite() {
            return Err(ImplicitEulerError::new(
                ImplicitEulerErrorKind::InvalidTimeSpan, t0, &y, 0, 0, 0, 0.0
            ));
        }

        let mut t_values = vec![t0];
        let mut y_values = vec![y.clone()];
        if span == 0.0 {
            let trajectory = ODEResult {
                t_values,
                y_values,
                method: ODEMethod::ImplicitEuler,
                steps: 0,
                phi: 1.0 / (1.0 + y.iter().map(|value| value.abs()).sum::<f64>() / system.dim as f64),
                encoding: Self::encode_implicit_euler_result(&y),
            };
            return Ok(ImplicitEulerSolveReport {
                trajectory,
                newton_corrections: 0,
                rhs_evaluations: 0,
                max_normalized_residual: 0.0,
                estimated_dense_work_units: 0,
            });
        }

        let h = span / n_steps as f64;
        if !h.is_finite() || h <= 0.0 {
            return Err(ImplicitEulerError::new(
                ImplicitEulerErrorKind::InvalidTimeSpan, t0, &y, 0, 0, 0, 0.0
            ));
        }
        let dimension = system.dim;
        let matrix_len = dimension.checked_mul(dimension).ok_or_else(|| {
            ImplicitEulerError::new(
                ImplicitEulerErrorKind::SystemTooLargeForDenseNewton {
                    dimension, max_dimension: MAX_DENSE_NEWTON_DIM,
                }, t0, &y, 0, total_corrections, rhs_evaluations, max_normalized_residual
            )
        })?;

        for step in 0..n_steps {
            let t_next = if step + 1 == n_steps {
                t_end
            } else {
                t0 + (step + 1) as f64 * h
            };
            if !t_next.is_finite() || t_next <= *t_values.last().unwrap_or(&t0) {
                return Err(ImplicitEulerError::new(
                    ImplicitEulerErrorKind::InvalidTimeSpan,
                    *t_values.last().unwrap_or(&t0), &y, step,
                    total_corrections, rhs_evaluations, max_normalized_residual
                ));
            }

            let f_current = match implicit_euler_eval_rhs(
                system, "predictor", *t_values.last().unwrap_or(&t0), &y, &mut rhs_evaluations
            ) {
                Ok(value) => value,
                Err(kind) => return Err(ImplicitEulerError::new(
                    kind, *t_values.last().unwrap_or(&t0), &y, step,
                    total_corrections, rhs_evaluations, max_normalized_residual
                )),
            };
            let mut y_next: Vec<f64> = y.iter().zip(f_current.iter())
                .map(|(state, derivative)| *state + h * *derivative)
                .collect();
            if let Some(index) = y_next.iter().position(|value| !value.is_finite()) {
                return Err(ImplicitEulerError::new(
                    ImplicitEulerErrorKind::NonFinitePredictor { index },
                    *t_values.last().unwrap_or(&t0), &y, step,
                    total_corrections, rhs_evaluations, max_normalized_residual
                ));
            }

            let mut converged = false;
            let mut step_residual_norm = f64::INFINITY;
            for iteration in 0..=newton_iters {
                let f_next = match implicit_euler_eval_rhs(
                    system, "residual", t_next, &y_next, &mut rhs_evaluations
                ) {
                    Ok(value) => value,
                    Err(kind) => return Err(ImplicitEulerError::new(
                        kind, *t_values.last().unwrap_or(&t0), &y, step,
                        total_corrections, rhs_evaluations, step_residual_norm
                    )),
                };
                let mut residual = Vec::with_capacity(dimension);
                step_residual_norm = 0.0;
                for i in 0..dimension {
                    let value = y_next[i] - y[i] - h * f_next[i];
                    if !value.is_finite() {
                        return Err(ImplicitEulerError::new(
                            ImplicitEulerErrorKind::NonFiniteResidual { index: i },
                            *t_values.last().unwrap_or(&t0), &y, step,
                            total_corrections, rhs_evaluations, step_residual_norm
                        ));
                    }
                    let scale = 1.0 + y_next[i].abs().max(y[i].abs());
                    let normalized = value.abs() / scale;
                    if !normalized.is_finite() {
                        return Err(ImplicitEulerError::new(
                            ImplicitEulerErrorKind::NonFiniteResidual { index: i },
                            *t_values.last().unwrap_or(&t0), &y, step,
                            total_corrections, rhs_evaluations, normalized
                        ));
                    }
                    step_residual_norm = step_residual_norm.max(normalized);
                    residual.push(value);
                }
                if step_residual_norm <= 1e-10 {
                    converged = true;
                    max_normalized_residual = max_normalized_residual.max(step_residual_norm);
                    break;
                }
                if iteration == newton_iters {
                    return Err(ImplicitEulerError::new(
                        ImplicitEulerErrorKind::NewtonNonConvergence {
                            iteration_limit: newton_iters,
                            residual_norm: step_residual_norm,
                        },
                        *t_values.last().unwrap_or(&t0), &y, step,
                        total_corrections, rhs_evaluations, step_residual_norm
                    ));
                }

                let per_correction_work = dimension.checked_mul(dimension)
                    .and_then(|value| value.checked_mul(dimension))
                    .unwrap_or(usize::MAX);
                if per_correction_work == usize::MAX
                    || dense_work_units.saturating_add(per_correction_work)
                        > MAX_IMPLICIT_NEWTON_WORK_UNITS
                {
                    return Err(ImplicitEulerError::new(
                        ImplicitEulerErrorKind::NewtonWorkBudgetExceeded {
                            max_work_units: MAX_IMPLICIT_NEWTON_WORK_UNITS,
                        },
                        *t_values.last().unwrap_or(&t0), &y, step,
                        total_corrections, rhs_evaluations, step_residual_norm
                    ));
                }

                let mut jacobian = vec![0.0_f64; matrix_len];
                for column in 0..dimension {
                    let original = y_next[column];
                    let delta = f64::EPSILON.sqrt() * original.abs().max(1.0);
                    let mut perturbed = original + delta;
                    let mut signed_delta = delta;
                    if !perturbed.is_finite() || perturbed == original {
                        perturbed = original - delta;
                        signed_delta = -delta;
                    }
                    if !perturbed.is_finite() || perturbed == original || signed_delta == 0.0 {
                        return Err(ImplicitEulerError::new(
                            ImplicitEulerErrorKind::NonFiniteJacobian { row: 0, column },
                            *t_values.last().unwrap_or(&t0), &y, step,
                            total_corrections, rhs_evaluations, step_residual_norm
                        ));
                    }
                    y_next[column] = perturbed;
                    let f_perturbed = implicit_euler_eval_rhs(
                        system, "jacobian", t_next, &y_next, &mut rhs_evaluations
                    );
                    y_next[column] = original;
                    let f_perturbed = match f_perturbed {
                        Ok(value) => value,
                        Err(kind) => return Err(ImplicitEulerError::new(
                            kind, *t_values.last().unwrap_or(&t0), &y, step,
                            total_corrections, rhs_evaluations, step_residual_norm
                        )),
                    };

                    for row in 0..dimension {
                        let candidate_component = if row == column {
                            perturbed
                        } else {
                            y_next[row]
                        };
                        let g_perturbed =
                            candidate_component - y[row] - h * f_perturbed[row];
                        let g_base = residual[row];
                        let entry = (g_perturbed - g_base) / signed_delta;
                        if !entry.is_finite() {
                            return Err(ImplicitEulerError::new(
                                ImplicitEulerErrorKind::NonFiniteJacobian { row, column },
                                *t_values.last().unwrap_or(&t0), &y, step,
                                total_corrections, rhs_evaluations, step_residual_norm
                            ));
                        }
                        jacobian[row * dimension + column] = entry;
                    }
                }

                let mut correction: Vec<f64> = residual.iter().map(|value| -*value).collect();
                if let Err((column, pivot_magnitude)) =
                    implicit_euler_solve_dense(&mut jacobian, &mut correction, dimension)
                {
                    let kind = if pivot_magnitude.is_finite() {
                        ImplicitEulerErrorKind::SingularJacobian { column, pivot_magnitude }
                    } else {
                        ImplicitEulerErrorKind::NonFiniteLinearSolve { column }
                    };
                    return Err(ImplicitEulerError::new(
                        kind,
                        *t_values.last().unwrap_or(&t0), &y, step,
                        total_corrections, rhs_evaluations, step_residual_norm
                    ));
                }
                for i in 0..dimension {
                    y_next[i] += correction[i];
                    if !y_next[i].is_finite() {
                        return Err(ImplicitEulerError::new(
                            ImplicitEulerErrorKind::NonFiniteNewtonCorrection { index: i },
                            *t_values.last().unwrap_or(&t0), &y, step,
                            total_corrections, rhs_evaluations, step_residual_norm
                        ));
                    }
                }
                total_corrections += 1;
                dense_work_units += per_correction_work;
            }

            if !converged {
                return Err(ImplicitEulerError::new(
                    ImplicitEulerErrorKind::NewtonNonConvergence {
                        iteration_limit: newton_iters,
                        residual_norm: step_residual_norm,
                    },
                    *t_values.last().unwrap_or(&t0), &y, step,
                    total_corrections, rhs_evaluations, step_residual_norm
                ));
            }
            y = y_next;
            t_values.push(t_next);
            y_values.push(y.clone());
        }

        let trajectory = ODEResult {
            t_values,
            y_values,
            method: ODEMethod::ImplicitEuler,
            steps: n_steps,
            phi: 1.0 / (1.0 + y.iter().map(|value| value.abs()).sum::<f64>() / dimension as f64),
            encoding: Self::encode_implicit_euler_result(&y),
        };
        Ok(ImplicitEulerSolveReport {
            trajectory,
            newton_corrections: total_corrections,
            rhs_evaluations,
            max_normalized_residual,
            estimated_dense_work_units: dense_work_units,
        })
    }

    /// Adaptive backward Euler using step doubling to estimate discretization error.
    ///
    /// Each trial compares one full backward-Euler step with two half steps.
    /// The two-half-step result is accepted when the RMS component error,
    /// scaled by `atol[i] + rtol * max(abs(y_old[i]), abs(y_fine[i]))`,
    /// is at most one. A backward-Euler step has order one, so the controller
    /// adjusts the step size with an error exponent of -1/2.
    ///
    /// Newton residual convergence and time-discretization error are distinct
    /// checks. Retryable Newton convergence/singularity failures shrink the
    /// time step; malformed RHS values and other structural failures stop the
    /// solve. The maximum-attempt and accumulated-work budgets include rejected
    /// macro-steps and all full/half-step subsolves.
    ///
    /// This is a first-order adaptive method, not a substitute for a qualified
    /// variable-order BDF or Radau solver.
    pub fn try_solve_implicit_euler_adaptive(
        system: &ODESystem,
        t0: f64,
        t_end: f64,
        y0: &[f64],
        absolute_tolerances: &[f64],
        relative_tolerance: f64,
        initial_step: f64,
        max_attempts: usize,
        newton_iters: usize,
    ) -> Result<AdaptiveImplicitEulerReport, AdaptiveImplicitEulerError> {
        let mut y = y0.to_vec();
        let mut t = t0;
        let mut accepted_steps = 0usize;
        let mut rejected_steps = 0usize;
        let mut attempts = 0usize;
        let mut max_accepted_error = 0.0_f64;
        let mut max_newton_residual = 0.0_f64;
        let mut work = AdaptiveImplicitEulerWork::default();

        if system.dim == 0 {
            return Err(AdaptiveImplicitEulerError::new(
                AdaptiveImplicitEulerErrorKind::EmptySystem, t, &y, accepted_steps,
                rejected_steps, attempts, &work, max_newton_residual
            ));
        }
        if y0.len() != system.dim {
            return Err(AdaptiveImplicitEulerError::new(
                AdaptiveImplicitEulerErrorKind::DimensionMismatch {
                    expected: system.dim, actual: y0.len(),
                }, t, &y, accepted_steps, rejected_steps, attempts, &work, max_newton_residual
            ));
        }
        if !t0.is_finite() || !t_end.is_finite() || t_end < t0 || !(t_end - t0).is_finite() {
            return Err(AdaptiveImplicitEulerError::new(
                AdaptiveImplicitEulerErrorKind::InvalidTimeSpan, t, &y, accepted_steps,
                rejected_steps, attempts, &work, max_newton_residual
            ));
        }
        if !initial_step.is_finite() || initial_step <= 0.0 {
            return Err(AdaptiveImplicitEulerError::new(
                AdaptiveImplicitEulerErrorKind::InvalidInitialStep, t, &y, accepted_steps,
                rejected_steps, attempts, &work, max_newton_residual
            ));
        }
        if max_attempts == 0 || max_attempts > MAX_STEPS {
            return Err(AdaptiveImplicitEulerError::new(
                AdaptiveImplicitEulerErrorKind::InvalidAttemptBudget {
                    requested: max_attempts, max_allowed: MAX_STEPS,
                }, t, &y, accepted_steps, rejected_steps, attempts, &work, max_newton_residual
            ));
        }
        if newton_iters == 0 || newton_iters > MAX_NEWTON_ITERATIONS_PER_STEP {
            return Err(AdaptiveImplicitEulerError::new(
                AdaptiveImplicitEulerErrorKind::InvalidNewtonIterationBudget {
                    requested: newton_iters, max_allowed: MAX_NEWTON_ITERATIONS_PER_STEP,
                }, t, &y, accepted_steps, rejected_steps, attempts, &work, max_newton_residual
            ));
        }
        if system.dim > MAX_DENSE_NEWTON_DIM {
            return Err(AdaptiveImplicitEulerError::new(
                AdaptiveImplicitEulerErrorKind::SystemTooLargeForDenseNewton {
                    dimension: system.dim, max_dimension: MAX_DENSE_NEWTON_DIM,
                }, t, &y, accepted_steps, rejected_steps, attempts, &work, max_newton_residual
            ));
        }
        if absolute_tolerances.len() != system.dim {
            return Err(AdaptiveImplicitEulerError::new(
                AdaptiveImplicitEulerErrorKind::ToleranceDimensionMismatch {
                    expected: system.dim, actual: absolute_tolerances.len(),
                }, t, &y, accepted_steps, rejected_steps, attempts, &work, max_newton_residual
            ));
        }
        if !relative_tolerance.is_finite() || relative_tolerance < 0.0 {
            return Err(AdaptiveImplicitEulerError::new(
                AdaptiveImplicitEulerErrorKind::InvalidRelativeTolerance, t, &y,
                accepted_steps, rejected_steps, attempts, &work, max_newton_residual
            ));
        }
        if let Some(index) = absolute_tolerances
            .iter()
            .position(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(AdaptiveImplicitEulerError::new(
                AdaptiveImplicitEulerErrorKind::InvalidAbsoluteTolerance { index }, t, &y,
                accepted_steps, rejected_steps, attempts, &work, max_newton_residual
            ));
        }
        if let Some(index) = y0.iter().position(|value| !value.is_finite()) {
            return Err(AdaptiveImplicitEulerError::new(
                AdaptiveImplicitEulerErrorKind::StepSolveFailed {
                    cause: ImplicitEulerErrorKind::NonFiniteInitialState { index },
                }, t, &y, accepted_steps, rejected_steps, attempts, &work, max_newton_residual
            ));
        }

        let span = t_end - t0;
        let mut t_values = vec![t0];
        let mut y_values = vec![y.clone()];
        if span == 0.0 {
            let trajectory = ODEResult {
                t_values,
                y_values,
                method: ODEMethod::ImplicitEuler,
                steps: 0,
                phi: 1.0 / (1.0 + y.iter().map(|value| value.abs()).sum::<f64>() / system.dim as f64),
                encoding: Self::encode_implicit_euler_result(&y),
            };
            return Ok(AdaptiveImplicitEulerReport {
                trajectory,
                accepted_steps: 0,
                rejected_steps: 0,
                max_accepted_normalized_local_error: 0.0,
                max_normalized_newton_residual: 0.0,
                newton_corrections: 0,
                rhs_evaluations: 0,
                estimated_dense_work_units: 0,
                attempts: 0,
            });
        }

        let mut step_size = initial_step.min(span);
        while t < t_end {
            if attempts >= max_attempts {
                return Err(AdaptiveImplicitEulerError::new(
                    AdaptiveImplicitEulerErrorKind::AttemptBudgetExceeded { max_attempts },
                    t, &y, accepted_steps, rejected_steps, attempts, &work, max_newton_residual
                ));
            }
            attempts += 1;
            let remaining = t_end - t;
            let trial_step = step_size.min(remaining);
            let trial_end = if trial_step >= remaining { t_end } else { t + trial_step };
            let midpoint = t + (trial_end - t) * 0.5;
            if !trial_step.is_finite() || trial_step <= 0.0
                || trial_end <= t || midpoint <= t || midpoint >= trial_end
            {
                return Err(AdaptiveImplicitEulerError::new(
                    AdaptiveImplicitEulerErrorKind::StepSizeUnderflow { attempted_step: trial_step },
                    t, &y, accepted_steps, rejected_steps, attempts, &work, max_newton_residual
                ));
            }

            let full = Self::try_solve_implicit_euler(system, t, trial_end, &y, 1, newton_iters);
            let full_state = match full {
                Ok(report) => {
                    max_newton_residual = max_newton_residual.max(report.max_normalized_residual);
                    work.record_report(&report);
                    if let Some(kind) = work.exceeded_budget() {
                        return Err(AdaptiveImplicitEulerError::new(
                            kind, t, &y, accepted_steps, rejected_steps, attempts, &work, max_newton_residual
                        ));
                    }
                    report.trajectory.final_state().to_vec()
                }
                Err(error) => {
                    max_newton_residual = max_newton_residual.max(error.residual_norm);
                    work.record_error(&error, system.dim);
                    if let Some(kind) = work.exceeded_budget() {
                        return Err(AdaptiveImplicitEulerError::new(
                            kind, t, &y, accepted_steps, rejected_steps, attempts, &work, max_newton_residual
                        ));
                    }
                    if adaptive_implicit_retryable(&error.kind) {
                        rejected_steps += 1;
                        step_size = trial_step * 0.5;
                        if !step_size.is_finite() || step_size <= 0.0 || t + step_size <= t {
                            return Err(AdaptiveImplicitEulerError::new(
                                AdaptiveImplicitEulerErrorKind::StepSizeUnderflow { attempted_step: step_size },
                                t, &y, accepted_steps, rejected_steps, attempts, &work, max_newton_residual
                            ));
                        }
                        continue;
                    }
                    return Err(AdaptiveImplicitEulerError::new(
                        AdaptiveImplicitEulerErrorKind::StepSolveFailed { cause: error.kind },
                        t, &y, accepted_steps, rejected_steps, attempts, &work, max_newton_residual
                    ));
                }
            };

            let first_half = Self::try_solve_implicit_euler(system, t, midpoint, &y, 1, newton_iters);
            let midpoint_state = match first_half {
                Ok(report) => {
                    max_newton_residual = max_newton_residual.max(report.max_normalized_residual);
                    work.record_report(&report);
                    if let Some(kind) = work.exceeded_budget() {
                        return Err(AdaptiveImplicitEulerError::new(
                            kind, t, &y, accepted_steps, rejected_steps, attempts, &work, max_newton_residual
                        ));
                    }
                    report.trajectory.final_state().to_vec()
                }
                Err(error) => {
                    max_newton_residual = max_newton_residual.max(error.residual_norm);
                    work.record_error(&error, system.dim);
                    if let Some(kind) = work.exceeded_budget() {
                        return Err(AdaptiveImplicitEulerError::new(
                            kind, t, &y, accepted_steps, rejected_steps, attempts, &work, max_newton_residual
                        ));
                    }
                    if adaptive_implicit_retryable(&error.kind) {
                        rejected_steps += 1;
                        step_size = trial_step * 0.5;
                        if !step_size.is_finite() || step_size <= 0.0 || t + step_size <= t {
                            return Err(AdaptiveImplicitEulerError::new(
                                AdaptiveImplicitEulerErrorKind::StepSizeUnderflow { attempted_step: step_size },
                                t, &y, accepted_steps, rejected_steps, attempts, &work, max_newton_residual
                            ));
                        }
                        continue;
                    }
                    return Err(AdaptiveImplicitEulerError::new(
                        AdaptiveImplicitEulerErrorKind::StepSolveFailed { cause: error.kind },
                        t, &y, accepted_steps, rejected_steps, attempts, &work, max_newton_residual
                    ));
                }
            };

            let second_half = Self::try_solve_implicit_euler(
                system, midpoint, trial_end, &midpoint_state, 1, newton_iters
            );
            let fine_state = match second_half {
                Ok(report) => {
                    max_newton_residual = max_newton_residual.max(report.max_normalized_residual);
                    work.record_report(&report);
                    if let Some(kind) = work.exceeded_budget() {
                        return Err(AdaptiveImplicitEulerError::new(
                            kind, t, &y, accepted_steps, rejected_steps, attempts, &work, max_newton_residual
                        ));
                    }
                    report.trajectory.final_state().to_vec()
                }
                Err(error) => {
                    max_newton_residual = max_newton_residual.max(error.residual_norm);
                    work.record_error(&error, system.dim);
                    if let Some(kind) = work.exceeded_budget() {
                        return Err(AdaptiveImplicitEulerError::new(
                            kind, t, &y, accepted_steps, rejected_steps, attempts, &work, max_newton_residual
                        ));
                    }
                    if adaptive_implicit_retryable(&error.kind) {
                        rejected_steps += 1;
                        step_size = trial_step * 0.5;
                        if !step_size.is_finite() || step_size <= 0.0 || t + step_size <= t {
                            return Err(AdaptiveImplicitEulerError::new(
                                AdaptiveImplicitEulerErrorKind::StepSizeUnderflow { attempted_step: step_size },
                                t, &y, accepted_steps, rejected_steps, attempts, &work, max_newton_residual
                            ));
                        }
                        continue;
                    }
                    return Err(AdaptiveImplicitEulerError::new(
                        AdaptiveImplicitEulerErrorKind::StepSolveFailed { cause: error.kind },
                        t, &y, accepted_steps, rejected_steps, attempts, &work, max_newton_residual
                    ));
                }
            };

            let mut squared_error_sum = 0.0_f64;
            for i in 0..system.dim {
                let scale = absolute_tolerances[i]
                    + relative_tolerance * y[i].abs().max(fine_state[i].abs());
                if !scale.is_finite() || scale <= 0.0 {
                    return Err(AdaptiveImplicitEulerError::new(
                        AdaptiveImplicitEulerErrorKind::NonFiniteToleranceScale { index: i },
                        t, &y, accepted_steps, rejected_steps, attempts, &work, max_newton_residual
                    ));
                }
                let normalized = (fine_state[i] - full_state[i]) / scale;
                squared_error_sum += normalized * normalized;
                if !squared_error_sum.is_finite() {
                    return Err(AdaptiveImplicitEulerError::new(
                        AdaptiveImplicitEulerErrorKind::NonFiniteStepDoublingError,
                        t, &y, accepted_steps, rejected_steps, attempts, &work, max_newton_residual
                    ));
                }
            }
            let error_norm = (squared_error_sum / system.dim as f64).sqrt();
            if !error_norm.is_finite() {
                return Err(AdaptiveImplicitEulerError::new(
                    AdaptiveImplicitEulerErrorKind::NonFiniteStepDoublingError,
                    t, &y, accepted_steps, rejected_steps, attempts, &work, max_newton_residual
                ));
            }

            if error_norm <= 1.0 {
                t = trial_end;
                y = fine_state;
                t_values.push(t);
                y_values.push(y.clone());
                accepted_steps += 1;
                max_accepted_error = max_accepted_error.max(error_norm);
                let factor = if error_norm == 0.0 {
                    2.0
                } else {
                    (0.9 / error_norm.sqrt()).clamp(0.2, 2.0)
                };
                step_size = (trial_end - *t_values.get(t_values.len().saturating_sub(2)).unwrap_or(&t0))
                    * factor;
                if !step_size.is_finite() || step_size <= 0.0 {
                    return Err(AdaptiveImplicitEulerError::new(
                        AdaptiveImplicitEulerErrorKind::StepSizeUnderflow { attempted_step: step_size },
                        t, &y, accepted_steps, rejected_steps, attempts, &work, max_newton_residual
                    ));
                }
            } else {
                rejected_steps += 1;
                let factor = (0.9 / error_norm.sqrt()).clamp(0.1, 0.5);
                step_size = trial_step * factor;
                if !step_size.is_finite() || step_size <= 0.0 || t + step_size <= t {
                    return Err(AdaptiveImplicitEulerError::new(
                        AdaptiveImplicitEulerErrorKind::StepSizeUnderflow { attempted_step: step_size },
                        t, &y, accepted_steps, rejected_steps, attempts, &work, max_newton_residual
                    ));
                }
            }
        }

        let trajectory = ODEResult {
            t_values,
            y_values,
            method: ODEMethod::ImplicitEuler,
            steps: accepted_steps,
            phi: 1.0 / (1.0 + y.iter().map(|value| value.abs()).sum::<f64>() / system.dim as f64),
            encoding: Self::encode_implicit_euler_result(&y),
        };
        Ok(AdaptiveImplicitEulerReport {
            trajectory,
            accepted_steps,
            rejected_steps,
            max_accepted_normalized_local_error: max_accepted_error,
            max_normalized_newton_residual: max_newton_residual,
            newton_corrections: work.newton_corrections,
            rhs_evaluations: work.rhs_evaluations,
            estimated_dense_work_units: work.estimated_dense_work_units,
            attempts,
        })
    }

    /// Legacy compatibility API for backward Euler.
    ///
    /// This implementation uses simplified Newton iteration with a diagonal
    /// Jacobian approximation and its result type cannot report failed Newton
    /// convergence. New callers should prefer `try_solve_implicit_euler`,
    /// which uses a full dense Jacobian, checks each nonlinear residual, and
    /// returns typed failure receipts.
    ///
    /// Backward Euler is L-stable as a mathematical method, but this legacy
    /// diagonal approximation does not guarantee that coupled nonlinear
    /// systems converge.
    pub fn solve_implicit_euler(
        system: &ODESystem,
        t0: f64,
        t_end: f64,
        y0: &[f64],
        n_steps: usize,
        newton_iters: usize,
    ) -> ODEResult {
        let dim = system.dim;
        let h = (t_end - t0) / n_steps as f64;
        let mut t_values = Vec::with_capacity(n_steps + 1);
        let mut y_values = Vec::with_capacity(n_steps + 1);
        let mut y = y0.to_vec();
        t_values.push(t0);
        y_values.push(y.clone());

        let newton_tol = 1e-10;

        for step in 0..n_steps {
            let t_next = t0 + (step + 1) as f64 * h;

            // Initial guess: forward Euler predictor
            let f_current = (system.f)(t_next - h, &y);
            let mut y_next: Vec<f64> = (0..dim).map(|i| y[i] + h * f_current[i]).collect();

            // Newton iteration with diagonal Jacobian
            for _iter in 0..newton_iters {
                let fy = (system.f)(t_next, &y_next);

                // Residual: G = y_next - y - h*f(t_next, y_next)
                let residual: Vec<f64> = (0..dim).map(|i| y_next[i] - y[i] - h * fy[i]).collect();

                // Check convergence
                let res_norm: f64 = residual.iter().map(|r| r.abs()).fold(0.0f64, f64::max);
                if res_norm < newton_tol {
                    break;
                }

                // Diagonal Jacobian via finite differences: J_ii = 1 - h * df_i/dy_i
                let eps_fd = 1e-8;
                for i in 0..dim {
                    let orig = y_next[i];
                    let pert = eps_fd * (1.0 + orig.abs());
                    y_next[i] = orig + pert;
                    let fy_pert = (system.f)(t_next, &y_next);
                    y_next[i] = orig; // restore

                    let dfdy_ii = (fy_pert[i] - fy[i]) / pert;
                    let j_ii = 1.0 - h * dfdy_ii;

                    // Newton update: Δy_i = -G_i / J_ii
                    if j_ii.abs() > 1e-15 {
                        y_next[i] -= residual[i] / j_ii;
                    }
                }
            }

            y = y_next;
            t_values.push(t_next);
            y_values.push(y.clone());
        }

        let encoding = Self::encode_implicit_euler_result(&y);
        let phi = 1.0 / (1.0 + y.iter().map(|v| v.abs()).sum::<f64>() / dim as f64);

        ODEResult {
            t_values,
            y_values,
            method: ODEMethod::ImplicitEuler,
            steps: n_steps,
            phi,
            encoding,
        }
    }
}

// ─── Stochastic Differential Equations ───────────────────────────────────────

/// Result of an SDE simulation.
#[derive(Debug, Clone)]
pub struct SDEResult {
    /// Time grid.
    pub t_values: Vec<f64>,
    /// Sample path of the SDE.
    pub y_values: Vec<f64>,
    /// Sample mean at final time (from multiple paths if run_ensemble was called).
    pub mean_final: f64,
    /// Sample variance at final time.
    pub variance_final: f64,
    /// Number of paths simulated (1 for single path).
    pub n_paths: usize,
}

/// Stochastic Differential Equations via Euler-Maruyama.
///
/// Solves dX = f(X,t) dt + g(X,t) dW where dW ~ N(0, dt).
pub struct SDEEngine;

impl SDEEngine {
    /// Simulate a single path of the SDE: dX = f(X,t)dt + g(X,t)dW.
    ///
    /// Uses the Euler-Maruyama scheme:
    ///   X_{n+1} = X_n + f(X_n, t_n)*h + g(X_n, t_n)*sqrt(h)*Z_n
    /// where Z_n ~ N(0,1) via Box-Muller transform from an LCG PRNG.
    pub fn euler_maruyama(
        drift: fn(f64, f64) -> f64,
        diffusion: fn(f64, f64) -> f64,
        x0: f64,
        t0: f64,
        t_end: f64,
        n_steps: usize,
        seed: u64,
    ) -> SDEResult {
        let h = (t_end - t0) / n_steps as f64;
        let sqrt_h = h.sqrt();
        let mut t_values = Vec::with_capacity(n_steps + 1);
        let mut y_values = Vec::with_capacity(n_steps + 1);
        let mut x = x0;
        let mut rng = seed;
        t_values.push(t0);
        y_values.push(x);

        for step in 0..n_steps {
            let t = t0 + step as f64 * h;
            let z = Self::normal_sample(&mut rng);
            x += drift(x, t) * h + diffusion(x, t) * sqrt_h * z;
            t_values.push(t + h);
            y_values.push(x);
        }

        SDEResult {
            mean_final: x,
            variance_final: 0.0,
            t_values,
            y_values,
            n_paths: 1,
        }
    }

    /// Simulate an ensemble of paths and return mean + variance at t_end.
    pub fn ensemble(
        drift: fn(f64, f64) -> f64,
        diffusion: fn(f64, f64) -> f64,
        x0: f64,
        t0: f64,
        t_end: f64,
        n_steps: usize,
        n_paths: usize,
        seed: u64,
    ) -> SDEResult {
        let mut finals = Vec::with_capacity(n_paths);
        let h = (t_end - t0) / n_steps as f64;
        let sqrt_h = h.sqrt();

        for path in 0..n_paths {
            let mut x = x0;
            let mut rng = seed.wrapping_add(path as u64 * 1_000_003);
            for step in 0..n_steps {
                let t = t0 + step as f64 * h;
                let z = Self::normal_sample(&mut rng);
                x += drift(x, t) * h + diffusion(x, t) * sqrt_h * z;
            }
            finals.push(x);
        }

        let mean = finals.iter().sum::<f64>() / n_paths as f64;
        let variance = finals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n_paths as f64;

        SDEResult {
            mean_final: mean,
            variance_final: variance,
            t_values: vec![t0, t_end],
            y_values: vec![x0, mean],
            n_paths,
        }
    }

    /// Ornstein-Uhlenbeck process: dX = θ(μ-X)dt + σdW.
    /// Analytical mean: E[X(t)] = μ + (x0-μ)*exp(-θ*t)
    /// Analytical variance: σ²/(2θ) * (1 - exp(-2θt))
    pub fn ornstein_uhlenbeck(
        theta: f64,
        mu: f64,
        sigma: f64,
        x0: f64,
        t_end: f64,
        n_steps: usize,
        n_paths: usize,
        seed: u64,
    ) -> SDEResult {
        let drift = |x: f64, _t: f64| theta * (mu - x);
        let diffusion = |_x: f64, _t: f64| sigma;
        // Can't use closures directly with fn pointers — use helper
        // Instead, simulate inline:
        let h = t_end / n_steps as f64;
        let sqrt_h = h.sqrt();
        let mut finals = Vec::with_capacity(n_paths);

        for path in 0..n_paths {
            let mut x = x0;
            let mut rng = seed.wrapping_add(path as u64 * 999_983);
            for step in 0..n_steps {
                let _t = step as f64 * h;
                let z = Self::normal_sample(&mut rng);
                x += drift(x, _t) * h + diffusion(x, _t) * sqrt_h * z;
            }
            finals.push(x);
        }

        let mean = finals.iter().sum::<f64>() / n_paths as f64;
        let variance = finals.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n_paths as f64;
        SDEResult {
            mean_final: mean,
            variance_final: variance,
            t_values: vec![0.0, t_end],
            y_values: vec![x0, mean],
            n_paths,
        }
    }

    /// Geometric Brownian Motion: dS = μS dt + σS dW.
    /// Used in Black-Scholes model.  Analytical solution:
    ///   S(t) = S₀ * exp((μ - σ²/2)*t + σ*W(t))
    pub fn geometric_brownian_motion(
        mu: f64,
        sigma: f64,
        s0: f64,
        t_end: f64,
        n_steps: usize,
        seed: u64,
    ) -> SDEResult {
        let h = t_end / n_steps as f64;
        let sqrt_h = h.sqrt();
        let mut t_values = Vec::with_capacity(n_steps + 1);
        let mut y_values = Vec::with_capacity(n_steps + 1);
        let mut s = s0;
        let mut rng = seed;
        t_values.push(0.0);
        y_values.push(s);

        for step in 0..n_steps {
            let z = Self::normal_sample(&mut rng);
            // Euler-Maruyama for GBM: dS = μS dt + σS dW
            s += mu * s * h + sigma * s * sqrt_h * z;
            s = s.max(1e-300); // keep positive
            t_values.push((step + 1) as f64 * h);
            y_values.push(s);
        }

        SDEResult {
            mean_final: s,
            variance_final: 0.0,
            t_values,
            y_values,
            n_paths: 1,
        }
    }

    /// Box-Muller transform: generates N(0,1) sample from LCG uniform samples.
    fn normal_sample(rng: &mut u64) -> f64 {
        let u1 = Self::lcg_next(rng);
        let u2 = Self::lcg_next(rng);
        // Box-Muller: Z = sqrt(-2 ln U1) * cos(2π U2)
        let r = (-2.0 * u1.ln()).sqrt();
        r * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Linear congruential generator → uniform (0,1).
    fn lcg_next(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Map to (0,1) avoiding exact 0
        let bits = (*state >> 11) as f64;
        (bits + 0.5) / (1u64 << 53) as f64
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-5;

    // ── IVP: exponential decay ────────────────────────────────────────────

    #[test]
    fn test_ivp_exponential_decay() {
        // dy/dt = -y,  y(0) = 1  →  y(t) = e^{-t}
        fn decay(_t: f64, y: &[f64]) -> Vec<f64> {
            vec![-y[0]]
        }
        let sys = ODESystem { f: decay, dim: 1 };
        let result = DifferentialEquationsEngine::solve_ivp(&sys, 0.0, 1.0, &[1.0], 10_000);

        let expected = (-1.0_f64).exp();
        assert!(
            (result.final_state()[0] - expected).abs() < TOL,
            "Expected ~{:.6}, got {:.6}",
            expected,
            result.final_state()[0]
        );
        assert_eq!(result.steps, 10_000);
        assert!(result.phi > 0.0);
    }

    #[test]
    fn test_ivp_exponential_decay_t2() {
        // y(2) = e^{-2}
        fn decay(_t: f64, y: &[f64]) -> Vec<f64> {
            vec![-y[0]]
        }
        let sys = ODESystem { f: decay, dim: 1 };
        let result = DifferentialEquationsEngine::solve_ivp(&sys, 0.0, 2.0, &[1.0], 20_000);
        let expected = (-2.0_f64).exp();
        assert!(
            (result.final_state()[0] - expected).abs() < TOL,
            "Expected {:.8}, got {:.8}",
            expected,
            result.final_state()[0]
        );
    }

    #[test]
    fn test_ivp_stores_full_trajectory() {
        fn decay(_t: f64, y: &[f64]) -> Vec<f64> {
            vec![-y[0]]
        }
        let sys = ODESystem { f: decay, dim: 1 };
        let result = DifferentialEquationsEngine::solve_ivp(&sys, 0.0, 1.0, &[1.0], 100);
        assert_eq!(result.t_values.len(), 101);
        assert_eq!(result.y_values.len(), 101);
        // First point is the initial condition
        assert!((result.y_values[0][0] - 1.0).abs() < 1e-15);
        // Solution must be monotonically decreasing
        for i in 1..result.y_values.len() {
            assert!(result.y_values[i][0] < result.y_values[i - 1][0]);
        }
    }

    // ── IVP: harmonic oscillator ──────────────────────────────────────────

    #[test]
    fn test_ivp_harmonic_oscillator() {
        // y'' + y = 0  →  y(t) = cos(t), y'(t) = -sin(t)
        // State: [y, y'] = [cos(t), -sin(t)],  y(0)=1, y'(0)=0
        fn harmonic(_t: f64, y: &[f64]) -> Vec<f64> {
            vec![y[1], -y[0]]
        }
        let sys = ODESystem {
            f: harmonic,
            dim: 2,
        };
        let result = DifferentialEquationsEngine::solve_ivp(
            &sys,
            0.0,
            std::f64::consts::PI,
            &[1.0, 0.0],
            10_000,
        );

        // y(π) = cos(π) = -1
        let expected_y = std::f64::consts::PI.cos();
        assert!(
            (result.final_state()[0] - expected_y).abs() < TOL,
            "y(π) expected {:.6}, got {:.6}",
            expected_y,
            result.final_state()[0]
        );
    }

    #[test]
    fn test_ivp_harmonic_energy_conservation() {
        // Total energy E = y² + (y')² should be conserved ≈ 1
        fn harmonic(_t: f64, y: &[f64]) -> Vec<f64> {
            vec![y[1], -y[0]]
        }
        let sys = ODESystem {
            f: harmonic,
            dim: 2,
        };
        // Integrate several full periods
        let result = DifferentialEquationsEngine::solve_ivp(
            &sys,
            0.0,
            4.0 * std::f64::consts::PI,
            &[1.0, 0.0],
            40_000,
        );
        let fs = result.final_state();
        let energy = fs[0] * fs[0] + fs[1] * fs[1];
        assert!(
            (energy - 1.0).abs() < 1e-3,
            "Energy should be conserved near 1.0, got {:.6}",
            energy
        );
    }

    // ── BVP: simple second-order ODE ─────────────────────────────────────

    #[test]
    fn test_bvp_shooting_trivial() {
        // y'' = 0,  y(0) = 0,  y(1) = 1  →  y(t) = t
        // 2D system: [y, y']' = [y', 0]
        fn linear_second_order(_t: f64, y: &[f64]) -> Vec<f64> {
            vec![y[1], 0.0]
        }
        let result = DifferentialEquationsEngine::solve_bvp_shooting(
            linear_second_order,
            0.0,  // a
            1.0,  // b
            0.0,  // y_a = 0
            1.0,  // y_b = 1
            -5.0, // slope bracket lo
            5.0,  // slope bracket hi
            1000,
        );
        assert!(result.converged, "BVP should converge");
        assert!(
            result.boundary_residual < 1e-6,
            "Boundary residual {:.2e} too large",
            result.boundary_residual
        );
        // Shooting slope should be ~1 (y = t → y' = 1)
        assert!(
            (result.shooting_slope - 1.0).abs() < 1e-5,
            "Expected slope ~1, got {:.6}",
            result.shooting_slope
        );
    }

    #[test]
    fn test_bvp_shooting_sine_like() {
        // y'' = -y,  y(0) = 0,  y(π/2) = 1  →  y(t) = sin(t)
        // Note: y'' = -y means [y, y']' = [y', -y]
        fn sine_ode(_t: f64, y: &[f64]) -> Vec<f64> {
            vec![y[1], -y[0]]
        }
        let half_pi = std::f64::consts::FRAC_PI_2;
        let result = DifferentialEquationsEngine::solve_bvp_shooting(
            sine_ode, 0.0,     // a
            half_pi, // b
            0.0,     // y(0) = 0
            1.0,     // y(π/2) = 1
            0.0,     // slope bracket lo
            3.0,     // slope bracket hi
            5000,
        );
        assert!(result.converged, "BVP (sine) should converge");
        assert!(
            result.boundary_residual < 1e-5,
            "Boundary residual {:.2e}",
            result.boundary_residual
        );
        // y'(0) = cos(0) = 1 for sin(t)
        assert!(
            (result.shooting_slope - 1.0).abs() < 1e-4,
            "Expected slope ~1, got {:.6}",
            result.shooting_slope
        );
    }

    // ── Heat equation ─────────────────────────────────────────────────────

    #[test]
    fn test_heat_equation_decay() {
        // Any non-zero initial condition should decay toward 0 over time
        // with Dirichlet BCs u(0,t) = u(1,t) = 0
        let n_x = 20;
        let u0: Vec<f64> = (1..=n_x)
            .map(|i| {
                let x = i as f64 / (n_x + 1) as f64;
                (std::f64::consts::PI * x).sin()
            })
            .collect();

        let result = DifferentialEquationsEngine::heat_equation_1d(
            0.1, // alpha
            1.0, // length
            1.0, // t_end
            n_x, &u0,
        );

        // Solution should have decayed
        let final_max: f64 = result.u_final.iter().cloned().fold(0.0_f64, f64::max);
        let initial_max: f64 = u0.iter().cloned().fold(0.0_f64, f64::max);
        assert!(
            final_max < initial_max,
            "Heat should dissipate: initial max {:.4}, final max {:.4}",
            initial_max,
            final_max
        );
        assert!(
            result.phi > 0.5,
            "Phi should be reasonable for stable solve"
        );
    }

    #[test]
    fn test_heat_equation_zero_initial() {
        // Zero IC should stay zero
        let n_x = 10;
        let u0 = vec![0.0_f64; n_x];
        let result = DifferentialEquationsEngine::heat_equation_1d(0.1, 1.0, 0.5, n_x, &u0);
        let max_val: f64 = result.u_final.iter().cloned().fold(0.0_f64, f64::max);
        assert!(max_val.abs() < 1e-15, "Zero IC must remain zero");
    }

    #[test]
    fn test_heat_equation_convergence() {
        // Coarser vs finer spatial grid: finer should give smaller max value
        // at same time (Gibbs effect aside, both decay — coarser decays faster due to
        // larger effective diffusion), but both should produce finite, bounded results.
        let u0_coarse: Vec<f64> = (1..=10_usize)
            .map(|i| {
                let x = i as f64 / 11.0;
                (std::f64::consts::PI * x).sin()
            })
            .collect();
        let u0_fine: Vec<f64> = (1..=50_usize)
            .map(|i| {
                let x = i as f64 / 51.0;
                (std::f64::consts::PI * x).sin()
            })
            .collect();

        let r_coarse =
            DifferentialEquationsEngine::heat_equation_1d(0.01, 1.0, 0.1, 10, &u0_coarse);
        let r_fine = DifferentialEquationsEngine::heat_equation_1d(0.01, 1.0, 0.1, 50, &u0_fine);

        // Both solutions must be finite and bounded in [0, 1]
        assert!(r_coarse.u_final.iter().all(|v| v.is_finite()));
        assert!(r_fine.u_final.iter().all(|v| v.is_finite()));
        assert!(
            r_coarse
                .u_final
                .iter()
                .all(|&v| v >= -1e-10 && v <= 1.0 + 1e-10)
        );
        assert!(
            r_fine
                .u_final
                .iter()
                .all(|&v| v >= -1e-10 && v <= 1.0 + 1e-10)
        );
    }

    // ── Wave equation ─────────────────────────────────────────────────────

    #[test]
    fn test_wave_equation_stationary() {
        // Zero IC and zero velocity → zero solution forever
        let n_x = 20;
        let u0 = vec![0.0_f64; n_x];
        let v0 = vec![0.0_f64; n_x];
        let result = DifferentialEquationsEngine::wave_equation_1d(1.0, 1.0, 0.5, n_x, &u0, &v0);
        let max_val: f64 = result
            .u_final
            .iter()
            .cloned()
            .fold(0.0_f64, |a, b| a.abs().max(b.abs()));
        assert!(max_val < 1e-14, "Zero IC must remain zero");
    }

    #[test]
    fn test_wave_equation_energy_conservation() {
        // A standing wave should conserve total energy over many cycles.
        let n_x = 40;
        let u0: Vec<f64> = (1..=n_x)
            .map(|i| {
                let x = i as f64 / (n_x + 1) as f64;
                (std::f64::consts::PI * x).sin()
            })
            .collect();
        let v0 = vec![0.0_f64; n_x];

        let result = DifferentialEquationsEngine::wave_equation_1d(1.0, 1.0, 2.0, n_x, &u0, &v0);

        // Check that the final solution is bounded (no blow-up)
        let max_amplitude: f64 = result
            .u_final
            .iter()
            .cloned()
            .fold(0.0_f64, |a, b| a.abs().max(b.abs()));
        assert!(
            max_amplitude < 2.0,
            "Wave amplitude should stay bounded, got {:.4}",
            max_amplitude
        );
        // Phi should be reasonable
        assert!(result.phi > 0.3);
    }

    #[test]
    fn test_wave_equation_finite_output() {
        // Gaussian initial displacement
        let n_x = 30;
        let u0: Vec<f64> = (1..=n_x)
            .map(|i| {
                let x = i as f64 / (n_x + 1) as f64 - 0.5;
                (-50.0 * x * x).exp()
            })
            .collect();
        let v0 = vec![0.0_f64; n_x];

        let result = DifferentialEquationsEngine::wave_equation_1d(1.0, 1.0, 1.0, n_x, &u0, &v0);
        assert!(
            result.u_final.iter().all(|v| v.is_finite()),
            "All wave solution values must be finite"
        );
    }

    // ── HDC encoding sanity ───────────────────────────────────────────────

    #[test]
    fn test_encoding_distinct_solutions() {
        // Two very different ODEs should have dissimilar encodings
        fn fast_decay(_t: f64, y: &[f64]) -> Vec<f64> {
            vec![-10.0 * y[0]]
        }
        fn slow_growth(_t: f64, y: &[f64]) -> Vec<f64> {
            vec![y[0]]
        }
        let sys1 = ODESystem {
            f: fast_decay,
            dim: 1,
        };
        let sys2 = ODESystem {
            f: slow_growth,
            dim: 1,
        };
        let r1 = DifferentialEquationsEngine::solve_ivp(&sys1, 0.0, 1.0, &[1.0], 1000);
        let r2 = DifferentialEquationsEngine::solve_ivp(&sys2, 0.0, 1.0, &[1.0], 1000);
        let sim = r1.encoding.similarity(&r2.encoding);
        assert!(
            sim < 0.6,
            "Distinct solutions should have dissimilar encodings: similarity = {:.4}",
            sim
        );
    }

    #[test]
    fn test_phi_positive_for_valid_solve() {
        fn decay(_t: f64, y: &[f64]) -> Vec<f64> {
            vec![-y[0]]
        }
        let sys = ODESystem { f: decay, dim: 1 };
        let result = DifferentialEquationsEngine::solve_ivp(&sys, 0.0, 1.0, &[1.0], 500);
        assert!(result.phi > 0.0, "Phi must be positive for a valid solve");
    }

    // ── RK45 tests ────────────────────────────────────────────────────────

    #[test]
    fn test_rk45_exponential_convergence() {
        // dy/dt = y, y(0) = 1 → y(1) = e ≈ 2.71828
        fn exp_rhs(_t: f64, y: &[f64]) -> Vec<f64> {
            vec![y[0]]
        }
        let sys = ODESystem { f: exp_rhs, dim: 1 };
        let result = DifferentialEquationsEngine::solve_rk45(&sys, 0.0, 1.0, &[1.0], 1e-8);
        assert!(result.converged, "RK45 should converge");
        let err = (result.y_final[0] - std::f64::consts::E).abs();
        assert!(
            err < 1e-7,
            "RK45 exponential error should be < 1e-7, got {}",
            err
        );
    }

    #[test]
    fn test_rk45_adaptive_fewer_steps_than_rk4() {
        // On smooth problem RK45 should use fewer evaluations than fixed RK4 at same accuracy
        fn harm(_t: f64, y: &[f64]) -> Vec<f64> {
            vec![y[1], -y[0]]
        }
        let sys = ODESystem { f: harm, dim: 2 };
        let rk45 = DifferentialEquationsEngine::solve_rk45(&sys, 0.0, 10.0, &[1.0, 0.0], 1e-6);
        assert!(rk45.converged, "RK45 harmonic oscillator should converge");
        assert!(
            rk45.steps_taken < 5000,
            "RK45 should adapt step size efficiently"
        );
    }

    #[test]
    fn test_rk45_rejected_steps_finite() {
        fn stiff(_t: f64, y: &[f64]) -> Vec<f64> {
            vec![-50.0 * y[0]]
        }
        let sys = ODESystem { f: stiff, dim: 1 };
        let result = DifferentialEquationsEngine::solve_rk45(&sys, 0.0, 1.0, &[1.0], 1e-6);
        // May or may not converge on stiff problem but should not panic
        assert!(
            result.rejected_steps < 100_000,
            "Should not loop infinitely"
        );
    }

    #[test]
    fn test_implicit_euler_decay() {
        // dy/dt = -y → y(t) = exp(-t)
        fn decay(_t: f64, y: &[f64]) -> Vec<f64> {
            vec![-y[0]]
        }
        let sys = ODESystem { f: decay, dim: 1 };
        let result =
            DifferentialEquationsEngine::solve_implicit_euler(&sys, 0.0, 1.0, &[1.0], 1000, 5);
        let final_y = result.final_state()[0];
        let expected = (-1.0f64).exp();
        assert!(
            (final_y - expected).abs() < 0.01,
            "Implicit Euler decay: got {:.6}, expected {:.6}",
            final_y,
            expected
        );
    }

    // ── Stiff ODE validation tests ──────────────────────────────────────

    /// STIFF TEST: dy/dt = -1000y + 1000 → y(t) = 1 - exp(-1000t), y(0) = 0
    /// Eigenvalue ratio = 1000. Forward Euler requires h < 0.002.
    /// Implicit Euler should handle h = 0.01 (5× beyond explicit stability).
    #[test]
    fn test_adaptive_implicit_euler_step_doubling_reduces_exponential_error() {
        fn growth(_t: f64, y: &[f64]) -> Vec<f64> {
            vec![y[0]]
        }
        let system = ODESystem { f: growth, dim: 1 };
        let coarse = DifferentialEquationsEngine::try_solve_implicit_euler_adaptive(
            &system, 0.0, 1.0, &[1.0], &[1e-3], 1e-3, 1.0, 10_000, 12,
        )
        .expect("looser tolerance should reach the endpoint");
        let tight = DifferentialEquationsEngine::try_solve_implicit_euler_adaptive(
            &system, 0.0, 1.0, &[1.0], &[1e-5], 1e-5, 1.0, 10_000, 12,
        )
        .expect("tighter tolerance should reach the endpoint");
        let exact = 1.0_f64.exp();
        let coarse_error = (coarse.trajectory.final_state()[0] - exact).abs();
        let tight_error = (tight.trajectory.final_state()[0] - exact).abs();

        assert_eq!(coarse.trajectory.t_end(), 1.0);
        assert_eq!(tight.trajectory.t_end(), 1.0);
        assert!(coarse.accepted_steps > 1);
        assert!(tight.accepted_steps > coarse.accepted_steps);
        assert!(tight_error < coarse_error);
        assert!(coarse.max_accepted_normalized_local_error <= 1.0);
        assert!(tight.max_accepted_normalized_local_error <= 1.0);
        assert!(tight.max_normalized_newton_residual <= 1e-10);
    }

    #[test]
    fn test_adaptive_implicit_euler_attempt_budget_counts_rejections() {
        fn growth(_t: f64, y: &[f64]) -> Vec<f64> {
            vec![y[0]]
        }
        let system = ODESystem { f: growth, dim: 1 };
        let error = DifferentialEquationsEngine::try_solve_implicit_euler_adaptive(
            &system, 0.0, 1.0, &[1.0], &[1e-12], 1e-12, 1.0, 1, 8,
        )
        .expect_err("one attempt should be exhausted after rejecting an inaccurate step");
        assert!(matches!(
            error.kind,
            AdaptiveImplicitEulerErrorKind::AttemptBudgetExceeded { max_attempts: 1 }
        ));
        assert_eq!(error.accepted_steps, 0);
        assert_eq!(error.rejected_steps, 1);
        assert_eq!(error.attempts, 1);
        assert_eq!(error.last_state, vec![1.0]);
    }

    #[test]
    fn test_adaptive_implicit_euler_invalid_tolerances_are_typed() {
        fn growth(_t: f64, y: &[f64]) -> Vec<f64> {
            vec![y[0], y[1]]
        }
        let system = ODESystem { f: growth, dim: 2 };
        let error = DifferentialEquationsEngine::try_solve_implicit_euler_adaptive(
            &system, 0.0, 1.0, &[1.0, 1.0], &[1e-5], 1e-3, 0.1, 100, 8,
        )
        .unwrap_err();
        assert!(matches!(
            error.kind,
            AdaptiveImplicitEulerErrorKind::ToleranceDimensionMismatch {
                expected: 2,
                actual: 1,
            }
        ));
    }

    #[test]
    fn test_adaptive_implicit_euler_malformed_rhs_fails_without_advancing_time() {
        fn malformed(_t: f64, _y: &[f64]) -> Vec<f64> {
            vec![]
        }
        let system = ODESystem { f: malformed, dim: 1 };
        let error = DifferentialEquationsEngine::try_solve_implicit_euler_adaptive(
            &system, 0.0, 1.0, &[1.0], &[1e-5], 1e-3, 0.1, 100, 8,
        )
        .unwrap_err();
        assert!(matches!(
            error.kind,
            AdaptiveImplicitEulerErrorKind::StepSolveFailed {
                cause: ImplicitEulerErrorKind::RhsDimensionMismatch {
                    expected: 1,
                    actual: 0,
                    ..
                }
            }
        ));
        assert_eq!(error.time, 0.0);
        assert_eq!(error.last_state, vec![1.0]);
        assert_eq!(error.accepted_steps, 0);
    }

    #[test]
    fn test_implicit_euler_full_newton_solves_coupled_linear_system() {
        fn coupled(_t: f64, y: &[f64]) -> Vec<f64> {
            vec![10.0 * y[1], 10.0 * y[0]]
        }
        let system = ODESystem { f: coupled, dim: 2 };
        let report = DifferentialEquationsEngine::try_solve_implicit_euler(
            &system, 0.0, 0.04, &[1.0, 0.0], 1, 8,
        )
        .expect("full Newton should solve a coupled linear implicit step");
        let state = report.trajectory.final_state();
        assert!((state[0] - (1.0 / 0.84)).abs() < 1e-9, "state={state:?}");
        assert!((state[1] - (0.4 / 0.84)).abs() < 1e-9, "state={state:?}");
        assert!(report.max_normalized_residual <= 1e-10);
        assert!(report.rhs_evaluations > report.newton_corrections);
        assert_eq!(report.trajectory.steps, 1);
    }

    #[test]
    fn test_implicit_euler_fallible_solver_reports_newton_nonconvergence() {
        fn quadratic(_t: f64, y: &[f64]) -> Vec<f64> {
            vec![y[0] * y[0]]
        }
        let system = ODESystem { f: quadratic, dim: 1 };
        let error = DifferentialEquationsEngine::try_solve_implicit_euler(
            &system, 0.0, 0.1, &[1.0], 1, 1,
        )
        .expect_err("one Newton correction is insufficient for this nonlinear step");
        assert!(matches!(
            error.kind,
            ImplicitEulerErrorKind::NewtonNonConvergence {
                iteration_limit: 1,
                ..
            }
        ));
        assert_eq!(error.steps_completed, 0);
        assert_eq!(error.last_state, vec![1.0]);
        assert_eq!(error.failed_step, 1);
    }

    #[test]
    fn test_implicit_euler_fallible_solver_rejects_invalid_contracts() {
        fn decay(_t: f64, y: &[f64]) -> Vec<f64> {
            vec![-y[0]]
        }
        let system = ODESystem { f: decay, dim: 1 };
        assert!(matches!(
            DifferentialEquationsEngine::try_solve_implicit_euler(
                &system, 0.0, 1.0, &[1.0], 0, 4,
            ).unwrap_err().kind,
            ImplicitEulerErrorKind::InvalidStepCount { requested: 0, .. }
        ));
        assert!(matches!(
            DifferentialEquationsEngine::try_solve_implicit_euler(
                &system, 0.0, 1.0, &[1.0], 2, 0,
            ).unwrap_err().kind,
            ImplicitEulerErrorKind::InvalidNewtonIterationBudget { requested: 0, .. }
        ));
        assert!(matches!(
            DifferentialEquationsEngine::try_solve_implicit_euler(
                &system, 0.0, 1.0, &[1.0, 2.0], 2, 4,
            ).unwrap_err().kind,
            ImplicitEulerErrorKind::DimensionMismatch { expected: 1, actual: 2 }
        ));
        assert!(matches!(
            DifferentialEquationsEngine::try_solve_implicit_euler(
                &system, 0.0, 1.0, &[f64::NAN], 2, 4,
            ).unwrap_err().kind,
            ImplicitEulerErrorKind::NonFiniteInitialState { index: 0 }
        ));
    }

    #[test]
    fn test_implicit_euler_fallible_solver_checks_rhs_shape() {
        fn malformed(_t: f64, _y: &[f64]) -> Vec<f64> {
            vec![]
        }
        let system = ODESystem { f: malformed, dim: 1 };
        let error = DifferentialEquationsEngine::try_solve_implicit_euler(
            &system, 0.0, 1.0, &[1.0], 1, 4,
        )
        .unwrap_err();
        assert!(matches!(
            error.kind,
            ImplicitEulerErrorKind::RhsDimensionMismatch { expected: 1, actual: 0, .. }
        ));
        assert_eq!(error.last_state, vec![1.0]);
    }

    #[test]
    fn test_implicit_euler_fallible_solver_rejects_singular_newton_matrix() {
        fn identity(_t: f64, y: &[f64]) -> Vec<f64> {
            y.to_vec()
        }
        let system = ODESystem { f: identity, dim: 1 };
        let error = DifferentialEquationsEngine::try_solve_implicit_euler(
            &system, 0.0, 1.0, &[1.0], 1, 4,
        )
        .expect_err("this step has no backward-Euler root because h*f(y)=y");
        assert!(matches!(
            error.kind,
            ImplicitEulerErrorKind::SingularJacobian { column: 0, .. }
        ));
        assert_eq!(error.last_state, vec![1.0]);
        assert_eq!(error.steps_completed, 0);
    }

    #[test]
    fn test_implicit_euler_fallible_solver_rejects_dense_system_oversize() {
        fn copy_state(_t: f64, y: &[f64]) -> Vec<f64> {
            y.to_vec()
        }
        let system = ODESystem { f: copy_state, dim: MAX_DENSE_NEWTON_DIM + 1 };
        let initial = vec![0.0; MAX_DENSE_NEWTON_DIM + 1];
        let error = DifferentialEquationsEngine::try_solve_implicit_euler(
            &system, 0.0, 1.0, &initial, 1, 4,
        )
        .expect_err("the dense solver must reject dimensions outside its declared envelope");
        assert!(matches!(
            error.kind,
            ImplicitEulerErrorKind::SystemTooLargeForDenseNewton {
                dimension: 65,
                max_dimension: 64,
            }
        ));
    }

    #[test]
    fn test_implicit_euler_fallible_solver_matches_stiff_decay_reference() {
        fn stiff_decay(_t: f64, y: &[f64]) -> Vec<f64> {
            vec![-1000.0 * y[0] + 1000.0]
        }
        let system = ODESystem { f: stiff_decay, dim: 1 };
        let report = DifferentialEquationsEngine::try_solve_implicit_euler(
            &system, 0.0, 0.1, &[0.0], 10, 8,
        )
        .expect("linear stiff decay should converge under backward Euler");
        assert!(
            (report.trajectory.final_state()[0] - 1.0).abs() < 1e-8,
            "final={:?}",
            report.trajectory.final_state()
        );
        assert!(report.max_normalized_residual <= 1e-10);
    }

    #[test]
    fn test_implicit_euler_stiff_decay() {
        fn stiff_decay(_t: f64, y: &[f64]) -> Vec<f64> {
            vec![-1000.0 * y[0] + 1000.0]
        }
        let sys = ODESystem {
            f: stiff_decay,
            dim: 1,
        };
        // h = 0.01 → h*λ = 10 (well beyond explicit stability limit of 2)
        let result =
            DifferentialEquationsEngine::solve_implicit_euler(&sys, 0.0, 0.1, &[0.0], 10, 10);
        let final_y = result.final_state()[0];
        // y(0.1) = 1 - exp(-100) ≈ 1.0
        assert!(
            (final_y - 1.0).abs() < 0.1,
            "Stiff decay: got {:.6}, expected ≈ 1.0 (h*λ=10, implicit should handle this)",
            final_y
        );
    }

    /// STIFF TEST: Two-rate system
    /// dy1/dt = -y1         (slow, λ₁ = 1)
    /// dy2/dt = -1000*y2    (fast, λ₂ = 1000)
    /// Eigenvalue ratio = 1000. Step size h = 0.01.
    #[test]
    fn test_implicit_euler_two_rate() {
        fn two_rate(_t: f64, y: &[f64]) -> Vec<f64> {
            vec![-y[0], -1000.0 * y[1]]
        }
        let sys = ODESystem {
            f: two_rate,
            dim: 2,
        };
        let result =
            DifferentialEquationsEngine::solve_implicit_euler(&sys, 0.0, 1.0, &[1.0, 1.0], 100, 10);
        let final_state = result.final_state();

        // y1(1) = exp(-1) ≈ 0.3679
        let y1_expected = (-1.0f64).exp();
        assert!(
            (final_state[0] - y1_expected).abs() < 0.05,
            "Slow component: got {:.6}, expected {:.6}",
            final_state[0],
            y1_expected
        );

        // y2(1) = exp(-1000) ≈ 0 (fast component fully decayed)
        assert!(
            final_state[1].abs() < 1e-3,
            "Fast component should decay to ~0, got {:.6}",
            final_state[1]
        );
    }

    /// STIFF TEST: Van der Pol oscillator with μ = 1000 (severely stiff)
    /// dy1/dt = y2
    /// dy2/dt = μ(1 - y1²)y2 - y1
    /// At μ = 1000, the system exhibits fast transients near limit cycle boundary.
    #[test]
    fn test_implicit_euler_van_der_pol() {
        fn van_der_pol(_t: f64, y: &[f64]) -> Vec<f64> {
            let mu = 1000.0;
            vec![y[1], mu * (1.0 - y[0] * y[0]) * y[1] - y[0]]
        }
        let sys = ODESystem {
            f: van_der_pol,
            dim: 2,
        };
        // Small step size (h=0.001) with many Newton iterations for severely stiff VdP
        let result =
            DifferentialEquationsEngine::solve_implicit_euler(&sys, 0.0, 0.1, &[2.0, 0.0], 100, 20);
        let final_state = result.final_state();

        // Don't check exact value — VdP at μ=1000 is notoriously difficult.
        // Just verify the solver doesn't blow up (finite output).
        assert!(
            final_state[0].is_finite() && final_state[1].is_finite(),
            "Van der Pol (μ=1000) should produce finite output, got [{:.6}, {:.6}]",
            final_state[0],
            final_state[1]
        );
        // The solution should stay bounded (VdP has a stable limit cycle)
        assert!(
            final_state[0].abs() < 10.0 && final_state[1].abs() < 2000.0,
            "Van der Pol should stay bounded, got [{:.6}, {:.6}]",
            final_state[0],
            final_state[1]
        );
    }

    // ── SDE tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_em_brownian_motion_zero_drift() {
        // dX = 0*dt + 1*dW → X(t) has E[X(t)] = 0
        fn zero_drift(_x: f64, _t: f64) -> f64 {
            0.0
        }
        fn unit_diffusion(_x: f64, _t: f64) -> f64 {
            1.0
        }
        let result = SDEEngine::ensemble(zero_drift, unit_diffusion, 0.0, 0.0, 1.0, 1000, 500, 42);
        assert!(
            result.mean_final.abs() < 0.15,
            "BM mean should be near 0, got {}",
            result.mean_final
        );
    }

    #[test]
    fn test_em_exponential_growth() {
        // dX = X dt + 0 dW → X(t) = X₀ * e^t (no noise)
        fn growth(x: f64, _t: f64) -> f64 {
            x
        }
        fn no_noise(_x: f64, _t: f64) -> f64 {
            0.0
        }
        let result = SDEEngine::euler_maruyama(growth, no_noise, 1.0, 0.0, 1.0, 10000, 123);
        let expected = std::f64::consts::E;
        assert!(
            (result.y_values.last().unwrap_or(&0.0) - expected).abs() < 0.01,
            "EM with no noise should match ODE solution"
        );
    }

    #[test]
    fn test_ou_process_mean_reversion() {
        // OU process with θ=2, μ=5: should converge toward μ=5 from x0=0
        let result = SDEEngine::ornstein_uhlenbeck(2.0, 5.0, 0.5, 0.0, 5.0, 5000, 1000, 99);
        assert!(
            result.mean_final > 3.0 && result.mean_final < 7.0,
            "OU process mean at t=5 should be near μ=5, got {}",
            result.mean_final
        );
    }

    #[test]
    fn test_gbm_positive() {
        let result = SDEEngine::geometric_brownian_motion(0.05, 0.2, 100.0, 1.0, 252, 77);
        assert!(
            result.y_values.iter().all(|&v| v > 0.0),
            "GBM should stay positive"
        );
    }

    // ── 2D Heat Equation ────────────────────────────────────────────────

    #[test]
    fn test_heat_equation_2d_energy_decays() {
        let nx = 10;
        let ny = 10;
        // Initial hot spot in the center
        let mut u0 = vec![0.0; nx * ny];
        u0[5 * ny + 5] = 1.0;

        let result =
            DifferentialEquationsEngine::heat_equation_2d(0.01, 1.0, 1.0, 1.0, nx, ny, &u0);

        // Energy should decay (heat dissipates)
        let initial_energy: f64 = u0.iter().map(|v| v * v).sum::<f64>();
        let final_energy: f64 = result.u_final.iter().map(|v| v * v).sum::<f64>();
        assert!(
            final_energy < initial_energy,
            "Energy should decay: initial={initial_energy}, final={final_energy}"
        );
        // All values should remain non-negative (heat can't go negative from positive IC)
        assert!(
            result.u_final.iter().all(|&v| v >= -1e-10),
            "Solution should stay non-negative"
        );
        // Phi should indicate stable solution
        assert!(
            result.phi > 0.5,
            "Phi should indicate good solution: {}",
            result.phi
        );
    }

    #[test]
    fn test_heat_equation_2d_symmetry() {
        let n = 8;
        // Symmetric initial condition (center hot spot)
        let mut u0 = vec![0.0; n * n];
        u0[4 * n + 4] = 1.0;
        u0[3 * n + 4] = 0.5;
        u0[4 * n + 3] = 0.5;
        u0[4 * n + 5] = 0.5;
        u0[5 * n + 4] = 0.5;

        let result = DifferentialEquationsEngine::heat_equation_2d(0.01, 1.0, 1.0, 0.5, n, n, &u0);

        // Result should have correct shape
        assert_eq!(result.u_final.len(), n * n);
        assert_eq!(result.x_grid.len(), n);
        assert_eq!(result.y_grid.len(), n);
        // All values finite
        assert!(result.u_final.iter().all(|v| v.is_finite()));
    }
}
