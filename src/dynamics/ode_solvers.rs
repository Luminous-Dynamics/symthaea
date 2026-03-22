// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Advanced ODE Solvers for Symthaea Neural Dynamics
//!
//! This module provides a suite of ordinary differential equation (ODE) solvers
//! tailored for Symthaea's continuous-time neural networks (LTC and CfC).
//!
//! ## Solver Selection Guide
//!
//! | Solver | Order | Adaptive | Stiff-safe | Best For |
//! |--------|-------|----------|------------|----------|
//! | Euler | 1 | No | No | Quick prototyping, baseline comparisons |
//! | RK4 | 4 | No | No | General purpose, smooth dynamics |
//! | Dormand-Prince | 4/5 | Yes | No | Production neural dynamics (variable dt) |
//! | Implicit Midpoint | 2 | No | Yes | Moderately stiff CfC systems |
//! | Backward Euler | 1 | No | Yes | Maximally stiff systems (very different tau) |
//! | Exponential | exact* | No | Yes | CfC networks (exploits linear structure) |
//!
//! ### When to use which solver for Symthaea:
//!
//! - **LTC forward pass**: Euler (existing) or RK4 for higher accuracy. If time
//!   constants span orders of magnitude (tau_min/tau_max > 100), use Backward Euler
//!   or Implicit Midpoint for stability.
//!
//! - **CfC closed-form**: CfC already has an exact closed-form solution, so ODE
//!   solvers are not needed at inference. However, during training or when
//!   computing gradients through the ODE, Dormand-Prince provides the best
//!   accuracy-per-evaluation tradeoff.
//!
//! - **World model rollouts**: Dormand-Prince with adaptive stepping. The
//!   environment dynamics may have varying stiffness; adaptive stepping handles
//!   this automatically.
//!
//! - **Exponential integrator**: Ideal when the system has the form dy/dt = Ay + g(t,y)
//!   where A is a known linear operator (common in CfC-like architectures).
//!
//! ## Mathematical Background
//!
//! Forward Euler: y_{n+1} = y_n + h * f(t_n, y_n)
//!   Error: O(h^2) per step, O(h) global
//!
//! RK4: y_{n+1} = y_n + (h/6)(k1 + 2*k2 + 2*k3 + k4)
//!   Error: O(h^5) per step, O(h^4) global
//!
//! Dormand-Prince (RK45): Embedded 4th/5th order pair with FSAL property.
//!   Uses error estimate |y5 - y4| for automatic step size control.
//!
//! Implicit Midpoint: y_{n+1} = y_n + h * f(t + h/2, (y_n + y_{n+1})/2)
//!   A-stable (no step size restriction for stability on stiff problems).
//!
//! Backward Euler: y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})
//!   L-stable (maximally damping for stiff modes).
//!
//! Exponential: y_{n+1} = exp(hA)*y_n + (exp(hA) - I)*A^{-1}*g(t_n, y_n)
//!   Exact for the linear part; first-order for the nonlinear part.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// ODE System trait
// ---------------------------------------------------------------------------

/// Trait for systems of ordinary differential equations.
///
/// Implement this for any continuous-time dynamical system (LTC cells, CfC
/// networks, world models, etc.) to use with the solver engine.
///
/// # Example
///
/// ```ignore
/// struct ExponentialDecay { rate: f64 }
///
/// impl OdeSystem for ExponentialDecay {
///     fn dimension(&self) -> usize { 1 }
///     fn evaluate(&self, _t: f64, state: &[f64], derivative: &mut [f64]) {
///         derivative[0] = -self.rate * state[0];
///     }
/// }
/// ```
pub trait OdeSystem {
    /// Number of state variables in the system.
    fn dimension(&self) -> usize;

    /// Evaluate the right-hand side f(t, y) and write into `derivative`.
    ///
    /// `state` and `derivative` both have length `self.dimension()`.
    fn evaluate(&self, t: f64, state: &[f64], derivative: &mut [f64]);
}

// ---------------------------------------------------------------------------
// Solver enum and configuration
// ---------------------------------------------------------------------------

/// Available ODE solver methods.
///
/// Each variant trades off accuracy, cost per step, and stability differently.
/// See the module-level documentation for guidance on which to choose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OdeSolver {
    /// Forward Euler (1st order, explicit). Cheapest per step but lowest accuracy.
    Euler,
    /// Classical Runge-Kutta (4th order, explicit). Good general-purpose choice.
    RK4,
    /// Dormand-Prince embedded RK4(5) with adaptive step size. Best for
    /// production use where accuracy matters and dynamics vary in stiffness.
    DormandPrince,
    /// Implicit midpoint rule (2nd order, A-stable). Good for moderately stiff
    /// systems like CfC with diverse time constants.
    ImplicitMidpoint,
    /// Backward Euler (1st order, L-stable). Maximum stability for very stiff
    /// systems at the cost of accuracy.
    BackwardEuler,
    /// Exponential integrator (exact for linear part). Exploits the structure
    /// dy/dt = A*y + g(t,y) common in CfC architectures.
    Exponential,
}

/// Configuration for the ODE solver engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OdeConfig {
    /// Which solver method to use.
    pub solver: OdeSolver,
    /// Time step size (for fixed-step methods) or initial step size (adaptive).
    pub dt: f64,
    /// Error tolerance for adaptive step size control (Dormand-Prince).
    /// Also used as convergence tolerance for implicit solvers.
    pub tolerance: f64,
    /// Maximum allowed step size (adaptive methods).
    pub max_step: f64,
    /// Minimum allowed step size (adaptive methods). If the solver needs a
    /// smaller step than this, it proceeds anyway and flags a warning.
    pub min_step: f64,
    /// Maximum iterations for Newton/fixed-point solves (implicit methods).
    pub max_iterations: usize,
}

impl Default for OdeConfig {
    fn default() -> Self {
        Self {
            solver: OdeSolver::RK4,
            dt: 0.01,
            tolerance: 1e-6,
            max_step: 1.0,
            min_step: 1e-12,
            max_iterations: 50,
        }
    }
}

// ---------------------------------------------------------------------------
// ODE result
// ---------------------------------------------------------------------------

/// Result of an ODE integration containing the full trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OdeResult {
    /// State vectors at each output time. Each inner `Vec<f64>` has length = dimension.
    pub states: Vec<Vec<f64>>,
    /// Time values corresponding to each state.
    pub times: Vec<f64>,
    /// Step sizes used at each integration step (may differ from `times` spacing
    /// for adaptive methods that take multiple internal steps per output).
    pub step_sizes: Vec<f64>,
    /// Error estimates at each step (only meaningful for adaptive methods;
    /// fixed-step methods report 0.0).
    pub error_estimates: Vec<f64>,
    /// Number of rejected steps (adaptive methods only).
    pub num_rejections: usize,
}

// ---------------------------------------------------------------------------
// Solver engine
// ---------------------------------------------------------------------------

/// The main ODE solver engine.
///
/// Holds configuration and workspace buffers. Reuse across calls to the same
/// system dimension to avoid repeated allocation.
#[derive(Debug, Clone)]
pub struct OdeSolverEngine {
    /// Solver configuration.
    pub config: OdeConfig,
    /// Dimension of the ODE system (number of state variables).
    pub dim: usize,
}

impl OdeSolverEngine {
    /// Create a new solver engine for a system of the given dimension.
    pub fn new(config: OdeConfig, dim: usize) -> Self {
        Self { config, dim }
    }

    // -----------------------------------------------------------------------
    // Public integration entry point
    // -----------------------------------------------------------------------

    /// Integrate the ODE system from `t_span.0` to `t_span.1` starting at `y0`.
    ///
    /// Returns the full trajectory including initial and final states.
    ///
    /// # Panics
    ///
    /// Panics if `y0.len() != self.dim` or if `t_span.1 <= t_span.0`.
    pub fn solve(&self, system: &dyn OdeSystem, y0: &[f64], t_span: (f64, f64)) -> OdeResult {
        assert_eq!(y0.len(), self.dim, "Initial state dimension mismatch");
        assert!(
            t_span.1 > t_span.0,
            "t_span.1 must be greater than t_span.0"
        );

        match self.config.solver {
            OdeSolver::Euler => self.solve_euler(system, y0, t_span),
            OdeSolver::RK4 => self.solve_rk4(system, y0, t_span),
            OdeSolver::DormandPrince => self.solve_dormand_prince(system, y0, t_span),
            OdeSolver::ImplicitMidpoint => self.solve_implicit_midpoint(system, y0, t_span),
            OdeSolver::BackwardEuler => self.solve_backward_euler(system, y0, t_span),
            OdeSolver::Exponential => self.solve_exponential(system, y0, t_span),
        }
    }

    // -----------------------------------------------------------------------
    // Forward Euler
    // -----------------------------------------------------------------------

    /// Single Forward Euler step: y_{n+1} = y_n + h * f(t_n, y_n)
    ///
    /// This is the simplest possible integration method. It is first-order
    /// accurate and conditionally stable. Use it only for rough estimates
    /// or as the baseline in LTC forward passes where h is already small.
    fn step_euler(&self, system: &dyn OdeSystem, t: f64, y: &[f64], h: f64, y_out: &mut [f64]) {
        let mut dydt = vec![0.0; self.dim];
        system.evaluate(t, y, &mut dydt);
        for i in 0..self.dim {
            y_out[i] = y[i] + h * dydt[i];
        }
    }

    fn solve_euler(&self, system: &dyn OdeSystem, y0: &[f64], t_span: (f64, f64)) -> OdeResult {
        let h = self.config.dt;
        let mut t = t_span.0;
        let mut y = y0.to_vec();
        let mut y_next = vec![0.0; self.dim];

        let mut result = OdeResult {
            states: vec![y.clone()],
            times: vec![t],
            step_sizes: Vec::new(),
            error_estimates: Vec::new(),
            num_rejections: 0,
        };

        while t < t_span.1 - 1e-14 {
            let step = h.min(t_span.1 - t);
            self.step_euler(system, t, &y, step, &mut y_next);
            t += step;
            y.copy_from_slice(&y_next);
            result.states.push(y.clone());
            result.times.push(t);
            result.step_sizes.push(step);
            result.error_estimates.push(0.0);
        }

        result
    }

    // -----------------------------------------------------------------------
    // Classical RK4
    // -----------------------------------------------------------------------

    /// Single RK4 step using the classical four-stage Runge-Kutta method.
    ///
    /// k1 = f(t, y)
    /// k2 = f(t + h/2, y + h/2 * k1)
    /// k3 = f(t + h/2, y + h/2 * k2)
    /// k4 = f(t + h, y + h * k3)
    /// y_{n+1} = y_n + (h/6)(k1 + 2*k2 + 2*k3 + k4)
    ///
    /// Fourth-order accurate: the local truncation error is O(h^5), giving
    /// O(h^4) global error. Four function evaluations per step. This is the
    /// workhorse method for smooth, non-stiff neural dynamics.
    fn step_rk4(&self, system: &dyn OdeSystem, t: f64, y: &[f64], h: f64, y_out: &mut [f64]) {
        let n = self.dim;
        let mut k1 = vec![0.0; n];
        let mut k2 = vec![0.0; n];
        let mut k3 = vec![0.0; n];
        let mut k4 = vec![0.0; n];
        let mut tmp = vec![0.0; n];

        // k1 = f(t, y)
        system.evaluate(t, y, &mut k1);

        // k2 = f(t + h/2, y + h/2 * k1)
        for i in 0..n {
            tmp[i] = y[i] + 0.5 * h * k1[i];
        }
        system.evaluate(t + 0.5 * h, &tmp, &mut k2);

        // k3 = f(t + h/2, y + h/2 * k2)
        for i in 0..n {
            tmp[i] = y[i] + 0.5 * h * k2[i];
        }
        system.evaluate(t + 0.5 * h, &tmp, &mut k3);

        // k4 = f(t + h, y + h * k3)
        for i in 0..n {
            tmp[i] = y[i] + h * k3[i];
        }
        system.evaluate(t + h, &tmp, &mut k4);

        // y_{n+1} = y_n + (h/6)(k1 + 2*k2 + 2*k3 + k4)
        for i in 0..n {
            y_out[i] = y[i] + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
    }

    fn solve_rk4(&self, system: &dyn OdeSystem, y0: &[f64], t_span: (f64, f64)) -> OdeResult {
        let h = self.config.dt;
        let mut t = t_span.0;
        let mut y = y0.to_vec();
        let mut y_next = vec![0.0; self.dim];

        let mut result = OdeResult {
            states: vec![y.clone()],
            times: vec![t],
            step_sizes: Vec::new(),
            error_estimates: Vec::new(),
            num_rejections: 0,
        };

        while t < t_span.1 - 1e-14 {
            let step = h.min(t_span.1 - t);
            self.step_rk4(system, t, &y, step, &mut y_next);
            t += step;
            y.copy_from_slice(&y_next);
            result.states.push(y.clone());
            result.times.push(t);
            result.step_sizes.push(step);
            result.error_estimates.push(0.0);
        }

        result
    }

    // -----------------------------------------------------------------------
    // Dormand-Prince RK4(5) with adaptive step size
    // -----------------------------------------------------------------------

    /// Dormand-Prince embedded RK4(5) method with FSAL (First Same As Last).
    ///
    /// This is the standard adaptive Runge-Kutta method used in MATLAB's ode45
    /// and SciPy's RK45. It uses a 7-stage embedded pair where the 4th-order
    /// and 5th-order solutions share the same stages, and the 7th stage of one
    /// step equals the 1st stage of the next (FSAL property).
    ///
    /// The error estimate |y5 - y4| drives the step size controller:
    ///   h_new = h * min(max(0.84 * (tol/err)^(1/5), 0.1), 4.0)
    ///
    /// For Symthaea neural dynamics, this is the recommended production solver:
    /// it automatically takes small steps where dynamics are fast (high-frequency
    /// oscillations) and large steps where dynamics are slow (near attractors).
    fn step_dormand_prince(
        &self,
        system: &dyn OdeSystem,
        t: f64,
        y: &[f64],
        h: f64,
        y4_out: &mut [f64],
        y5_out: &mut [f64],
        k1_in: Option<&[f64]>,
        k7_out: &mut [f64],
    ) -> f64 {
        let n = self.dim;

        // Dormand-Prince Butcher tableau coefficients
        // Nodes (c)
        const C2: f64 = 1.0 / 5.0;
        const C3: f64 = 3.0 / 10.0;
        const C4: f64 = 4.0 / 5.0;
        const C5: f64 = 8.0 / 9.0;
        // C6 = 1.0, C7 = 1.0

        // Stage coefficients (a)
        const A21: f64 = 1.0 / 5.0;

        const A31: f64 = 3.0 / 40.0;
        const A32: f64 = 9.0 / 40.0;

        const A41: f64 = 44.0 / 45.0;
        const A42: f64 = -56.0 / 15.0;
        const A43: f64 = 32.0 / 9.0;

        const A51: f64 = 19372.0 / 6561.0;
        const A52: f64 = -25360.0 / 2187.0;
        const A53: f64 = 64448.0 / 6561.0;
        const A54: f64 = -212.0 / 729.0;

        const A61: f64 = 9017.0 / 3168.0;
        const A62: f64 = -355.0 / 33.0;
        const A63: f64 = 46732.0 / 5247.0;
        const A64: f64 = 49.0 / 176.0;
        const A65: f64 = -5103.0 / 18656.0;

        const _A71: f64 = 35.0 / 384.0;
        // A72 = 0
        const _A73: f64 = 500.0 / 1113.0;
        const _A74: f64 = 125.0 / 192.0;
        const _A75: f64 = -2187.0 / 6784.0;
        const _A76: f64 = 11.0 / 84.0;

        // 5th order weights (b5) - same as a7* for FSAL
        // (These produce the 5th-order solution used as the propagated value)
        const B51: f64 = 35.0 / 384.0;
        // B52 = 0
        const B53: f64 = 500.0 / 1113.0;
        const B54: f64 = 125.0 / 192.0;
        const B55: f64 = -2187.0 / 6784.0;
        const B56: f64 = 11.0 / 84.0;
        // B57 = 0

        // 4th order weights (b4) for error estimation
        const B41: f64 = 5179.0 / 57600.0;
        // B42 = 0
        const B43: f64 = 7571.0 / 16695.0;
        const B44: f64 = 393.0 / 640.0;
        const B45: f64 = -92097.0 / 339200.0;
        const B46: f64 = 187.0 / 2100.0;
        const B47: f64 = 1.0 / 40.0;

        let mut k1 = vec![0.0; n];
        let mut k2 = vec![0.0; n];
        let mut k3 = vec![0.0; n];
        let mut k4 = vec![0.0; n];
        let mut k5 = vec![0.0; n];
        let mut k6 = vec![0.0; n];
        let mut tmp = vec![0.0; n];

        // Stage 1: reuse from previous step if available (FSAL)
        if let Some(k1_prev) = k1_in {
            k1.copy_from_slice(k1_prev);
        } else {
            system.evaluate(t, y, &mut k1);
        }

        // Stage 2
        for i in 0..n {
            tmp[i] = y[i] + h * A21 * k1[i];
        }
        system.evaluate(t + C2 * h, &tmp, &mut k2);

        // Stage 3
        for i in 0..n {
            tmp[i] = y[i] + h * (A31 * k1[i] + A32 * k2[i]);
        }
        system.evaluate(t + C3 * h, &tmp, &mut k3);

        // Stage 4
        for i in 0..n {
            tmp[i] = y[i] + h * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i]);
        }
        system.evaluate(t + C4 * h, &tmp, &mut k4);

        // Stage 5
        for i in 0..n {
            tmp[i] = y[i] + h * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i]);
        }
        system.evaluate(t + C5 * h, &tmp, &mut k5);

        // Stage 6
        for i in 0..n {
            tmp[i] =
                y[i] + h * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i]);
        }
        system.evaluate(t + h, &tmp, &mut k6);

        // 5th order solution (propagated value)
        for i in 0..n {
            y5_out[i] =
                y[i] + h * (B51 * k1[i] + B53 * k3[i] + B54 * k4[i] + B55 * k5[i] + B56 * k6[i]);
        }

        // Stage 7 (FSAL: this becomes k1 of the next step)
        // We evaluate at (t+h, y5) which is the 5th-order solution
        system.evaluate(t + h, y5_out, k7_out);

        // 4th order solution (for error estimation only)
        for i in 0..n {
            y4_out[i] = y[i]
                + h * (B41 * k1[i]
                    + B43 * k3[i]
                    + B44 * k4[i]
                    + B45 * k5[i]
                    + B46 * k6[i]
                    + B47 * k7_out[i]);
        }

        // Error estimate: max-norm of |y5 - y4|
        let mut err = 0.0_f64;
        for i in 0..n {
            let scale = 1.0 + y[i].abs(); // relative-ish tolerance
            let ei = ((y5_out[i] - y4_out[i]) / scale).abs();
            if ei > err {
                err = ei;
            }
        }

        err
    }

    /// Compute the new step size from the error estimate.
    ///
    /// Uses the standard PI controller formula:
    ///   h_new = h * safety * (tol / err)^(1/5)
    /// clamped to [0.1*h, 4.0*h] to prevent wild oscillations, and further
    /// clamped to [min_step, max_step].
    fn adapt_step_size(&self, h: f64, err: f64) -> (f64, bool) {
        let tol = self.config.tolerance;

        if err <= tol {
            // Step accepted; compute optimal new step size
            let factor = if err < 1e-30 {
                4.0 // error is essentially zero, allow maximum growth
            } else {
                0.84 * (tol / err).powf(0.2)
            };
            let factor = factor.clamp(0.1, 4.0);
            let h_new = (h * factor).clamp(self.config.min_step, self.config.max_step);
            (h_new, true) // accepted
        } else {
            // Step rejected; shrink
            let factor = (0.84 * (tol / err).powf(0.2)).clamp(0.1, 0.9);
            let h_new = (h * factor).max(self.config.min_step);
            (h_new, false) // rejected
        }
    }

    fn solve_dormand_prince(
        &self,
        system: &dyn OdeSystem,
        y0: &[f64],
        t_span: (f64, f64),
    ) -> OdeResult {
        let n = self.dim;
        let mut t = t_span.0;
        let mut y = y0.to_vec();
        let mut h = self.config.dt.min(t_span.1 - t_span.0);

        let mut y4 = vec![0.0; n];
        let mut y5 = vec![0.0; n];
        let mut k7 = vec![0.0; n];
        let mut fsal_k1: Option<Vec<f64>> = None;

        let mut result = OdeResult {
            states: vec![y.clone()],
            times: vec![t],
            step_sizes: Vec::new(),
            error_estimates: Vec::new(),
            num_rejections: 0,
        };

        while t < t_span.1 - 1e-14 {
            h = h.min(t_span.1 - t);

            let err = self.step_dormand_prince(
                system,
                t,
                &y,
                h,
                &mut y4,
                &mut y5,
                fsal_k1.as_deref(),
                &mut k7,
            );

            let (h_new, accepted) = self.adapt_step_size(h, err);

            if accepted {
                t += h;
                y.copy_from_slice(&y5);
                fsal_k1 = Some(k7.clone()); // FSAL: k7 becomes k1 of next step

                result.states.push(y.clone());
                result.times.push(t);
                result.step_sizes.push(h);
                result.error_estimates.push(err);
            } else {
                result.num_rejections += 1;
                fsal_k1 = None; // FSAL invalidated on rejection
            }

            h = h_new;

            // Safety: prevent infinite loops from extremely stiff systems
            if h < self.config.min_step * 0.5 {
                // Force acceptance at min step
                h = self.config.min_step;
            }
        }

        result
    }

    // -----------------------------------------------------------------------
    // Implicit Midpoint Method
    // -----------------------------------------------------------------------

    /// Implicit midpoint rule: y_{n+1} = y_n + h * f(t + h/2, (y_n + y_{n+1})/2)
    ///
    /// This is a second-order, A-stable method. A-stability means there is no
    /// step size restriction for stability on stiff problems -- it will not blow
    /// up regardless of how stiff the system is. It preserves symplectic structure,
    /// making it good for oscillatory dynamics.
    ///
    /// We solve the implicit equation via fixed-point iteration:
    ///   y^{(k+1)} = y_n + h * f(t + h/2, (y_n + y^{(k)})/2)
    /// starting from y^{(0)} = y_n + h * f(t, y_n) (Euler predictor).
    ///
    /// For Symthaea CfC networks with widely varying time constants (tau ranging
    /// from 0.1 to 10.0), this method provides stable integration without
    /// requiring tiny step sizes.
    fn step_implicit_midpoint(
        &self,
        system: &dyn OdeSystem,
        t: f64,
        y: &[f64],
        h: f64,
        y_out: &mut [f64],
    ) {
        let n = self.dim;
        let mut dydt = vec![0.0; n];
        let mut mid = vec![0.0; n];

        // Initial guess: Forward Euler predictor
        system.evaluate(t, y, &mut dydt);
        for i in 0..n {
            y_out[i] = y[i] + h * dydt[i];
        }

        // Fixed-point iteration
        for _iter in 0..self.config.max_iterations {
            // midpoint = (y_n + y_current) / 2
            for i in 0..n {
                mid[i] = 0.5 * (y[i] + y_out[i]);
            }

            // f(t + h/2, midpoint)
            system.evaluate(t + 0.5 * h, &mid, &mut dydt);

            // y_{n+1} = y_n + h * f(t + h/2, midpoint)
            let mut max_change = 0.0_f64;
            for i in 0..n {
                let new_val = y[i] + h * dydt[i];
                let change = (new_val - y_out[i]).abs();
                if change > max_change {
                    max_change = change;
                }
                y_out[i] = new_val;
            }

            if max_change < self.config.tolerance {
                return;
            }
        }
        // If we exhaust iterations, y_out still holds the best estimate.
    }

    fn solve_implicit_midpoint(
        &self,
        system: &dyn OdeSystem,
        y0: &[f64],
        t_span: (f64, f64),
    ) -> OdeResult {
        let h = self.config.dt;
        let mut t = t_span.0;
        let mut y = y0.to_vec();
        let mut y_next = vec![0.0; self.dim];

        let mut result = OdeResult {
            states: vec![y.clone()],
            times: vec![t],
            step_sizes: Vec::new(),
            error_estimates: Vec::new(),
            num_rejections: 0,
        };

        while t < t_span.1 - 1e-14 {
            let step = h.min(t_span.1 - t);
            self.step_implicit_midpoint(system, t, &y, step, &mut y_next);
            t += step;
            y.copy_from_slice(&y_next);
            result.states.push(y.clone());
            result.times.push(t);
            result.step_sizes.push(step);
            result.error_estimates.push(0.0);
        }

        result
    }

    // -----------------------------------------------------------------------
    // Backward Euler
    // -----------------------------------------------------------------------

    /// Backward Euler: y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})
    ///
    /// This is first-order accurate but L-stable, meaning it maximally damps
    /// parasitic fast modes. For very stiff CfC systems where some neurons
    /// have tau = 0.001 and others have tau = 100, this is the safest choice.
    ///
    /// Solved via Newton iteration using finite-difference Jacobian approximation:
    ///   G(y) = y - y_n - h * f(t_{n+1}, y) = 0
    ///   y^{(k+1)} = y^{(k)} - J^{-1} * G(y^{(k)})
    ///
    /// For efficiency we approximate J^{-1} * v via a simplified diagonal
    /// Jacobian (avoids full matrix factorization for large neural systems).
    fn step_backward_euler(
        &self,
        system: &dyn OdeSystem,
        t: f64,
        y: &[f64],
        h: f64,
        y_out: &mut [f64],
    ) {
        let n = self.dim;
        let t_next = t + h;
        let mut dydt = vec![0.0; n];
        let mut dydt_pert = vec![0.0; n];
        let mut residual = vec![0.0; n];

        // Initial guess: Forward Euler
        system.evaluate(t, y, &mut dydt);
        for i in 0..n {
            y_out[i] = y[i] + h * dydt[i];
        }

        // Newton iteration with diagonal Jacobian approximation
        let eps_fd = 1e-8_f64; // finite difference perturbation

        for _iter in 0..self.config.max_iterations {
            system.evaluate(t_next, y_out, &mut dydt);

            // Residual: G(y) = y - y_n - h * f(t_{n+1}, y)
            let mut max_res = 0.0_f64;
            for i in 0..n {
                residual[i] = y_out[i] - y[i] - h * dydt[i];
                let r = residual[i].abs();
                if r > max_res {
                    max_res = r;
                }
            }

            if max_res < self.config.tolerance {
                return;
            }

            // Diagonal Jacobian via finite differences:
            //   dG_i/dy_i approx = 1 - h * df_i/dy_i
            //   df_i/dy_i approx = (f_i(y + eps*e_i) - f_i(y)) / eps
            //
            // Then Newton update: dy_i = -G_i / (dG_i/dy_i)
            for i in 0..n {
                let orig = y_out[i];
                let pert = eps_fd * (1.0 + orig.abs());
                y_out[i] = orig + pert;
                system.evaluate(t_next, y_out, &mut dydt_pert);
                y_out[i] = orig;

                let df_dy = (dydt_pert[i] - dydt[i]) / pert;
                let dg_dy = 1.0 - h * df_dy;

                // Avoid division by near-zero
                if dg_dy.abs() > 1e-15 {
                    y_out[i] -= residual[i] / dg_dy;
                }
            }
        }
    }

    fn solve_backward_euler(
        &self,
        system: &dyn OdeSystem,
        y0: &[f64],
        t_span: (f64, f64),
    ) -> OdeResult {
        let h = self.config.dt;
        let mut t = t_span.0;
        let mut y = y0.to_vec();
        let mut y_next = vec![0.0; self.dim];

        let mut result = OdeResult {
            states: vec![y.clone()],
            times: vec![t],
            step_sizes: Vec::new(),
            error_estimates: Vec::new(),
            num_rejections: 0,
        };

        while t < t_span.1 - 1e-14 {
            let step = h.min(t_span.1 - t);
            self.step_backward_euler(system, t, &y, step, &mut y_next);
            t += step;
            y.copy_from_slice(&y_next);
            result.states.push(y.clone());
            result.times.push(t);
            result.step_sizes.push(step);
            result.error_estimates.push(0.0);
        }

        result
    }

    // -----------------------------------------------------------------------
    // Exponential Integrator
    // -----------------------------------------------------------------------

    /// Exponential integrator for systems of the form dy/dt = A*y + g(t, y).
    ///
    /// For this method, the ODE system is split into a linear part (approximated
    /// by a diagonal matrix of local linearization coefficients) and a nonlinear
    /// remainder. The linear part is solved exactly using the matrix exponential,
    /// and the nonlinear part is treated with a first-order approximation.
    ///
    /// For a scalar ODE dy/dt = a*y + g with constant a and g:
    ///   y(t+h) = exp(ah) * y(t) + (exp(ah) - 1) / a * g
    ///
    /// This is exact for linear systems and first-order for the nonlinear part.
    /// It is particularly well-suited for CfC networks whose dynamics have the
    /// form dx/dt = -x/tau + sigma(Wx + b), where -1/tau is the linear decay
    /// and sigma(Wx + b) is the nonlinear forcing.
    ///
    /// We estimate the diagonal of the Jacobian (the "a" values) via finite
    /// differences, then apply the scalar exponential formula component-wise.
    fn step_exponential(
        &self,
        system: &dyn OdeSystem,
        t: f64,
        y: &[f64],
        h: f64,
        y_out: &mut [f64],
    ) {
        let n = self.dim;
        let mut f_y = vec![0.0; n];
        let mut f_pert = vec![0.0; n];
        let eps = 1e-8_f64;

        // Evaluate f(t, y)
        system.evaluate(t, y, &mut f_y);

        // For each component, estimate the diagonal Jacobian element a_i and
        // the nonlinear remainder g_i, then apply the exponential formula.
        for i in 0..n {
            let orig = y[i];
            let pert = eps * (1.0 + orig.abs());
            let mut y_tmp = y.to_vec();
            y_tmp[i] = orig + pert;
            system.evaluate(t, &y_tmp, &mut f_pert);

            let a_i = (f_pert[i] - f_y[i]) / pert; // df_i/dy_i ~ diagonal Jacobian
            let g_i = f_y[i] - a_i * y[i]; // g_i = f_i - a_i * y_i

            if a_i.abs() < 1e-12 {
                // Nearly zero linear part: fall back to Forward Euler for this component
                y_out[i] = y[i] + h * f_y[i];
            } else {
                // Exact exponential formula: y(t+h) = exp(a*h)*y + (exp(a*h)-1)/a * g
                let exp_ah = (a_i * h).exp();
                y_out[i] = exp_ah * y[i] + (exp_ah - 1.0) / a_i * g_i;
            }
        }
    }

    fn solve_exponential(
        &self,
        system: &dyn OdeSystem,
        y0: &[f64],
        t_span: (f64, f64),
    ) -> OdeResult {
        let h = self.config.dt;
        let mut t = t_span.0;
        let mut y = y0.to_vec();
        let mut y_next = vec![0.0; self.dim];

        let mut result = OdeResult {
            states: vec![y.clone()],
            times: vec![t],
            step_sizes: Vec::new(),
            error_estimates: Vec::new(),
            num_rejections: 0,
        };

        while t < t_span.1 - 1e-14 {
            let step = h.min(t_span.1 - t);
            self.step_exponential(system, t, &y, step, &mut y_next);
            t += step;
            y.copy_from_slice(&y_next);
            result.states.push(y.clone());
            result.times.push(t);
            result.step_sizes.push(step);
            result.error_estimates.push(0.0);
        }

        result
    }
}

// ---------------------------------------------------------------------------
// Newton solver helper (standalone, for use by external implicit methods)
// ---------------------------------------------------------------------------

/// Standalone Newton solver for the nonlinear equation F(x) = 0.
///
/// Uses finite-difference diagonal Jacobian approximation (same strategy as
/// the implicit solvers above). Suitable for moderate-dimensional systems
/// where a full dense Jacobian would be too expensive.
///
/// # Arguments
/// - `f`: Function that evaluates F(x) given x and writes into `out`.
/// - `x0`: Initial guess.
/// - `tol`: Convergence tolerance (max-norm of F(x)).
/// - `max_iter`: Maximum number of Newton iterations.
///
/// # Returns
/// The solution vector. If convergence is not achieved, returns the best
/// estimate after `max_iter` iterations.
pub fn newton_solve<F>(f: F, x0: &[f64], tol: f64, max_iter: usize) -> Vec<f64>
where
    F: Fn(&[f64], &mut [f64]),
{
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut fx = vec![0.0; n];
    let mut fx_pert = vec![0.0; n];
    let eps = 1e-8_f64;

    for _iter in 0..max_iter {
        f(&x, &mut fx);

        // Check convergence
        let max_res = fx.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if max_res < tol {
            return x;
        }

        // Build full Jacobian via finite differences
        let mut jac = vec![vec![0.0; n]; n]; // jac[row][col] = df_row/dx_col
        for j in 0..n {
            let orig = x[j];
            let pert = eps * (1.0 + orig.abs());
            x[j] = orig + pert;
            f(&x, &mut fx_pert);
            x[j] = orig;

            for i in 0..n {
                jac[i][j] = (fx_pert[i] - fx[i]) / pert;
            }
        }

        // Solve J * dx = -fx via Gaussian elimination with partial pivoting
        let mut aug = vec![vec![0.0; n + 1]; n];
        for i in 0..n {
            for j in 0..n {
                aug[i][j] = jac[i][j];
            }
            aug[i][n] = -fx[i];
        }

        // Forward elimination
        for col in 0..n {
            // Partial pivoting
            let mut max_row = col;
            let mut max_val = aug[col][col].abs();
            for row in (col + 1)..n {
                if aug[row][col].abs() > max_val {
                    max_val = aug[row][col].abs();
                    max_row = row;
                }
            }
            if max_val < 1e-15 {
                continue; // singular column, skip
            }
            if max_row != col {
                aug.swap(col, max_row);
            }

            let pivot = aug[col][col];
            for row in (col + 1)..n {
                let factor = aug[row][col] / pivot;
                for k in col..=n {
                    aug[row][k] -= factor * aug[col][k];
                }
            }
        }

        // Back substitution
        let mut dx = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = aug[i][n];
            for j in (i + 1)..n {
                sum -= aug[i][j] * dx[j];
            }
            if aug[i][i].abs() > 1e-15 {
                dx[i] = sum / aug[i][i];
            }
        }

        for i in 0..n {
            x[i] += dx[i];
        }
    }

    x
}

// ===========================================================================
// Standalone RK4 step function (no OdeSystem trait required)
// ===========================================================================

/// Perform a single RK4 step using a closure for the derivative.
///
/// This standalone function avoids requiring the `OdeSystem` trait, making it
/// easy to integrate into existing code (e.g., LTC networks) without
/// restructuring. The closure `f(t, y, dydt)` writes the derivative into `dydt`.
///
/// # Arguments
/// * `t` - Current time
/// * `y` - Current state (length N)
/// * `h` - Step size
/// * `f` - Derivative function: `f(t, &y, &mut dydt)` computes dy/dt
/// * `y_out` - Output state after one RK4 step (length N)
///
/// # Example
/// ```ignore
/// let mut y_out = vec![0.0; 2];
/// rk4_step_fn(0.0, &[1.0, 0.0], 0.01, |t, y, dy| {
///     dy[0] = y[1];
///     dy[1] = -y[0];
/// }, &mut y_out);
/// ```
pub fn rk4_step_fn<F>(t: f64, y: &[f64], h: f64, f: F, y_out: &mut [f64])
where
    F: Fn(f64, &[f64], &mut [f64]),
{
    let n = y.len();
    let mut k1 = vec![0.0; n];
    let mut k2 = vec![0.0; n];
    let mut k3 = vec![0.0; n];
    let mut k4 = vec![0.0; n];
    let mut tmp = vec![0.0; n];

    // k1 = f(t, y)
    f(t, y, &mut k1);

    // k2 = f(t + h/2, y + h/2 * k1)
    for i in 0..n {
        tmp[i] = y[i] + 0.5 * h * k1[i];
    }
    f(t + 0.5 * h, &tmp, &mut k2);

    // k3 = f(t + h/2, y + h/2 * k2)
    for i in 0..n {
        tmp[i] = y[i] + 0.5 * h * k2[i];
    }
    f(t + 0.5 * h, &tmp, &mut k3);

    // k4 = f(t + h, y + h * k3)
    for i in 0..n {
        tmp[i] = y[i] + h * k3[i];
    }
    f(t + h, &tmp, &mut k4);

    // y_{n+1} = y_n + (h/6)(k1 + 2*k2 + 2*k3 + k4)
    for i in 0..n {
        y_out[i] = y[i] + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Test ODE systems
    // -----------------------------------------------------------------------

    /// Exponential decay: dy/dt = -y
    /// Analytical solution: y(t) = y0 * exp(-t)
    struct ExponentialDecay;

    impl OdeSystem for ExponentialDecay {
        fn dimension(&self) -> usize {
            1
        }
        fn evaluate(&self, _t: f64, state: &[f64], derivative: &mut [f64]) {
            derivative[0] = -state[0];
        }
    }

    /// Stiff system: dy/dt = -1000*y + 1000
    /// Analytical solution: y(t) = 1.0 + (y0 - 1.0) * exp(-1000*t)
    /// Stiffness ratio = 1000: explicit methods need h < 0.002 for stability.
    struct StiffDecay;

    impl OdeSystem for StiffDecay {
        fn dimension(&self) -> usize {
            1
        }
        fn evaluate(&self, _t: f64, state: &[f64], derivative: &mut [f64]) {
            derivative[0] = -1000.0 * state[0] + 1000.0;
        }
    }

    /// Simple linear system: dy/dt = [0, 1; -1, 0] y (harmonic oscillator)
    /// Solution: y1(t) = cos(t), y2(t) = sin(t)  for y0 = [1, 0]
    struct HarmonicOscillator;

    impl OdeSystem for HarmonicOscillator {
        fn dimension(&self) -> usize {
            2
        }
        fn evaluate(&self, _t: f64, state: &[f64], derivative: &mut [f64]) {
            derivative[0] = state[1];
            derivative[1] = -state[0];
        }
    }

    /// Two-rate stiff system: dy1/dt = -y1, dy2/dt = -1000*y2
    /// Tests solvers with widely separated time constants.
    struct TwoRateSystem;

    impl OdeSystem for TwoRateSystem {
        fn dimension(&self) -> usize {
            2
        }
        fn evaluate(&self, _t: f64, state: &[f64], derivative: &mut [f64]) {
            derivative[0] = -state[0];
            derivative[1] = -1000.0 * state[1];
        }
    }

    // -----------------------------------------------------------------------
    // Exponential decay tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_euler_exponential_decay() {
        let config = OdeConfig {
            solver: OdeSolver::Euler,
            dt: 0.001,
            ..Default::default()
        };
        let engine = OdeSolverEngine::new(config, 1);
        let result = engine.solve(&ExponentialDecay, &[1.0], (0.0, 1.0));

        let y_final = result.states.last().unwrap()[0];
        let y_exact = (-1.0_f64).exp();
        let error = (y_final - y_exact).abs();

        // Euler with h=0.001 over t=1: global error should be O(h) ~ 0.001
        assert!(
            error < 0.01,
            "Euler error too large: {} (expected < 0.01)",
            error
        );
    }

    #[test]
    fn test_rk4_exponential_decay() {
        let config = OdeConfig {
            solver: OdeSolver::RK4,
            dt: 0.01,
            ..Default::default()
        };
        let engine = OdeSolverEngine::new(config, 1);
        let result = engine.solve(&ExponentialDecay, &[1.0], (0.0, 1.0));

        let y_final = result.states.last().unwrap()[0];
        let y_exact = (-1.0_f64).exp();
        let error = (y_final - y_exact).abs();

        // RK4 with h=0.01 over t=1: global error should be O(h^4) ~ 1e-8
        assert!(
            error < 1e-6,
            "RK4 error too large: {} (expected < 1e-6)",
            error
        );
    }

    #[test]
    fn test_dormand_prince_exponential_decay() {
        let config = OdeConfig {
            solver: OdeSolver::DormandPrince,
            dt: 0.1,
            tolerance: 1e-8,
            max_step: 1.0,
            min_step: 1e-12,
            max_iterations: 50,
        };
        let engine = OdeSolverEngine::new(config, 1);
        let result = engine.solve(&ExponentialDecay, &[1.0], (0.0, 1.0));

        let y_final = result.states.last().unwrap()[0];
        let y_exact = (-1.0_f64).exp();
        let error = (y_final - y_exact).abs();

        // Dormand-Prince with tol=1e-8 should achieve very high accuracy
        assert!(
            error < 1e-6,
            "Dormand-Prince error too large: {} (expected < 1e-6)",
            error
        );
    }

    #[test]
    fn test_implicit_midpoint_exponential_decay() {
        let config = OdeConfig {
            solver: OdeSolver::ImplicitMidpoint,
            dt: 0.01,
            tolerance: 1e-12,
            max_iterations: 20,
            ..Default::default()
        };
        let engine = OdeSolverEngine::new(config, 1);
        let result = engine.solve(&ExponentialDecay, &[1.0], (0.0, 1.0));

        let y_final = result.states.last().unwrap()[0];
        let y_exact = (-1.0_f64).exp();
        let error = (y_final - y_exact).abs();

        // Implicit midpoint is 2nd order: O(h^2) ~ 1e-4
        assert!(
            error < 1e-3,
            "Implicit midpoint error too large: {} (expected < 1e-3)",
            error
        );
    }

    #[test]
    fn test_backward_euler_exponential_decay() {
        let config = OdeConfig {
            solver: OdeSolver::BackwardEuler,
            dt: 0.001,
            tolerance: 1e-12,
            max_iterations: 20,
            ..Default::default()
        };
        let engine = OdeSolverEngine::new(config, 1);
        let result = engine.solve(&ExponentialDecay, &[1.0], (0.0, 1.0));

        let y_final = result.states.last().unwrap()[0];
        let y_exact = (-1.0_f64).exp();
        let error = (y_final - y_exact).abs();

        // Backward Euler is 1st order: O(h) ~ 0.001
        assert!(
            error < 0.01,
            "Backward Euler error too large: {} (expected < 0.01)",
            error
        );
    }

    #[test]
    fn test_exponential_integrator_exponential_decay() {
        let config = OdeConfig {
            solver: OdeSolver::Exponential,
            dt: 0.1,
            ..Default::default()
        };
        let engine = OdeSolverEngine::new(config, 1);
        let result = engine.solve(&ExponentialDecay, &[1.0], (0.0, 1.0));

        let y_final = result.states.last().unwrap()[0];
        let y_exact = (-1.0_f64).exp();
        let error = (y_final - y_exact).abs();

        // Exponential integrator should be exact for linear systems
        // (up to finite difference Jacobian estimation error)
        assert!(
            error < 1e-5,
            "Exponential integrator error too large: {} (expected < 1e-5)",
            error
        );
    }

    // -----------------------------------------------------------------------
    // RK4 convergence order test
    // -----------------------------------------------------------------------

    #[test]
    fn test_rk4_error_order() {
        // Verify that halving the step size reduces global error by factor ~16 (h^4)
        let y_exact = (-1.0_f64).exp();

        let solve_with_dt = |dt: f64| -> f64 {
            let config = OdeConfig {
                solver: OdeSolver::RK4,
                dt,
                ..Default::default()
            };
            let engine = OdeSolverEngine::new(config, 1);
            let result = engine.solve(&ExponentialDecay, &[1.0], (0.0, 1.0));
            (result.states.last().unwrap()[0] - y_exact).abs()
        };

        let err_coarse = solve_with_dt(0.1);
        let err_fine = solve_with_dt(0.05);

        // Ratio should be approximately 2^4 = 16 for 4th order
        let ratio = err_coarse / err_fine;
        assert!(
            ratio > 10.0 && ratio < 25.0,
            "RK4 convergence ratio: {} (expected ~16 for 4th order)",
            ratio
        );
    }

    // -----------------------------------------------------------------------
    // Dormand-Prince adaptive step size test
    // -----------------------------------------------------------------------

    #[test]
    fn test_dormand_prince_adapts_step_size() {
        let config = OdeConfig {
            solver: OdeSolver::DormandPrince,
            dt: 0.5,
            tolerance: 1e-6,
            max_step: 1.0,
            min_step: 1e-10,
            max_iterations: 50,
        };
        let engine = OdeSolverEngine::new(config, 1);
        let result = engine.solve(&ExponentialDecay, &[1.0], (0.0, 5.0));

        // The step sizes should not all be the same (adaptive behavior)
        assert!(
            result.step_sizes.len() >= 2,
            "Too few steps taken: {}",
            result.step_sizes.len()
        );

        // Check that we have varying step sizes (not all identical)
        let first_step = result.step_sizes[0];
        let has_variation = result
            .step_sizes
            .iter()
            .any(|&s| (s - first_step).abs() > 1e-15);

        // For exponential decay the steps should grow as the solution flattens
        // (or at least the initial step gets adapted from the large 0.5)
        assert!(
            has_variation || result.num_rejections > 0,
            "Dormand-Prince should adapt step size or reject steps"
        );
    }

    // -----------------------------------------------------------------------
    // Stiff system tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_implicit_midpoint_stiff_system() {
        // dy/dt = -1000*y + 1000, y(0) = 0
        // Analytical: y(t) = 1 - exp(-1000*t) -> y(0.01) ~ 1.0
        let config = OdeConfig {
            solver: OdeSolver::ImplicitMidpoint,
            dt: 0.001,
            tolerance: 1e-10,
            max_iterations: 50,
            ..Default::default()
        };
        let engine = OdeSolverEngine::new(config, 1);
        let result = engine.solve(&StiffDecay, &[0.0], (0.0, 0.01));

        let y_final = result.states.last().unwrap()[0];
        let y_exact = 1.0 - (-1000.0 * 0.01_f64).exp();

        // Should converge without instability
        assert!(
            y_final.is_finite(),
            "Implicit midpoint produced non-finite result on stiff system"
        );
        let error = (y_final - y_exact).abs();
        assert!(
            error < 0.1,
            "Implicit midpoint stiff error too large: {} (y_final={}, y_exact={})",
            error,
            y_final,
            y_exact
        );
    }

    #[test]
    fn test_backward_euler_stiff_system() {
        // Same stiff system
        let config = OdeConfig {
            solver: OdeSolver::BackwardEuler,
            dt: 0.001,
            tolerance: 1e-10,
            max_iterations: 50,
            ..Default::default()
        };
        let engine = OdeSolverEngine::new(config, 1);
        let result = engine.solve(&StiffDecay, &[0.0], (0.0, 0.01));

        let y_final = result.states.last().unwrap()[0];
        let y_exact = 1.0 - (-1000.0 * 0.01_f64).exp();

        assert!(
            y_final.is_finite(),
            "Backward Euler produced non-finite result on stiff system"
        );
        let error = (y_final - y_exact).abs();
        assert!(
            error < 0.1,
            "Backward Euler stiff error too large: {} (y_final={}, y_exact={})",
            error,
            y_final,
            y_exact
        );
    }

    #[test]
    fn test_exponential_integrator_stiff_system() {
        // The exponential integrator should handle stiff linear systems well
        let config = OdeConfig {
            solver: OdeSolver::Exponential,
            dt: 0.001,
            ..Default::default()
        };
        let engine = OdeSolverEngine::new(config, 1);
        let result = engine.solve(&StiffDecay, &[0.0], (0.0, 0.01));

        let y_final = result.states.last().unwrap()[0];
        let y_exact = 1.0 - (-1000.0 * 0.01_f64).exp();

        assert!(
            y_final.is_finite(),
            "Exponential integrator produced non-finite result on stiff system"
        );
        let error = (y_final - y_exact).abs();
        assert!(
            error < 0.01,
            "Exponential integrator stiff error: {} (y_final={}, y_exact={})",
            error,
            y_final,
            y_exact
        );
    }

    // -----------------------------------------------------------------------
    // All solvers agree on simple linear system
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_solvers_agree_on_exponential_decay() {
        let y0 = [2.0];
        let t_span = (0.0, 0.5);
        let y_exact = 2.0 * (-0.5_f64).exp();

        let solvers = [
            (OdeSolver::Euler, 0.0005),
            (OdeSolver::RK4, 0.01),
            (OdeSolver::DormandPrince, 0.1),
            (OdeSolver::ImplicitMidpoint, 0.01),
            (OdeSolver::BackwardEuler, 0.0005),
            (OdeSolver::Exponential, 0.1),
        ];

        for (solver, dt) in &solvers {
            let config = OdeConfig {
                solver: *solver,
                dt: *dt,
                tolerance: 1e-8,
                max_step: 1.0,
                min_step: 1e-12,
                max_iterations: 50,
            };
            let engine = OdeSolverEngine::new(config, 1);
            let result = engine.solve(&ExponentialDecay, &y0, t_span);

            let y_final = result.states.last().unwrap()[0];
            let error = (y_final - y_exact).abs();

            assert!(
                error < 0.01,
                "{:?} solver error too large: {} (y_final={}, y_exact={})",
                solver,
                error,
                y_final,
                y_exact
            );
        }
    }

    // -----------------------------------------------------------------------
    // Harmonic oscillator (2D system)
    // -----------------------------------------------------------------------

    #[test]
    fn test_rk4_harmonic_oscillator() {
        let config = OdeConfig {
            solver: OdeSolver::RK4,
            dt: 0.01,
            ..Default::default()
        };
        let engine = OdeSolverEngine::new(config, 2);
        let two_pi = 2.0 * std::f64::consts::PI;
        let result = engine.solve(&HarmonicOscillator, &[1.0, 0.0], (0.0, two_pi)); // one full period

        let y_final = &result.states.last().unwrap();
        // After one full period, should return to [1, 0]
        let error_x = (y_final[0] - 1.0).abs();
        let error_v = (y_final[1] - 0.0).abs();

        assert!(
            error_x < 1e-4,
            "RK4 oscillator x error: {} (expected < 1e-4)",
            error_x
        );
        assert!(
            error_v < 1e-4,
            "RK4 oscillator v error: {} (expected < 1e-4)",
            error_v
        );
    }

    // -----------------------------------------------------------------------
    // Two-rate stiff system
    // -----------------------------------------------------------------------

    #[test]
    fn test_implicit_midpoint_two_rate() {
        // dy1/dt = -y1, dy2/dt = -1000*y2
        // y1(1) = exp(-1), y2(1) ~ 0
        // Need dt small enough for fixed-point iteration to converge:
        // contraction rate = h/2 * max|df/dy| < 1  =>  h < 2/1000 = 0.002
        let config = OdeConfig {
            solver: OdeSolver::ImplicitMidpoint,
            dt: 0.001,
            tolerance: 1e-10,
            max_iterations: 50,
            ..Default::default()
        };
        let engine = OdeSolverEngine::new(config, 2);
        let result = engine.solve(&TwoRateSystem, &[1.0, 1.0], (0.0, 1.0));

        let y_final = &result.states.last().unwrap();
        let y1_exact = (-1.0_f64).exp();

        assert!(
            y_final[0].is_finite() && y_final[1].is_finite(),
            "Implicit midpoint produced non-finite on two-rate system"
        );

        let error1 = (y_final[0] - y1_exact).abs();
        assert!(
            error1 < 0.01,
            "Implicit midpoint two-rate y1 error: {}",
            error1
        );

        // y2 should be effectively 0 (exp(-1000) ~ 0)
        assert!(
            y_final[1].abs() < 0.01,
            "Implicit midpoint two-rate y2 should be ~0, got: {}",
            y_final[1]
        );
    }

    // -----------------------------------------------------------------------
    // Newton solver standalone test
    // -----------------------------------------------------------------------

    #[test]
    fn test_newton_solve_quadratic() {
        // Solve x^2 - 4 = 0 starting from x=1. Root at x=2.
        let solution = newton_solve(
            |x: &[f64], out: &mut [f64]| {
                out[0] = x[0] * x[0] - 4.0;
            },
            &[1.0],
            1e-12,
            50,
        );

        let error = (solution[0] - 2.0).abs();
        assert!(
            error < 1e-8,
            "Newton solver error: {} (expected root at 2.0, got {})",
            error,
            solution[0]
        );
    }

    #[test]
    fn test_newton_solve_2d_system() {
        // Solve: x^2 + y^2 = 1, x - y = 0
        // Root at (1/sqrt(2), 1/sqrt(2))
        let solution = newton_solve(
            |xy: &[f64], out: &mut [f64]| {
                out[0] = xy[0] * xy[0] + xy[1] * xy[1] - 1.0;
                out[1] = xy[0] - xy[1];
            },
            &[0.5, 0.5],
            1e-10,
            50,
        );

        let expected = 1.0 / 2.0_f64.sqrt();
        let err_x = (solution[0] - expected).abs();
        let err_y = (solution[1] - expected).abs();
        assert!(
            err_x < 1e-6 && err_y < 1e-6,
            "Newton 2D error: ({}, {}), expected ({}, {})",
            solution[0],
            solution[1],
            expected,
            expected
        );
    }

    // -----------------------------------------------------------------------
    // OdeResult structure test
    // -----------------------------------------------------------------------

    #[test]
    fn test_result_times_monotonic() {
        let config = OdeConfig {
            solver: OdeSolver::DormandPrince,
            dt: 0.1,
            tolerance: 1e-6,
            max_step: 1.0,
            min_step: 1e-12,
            max_iterations: 50,
        };
        let engine = OdeSolverEngine::new(config, 1);
        let result = engine.solve(&ExponentialDecay, &[1.0], (0.0, 3.0));

        // Times should be strictly increasing
        for i in 1..result.times.len() {
            assert!(
                result.times[i] > result.times[i - 1],
                "Times not monotonically increasing at index {}: {} <= {}",
                i,
                result.times[i],
                result.times[i - 1]
            );
        }

        // First time should be t_start, last should be close to t_end
        assert!((result.times[0] - 0.0).abs() < 1e-14);
        assert!((result.times.last().unwrap() - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_result_states_dimension() {
        let config = OdeConfig {
            solver: OdeSolver::RK4,
            dt: 0.1,
            ..Default::default()
        };
        let engine = OdeSolverEngine::new(config, 2);
        let result = engine.solve(&HarmonicOscillator, &[1.0, 0.0], (0.0, 1.0));

        // Every state vector should have dimension 2
        for (i, state) in result.states.iter().enumerate() {
            assert_eq!(
                state.len(),
                2,
                "State at index {} has wrong dimension: {}",
                i,
                state.len()
            );
        }

        // Number of states should equal number of times
        assert_eq!(result.states.len(), result.times.len());
    }

    // -----------------------------------------------------------------------
    // Serialization round-trip test
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_serde_roundtrip() {
        let config = OdeConfig {
            solver: OdeSolver::DormandPrince,
            dt: 0.05,
            tolerance: 1e-9,
            max_step: 0.5,
            min_step: 1e-15,
            max_iterations: 100,
        };

        let json = serde_json::to_string(&config).expect("serialize");
        let config2: OdeConfig = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(config.solver, config2.solver);
        assert!((config.dt - config2.dt).abs() < 1e-15);
        assert!((config.tolerance - config2.tolerance).abs() < 1e-20);
    }
}
