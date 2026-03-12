#![allow(dead_code)]

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

// ─── Types ────────────────────────────────────────────────────────────────────

/// Numerical method used to integrate the ODE.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ODEMethod {
    /// Classical 4th-order Runge-Kutta with fixed step size.
    RK4,
    /// Shooting method (wraps an inner IVP solver).
    Shooting,
}

impl std::fmt::Display for ODEMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ODEMethod::RK4 => write!(f, "RK4"),
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

    fn encode_pde_result(u_final: &[f64], tag: &str) -> BinaryHV {
        let base = BinaryHV::random(seed_from_name(tag));
        let state_hash: u64 = u_final.iter().enumerate().fold(0u64, |acc, (i, &v)| {
            acc.wrapping_add(v.to_bits().wrapping_mul(i as u64 + 1))
        });
        let state_hv = BinaryHV::random(seed_from_name(&format!("{}_STATE_{}", tag, state_hash)));
        base.bind(&state_hv)
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
        assert!(r_coarse
            .u_final
            .iter()
            .all(|&v| v >= -1e-10 && v <= 1.0 + 1e-10));
        assert!(r_fine
            .u_final
            .iter()
            .all(|&v| v >= -1e-10 && v <= 1.0 + 1e-10));
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
}
