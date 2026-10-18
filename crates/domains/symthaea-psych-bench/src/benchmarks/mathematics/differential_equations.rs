// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Differential Equations Assessment.
//!
//! Tests numerical ODE/PDE solvers against problems with known closed-form
//! solutions using the actual `DifferentialEquationsEngine` from symthaea-core.
//!
//! ## Problems
//!
//! 1. **IVP — Exponential Growth**: dy/dt = y, y(0) = 1 → y(t) = eᵗ
//!    Checks RK4 solution at t=1 against e ≈ 2.71828.
//!
//! 2. **BVP — Harmonic Oscillator**: y'' + y = 0 on [0, π/2], with
//!    y(0)=0 and y(π/2)=1. The unique solution is y(t)=sin(t); score the
//!    shooting slope and the full trajectory so a zero solution cannot pass.
//!
//! 3. **PDE — 1-D Heat Equation**: u(x,0)=sin(πx/L) with zero Dirichlet
//!    boundaries. Compare the computed solution with the analytical eigenmode
//!    at the requested time; energy decay remains only a qualitative invariant.
//!
//! Legacy broad smoke-test targets (historical project values, not literature norms):
//! - ivp_relative_error:    < 0.01
//! - bvp_converged_rate:    0.88
//! - pde_energy_decay_rate: 0.95
//!
//! Provisional exact-reference gates (project targets, not literature norms):
//! - BVP slope error and max trajectory absolute error: < 1e-4.
//! - Heat-mode relative L2 error: < 1e-2.

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::{BenchmarkProvenance, PsychBenchmark};
use symthaea_core::hdc::differential_equations::{DifferentialEquationsEngine, ODESystem};

/// Differential Equations Assessment benchmark.
pub struct DifferentialEquationsBenchmark;

struct TrialResult {
    ivp_relative_error: f64,
    bvp_converged: f64,
    bvp_boundary_residual: f64,
    bvp_slope_abs_error: f64,
    bvp_max_abs_error: f64,
    pde_energy_decay: f64,
    pde_relative_l2_error: f64,
}

// ── IVP helpers ──────────────────────────────────────────────────────────────

/// dy/dt = y  (exact solution: y(t) = e^t)
fn exponential_rhs(_t: f64, y: &[f64]) -> Vec<f64> {
    vec![y[0]]
}

// ── BVP helpers ──────────────────────────────────────────────────────────────

/// y'' + y = 0  →  [y, y']' = [y', -y]  (exact solution: y(t) = sin(t))
fn harmonic_oscillator_rhs(_t: f64, y: &[f64]) -> Vec<f64> {
    vec![y[1], -y[0]]
}

// ── PDE helpers ──────────────────────────────────────────────────────────────

/// First Dirichlet eigenmode: u(x,0) = sin(πx/L).
///
/// Its analytical heat-equation solution is
/// u(x,t) = exp(-α(π/L)²t) sin(πx/L), giving an independent reference.
fn sine_mode_ic(n_x: usize, length: f64) -> Vec<f64> {
    let dx = length / (n_x + 1) as f64;
    (1..=n_x)
        .map(|i| (std::f64::consts::PI * i as f64 * dx / length).sin())
        .collect()
}

impl DifferentialEquationsBenchmark {
    fn run_trial(&self, _config: &BenchmarkConfig, _trial_idx: usize) -> TrialResult {
        // ── 1. IVP: Exponential growth dy/dt = y ─────────────────────────
        let ivp_system = ODESystem {
            f: exponential_rhs,
            dim: 1,
        };
        let ivp = DifferentialEquationsEngine::solve_ivp(
            &ivp_system,
            0.0,    // t0
            1.0,    // t_end
            &[1.0], // y0 = 1
            1000,   // steps
        );
        let y_at_1 = ivp.final_state()[0];
        let exact_e = std::f64::consts::E;
        let ivp_relative_error = (y_at_1 - exact_e).abs() / exact_e;

        // ── 2. BVP: Harmonic oscillator y'' + y = 0 on [0, π/2] ─────────
        // y(0)=0, y(π/2)=1 gives the unique nontrivial solution y(t)=sin(t).
        // This rejects the former degenerate zero solution that satisfied
        // y(0)=y(π)=0 with a tiny boundary residual.
        let bvp = DifferentialEquationsEngine::solve_bvp_shooting(
            harmonic_oscillator_rhs,
            0.0,                         // a (left boundary t)
            std::f64::consts::FRAC_PI_2, // b (right boundary t)
            0.0,                         // y(a) = 0
            1.0,                         // y(b) = 1; excludes zero solution
            -10.0,                       // slope_lo bracket
            10.0,                        // slope_hi bracket
            500,                         // IVP steps
        );
        let bvp_max_abs_error = bvp
            .trajectory
            .t_values
            .iter()
            .zip(&bvp.trajectory.y_values)
            .map(|(&t, state)| (state[0] - t.sin()).abs())
            .fold(0.0_f64, f64::max);
        let bvp_boundary_residual = bvp.boundary_residual;
        let bvp_slope_abs_error = (bvp.shooting_slope - 1.0).abs();
        let bvp_converged = if bvp.converged
            && bvp_boundary_residual < 1e-4
            && bvp_slope_abs_error < 1e-4
            && bvp_max_abs_error < 1e-4
        {
            1.0
        } else {
            0.0
        };

        // ── 3. PDE: 1-D Heat Equation — analytical eigenmode comparison ───
        let n_x = 20usize;
        let length = 1.0_f64;
        let alpha = 0.01_f64; // thermal diffusivity
        let t_end = 0.5_f64;
        let u0 = sine_mode_ic(n_x, length);
        let initial_energy: f64 = u0.iter().map(|v| v * v).sum::<f64>().sqrt();

        let pde = DifferentialEquationsEngine::heat_equation_1d(alpha, length, t_end, n_x, &u0);
        let final_energy: f64 = pde.u_final.iter().map(|v| v * v).sum::<f64>().sqrt();
        let decay = (-alpha * (std::f64::consts::PI / length).powi(2) * t_end).exp();
        let exact: Vec<f64> = pde
            .x_grid
            .iter()
            .map(|&x| decay * (std::f64::consts::PI * x / length).sin())
            .collect();
        let exact_norm_squared: f64 = exact.iter().map(|v| v * v).sum();
        let squared_error: f64 = pde
            .u_final
            .iter()
            .zip(&exact)
            .map(|(actual, expected)| (actual - expected).powi(2))
            .sum();
        let pde_relative_l2_error = if exact_norm_squared > 0.0 {
            (squared_error / exact_norm_squared).sqrt()
        } else {
            f64::INFINITY
        };

        // Retain the qualitative invariant, but do not treat it as accuracy.
        let pde_energy_decay = if initial_energy > 0.0
            && final_energy < initial_energy * 0.99
            && final_energy >= 0.0
        {
            1.0
        } else {
            0.0
        };

        TrialResult {
            ivp_relative_error,
            bvp_converged,
            bvp_boundary_residual,
            bvp_slope_abs_error,
            bvp_max_abs_error,
            pde_energy_decay,
            pde_relative_l2_error,
        }
    }
}

impl PsychBenchmark for DifferentialEquationsBenchmark {
    fn name(&self) -> &str {
        "Mathematics::DifferentialEquations"
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Numerical ODE/PDE Solving Assessment",
            citation: "Burden & Faires (2010)",
            year: 2010,
            doi: Some("10.1007/978-0-8176-8259-0"),
        })
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = std::time::Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let mut ivp_errors = Vec::with_capacity(config.trials_per_condition);
        let mut bvp_rates = Vec::with_capacity(config.trials_per_condition);
        let mut bvp_boundary_residuals = Vec::with_capacity(config.trials_per_condition);
        let mut bvp_slope_errors = Vec::with_capacity(config.trials_per_condition);
        let mut bvp_errors = Vec::with_capacity(config.trials_per_condition);
        let mut pde_decays = Vec::with_capacity(config.trials_per_condition);
        let mut pde_errors = Vec::with_capacity(config.trials_per_condition);

        for trial in 0..config.trials_per_condition {
            let r = self.run_trial(config, trial);
            ivp_errors.push(r.ivp_relative_error);
            bvp_rates.push(r.bvp_converged);
            bvp_boundary_residuals.push(r.bvp_boundary_residual);
            bvp_slope_errors.push(r.bvp_slope_abs_error);
            bvp_errors.push(r.bvp_max_abs_error);
            pde_decays.push(r.pde_energy_decay);
            pde_errors.push(r.pde_relative_l2_error);
        }

        result.insert("ivp_relative_error", MetricValue::from_samples(&ivp_errors));
        result.insert("bvp_converged_rate", MetricValue::from_samples(&bvp_rates));
        result.insert(
            "bvp_boundary_residual",
            MetricValue::from_samples(&bvp_boundary_residuals),
        );
        result.insert(
            "bvp_slope_abs_error",
            MetricValue::from_samples(&bvp_slope_errors),
        );
        result.insert(
            "bvp_max_abs_error",
            MetricValue::from_samples(&bvp_errors),
        );
        result.insert(
            "pde_energy_decay_rate",
            MetricValue::from_samples(&pde_decays),
        );
        result.insert(
            "pde_relative_l2_error",
            MetricValue::from_samples(&pde_errors),
        );

        result.conditions = 3; // IVP, BVP, PDE
        result.trials_per_condition = config.trials_per_condition;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> BenchmarkConfig {
        BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 3,
            ..Default::default()
        }
    }

    #[test]
    fn test_diffeq_runs_and_has_metrics() {
        let result = DifferentialEquationsBenchmark.run(&test_config());
        assert!(result.metrics.contains_key("ivp_relative_error"));
        assert!(result.metrics.contains_key("bvp_converged_rate"));
        assert!(result.metrics.contains_key("bvp_boundary_residual"));
        assert!(result.metrics.contains_key("bvp_slope_abs_error"));
        assert!(result.metrics.contains_key("bvp_max_abs_error"));
        assert!(result.metrics.contains_key("pde_energy_decay_rate"));
        assert!(result.metrics.contains_key("pde_relative_l2_error"));
    }

    #[test]
    fn test_diffeq_metrics_finite() {
        let result = DifferentialEquationsBenchmark.run(&test_config());
        for (key, val) in &result.metrics {
            assert!(val.mean.is_finite(), "metric {} is not finite", key);
        }
    }

    #[test]
    fn test_ivp_rk4_accuracy() {
        // RK4 on dy/dt=y should match e^1 to within 1e-7 with 1000 steps
        let system = ODESystem {
            f: exponential_rhs,
            dim: 1,
        };
        let result = DifferentialEquationsEngine::solve_ivp(&system, 0.0, 1.0, &[1.0], 1000);
        let error = (result.final_state()[0] - std::f64::consts::E).abs();
        assert!(
            error < 1e-7,
            "RK4 error {:.2e} exceeds 1e-7 on dy/dt=y",
            error
        );
    }

    #[test]
    fn test_bvp_harmonic_converges() {
        let bvp = DifferentialEquationsEngine::solve_bvp_shooting(
            harmonic_oscillator_rhs,
            0.0,
            std::f64::consts::FRAC_PI_2,
            0.0,
            1.0,
            -10.0,
            10.0,
            500,
        );
        assert!(bvp.converged, "BVP shooting method should converge");
        assert!(
            bvp.boundary_residual < 1e-4,
            "Boundary residual {:.2e} too large",
            bvp.boundary_residual
        );
        assert!(
            (bvp.shooting_slope - 1.0).abs() < 1e-4,
            "Expected nontrivial shooting slope 1, got {:.6}",
            bvp.shooting_slope
        );
        let max_abs_error = bvp
            .trajectory
            .t_values
            .iter()
            .zip(&bvp.trajectory.y_values)
            .map(|(&t, state)| (state[0] - t.sin()).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_abs_error < 1e-4,
            "Trajectory error {:.2e} exceeds 1e-4",
            max_abs_error
        );
    }

    #[test]
    fn test_pde_heat_energy_decays() {
        let n_x = 20;
        let length = 1.0;
        let u0 = sine_mode_ic(n_x, length);
        let initial_energy: f64 = u0.iter().map(|v| v * v).sum::<f64>().sqrt();

        let alpha = 0.01_f64;
        let t_end = 0.5_f64;
        let pde = DifferentialEquationsEngine::heat_equation_1d(alpha, length, t_end, n_x, &u0);
        let final_energy: f64 = pde.u_final.iter().map(|v| v * v).sum::<f64>().sqrt();
        let decay = (-alpha * (std::f64::consts::PI / length).powi(2) * t_end).exp();
        let exact: Vec<f64> = pde
            .x_grid
            .iter()
            .map(|&x| decay * (std::f64::consts::PI * x / length).sin())
            .collect();
        let exact_norm_squared: f64 = exact.iter().map(|v| v * v).sum();
        let squared_error: f64 = pde
            .u_final
            .iter()
            .zip(&exact)
            .map(|(actual, expected)| (actual - expected).powi(2))
            .sum();
        let relative_l2_error = (squared_error / exact_norm_squared).sqrt();
        assert!(
            relative_l2_error < 1e-2,
            "Heat-equation relative L2 error {:.3e} exceeds 1e-2",
            relative_l2_error
        );

        assert!(
            final_energy < initial_energy,
            "Heat equation: final energy ({:.4}) should be < initial ({:.4})",
            final_energy,
            initial_energy
        );
        assert!(
            final_energy >= 0.0,
            "Energy must remain non-negative, got {:.4}",
            final_energy
        );
    }
    #[test]
    fn test_heat_mode_refinement_observed_order() {
        // This tests the convergence trend, not just one hand-picked resolution.
        // The explicit solver chooses dt from dx, so the refinement schedule is
        // coupled; all parameters and resulting step counts remain deterministic.
        let mut relative_errors = Vec::new();
        for n_x in [20usize, 40, 80] {
            let length = 1.0_f64;
            let alpha = 0.01_f64;
            let t_end = 0.5_f64;
            let u0 = sine_mode_ic(n_x, length);
            let pde =
                DifferentialEquationsEngine::heat_equation_1d(alpha, length, t_end, n_x, &u0);
            let decay = (-alpha * (std::f64::consts::PI / length).powi(2) * t_end).exp();
            let exact: Vec<f64> = pde
                .x_grid
                .iter()
                .map(|&x| decay * (std::f64::consts::PI * x / length).sin())
                .collect();
            let exact_norm_squared: f64 = exact.iter().map(|value| value * value).sum();
            let squared_error: f64 = pde
                .u_final
                .iter()
                .zip(&exact)
                .map(|(actual, expected)| (actual - expected).powi(2))
                .sum();
            relative_errors.push((squared_error / exact_norm_squared).sqrt());
        }

        assert!(
            relative_errors[1] < relative_errors[0]
                && relative_errors[2] < relative_errors[1],
            "Heat-mode errors must decrease under refinement: {:?}",
            relative_errors
        );

        let order_20_40 = (relative_errors[0] / relative_errors[1]).log2();
        let order_40_80 = (relative_errors[1] / relative_errors[2]).log2();
        assert!(
            order_20_40 >= 1.7 && order_40_80 >= 1.7,
            "Expected observed order >= 1.7 over both refinement intervals; errors={:?}, orders=({:.3}, {:.3})",
            relative_errors,
            order_20_40,
            order_40_80
        );
    }

}
