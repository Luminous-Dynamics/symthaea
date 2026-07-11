// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! The Lotka-Volterra predator-prey model (analytic ODE form).
//!
//! ```text
//! dx/dt =  α·x − β·x·y      (prey)
//! dy/dt =  δ·x·y − γ·y      (predator)
//! ```
//!
//! This is the closed-form / ODE counterpart to `symthaea-alife`'s *agent-based*
//! predator-prey sim (which derives interaction rates from real agent choices);
//! here the rates are the model parameters.

/// Lotka-Volterra parameters: prey growth `alpha`, predation `beta`, predator
/// gain-per-prey `delta`, predator death `gamma`.
#[derive(Debug, Clone, Copy)]
pub struct LotkaVolterra {
    pub alpha: f64,
    pub beta: f64,
    pub delta: f64,
    pub gamma: f64,
}

impl LotkaVolterra {
    /// The non-trivial coexistence equilibrium `(x*, y*) = (γ/δ, α/β)`, at which
    /// both populations are stationary.
    pub fn equilibrium(&self) -> (f64, f64) {
        (self.gamma / self.delta, self.alpha / self.beta)
    }

    /// The derivatives (dx/dt, dy/dt) at a state.
    pub fn derivatives(&self, x: f64, y: f64) -> (f64, f64) {
        (
            self.alpha * x - self.beta * x * y,
            self.delta * x * y - self.gamma * y,
        )
    }

    /// The conserved quantity `V = δx − γ·ln x + βy − α·ln y`, invariant along
    /// every trajectory (the model's first integral). Requires `x, y > 0`.
    pub fn conserved_quantity(&self, x: f64, y: f64) -> f64 {
        self.delta * x - self.gamma * x.ln() + self.beta * y - self.alpha * y.ln()
    }

    /// Integrate the trajectory with classic RK4 (via `symthaea-numerical`),
    /// returning `(x, y)` at each of `steps` points (excluding the initial
    /// state). Step size is `dt`.
    pub fn simulate(&self, x0: f64, y0: f64, dt: f64, steps: usize) -> Vec<(f64, f64)> {
        let t1 = dt * steps as f64;
        symthaea_numerical::rk4_system(
            |_, s| {
                let (dx, dy) = self.derivatives(s[0], s[1]);
                vec![dx, dy]
            },
            vec![x0, y0],
            0.0,
            t1,
            steps,
        )
        .into_iter()
        .map(|(_, s)| (s[0], s[1]))
        .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> LotkaVolterra {
        LotkaVolterra {
            alpha: 1.0,
            beta: 0.1,
            delta: 0.075,
            gamma: 1.5,
        }
    }

    #[test]
    fn equilibrium_has_zero_derivatives() {
        let m = model();
        let (xs, ys) = m.equilibrium();
        assert!((xs - 20.0).abs() < 1e-12); // γ/δ = 1.5/0.075
        assert!((ys - 10.0).abs() < 1e-12); // α/β = 1.0/0.1
        let (dx, dy) = m.derivatives(xs, ys);
        assert!(dx.abs() < 1e-9 && dy.abs() < 1e-9);
    }

    #[test]
    fn conserved_quantity_is_invariant_along_trajectory() {
        let m = model();
        let (x0, y0) = (10.0, 5.0);
        let v0 = m.conserved_quantity(x0, y0);
        let traj = m.simulate(x0, y0, 0.001, 20_000); // ~20 time units
        // V should stay constant to good precision under RK4.
        for &(x, y) in traj.iter().step_by(1000) {
            assert!((m.conserved_quantity(x, y) - v0).abs() < 1e-3, "V drifted");
        }
    }

    #[test]
    fn populations_oscillate_and_stay_positive() {
        let m = model();
        let traj = m.simulate(10.0, 5.0, 0.01, 2000);
        assert!(traj.iter().all(|&(x, y)| x > 0.0 && y > 0.0));
        // The prey population should both rise above and fall below its start.
        let max_x = traj.iter().map(|&(x, _)| x).fold(f64::MIN, f64::max);
        let min_x = traj.iter().map(|&(x, _)| x).fold(f64::MAX, f64::min);
        assert!(max_x > 10.0 && min_x < 10.0);
    }
}
