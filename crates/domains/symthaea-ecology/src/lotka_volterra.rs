// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! The Lotka-Volterra predator-prey model (analytic ODE form).

use crate::error::{ModelError, require_positive};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LotkaVolterra {
    pub alpha: f64,
    pub beta: f64,
    pub delta: f64,
    pub gamma: f64,
}

impl LotkaVolterra {
    pub fn try_new(alpha: f64, beta: f64, delta: f64, gamma: f64) -> Result<Self, ModelError> {
        let model = Self {
            alpha,
            beta,
            delta,
            gamma,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        require_positive("alpha", self.alpha)?;
        require_positive("beta", self.beta)?;
        require_positive("delta", self.delta)?;
        require_positive("gamma", self.gamma)?;
        Ok(())
    }

    pub fn equilibrium(&self) -> (f64, f64) {
        (self.gamma / self.delta, self.alpha / self.beta)
    }

    pub fn try_equilibrium(&self) -> Result<(f64, f64), ModelError> {
        self.validate()?;
        Ok(self.equilibrium())
    }

    pub fn derivatives(&self, x: f64, y: f64) -> (f64, f64) {
        (
            self.alpha * x - self.beta * x * y,
            self.delta * x * y - self.gamma * y,
        )
    }

    /// Jacobian matrix at `(x, y)`.
    pub fn jacobian(&self, x: f64, y: f64) -> [[f64; 2]; 2] {
        [
            [self.alpha - self.beta * y, -self.beta * x],
            [self.delta * y, self.delta * x - self.gamma],
        ]
    }

    /// Small-oscillation angular frequency around the coexistence equilibrium.
    pub fn equilibrium_angular_frequency(&self) -> f64 {
        (self.alpha * self.gamma).sqrt()
    }

    pub fn conserved_quantity(&self, x: f64, y: f64) -> f64 {
        self.delta * x - self.gamma * x.ln() + self.beta * y - self.alpha * y.ln()
    }

    pub fn try_conserved_quantity(&self, x: f64, y: f64) -> Result<f64, ModelError> {
        self.validate()?;
        require_positive("prey_population", x)?;
        require_positive("predator_population", y)?;
        Ok(self.conserved_quantity(x, y))
    }

    pub fn simulate(&self, x0: f64, y0: f64, dt: f64, steps: usize) -> Vec<(f64, f64)> {
        crate::integration::simulate_pair_unchecked(x0, y0, dt, steps, |x, y| {
            self.derivatives(x, y)
        })
    }

    pub fn try_simulate(
        &self,
        x0: f64,
        y0: f64,
        dt: f64,
        steps: usize,
    ) -> Result<Vec<(f64, f64)>, ModelError> {
        self.validate()?;
        require_positive("initial_prey", x0)?;
        require_positive("initial_predator", y0)?;
        require_positive("dt", dt)?;
        if steps == 0 {
            return Err(ModelError::ZeroSteps);
        }
        Ok(
            crate::integration::simulate_positive_pair(x0, y0, dt, steps, |x, y| {
                self.derivatives(x, y)
            })?
            .into_iter()
            .map(|sample| (sample.first, sample.second))
            .collect(),
        )
    }

    /// Guarded RK4 trajectory with explicit timestamps and the initial state.
    pub fn try_simulate_timestamped(
        &self,
        x0: f64,
        y0: f64,
        dt: f64,
        steps: usize,
    ) -> Result<Vec<crate::integration::PopulationPairSample>, ModelError> {
        self.validate()?;
        require_positive("initial_prey", x0)?;
        require_positive("initial_predator", y0)?;
        require_positive("dt", dt)?;
        if steps == 0 {
            return Err(ModelError::ZeroSteps);
        }
        crate::integration::simulate_positive_pair(x0, y0, dt, steps, |x, y| self.derivatives(x, y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> LotkaVolterra {
        LotkaVolterra::try_new(1.0, 0.1, 0.075, 1.5).unwrap()
    }

    #[test]
    fn equilibrium_has_zero_derivatives() {
        let m = model();
        let (xs, ys) = m.try_equilibrium().unwrap();
        assert!((xs - 20.0).abs() < 1e-12);
        assert!((ys - 10.0).abs() < 1e-12);
        let (dx, dy) = m.derivatives(xs, ys);
        assert!(dx.abs() < 1e-9 && dy.abs() < 1e-9);
    }

    #[test]
    fn equilibrium_jacobian_has_zero_trace() {
        let m = model();
        let (x, y) = m.equilibrium();
        let j = m.jacobian(x, y);
        assert!((j[0][0] + j[1][1]).abs() < 1e-12);
        assert!((m.equilibrium_angular_frequency() - (1.5_f64).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn conserved_quantity_is_invariant_along_trajectory() {
        let m = model();
        let (x0, y0) = (10.0, 5.0);
        let v0 = m.try_conserved_quantity(x0, y0).unwrap();
        let traj = m.try_simulate(x0, y0, 0.001, 20_000).unwrap();
        for &(x, y) in traj.iter().step_by(1000) {
            assert!((m.conserved_quantity(x, y) - v0).abs() < 1e-3);
        }
    }

    #[test]
    fn checked_simulation_rejects_invalid_inputs() {
        assert!(model().try_simulate(-1.0, 5.0, 0.01, 10).is_err());
        assert!(model().try_simulate(10.0, 5.0, 0.0, 10).is_err());
        assert!(model().try_simulate(10.0, 5.0, 0.01, 0).is_err());
        assert!(matches!(
            model().try_simulate(10.0, 5.0, 100.0, 2),
            Err(ModelError::IntegrationDomainViolation { .. })
        ));
    }

    #[test]
    fn populations_oscillate_and_stay_positive() {
        let traj = model().try_simulate(10.0, 5.0, 0.01, 2000).unwrap();
        assert!(traj.iter().all(|&(x, y)| x > 0.0 && y > 0.0));
        let max_x = traj.iter().map(|&(x, _)| x).fold(f64::MIN, f64::max);
        let min_x = traj.iter().map(|&(x, _)| x).fold(f64::MAX, f64::min);
        assert!(max_x > 10.0 && min_x < 10.0);
    }
    #[test]
    fn timestamped_trajectory_includes_initial_state() {
        let trajectory = model()
            .try_simulate_timestamped(10.0, 5.0, 0.01, 10)
            .unwrap();
        assert_eq!(trajectory.len(), 11);
        assert_eq!(trajectory[0].time, 0.0);
        assert_eq!(trajectory[0].first, 10.0);
        assert!((trajectory[10].time - 0.1).abs() < 1e-12);
    }

    #[test]
    fn guarded_trajectory_rejects_unresolved_step() {
        assert!(matches!(
            model().try_simulate_timestamped(10.0, 5.0, 100.0, 2),
            Err(ModelError::IntegrationDomainViolation { .. })
        ));
    }
}
