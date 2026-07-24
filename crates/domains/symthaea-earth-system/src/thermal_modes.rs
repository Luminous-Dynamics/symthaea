// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Exact linear modes for the two-box surface/deep-ocean climate model.
//!
//! The two-box equations are linear under constant forcing. Exposing their
//! eigenmodes gives an independent oracle for the RK4 implementation and makes
//! the fast mixed-layer and slow deep-ocean timescales explicit.

use crate::error::{ModelError, require_finite, require_non_negative, require_positive};
use crate::transient::{SimulationGrid, TwoBoxClimateModel, TwoBoxState};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoBoxModes {
    /// Less-negative eigenvalue, s⁻¹.
    pub slow_decay_rate: f64,
    /// More-negative eigenvalue, s⁻¹.
    pub fast_decay_rate: f64,
    pub slow_timescale_seconds: f64,
    pub fast_timescale_seconds: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoBoxConvergenceReport {
    pub duration_seconds: f64,
    pub coarse_surface_error: f64,
    pub coarse_deep_error: f64,
    pub fine_surface_error: f64,
    pub fine_deep_error: f64,
    pub observed_surface_order: Option<f64>,
    pub observed_deep_order: Option<f64>,
}

impl TwoBoxClimateModel {
    fn anomaly_matrix(&self) -> [[f64; 2]; 2] {
        [
            [
                -(self.feedback + self.ocean_exchange) / self.surface_heat_capacity,
                self.ocean_exchange / self.surface_heat_capacity,
            ],
            [
                self.ocean_exchange / self.deep_heat_capacity,
                -self.ocean_exchange / self.deep_heat_capacity,
            ],
        ]
    }

    /// Analytic decay modes under constant forcing.
    pub fn modes(&self) -> Result<TwoBoxModes, ModelError> {
        self.validate()?;
        let matrix = self.anomaly_matrix();
        let half_trace = 0.5 * (matrix[0][0] + matrix[1][1]);
        let half_difference = 0.5 * (matrix[0][0] - matrix[1][1]);
        let discriminant = (half_difference * half_difference + matrix[0][1] * matrix[1][0]).sqrt();
        let slow_decay_rate = half_trace + discriminant;
        let fast_decay_rate = half_trace - discriminant;
        if !(slow_decay_rate < 0.0 && fast_decay_rate < 0.0) {
            return Err(ModelError::SingularCalibration {
                reason: "two-box climate modes are not both decaying",
            });
        }
        Ok(TwoBoxModes {
            slow_decay_rate,
            fast_decay_rate,
            slow_timescale_seconds: -1.0 / slow_decay_rate,
            fast_timescale_seconds: -1.0 / fast_decay_rate,
        })
    }

    /// Exact two-box state after constant forcing for `elapsed_seconds`.
    pub fn exact_constant_forcing(
        &self,
        initial_state: TwoBoxState,
        forcing: f64,
        elapsed_seconds: f64,
    ) -> Result<TwoBoxState, ModelError> {
        self.validate()?;
        require_positive("surface_temperature", initial_state.surface_temperature)?;
        require_positive("deep_temperature", initial_state.deep_temperature)?;
        require_finite("forcing", forcing)?;
        require_non_negative("elapsed_seconds", elapsed_seconds)?;
        let equilibrium =
            self.equilibrium_state(forcing)
                .ok_or(ModelError::SingularCalibration {
                    reason: "two-box equilibrium requires positive feedback",
                })?;
        if elapsed_seconds == 0.0 {
            return Ok(initial_state);
        }

        let matrix = self.anomaly_matrix();
        let modes = self.modes()?;
        let z = [
            initial_state.surface_temperature - equilibrium.surface_temperature,
            initial_state.deep_temperature - equilibrium.deep_temperature,
        ];
        let separation = modes.slow_decay_rate - modes.fast_decay_rate;
        let evolved = if separation.abs() > 64.0 * f64::EPSILON {
            let slow_projection = [
                (matrix[0][0] - modes.fast_decay_rate) * z[0] + matrix[0][1] * z[1],
                matrix[1][0] * z[0] + (matrix[1][1] - modes.fast_decay_rate) * z[1],
            ];
            let fast_projection = [
                (matrix[0][0] - modes.slow_decay_rate) * z[0] + matrix[0][1] * z[1],
                matrix[1][0] * z[0] + (matrix[1][1] - modes.slow_decay_rate) * z[1],
            ];
            let slow_factor = (modes.slow_decay_rate * elapsed_seconds).exp();
            let fast_factor = (modes.fast_decay_rate * elapsed_seconds).exp();
            [
                (slow_factor * slow_projection[0] - fast_factor * fast_projection[0]) / separation,
                (slow_factor * slow_projection[1] - fast_factor * fast_projection[1]) / separation,
            ]
        } else {
            let rate = 0.5 * (modes.slow_decay_rate + modes.fast_decay_rate);
            let shifted = [
                (matrix[0][0] - rate) * z[0] + matrix[0][1] * z[1],
                matrix[1][0] * z[0] + (matrix[1][1] - rate) * z[1],
            ];
            let factor = (rate * elapsed_seconds).exp();
            [
                factor * (z[0] + elapsed_seconds * shifted[0]),
                factor * (z[1] + elapsed_seconds * shifted[1]),
            ]
        };
        let state = TwoBoxState {
            surface_temperature: equilibrium.surface_temperature + evolved[0],
            deep_temperature: equilibrium.deep_temperature + evolved[1],
        };
        require_positive("surface_temperature", state.surface_temperature)?;
        require_positive("deep_temperature", state.deep_temperature)?;
        Ok(state)
    }

    /// RK4 convergence evidence against the exact two-box constant-forcing state.
    pub fn constant_forcing_convergence(
        &self,
        initial_state: TwoBoxState,
        forcing: f64,
        coarse_grid: SimulationGrid,
    ) -> Result<TwoBoxConvergenceReport, ModelError> {
        let grid = SimulationGrid::try_new(coarse_grid.dt_seconds, coarse_grid.steps)?;
        let duration_seconds = grid.duration_seconds();
        let exact = self.exact_constant_forcing(initial_state, forcing, duration_seconds)?;
        let coarse = *self
            .simulate_constant_forcing(initial_state, forcing, grid.dt_seconds, grid.steps)?
            .last()
            .ok_or(ModelError::ZeroSteps)?;
        let fine_steps = grid
            .steps
            .checked_mul(2)
            .ok_or(ModelError::ScheduleTooLarge {
                requested: usize::MAX,
                maximum: usize::MAX / 2,
            })?;
        let fine = *self
            .simulate_constant_forcing(initial_state, forcing, 0.5 * grid.dt_seconds, fine_steps)?
            .last()
            .ok_or(ModelError::ZeroSteps)?;
        let coarse_surface_error = (coarse.surface_temperature - exact.surface_temperature).abs();
        let coarse_deep_error = (coarse.deep_temperature - exact.deep_temperature).abs();
        let fine_surface_error = (fine.surface_temperature - exact.surface_temperature).abs();
        let fine_deep_error = (fine.deep_temperature - exact.deep_temperature).abs();
        Ok(TwoBoxConvergenceReport {
            duration_seconds,
            coarse_surface_error,
            coarse_deep_error,
            fine_surface_error,
            fine_deep_error,
            observed_surface_order: empirical_order(coarse_surface_error, fine_surface_error),
            observed_deep_order: empirical_order(coarse_deep_error, fine_deep_error),
        })
    }
}

fn empirical_order(coarse: f64, fine: f64) -> Option<f64> {
    if coarse > 0.0 && fine > 0.0 {
        Some((coarse / fine).log2())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transient::SECONDS_PER_YEAR;

    #[test]
    fn earthlike_modes_are_ordered_and_decaying() {
        let modes = TwoBoxClimateModel::earthlike().modes().unwrap();
        assert!(modes.fast_decay_rate < modes.slow_decay_rate);
        assert!(modes.fast_timescale_seconds < modes.slow_timescale_seconds);
        assert!(modes.fast_timescale_seconds > 0.0);
    }

    #[test]
    fn exact_solution_starts_at_initial_and_relaxes_to_equilibrium() {
        let model = TwoBoxClimateModel::earthlike();
        let initial = TwoBoxState {
            surface_temperature: 288.0,
            deep_temperature: 288.0,
        };
        assert_eq!(
            model.exact_constant_forcing(initial, 4.0, 0.0).unwrap(),
            initial
        );
        let late = model
            .exact_constant_forcing(
                initial,
                4.0,
                100.0 * model.modes().unwrap().slow_timescale_seconds,
            )
            .unwrap();
        let equilibrium = model.equilibrium_state(4.0).unwrap();
        assert!((late.surface_temperature - equilibrium.surface_temperature).abs() < 1e-10);
        assert!((late.deep_temperature - equilibrium.deep_temperature).abs() < 1e-10);
    }

    #[test]
    fn rk4_converges_at_fourth_order_against_exact_modes() {
        let model = TwoBoxClimateModel::earthlike();
        let report = model
            .constant_forcing_convergence(
                TwoBoxState {
                    surface_temperature: 288.0,
                    deep_temperature: 288.0,
                },
                4.0,
                SimulationGrid::try_new(0.5 * SECONDS_PER_YEAR, 40).unwrap(),
            )
            .unwrap();
        assert!(report.fine_surface_error < report.coarse_surface_error);
        assert!(report.fine_deep_error < report.coarse_deep_error);
        assert!(report.observed_surface_order.unwrap() > 3.5);
        assert!(report.observed_deep_order.unwrap() > 3.5);
    }
}
