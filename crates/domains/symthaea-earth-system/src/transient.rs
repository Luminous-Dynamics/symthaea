// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Minimal transient energy-balance models.
//!
//! These models add thermal inertia to the crate's equilibrium baselines. They
//! use explicit SI units and deterministic RK4 stepping. The one-box model also
//! exposes its exact constant-forcing solution as a numerical oracle. They are
//! not ocean circulation or a resolved carbon-climate model.

use crate::error::{ModelError, require_finite, require_non_negative, require_positive};
use crate::forcing::ForcingProtocol;

/// Mean Gregorian year in seconds.
pub const SECONDS_PER_YEAR: f64 = 365.2425 * 24.0 * 60.0 * 60.0;

/// Hard bound for directly allocated fixed-step trajectories.
pub const MAX_TRAJECTORY_STEPS: usize = 1_000_000;

pub(crate) fn validate_trajectory_capacity(step_size: f64, steps: usize) -> Result<(), ModelError> {
    require_positive("integration_step", step_size)?;
    if steps > MAX_TRAJECTORY_STEPS {
        return Err(ModelError::TrajectoryTooLarge {
            requested: steps,
            maximum: MAX_TRAJECTORY_STEPS,
        });
    }
    steps.checked_add(1).ok_or(ModelError::TrajectoryTooLarge {
        requested: usize::MAX,
        maximum: MAX_TRAJECTORY_STEPS,
    })?;
    require_finite("integration_duration", step_size * steps as f64)
}

/// Explicit time grid for reproducible transient experiments.
///
/// `steps` is the number of integration intervals. Trajectories returned by
/// the `*_including_initial` methods contain `steps + 1` samples.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimulationGrid {
    pub dt_seconds: f64,
    pub steps: usize,
}

impl SimulationGrid {
    pub fn try_new(dt_seconds: f64, steps: usize) -> Result<Self, ModelError> {
        validate_trajectory_capacity(dt_seconds, steps)?;
        if steps == 0 {
            return Err(ModelError::ZeroSteps);
        }
        Ok(Self { dt_seconds, steps })
    }

    pub fn duration_seconds(&self) -> f64 {
        self.dt_seconds * self.steps as f64
    }

    pub fn sample_count_including_initial(&self) -> usize {
        self.steps + 1
    }
}

/// RK4 convergence evidence against the exact one-box constant-forcing solution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OneBoxConvergenceReport {
    pub coarse_dt_seconds: f64,
    pub coarse_absolute_error: f64,
    pub fine_absolute_error: f64,
    /// Empirical order from halving the step, when both errors are non-zero.
    pub observed_order: Option<f64>,
}

/// A timestamped one-box state with its instantaneous energy-budget diagnostic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OneBoxSample {
    pub time_seconds: f64,
    pub temperature: f64,
    pub forcing: f64,
    pub radiative_imbalance: f64,
}

/// One-box global-mean climate model:
/// `C dT/dt = F - λ(T - T₀)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OneBoxClimateModel {
    /// Effective heat capacity, J·m⁻²·K⁻¹.
    pub heat_capacity: f64,
    /// Restoring climate-feedback parameter, W·m⁻²·K⁻¹.
    pub feedback: f64,
    /// Unforced reference temperature, K.
    pub baseline_temperature: f64,
}

impl OneBoxClimateModel {
    pub fn try_new(
        heat_capacity: f64,
        feedback: f64,
        baseline_temperature: f64,
    ) -> Result<Self, ModelError> {
        let model = Self {
            heat_capacity,
            feedback,
            baseline_temperature,
        };
        model.validate()?;
        Ok(model)
    }

    /// Illustrative mixed-layer-like parameters, not an observational fit.
    pub fn earthlike() -> Self {
        Self {
            heat_capacity: 4.0e8,
            feedback: 1.2,
            baseline_temperature: 288.0,
        }
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        require_positive("heat_capacity", self.heat_capacity)?;
        require_positive("feedback", self.feedback)?;
        require_positive("baseline_temperature", self.baseline_temperature)?;
        Ok(())
    }

    /// Net top-of-atmosphere energy imbalance, W/m².
    pub fn radiative_imbalance(&self, temperature: f64, forcing: f64) -> f64 {
        forcing - self.feedback * (temperature - self.baseline_temperature)
    }

    /// Temperature tendency in K/s under radiative forcing `forcing` (W/m²).
    pub fn tendency(&self, temperature: f64, forcing: f64) -> f64 {
        self.radiative_imbalance(temperature, forcing) / self.heat_capacity
    }

    /// Equilibrium temperature under constant forcing.
    pub fn equilibrium_temperature(&self, forcing: f64) -> f64 {
        self.baseline_temperature + forcing / self.feedback
    }

    /// E-folding response time, seconds.
    pub fn response_time(&self) -> f64 {
        self.heat_capacity / self.feedback
    }

    /// Exact solution under constant forcing.
    pub fn exact_constant_forcing(
        &self,
        initial_temperature: f64,
        forcing: f64,
        elapsed_seconds: f64,
    ) -> Result<f64, ModelError> {
        self.validate()?;
        require_positive("initial_temperature", initial_temperature)?;
        require_finite("forcing", forcing)?;
        require_non_negative("elapsed_seconds", elapsed_seconds)?;
        let equilibrium = self.equilibrium_temperature(forcing);
        Ok(equilibrium
            + (initial_temperature - equilibrium) * (-elapsed_seconds / self.response_time()).exp())
    }

    /// Advance one RK4 step under constant forcing.
    pub fn step_rk4(
        &self,
        temperature: f64,
        forcing: f64,
        dt_seconds: f64,
    ) -> Result<f64, ModelError> {
        self.validate()?;
        require_positive("temperature", temperature)?;
        require_finite("forcing", forcing)?;
        require_positive("dt_seconds", dt_seconds)?;

        let k1 = self.tendency(temperature, forcing);
        require_finite("temperature_tendency_stage_1", k1)?;
        let stage2 = temperature + 0.5 * dt_seconds * k1;
        require_positive("temperature_stage_2", stage2)?;
        let k2 = self.tendency(stage2, forcing);
        require_finite("temperature_tendency_stage_2", k2)?;
        let stage3 = temperature + 0.5 * dt_seconds * k2;
        require_positive("temperature_stage_3", stage3)?;
        let k3 = self.tendency(stage3, forcing);
        require_finite("temperature_tendency_stage_3", k3)?;
        let stage4 = temperature + dt_seconds * k3;
        require_positive("temperature_stage_4", stage4)?;
        let k4 = self.tendency(stage4, forcing);
        require_finite("temperature_tendency_stage_4", k4)?;
        let next = temperature + dt_seconds * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
        require_positive("temperature", next)?;
        Ok(next)
    }

    /// Advance one RK4 step while evaluating a protocol at RK4 substeps.
    pub fn step_rk4_protocol(
        &self,
        temperature: f64,
        start_time_seconds: f64,
        dt_seconds: f64,
        protocol: &ForcingProtocol,
    ) -> Result<f64, ModelError> {
        self.validate()?;
        protocol.validate()?;
        require_positive("temperature", temperature)?;
        require_non_negative("start_time_seconds", start_time_seconds)?;
        require_positive("dt_seconds", dt_seconds)?;

        let half_time = start_time_seconds + 0.5 * dt_seconds;
        let end_time = start_time_seconds + dt_seconds;
        let k1 = self.tendency(temperature, protocol.at(start_time_seconds)?);
        require_finite("temperature_tendency_stage_1", k1)?;
        let stage2 = temperature + 0.5 * dt_seconds * k1;
        require_positive("temperature_stage_2", stage2)?;
        let k2 = self.tendency(stage2, protocol.at(half_time)?);
        require_finite("temperature_tendency_stage_2", k2)?;
        let stage3 = temperature + 0.5 * dt_seconds * k2;
        require_positive("temperature_stage_3", stage3)?;
        let k3 = self.tendency(stage3, protocol.at(half_time)?);
        require_finite("temperature_tendency_stage_3", k3)?;
        let stage4 = temperature + dt_seconds * k3;
        require_positive("temperature_stage_4", stage4)?;
        let k4 = self.tendency(stage4, protocol.at(end_time)?);
        require_finite("temperature_tendency_stage_4", k4)?;
        let next = temperature + dt_seconds * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
        require_positive("temperature", next)?;
        Ok(next)
    }

    /// Advance one RK4 step while treating a protocol breakpoint at the end
    /// of the interval as a left-hand limit. This is used by event-aligned
    /// schedules so rectangular pulses are not numerically smeared.
    fn step_rk4_protocol_event_aligned(
        &self,
        temperature: f64,
        start_time_seconds: f64,
        dt_seconds: f64,
        protocol: &ForcingProtocol,
    ) -> Result<f64, ModelError> {
        self.validate()?;
        protocol.validate()?;
        require_positive("temperature", temperature)?;
        require_non_negative("start_time_seconds", start_time_seconds)?;
        require_positive("dt_seconds", dt_seconds)?;

        let half_time = start_time_seconds + 0.5 * dt_seconds;
        let end_time = start_time_seconds + dt_seconds;
        let k1 = self.tendency(temperature, protocol.at(start_time_seconds)?);
        require_finite("temperature_tendency_stage_1", k1)?;
        let stage2 = temperature + 0.5 * dt_seconds * k1;
        require_positive("temperature_stage_2", stage2)?;
        let k2 = self.tendency(stage2, protocol.at(half_time)?);
        require_finite("temperature_tendency_stage_2", k2)?;
        let stage3 = temperature + 0.5 * dt_seconds * k2;
        require_positive("temperature_stage_3", stage3)?;
        let k3 = self.tendency(stage3, protocol.at(half_time)?);
        require_finite("temperature_tendency_stage_3", k3)?;
        let stage4 = temperature + dt_seconds * k3;
        require_positive("temperature_stage_4", stage4)?;
        let k4 = self.tendency(stage4, protocol.at_left_limit(end_time)?);
        require_finite("temperature_tendency_stage_4", k4)?;
        let next = temperature + dt_seconds * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
        require_positive("temperature", next)?;
        Ok(next)
    }

    /// Simulate constant forcing, returning states after each step.
    pub fn simulate_constant_forcing(
        &self,
        initial_temperature: f64,
        forcing: f64,
        dt_seconds: f64,
        steps: usize,
    ) -> Result<Vec<f64>, ModelError> {
        self.validate()?;
        require_positive("initial_temperature", initial_temperature)?;
        require_finite("forcing", forcing)?;
        validate_trajectory_capacity(dt_seconds, steps)?;

        let mut trajectory = Vec::with_capacity(steps);
        let mut temperature = initial_temperature;
        for _ in 0..steps {
            temperature = self.step_rk4(temperature, forcing, dt_seconds)?;
            trajectory.push(temperature);
        }
        Ok(trajectory)
    }

    /// Simulate a forcing protocol and retain forcing and imbalance diagnostics.
    pub fn simulate_protocol(
        &self,
        initial_temperature: f64,
        protocol: &ForcingProtocol,
        dt_seconds: f64,
        steps: usize,
    ) -> Result<Vec<OneBoxSample>, ModelError> {
        self.validate()?;
        protocol.validate()?;
        require_positive("initial_temperature", initial_temperature)?;
        validate_trajectory_capacity(dt_seconds, steps)?;

        let mut trajectory = Vec::with_capacity(steps);
        let mut temperature = initial_temperature;
        let mut time_seconds = 0.0;
        for _ in 0..steps {
            temperature =
                self.step_rk4_protocol(temperature, time_seconds, dt_seconds, protocol)?;
            time_seconds += dt_seconds;
            let forcing = protocol.at(time_seconds)?;
            trajectory.push(OneBoxSample {
                time_seconds,
                temperature,
                forcing,
                radiative_imbalance: self.radiative_imbalance(temperature, forcing),
            });
        }
        Ok(trajectory)
    }
    /// Simulate a forcing protocol with an explicit grid and include the
    /// initial state at `t = 0`.
    pub fn simulate_protocol_including_initial(
        &self,
        initial_temperature: f64,
        protocol: &ForcingProtocol,
        grid: SimulationGrid,
    ) -> Result<Vec<OneBoxSample>, ModelError> {
        self.validate()?;
        protocol.validate()?;
        require_positive("initial_temperature", initial_temperature)?;
        let grid = SimulationGrid::try_new(grid.dt_seconds, grid.steps)?;

        let mut samples = Vec::with_capacity(grid.sample_count_including_initial());
        let mut temperature = initial_temperature;
        let mut time_seconds = 0.0;
        let forcing = protocol.at(time_seconds)?;
        samples.push(OneBoxSample {
            time_seconds,
            temperature,
            forcing,
            radiative_imbalance: self.radiative_imbalance(temperature, forcing),
        });

        for _ in 0..grid.steps {
            temperature =
                self.step_rk4_protocol(temperature, time_seconds, grid.dt_seconds, protocol)?;
            time_seconds += grid.dt_seconds;
            let forcing = protocol.at(time_seconds)?;
            samples.push(OneBoxSample {
                time_seconds,
                temperature,
                forcing,
                radiative_imbalance: self.radiative_imbalance(temperature, forcing),
            });
        }
        Ok(samples)
    }

    /// Simulate through an arbitrary duration using a nominal maximum step
    /// and exact splits at every protocol breakpoint. The result includes the
    /// initial state and every generated interval endpoint.
    pub fn simulate_protocol_event_aligned(
        &self,
        initial_temperature: f64,
        protocol: &ForcingProtocol,
        nominal_dt_seconds: f64,
        duration_seconds: f64,
    ) -> Result<Vec<OneBoxSample>, ModelError> {
        self.validate()?;
        protocol.validate()?;
        require_positive("initial_temperature", initial_temperature)?;
        let intervals = crate::schedule::event_aligned_intervals(
            duration_seconds,
            nominal_dt_seconds,
            &protocol.integration_events(),
        )?;
        let mut samples = Vec::with_capacity(intervals.len() + 1);
        let mut temperature = initial_temperature;
        let initial_forcing = protocol.at(0.0)?;
        samples.push(OneBoxSample {
            time_seconds: 0.0,
            temperature,
            forcing: initial_forcing,
            radiative_imbalance: self.radiative_imbalance(temperature, initial_forcing),
        });
        for interval in intervals {
            temperature = self.step_rk4_protocol_event_aligned(
                temperature,
                interval.start,
                interval.duration(),
                protocol,
            )?;
            let forcing = protocol.at(interval.end)?;
            samples.push(OneBoxSample {
                time_seconds: interval.end,
                temperature,
                forcing,
                radiative_imbalance: self.radiative_imbalance(temperature, forcing),
            });
        }
        Ok(samples)
    }

    /// Compare two RK4 resolutions against the exact constant-forcing result.
    pub fn constant_forcing_convergence(
        &self,
        initial_temperature: f64,
        forcing: f64,
        coarse_grid: SimulationGrid,
    ) -> Result<OneBoxConvergenceReport, ModelError> {
        let coarse_grid = SimulationGrid::try_new(coarse_grid.dt_seconds, coarse_grid.steps)?;
        let duration = coarse_grid.duration_seconds();
        let exact = self.exact_constant_forcing(initial_temperature, forcing, duration)?;
        let coarse = *self
            .simulate_constant_forcing(
                initial_temperature,
                forcing,
                coarse_grid.dt_seconds,
                coarse_grid.steps,
            )?
            .last()
            .ok_or(ModelError::ZeroSteps)?;
        let fine = *self
            .simulate_constant_forcing(
                initial_temperature,
                forcing,
                0.5 * coarse_grid.dt_seconds,
                2 * coarse_grid.steps,
            )?
            .last()
            .ok_or(ModelError::ZeroSteps)?;
        let coarse_absolute_error = (coarse - exact).abs();
        let fine_absolute_error = (fine - exact).abs();
        let observed_order = if coarse_absolute_error > 0.0 && fine_absolute_error > 0.0 {
            Some((coarse_absolute_error / fine_absolute_error).log2())
        } else {
            None
        };
        Ok(OneBoxConvergenceReport {
            coarse_dt_seconds: coarse_grid.dt_seconds,
            coarse_absolute_error,
            fine_absolute_error,
            observed_order,
        })
    }
}

/// Surface/deep-ocean temperature state for [`TwoBoxClimateModel`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoBoxState {
    pub surface_temperature: f64,
    pub deep_temperature: f64,
}

/// A timestamped two-box state with explicit energy-flow diagnostics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoBoxSample {
    pub time_seconds: f64,
    pub state: TwoBoxState,
    pub forcing: f64,
    pub top_of_atmosphere_imbalance: f64,
    pub ocean_heat_flux: f64,
    pub heat_content_anomaly: f64,
}

/// Two-box climate model with surface-to-deep heat exchange.
///
/// ```text
/// Cₛ dTₛ/dt = F - λ(Tₛ-T₀) - κ(Tₛ-T_d)
/// C_d dT_d/dt = κ(Tₛ-T_d)
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoBoxClimateModel {
    pub surface_heat_capacity: f64,
    pub deep_heat_capacity: f64,
    pub feedback: f64,
    pub ocean_exchange: f64,
    pub baseline_temperature: f64,
}

impl TwoBoxClimateModel {
    pub fn try_new(
        surface_heat_capacity: f64,
        deep_heat_capacity: f64,
        feedback: f64,
        ocean_exchange: f64,
        baseline_temperature: f64,
    ) -> Result<Self, ModelError> {
        let model = Self {
            surface_heat_capacity,
            deep_heat_capacity,
            feedback,
            ocean_exchange,
            baseline_temperature,
        };
        model.validate()?;
        Ok(model)
    }

    /// Illustrative parameters suitable for deterministic experiments.
    pub fn earthlike() -> Self {
        Self {
            surface_heat_capacity: 4.0e8,
            deep_heat_capacity: 1.0e10,
            feedback: 1.2,
            ocean_exchange: 0.7,
            baseline_temperature: 288.0,
        }
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        require_positive("surface_heat_capacity", self.surface_heat_capacity)?;
        require_positive("deep_heat_capacity", self.deep_heat_capacity)?;
        require_non_negative("feedback", self.feedback)?;
        require_non_negative("ocean_exchange", self.ocean_exchange)?;
        require_positive("baseline_temperature", self.baseline_temperature)?;
        Ok(())
    }

    /// Net top-of-atmosphere energy imbalance, W/m².
    pub fn top_of_atmosphere_imbalance(&self, state: TwoBoxState, forcing: f64) -> f64 {
        forcing - self.feedback * (state.surface_temperature - self.baseline_temperature)
    }

    /// Downward surface-to-deep heat flux, W/m².
    pub fn ocean_heat_flux(&self, state: TwoBoxState) -> f64 {
        self.ocean_exchange * (state.surface_temperature - state.deep_temperature)
    }

    /// Temperature derivatives `(dT_surface/dt, dT_deep/dt)` in K/s.
    pub fn derivatives(&self, state: TwoBoxState, forcing: f64) -> TwoBoxState {
        let exchange = self.ocean_heat_flux(state);
        TwoBoxState {
            surface_temperature: (self.top_of_atmosphere_imbalance(state, forcing) - exchange)
                / self.surface_heat_capacity,
            deep_temperature: exchange / self.deep_heat_capacity,
        }
    }

    /// Rate of total heat-content change implied by the derivatives, W/m².
    pub fn heat_content_tendency(&self, state: TwoBoxState, forcing: f64) -> f64 {
        let tendency = self.derivatives(state, forcing);
        self.surface_heat_capacity * tendency.surface_temperature
            + self.deep_heat_capacity * tendency.deep_temperature
    }

    /// Constant-forcing equilibrium when `feedback > 0`.
    pub fn equilibrium_state(&self, forcing: f64) -> Option<TwoBoxState> {
        (self.feedback > 0.0).then(|| {
            let temperature = self.baseline_temperature + forcing / self.feedback;
            TwoBoxState {
                surface_temperature: temperature,
                deep_temperature: temperature,
            }
        })
    }

    /// Heat-content anomaly relative to the baseline, J/m².
    pub fn heat_content_anomaly(&self, state: TwoBoxState) -> f64 {
        self.surface_heat_capacity * (state.surface_temperature - self.baseline_temperature)
            + self.deep_heat_capacity * (state.deep_temperature - self.baseline_temperature)
    }

    pub fn step_rk4(
        &self,
        state: TwoBoxState,
        forcing: f64,
        dt_seconds: f64,
    ) -> Result<TwoBoxState, ModelError> {
        self.validate()?;
        require_positive("surface_temperature", state.surface_temperature)?;
        require_positive("deep_temperature", state.deep_temperature)?;
        require_finite("forcing", forcing)?;
        require_positive("dt_seconds", dt_seconds)?;

        let k1 = self.derivatives(state, forcing);
        validate_two_box_derivative(k1, 1)?;
        let stage2 = add_scaled(state, k1, 0.5 * dt_seconds);
        validate_two_box_state(stage2, 2)?;
        let k2 = self.derivatives(stage2, forcing);
        validate_two_box_derivative(k2, 2)?;
        let stage3 = add_scaled(state, k2, 0.5 * dt_seconds);
        validate_two_box_state(stage3, 3)?;
        let k3 = self.derivatives(stage3, forcing);
        validate_two_box_derivative(k3, 3)?;
        let stage4 = add_scaled(state, k3, dt_seconds);
        validate_two_box_state(stage4, 4)?;
        let k4 = self.derivatives(stage4, forcing);
        validate_two_box_derivative(k4, 4)?;
        let next = combine_rk4(state, k1, k2, k3, k4, dt_seconds);
        validate_two_box_state(next, 0)?;
        Ok(next)
    }

    /// Advance one RK4 step while evaluating a protocol at RK4 substeps.
    pub fn step_rk4_protocol(
        &self,
        state: TwoBoxState,
        start_time_seconds: f64,
        dt_seconds: f64,
        protocol: &ForcingProtocol,
    ) -> Result<TwoBoxState, ModelError> {
        self.validate()?;
        protocol.validate()?;
        require_positive("surface_temperature", state.surface_temperature)?;
        require_positive("deep_temperature", state.deep_temperature)?;
        require_non_negative("start_time_seconds", start_time_seconds)?;
        require_positive("dt_seconds", dt_seconds)?;

        let half_time = start_time_seconds + 0.5 * dt_seconds;
        let end_time = start_time_seconds + dt_seconds;
        let k1 = self.derivatives(state, protocol.at(start_time_seconds)?);
        validate_two_box_derivative(k1, 1)?;
        let stage2 = add_scaled(state, k1, 0.5 * dt_seconds);
        validate_two_box_state(stage2, 2)?;
        let k2 = self.derivatives(stage2, protocol.at(half_time)?);
        validate_two_box_derivative(k2, 2)?;
        let stage3 = add_scaled(state, k2, 0.5 * dt_seconds);
        validate_two_box_state(stage3, 3)?;
        let k3 = self.derivatives(stage3, protocol.at(half_time)?);
        validate_two_box_derivative(k3, 3)?;
        let stage4 = add_scaled(state, k3, dt_seconds);
        validate_two_box_state(stage4, 4)?;
        let k4 = self.derivatives(stage4, protocol.at(end_time)?);
        validate_two_box_derivative(k4, 4)?;
        let next = combine_rk4(state, k1, k2, k3, k4, dt_seconds);
        validate_two_box_state(next, 0)?;
        Ok(next)
    }

    pub fn simulate_constant_forcing(
        &self,
        initial_state: TwoBoxState,
        forcing: f64,
        dt_seconds: f64,
        steps: usize,
    ) -> Result<Vec<TwoBoxState>, ModelError> {
        self.validate()?;
        require_positive("surface_temperature", initial_state.surface_temperature)?;
        require_positive("deep_temperature", initial_state.deep_temperature)?;
        require_finite("forcing", forcing)?;
        validate_trajectory_capacity(dt_seconds, steps)?;
        let mut trajectory = Vec::with_capacity(steps);
        let mut state = initial_state;
        for _ in 0..steps {
            state = self.step_rk4(state, forcing, dt_seconds)?;
            trajectory.push(state);
        }
        Ok(trajectory)
    }

    /// Simulate a forcing protocol and retain energy-flow diagnostics.
    pub fn simulate_protocol(
        &self,
        initial_state: TwoBoxState,
        protocol: &ForcingProtocol,
        dt_seconds: f64,
        steps: usize,
    ) -> Result<Vec<TwoBoxSample>, ModelError> {
        self.validate()?;
        protocol.validate()?;
        require_positive("surface_temperature", initial_state.surface_temperature)?;
        require_positive("deep_temperature", initial_state.deep_temperature)?;
        validate_trajectory_capacity(dt_seconds, steps)?;

        let mut trajectory = Vec::with_capacity(steps);
        let mut state = initial_state;
        let mut time_seconds = 0.0;
        for _ in 0..steps {
            state = self.step_rk4_protocol(state, time_seconds, dt_seconds, protocol)?;
            time_seconds += dt_seconds;
            let forcing = protocol.at(time_seconds)?;
            trajectory.push(TwoBoxSample {
                time_seconds,
                state,
                forcing,
                top_of_atmosphere_imbalance: self.top_of_atmosphere_imbalance(state, forcing),
                ocean_heat_flux: self.ocean_heat_flux(state),
                heat_content_anomaly: self.heat_content_anomaly(state),
            });
        }
        Ok(trajectory)
    }
    /// Simulate a forcing protocol on an explicit grid and include the initial
    /// surface/deep state at `t = 0`.
    pub fn simulate_protocol_including_initial(
        &self,
        initial_state: TwoBoxState,
        protocol: &ForcingProtocol,
        grid: SimulationGrid,
    ) -> Result<Vec<TwoBoxSample>, ModelError> {
        self.validate()?;
        protocol.validate()?;
        require_positive("surface_temperature", initial_state.surface_temperature)?;
        require_positive("deep_temperature", initial_state.deep_temperature)?;
        let grid = SimulationGrid::try_new(grid.dt_seconds, grid.steps)?;

        let mut samples = Vec::with_capacity(grid.sample_count_including_initial());
        let mut state = initial_state;
        let mut time_seconds = 0.0;
        let forcing = protocol.at(time_seconds)?;
        samples.push(TwoBoxSample {
            time_seconds,
            state,
            forcing,
            top_of_atmosphere_imbalance: self.top_of_atmosphere_imbalance(state, forcing),
            ocean_heat_flux: self.ocean_heat_flux(state),
            heat_content_anomaly: self.heat_content_anomaly(state),
        });

        for _ in 0..grid.steps {
            state = self.step_rk4_protocol(state, time_seconds, grid.dt_seconds, protocol)?;
            time_seconds += grid.dt_seconds;
            let forcing = protocol.at(time_seconds)?;
            samples.push(TwoBoxSample {
                time_seconds,
                state,
                forcing,
                top_of_atmosphere_imbalance: self.top_of_atmosphere_imbalance(state, forcing),
                ocean_heat_flux: self.ocean_heat_flux(state),
                heat_content_anomaly: self.heat_content_anomaly(state),
            });
        }
        Ok(samples)
    }
}

fn validate_two_box_state(state: TwoBoxState, stage: usize) -> Result<(), ModelError> {
    let surface = match stage {
        0 => "surface_temperature",
        2 => "surface_temperature_stage_2",
        3 => "surface_temperature_stage_3",
        _ => "surface_temperature_stage_4",
    };
    let deep = match stage {
        0 => "deep_temperature",
        2 => "deep_temperature_stage_2",
        3 => "deep_temperature_stage_3",
        _ => "deep_temperature_stage_4",
    };
    require_positive(surface, state.surface_temperature)?;
    require_positive(deep, state.deep_temperature)?;
    Ok(())
}

fn validate_two_box_derivative(derivative: TwoBoxState, stage: usize) -> Result<(), ModelError> {
    let surface = match stage {
        1 => "surface_tendency_stage_1",
        2 => "surface_tendency_stage_2",
        3 => "surface_tendency_stage_3",
        _ => "surface_tendency_stage_4",
    };
    let deep = match stage {
        1 => "deep_tendency_stage_1",
        2 => "deep_tendency_stage_2",
        3 => "deep_tendency_stage_3",
        _ => "deep_tendency_stage_4",
    };
    require_finite(surface, derivative.surface_temperature)?;
    require_finite(deep, derivative.deep_temperature)?;
    Ok(())
}

fn add_scaled(state: TwoBoxState, derivative: TwoBoxState, scale: f64) -> TwoBoxState {
    TwoBoxState {
        surface_temperature: state.surface_temperature + scale * derivative.surface_temperature,
        deep_temperature: state.deep_temperature + scale * derivative.deep_temperature,
    }
}

fn combine_rk4(
    state: TwoBoxState,
    k1: TwoBoxState,
    k2: TwoBoxState,
    k3: TwoBoxState,
    k4: TwoBoxState,
    dt_seconds: f64,
) -> TwoBoxState {
    TwoBoxState {
        surface_temperature: state.surface_temperature
            + dt_seconds
                * (k1.surface_temperature
                    + 2.0 * k2.surface_temperature
                    + 2.0 * k3.surface_temperature
                    + k4.surface_temperature)
                / 6.0,
        deep_temperature: state.deep_temperature
            + dt_seconds
                * (k1.deep_temperature
                    + 2.0 * k2.deep_temperature
                    + 2.0 * k3.deep_temperature
                    + k4.deep_temperature)
                / 6.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_box_converges_toward_forced_equilibrium() {
        let model = OneBoxClimateModel::earthlike();
        let forcing = 3.7;
        let equilibrium = model.equilibrium_temperature(forcing);
        let trajectory = model
            .simulate_constant_forcing(288.0, forcing, 0.1 * SECONDS_PER_YEAR, 1000)
            .unwrap();
        let final_temperature = *trajectory.last().unwrap();
        assert!((final_temperature - equilibrium).abs() < 1e-3);
    }

    #[test]
    fn exact_one_box_solution_is_a_rk4_oracle() {
        let model = OneBoxClimateModel::earthlike();
        let dt = 0.05 * SECONDS_PER_YEAR;
        let steps = 200;
        let numerical = model
            .simulate_constant_forcing(288.0, 3.7, dt, steps)
            .unwrap();
        let exact = model
            .exact_constant_forcing(288.0, 3.7, dt * steps as f64)
            .unwrap();
        assert!((numerical.last().unwrap() - exact).abs() < 1e-9);
    }

    #[test]
    fn one_response_time_closes_one_minus_inverse_e_of_gap() {
        let model = OneBoxClimateModel::earthlike();
        let equilibrium = model.equilibrium_temperature(3.7);
        let temperature = model
            .exact_constant_forcing(288.0, 3.7, model.response_time())
            .unwrap();
        let closed_fraction = (temperature - 288.0) / (equilibrium - 288.0);
        assert!((closed_fraction - (1.0 - (-1.0_f64).exp())).abs() < 1e-12);
    }

    #[test]
    fn constant_protocol_matches_constant_step() {
        let model = OneBoxClimateModel::earthlike();
        let protocol = ForcingProtocol::constant(3.7).unwrap();
        let dt = 0.25 * SECONDS_PER_YEAR;
        let direct = model.step_rk4(288.0, 3.7, dt).unwrap();
        let protocol_step = model.step_rk4_protocol(288.0, 0.0, dt, &protocol).unwrap();
        assert!((direct - protocol_step).abs() < 1e-12);
    }

    #[test]
    fn two_box_equilibrium_zeroes_derivatives() {
        let model = TwoBoxClimateModel::earthlike();
        let state = model.equilibrium_state(3.7).unwrap();
        let derivative = model.derivatives(state, 3.7);
        assert!(derivative.surface_temperature.abs() < 1e-15);
        assert!(derivative.deep_temperature.abs() < 1e-15);
    }

    #[test]
    fn two_box_budget_closes_at_derivative_level() {
        let model = TwoBoxClimateModel::earthlike();
        let state = TwoBoxState {
            surface_temperature: 289.0,
            deep_temperature: 287.5,
        };
        let forcing = 2.0;
        let tendency = model.heat_content_tendency(state, forcing);
        let toa = model.top_of_atmosphere_imbalance(state, forcing);
        assert!((tendency - toa).abs() < 1e-12);
    }

    #[test]
    fn internal_ocean_exchange_conserves_total_heat() {
        let model = TwoBoxClimateModel::try_new(4.0e8, 1.0e10, 0.0, 0.7, 288.0).unwrap();
        let initial = TwoBoxState {
            surface_temperature: 290.0,
            deep_temperature: 287.0,
        };
        let before = model.heat_content_anomaly(initial);
        let after_state = model
            .step_rk4(initial, 0.0, 0.01 * SECONDS_PER_YEAR)
            .unwrap();
        let after = model.heat_content_anomaly(after_state);
        assert!(
            (after - before).abs() < 1e-3,
            "heat drift={}",
            after - before
        );
    }

    #[test]
    fn ocean_uptake_delays_surface_warming() {
        let one_box = OneBoxClimateModel::earthlike();
        let two_box = TwoBoxClimateModel::earthlike();
        let dt = 0.25 * SECONDS_PER_YEAR;
        let one = one_box.step_rk4(288.0, 3.7, dt).unwrap();
        let two = two_box
            .step_rk4(
                TwoBoxState {
                    surface_temperature: 288.0,
                    deep_temperature: 288.0,
                },
                3.7,
                dt,
            )
            .unwrap();
        assert!(two.surface_temperature <= one + 1e-12);
    }

    #[test]
    fn protocol_samples_are_timestamped_and_diagnostic() {
        let model = OneBoxClimateModel::earthlike();
        let protocol = ForcingProtocol::linear_ramp(0.0, 4.0, SECONDS_PER_YEAR).unwrap();
        let samples = model
            .simulate_protocol(288.0, &protocol, 0.25 * SECONDS_PER_YEAR, 4)
            .unwrap();
        assert_eq!(samples.len(), 4);
        assert!((samples[3].time_seconds - SECONDS_PER_YEAR).abs() < 1e-9);
        assert!((samples[3].forcing - 4.0).abs() < 1e-12);
        assert!(samples.iter().all(|sample| sample.temperature.is_finite()));
    }
    #[test]
    fn explicit_grid_includes_initial_sample() {
        let model = OneBoxClimateModel::earthlike();
        let protocol = ForcingProtocol::constant(3.7).unwrap();
        let grid = SimulationGrid::try_new(0.25 * SECONDS_PER_YEAR, 4).unwrap();
        let samples = model
            .simulate_protocol_including_initial(288.0, &protocol, grid)
            .unwrap();
        assert_eq!(samples.len(), 5);
        assert_eq!(samples[0].time_seconds, 0.0);
        assert_eq!(samples[0].temperature, 288.0);
        assert!((samples.last().unwrap().time_seconds - SECONDS_PER_YEAR).abs() < 1e-9);
    }

    #[test]
    fn rk4_convergence_report_improves_when_step_is_halved() {
        let model = OneBoxClimateModel::earthlike();
        let grid = SimulationGrid::try_new(SECONDS_PER_YEAR, 20).unwrap();
        let report = model
            .constant_forcing_convergence(288.0, 3.7, grid)
            .unwrap();
        assert!(report.fine_absolute_error < report.coarse_absolute_error);
        assert!(report.observed_order.unwrap() > 3.5);
    }

    #[test]
    fn two_box_explicit_grid_includes_initial_sample() {
        let model = TwoBoxClimateModel::earthlike();
        let protocol = ForcingProtocol::constant(1.0).unwrap();
        let grid = SimulationGrid::try_new(0.1 * SECONDS_PER_YEAR, 2).unwrap();
        let initial = TwoBoxState {
            surface_temperature: 288.0,
            deep_temperature: 288.0,
        };
        let samples = model
            .simulate_protocol_including_initial(initial, &protocol, grid)
            .unwrap();
        assert_eq!(samples.len(), 3);
        assert_eq!(samples[0].state, initial);
    }

    #[test]
    fn oversized_steps_fail_before_leaving_physical_temperature_domain() {
        let one_box = OneBoxClimateModel::earthlike();
        assert!(one_box.step_rk4(288.0, -1.0e12, 1.0e9).is_err());

        let two_box = TwoBoxClimateModel::earthlike();
        let state = TwoBoxState {
            surface_temperature: 288.0,
            deep_temperature: 288.0,
        };
        assert!(two_box.step_rk4(state, -1.0e12, 1.0e9).is_err());
    }

    #[test]
    fn event_aligned_pulse_matches_piecewise_exact_solution() {
        let model = OneBoxClimateModel::earthlike();
        let pulse = ForcingProtocol::pulse(0.0, 4.0, 2.5, 7.25).unwrap();
        let samples = model
            .simulate_protocol_event_aligned(288.0, &pulse, 4.0, 10.0)
            .unwrap();
        assert!(samples.iter().any(|sample| sample.time_seconds == 2.5));
        assert!(samples.iter().any(|sample| sample.time_seconds == 7.25));

        let at_start = model.exact_constant_forcing(288.0, 0.0, 2.5).unwrap();
        let at_end = model
            .exact_constant_forcing(at_start, 4.0, 7.25 - 2.5)
            .unwrap();
        let exact = model
            .exact_constant_forcing(at_end, 0.0, 10.0 - 7.25)
            .unwrap();
        assert!((samples.last().unwrap().temperature - exact).abs() < 2.0e-7);
    }

    #[test]
    fn simulation_grid_rejects_unbounded_or_overflowing_trajectories() {
        assert!(matches!(
            SimulationGrid::try_new(1.0, MAX_TRAJECTORY_STEPS + 1),
            Err(ModelError::TrajectoryTooLarge { .. })
        ));
        assert!(SimulationGrid::try_new(f64::MAX, 2).is_err());
    }
}
