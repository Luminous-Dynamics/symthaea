// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Mass-conserving reduced-order carbon-cycle and carbon-climate dynamics.
//!
//! The carbon cycle is a reversible two-box anomaly model, not a calibrated
//! representation of the ocean, land biosphere, carbonate chemistry, or
//! permafrost. Its purpose is to provide a transparent dynamic bridge from an
//! emissions rate to atmospheric concentration while preserving carbon exactly
//! at the differential-equation level.

use crate::energy_balance::try_co2_radiative_forcing_myhre1998;
use crate::error::{ModelError, require_finite, require_non_negative, require_positive};
use crate::transient::{OneBoxClimateModel, SECONDS_PER_YEAR, validate_trajectory_capacity};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EmissionsProtocol {
    /// Constant emissions rate in GtC/year.
    Constant { rate_gtc_per_year: f64 },
    /// Linear ramp between emissions rates, followed by a hold.
    LinearRamp {
        initial_gtc_per_year: f64,
        final_gtc_per_year: f64,
        duration_years: f64,
    },
    /// Rectangular emissions pulse over `[start_years, end_years)`.
    Pulse {
        baseline_gtc_per_year: f64,
        anomaly_gtc_per_year: f64,
        start_years: f64,
        end_years: f64,
    },
}

impl EmissionsProtocol {
    pub fn constant(rate_gtc_per_year: f64) -> Result<Self, ModelError> {
        let protocol = Self::Constant { rate_gtc_per_year };
        protocol.validate()?;
        Ok(protocol)
    }

    pub fn linear_ramp(
        initial_gtc_per_year: f64,
        final_gtc_per_year: f64,
        duration_years: f64,
    ) -> Result<Self, ModelError> {
        let protocol = Self::LinearRamp {
            initial_gtc_per_year,
            final_gtc_per_year,
            duration_years,
        };
        protocol.validate()?;
        Ok(protocol)
    }

    pub fn pulse(
        baseline_gtc_per_year: f64,
        anomaly_gtc_per_year: f64,
        start_years: f64,
        end_years: f64,
    ) -> Result<Self, ModelError> {
        let protocol = Self::Pulse {
            baseline_gtc_per_year,
            anomaly_gtc_per_year,
            start_years,
            end_years,
        };
        protocol.validate()?;
        Ok(protocol)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        match *self {
            Self::Constant { rate_gtc_per_year } => {
                require_finite("rate_gtc_per_year", rate_gtc_per_year)
            }
            Self::LinearRamp {
                initial_gtc_per_year,
                final_gtc_per_year,
                duration_years,
            } => {
                require_finite("initial_gtc_per_year", initial_gtc_per_year)?;
                require_finite("final_gtc_per_year", final_gtc_per_year)?;
                require_positive("duration_years", duration_years)
            }
            Self::Pulse {
                baseline_gtc_per_year,
                anomaly_gtc_per_year,
                start_years,
                end_years,
            } => {
                require_finite("baseline_gtc_per_year", baseline_gtc_per_year)?;
                require_finite("anomaly_gtc_per_year", anomaly_gtc_per_year)?;
                require_non_negative("start_years", start_years)?;
                require_non_negative("end_years", end_years)?;
                if start_years >= end_years {
                    return Err(ModelError::InvalidOrdering {
                        lower: "start_years",
                        lower_value: start_years,
                        upper: "end_years",
                        upper_value: end_years,
                    });
                }
                Ok(())
            }
        }
    }

    /// Times where the protocol changes value or slope.
    pub fn integration_events(&self) -> Vec<f64> {
        match *self {
            Self::Constant { .. } => Vec::new(),
            Self::LinearRamp { duration_years, .. } => vec![duration_years],
            Self::Pulse {
                start_years,
                end_years,
                ..
            } => vec![start_years, end_years],
        }
    }

    /// Left-hand emissions value at a protocol breakpoint.
    pub fn at_left_limit(&self, time_years: f64) -> Result<f64, ModelError> {
        self.validate()?;
        require_non_negative("time_years", time_years)?;
        Ok(match *self {
            Self::Pulse {
                baseline_gtc_per_year,
                anomaly_gtc_per_year,
                start_years,
                end_years,
            } if time_years == end_years => baseline_gtc_per_year + anomaly_gtc_per_year,
            Self::Pulse {
                baseline_gtc_per_year,
                start_years,
                ..
            } if time_years == start_years => baseline_gtc_per_year,
            _ => return self.at(time_years),
        })
    }

    pub fn at(&self, time_years: f64) -> Result<f64, ModelError> {
        self.validate()?;
        require_non_negative("time_years", time_years)?;
        Ok(match *self {
            Self::Constant { rate_gtc_per_year } => rate_gtc_per_year,
            Self::LinearRamp {
                initial_gtc_per_year,
                final_gtc_per_year,
                duration_years,
            } => {
                let fraction = (time_years / duration_years).clamp(0.0, 1.0);
                initial_gtc_per_year + fraction * (final_gtc_per_year - initial_gtc_per_year)
            }
            Self::Pulse {
                baseline_gtc_per_year,
                anomaly_gtc_per_year,
                start_years,
                end_years,
            } => {
                if time_years >= start_years && time_years < end_years {
                    baseline_gtc_per_year + anomaly_gtc_per_year
                } else {
                    baseline_gtc_per_year
                }
            }
        })
    }

    /// Analytic cumulative emissions over `[start_years, end_years]`, in GtC.
    pub fn cumulative_between(&self, start_years: f64, end_years: f64) -> Result<f64, ModelError> {
        self.validate()?;
        require_non_negative("start_years", start_years)?;
        require_non_negative("end_years", end_years)?;
        if start_years > end_years {
            return Err(ModelError::InvalidOrdering {
                lower: "start_years",
                lower_value: start_years,
                upper: "end_years",
                upper_value: end_years,
            });
        }
        Ok(self.cumulative_through(end_years) - self.cumulative_through(start_years))
    }

    fn cumulative_through(&self, time_years: f64) -> f64 {
        match *self {
            Self::Constant { rate_gtc_per_year } => rate_gtc_per_year * time_years,
            Self::LinearRamp {
                initial_gtc_per_year,
                final_gtc_per_year,
                duration_years,
            } => {
                let change = final_gtc_per_year - initial_gtc_per_year;
                if time_years <= duration_years {
                    initial_gtc_per_year * time_years
                        + 0.5 * change * time_years.powi(2) / duration_years
                } else {
                    0.5 * (initial_gtc_per_year + final_gtc_per_year) * duration_years
                        + final_gtc_per_year * (time_years - duration_years)
                }
            }
            Self::Pulse {
                baseline_gtc_per_year,
                anomaly_gtc_per_year,
                start_years,
                end_years,
            } => {
                let pulse_duration =
                    (time_years.min(end_years) - start_years).clamp(0.0, end_years - start_years);
                baseline_gtc_per_year * time_years + anomaly_gtc_per_year * pulse_duration
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CarbonState {
    /// Atmospheric carbon anomaly relative to the concentration baseline, GtC.
    pub atmosphere_gtc: f64,
    /// Carbon anomaly in a reversible aggregate reservoir, GtC.
    pub reservoir_gtc: f64,
}

impl CarbonState {
    pub fn total_anomaly_gtc(&self) -> f64 {
        self.atmosphere_gtc + self.reservoir_gtc
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoBoxCarbonCycle {
    /// Atmosphere-to-reservoir transfer rate, year⁻¹.
    pub atmosphere_to_reservoir_per_year: f64,
    /// Reservoir-to-atmosphere return rate, year⁻¹.
    pub reservoir_to_atmosphere_per_year: f64,
    /// Concentration baseline corresponding to zero atmospheric anomaly, ppm.
    pub baseline_co2_ppm: f64,
    /// Atmospheric carbon mass per ppm concentration change, GtC/ppm.
    pub gtc_per_ppm: f64,
}

impl TwoBoxCarbonCycle {
    pub fn try_new(
        atmosphere_to_reservoir_per_year: f64,
        reservoir_to_atmosphere_per_year: f64,
        baseline_co2_ppm: f64,
        gtc_per_ppm: f64,
    ) -> Result<Self, ModelError> {
        let model = Self {
            atmosphere_to_reservoir_per_year,
            reservoir_to_atmosphere_per_year,
            baseline_co2_ppm,
            gtc_per_ppm,
        };
        model.validate()?;
        Ok(model)
    }

    /// Illustrative exchange parameters for deterministic experiments; not an
    /// observationally fitted carbon-cycle emulator.
    pub fn illustrative() -> Self {
        Self {
            atmosphere_to_reservoir_per_year: 0.20,
            reservoir_to_atmosphere_per_year: 0.01,
            baseline_co2_ppm: 280.0,
            gtc_per_ppm: 2.12,
        }
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        require_non_negative(
            "atmosphere_to_reservoir_per_year",
            self.atmosphere_to_reservoir_per_year,
        )?;
        require_non_negative(
            "reservoir_to_atmosphere_per_year",
            self.reservoir_to_atmosphere_per_year,
        )?;
        require_positive("baseline_co2_ppm", self.baseline_co2_ppm)?;
        require_positive("gtc_per_ppm", self.gtc_per_ppm)?;
        Ok(())
    }

    /// Sum of the reversible exchange rates, year⁻¹.
    pub fn exchange_rate_sum(&self) -> f64 {
        self.atmosphere_to_reservoir_per_year + self.reservoir_to_atmosphere_per_year
    }

    /// E-folding exchange time in years, or `None` when both rates are zero.
    pub fn exchange_timescale_years(&self) -> Option<f64> {
        let rate = self.exchange_rate_sum();
        (rate > 0.0).then(|| 1.0 / rate)
    }

    /// Long-run atmospheric fraction of a fixed total anomaly under exchange.
    pub fn equilibrium_atmospheric_fraction(&self) -> Option<f64> {
        let rate = self.exchange_rate_sum();
        (rate > 0.0).then(|| self.reservoir_to_atmosphere_per_year / rate)
    }

    /// Exact atmospheric/reservoir partition after an instantaneous pulse is
    /// placed in the atmosphere at `t = 0`, with zero subsequent emissions.
    ///
    /// This is the impulse response of the reduced reversible two-box model.
    /// It is not an observational airborne-fraction curve.
    pub fn exact_atmospheric_pulse_response(
        &self,
        atmospheric_pulse_gtc: f64,
        elapsed_years: f64,
    ) -> Result<CarbonState, ModelError> {
        self.validate()?;
        require_finite("atmospheric_pulse_gtc", atmospheric_pulse_gtc)?;
        require_non_negative("elapsed_years", elapsed_years)?;

        let exchange_rate = self.exchange_rate_sum();
        let atmosphere = if exchange_rate == 0.0 {
            atmospheric_pulse_gtc
        } else {
            let equilibrium_fraction = self.reservoir_to_atmosphere_per_year / exchange_rate;
            atmospheric_pulse_gtc
                * (equilibrium_fraction
                    + (1.0 - equilibrium_fraction) * (-exchange_rate * elapsed_years).exp())
        };
        let state = CarbonState {
            atmosphere_gtc: atmosphere,
            reservoir_gtc: atmospheric_pulse_gtc - atmosphere,
        };
        self.validate_state(state, "final")?;
        Ok(state)
    }

    /// Fraction of a unit atmospheric pulse remaining in the atmosphere.
    pub fn pulse_airborne_fraction(&self, elapsed_years: f64) -> Result<f64, ModelError> {
        Ok(self
            .exact_atmospheric_pulse_response(1.0, elapsed_years)?
            .atmosphere_gtc)
    }

    /// Exact state under a constant emissions rate.
    ///
    /// This solves the linear two-box anomaly system analytically and is an
    /// oracle for numerical integration. It is not a calibrated carbon-cycle
    /// response function.
    pub fn exact_constant_emissions(
        &self,
        initial_state: CarbonState,
        emissions_gtc_per_year: f64,
        elapsed_years: f64,
    ) -> Result<CarbonState, ModelError> {
        self.validate()?;
        self.validate_state(initial_state, "initial")?;
        require_finite("emissions_gtc_per_year", emissions_gtc_per_year)?;
        require_non_negative("elapsed_years", elapsed_years)?;

        let total_initial = initial_state.total_anomaly_gtc();
        let total = total_initial + emissions_gtc_per_year * elapsed_years;
        let exchange_rate = self.exchange_rate_sum();
        let atmosphere = if exchange_rate == 0.0 {
            initial_state.atmosphere_gtc + emissions_gtc_per_year * elapsed_years
        } else {
            let atmospheric_slope =
                self.reservoir_to_atmosphere_per_year * emissions_gtc_per_year / exchange_rate;
            let atmospheric_intercept = (emissions_gtc_per_year
                + self.reservoir_to_atmosphere_per_year * total_initial
                - atmospheric_slope)
                / exchange_rate;
            atmospheric_slope * elapsed_years
                + atmospheric_intercept
                + (initial_state.atmosphere_gtc - atmospheric_intercept)
                    * (-exchange_rate * elapsed_years).exp()
        };
        let state = CarbonState {
            atmosphere_gtc: atmosphere,
            reservoir_gtc: total - atmosphere,
        };
        self.validate_state(state, "final")?;
        Ok(state)
    }

    /// Integrated carbon-budget residual, GtC.
    pub fn integrated_mass_balance_residual(
        &self,
        initial_state: CarbonState,
        final_state: CarbonState,
        cumulative_emissions_gtc: f64,
    ) -> Result<f64, ModelError> {
        self.validate()?;
        self.validate_state(initial_state, "initial")?;
        self.validate_state(final_state, "final")?;
        require_finite("cumulative_emissions_gtc", cumulative_emissions_gtc)?;
        Ok(final_state.total_anomaly_gtc()
            - initial_state.total_anomaly_gtc()
            - cumulative_emissions_gtc)
    }

    pub fn atmospheric_co2_ppm(&self, state: CarbonState) -> Result<f64, ModelError> {
        self.validate()?;
        let concentration = self.baseline_co2_ppm + state.atmosphere_gtc / self.gtc_per_ppm;
        require_positive("atmospheric_co2_ppm", concentration)?;
        Ok(concentration)
    }

    fn validate_state(&self, state: CarbonState, stage: &'static str) -> Result<(), ModelError> {
        let atmosphere = match stage {
            "initial" => "atmosphere_gtc",
            "stage_2" => "atmosphere_gtc_stage_2",
            "stage_3" => "atmosphere_gtc_stage_3",
            "stage_4" => "atmosphere_gtc_stage_4",
            _ => "atmosphere_gtc_final",
        };
        let reservoir = match stage {
            "initial" => "reservoir_gtc",
            "stage_2" => "reservoir_gtc_stage_2",
            "stage_3" => "reservoir_gtc_stage_3",
            "stage_4" => "reservoir_gtc_stage_4",
            _ => "reservoir_gtc_final",
        };
        require_finite(atmosphere, state.atmosphere_gtc)?;
        require_finite(reservoir, state.reservoir_gtc)?;
        self.atmospheric_co2_ppm(state)?;
        Ok(())
    }

    pub fn derivatives(&self, state: CarbonState, emissions_gtc_per_year: f64) -> CarbonState {
        let uptake = self.atmosphere_to_reservoir_per_year * state.atmosphere_gtc;
        let return_flux = self.reservoir_to_atmosphere_per_year * state.reservoir_gtc;
        CarbonState {
            atmosphere_gtc: emissions_gtc_per_year - uptake + return_flux,
            reservoir_gtc: uptake - return_flux,
        }
    }

    /// Differential mass-balance residual; exactly zero in real arithmetic.
    pub fn mass_balance_residual(&self, state: CarbonState, emissions_gtc_per_year: f64) -> f64 {
        let derivative = self.derivatives(state, emissions_gtc_per_year);
        derivative.total_anomaly_gtc() - emissions_gtc_per_year
    }

    pub fn step_rk4_protocol(
        &self,
        state: CarbonState,
        start_time_years: f64,
        dt_years: f64,
        protocol: &EmissionsProtocol,
    ) -> Result<CarbonState, ModelError> {
        self.validate()?;
        protocol.validate()?;
        require_non_negative("start_time_years", start_time_years)?;
        require_positive("dt_years", dt_years)?;
        require_finite("atmosphere_gtc", state.atmosphere_gtc)?;
        require_finite("reservoir_gtc", state.reservoir_gtc)?;

        let half_time = start_time_years + 0.5 * dt_years;
        let end_time = start_time_years + dt_years;
        self.validate_state(state, "initial")?;
        let k1 = self.derivatives(state, protocol.at(start_time_years)?);
        validate_carbon_derivative(k1, 1)?;
        let stage2 = add_carbon_scaled(state, k1, 0.5 * dt_years);
        self.validate_state(stage2, "stage_2")?;
        let k2 = self.derivatives(stage2, protocol.at(half_time)?);
        validate_carbon_derivative(k2, 2)?;
        let stage3 = add_carbon_scaled(state, k2, 0.5 * dt_years);
        self.validate_state(stage3, "stage_3")?;
        let k3 = self.derivatives(stage3, protocol.at(half_time)?);
        validate_carbon_derivative(k3, 3)?;
        let stage4 = add_carbon_scaled(state, k3, dt_years);
        self.validate_state(stage4, "stage_4")?;
        let k4 = self.derivatives(stage4, protocol.at(end_time)?);
        validate_carbon_derivative(k4, 4)?;
        let next = combine_carbon_rk4(state, k1, k2, k3, k4, dt_years);
        self.validate_state(next, "final")?;
        Ok(next)
    }

    fn step_rk4_protocol_event_aligned(
        &self,
        state: CarbonState,
        start_time_years: f64,
        dt_years: f64,
        protocol: &EmissionsProtocol,
    ) -> Result<CarbonState, ModelError> {
        self.validate()?;
        protocol.validate()?;
        require_non_negative("start_time_years", start_time_years)?;
        require_positive("dt_years", dt_years)?;
        self.validate_state(state, "initial")?;

        let half_time = start_time_years + 0.5 * dt_years;
        let end_time = start_time_years + dt_years;
        let k1 = self.derivatives(state, protocol.at(start_time_years)?);
        validate_carbon_derivative(k1, 1)?;
        let stage2 = add_carbon_scaled(state, k1, 0.5 * dt_years);
        self.validate_state(stage2, "stage_2")?;
        let k2 = self.derivatives(stage2, protocol.at(half_time)?);
        validate_carbon_derivative(k2, 2)?;
        let stage3 = add_carbon_scaled(state, k2, 0.5 * dt_years);
        self.validate_state(stage3, "stage_3")?;
        let k3 = self.derivatives(stage3, protocol.at(half_time)?);
        validate_carbon_derivative(k3, 3)?;
        let stage4 = add_carbon_scaled(state, k3, dt_years);
        self.validate_state(stage4, "stage_4")?;
        let k4 = self.derivatives(stage4, protocol.at_left_limit(end_time)?);
        validate_carbon_derivative(k4, 4)?;
        let next = combine_carbon_rk4(state, k1, k2, k3, k4, dt_years);
        self.validate_state(next, "final")?;
        Ok(next)
    }

    /// Event-aligned integration over an arbitrary duration. The trajectory
    /// includes the initial state and splits exactly at protocol breakpoints.
    pub fn simulate_protocol_event_aligned(
        &self,
        initial_state: CarbonState,
        protocol: &EmissionsProtocol,
        nominal_dt_years: f64,
        duration_years: f64,
    ) -> Result<Vec<CarbonSample>, ModelError> {
        self.validate()?;
        protocol.validate()?;
        self.validate_state(initial_state, "initial")?;
        let intervals = crate::schedule::event_aligned_intervals(
            duration_years,
            nominal_dt_years,
            &protocol.integration_events(),
        )?;
        let mut samples = Vec::with_capacity(intervals.len() + 1);
        let mut state = initial_state;
        samples.push(self.sample(0.0, state, protocol.at(0.0)?)?);
        for interval in intervals {
            state = self.step_rk4_protocol_event_aligned(
                state,
                interval.start,
                interval.duration(),
                protocol,
            )?;
            samples.push(self.sample(interval.end, state, protocol.at(interval.end)?)?);
        }
        Ok(samples)
    }

    pub fn simulate_protocol_including_initial(
        &self,
        initial_state: CarbonState,
        protocol: &EmissionsProtocol,
        dt_years: f64,
        steps: usize,
    ) -> Result<Vec<CarbonSample>, ModelError> {
        self.validate()?;
        protocol.validate()?;
        validate_trajectory_capacity(dt_years, steps)?;
        if steps == 0 {
            return Err(ModelError::ZeroSteps);
        }
        require_finite("atmosphere_gtc", initial_state.atmosphere_gtc)?;
        require_finite("reservoir_gtc", initial_state.reservoir_gtc)?;

        let mut samples = Vec::with_capacity(steps + 1);
        let mut state = initial_state;
        let mut time_years = 0.0;
        samples.push(self.sample(time_years, state, protocol.at(time_years)?)?);
        for _ in 0..steps {
            state = self.step_rk4_protocol(state, time_years, dt_years, protocol)?;
            time_years += dt_years;
            samples.push(self.sample(time_years, state, protocol.at(time_years)?)?);
        }
        Ok(samples)
    }

    fn sample(
        &self,
        time_years: f64,
        state: CarbonState,
        emissions_gtc_per_year: f64,
    ) -> Result<CarbonSample, ModelError> {
        let atmospheric_co2_ppm = self.atmospheric_co2_ppm(state)?;
        Ok(CarbonSample {
            time_years,
            state,
            emissions_gtc_per_year,
            atmospheric_co2_ppm,
            radiative_forcing: try_co2_radiative_forcing_myhre1998(
                atmospheric_co2_ppm,
                self.baseline_co2_ppm,
            )?,
            mass_balance_residual: self.mass_balance_residual(state, emissions_gtc_per_year),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CarbonSample {
    pub time_years: f64,
    pub state: CarbonState,
    pub emissions_gtc_per_year: f64,
    pub atmospheric_co2_ppm: f64,
    pub radiative_forcing: f64,
    pub mass_balance_residual: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CarbonClimateState {
    pub carbon: CarbonState,
    pub temperature: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CarbonClimateModel {
    pub carbon_cycle: TwoBoxCarbonCycle,
    pub climate: OneBoxClimateModel,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CarbonClimateSample {
    pub time_years: f64,
    pub state: CarbonClimateState,
    pub emissions_gtc_per_year: f64,
    pub atmospheric_co2_ppm: f64,
    pub radiative_forcing: f64,
    pub radiative_imbalance: f64,
    pub carbon_mass_balance_residual: f64,
}

impl CarbonClimateModel {
    pub fn try_new(
        carbon_cycle: TwoBoxCarbonCycle,
        climate: OneBoxClimateModel,
    ) -> Result<Self, ModelError> {
        carbon_cycle.validate()?;
        climate.validate()?;
        Ok(Self {
            carbon_cycle,
            climate,
        })
    }

    /// Derivatives per year. Climate tendency is converted from K/s to K/year.
    pub fn derivatives(
        &self,
        state: CarbonClimateState,
        emissions_gtc_per_year: f64,
    ) -> Result<CarbonClimateState, ModelError> {
        self.carbon_cycle.validate()?;
        self.climate.validate()?;
        self.carbon_cycle.validate_state(state.carbon, "initial")?;
        require_positive("temperature", state.temperature)?;
        require_finite("emissions_gtc_per_year", emissions_gtc_per_year)?;
        let carbon = self
            .carbon_cycle
            .derivatives(state.carbon, emissions_gtc_per_year);
        let concentration = self.carbon_cycle.atmospheric_co2_ppm(state.carbon)?;
        let forcing =
            try_co2_radiative_forcing_myhre1998(concentration, self.carbon_cycle.baseline_co2_ppm)?;
        Ok(CarbonClimateState {
            carbon,
            temperature: self.climate.tendency(state.temperature, forcing) * SECONDS_PER_YEAR,
        })
    }

    pub fn step_rk4_protocol(
        &self,
        state: CarbonClimateState,
        start_time_years: f64,
        dt_years: f64,
        protocol: &EmissionsProtocol,
    ) -> Result<CarbonClimateState, ModelError> {
        self.carbon_cycle.validate()?;
        self.climate.validate()?;
        protocol.validate()?;
        require_positive("temperature", state.temperature)?;
        require_non_negative("start_time_years", start_time_years)?;
        require_positive("dt_years", dt_years)?;

        let half_time = start_time_years + 0.5 * dt_years;
        let end_time = start_time_years + dt_years;
        validate_chain_state(self, state, "initial")?;
        let k1 = self.derivatives(state, protocol.at(start_time_years)?)?;
        validate_chain_derivative(k1, 1)?;
        let stage2 = add_chain_scaled(state, k1, 0.5 * dt_years);
        validate_chain_state(self, stage2, "stage_2")?;
        let k2 = self.derivatives(stage2, protocol.at(half_time)?)?;
        validate_chain_derivative(k2, 2)?;
        let stage3 = add_chain_scaled(state, k2, 0.5 * dt_years);
        validate_chain_state(self, stage3, "stage_3")?;
        let k3 = self.derivatives(stage3, protocol.at(half_time)?)?;
        validate_chain_derivative(k3, 3)?;
        let stage4 = add_chain_scaled(state, k3, dt_years);
        validate_chain_state(self, stage4, "stage_4")?;
        let k4 = self.derivatives(stage4, protocol.at(end_time)?)?;
        validate_chain_derivative(k4, 4)?;
        let next = combine_chain_rk4(state, k1, k2, k3, k4, dt_years);
        validate_chain_state(self, next, "final")?;
        Ok(next)
    }

    fn step_rk4_protocol_event_aligned(
        &self,
        state: CarbonClimateState,
        start_time_years: f64,
        dt_years: f64,
        protocol: &EmissionsProtocol,
    ) -> Result<CarbonClimateState, ModelError> {
        self.carbon_cycle.validate()?;
        self.climate.validate()?;
        protocol.validate()?;
        require_non_negative("start_time_years", start_time_years)?;
        require_positive("dt_years", dt_years)?;
        validate_chain_state(self, state, "initial")?;

        let half_time = start_time_years + 0.5 * dt_years;
        let end_time = start_time_years + dt_years;
        let k1 = self.derivatives(state, protocol.at(start_time_years)?)?;
        validate_chain_derivative(k1, 1)?;
        let stage2 = add_chain_scaled(state, k1, 0.5 * dt_years);
        validate_chain_state(self, stage2, "stage_2")?;
        let k2 = self.derivatives(stage2, protocol.at(half_time)?)?;
        validate_chain_derivative(k2, 2)?;
        let stage3 = add_chain_scaled(state, k2, 0.5 * dt_years);
        validate_chain_state(self, stage3, "stage_3")?;
        let k3 = self.derivatives(stage3, protocol.at(half_time)?)?;
        validate_chain_derivative(k3, 3)?;
        let stage4 = add_chain_scaled(state, k3, dt_years);
        validate_chain_state(self, stage4, "stage_4")?;
        let k4 = self.derivatives(stage4, protocol.at_left_limit(end_time)?)?;
        validate_chain_derivative(k4, 4)?;
        let next = combine_chain_rk4(state, k1, k2, k3, k4, dt_years);
        validate_chain_state(self, next, "final")?;
        Ok(next)
    }

    pub fn simulate_protocol_event_aligned(
        &self,
        initial_state: CarbonClimateState,
        protocol: &EmissionsProtocol,
        nominal_dt_years: f64,
        duration_years: f64,
    ) -> Result<Vec<CarbonClimateSample>, ModelError> {
        let intervals = crate::schedule::event_aligned_intervals(
            duration_years,
            nominal_dt_years,
            &protocol.integration_events(),
        )?;
        validate_chain_state(self, initial_state, "initial")?;
        let mut samples = Vec::with_capacity(intervals.len() + 1);
        let mut state = initial_state;
        samples.push(self.sample(0.0, state, protocol.at(0.0)?)?);
        for interval in intervals {
            state = self.step_rk4_protocol_event_aligned(
                state,
                interval.start,
                interval.duration(),
                protocol,
            )?;
            samples.push(self.sample(interval.end, state, protocol.at(interval.end)?)?);
        }
        Ok(samples)
    }

    pub fn simulate_protocol_including_initial(
        &self,
        initial_state: CarbonClimateState,
        protocol: &EmissionsProtocol,
        dt_years: f64,
        steps: usize,
    ) -> Result<Vec<CarbonClimateSample>, ModelError> {
        validate_trajectory_capacity(dt_years, steps)?;
        if steps == 0 {
            return Err(ModelError::ZeroSteps);
        }
        let mut samples = Vec::with_capacity(steps + 1);
        let mut state = initial_state;
        let mut time_years = 0.0;
        samples.push(self.sample(time_years, state, protocol.at(time_years)?)?);
        for _ in 0..steps {
            state = self.step_rk4_protocol(state, time_years, dt_years, protocol)?;
            time_years += dt_years;
            samples.push(self.sample(time_years, state, protocol.at(time_years)?)?);
        }
        Ok(samples)
    }

    fn sample(
        &self,
        time_years: f64,
        state: CarbonClimateState,
        emissions_gtc_per_year: f64,
    ) -> Result<CarbonClimateSample, ModelError> {
        let atmospheric_co2_ppm = self.carbon_cycle.atmospheric_co2_ppm(state.carbon)?;
        let radiative_forcing = try_co2_radiative_forcing_myhre1998(
            atmospheric_co2_ppm,
            self.carbon_cycle.baseline_co2_ppm,
        )?;
        Ok(CarbonClimateSample {
            time_years,
            state,
            emissions_gtc_per_year,
            atmospheric_co2_ppm,
            radiative_forcing,
            radiative_imbalance: self
                .climate
                .radiative_imbalance(state.temperature, radiative_forcing),
            carbon_mass_balance_residual: self
                .carbon_cycle
                .mass_balance_residual(state.carbon, emissions_gtc_per_year),
        })
    }
}

fn validate_carbon_derivative(derivative: CarbonState, stage: usize) -> Result<(), ModelError> {
    let atmosphere = match stage {
        1 => "atmosphere_tendency_stage_1",
        2 => "atmosphere_tendency_stage_2",
        3 => "atmosphere_tendency_stage_3",
        _ => "atmosphere_tendency_stage_4",
    };
    let reservoir = match stage {
        1 => "reservoir_tendency_stage_1",
        2 => "reservoir_tendency_stage_2",
        3 => "reservoir_tendency_stage_3",
        _ => "reservoir_tendency_stage_4",
    };
    require_finite(atmosphere, derivative.atmosphere_gtc)?;
    require_finite(reservoir, derivative.reservoir_gtc)?;
    Ok(())
}

fn validate_chain_state(
    model: &CarbonClimateModel,
    state: CarbonClimateState,
    stage: &'static str,
) -> Result<(), ModelError> {
    model.carbon_cycle.validate_state(state.carbon, stage)?;
    let temperature = match stage {
        "initial" => "temperature",
        "stage_2" => "temperature_stage_2",
        "stage_3" => "temperature_stage_3",
        "stage_4" => "temperature_stage_4",
        _ => "temperature_final",
    };
    require_positive(temperature, state.temperature)?;
    Ok(())
}

fn validate_chain_derivative(
    derivative: CarbonClimateState,
    stage: usize,
) -> Result<(), ModelError> {
    validate_carbon_derivative(derivative.carbon, stage)?;
    let temperature = match stage {
        1 => "temperature_tendency_stage_1",
        2 => "temperature_tendency_stage_2",
        3 => "temperature_tendency_stage_3",
        _ => "temperature_tendency_stage_4",
    };
    require_finite(temperature, derivative.temperature)?;
    Ok(())
}

fn add_carbon_scaled(state: CarbonState, derivative: CarbonState, scale: f64) -> CarbonState {
    CarbonState {
        atmosphere_gtc: state.atmosphere_gtc + scale * derivative.atmosphere_gtc,
        reservoir_gtc: state.reservoir_gtc + scale * derivative.reservoir_gtc,
    }
}

fn combine_carbon_rk4(
    state: CarbonState,
    k1: CarbonState,
    k2: CarbonState,
    k3: CarbonState,
    k4: CarbonState,
    dt: f64,
) -> CarbonState {
    CarbonState {
        atmosphere_gtc: state.atmosphere_gtc
            + dt * (k1.atmosphere_gtc
                + 2.0 * k2.atmosphere_gtc
                + 2.0 * k3.atmosphere_gtc
                + k4.atmosphere_gtc)
                / 6.0,
        reservoir_gtc: state.reservoir_gtc
            + dt * (k1.reservoir_gtc
                + 2.0 * k2.reservoir_gtc
                + 2.0 * k3.reservoir_gtc
                + k4.reservoir_gtc)
                / 6.0,
    }
}

fn add_chain_scaled(
    state: CarbonClimateState,
    derivative: CarbonClimateState,
    scale: f64,
) -> CarbonClimateState {
    CarbonClimateState {
        carbon: add_carbon_scaled(state.carbon, derivative.carbon, scale),
        temperature: state.temperature + scale * derivative.temperature,
    }
}

fn combine_chain_rk4(
    state: CarbonClimateState,
    k1: CarbonClimateState,
    k2: CarbonClimateState,
    k3: CarbonClimateState,
    k4: CarbonClimateState,
    dt: f64,
) -> CarbonClimateState {
    CarbonClimateState {
        carbon: combine_carbon_rk4(state.carbon, k1.carbon, k2.carbon, k3.carbon, k4.carbon, dt),
        temperature: state.temperature
            + dt * (k1.temperature + 2.0 * k2.temperature + 2.0 * k3.temperature + k4.temperature)
                / 6.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_state() -> CarbonState {
        CarbonState {
            atmosphere_gtc: 0.0,
            reservoir_gtc: 0.0,
        }
    }

    #[test]
    fn derivative_level_carbon_budget_closes() {
        let model = TwoBoxCarbonCycle::illustrative();
        let state = CarbonState {
            atmosphere_gtc: 100.0,
            reservoir_gtc: 50.0,
        };
        assert!(model.mass_balance_residual(state, 10.0).abs() < 1e-12);
    }

    #[test]
    fn constant_emissions_accumulate_exact_total_anomaly() {
        let model = TwoBoxCarbonCycle::illustrative();
        let protocol = EmissionsProtocol::constant(10.0).unwrap();
        let samples = model
            .simulate_protocol_including_initial(zero_state(), &protocol, 0.1, 100)
            .unwrap();
        let total = samples.last().unwrap().state.total_anomaly_gtc();
        assert!((total - 100.0).abs() < 1e-9);
        assert_eq!(samples.len(), 101);
    }

    #[test]
    fn zero_emissions_preserve_total_carbon_during_exchange() {
        let model = TwoBoxCarbonCycle::illustrative();
        let protocol = EmissionsProtocol::constant(0.0).unwrap();
        let initial = CarbonState {
            atmosphere_gtc: 100.0,
            reservoir_gtc: 0.0,
        };
        let samples = model
            .simulate_protocol_including_initial(initial, &protocol, 0.05, 400)
            .unwrap();
        assert!(
            samples
                .iter()
                .all(|sample| { (sample.state.total_anomaly_gtc() - 100.0).abs() < 1e-9 })
        );
        assert!(samples.last().unwrap().state.atmosphere_gtc < 100.0);
    }

    #[test]
    fn atmospheric_mass_maps_to_concentration() {
        let model = TwoBoxCarbonCycle::illustrative();
        let concentration = model
            .atmospheric_co2_ppm(CarbonState {
                atmosphere_gtc: 2.12,
                reservoir_gtc: 0.0,
            })
            .unwrap();
        assert!((concentration - 281.0).abs() < 1e-12);
    }

    #[test]
    fn zero_emissions_baseline_remains_stationary() {
        let model = CarbonClimateModel::try_new(
            TwoBoxCarbonCycle::illustrative(),
            OneBoxClimateModel::earthlike(),
        )
        .unwrap();
        let protocol = EmissionsProtocol::constant(0.0).unwrap();
        let initial = CarbonClimateState {
            carbon: zero_state(),
            temperature: 288.0,
        };
        let samples = model
            .simulate_protocol_including_initial(initial, &protocol, 1.0, 20)
            .unwrap();
        assert!(samples.iter().all(|sample| {
            sample.state.carbon.total_anomaly_gtc().abs() < 1e-12
                && (sample.state.temperature - 288.0).abs() < 1e-12
                && sample.radiative_forcing.abs() < 1e-12
        }));
    }

    #[test]
    fn positive_emissions_raise_concentration_and_temperature() {
        let model = CarbonClimateModel::try_new(
            TwoBoxCarbonCycle::illustrative(),
            OneBoxClimateModel::earthlike(),
        )
        .unwrap();
        let protocol = EmissionsProtocol::constant(10.0).unwrap();
        let initial = CarbonClimateState {
            carbon: zero_state(),
            temperature: 288.0,
        };
        let samples = model
            .simulate_protocol_including_initial(initial, &protocol, 0.1, 500)
            .unwrap();
        let final_sample = samples.last().unwrap();
        assert!(final_sample.atmospheric_co2_ppm > 280.0);
        assert!(final_sample.radiative_forcing > 0.0);
        assert!(final_sample.state.temperature > 288.0);
    }
    #[test]
    fn carbon_integrators_fail_before_invalid_concentration_stages() {
        let cycle = TwoBoxCarbonCycle::illustrative();
        let protocol = EmissionsProtocol::constant(-10_000.0).unwrap();
        assert!(
            cycle
                .step_rk4_protocol(zero_state(), 0.0, 1.0, &protocol)
                .is_err()
        );

        let chain = CarbonClimateModel::try_new(cycle, OneBoxClimateModel::earthlike()).unwrap();
        let initial = CarbonClimateState {
            carbon: zero_state(),
            temperature: 288.0,
        };
        assert!(
            chain
                .step_rk4_protocol(initial, 0.0, 1.0, &protocol)
                .is_err()
        );
    }

    #[test]
    fn analytic_emissions_integrals_match_protocol_geometry() {
        let constant = EmissionsProtocol::constant(10.0).unwrap();
        assert_eq!(constant.cumulative_between(2.0, 5.0).unwrap(), 30.0);

        let ramp = EmissionsProtocol::linear_ramp(0.0, 10.0, 10.0).unwrap();
        assert!((ramp.cumulative_between(0.0, 10.0).unwrap() - 50.0).abs() < 1e-12);
        assert!((ramp.cumulative_between(0.0, 20.0).unwrap() - 150.0).abs() < 1e-12);

        let pulse = EmissionsProtocol::pulse(1.0, 4.0, 2.0, 5.0).unwrap();
        assert!((pulse.cumulative_between(0.0, 10.0).unwrap() - 22.0).abs() < 1e-12);
        assert_eq!(pulse.cumulative_between(3.0, 3.0).unwrap(), 0.0);
    }

    #[test]
    fn exact_constant_emissions_matches_resolved_rk4() {
        let model = TwoBoxCarbonCycle::illustrative();
        let initial = CarbonState {
            atmosphere_gtc: 20.0,
            reservoir_gtc: 5.0,
        };
        let emissions = 10.0;
        let elapsed = 20.0;
        let exact = model
            .exact_constant_emissions(initial, emissions, elapsed)
            .unwrap();
        let protocol = EmissionsProtocol::constant(emissions).unwrap();
        let numerical = model
            .simulate_protocol_including_initial(initial, &protocol, 0.001, 20_000)
            .unwrap();
        let final_state = numerical.last().unwrap().state;
        assert!((final_state.atmosphere_gtc - exact.atmosphere_gtc).abs() < 1e-8);
        assert!((final_state.reservoir_gtc - exact.reservoir_gtc).abs() < 1e-8);
        let cumulative = protocol.cumulative_between(0.0, elapsed).unwrap();
        assert!(
            model
                .integrated_mass_balance_residual(initial, final_state, cumulative)
                .unwrap()
                .abs()
                < 1e-8
        );
    }

    #[test]
    fn zero_exchange_exact_solution_keeps_reservoir_fixed() {
        let model = TwoBoxCarbonCycle::try_new(0.0, 0.0, 280.0, 2.12).unwrap();
        assert_eq!(model.exchange_timescale_years(), None);
        assert_eq!(model.equilibrium_atmospheric_fraction(), None);
        let initial = CarbonState {
            atmosphere_gtc: 5.0,
            reservoir_gtc: 7.0,
        };
        let final_state = model.exact_constant_emissions(initial, 3.0, 4.0).unwrap();
        assert_eq!(final_state.atmosphere_gtc, 17.0);
        assert_eq!(final_state.reservoir_gtc, 7.0);
    }

    #[test]
    fn event_aligned_pulse_preserves_analytic_emissions_budget() {
        let model = TwoBoxCarbonCycle::illustrative();
        let protocol = EmissionsProtocol::pulse(1.0, 9.0, 2.5, 7.25).unwrap();
        let samples = model
            .simulate_protocol_event_aligned(
                CarbonState {
                    atmosphere_gtc: 0.0,
                    reservoir_gtc: 0.0,
                },
                &protocol,
                4.0,
                10.0,
            )
            .unwrap();
        let final_total = samples.last().unwrap().state.total_anomaly_gtc();
        let expected = protocol.cumulative_between(0.0, 10.0).unwrap();
        assert!((final_total - expected).abs() < 1.0e-10);
        assert!(samples.iter().any(|sample| sample.time_years == 2.5));
        assert!(samples.iter().any(|sample| sample.time_years == 7.25));
    }

    #[test]
    fn atmospheric_pulse_response_conserves_mass_and_relaxes_exactly() {
        let model = TwoBoxCarbonCycle::illustrative();
        let pulse = 100.0;
        let initial = model.exact_atmospheric_pulse_response(pulse, 0.0).unwrap();
        assert_eq!(initial.atmosphere_gtc, pulse);
        assert_eq!(initial.reservoir_gtc, 0.0);

        let late = model
            .exact_atmospheric_pulse_response(pulse, 1.0e6)
            .unwrap();
        let expected_fraction = model.equilibrium_atmospheric_fraction().unwrap();
        assert!((late.total_anomaly_gtc() - pulse).abs() < 1.0e-12);
        assert!((late.atmosphere_gtc / pulse - expected_fraction).abs() < 1.0e-12);
    }

    #[test]
    fn pulse_airborne_fraction_matches_zero_emissions_rk4() {
        let model = TwoBoxCarbonCycle::illustrative();
        let pulse = 100.0;
        let elapsed = 25.0;
        let exact = model
            .exact_atmospheric_pulse_response(pulse, elapsed)
            .unwrap();
        let protocol = EmissionsProtocol::constant(0.0).unwrap();
        let numerical = model
            .simulate_protocol_including_initial(
                CarbonState {
                    atmosphere_gtc: pulse,
                    reservoir_gtc: 0.0,
                },
                &protocol,
                0.01,
                2_500,
            )
            .unwrap();
        let final_state = numerical.last().unwrap().state;
        assert!((final_state.atmosphere_gtc - exact.atmosphere_gtc).abs() < 1.0e-8);
        assert!((final_state.reservoir_gtc - exact.reservoir_gtc).abs() < 1.0e-8);
        assert!(
            (model.pulse_airborne_fraction(elapsed).unwrap() - exact.atmosphere_gtc / pulse).abs()
                < 1.0e-12
        );
    }
}
