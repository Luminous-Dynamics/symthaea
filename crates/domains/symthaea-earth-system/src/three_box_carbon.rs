// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Mass-conserving three-reservoir carbon-cycle baseline.
//!
//! This model adds a second exchange timescale to the two-box oracle while
//! remaining deliberately linear and configurable. The reservoirs are labels
//! for fast and slow reversible storage, not calibrated ocean or land pools.

use crate::carbon_cycle::EmissionsProtocol;
use crate::energy_balance::try_co2_radiative_forcing_myhre1998;
use crate::error::{ModelError, require_finite, require_non_negative, require_positive};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThreeBoxCarbonState {
    pub atmosphere_gtc: f64,
    pub fast_reservoir_gtc: f64,
    pub slow_reservoir_gtc: f64,
}

impl ThreeBoxCarbonState {
    pub fn total_anomaly_gtc(&self) -> f64 {
        self.atmosphere_gtc + self.fast_reservoir_gtc + self.slow_reservoir_gtc
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThreeBoxCarbonCycle {
    pub atmosphere_to_fast_per_year: f64,
    pub fast_to_atmosphere_per_year: f64,
    pub fast_to_slow_per_year: f64,
    pub slow_to_fast_per_year: f64,
    pub baseline_co2_ppm: f64,
    pub gtc_per_ppm: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThreeBoxEquilibriumFractions {
    pub atmosphere: f64,
    pub fast_reservoir: f64,
    pub slow_reservoir: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThreeBoxCarbonSample {
    pub time_years: f64,
    pub state: ThreeBoxCarbonState,
    pub emissions_gtc_per_year: f64,
    pub atmospheric_co2_ppm: f64,
    pub radiative_forcing: f64,
    pub mass_balance_residual: f64,
}

impl ThreeBoxCarbonCycle {
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        atmosphere_to_fast_per_year: f64,
        fast_to_atmosphere_per_year: f64,
        fast_to_slow_per_year: f64,
        slow_to_fast_per_year: f64,
        baseline_co2_ppm: f64,
        gtc_per_ppm: f64,
    ) -> Result<Self, ModelError> {
        let model = Self {
            atmosphere_to_fast_per_year,
            fast_to_atmosphere_per_year,
            fast_to_slow_per_year,
            slow_to_fast_per_year,
            baseline_co2_ppm,
            gtc_per_ppm,
        };
        model.validate()?;
        Ok(model)
    }

    /// Illustrative fast/slow exchange rates, not an observational fit.
    pub fn illustrative() -> Self {
        Self {
            atmosphere_to_fast_per_year: 0.20,
            fast_to_atmosphere_per_year: 0.04,
            fast_to_slow_per_year: 0.02,
            slow_to_fast_per_year: 0.001,
            baseline_co2_ppm: 280.0,
            gtc_per_ppm: 2.12,
        }
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        require_positive(
            "atmosphere_to_fast_per_year",
            self.atmosphere_to_fast_per_year,
        )?;
        require_positive(
            "fast_to_atmosphere_per_year",
            self.fast_to_atmosphere_per_year,
        )?;
        require_positive("fast_to_slow_per_year", self.fast_to_slow_per_year)?;
        require_positive("slow_to_fast_per_year", self.slow_to_fast_per_year)?;
        require_positive("baseline_co2_ppm", self.baseline_co2_ppm)?;
        require_positive("gtc_per_ppm", self.gtc_per_ppm)?;
        Ok(())
    }

    pub fn equilibrium_fractions(&self) -> Result<ThreeBoxEquilibriumFractions, ModelError> {
        self.validate()?;
        let fast_per_atmosphere =
            self.atmosphere_to_fast_per_year / self.fast_to_atmosphere_per_year;
        let slow_per_fast = self.fast_to_slow_per_year / self.slow_to_fast_per_year;
        let denominator = 1.0 + fast_per_atmosphere * (1.0 + slow_per_fast);
        Ok(ThreeBoxEquilibriumFractions {
            atmosphere: 1.0 / denominator,
            fast_reservoir: fast_per_atmosphere / denominator,
            slow_reservoir: fast_per_atmosphere * slow_per_fast / denominator,
        })
    }

    pub fn equilibrium_partition(
        &self,
        total_anomaly_gtc: f64,
    ) -> Result<ThreeBoxCarbonState, ModelError> {
        require_finite("total_anomaly_gtc", total_anomaly_gtc)?;
        let fractions = self.equilibrium_fractions()?;
        Ok(ThreeBoxCarbonState {
            atmosphere_gtc: fractions.atmosphere * total_anomaly_gtc,
            fast_reservoir_gtc: fractions.fast_reservoir * total_anomaly_gtc,
            slow_reservoir_gtc: fractions.slow_reservoir * total_anomaly_gtc,
        })
    }

    pub fn atmospheric_co2_ppm(&self, state: ThreeBoxCarbonState) -> Result<f64, ModelError> {
        self.validate_state(state, "state")?;
        let concentration = self.baseline_co2_ppm + state.atmosphere_gtc / self.gtc_per_ppm;
        require_positive("atmospheric_co2_ppm", concentration)?;
        Ok(concentration)
    }

    pub fn derivatives(
        &self,
        state: ThreeBoxCarbonState,
        emissions_gtc_per_year: f64,
    ) -> ThreeBoxCarbonState {
        let atmosphere_to_fast = self.atmosphere_to_fast_per_year * state.atmosphere_gtc;
        let fast_to_atmosphere = self.fast_to_atmosphere_per_year * state.fast_reservoir_gtc;
        let fast_to_slow = self.fast_to_slow_per_year * state.fast_reservoir_gtc;
        let slow_to_fast = self.slow_to_fast_per_year * state.slow_reservoir_gtc;
        ThreeBoxCarbonState {
            atmosphere_gtc: emissions_gtc_per_year - atmosphere_to_fast + fast_to_atmosphere,
            fast_reservoir_gtc: atmosphere_to_fast - fast_to_atmosphere - fast_to_slow
                + slow_to_fast,
            slow_reservoir_gtc: fast_to_slow - slow_to_fast,
        }
    }

    pub fn mass_balance_residual(
        &self,
        state: ThreeBoxCarbonState,
        emissions_gtc_per_year: f64,
    ) -> f64 {
        self.derivatives(state, emissions_gtc_per_year)
            .total_anomaly_gtc()
            - emissions_gtc_per_year
    }

    pub fn step_rk4_protocol(
        &self,
        state: ThreeBoxCarbonState,
        start_time_years: f64,
        dt_years: f64,
        protocol: &EmissionsProtocol,
    ) -> Result<ThreeBoxCarbonState, ModelError> {
        self.step_rk4_protocol_with_endpoint(state, start_time_years, dt_years, protocol, false)
    }

    fn step_rk4_protocol_with_endpoint(
        &self,
        state: ThreeBoxCarbonState,
        start_time_years: f64,
        dt_years: f64,
        protocol: &EmissionsProtocol,
        left_endpoint: bool,
    ) -> Result<ThreeBoxCarbonState, ModelError> {
        self.validate()?;
        protocol.validate()?;
        require_non_negative("start_time_years", start_time_years)?;
        require_positive("dt_years", dt_years)?;
        self.validate_state(state, "initial")?;
        let half_time = start_time_years + 0.5 * dt_years;
        let end_time = start_time_years + dt_years;
        let k1 = self.derivatives(state, protocol.at(start_time_years)?);
        validate_derivative(k1, 1)?;
        let stage2 = add_scaled(state, k1, 0.5 * dt_years);
        self.validate_state(stage2, "stage_2")?;
        let k2 = self.derivatives(stage2, protocol.at(half_time)?);
        validate_derivative(k2, 2)?;
        let stage3 = add_scaled(state, k2, 0.5 * dt_years);
        self.validate_state(stage3, "stage_3")?;
        let k3 = self.derivatives(stage3, protocol.at(half_time)?);
        validate_derivative(k3, 3)?;
        let stage4 = add_scaled(state, k3, dt_years);
        self.validate_state(stage4, "stage_4")?;
        let endpoint_emissions = if left_endpoint {
            protocol.at_left_limit(end_time)?
        } else {
            protocol.at(end_time)?
        };
        let k4 = self.derivatives(stage4, endpoint_emissions);
        validate_derivative(k4, 4)?;
        let next = combine_rk4(state, k1, k2, k3, k4, dt_years);
        self.validate_state(next, "final")?;
        Ok(next)
    }

    pub fn simulate_protocol_event_aligned(
        &self,
        initial_state: ThreeBoxCarbonState,
        protocol: &EmissionsProtocol,
        nominal_dt_years: f64,
        duration_years: f64,
    ) -> Result<Vec<ThreeBoxCarbonSample>, ModelError> {
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
            state = self.step_rk4_protocol_with_endpoint(
                state,
                interval.start,
                interval.duration(),
                protocol,
                true,
            )?;
            samples.push(self.sample(interval.end, state, protocol.at(interval.end)?)?);
        }
        Ok(samples)
    }

    fn sample(
        &self,
        time_years: f64,
        state: ThreeBoxCarbonState,
        emissions_gtc_per_year: f64,
    ) -> Result<ThreeBoxCarbonSample, ModelError> {
        let atmospheric_co2_ppm = self.atmospheric_co2_ppm(state)?;
        Ok(ThreeBoxCarbonSample {
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

    fn validate_state(
        &self,
        state: ThreeBoxCarbonState,
        stage: &'static str,
    ) -> Result<(), ModelError> {
        let atmosphere = match stage {
            "initial" => "atmosphere_gtc_initial",
            "stage_2" => "atmosphere_gtc_stage_2",
            "stage_3" => "atmosphere_gtc_stage_3",
            "stage_4" => "atmosphere_gtc_stage_4",
            _ => "atmosphere_gtc",
        };
        require_finite(atmosphere, state.atmosphere_gtc)?;
        require_finite("fast_reservoir_gtc", state.fast_reservoir_gtc)?;
        require_finite("slow_reservoir_gtc", state.slow_reservoir_gtc)?;
        let concentration = self.baseline_co2_ppm + state.atmosphere_gtc / self.gtc_per_ppm;
        require_positive("atmospheric_co2_ppm", concentration)?;
        Ok(())
    }
}

fn add_scaled(
    state: ThreeBoxCarbonState,
    derivative: ThreeBoxCarbonState,
    scale: f64,
) -> ThreeBoxCarbonState {
    ThreeBoxCarbonState {
        atmosphere_gtc: state.atmosphere_gtc + scale * derivative.atmosphere_gtc,
        fast_reservoir_gtc: state.fast_reservoir_gtc + scale * derivative.fast_reservoir_gtc,
        slow_reservoir_gtc: state.slow_reservoir_gtc + scale * derivative.slow_reservoir_gtc,
    }
}

fn combine_rk4(
    state: ThreeBoxCarbonState,
    k1: ThreeBoxCarbonState,
    k2: ThreeBoxCarbonState,
    k3: ThreeBoxCarbonState,
    k4: ThreeBoxCarbonState,
    dt: f64,
) -> ThreeBoxCarbonState {
    ThreeBoxCarbonState {
        atmosphere_gtc: state.atmosphere_gtc
            + dt * (k1.atmosphere_gtc
                + 2.0 * k2.atmosphere_gtc
                + 2.0 * k3.atmosphere_gtc
                + k4.atmosphere_gtc)
                / 6.0,
        fast_reservoir_gtc: state.fast_reservoir_gtc
            + dt * (k1.fast_reservoir_gtc
                + 2.0 * k2.fast_reservoir_gtc
                + 2.0 * k3.fast_reservoir_gtc
                + k4.fast_reservoir_gtc)
                / 6.0,
        slow_reservoir_gtc: state.slow_reservoir_gtc
            + dt * (k1.slow_reservoir_gtc
                + 2.0 * k2.slow_reservoir_gtc
                + 2.0 * k3.slow_reservoir_gtc
                + k4.slow_reservoir_gtc)
                / 6.0,
    }
}

fn validate_derivative(derivative: ThreeBoxCarbonState, stage: usize) -> Result<(), ModelError> {
    let atmosphere = match stage {
        1 => "atmosphere_tendency_stage_1",
        2 => "atmosphere_tendency_stage_2",
        3 => "atmosphere_tendency_stage_3",
        _ => "atmosphere_tendency_stage_4",
    };
    require_finite(atmosphere, derivative.atmosphere_gtc)?;
    require_finite("fast_reservoir_tendency", derivative.fast_reservoir_gtc)?;
    require_finite("slow_reservoir_tendency", derivative.slow_reservoir_gtc)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equilibrium_fractions_are_normalized_and_stationary() {
        let model = ThreeBoxCarbonCycle::illustrative();
        let fractions = model.equilibrium_fractions().unwrap();
        assert!(
            (fractions.atmosphere + fractions.fast_reservoir + fractions.slow_reservoir - 1.0)
                .abs()
                < 1e-12
        );
        let state = model.equilibrium_partition(100.0).unwrap();
        let derivative = model.derivatives(state, 0.0);
        assert!(derivative.atmosphere_gtc.abs() < 1e-12);
        assert!(derivative.fast_reservoir_gtc.abs() < 1e-12);
        assert!(derivative.slow_reservoir_gtc.abs() < 1e-12);
    }

    #[test]
    fn derivative_conserves_total_carbon() {
        let model = ThreeBoxCarbonCycle::illustrative();
        let state = ThreeBoxCarbonState {
            atmosphere_gtc: 40.0,
            fast_reservoir_gtc: 20.0,
            slow_reservoir_gtc: 10.0,
        };
        assert!(model.mass_balance_residual(state, 7.0).abs() < 1e-12);
    }

    #[test]
    fn pulse_integration_closes_the_emissions_budget() {
        let model = ThreeBoxCarbonCycle::illustrative();
        let protocol = EmissionsProtocol::pulse(1.0, 9.0, 2.5, 7.25).unwrap();
        let samples = model
            .simulate_protocol_event_aligned(
                ThreeBoxCarbonState {
                    atmosphere_gtc: 0.0,
                    fast_reservoir_gtc: 0.0,
                    slow_reservoir_gtc: 0.0,
                },
                &protocol,
                0.25,
                10.0,
            )
            .unwrap();
        let total = samples.last().unwrap().state.total_anomaly_gtc();
        let expected = protocol.cumulative_between(0.0, 10.0).unwrap();
        assert!((total - expected).abs() < 1e-10);
    }
}
