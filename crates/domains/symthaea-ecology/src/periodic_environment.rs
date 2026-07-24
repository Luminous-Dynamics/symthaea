// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Smooth periodic environmental drivers for seasonal experiments.
//!
//! These signals are deterministic forcing protocols, not climatologies. The
//! caller chooses the time unit, period, phases, and ecological calibration.

use crate::environment::EnvironmentalDrivers;
use crate::environment_timeline::EnvironmentalDriverSource;
use crate::error::{ModelError, require_finite, require_non_negative, require_positive};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeriodicSignal {
    pub mean: f64,
    pub amplitude: f64,
    pub phase_radians: f64,
}

impl PeriodicSignal {
    pub fn try_new(mean: f64, amplitude: f64, phase_radians: f64) -> Result<Self, ModelError> {
        require_finite("periodic_mean", mean)?;
        require_finite("periodic_amplitude", amplitude)?;
        require_finite("phase_radians", phase_radians)?;
        Ok(Self {
            mean,
            amplitude,
            phase_radians,
        })
    }

    pub fn at_angle(&self, angle: f64) -> f64 {
        self.mean + self.amplitude * (angle + self.phase_radians).sin()
    }

    pub fn minimum(&self) -> f64 {
        self.mean - self.amplitude.abs()
    }

    pub fn maximum(&self) -> f64 {
        self.mean + self.amplitude.abs()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeriodicEnvironment {
    pub temperature: PeriodicSignal,
    pub productivity: PeriodicSignal,
    pub disturbance: PeriodicSignal,
    pub period: f64,
}

impl PeriodicEnvironment {
    pub fn try_new(
        temperature: PeriodicSignal,
        productivity: PeriodicSignal,
        disturbance: PeriodicSignal,
        period: f64,
    ) -> Result<Self, ModelError> {
        let environment = Self {
            temperature,
            productivity,
            disturbance,
            period,
        };
        environment.validate()?;
        Ok(environment)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        require_positive("period", self.period)?;
        for signal in [self.temperature, self.productivity, self.disturbance] {
            require_finite("periodic_mean", signal.mean)?;
            require_finite("periodic_amplitude", signal.amplitude)?;
            require_finite("phase_radians", signal.phase_radians)?;
        }
        require_positive("minimum_temperature", self.temperature.minimum())?;
        require_positive("minimum_productivity", self.productivity.minimum())?;
        require_non_negative("minimum_disturbance", self.disturbance.minimum())?;
        Ok(())
    }

    pub fn at(&self, time: f64) -> Result<EnvironmentalDrivers, ModelError> {
        self.validate()?;
        require_non_negative("periodic_time", time)?;
        let angle = core::f64::consts::TAU * time / self.period;
        EnvironmentalDrivers::try_new(
            self.temperature.at_angle(angle),
            self.productivity.at_angle(angle),
            self.disturbance.at_angle(angle),
        )
    }
}

impl EnvironmentalDriverSource for PeriodicEnvironment {
    fn drivers_at(&self, time: f64) -> Result<EnvironmentalDrivers, ModelError> {
        self.at(time)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GaussianThermalResponse, LogisticEnvironmentCoupling, LogisticModel,
        simulate_logistic_driver_source,
    };

    fn environment() -> PeriodicEnvironment {
        PeriodicEnvironment::try_new(
            PeriodicSignal::try_new(293.0, 5.0, 0.0).unwrap(),
            PeriodicSignal::try_new(1.0, 0.2, core::f64::consts::FRAC_PI_2).unwrap(),
            PeriodicSignal::try_new(0.1, 0.05, core::f64::consts::PI).unwrap(),
            12.0,
        )
        .unwrap()
    }

    #[test]
    fn periodic_drivers_repeat_exactly() {
        let environment = environment();
        let start = environment.at(0.0).unwrap();
        let cycle = environment.at(12.0).unwrap();
        assert!((start.temperature - cycle.temperature).abs() < 1e-12);
        assert!((start.productivity - cycle.productivity).abs() < 1e-12);
        assert!((start.disturbance - cycle.disturbance).abs() < 1e-12);
        assert!((environment.at(3.0).unwrap().temperature - 298.0).abs() < 1e-12);
    }

    #[test]
    fn impossible_negative_driver_cycles_are_rejected() {
        let invalid = PeriodicEnvironment::try_new(
            PeriodicSignal::try_new(293.0, 5.0, 0.0).unwrap(),
            PeriodicSignal::try_new(0.1, 0.2, 0.0).unwrap(),
            PeriodicSignal::try_new(0.0, 0.0, 0.0).unwrap(),
            12.0,
        );
        assert!(invalid.is_err());
    }

    #[test]
    fn generic_driver_replay_accepts_periodic_environment() {
        let coupling = LogisticEnvironmentCoupling::try_new(
            LogisticModel::try_new(0.5, 100.0).unwrap(),
            GaussianThermalResponse::try_new(293.0, 10.0, 0.1).unwrap(),
            1.0,
            0.5,
        )
        .unwrap();
        let samples =
            simulate_logistic_driver_source(&coupling, 10.0, &environment(), 0.05, 480).unwrap();
        assert_eq!(samples.len(), 481);
        assert!(samples.iter().all(|sample| sample.population > 0.0));
        assert_eq!(samples.first().unwrap().time, 0.0);
        assert!((samples.last().unwrap().time - 24.0).abs() < 1e-12);
    }
}
