// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Versioned persistence for qualified Genesis resource-source profiles.
//!
//! Arbitrary `FnMut` resource closures are intentionally not snapshot-able: their hidden state is
//! unknowable to `Population`. Exact execution resume therefore requires an explicit environment
//! profile whose causal state can be validated and restored. This module defines two such profiles:
//!
//! - [`EnvironmentSnapshotV1`] for the pure synthetic environment parameters. Its execution clock
//!   remains external and must be bound by the higher execution capsule.
//! - [`EarthForcedEnvironmentSnapshotV1`] for the stateful integrated climate environment,
//!   including its private seasonal-phase tick and every `IceAlbedoModel` parameter.

use serde::{Deserialize, Serialize};
use symthaea_earth_system::IceAlbedoModel;

use crate::{EarthForcedEnvironment, Environment};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvironmentSnapshotV1 {
    mean_bits: u64,
    amplitude_bits: u64,
    period_bits: u64,
    noise_seed: u64,
    noise_amplitude_bits: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ValidatedEnvironmentSnapshotV1 {
    environment: Environment,
    snapshot: EnvironmentSnapshotV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvironmentSnapshotErrorV1 {
    NonFinite { field: &'static str },
    NonPositive { field: &'static str },
}

impl EnvironmentSnapshotV1 {
    pub fn from_environment(environment: Environment) -> Self {
        Self {
            mean_bits: environment.mean.to_bits(),
            amplitude_bits: environment.amplitude.to_bits(),
            period_bits: environment.period.to_bits(),
            noise_seed: environment.noise_seed,
            noise_amplitude_bits: environment.noise_amplitude.to_bits(),
        }
    }

    pub fn validate(
        self,
    ) -> Result<ValidatedEnvironmentSnapshotV1, EnvironmentSnapshotErrorV1> {
        let environment = Environment {
            mean: finite("mean", self.mean_bits)?,
            amplitude: finite("amplitude", self.amplitude_bits)?,
            period: finite("period", self.period_bits)?,
            noise_seed: self.noise_seed,
            noise_amplitude: finite("noise_amplitude", self.noise_amplitude_bits)?,
        };
        if environment.period <= 0.0 {
            return Err(EnvironmentSnapshotErrorV1::NonPositive { field: "period" });
        }
        Ok(ValidatedEnvironmentSnapshotV1 {
            environment,
            snapshot: self,
        })
    }
}

impl ValidatedEnvironmentSnapshotV1 {
    pub fn environment(&self) -> Environment {
        self.environment
    }

    pub fn as_snapshot(&self) -> &EnvironmentSnapshotV1 {
        &self.snapshot
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct IceAlbedoModelSnapshotV1 {
    solar_constant_bits: u64,
    emissivity_bits: u64,
    albedo_ice_bits: u64,
    albedo_warm_bits: u64,
    t_ice_bits: u64,
    t_warm_bits: u64,
}

impl IceAlbedoModelSnapshotV1 {
    pub(crate) fn from_model(model: IceAlbedoModel) -> Self {
        Self {
            solar_constant_bits: model.solar_constant.to_bits(),
            emissivity_bits: model.emissivity.to_bits(),
            albedo_ice_bits: model.albedo_ice.to_bits(),
            albedo_warm_bits: model.albedo_warm.to_bits(),
            t_ice_bits: model.t_ice.to_bits(),
            t_warm_bits: model.t_warm.to_bits(),
        }
    }

    fn validate(self) -> Result<IceAlbedoModel, EarthForcedEnvironmentSnapshotErrorV1> {
        let model = IceAlbedoModel {
            solar_constant: earth_finite("model.solar_constant", self.solar_constant_bits)?,
            emissivity: earth_finite("model.emissivity", self.emissivity_bits)?,
            albedo_ice: earth_finite("model.albedo_ice", self.albedo_ice_bits)?,
            albedo_warm: earth_finite("model.albedo_warm", self.albedo_warm_bits)?,
            t_ice: earth_finite("model.t_ice", self.t_ice_bits)?,
            t_warm: earth_finite("model.t_warm", self.t_warm_bits)?,
        };
        if model.validate().is_err() {
            return Err(EarthForcedEnvironmentSnapshotErrorV1::InvalidIceAlbedoModel);
        }
        Ok(model)
    }
}

/// Complete stateful climate-resource continuation state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EarthForcedEnvironmentSnapshotV1 {
    model: IceAlbedoModelSnapshotV1,
    temperature_bits: u64,
    seasonal_amplitude_bits: u64,
    seasonal_period_ticks_bits: u64,
    dt_seconds_bits: u64,
    secular_drift_per_tick_bits: u64,
    tick: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ValidatedEarthForcedEnvironmentSnapshotV1 {
    model: IceAlbedoModel,
    temperature: f64,
    seasonal_amplitude: f64,
    seasonal_period_ticks: f64,
    dt_seconds: f64,
    secular_drift_per_tick: f64,
    tick: u64,
    snapshot: EarthForcedEnvironmentSnapshotV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EarthForcedEnvironmentSnapshotErrorV1 {
    NonFinite { field: &'static str },
    NonPositive { field: &'static str },
    InvalidIceAlbedoModel,
    TickExhausted,
}

impl EarthForcedEnvironmentSnapshotV1 {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_live_parts(
        model: IceAlbedoModel,
        temperature: f64,
        seasonal_amplitude: f64,
        seasonal_period_ticks: f64,
        dt_seconds: f64,
        secular_drift_per_tick: f64,
        tick: u64,
    ) -> Self {
        Self {
            model: IceAlbedoModelSnapshotV1::from_model(model),
            temperature_bits: temperature.to_bits(),
            seasonal_amplitude_bits: seasonal_amplitude.to_bits(),
            seasonal_period_ticks_bits: seasonal_period_ticks.to_bits(),
            dt_seconds_bits: dt_seconds.to_bits(),
            secular_drift_per_tick_bits: secular_drift_per_tick.to_bits(),
            tick,
        }
    }

    pub fn validate(
        self,
    ) -> Result<ValidatedEarthForcedEnvironmentSnapshotV1, EarthForcedEnvironmentSnapshotErrorV1>
    {
        let model = self.model.validate()?;
        let temperature = earth_finite("temperature", self.temperature_bits)?;
        let seasonal_amplitude = earth_finite("seasonal_amplitude", self.seasonal_amplitude_bits)?;
        let seasonal_period_ticks = earth_finite(
            "seasonal_period_ticks",
            self.seasonal_period_ticks_bits,
        )?;
        let dt_seconds = earth_finite("dt_seconds", self.dt_seconds_bits)?;
        let secular_drift_per_tick = earth_finite(
            "secular_drift_per_tick",
            self.secular_drift_per_tick_bits,
        )?;

        if temperature <= 0.0 {
            return Err(EarthForcedEnvironmentSnapshotErrorV1::NonPositive {
                field: "temperature",
            });
        }
        if seasonal_period_ticks <= 0.0 {
            return Err(EarthForcedEnvironmentSnapshotErrorV1::NonPositive {
                field: "seasonal_period_ticks",
            });
        }
        if dt_seconds <= 0.0 {
            return Err(EarthForcedEnvironmentSnapshotErrorV1::NonPositive {
                field: "dt_seconds",
            });
        }
        if self.tick == u64::MAX {
            return Err(EarthForcedEnvironmentSnapshotErrorV1::TickExhausted);
        }

        Ok(ValidatedEarthForcedEnvironmentSnapshotV1 {
            model,
            temperature,
            seasonal_amplitude,
            seasonal_period_ticks,
            dt_seconds,
            secular_drift_per_tick,
            tick: self.tick,
            snapshot: self,
        })
    }
}

impl ValidatedEarthForcedEnvironmentSnapshotV1 {
    pub fn model(&self) -> IceAlbedoModel {
        self.model
    }

    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    pub fn seasonal_amplitude(&self) -> f64 {
        self.seasonal_amplitude
    }

    pub fn seasonal_period_ticks(&self) -> f64 {
        self.seasonal_period_ticks
    }

    pub fn dt_seconds(&self) -> f64 {
        self.dt_seconds
    }

    pub fn secular_drift_per_tick(&self) -> f64 {
        self.secular_drift_per_tick
    }

    pub fn tick(&self) -> u64 {
        self.tick
    }

    pub fn as_snapshot(&self) -> &EarthForcedEnvironmentSnapshotV1 {
        &self.snapshot
    }
}

fn finite(field: &'static str, bits: u64) -> Result<f64, EnvironmentSnapshotErrorV1> {
    let value = f64::from_bits(bits);
    if !value.is_finite() {
        return Err(EnvironmentSnapshotErrorV1::NonFinite { field });
    }
    Ok(value)
}

fn earth_finite(
    field: &'static str,
    bits: u64,
) -> Result<f64, EarthForcedEnvironmentSnapshotErrorV1> {
    let value = f64::from_bits(bits);
    if !value.is_finite() {
        return Err(EarthForcedEnvironmentSnapshotErrorV1::NonFinite { field });
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_environment_parameters_round_trip_bit_exactly() {
        let original = Environment {
            mean: -0.0,
            amplitude: 0.3125,
            period: 137.25,
            noise_seed: 0,
            noise_amplitude: 0.0175,
        };
        let encoded = serde_json::to_string(&EnvironmentSnapshotV1::from_environment(original))
            .expect("serialize environment");
        let raw: EnvironmentSnapshotV1 = serde_json::from_str(&encoded).expect("deserialize");
        let restored = raw.validate().expect("validate").environment();
        assert_eq!(original.mean.to_bits(), restored.mean.to_bits());
        assert_eq!(original.amplitude.to_bits(), restored.amplitude.to_bits());
        assert_eq!(original.period.to_bits(), restored.period.to_bits());
        assert_eq!(original.noise_seed, restored.noise_seed);
        assert_eq!(original.noise_amplitude.to_bits(), restored.noise_amplitude.to_bits());
        for tick in [0, 1, 7, 99, 10_000] {
            assert_eq!(original.resource_at(tick).to_bits(), restored.resource_at(tick).to_bits());
        }
    }

    #[test]
    fn pure_environment_rejects_nonpositive_period() {
        let mut raw = EnvironmentSnapshotV1::from_environment(Environment::default());
        raw.period_bits = 0.0_f64.to_bits();
        assert_eq!(
            raw.validate().unwrap_err(),
            EnvironmentSnapshotErrorV1::NonPositive { field: "period" }
        );
    }

    #[test]
    fn earth_snapshot_rejects_exhausted_continuation_tick() {
        let model = IceAlbedoModel::earth();
        let raw = EarthForcedEnvironmentSnapshotV1::from_live_parts(
            model,
            288.0,
            10.0,
            200.0,
            3600.0,
            0.0,
            u64::MAX,
        );
        assert_eq!(
            raw.validate().unwrap_err(),
            EarthForcedEnvironmentSnapshotErrorV1::TickExhausted
        );
    }

    #[test]
    fn invalid_ice_albedo_model_cannot_become_environment_authority() {
        let mut model = IceAlbedoModel::earth();
        model.emissivity = 2.0;
        let raw = EarthForcedEnvironmentSnapshotV1::from_live_parts(
            model, 288.0, 10.0, 200.0, 3600.0, 0.0, 7,
        );
        assert_eq!(
            raw.validate().unwrap_err(),
            EarthForcedEnvironmentSnapshotErrorV1::InvalidIceAlbedoModel
        );
    }

    #[test]
    fn serialized_validated_earth_restore_preserves_exact_future_climate_trajectory() {
        let mut uninterrupted = EarthForcedEnvironment::earth_like(173.0).with_secular_drift(-0.017);
        for _ in 0..137 {
            uninterrupted.step();
        }

        let encoded = serde_json::to_string(&uninterrupted.snapshot_v1())
            .expect("serialize Earth environment snapshot");
        let raw: EarthForcedEnvironmentSnapshotV1 =
            serde_json::from_str(&encoded).expect("deserialize Earth environment snapshot");
        let validated = raw.validate().expect("validate Earth environment snapshot");
        let mut restored = EarthForcedEnvironment::from_validated_snapshot_v1(&validated);

        assert_eq!(uninterrupted.snapshot_v1(), restored.snapshot_v1());
        for step in 0..512 {
            assert_eq!(
                uninterrupted.step().to_bits(),
                restored.step().to_bits(),
                "resource trajectory diverged after restored step {step}"
            );
            assert_eq!(
                uninterrupted.snapshot_v1(),
                restored.snapshot_v1(),
                "climate state diverged after restored step {step}"
            );
        }
    }

    #[test]
    fn tick_exhaustion_panics_before_any_climate_state_mutation() {
        let raw = EarthForcedEnvironmentSnapshotV1::from_live_parts(
            IceAlbedoModel::earth(),
            288.0,
            10.0,
            200.0,
            3600.0,
            -0.1,
            u64::MAX - 1,
        );
        let validated = raw.validate().expect("penultimate tick is a valid continuation state");
        let mut environment = EarthForcedEnvironment::from_validated_snapshot_v1(&validated);

        environment.step();
        let before = environment.snapshot_v1();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| environment.step()));
        assert!(panic.is_err(), "exhausted environment must fail closed");
        assert_eq!(before, environment.snapshot_v1());
    }
}
