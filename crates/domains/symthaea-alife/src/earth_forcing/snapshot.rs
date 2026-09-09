// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bit-exact persistence contract for [`super::EarthForcedEnvironment`].
//!
//! The environment is causal state, not merely an input function. Exact continuation requires the
//! complete mutable ice-albedo model, integrator temperature, forcing parameters, timestep, secular
//! drift, and the private seasonal tick. Raw persisted bytes remain non-authoritative until this
//! module validates both the Earth-system model and the ALife integration parameters.

use serde::{Deserialize, Serialize};
use symthaea_earth_system::{IceAlbedoModel, ModelError};

use super::EarthForcedEnvironment;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EarthForcedEnvironmentSnapshotV1 {
    model_solar_constant_bits: u64,
    model_emissivity_bits: u64,
    model_albedo_ice_bits: u64,
    model_albedo_warm_bits: u64,
    model_t_ice_bits: u64,
    model_t_warm_bits: u64,
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
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EarthForcedEnvironmentSnapshotErrorV1 {
    Model(ModelError),
    NonFinite { field: &'static str },
    NonPositive { field: &'static str },
}

impl EarthForcedEnvironment {
    /// Capture every environment-owned causal field at a stable between-step boundary.
    pub fn snapshot_v1(&self) -> EarthForcedEnvironmentSnapshotV1 {
        EarthForcedEnvironmentSnapshotV1 {
            model_solar_constant_bits: self.model.solar_constant.to_bits(),
            model_emissivity_bits: self.model.emissivity.to_bits(),
            model_albedo_ice_bits: self.model.albedo_ice.to_bits(),
            model_albedo_warm_bits: self.model.albedo_warm.to_bits(),
            model_t_ice_bits: self.model.t_ice.to_bits(),
            model_t_warm_bits: self.model.t_warm.to_bits(),
            temperature_bits: self.temperature.to_bits(),
            seasonal_amplitude_bits: self.seasonal_amplitude.to_bits(),
            seasonal_period_ticks_bits: self.seasonal_period_ticks.to_bits(),
            dt_seconds_bits: self.dt_seconds.to_bits(),
            secular_drift_per_tick_bits: self.secular_drift_per_tick.to_bits(),
            tick: self.tick,
        }
    }

    /// Recreate live forcing state only from a validated, non-serializable snapshot capability.
    pub fn from_validated_snapshot_v1(
        snapshot: &ValidatedEarthForcedEnvironmentSnapshotV1,
    ) -> Self {
        Self {
            model: snapshot.model,
            temperature: snapshot.temperature,
            seasonal_amplitude: snapshot.seasonal_amplitude,
            seasonal_period_ticks: snapshot.seasonal_period_ticks,
            dt_seconds: snapshot.dt_seconds,
            secular_drift_per_tick: snapshot.secular_drift_per_tick,
            tick: snapshot.tick,
        }
    }
}

impl EarthForcedEnvironmentSnapshotV1 {
    /// Validate persisted forcing state before it can become live execution authority.
    pub fn validate(
        self,
    ) -> Result<ValidatedEarthForcedEnvironmentSnapshotV1, EarthForcedEnvironmentSnapshotErrorV1>
    {
        let model = IceAlbedoModel {
            solar_constant: f64::from_bits(self.model_solar_constant_bits),
            emissivity: f64::from_bits(self.model_emissivity_bits),
            albedo_ice: f64::from_bits(self.model_albedo_ice_bits),
            albedo_warm: f64::from_bits(self.model_albedo_warm_bits),
            t_ice: f64::from_bits(self.model_t_ice_bits),
            t_warm: f64::from_bits(self.model_t_warm_bits),
        };
        model
            .validate()
            .map_err(EarthForcedEnvironmentSnapshotErrorV1::Model)?;

        let temperature = finite("temperature", self.temperature_bits)?;
        let seasonal_amplitude = finite("seasonal_amplitude", self.seasonal_amplitude_bits)?;
        let seasonal_period_ticks = positive("seasonal_period_ticks", self.seasonal_period_ticks_bits)?;
        let dt_seconds = positive("dt_seconds", self.dt_seconds_bits)?;
        let secular_drift_per_tick = finite(
            "secular_drift_per_tick",
            self.secular_drift_per_tick_bits,
        )?;

        Ok(ValidatedEarthForcedEnvironmentSnapshotV1 {
            model,
            temperature,
            seasonal_amplitude,
            seasonal_period_ticks,
            dt_seconds,
            secular_drift_per_tick,
            tick: self.tick,
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
}

fn finite(
    field: &'static str,
    bits: u64,
) -> Result<f64, EarthForcedEnvironmentSnapshotErrorV1> {
    let value = f64::from_bits(bits);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(EarthForcedEnvironmentSnapshotErrorV1::NonFinite { field })
    }
}

fn positive(
    field: &'static str,
    bits: u64,
) -> Result<f64, EarthForcedEnvironmentSnapshotErrorV1> {
    let value = finite(field, bits)?;
    if value > 0.0 {
        Ok(value)
    } else {
        Err(EarthForcedEnvironmentSnapshotErrorV1::NonPositive { field })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(env: &EarthForcedEnvironment) -> EarthForcedEnvironment {
        let json = serde_json::to_string(&env.snapshot_v1()).expect("serialize environment");
        let raw: EarthForcedEnvironmentSnapshotV1 =
            serde_json::from_str(&json).expect("deserialize environment");
        let validated = raw.validate().expect("validate environment snapshot");
        EarthForcedEnvironment::from_validated_snapshot_v1(&validated)
    }

    #[test]
    fn serialized_restore_preserves_every_causal_field_bit_exactly() {
        let mut env = EarthForcedEnvironment::earth_like(197.5).with_secular_drift(-0.03125);
        env.model.emissivity = 0.615;
        env.model.albedo_ice = 0.58;
        env.model.albedo_warm = 0.29;
        env.model.t_ice = 262.5;
        env.model.t_warm = 284.25;
        env.seasonal_amplitude *= 0.875;
        env.dt_seconds *= 1.03125;
        for _ in 0..137 {
            let _ = env.step();
        }

        let encoded = serde_json::to_string(&env.snapshot_v1()).expect("serialize");
        let restored = round_trip(&env);
        let restored_encoded =
            serde_json::to_string(&restored.snapshot_v1()).expect("serialize restored");
        assert_eq!(encoded, restored_encoded);
    }

    #[test]
    fn restored_environment_has_identical_long_future() {
        let mut uninterrupted =
            EarthForcedEnvironment::earth_like(211.0).with_secular_drift(-0.0175);
        for _ in 0..173 {
            let _ = uninterrupted.step();
        }
        let mut restored = round_trip(&uninterrupted);

        for step in 173..1200 {
            let a = uninterrupted.step();
            let b = restored.step();
            assert_eq!(a.to_bits(), b.to_bits(), "resource drift at step {step}");
            assert_eq!(
                uninterrupted.temperature.to_bits(),
                restored.temperature.to_bits(),
                "temperature drift at step {step}"
            );
            assert_eq!(
                serde_json::to_string(&uninterrupted.snapshot_v1()).unwrap(),
                serde_json::to_string(&restored.snapshot_v1()).unwrap(),
                "environment causal state drift at step {step}"
            );
        }
    }

    #[test]
    fn invalid_earth_model_cannot_become_restore_authority() {
        let mut raw = EarthForcedEnvironment::earth_like(200.0).snapshot_v1();
        raw.model_solar_constant_bits = 0.0_f64.to_bits();
        assert!(matches!(
            raw.validate(),
            Err(EarthForcedEnvironmentSnapshotErrorV1::Model(_))
        ));
    }

    #[test]
    fn nonfinite_integrator_state_fails_closed() {
        let mut raw = EarthForcedEnvironment::earth_like(200.0).snapshot_v1();
        raw.temperature_bits = f64::NAN.to_bits();
        assert_eq!(
            raw.validate(),
            Err(EarthForcedEnvironmentSnapshotErrorV1::NonFinite {
                field: "temperature"
            })
        );
    }

    #[test]
    fn nonpositive_period_or_timestep_fails_closed() {
        let mut period = EarthForcedEnvironment::earth_like(200.0).snapshot_v1();
        period.seasonal_period_ticks_bits = 0.0_f64.to_bits();
        assert_eq!(
            period.validate(),
            Err(EarthForcedEnvironmentSnapshotErrorV1::NonPositive {
                field: "seasonal_period_ticks"
            })
        );

        let mut dt = EarthForcedEnvironment::earth_like(200.0).snapshot_v1();
        dt.dt_seconds_bits = (-1.0_f64).to_bits();
        assert_eq!(
            dt.validate(),
            Err(EarthForcedEnvironmentSnapshotErrorV1::NonPositive {
                field: "dt_seconds"
            })
        );
    }
}
