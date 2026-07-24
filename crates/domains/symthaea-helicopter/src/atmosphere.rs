// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic lower-atmosphere model for density-altitude-aware flight physics.
//!
//! The model implements the International Standard Atmosphere tropospheric
//! lapse-rate equations over a deliberately bounded altitude range. It is a
//! reduced-order simulation input, not a weather forecast or certification
//! atmosphere. Temperature offsets let qualification scenarios exercise hot
//! and cold days without silently changing the calibrated sea-level constants.

use serde::{Deserialize, Serialize};

/// Configuration for a bounded standard-atmosphere model.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StandardAtmosphereConfig {
    pub sea_level_temperature_k: f64,
    pub sea_level_pressure_pa: f64,
    pub sea_level_density_kg_m3: f64,
    /// Temperature change per meter. The ISA troposphere value is -0.0065 K/m.
    pub lapse_rate_k_per_m: f64,
    pub gravity_mps2: f64,
    pub specific_gas_constant_j_kg_k: f64,
    pub heat_capacity_ratio: f64,
    pub minimum_altitude_m: f64,
    pub maximum_altitude_m: f64,
}

impl Default for StandardAtmosphereConfig {
    fn default() -> Self {
        Self {
            sea_level_temperature_k: 288.15,
            sea_level_pressure_pa: 101_325.0,
            sea_level_density_kg_m3: 1.225,
            lapse_rate_k_per_m: -0.0065,
            gravity_mps2: 9.806_65,
            specific_gas_constant_j_kg_k: 287.052_87,
            heat_capacity_ratio: 1.4,
            minimum_altitude_m: -500.0,
            maximum_altitude_m: 11_000.0,
        }
    }
}

impl StandardAtmosphereConfig {
    pub fn validate(&self) -> Result<(), AtmosphereError> {
        let positive = [
            self.sea_level_temperature_k,
            self.sea_level_pressure_pa,
            self.sea_level_density_kg_m3,
            self.gravity_mps2,
            self.specific_gas_constant_j_kg_k,
            self.heat_capacity_ratio,
        ];
        if !positive
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
            || !self.lapse_rate_k_per_m.is_finite()
            || self.lapse_rate_k_per_m >= 0.0
            || !self.minimum_altitude_m.is_finite()
            || !self.maximum_altitude_m.is_finite()
            || self.maximum_altitude_m <= self.minimum_altitude_m
            || self.sea_level_temperature_k + self.lapse_rate_k_per_m * self.maximum_altitude_m
                <= 0.0
        {
            return Err(AtmosphereError::InvalidConfiguration);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AtmosphereError {
    InvalidConfiguration,
    NonFiniteInput,
    AltitudeOutsideModel,
    NonPhysicalTemperature,
}

/// Atmospheric state at one geometric altitude.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AtmosphereSample {
    pub altitude_m: f64,
    pub temperature_k: f64,
    pub pressure_pa: f64,
    pub density_kg_m3: f64,
    pub density_ratio: f64,
    pub speed_of_sound_mps: f64,
    /// ISA altitude that has the same density as this sample.
    pub density_altitude_m: f64,
    /// True when `sample_bounded` clamped altitude to the supported model range.
    pub model_clamped: bool,
}

impl AtmosphereSample {
    pub fn sea_level() -> Self {
        Self {
            altitude_m: 0.0,
            temperature_k: 288.15,
            pressure_pa: 101_325.0,
            density_kg_m3: 1.225,
            density_ratio: 1.0,
            speed_of_sound_mps: 340.294,
            density_altitude_m: 0.0,
            model_clamped: false,
        }
    }
}

/// Deterministic ISA-troposphere sampler with an optional day-temperature offset.
#[derive(Debug, Clone, Copy)]
pub struct StandardAtmosphere {
    config: StandardAtmosphereConfig,
    temperature_offset_k: f64,
}

impl StandardAtmosphere {
    pub fn new(config: StandardAtmosphereConfig) -> Result<Self, AtmosphereError> {
        config.validate()?;
        Ok(Self {
            config,
            temperature_offset_k: 0.0,
        })
    }

    pub fn with_temperature_offset(
        config: StandardAtmosphereConfig,
        temperature_offset_k: f64,
    ) -> Result<Self, AtmosphereError> {
        config.validate()?;
        if !temperature_offset_k.is_finite() {
            return Err(AtmosphereError::NonFiniteInput);
        }
        let atmosphere = Self {
            config,
            temperature_offset_k,
        };
        // Check both range endpoints because the lapse is monotonic.
        for altitude in [config.minimum_altitude_m, config.maximum_altitude_m] {
            if atmosphere.temperature_at(altitude) <= 0.0 {
                return Err(AtmosphereError::NonPhysicalTemperature);
            }
        }
        Ok(atmosphere)
    }

    pub fn config(&self) -> StandardAtmosphereConfig {
        self.config
    }

    pub fn temperature_offset_k(&self) -> f64 {
        self.temperature_offset_k
    }

    fn temperature_at(&self, altitude_m: f64) -> f64 {
        self.config.sea_level_temperature_k
            + self.temperature_offset_k
            + self.config.lapse_rate_k_per_m * altitude_m
    }

    pub fn sample(&self, altitude_m: f64) -> Result<AtmosphereSample, AtmosphereError> {
        self.config.validate()?;
        if !altitude_m.is_finite() {
            return Err(AtmosphereError::NonFiniteInput);
        }
        if !(self.config.minimum_altitude_m..=self.config.maximum_altitude_m).contains(&altitude_m)
        {
            return Err(AtmosphereError::AltitudeOutsideModel);
        }

        let temperature_k = self.temperature_at(altitude_m);
        let sea_level_day_temperature_k =
            self.config.sea_level_temperature_k + self.temperature_offset_k;
        if temperature_k <= 0.0 || sea_level_day_temperature_k <= 0.0 {
            return Err(AtmosphereError::NonPhysicalTemperature);
        }

        let pressure_exponent = -self.config.gravity_mps2
            / (self.config.specific_gas_constant_j_kg_k * self.config.lapse_rate_k_per_m);
        let temperature_ratio = temperature_k / sea_level_day_temperature_k;
        let pressure_pa =
            self.config.sea_level_pressure_pa * temperature_ratio.powf(pressure_exponent);
        let density_kg_m3 =
            pressure_pa / (self.config.specific_gas_constant_j_kg_k * temperature_k);
        let density_ratio = density_kg_m3 / self.config.sea_level_density_kg_m3;
        let speed_of_sound_mps = (self.config.heat_capacity_ratio
            * self.config.specific_gas_constant_j_kg_k
            * temperature_k)
            .sqrt();

        // Invert the zero-offset ISA density relation. The density exponent is
        // pressure_exponent - 1 because rho = p/(R*T).
        let density_exponent = pressure_exponent - 1.0;
        let isa_temperature_ratio = density_ratio.max(1.0e-12).powf(1.0 / density_exponent);
        let density_altitude_m = self.config.sea_level_temperature_k
            * (isa_temperature_ratio - 1.0)
            / self.config.lapse_rate_k_per_m;

        Ok(AtmosphereSample {
            altitude_m,
            temperature_k,
            pressure_pa,
            density_kg_m3,
            density_ratio,
            speed_of_sound_mps,
            density_altitude_m,
            model_clamped: false,
        })
    }

    /// Sample while conservatively clamping geometric altitude to the declared
    /// model range. This avoids a high-altitude failure silently reverting to
    /// sea-level density. Non-finite altitude still fails closed.
    pub fn sample_bounded(&self, altitude_m: f64) -> Result<AtmosphereSample, AtmosphereError> {
        if !altitude_m.is_finite() {
            return Err(AtmosphereError::NonFiniteInput);
        }
        let bounded = altitude_m.clamp(
            self.config.minimum_altitude_m,
            self.config.maximum_altitude_m,
        );
        let mut sample = self.sample(bounded)?;
        sample.model_clamped = bounded != altitude_m;
        Ok(sample)
    }
}

impl Default for StandardAtmosphere {
    fn default() -> Self {
        Self::new(StandardAtmosphereConfig::default())
            .expect("default atmosphere configuration must remain valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sea_level_is_close_to_declared_reference() {
        let sample = StandardAtmosphere::default().sample(0.0).unwrap();
        assert!((sample.temperature_k - 288.15).abs() < 1.0e-9);
        assert!((sample.pressure_pa - 101_325.0).abs() < 1.0e-6);
        assert!((sample.density_kg_m3 - 1.225).abs() < 0.002);
        assert!((sample.density_altitude_m).abs() < 5.0);
    }

    #[test]
    fn density_and_pressure_fall_with_altitude() {
        let atmosphere = StandardAtmosphere::default();
        let sea = atmosphere.sample(0.0).unwrap();
        let mountain = atmosphere.sample(3_000.0).unwrap();
        assert!(mountain.pressure_pa < sea.pressure_pa);
        assert!(mountain.density_kg_m3 < sea.density_kg_m3);
        assert!(mountain.speed_of_sound_mps < sea.speed_of_sound_mps);
    }

    #[test]
    fn hot_day_increases_density_altitude() {
        let standard = StandardAtmosphere::default().sample(2_000.0).unwrap();
        let hot =
            StandardAtmosphere::with_temperature_offset(StandardAtmosphereConfig::default(), 20.0)
                .unwrap()
                .sample(2_000.0)
                .unwrap();
        assert!(hot.density_kg_m3 < standard.density_kg_m3);
        assert!(hot.density_altitude_m > standard.density_altitude_m);
    }

    #[test]
    fn unsupported_altitude_fails_closed() {
        assert_eq!(
            StandardAtmosphere::default().sample(12_000.0),
            Err(AtmosphereError::AltitudeOutsideModel)
        );
    }
    #[test]
    fn bounded_sampling_never_reverts_to_sea_level_at_high_altitude() {
        let atmosphere = StandardAtmosphere::default();
        let bounded = atmosphere.sample_bounded(20_000.0).unwrap();
        assert!(bounded.model_clamped);
        assert_eq!(bounded.altitude_m, atmosphere.config().maximum_altitude_m);
        assert!(bounded.density_kg_m3 < 0.5);
    }
}
