// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Stateful icing, precipitation, dust, and salt-aerosol degradation model.
//!
//! Wind alone is not a complete environment. This bounded model accumulates
//! exposure and emits conservative rotor, engine, and sensor effectiveness for
//! qualification campaigns. Coefficients are research assumptions until bound
//! to named-airframe calibration evidence.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EnvironmentalHazardConfig {
    pub minimum_icing_temperature_c: f64,
    pub maximum_icing_temperature_c: f64,
    pub icing_rate_mm_s_per_g_m3: f64,
    pub anti_ice_effectiveness: f64,
    pub melt_rate_mm_s_per_c: f64,
    pub dust_accumulation_per_mg_m3_s: f64,
    pub dust_washout_per_mm_h_s: f64,
    pub salt_accumulation_per_mg_m3_s: f64,
    pub rotor_loss_per_mm_ice: f64,
    pub engine_loss_per_dust_load: f64,
    pub sensor_time_constant_s: f64,
    pub caution_rotor_efficiency: f64,
    pub exit_rotor_efficiency: f64,
    pub land_engine_power_fraction: f64,
}

impl Default for EnvironmentalHazardConfig {
    fn default() -> Self {
        Self {
            minimum_icing_temperature_c: -20.0,
            maximum_icing_temperature_c: 2.0,
            icing_rate_mm_s_per_g_m3: 0.004,
            anti_ice_effectiveness: 0.85,
            melt_rate_mm_s_per_c: 0.002,
            dust_accumulation_per_mg_m3_s: 0.0005,
            dust_washout_per_mm_h_s: 0.00002,
            salt_accumulation_per_mg_m3_s: 0.0002,
            rotor_loss_per_mm_ice: 0.035,
            engine_loss_per_dust_load: 0.12,
            sensor_time_constant_s: 1.0,
            caution_rotor_efficiency: 0.9,
            exit_rotor_efficiency: 0.75,
            land_engine_power_fraction: 0.65,
        }
    }
}

impl EnvironmentalHazardConfig {
    pub fn validate(&self) -> Result<(), EnvironmentalHazardError> {
        if !self.minimum_icing_temperature_c.is_finite()
            || !self.maximum_icing_temperature_c.is_finite()
            || self.maximum_icing_temperature_c <= self.minimum_icing_temperature_c
            || !self.icing_rate_mm_s_per_g_m3.is_finite()
            || self.icing_rate_mm_s_per_g_m3 < 0.0
            || !self.anti_ice_effectiveness.is_finite()
            || !(0.0..=1.0).contains(&self.anti_ice_effectiveness)
            || !self.melt_rate_mm_s_per_c.is_finite()
            || self.melt_rate_mm_s_per_c < 0.0
            || !self.dust_accumulation_per_mg_m3_s.is_finite()
            || self.dust_accumulation_per_mg_m3_s < 0.0
            || !self.dust_washout_per_mm_h_s.is_finite()
            || self.dust_washout_per_mm_h_s < 0.0
            || !self.salt_accumulation_per_mg_m3_s.is_finite()
            || self.salt_accumulation_per_mg_m3_s < 0.0
            || !self.rotor_loss_per_mm_ice.is_finite()
            || self.rotor_loss_per_mm_ice < 0.0
            || !self.engine_loss_per_dust_load.is_finite()
            || self.engine_loss_per_dust_load < 0.0
            || !self.sensor_time_constant_s.is_finite()
            || self.sensor_time_constant_s <= 0.0
            || !valid_fraction(self.caution_rotor_efficiency)
            || !valid_fraction(self.exit_rotor_efficiency)
            || !valid_fraction(self.land_engine_power_fraction)
            || self.exit_rotor_efficiency >= self.caution_rotor_efficiency
        {
            return Err(EnvironmentalHazardError::InvalidConfiguration);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EnvironmentalHazardInput {
    pub temperature_c: f64,
    pub liquid_water_content_g_m3: f64,
    pub precipitation_mm_h: f64,
    pub dust_mg_m3: f64,
    pub salt_aerosol_mg_m3: f64,
    pub visibility_m: f64,
    pub true_airspeed_mps: f64,
    pub anti_ice_command: f64,
    pub dt_s: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EnvironmentalHazardLevel {
    Nominal,
    Caution,
    ExitEnvironment,
    LandAsSoonAsPracticable,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EnvironmentalHazardState {
    pub accreted_ice_mm: f64,
    pub dust_load: f64,
    pub salt_load: f64,
    pub sensor_occlusion: f64,
    pub main_rotor_efficiency: f64,
    pub tail_rotor_efficiency: f64,
    pub engine_power_fraction: f64,
    pub sensor_quality: f64,
    pub level: EnvironmentalHazardLevel,
    pub cumulative_exposure_s: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvironmentalHazardError {
    InvalidConfiguration,
    InvalidInput,
}

#[derive(Debug, Clone)]
pub struct EnvironmentalHazardModel {
    config: EnvironmentalHazardConfig,
    state: EnvironmentalHazardState,
}

impl Default for EnvironmentalHazardModel {
    fn default() -> Self {
        Self::new(EnvironmentalHazardConfig::default())
            .expect("default environmental-hazard config must remain valid")
    }
}

impl EnvironmentalHazardModel {
    pub fn new(config: EnvironmentalHazardConfig) -> Result<Self, EnvironmentalHazardError> {
        config.validate()?;
        Ok(Self {
            config,
            state: EnvironmentalHazardState {
                accreted_ice_mm: 0.0,
                dust_load: 0.0,
                salt_load: 0.0,
                sensor_occlusion: 0.0,
                main_rotor_efficiency: 1.0,
                tail_rotor_efficiency: 1.0,
                engine_power_fraction: 1.0,
                sensor_quality: 1.0,
                level: EnvironmentalHazardLevel::Nominal,
                cumulative_exposure_s: 0.0,
            },
        })
    }

    pub fn state(&self) -> EnvironmentalHazardState {
        self.state
    }

    pub fn reset(&mut self) {
        self.state = Self::new(self.config)
            .expect("validated environmental-hazard config remains valid")
            .state;
    }

    pub fn step(
        &mut self,
        input: EnvironmentalHazardInput,
    ) -> Result<EnvironmentalHazardState, EnvironmentalHazardError> {
        self.config.validate()?;
        if ![
            input.temperature_c,
            input.liquid_water_content_g_m3,
            input.precipitation_mm_h,
            input.dust_mg_m3,
            input.salt_aerosol_mg_m3,
            input.visibility_m,
            input.true_airspeed_mps,
            input.anti_ice_command,
            input.dt_s,
        ]
        .iter()
        .all(|value| value.is_finite())
            || input.liquid_water_content_g_m3 < 0.0
            || input.precipitation_mm_h < 0.0
            || input.dust_mg_m3 < 0.0
            || input.salt_aerosol_mg_m3 < 0.0
            || input.visibility_m <= 0.0
            || input.true_airspeed_mps < 0.0
            || !(0.0..=1.0).contains(&input.anti_ice_command)
            || input.dt_s <= 0.0
        {
            return Err(EnvironmentalHazardError::InvalidInput);
        }

        let in_icing_band = (self.config.minimum_icing_temperature_c
            ..=self.config.maximum_icing_temperature_c)
            .contains(&input.temperature_c);
        if in_icing_band && input.liquid_water_content_g_m3 > 0.0 {
            let airspeed_factor = (input.true_airspeed_mps / 30.0).clamp(0.1, 2.0);
            let anti_ice_reduction =
                1.0 - input.anti_ice_command * self.config.anti_ice_effectiveness;
            self.state.accreted_ice_mm += self.config.icing_rate_mm_s_per_g_m3
                * input.liquid_water_content_g_m3
                * airspeed_factor
                * anti_ice_reduction
                * input.dt_s;
        } else if input.temperature_c > self.config.maximum_icing_temperature_c {
            let excess_temperature = input.temperature_c - self.config.maximum_icing_temperature_c;
            self.state.accreted_ice_mm = (self.state.accreted_ice_mm
                - self.config.melt_rate_mm_s_per_c * excess_temperature * input.dt_s)
                .max(0.0);
        }

        self.state.dust_load +=
            self.config.dust_accumulation_per_mg_m3_s * input.dust_mg_m3 * input.dt_s;
        self.state.dust_load = (self.state.dust_load
            - self.config.dust_washout_per_mm_h_s * input.precipitation_mm_h * input.dt_s)
            .max(0.0);
        self.state.salt_load +=
            self.config.salt_accumulation_per_mg_m3_s * input.salt_aerosol_mg_m3 * input.dt_s;

        let visibility_occlusion = (1.0 - input.visibility_m / 5_000.0).clamp(0.0, 1.0);
        let precipitation_occlusion = (input.precipitation_mm_h / 50.0).clamp(0.0, 1.0);
        let dust_occlusion = (input.dust_mg_m3 / 20.0).clamp(0.0, 1.0);
        let target_occlusion = visibility_occlusion
            .max(precipitation_occlusion)
            .max(dust_occlusion);
        let alpha = 1.0 - (-input.dt_s / self.config.sensor_time_constant_s).exp();
        self.state.sensor_occlusion += alpha * (target_occlusion - self.state.sensor_occlusion);
        self.state.sensor_occlusion = self.state.sensor_occlusion.clamp(0.0, 1.0);

        self.state.main_rotor_efficiency =
            (1.0 - self.config.rotor_loss_per_mm_ice * self.state.accreted_ice_mm).clamp(0.2, 1.0);
        self.state.tail_rotor_efficiency = (1.0
            - 1.25 * self.config.rotor_loss_per_mm_ice * self.state.accreted_ice_mm)
            .clamp(0.15, 1.0);
        self.state.engine_power_fraction = (1.0
            - self.config.engine_loss_per_dust_load * self.state.dust_load
            - 0.02 * self.state.accreted_ice_mm)
            .clamp(0.0, 1.0);
        self.state.sensor_quality = (1.0 - self.state.sensor_occlusion).clamp(0.0, 1.0);
        self.state.cumulative_exposure_s += input.dt_s;

        self.state.level = if self.state.engine_power_fraction
            <= self.config.land_engine_power_fraction
            || self.state.tail_rotor_efficiency <= 0.5
        {
            EnvironmentalHazardLevel::LandAsSoonAsPracticable
        } else if self.state.main_rotor_efficiency <= self.config.exit_rotor_efficiency
            || self.state.sensor_quality <= 0.35
        {
            EnvironmentalHazardLevel::ExitEnvironment
        } else if self.state.main_rotor_efficiency <= self.config.caution_rotor_efficiency
            || self.state.sensor_quality <= 0.7
            || self.state.dust_load >= 0.5
        {
            EnvironmentalHazardLevel::Caution
        } else {
            EnvironmentalHazardLevel::Nominal
        };
        Ok(self.state)
    }
}

fn valid_fraction(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nominal_input() -> EnvironmentalHazardInput {
        EnvironmentalHazardInput {
            temperature_c: 15.0,
            liquid_water_content_g_m3: 0.0,
            precipitation_mm_h: 0.0,
            dust_mg_m3: 0.0,
            salt_aerosol_mg_m3: 0.0,
            visibility_m: 10_000.0,
            true_airspeed_mps: 30.0,
            anti_ice_command: 0.0,
            dt_s: 1.0,
        }
    }

    #[test]
    fn supercooled_water_accumulates_ice_and_reduces_rotor_authority() {
        let mut model = EnvironmentalHazardModel::default();
        let mut input = nominal_input();
        input.temperature_c = -5.0;
        input.liquid_water_content_g_m3 = 2.0;
        for _ in 0..200 {
            model.step(input).unwrap();
        }
        assert!(model.state().accreted_ice_mm > 0.0);
        assert!(model.state().main_rotor_efficiency < 1.0);
    }

    #[test]
    fn anti_ice_reduces_accretion_rate() {
        let mut unprotected = EnvironmentalHazardModel::default();
        let mut protected = EnvironmentalHazardModel::default();
        let mut input = nominal_input();
        input.temperature_c = -5.0;
        input.liquid_water_content_g_m3 = 2.0;
        unprotected.step(input).unwrap();
        input.anti_ice_command = 1.0;
        protected.step(input).unwrap();
        assert!(protected.state().accreted_ice_mm < unprotected.state().accreted_ice_mm);
    }

    #[test]
    fn dust_degrades_engine_and_sensor_quality() {
        let mut model = EnvironmentalHazardModel::default();
        let mut input = nominal_input();
        input.dust_mg_m3 = 20.0;
        input.visibility_m = 100.0;
        for _ in 0..100 {
            model.step(input).unwrap();
        }
        assert!(model.state().engine_power_fraction < 1.0);
        assert!(model.state().sensor_quality < 0.5);
        assert!(model.state().level >= EnvironmentalHazardLevel::Caution);
    }

    #[test]
    fn warm_air_melts_ice_gradually_not_instantly() {
        let mut model = EnvironmentalHazardModel::default();
        let mut input = nominal_input();
        input.temperature_c = -5.0;
        input.liquid_water_content_g_m3 = 5.0;
        for _ in 0..100 {
            model.step(input).unwrap();
        }
        let iced = model.state().accreted_ice_mm;
        input.temperature_c = 10.0;
        input.liquid_water_content_g_m3 = 0.0;
        model.step(input).unwrap();
        assert!(model.state().accreted_ice_mm < iced);
        assert!(model.state().accreted_ice_mm > 0.0);
    }
}
