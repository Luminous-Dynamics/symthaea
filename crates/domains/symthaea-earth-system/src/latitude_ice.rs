// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Temperature-dependent albedo feedback for the equal-area latitudinal EBM.
//!
//! This module adds a smooth, local transition from each band's declared warm
//! albedo to a caller-visible cold albedo. It is an executable ice-albedo
//! feedback baseline, not a dynamic sea-ice, snow, glacier, or ice-sheet model.

use crate::error::{ModelError, require_finite, require_fraction, require_positive};
use crate::latitude::{LatitudinalEnergyBalanceModel, MAX_LATITUDE_STEPS};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TemperatureDependentAlbedo {
    /// Albedo approached far below the transition temperature.
    pub cold_albedo: f64,
    /// Midpoint temperature of the smooth transition, K.
    pub transition_temperature: f64,
    /// Temperature width of the transition, K.
    pub transition_width: f64,
}

impl TemperatureDependentAlbedo {
    pub fn try_new(
        cold_albedo: f64,
        transition_temperature: f64,
        transition_width: f64,
    ) -> Result<Self, ModelError> {
        let feedback = Self {
            cold_albedo,
            transition_temperature,
            transition_width,
        };
        feedback.validate()?;
        Ok(feedback)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        require_fraction("cold_albedo", self.cold_albedo)?;
        require_positive("transition_temperature", self.transition_temperature)?;
        require_positive("transition_width", self.transition_width)
    }

    /// Smooth cold-state fraction in `[0, 1]`.
    pub fn cold_fraction(&self, temperature: f64) -> Result<f64, ModelError> {
        self.validate()?;
        require_positive("temperature", temperature)?;
        Ok(0.5
            * (1.0 - ((temperature - self.transition_temperature) / self.transition_width).tanh()))
    }

    pub fn effective_albedo(&self, warm_albedo: f64, temperature: f64) -> Result<f64, ModelError> {
        require_fraction("warm_albedo", warm_albedo)?;
        let fraction = self.cold_fraction(temperature)?;
        Ok(warm_albedo + fraction * (self.cold_albedo - warm_albedo))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LatitudinalIceAlbedoModel {
    pub climate: LatitudinalEnergyBalanceModel,
    pub albedo_feedback: TemperatureDependentAlbedo,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LatitudeIceSample {
    pub time_seconds: f64,
    pub temperatures: Vec<f64>,
    pub global_mean_temperature: f64,
    pub global_mean_albedo: f64,
    pub global_mean_cold_fraction: f64,
    pub global_energy_imbalance: f64,
}

impl LatitudinalIceAlbedoModel {
    pub fn try_new(
        climate: LatitudinalEnergyBalanceModel,
        albedo_feedback: TemperatureDependentAlbedo,
    ) -> Result<Self, ModelError> {
        let model = Self {
            climate,
            albedo_feedback,
        };
        model.validate()?;
        Ok(model)
    }

    /// Illustrative feedback layered on the existing Earth-like annual-mean
    /// zonal model. Parameters are not an observational ice-line fit.
    pub fn earthlike(bands: usize) -> Result<Self, ModelError> {
        Self::try_new(
            LatitudinalEnergyBalanceModel::earthlike(bands)?,
            TemperatureDependentAlbedo::try_new(0.68, 263.15, 5.0)?,
        )
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        self.climate.validate()?;
        self.albedo_feedback.validate()?;
        let warm_max = self
            .climate
            .equatorial_albedo
            .max(self.climate.polar_albedo);
        if self.albedo_feedback.cold_albedo < warm_max {
            return Err(ModelError::InvalidOrdering {
                lower: "maximum_warm_albedo",
                lower_value: warm_max,
                upper: "cold_albedo",
                upper_value: self.albedo_feedback.cold_albedo,
            });
        }
        Ok(())
    }

    pub fn band_albedo(&self, band: usize, temperature: f64) -> Result<f64, ModelError> {
        self.validate()?;
        self.validate_band(band)?;
        self.albedo_feedback
            .effective_albedo(self.climate.albedo(band), temperature)
    }

    pub fn band_cold_fraction(&self, band: usize, temperature: f64) -> Result<f64, ModelError> {
        self.validate()?;
        self.validate_band(band)?;
        self.albedo_feedback.cold_fraction(temperature)
    }

    pub fn global_mean_albedo(&self, temperatures: &[f64]) -> Result<f64, ModelError> {
        self.validate_temperatures(temperatures)?;
        let mut total = 0.0;
        for (band, temperature) in temperatures.iter().copied().enumerate() {
            total += self.band_albedo(band, temperature)?;
        }
        Ok(total / self.climate.bands as f64)
    }

    pub fn global_mean_cold_fraction(&self, temperatures: &[f64]) -> Result<f64, ModelError> {
        self.validate_temperatures(temperatures)?;
        let mut total = 0.0;
        for (band, temperature) in temperatures.iter().copied().enumerate() {
            total += self.band_cold_fraction(band, temperature)?;
        }
        Ok(total / self.climate.bands as f64)
    }

    pub fn global_energy_imbalance(
        &self,
        temperatures: &[f64],
        uniform_forcing: f64,
    ) -> Result<f64, ModelError> {
        self.validate_temperatures(temperatures)?;
        require_finite("uniform_forcing", uniform_forcing)?;
        let mut total = 0.0;
        for (band, temperature) in temperatures.iter().copied().enumerate() {
            let albedo = self.band_albedo(band, temperature)?;
            total += self.climate.insolation(band) * (1.0 - albedo) + uniform_forcing
                - self.climate.outgoing_longwave(temperature);
        }
        Ok(total / self.climate.bands as f64)
    }

    pub fn tendencies(
        &self,
        temperatures: &[f64],
        uniform_forcing: f64,
    ) -> Result<Vec<f64>, ModelError> {
        self.validate()?;
        self.validate_temperatures(temperatures)?;
        require_finite("uniform_forcing", uniform_forcing)?;
        let transport = self.climate.transport_convergence(temperatures)?;
        let mut tendencies = Vec::with_capacity(self.climate.bands);
        for (band, temperature) in temperatures.iter().copied().enumerate() {
            let albedo = self.band_albedo(band, temperature)?;
            tendencies.push(
                (self.climate.insolation(band) * (1.0 - albedo) + uniform_forcing
                    - self.climate.outgoing_longwave(temperature)
                    + transport[band])
                    / self.climate.heat_capacity,
            );
        }
        Ok(tendencies)
    }

    pub fn step_rk4(
        &self,
        temperatures: &[f64],
        uniform_forcing: f64,
        dt_seconds: f64,
    ) -> Result<Vec<f64>, ModelError> {
        require_positive("dt_seconds", dt_seconds)?;
        let k1 = self.tendencies(temperatures, uniform_forcing)?;
        let stage2 = add_scaled(temperatures, &k1, 0.5 * dt_seconds);
        self.validate_temperatures(&stage2)?;
        let k2 = self.tendencies(&stage2, uniform_forcing)?;
        let stage3 = add_scaled(temperatures, &k2, 0.5 * dt_seconds);
        self.validate_temperatures(&stage3)?;
        let k3 = self.tendencies(&stage3, uniform_forcing)?;
        let stage4 = add_scaled(temperatures, &k3, dt_seconds);
        self.validate_temperatures(&stage4)?;
        let k4 = self.tendencies(&stage4, uniform_forcing)?;
        let next: Vec<f64> = temperatures
            .iter()
            .zip(&k1)
            .zip(&k2)
            .zip(&k3)
            .zip(&k4)
            .map(|((((temperature, k1), k2), k3), k4)| {
                temperature + dt_seconds * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            })
            .collect();
        self.validate_temperatures(&next)?;
        Ok(next)
    }

    pub fn simulate(
        &self,
        initial_temperatures: &[f64],
        uniform_forcing: f64,
        dt_seconds: f64,
        steps: usize,
    ) -> Result<Vec<LatitudeIceSample>, ModelError> {
        self.validate()?;
        self.validate_temperatures(initial_temperatures)?;
        require_finite("uniform_forcing", uniform_forcing)?;
        require_positive("dt_seconds", dt_seconds)?;
        if steps == 0 {
            return Err(ModelError::ZeroSteps);
        }
        if steps > MAX_LATITUDE_STEPS {
            return Err(ModelError::TrajectoryTooLarge {
                requested: steps,
                maximum: MAX_LATITUDE_STEPS,
            });
        }
        require_finite("duration_seconds", dt_seconds * steps as f64)?;
        let capacity = steps.checked_add(1).ok_or(ModelError::TrajectoryTooLarge {
            requested: usize::MAX,
            maximum: MAX_LATITUDE_STEPS,
        })?;
        let mut samples = Vec::with_capacity(capacity);
        let mut temperatures = initial_temperatures.to_vec();
        samples.push(self.sample(0.0, temperatures.clone(), uniform_forcing)?);
        for step in 1..=steps {
            temperatures = self.step_rk4(&temperatures, uniform_forcing, dt_seconds)?;
            samples.push(self.sample(
                step as f64 * dt_seconds,
                temperatures.clone(),
                uniform_forcing,
            )?);
        }
        Ok(samples)
    }

    fn sample(
        &self,
        time_seconds: f64,
        temperatures: Vec<f64>,
        uniform_forcing: f64,
    ) -> Result<LatitudeIceSample, ModelError> {
        Ok(LatitudeIceSample {
            time_seconds,
            global_mean_temperature: self.climate.global_mean_temperature(&temperatures)?,
            global_mean_albedo: self.global_mean_albedo(&temperatures)?,
            global_mean_cold_fraction: self.global_mean_cold_fraction(&temperatures)?,
            global_energy_imbalance: self
                .global_energy_imbalance(&temperatures, uniform_forcing)?,
            temperatures,
        })
    }

    fn validate_temperatures(&self, temperatures: &[f64]) -> Result<(), ModelError> {
        if temperatures.len() != self.climate.bands {
            return Err(ModelError::DimensionMismatch {
                context: "latitude_temperatures",
                expected: self.climate.bands,
                found: temperatures.len(),
            });
        }
        for temperature in temperatures {
            require_positive("latitude_temperature", *temperature)?;
        }
        Ok(())
    }

    fn validate_band(&self, band: usize) -> Result<(), ModelError> {
        if band < self.climate.bands {
            Ok(())
        } else {
            Err(ModelError::OutOfRange {
                parameter: "latitude_band",
                value: band as f64,
                min: 0.0,
                max: (self.climate.bands - 1) as f64,
            })
        }
    }
}

fn add_scaled(values: &[f64], derivatives: &[f64], scale: f64) -> Vec<f64> {
    values
        .iter()
        .zip(derivatives)
        .map(|(value, derivative)| value + scale * derivative)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transient::SECONDS_PER_YEAR;

    #[test]
    fn cold_fraction_and_albedo_are_monotonic() {
        let feedback = TemperatureDependentAlbedo::try_new(0.7, 263.15, 5.0).unwrap();
        let cold = feedback.cold_fraction(245.0).unwrap();
        let warm = feedback.cold_fraction(285.0).unwrap();
        assert!(cold > warm);
        assert!(
            feedback.effective_albedo(0.3, 245.0).unwrap()
                > feedback.effective_albedo(0.3, 285.0).unwrap()
        );
    }

    #[test]
    fn global_tendency_closes_against_top_of_atmosphere_imbalance() {
        let model = LatitudinalIceAlbedoModel::earthlike(18).unwrap();
        let temperatures: Vec<f64> = (0..18)
            .map(|band| 280.0 + 8.0 * (1.0 - model.climate.x_center(band).abs()))
            .collect();
        let tendencies = model.tendencies(&temperatures, 1.5).unwrap();
        let mean_heat_tendency =
            tendencies.iter().sum::<f64>() / tendencies.len() as f64 * model.climate.heat_capacity;
        let imbalance = model.global_energy_imbalance(&temperatures, 1.5).unwrap();
        assert!((mean_heat_tendency - imbalance).abs() < 1e-10);
    }

    #[test]
    fn colder_zones_absorb_less_solar_energy() {
        let model = LatitudinalIceAlbedoModel::earthlike(18).unwrap();
        let band = 9;
        let warm_albedo = model.band_albedo(band, 290.0).unwrap();
        let cold_albedo = model.band_albedo(band, 245.0).unwrap();
        assert!(cold_albedo > warm_albedo);
        let warm_absorbed = model.climate.insolation(band) * (1.0 - warm_albedo);
        let cold_absorbed = model.climate.insolation(band) * (1.0 - cold_albedo);
        assert!(cold_absorbed < warm_absorbed);
    }

    #[test]
    fn mutated_feedback_that_reverses_ice_contrast_is_rejected() {
        let mut model = LatitudinalIceAlbedoModel::earthlike(18).unwrap();
        model.albedo_feedback.cold_albedo = 0.2;
        assert!(model.global_mean_albedo(&vec![280.0; 18]).is_err());
    }

    #[test]
    fn trajectory_includes_initial_state_and_stays_physical() {
        let model = LatitudinalIceAlbedoModel::earthlike(18).unwrap();
        let initial = vec![288.0; 18];
        let samples = model
            .simulate(&initial, 0.0, 0.05 * SECONDS_PER_YEAR, 20)
            .unwrap();
        assert_eq!(samples.len(), 21);
        assert_eq!(samples[0].time_seconds, 0.0);
        assert!(samples.iter().all(|sample| {
            sample.global_mean_temperature.is_finite()
                && (0.0..=1.0).contains(&sample.global_mean_albedo)
                && (0.0..=1.0).contains(&sample.global_mean_cold_fraction)
        }));
    }
}
