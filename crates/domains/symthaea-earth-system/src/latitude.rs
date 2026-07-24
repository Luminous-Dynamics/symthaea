// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Equal-area diffusive latitudinal energy-balance model.
//!
//! The coordinate `x = sin(latitude)` makes equal-width cells equal-area. The
//! transport operator is the conservative spherical diffusion term
//! `d/dx[(1-x²)dT/dx]`; face factors vanish at both poles, so internal heat
//! transport sums to zero without special boundary fluxes. Insolation uses a
//! caller-visible quadrupole approximation and outgoing longwave radiation is
//! linearized around a reference temperature.

use crate::error::{ModelError, require_finite, require_non_negative, require_positive};

pub const MAX_LATITUDE_BANDS: usize = 128;
pub const MAX_LATITUDE_STEPS: usize = 1_000_000;

#[derive(Debug, Clone, PartialEq)]
pub struct LatitudinalEnergyBalanceModel {
    pub bands: usize,
    /// Effective heat capacity, J m⁻² K⁻¹.
    pub heat_capacity: f64,
    /// Linear outgoing-longwave feedback, W m⁻² K⁻¹.
    pub outgoing_feedback: f64,
    /// Meridional diffusion coefficient, W m⁻² K⁻¹.
    pub diffusion: f64,
    pub reference_temperature: f64,
    /// Outgoing longwave radiation at the reference temperature, W/m².
    pub outgoing_at_reference: f64,
    pub solar_constant: f64,
    /// Coefficient multiplying `P2(sin(latitude))` in annual-mean insolation.
    pub insolation_quadrupole: f64,
    /// Fixed albedo at the equator.
    pub equatorial_albedo: f64,
    /// Fixed albedo at either pole.
    pub polar_albedo: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LatitudeSample {
    pub time_seconds: f64,
    pub temperatures: Vec<f64>,
    pub global_mean_temperature: f64,
    pub global_energy_imbalance: f64,
}

impl LatitudinalEnergyBalanceModel {
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        bands: usize,
        heat_capacity: f64,
        outgoing_feedback: f64,
        diffusion: f64,
        reference_temperature: f64,
        outgoing_at_reference: f64,
        solar_constant: f64,
        insolation_quadrupole: f64,
        equatorial_albedo: f64,
        polar_albedo: f64,
    ) -> Result<Self, ModelError> {
        let model = Self {
            bands,
            heat_capacity,
            outgoing_feedback,
            diffusion,
            reference_temperature,
            outgoing_at_reference,
            solar_constant,
            insolation_quadrupole,
            equatorial_albedo,
            polar_albedo,
        };
        model.validate()?;
        Ok(model)
    }

    /// Illustrative annual-mean Earth-like parameters, not an observational
    /// fit. The OLR intercept is chosen so the reference temperature has zero
    /// global-mean imbalance under the declared fixed albedo profile.
    pub fn earthlike(bands: usize) -> Result<Self, ModelError> {
        let mut model = Self::try_new(
            bands, 4.0e8, 2.0, 0.6, 288.0, 240.0, 1361.0, -0.48, 0.28, 0.55,
        )?;
        model.outgoing_at_reference = model.global_mean_absorbed_solar();
        Ok(model)
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        if !(4..=MAX_LATITUDE_BANDS).contains(&self.bands) {
            return Err(ModelError::OutOfRange {
                parameter: "latitude_bands",
                value: self.bands as f64,
                min: 4.0,
                max: MAX_LATITUDE_BANDS as f64,
            });
        }
        require_positive("heat_capacity", self.heat_capacity)?;
        require_positive("outgoing_feedback", self.outgoing_feedback)?;
        require_non_negative("diffusion", self.diffusion)?;
        require_positive("reference_temperature", self.reference_temperature)?;
        require_non_negative("outgoing_at_reference", self.outgoing_at_reference)?;
        require_positive("solar_constant", self.solar_constant)?;
        require_finite("insolation_quadrupole", self.insolation_quadrupole)?;
        require_fraction("equatorial_albedo", self.equatorial_albedo)?;
        require_fraction("polar_albedo", self.polar_albedo)?;
        for band in 0..self.bands {
            require_non_negative("annual_mean_insolation", self.insolation(band))?;
        }
        Ok(())
    }

    pub fn cell_width(&self) -> f64 {
        2.0 / self.bands as f64
    }

    /// Equal-area cell-center coordinate `x = sin(latitude)`.
    pub fn x_center(&self, band: usize) -> f64 {
        -1.0 + (band as f64 + 0.5) * self.cell_width()
    }

    pub fn latitude_radians(&self, band: usize) -> f64 {
        self.x_center(band).asin()
    }

    pub fn insolation(&self, band: usize) -> f64 {
        let dx = self.cell_width();
        let x_lower = -1.0 + band as f64 * dx;
        let x_upper = x_lower + dx;
        // Exact cell average of P2(x) = (3x² - 1)/2. Using the cell
        // center would bias the discrete global mean on every finite grid.
        let p2_mean = 0.5 * ((x_upper.powi(3) - x_upper) - (x_lower.powi(3) - x_lower)) / dx;
        self.solar_constant / 4.0 * (1.0 + self.insolation_quadrupole * p2_mean)
    }

    pub fn albedo(&self, band: usize) -> f64 {
        let x = self.x_center(band);
        self.equatorial_albedo + (self.polar_albedo - self.equatorial_albedo) * x * x
    }

    pub fn absorbed_solar(&self, band: usize) -> f64 {
        self.insolation(band) * (1.0 - self.albedo(band))
    }

    pub fn outgoing_longwave(&self, temperature: f64) -> f64 {
        self.outgoing_at_reference
            + self.outgoing_feedback * (temperature - self.reference_temperature)
    }

    pub fn global_mean_absorbed_solar(&self) -> f64 {
        (0..self.bands)
            .map(|band| self.absorbed_solar(band))
            .sum::<f64>()
            / self.bands as f64
    }

    pub fn global_mean_temperature(&self, temperatures: &[f64]) -> Result<f64, ModelError> {
        self.validate_temperatures(temperatures)?;
        Ok(temperatures.iter().sum::<f64>() / self.bands as f64)
    }

    pub fn global_energy_imbalance(
        &self,
        temperatures: &[f64],
        uniform_forcing: f64,
    ) -> Result<f64, ModelError> {
        self.validate_temperatures(temperatures)?;
        require_finite("uniform_forcing", uniform_forcing)?;
        Ok((0..self.bands)
            .map(|band| {
                self.absorbed_solar(band) + uniform_forcing
                    - self.outgoing_longwave(temperatures[band])
            })
            .sum::<f64>()
            / self.bands as f64)
    }

    /// Conservative spherical diffusion convergence in W/m² for each band.
    pub fn transport_convergence(&self, temperatures: &[f64]) -> Result<Vec<f64>, ModelError> {
        self.validate_temperatures(temperatures)?;
        let dx = self.cell_width();
        let mut face_flux = vec![0.0; self.bands + 1];
        for face in 1..self.bands {
            let x_face = -1.0 + face as f64 * dx;
            face_flux[face] = self.diffusion
                * (1.0 - x_face * x_face)
                * (temperatures[face] - temperatures[face - 1])
                / dx;
        }
        Ok((0..self.bands)
            .map(|band| (face_flux[band + 1] - face_flux[band]) / dx)
            .collect())
    }

    pub fn tendencies(
        &self,
        temperatures: &[f64],
        uniform_forcing: f64,
    ) -> Result<Vec<f64>, ModelError> {
        self.validate()?;
        self.validate_temperatures(temperatures)?;
        require_finite("uniform_forcing", uniform_forcing)?;
        let transport = self.transport_convergence(temperatures)?;
        Ok((0..self.bands)
            .map(|band| {
                (self.absorbed_solar(band) + uniform_forcing
                    - self.outgoing_longwave(temperatures[band])
                    + transport[band])
                    / self.heat_capacity
            })
            .collect())
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
    ) -> Result<Vec<LatitudeSample>, ModelError> {
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
        let duration = dt_seconds * steps as f64;
        require_finite("duration_seconds", duration)?;

        let mut temperatures = initial_temperatures.to_vec();
        let mut samples = Vec::with_capacity(steps + 1);
        samples.push(self.sample(0.0, &temperatures, uniform_forcing)?);
        for step in 1..=steps {
            temperatures = self.step_rk4(&temperatures, uniform_forcing, dt_seconds)?;
            samples.push(self.sample(step as f64 * dt_seconds, &temperatures, uniform_forcing)?);
        }
        Ok(samples)
    }

    fn sample(
        &self,
        time_seconds: f64,
        temperatures: &[f64],
        uniform_forcing: f64,
    ) -> Result<LatitudeSample, ModelError> {
        Ok(LatitudeSample {
            time_seconds,
            temperatures: temperatures.to_vec(),
            global_mean_temperature: self.global_mean_temperature(temperatures)?,
            global_energy_imbalance: self.global_energy_imbalance(temperatures, uniform_forcing)?,
        })
    }

    fn validate_temperatures(&self, temperatures: &[f64]) -> Result<(), ModelError> {
        if temperatures.len() != self.bands {
            return Err(ModelError::DimensionMismatch {
                context: "latitudinal temperature state",
                expected: self.bands,
                found: temperatures.len(),
            });
        }
        for &temperature in temperatures {
            require_positive("band_temperature", temperature)?;
        }
        Ok(())
    }
}

fn require_fraction(parameter: &'static str, value: f64) -> Result<(), ModelError> {
    require_finite(parameter, value)?;
    if (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(ModelError::OutOfRange {
            parameter,
            value,
            min: 0.0,
            max: 1.0,
        })
    }
}

fn add_scaled(state: &[f64], derivative: &[f64], scale: f64) -> Vec<f64> {
    state
        .iter()
        .zip(derivative)
        .map(|(state, derivative)| state + scale * derivative)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transient::SECONDS_PER_YEAR;

    #[test]
    fn equal_area_insolation_has_the_declared_global_mean() {
        for bands in [4, 18, 32, 127] {
            let model = LatitudinalEnergyBalanceModel::earthlike(bands).unwrap();
            let mean = (0..model.bands)
                .map(|band| model.insolation(band))
                .sum::<f64>()
                / model.bands as f64;
            assert!((mean - model.solar_constant / 4.0).abs() < 1.0e-12);
        }
    }

    #[test]
    fn spherical_transport_is_globally_conservative() {
        let model = LatitudinalEnergyBalanceModel::earthlike(18).unwrap();
        let temperatures: Vec<_> = (0..model.bands)
            .map(|band| 260.0 + 30.0 * (1.0 - model.x_center(band).abs()))
            .collect();
        let convergence = model.transport_convergence(&temperatures).unwrap();
        assert!(convergence.iter().sum::<f64>().abs() < 1.0e-10);
    }

    #[test]
    fn global_tendency_matches_top_of_atmosphere_imbalance() {
        let model = LatitudinalEnergyBalanceModel::earthlike(18).unwrap();
        let temperatures = vec![288.0; model.bands];
        let forcing = 3.7;
        let tendencies = model.tendencies(&temperatures, forcing).unwrap();
        let mean_heat_tendency =
            model.heat_capacity * tendencies.iter().sum::<f64>() / model.bands as f64;
        let imbalance = model
            .global_energy_imbalance(&temperatures, forcing)
            .unwrap();
        assert!((mean_heat_tendency - imbalance).abs() < 1.0e-10);
    }

    #[test]
    fn calibrated_reference_has_reference_global_mean_at_equilibrium() {
        let model = LatitudinalEnergyBalanceModel::earthlike(18).unwrap();
        let initial = vec![288.0; model.bands];
        let samples = model
            .simulate(&initial, 0.0, 10.0 * 24.0 * 3600.0, 5_000)
            .unwrap();
        let final_sample = samples.last().unwrap();
        assert!((final_sample.global_mean_temperature - 288.0).abs() < 1.0e-5);
        assert!(final_sample.global_energy_imbalance.abs() < 1.0e-5);
        assert!(final_sample.temperatures[model.bands / 2] > final_sample.temperatures[0]);
        assert!(final_sample.time_seconds > 100.0 * SECONDS_PER_YEAR);
    }
}
