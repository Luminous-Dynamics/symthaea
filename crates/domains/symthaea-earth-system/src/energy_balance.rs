// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Zero-dimensional radiative-equilibrium climate baselines.
//!
//! The planet is treated as a single point in radiative equilibrium: absorbed
//! shortwave = emitted longwave. These models are deliberately reduced-order
//! reference calculations, not a resolved atmosphere or complete Earth-system
//! simulation.
//!
//! All SI units (W/m², K).

use crate::error::{ModelError, require_finite, require_fraction, require_positive};

/// Stefan-Boltzmann constant σ (W·m⁻²·K⁻⁴), CODATA 2018.
pub const STEFAN_BOLTZMANN: f64 = 5.670_374_419e-8;

/// Earth's total solar irradiance (solar constant), W/m².
pub const SOLAR_CONSTANT_EARTH: f64 = 1361.0;

/// Earth's Bond albedo (fraction of shortwave reflected).
pub const EARTH_ALBEDO: f64 = 0.30;

/// Pre-industrial CO₂ concentration (ppm), the standard forcing baseline.
pub const CO2_PREINDUSTRIAL_PPM: f64 = 280.0;

/// Blackbody temperature radiating a given longwave flux: `T = (F/σ)^¼`.
pub fn stefan_boltzmann_temperature(flux: f64) -> f64 {
    (flux / STEFAN_BOLTZMANN).powf(0.25)
}

/// Checked form of [`stefan_boltzmann_temperature`].
pub fn try_stefan_boltzmann_temperature(flux: f64) -> Result<f64, ModelError> {
    require_finite("flux", flux)?;
    if flux < 0.0 {
        return Err(ModelError::OutOfRange {
            parameter: "flux",
            value: flux,
            min: 0.0,
            max: f64::INFINITY,
        });
    }
    Ok(stefan_boltzmann_temperature(flux))
}

/// Longwave flux emitted by a blackbody at temperature `t`: `F = σT⁴`.
pub fn blackbody_flux(t: f64) -> f64 {
    STEFAN_BOLTZMANN * t.powi(4)
}

/// Checked form of [`blackbody_flux`].
pub fn try_blackbody_flux(t: f64) -> Result<f64, ModelError> {
    require_finite("temperature", t)?;
    if t < 0.0 {
        return Err(ModelError::OutOfRange {
            parameter: "temperature",
            value: t,
            min: 0.0,
            max: f64::INFINITY,
        });
    }
    Ok(blackbody_flux(t))
}

/// Planetary effective (emission) temperature with no greenhouse effect.
///
/// Absorbed shortwave averaged over the sphere is `S(1-α)/4`; equating to
/// `σT⁴` gives `T_eff = (S(1-α)/(4σ))^¼`.
pub fn effective_temperature(solar_constant: f64, albedo: f64) -> f64 {
    let absorbed = solar_constant * (1.0 - albedo) / 4.0;
    stefan_boltzmann_temperature(absorbed)
}

/// Checked form of [`effective_temperature`].
pub fn try_effective_temperature(solar_constant: f64, albedo: f64) -> Result<f64, ModelError> {
    require_positive("solar_constant", solar_constant)?;
    require_fraction("albedo", albedo)?;
    Ok(effective_temperature(solar_constant, albedo))
}

/// Surface temperature under an effective outgoing-longwave emissivity.
///
/// `T_s = (S(1-α)/(4·ε_eff·σ))^¼`. This is a calibrated, reduced-order
/// parameterization of the net longwave response, not the emissivity of a
/// literal single atmospheric layer. `ε_eff = 1` recovers
/// [`effective_temperature`].
pub fn effective_emissivity_surface_temperature(
    solar_constant: f64,
    albedo: f64,
    effective_olr_emissivity: f64,
) -> f64 {
    let absorbed = solar_constant * (1.0 - albedo) / 4.0;
    (absorbed / (effective_olr_emissivity * STEFAN_BOLTZMANN)).powf(0.25)
}

/// Checked form of [`effective_emissivity_surface_temperature`].
pub fn try_effective_emissivity_surface_temperature(
    solar_constant: f64,
    albedo: f64,
    effective_olr_emissivity: f64,
) -> Result<f64, ModelError> {
    require_positive("solar_constant", solar_constant)?;
    require_fraction("albedo", albedo)?;
    require_positive("effective_olr_emissivity", effective_olr_emissivity)?;
    if effective_olr_emissivity > 1.0 {
        return Err(ModelError::OutOfRange {
            parameter: "effective_olr_emissivity",
            value: effective_olr_emissivity,
            min: f64::MIN_POSITIVE,
            max: 1.0,
        });
    }
    Ok(effective_emissivity_surface_temperature(
        solar_constant,
        albedo,
        effective_olr_emissivity,
    ))
}

/// Backward-compatible alias for [`effective_emissivity_surface_temperature`].
#[deprecated(
    since = "0.1.1",
    note = "this is an effective OLR parameterization, not a literal grey-atmosphere layer; use effective_emissivity_surface_temperature"
)]
pub fn grey_atmosphere_surface_temperature(
    solar_constant: f64,
    albedo: f64,
    emissivity: f64,
) -> f64 {
    effective_emissivity_surface_temperature(solar_constant, albedo, emissivity)
}

/// CO₂ radiative forcing (W/m²) using Myhre et al. (1998):
/// `ΔF = 5.35·ln(C/C₀)`.
pub fn co2_radiative_forcing_myhre1998(concentration_ppm: f64, baseline_ppm: f64) -> f64 {
    5.35 * (concentration_ppm / baseline_ppm).ln()
}

/// Checked form of [`co2_radiative_forcing_myhre1998`].
pub fn try_co2_radiative_forcing_myhre1998(
    concentration_ppm: f64,
    baseline_ppm: f64,
) -> Result<f64, ModelError> {
    require_positive("concentration_ppm", concentration_ppm)?;
    require_positive("baseline_ppm", baseline_ppm)?;
    Ok(co2_radiative_forcing_myhre1998(
        concentration_ppm,
        baseline_ppm,
    ))
}

/// Backward-compatible alias for [`co2_radiative_forcing_myhre1998`].
#[deprecated(
    since = "0.1.1",
    note = "use the versioned co2_radiative_forcing_myhre1998 function"
)]
pub fn co2_radiative_forcing(concentration_ppm: f64, baseline_ppm: f64) -> f64 {
    co2_radiative_forcing_myhre1998(concentration_ppm, baseline_ppm)
}

/// Equilibrium surface warming from a radiative forcing: `ΔT = λ·ΔF`.
///
/// `climate_sensitivity_param` λ (K per W/m²); ~0.8 gives ~3 K per CO₂ doubling.
pub fn equilibrium_warming(forcing: f64, climate_sensitivity_param: f64) -> f64 {
    climate_sensitivity_param * forcing
}

/// A reduced-order planetary radiative-equilibrium configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnergyBalanceModel {
    /// Solar constant at the planet (W/m²).
    pub solar_constant: f64,
    /// Bond albedo.
    pub albedo: f64,
    /// Effective outgoing-longwave emissivity (1.0 = blackbody emission).
    ///
    /// The field name is retained for source compatibility. It is not the
    /// emissivity of a literal single atmospheric layer.
    pub emissivity: f64,
}

impl EnergyBalanceModel {
    /// Construct and validate a reduced-order energy-balance model.
    pub fn try_new(
        solar_constant: f64,
        albedo: f64,
        effective_olr_emissivity: f64,
    ) -> Result<Self, ModelError> {
        let model = Self {
            solar_constant,
            albedo,
            emissivity: effective_olr_emissivity,
        };
        model.validate()?;
        Ok(model)
    }

    /// Validate all physical parameter domains.
    pub fn validate(&self) -> Result<(), ModelError> {
        require_positive("solar_constant", self.solar_constant)?;
        require_fraction("albedo", self.albedo)?;
        require_positive("effective_olr_emissivity", self.emissivity)?;
        if self.emissivity > 1.0 {
            return Err(ModelError::OutOfRange {
                parameter: "effective_olr_emissivity",
                value: self.emissivity,
                min: f64::MIN_POSITIVE,
                max: 1.0,
            });
        }
        Ok(())
    }

    /// Earth with an effective OLR emissivity calibrated so the surface sits
    /// near the observed ~288 K global mean.
    pub fn earth() -> EnergyBalanceModel {
        EnergyBalanceModel {
            solar_constant: SOLAR_CONSTANT_EARTH,
            albedo: EARTH_ALBEDO,
            emissivity: 0.615,
        }
    }

    /// Effective outgoing-longwave emissivity.
    pub fn effective_olr_emissivity(&self) -> f64 {
        self.emissivity
    }

    /// Effective (no-greenhouse) temperature.
    pub fn effective_temperature(&self) -> f64 {
        effective_temperature(self.solar_constant, self.albedo)
    }

    /// Surface temperature under the calibrated effective-emissivity model.
    pub fn surface_temperature(&self) -> f64 {
        effective_emissivity_surface_temperature(self.solar_constant, self.albedo, self.emissivity)
    }

    /// Calibrated surface-minus-effective temperature difference, K.
    pub fn greenhouse_warming(&self) -> f64 {
        self.surface_temperature() - self.effective_temperature()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn earth_effective_temperature_matches_canonical() {
        let t = effective_temperature(SOLAR_CONSTANT_EARTH, EARTH_ALBEDO);
        assert!((t - 254.6).abs() < 0.5, "T_eff = {t}");
    }

    #[test]
    fn stefan_boltzmann_round_trips() {
        let f = blackbody_flux(288.0);
        assert!((stefan_boltzmann_temperature(f) - 288.0).abs() < 1e-6);
    }

    #[test]
    fn effective_emissivity_one_recovers_effective_temperature() {
        let t_parameterized =
            effective_emissivity_surface_temperature(SOLAR_CONSTANT_EARTH, EARTH_ALBEDO, 1.0);
        let t_eff = effective_temperature(SOLAR_CONSTANT_EARTH, EARTH_ALBEDO);
        assert!((t_parameterized - t_eff).abs() < 1e-9);
    }

    #[test]
    fn co2_doubling_forcing_is_canonical() {
        let df =
            co2_radiative_forcing_myhre1998(2.0 * CO2_PREINDUSTRIAL_PPM, CO2_PREINDUSTRIAL_PPM);
        assert!((df - 3.708).abs() < 0.01, "ΔF = {df}");
    }

    #[test]
    fn co2_doubling_warming_in_ipcc_range() {
        let df = co2_radiative_forcing_myhre1998(560.0, 280.0);
        let dt = equilibrium_warming(df, 0.8);
        assert!((2.0..4.0).contains(&dt), "ECS = {dt}");
    }

    #[test]
    fn earth_calibration_lifts_surface_above_effective() {
        let m = EnergyBalanceModel::earth();
        assert!(m.surface_temperature() > m.effective_temperature());
        assert!(
            (m.greenhouse_warming() - 33.0).abs() < 3.0,
            "ΔT_gh = {}",
            m.greenhouse_warming()
        );
    }
    #[test]
    fn checked_apis_reject_invalid_domains() {
        assert!(try_effective_temperature(-1.0, 0.3).is_err());
        assert!(try_effective_temperature(1361.0, 1.2).is_err());
        assert!(try_effective_emissivity_surface_temperature(1361.0, 0.3, 0.0).is_err());
        assert!(try_co2_radiative_forcing_myhre1998(0.0, 280.0).is_err());
        assert!(EnergyBalanceModel::try_new(1361.0, 0.3, 0.615).is_ok());
    }
}
