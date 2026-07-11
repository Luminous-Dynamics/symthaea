// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Zero-dimensional energy-balance climate model (EBM).
//!
//! The planet is treated as a single point in radiative equilibrium: absorbed
//! shortwave = emitted longwave. This is the foundational climate model
//! (Budyko 1969, Sellers 1969) and reproduces planetary effective temperatures
//! and CO₂ radiative forcing to good accuracy despite its simplicity.
//!
//! All SI units (W/m², K).

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

/// Longwave flux emitted by a blackbody at temperature `t`: `F = σT⁴`.
pub fn blackbody_flux(t: f64) -> f64 {
    STEFAN_BOLTZMANN * t.powi(4)
}

/// Planetary effective (emission) temperature with no greenhouse effect.
///
/// Absorbed shortwave averaged over the sphere is `S(1-α)/4`; equating to
/// `σT⁴` gives `T_eff = (S(1-α)/(4σ))^¼`.
pub fn effective_temperature(solar_constant: f64, albedo: f64) -> f64 {
    let absorbed = solar_constant * (1.0 - albedo) / 4.0;
    stefan_boltzmann_temperature(absorbed)
}

/// Surface temperature with a grey atmosphere of longwave emissivity `emissivity`.
///
/// A lower effective emissivity traps longwave (greenhouse): `T_s =
/// (S(1-α)/(4·ε·σ))^¼`. `emissivity = 1` recovers [`effective_temperature`].
pub fn grey_atmosphere_surface_temperature(
    solar_constant: f64,
    albedo: f64,
    emissivity: f64,
) -> f64 {
    let absorbed = solar_constant * (1.0 - albedo) / 4.0;
    (absorbed / (emissivity * STEFAN_BOLTZMANN)).powf(0.25)
}

/// CO₂ radiative forcing (W/m²): `ΔF = 5.35·ln(C/C₀)` (Myhre et al. 1998).
pub fn co2_radiative_forcing(concentration_ppm: f64, baseline_ppm: f64) -> f64 {
    5.35 * (concentration_ppm / baseline_ppm).ln()
}

/// Equilibrium surface warming from a radiative forcing: `ΔT = λ·ΔF`.
///
/// `climate_sensitivity_param` λ (K per W/m²); ~0.8 gives ~3 K per CO₂ doubling.
pub fn equilibrium_warming(forcing: f64, climate_sensitivity_param: f64) -> f64 {
    climate_sensitivity_param * forcing
}

/// A planetary energy-balance configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnergyBalanceModel {
    /// Solar constant at the planet (W/m²).
    pub solar_constant: f64,
    /// Bond albedo.
    pub albedo: f64,
    /// Grey-atmosphere longwave emissivity (1.0 = no greenhouse).
    pub emissivity: f64,
}

impl EnergyBalanceModel {
    /// Earth with a grey-atmosphere emissivity tuned so the surface sits near
    /// the observed ~288 K global mean.
    pub fn earth() -> EnergyBalanceModel {
        EnergyBalanceModel {
            solar_constant: SOLAR_CONSTANT_EARTH,
            albedo: EARTH_ALBEDO,
            emissivity: 0.615,
        }
    }

    /// Effective (no-greenhouse) temperature.
    pub fn effective_temperature(&self) -> f64 {
        effective_temperature(self.solar_constant, self.albedo)
    }

    /// Surface temperature including the grey-atmosphere greenhouse effect.
    pub fn surface_temperature(&self) -> f64 {
        grey_atmosphere_surface_temperature(self.solar_constant, self.albedo, self.emissivity)
    }

    /// Greenhouse warming (surface − effective), K.
    pub fn greenhouse_warming(&self) -> f64 {
        self.surface_temperature() - self.effective_temperature()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn earth_effective_temperature_matches_canonical() {
        // Canonical Earth effective temperature ≈ 255 K.
        let t = effective_temperature(SOLAR_CONSTANT_EARTH, EARTH_ALBEDO);
        assert!((t - 254.6).abs() < 0.5, "T_eff = {t}");
    }

    #[test]
    fn stefan_boltzmann_round_trips() {
        let f = blackbody_flux(288.0);
        assert!((stefan_boltzmann_temperature(f) - 288.0).abs() < 1e-6);
    }

    #[test]
    fn emissivity_one_recovers_effective_temperature() {
        let t_grey = grey_atmosphere_surface_temperature(SOLAR_CONSTANT_EARTH, EARTH_ALBEDO, 1.0);
        let t_eff = effective_temperature(SOLAR_CONSTANT_EARTH, EARTH_ALBEDO);
        assert!((t_grey - t_eff).abs() < 1e-9);
    }

    #[test]
    fn co2_doubling_forcing_is_canonical() {
        // Doubling CO₂ ⇒ ΔF = 5.35·ln(2) ≈ 3.71 W/m².
        let df = co2_radiative_forcing(2.0 * CO2_PREINDUSTRIAL_PPM, CO2_PREINDUSTRIAL_PPM);
        assert!((df - 3.708).abs() < 0.01, "ΔF = {df}");
    }

    #[test]
    fn co2_doubling_warming_in_ipcc_range() {
        // λ ≈ 0.8 K/(W/m²) ⇒ ~3 K equilibrium sensitivity.
        let df = co2_radiative_forcing(560.0, 280.0);
        let dt = equilibrium_warming(df, 0.8);
        assert!((2.0..4.0).contains(&dt), "ECS = {dt}");
    }

    #[test]
    fn earth_greenhouse_lifts_surface_above_effective() {
        let m = EnergyBalanceModel::earth();
        assert!(m.surface_temperature() > m.effective_temperature());
        // Observed greenhouse warming is ~33 K (288 vs 255).
        assert!(
            (m.greenhouse_warming() - 33.0).abs() < 3.0,
            "ΔT_gh = {}",
            m.greenhouse_warming()
        );
    }
}
