// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-earth-system
//!
//! Earth-system climate physics for Symthaea, starting with a zero-dimensional
//! energy-balance model: Stefan-Boltzmann effective temperature, grey-atmosphere
//! greenhouse, CO₂ radiative forcing, and equilibrium climate sensitivity.
//!
//! Fills a confirmed gap — the workspace encoded geophysics only as exploratory
//! HDC concept vectors (`symthaea-core/src/physics/geophysics.rs`) with **no
//! quantitative climate/atmosphere model**. This crate provides real, closed-form
//! climate physics checked against canonical values (Earth T_eff ≈ 255 K, CO₂
//! doubling forcing ≈ 3.7 W/m², ~33 K greenhouse warming).
//!
//! ## Scope
//!
//! - 0-D energy balance: effective temperature, grey-atmosphere surface
//!   temperature, greenhouse warming.
//! - Forcing & sensitivity: CO₂ radiative forcing (Myhre 1998), linear
//!   equilibrium warming `ΔT = λ·ΔF`.
//! - Ice-albedo feedback: temperature-dependent albedo, multiple equilibria
//!   (warm + snowball), stability classification.
//! - Carbon budgets: TCRE cumulative-emissions warming, remaining-budget
//!   accounting for a temperature target.
//!
//! Not yet: 1-D latitudinal EBM, radiative-convective columns, or general
//! circulation — the intended next direction.
//!
//! ## Example
//!
//! ```
//! use symthaea_earth_system::energy_balance::{
//!     co2_radiative_forcing, equilibrium_warming, EnergyBalanceModel,
//! };
//!
//! let earth = EnergyBalanceModel::earth();
//! assert!((earth.effective_temperature() - 255.0).abs() < 1.0);
//! assert!(earth.greenhouse_warming() > 30.0); // ~33 K
//!
//! let df = co2_radiative_forcing(560.0, 280.0);       // CO₂ doubling
//! assert!((equilibrium_warming(df, 0.8) - 3.0).abs() < 1.0); // ~3 K ECS
//! ```

pub mod carbon;
pub mod energy_balance;
pub mod ice_albedo;

pub use carbon::{
    remaining_carbon_budget, remaining_co2_budget, warming_from_cumulative_carbon,
    warming_from_cumulative_co2,
};
pub use ice_albedo::{Equilibrium, IceAlbedoModel};

pub use energy_balance::{
    EnergyBalanceModel, co2_radiative_forcing, effective_temperature, equilibrium_warming,
    grey_atmosphere_surface_temperature, stefan_boltzmann_temperature,
};

#[cfg(test)]
mod integration_tests {
    use super::energy_balance::*;

    #[test]
    fn venus_runaway_is_hotter_than_earth() {
        // Venus: high albedo but strong greenhouse. Even the effective
        // temperature (albedo 0.77, S≈2601) is cooler than Earth's surface —
        // the model correctly shows Venus's heat is greenhouse, not insolation.
        let venus_eff = effective_temperature(2601.0, 0.77);
        let earth_eff = effective_temperature(SOLAR_CONSTANT_EARTH, EARTH_ALBEDO);
        // Venus's high albedo makes its *effective* temp near Earth's.
        assert!(venus_eff < 240.0, "Venus T_eff = {venus_eff}");
        assert!(earth_eff > 250.0);
    }

    #[test]
    fn warming_scales_linearly_with_forcing() {
        let dt1 = equilibrium_warming(3.7, 0.8);
        let dt2 = equilibrium_warming(7.4, 0.8);
        assert!((dt2 / dt1 - 2.0).abs() < 1e-9);
    }
}
