// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-earth-system
//!
//! Reduced-order quantitative Earth-system baselines for Symthaea: radiative
//! equilibrium, CO₂ forcing, ice-albedo feedback, transient thermal response,
//! central-TCRE arithmetic, a reversible mass-conserving carbon-cycle oracle,
//! calibration, deterministic ensembles, exact carbon and thermal-mode oracles,
//! event-aligned and periodic forcing, a conservative equal-area latitudinal EBM,
//! guarded integration, critical-slowing diagnostics, exact piecewise mitigation
//! pathways, temperature-dependent zonal albedo, multi-timescale sea-level
//! response, a conserved land-water bucket, exact soil-carbon and nutrient turnover,
//! finite-interval productivity accounting, and dependency-neutral climate,
//! hydrology, nutrient, productivity, and biogeochemical driver export.
//!
//! Version 0.1 remains deliberately reduced-order. It is not a resolved
//! atmosphere-ocean model, a calibrated multi-reservoir carbon-cycle emulator,
//! a resolved land-surface or groundwater model, calibrated soil or nutrient biogeochemistry,
//! a general circulation model, an assessed sea-level projection, or a complete
//! ecological forecast. Its role is
//! to provide small, inspectable reference models and coupling experiments.

pub mod calibration;
pub mod carbon;
pub mod carbon_cycle;
pub mod driver;
pub mod emissions_schedule;
pub mod energy_balance;
pub mod ensemble;
pub mod error;
pub mod forcing;
pub mod hydrology;
pub mod ice_albedo;
pub mod latitude;
pub mod latitude_ice;
pub mod nutrient;
pub mod productivity;
pub mod schedule;
pub mod sea_level;
pub mod soil_carbon;
pub mod thermal_modes;
pub mod three_box_carbon;
pub mod transient;

pub use calibration::{
    EffectiveEmissivityCalibration, ObservationErrorSummary, OneBoxCalibration,
    calibrate_effective_olr_emissivity, one_box_constant_forcing_error,
};
pub use carbon::{
    BudgetHeadroomRange, DEFAULT_TCRE_ESTIMATE, TcreEstimate, WarmingRange,
    central_tcre_budget_headroom_carbon, central_tcre_budget_headroom_co2, remaining_carbon_budget,
    remaining_co2_budget, warming_from_cumulative_carbon, warming_from_cumulative_co2,
};
pub use carbon_cycle::{
    CarbonClimateModel, CarbonClimateSample, CarbonClimateState, CarbonSample, CarbonState,
    EmissionsProtocol, TwoBoxCarbonCycle,
};
pub use driver::{
    HydrologyDriverSample, LatitudeBandDriverSample, NutrientDriverSample,
    ProductivityDriverSample, SoilCarbonDriverSample, TemperatureDriverSample, hydrology_drivers,
    latitude_temperature_drivers, nutrient_drivers, one_box_temperature_drivers,
    productivity_driver, soil_carbon_drivers, two_box_surface_temperature_drivers,
};
pub use emissions_schedule::{
    EmissionsStage, MAX_EMISSIONS_STAGES, PiecewiseConstantEmissions, ScheduledCarbonSample,
};
pub use ensemble::{
    EnsembleSummary, MAX_ENSEMBLE_MEMBERS, OneBoxEnsembleCase, OneBoxEnsembleOutcome,
    run_one_box_ensemble, summarize_horizon_warming,
};
pub use error::ModelError;
pub use forcing::ForcingProtocol;
pub use hydrology::{HydrologyBucket, HydrologySample, HydrologyState};
pub use ice_albedo::{
    ClimateLinearStability, ClimateRecoveryDiagnostic, Equilibrium, EquilibriumSlice,
    IceAlbedoModel, SaddleNode,
};
pub use latitude::{
    LatitudeSample, LatitudinalEnergyBalanceModel, MAX_LATITUDE_BANDS, MAX_LATITUDE_STEPS,
};
pub use latitude_ice::{LatitudeIceSample, LatitudinalIceAlbedoModel, TemperatureDependentAlbedo};
pub use nutrient::{NutrientSample, NutrientState, TwoPoolNutrientCycle};
pub use productivity::{EcosystemProductivityModel, ProductivityLedger, ProductivityLimitation};
pub use schedule::{IntegrationInterval, MAX_SCHEDULE_INTERVALS, event_aligned_intervals};
pub use sea_level::{SeaLevelComponent, SeaLevelResponseModel, SeaLevelSample, SeaLevelState};
pub use soil_carbon::{SoilCarbonSample, SoilCarbonState, TwoPoolSoilCarbon};
pub use thermal_modes::{TwoBoxConvergenceReport, TwoBoxModes};
pub use three_box_carbon::{
    ThreeBoxCarbonCycle, ThreeBoxCarbonSample, ThreeBoxCarbonState, ThreeBoxEquilibriumFractions,
};
pub use transient::{
    MAX_TRAJECTORY_STEPS, OneBoxClimateModel, OneBoxConvergenceReport, OneBoxSample,
    SECONDS_PER_YEAR, SimulationGrid, TwoBoxClimateModel, TwoBoxSample, TwoBoxState,
};

pub use energy_balance::{
    EnergyBalanceModel, co2_radiative_forcing, co2_radiative_forcing_myhre1998,
    effective_emissivity_surface_temperature, effective_temperature, equilibrium_warming,
    grey_atmosphere_surface_temperature, stefan_boltzmann_temperature, try_blackbody_flux,
    try_co2_radiative_forcing_myhre1998, try_effective_emissivity_surface_temperature,
    try_effective_temperature, try_stefan_boltzmann_temperature,
};

#[cfg(test)]
mod integration_tests {
    use super::energy_balance::*;

    #[test]
    fn venus_high_albedo_lowers_effective_temperature() {
        let venus_eff = effective_temperature(2601.0, 0.77);
        let earth_eff = effective_temperature(SOLAR_CONSTANT_EARTH, EARTH_ALBEDO);
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
