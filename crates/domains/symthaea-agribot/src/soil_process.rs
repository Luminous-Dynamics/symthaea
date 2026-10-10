// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! First-principles mass and sensible-heat accounting for regenerative soil systems.
//!
//! This module intentionally separates identities derived from conservation laws
//! from empirical process parameters. It is not a pyrolysis kinetics model, soil
//! chemistry solver, microbial ecology model, or agronomic recommendation engine.
//! Inputs must be tied to source IDs / laboratory records by the calling application.
//! No empirical parameter is assigned a default value.

use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

/// Validated model-input error. No invalid values are silently clamped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SoilProcessError {
    pub field: &'static str,
    pub reason: &'static str,
}

impl SoilProcessError {
    fn new(field: &'static str, reason: &'static str) -> Self {
        Self { field, reason }
    }
}

impl fmt::Display for SoilProcessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.reason)
    }
}

impl Error for SoilProcessError {}

fn finite(value: f64, field: &'static str) -> Result<(), SoilProcessError> {
    if !value.is_finite() {
        return Err(SoilProcessError::new(field, "must be finite"));
    }
    Ok(())
}

fn positive(value: f64, field: &'static str) -> Result<(), SoilProcessError> {
    finite_nonnegative(value, field)?;
    if value == 0.0 {
        return Err(SoilProcessError::new(field, "must be greater than zero"));
    }
    Ok(())
}

fn finite_nonnegative(value: f64, field: &'static str) -> Result<(), SoilProcessError> {
    if !value.is_finite() {
        return Err(SoilProcessError::new(field, "must be finite"));
    }
    if value < 0.0 {
        return Err(SoilProcessError::new(field, "cannot be negative"));
    }
    Ok(())
}

fn fraction(value: f64, field: &'static str, allow_zero: bool) -> Result<(), SoilProcessError> {
    finite_nonnegative(value, field)?;
    if value > 1.0 || (!allow_zero && value == 0.0) {
        return Err(SoilProcessError::new(
            field,
            "must be within the allowed fraction interval",
        ));
    }
    Ok(())
}

/// Input to a lumped biomass-pyrolysis mass and thermal-duty calculation.
///
/// Fractions are unitless on explicitly named bases. Dry-feedstock carbon fraction,
/// char yield, and char carbon fraction are measured/estimated process/material
/// parameters; conservation laws do not determine them from feed mass alone.
///
/// All heat capacities are in kJ/(kg·K), latent heat in kJ/kg, temperatures in °C,
/// and masses in kg. Water boiling temperature is explicit (rather than hard-coded
/// to 100 °C) so pressure/altitude assumptions remain visible. This simplified heat
/// balance covers dry-feedstock sensible heating, feed water heating/vaporization
/// and steam superheating, and reactor sensible heating. It excludes reaction enthalpy,
/// exhaust sensible heat, heat recovery, char cooling, and detailed reactor dynamics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PyrolysisBatchInput {
    /// Wet feedstock mass entering the process, kg.
    pub wet_feedstock_kg: f64,
    /// Feed water divided by total wet feedstock mass, between 0 and 1.
    pub moisture_fraction_wet_basis: f64,
    /// Elemental carbon mass fraction of dry feedstock, between 0 and 1.
    pub feedstock_carbon_fraction_dry: f64,
    /// Char mass / dry feedstock mass. This is an empirical process parameter.
    pub char_yield_fraction_dry_basis: f64,
    /// Elemental carbon mass fraction of the dry char product.
    pub char_carbon_fraction_dry: f64,
    pub ambient_temperature_c: f64,
    pub target_temperature_c: f64,
    pub water_boiling_temperature_c: f64,
    pub dry_feedstock_heat_capacity_kj_per_kg_k: f64,
    pub liquid_water_heat_capacity_kj_per_kg_k: f64,
    pub water_latent_heat_kj_per_kg: f64,
    pub steam_heat_capacity_kj_per_kg_k: f64,
    pub reactor_mass_kg: f64,
    pub reactor_heat_capacity_kj_per_kg_k: f64,
    /// Aggregate useful-heat / supplied-heat ratio, including all losses represented
    /// by this simple model. This must be calibrated or set as an explicitly labelled
    /// scenario assumption; it is not inferred from first principles here.
    pub effective_heat_efficiency: f64,
}

/// Computed mass and heat quantities. The heat field is a lower-order duty estimate
/// for the declared terms, not a complete plant energy balance or safety assessment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PyrolysisBatchResult {
    pub dry_feedstock_kg: f64,
    pub feed_water_kg: f64,
    pub char_product_kg: f64,
    /// Dry-feedstock mass not represented by the char product; includes all other
    /// reaction products and is not a claim about gas yield, tar yield, or emissions.
    pub non_char_dry_products_residual_kg: f64,
    pub feedstock_carbon_kg: f64,
    pub char_carbon_kg: f64,
    /// Difference in feedstock and char carbon; partition among gases, liquids,
    /// deposits, or emissions is not modelled.
    pub carbon_not_in_char_kg: f64,
    pub carbon_retained_in_char_fraction: f64,
    pub dry_feedstock_sensible_heat_mj: f64,
    pub water_heating_and_vaporization_heat_mj: f64,
    pub reactor_sensible_heat_mj: f64,
    pub declared_terms_heat_duty_mj: f64,
    pub estimated_supplied_heat_mj: f64,
}

/// Compute biomass/char mass and a simplified heat duty from explicit inputs.
///
/// Formula basis:
/// - dry biomass = wet biomass × (1 − wet-basis moisture fraction)
/// - char mass = dry biomass × empirical dry-basis char yield
/// - char carbon = char mass × char carbon fraction
/// - useful heat = sum(mass × heat capacity × temperature rise) + water latent heat
/// - supplied heat estimate = useful heat / aggregate useful-heat efficiency
///
/// All calculations fail closed for invalid inputs, impossible carbon retention, or
/// overflow. The caller is responsible for preserving input provenance and uncertainty.
pub fn calculate_pyrolysis_batch(
    input: &PyrolysisBatchInput,
) -> Result<PyrolysisBatchResult, SoilProcessError> {
    finite_nonnegative(input.wet_feedstock_kg, "wet_feedstock_kg")?;
    fraction(input.moisture_fraction_wet_basis, "moisture_fraction_wet_basis", true)?;
    fraction(input.feedstock_carbon_fraction_dry, "feedstock_carbon_fraction_dry", true)?;
    fraction(input.char_yield_fraction_dry_basis, "char_yield_fraction_dry_basis", true)?;
    fraction(input.char_carbon_fraction_dry, "char_carbon_fraction_dry", true)?;
    finite(input.ambient_temperature_c, "ambient_temperature_c")?;
    finite(input.target_temperature_c, "target_temperature_c")?;
    finite(input.water_boiling_temperature_c, "water_boiling_temperature_c")?;
    positive(
        input.dry_feedstock_heat_capacity_kj_per_kg_k,
        "dry_feedstock_heat_capacity_kj_per_kg_k",
    )?;
    positive(
        input.liquid_water_heat_capacity_kj_per_kg_k,
        "liquid_water_heat_capacity_kj_per_kg_k",
    )?;
    positive(input.water_latent_heat_kj_per_kg, "water_latent_heat_kj_per_kg")?;
    positive(
        input.steam_heat_capacity_kj_per_kg_k,
        "steam_heat_capacity_kj_per_kg_k",
    )?;
    finite_nonnegative(input.reactor_mass_kg, "reactor_mass_kg")?;
    positive(
        input.reactor_heat_capacity_kj_per_kg_k,
        "reactor_heat_capacity_kj_per_kg_k",
    )?;
    fraction(input.effective_heat_efficiency, "effective_heat_efficiency", false)?;

    if input.target_temperature_c <= input.ambient_temperature_c {
        return Err(SoilProcessError::new(
            "target_temperature_c",
            "must exceed ambient temperature",
        ));
    }
    if input.water_boiling_temperature_c <= input.ambient_temperature_c
        || input.target_temperature_c <= input.water_boiling_temperature_c
    {
        return Err(SoilProcessError::new(
            "water_boiling_temperature_c",
            "must lie strictly between ambient and target temperatures",
        ));
    }

    let dry_feedstock_kg =
        input.wet_feedstock_kg * (1.0 - input.moisture_fraction_wet_basis);
    let feed_water_kg = input.wet_feedstock_kg * input.moisture_fraction_wet_basis;
    let char_product_kg = dry_feedstock_kg * input.char_yield_fraction_dry_basis;
    let non_char_dry_products_residual_kg = dry_feedstock_kg - char_product_kg;
    let feedstock_carbon_kg = dry_feedstock_kg * input.feedstock_carbon_fraction_dry;
    let char_carbon_kg = char_product_kg * input.char_carbon_fraction_dry;

    if !feedstock_carbon_kg.is_finite() || !char_carbon_kg.is_finite() {
        return Err(SoilProcessError::new("carbon_balance", "calculation overflow"));
    }
    if char_carbon_kg > feedstock_carbon_kg + f64::EPSILON * feedstock_carbon_kg.max(1.0) {
        return Err(SoilProcessError::new(
            "char_carbon_fraction_dry",
            "implies more char carbon than supplied feedstock carbon",
        ));
    }
    let carbon_not_in_char_kg = (feedstock_carbon_kg - char_carbon_kg).max(0.0);
    let carbon_retained_in_char_fraction = if feedstock_carbon_kg == 0.0 {
        if char_carbon_kg > 0.0 {
            return Err(SoilProcessError::new("carbon_balance", "positive char carbon from zero feedstock carbon"));
        }
        0.0
    } else {
        char_carbon_kg / feedstock_carbon_kg
    };

    let delta_t_dry = input.target_temperature_c - input.ambient_temperature_c;
    let delta_t_water = input.water_boiling_temperature_c - input.ambient_temperature_c;
    let delta_t_steam = input.target_temperature_c - input.water_boiling_temperature_c;
    let dry_heat_kj =
        dry_feedstock_kg * input.dry_feedstock_heat_capacity_kj_per_kg_k * delta_t_dry;
    let water_heat_kj = feed_water_kg
        * (input.liquid_water_heat_capacity_kj_per_kg_k * delta_t_water
            + input.water_latent_heat_kj_per_kg
            + input.steam_heat_capacity_kj_per_kg_k * delta_t_steam);
    let reactor_heat_kj =
        input.reactor_mass_kg * input.reactor_heat_capacity_kj_per_kg_k * delta_t_dry;
    let declared_terms_heat_duty_mj = (dry_heat_kj + water_heat_kj + reactor_heat_kj) / 1000.0;
    let estimated_supplied_heat_mj =
        declared_terms_heat_duty_mj / input.effective_heat_efficiency;

    let scalars = [
        dry_feedstock_kg,
        feed_water_kg,
        char_product_kg,
        non_char_dry_products_residual_kg,
        feedstock_carbon_kg,
        char_carbon_kg,
        carbon_not_in_char_kg,
        carbon_retained_in_char_fraction,
        dry_heat_kj,
        water_heat_kj,
        reactor_heat_kj,
        declared_terms_heat_duty_mj,
        estimated_supplied_heat_mj,
    ];
    if scalars.iter().any(|value| !value.is_finite() || *value < 0.0) {
        return Err(SoilProcessError::new("result", "non-finite or negative result"));
    }

    Ok(PyrolysisBatchResult {
        dry_feedstock_kg,
        feed_water_kg,
        char_product_kg,
        non_char_dry_products_residual_kg,
        feedstock_carbon_kg,
        char_carbon_kg,
        carbon_not_in_char_kg,
        carbon_retained_in_char_fraction,
        dry_feedstock_sensible_heat_mj: dry_heat_kj / 1000.0,
        water_heating_and_vaporization_heat_mj: water_heat_kj / 1000.0,
        reactor_sensible_heat_mj: reactor_heat_kj / 1000.0,
        declared_terms_heat_duty_mj,
        estimated_supplied_heat_mj,
    })
}

/// Elemental nutrient concentrations in kg/m³ of incoming stream.
///
/// The caller must identify sampling method, analysis date, and source in the evidence
/// record associated with this calculation. Do not assume untreated wastewater or
/// untested waste is safe to recover, transport, or apply.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NutrientConcentrationsKgPerM3 {
    pub nitrogen: f64,
    pub phosphorus: f64,
    pub potassium: f64,
}

/// Per-element recovery and seasonal plant-availability fractions. These are
/// process-/agronomy-dependent parameters, not universal constants.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NutrientRecoveryFractions {
    pub nitrogen_recovered: f64,
    pub phosphorus_recovered: f64,
    pub potassium_recovered: f64,
    pub nitrogen_plant_available_in_period: f64,
    pub phosphorus_plant_available_in_period: f64,
    pub potassium_plant_available_in_period: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveredNutrientMassKg {
    pub nitrogen_in: f64,
    pub phosphorus_in: f64,
    pub potassium_in: f64,
    pub nitrogen_recovered: f64,
    pub phosphorus_recovered: f64,
    pub potassium_recovered: f64,
    pub nitrogen_plant_available_in_period: f64,
    pub phosphorus_plant_available_in_period: f64,
    pub potassium_plant_available_in_period: f64,
}

/// Calculate nutrient mass in a stream and the amount recovered/available.
///
/// This is a mass-balance identity conditional on measured concentrations and explicitly
/// supplied recovery/availability fractions. Those fractions must be measured or
/// calibrated for the specific process; this routine does not predict treatment chemistry.
pub fn calculate_recovered_nutrients(
    volume_m3: f64,
    concentrations_kg_per_m3: NutrientConcentrationsKgPerM3,
    fractions: NutrientRecoveryFractions,
) -> Result<RecoveredNutrientMassKg, SoilProcessError> {
    finite_nonnegative(volume_m3, "volume_m3")?;
    for (name, value) in [
        ("nitrogen_concentration", concentrations_kg_per_m3.nitrogen),
        ("phosphorus_concentration", concentrations_kg_per_m3.phosphorus),
        ("potassium_concentration", concentrations_kg_per_m3.potassium),
    ] {
        finite_nonnegative(value, name)?;
    }
    for (name, value) in [
        ("nitrogen_recovered", fractions.nitrogen_recovered),
        ("phosphorus_recovered", fractions.phosphorus_recovered),
        ("potassium_recovered", fractions.potassium_recovered),
        ("nitrogen_plant_available_in_period", fractions.nitrogen_plant_available_in_period),
        ("phosphorus_plant_available_in_period", fractions.phosphorus_plant_available_in_period),
        ("potassium_plant_available_in_period", fractions.potassium_plant_available_in_period),
    ] {
        fraction(value, name, true)?;
    }

    let nitrogen_in = volume_m3 * concentrations_kg_per_m3.nitrogen;
    let phosphorus_in = volume_m3 * concentrations_kg_per_m3.phosphorus;
    let potassium_in = volume_m3 * concentrations_kg_per_m3.potassium;
    let nitrogen_recovered = nitrogen_in * fractions.nitrogen_recovered;
    let phosphorus_recovered = phosphorus_in * fractions.phosphorus_recovered;
    let potassium_recovered = potassium_in * fractions.potassium_recovered;
    let nitrogen_plant_available_in_period =
        nitrogen_recovered * fractions.nitrogen_plant_available_in_period;
    let phosphorus_plant_available_in_period =
        phosphorus_recovered * fractions.phosphorus_plant_available_in_period;
    let potassium_plant_available_in_period =
        potassium_recovered * fractions.potassium_plant_available_in_period;

    let result = RecoveredNutrientMassKg {
        nitrogen_in,
        phosphorus_in,
        potassium_in,
        nitrogen_recovered,
        phosphorus_recovered,
        potassium_recovered,
        nitrogen_plant_available_in_period,
        phosphorus_plant_available_in_period,
        potassium_plant_available_in_period,
    };
    let values = [
        result.nitrogen_in,
        result.phosphorus_in,
        result.potassium_in,
        result.nitrogen_recovered,
        result.phosphorus_recovered,
        result.potassium_recovered,
        result.nitrogen_plant_available_in_period,
        result.phosphorus_plant_available_in_period,
        result.potassium_plant_available_in_period,
    ];
    if values.iter().any(|value| !value.is_finite() || *value < 0.0) {
        return Err(SoilProcessError::new("result", "non-finite or negative result"));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pyrolysis_input() -> PyrolysisBatchInput {
        PyrolysisBatchInput {
            wet_feedstock_kg: 1_000.0,
            moisture_fraction_wet_basis: 0.20,
            feedstock_carbon_fraction_dry: 0.48,
            char_yield_fraction_dry_basis: 0.30,
            char_carbon_fraction_dry: 0.75,
            ambient_temperature_c: 25.0,
            target_temperature_c: 500.0,
            water_boiling_temperature_c: 100.0,
            dry_feedstock_heat_capacity_kj_per_kg_k: 1.5,
            liquid_water_heat_capacity_kj_per_kg_k: 4.18,
            water_latent_heat_kj_per_kg: 2_257.0,
            steam_heat_capacity_kj_per_kg_k: 2.0,
            reactor_mass_kg: 500.0,
            reactor_heat_capacity_kj_per_kg_k: 0.50,
            effective_heat_efficiency: 0.65,
        }
    }

    #[test]
    fn pyrolysis_mass_and_carbon_balances_close_for_valid_scenario() {
        let result = calculate_pyrolysis_batch(&pyrolysis_input()).unwrap();
        assert!((result.dry_feedstock_kg - 800.0).abs() < 1e-10);
        assert!((result.feed_water_kg - 200.0).abs() < 1e-10);
        assert!((result.char_product_kg - 240.0).abs() < 1e-10);
        assert!((result.non_char_dry_products_residual_kg + result.char_product_kg
            - result.dry_feedstock_kg).abs() < 1e-10);
        assert!((result.feedstock_carbon_kg - 384.0).abs() < 1e-10);
        assert!((result.char_carbon_kg - 180.0).abs() < 1e-10);
        assert!((result.carbon_not_in_char_kg + result.char_carbon_kg
            - result.feedstock_carbon_kg).abs() < 1e-10);
        assert!((result.carbon_retained_in_char_fraction - (180.0 / 384.0)).abs() < 1e-12);
        assert!(result.declared_terms_heat_duty_mj > 0.0);
        assert!(result.estimated_supplied_heat_mj > result.declared_terms_heat_duty_mj);
    }

    #[test]
    fn subzero_ambient_temperature_is_supported() {
        let mut input = pyrolysis_input();
        input.ambient_temperature_c = -10.0;
        let result = calculate_pyrolysis_batch(&input).unwrap();
        assert!(result.declared_terms_heat_duty_mj > 0.0);
    }

    #[test]
    fn dry_feedstock_has_zero_water_duty() {
        let mut input = pyrolysis_input();
        input.moisture_fraction_wet_basis = 0.0;
        let result = calculate_pyrolysis_batch(&input).unwrap();
        assert_eq!(result.feed_water_kg, 0.0);
        assert_eq!(result.water_heating_and_vaporization_heat_mj, 0.0);
    }

    #[test]
    fn impossible_carbon_yield_fails_closed() {
        let mut input = pyrolysis_input();
        input.char_yield_fraction_dry_basis = 0.9;
        input.char_carbon_fraction_dry = 0.9;
        assert!(calculate_pyrolysis_batch(&input).is_err());
    }

    #[test]
    fn pyrolysis_rejects_nan_negative_fraction_and_invalid_temperatures() {
        let mut input = pyrolysis_input();
        input.wet_feedstock_kg = f64::NAN;
        assert!(calculate_pyrolysis_batch(&input).is_err());

        input = pyrolysis_input();
        input.moisture_fraction_wet_basis = -0.01;
        assert!(calculate_pyrolysis_batch(&input).is_err());

        input = pyrolysis_input();
        input.target_temperature_c = input.ambient_temperature_c;
        assert!(calculate_pyrolysis_batch(&input).is_err());

        input = pyrolysis_input();
        input.effective_heat_efficiency = 1.1;
        assert!(calculate_pyrolysis_batch(&input).is_err());
    }

    #[test]
    fn recovered_nutrients_are_element_specific_and_bounded() {
        let result = calculate_recovered_nutrients(
            100.0,
            NutrientConcentrationsKgPerM3 {
                nitrogen: 0.04,
                phosphorus: 0.01,
                potassium: 0.02,
            },
            NutrientRecoveryFractions {
                nitrogen_recovered: 0.70,
                phosphorus_recovered: 0.85,
                potassium_recovered: 0.40,
                nitrogen_plant_available_in_period: 0.60,
                phosphorus_plant_available_in_period: 0.50,
                potassium_plant_available_in_period: 0.90,
            },
        )
        .unwrap();

        assert!((result.nitrogen_in - 4.0).abs() < 1e-12);
        assert!((result.phosphorus_in - 1.0).abs() < 1e-12);
        assert!((result.potassium_in - 2.0).abs() < 1e-12);
        assert!((result.nitrogen_recovered - 2.8).abs() < 1e-12);
        assert!((result.phosphorus_recovered - 0.85).abs() < 1e-12);
        assert!((result.potassium_recovered - 0.8).abs() < 1e-12);
        assert!((result.nitrogen_plant_available_in_period - 1.68).abs() < 1e-12);
        assert!(result.nitrogen_recovered <= result.nitrogen_in);
        assert!(result.phosphorus_recovered <= result.phosphorus_in);
        assert!(result.potassium_recovered <= result.potassium_in);
    }

    #[test]
    fn nutrient_recovery_rejects_invalid_parameters_and_overflow() {
        let concentrations = NutrientConcentrationsKgPerM3 {
            nitrogen: 0.1,
            phosphorus: 0.02,
            potassium: 0.03,
        };
        let mut fractions = NutrientRecoveryFractions {
            nitrogen_recovered: 0.8,
            phosphorus_recovered: 0.8,
            potassium_recovered: 0.8,
            nitrogen_plant_available_in_period: 0.5,
            phosphorus_plant_available_in_period: 0.5,
            potassium_plant_available_in_period: 0.5,
        };
        assert!(calculate_recovered_nutrients(-1.0, concentrations, fractions).is_err());
        fractions.nitrogen_recovered = 1.1;
        assert!(calculate_recovered_nutrients(10.0, concentrations, fractions).is_err());
        fractions.nitrogen_recovered = 0.8;
        assert!(calculate_recovered_nutrients(f64::MAX, concentrations, fractions).is_err());
    }

    #[test]
    fn serde_roundtrip_preserves_explicit_assumptions() {
        let input = pyrolysis_input();
        let json = serde_json::to_string(&input).unwrap();
        let decoded: PyrolysisBatchInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, input);
    }
}
