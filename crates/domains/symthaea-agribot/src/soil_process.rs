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

/// Provenance class for one input group. A literature value or estimate remains
/// distinct from a direct measurement; a scenario is never an observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    Measured,
    Literature,
    Estimated,
    Scenario,
}

/// Reference to an immutable evidence record or versioned parameter source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceRef {
    pub evidence_id: String,
    pub kind: EvidenceKind,
}

impl EvidenceRef {
    fn validate(&self, field: &'static str) -> Result<(), SoilProcessError> {
        if self.evidence_id.trim().is_empty() {
            return Err(SoilProcessError::new(field, "evidence ID cannot be empty"));
        }
        Ok(())
    }
}

/// Separate provenance for feedstock observations, empirical process parameters,
/// thermophysical values, reactor design data, and the exact input snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PyrolysisEvidence {
    pub feedstock: EvidenceRef,
    pub process_parameters: EvidenceRef,
    pub thermophysical_properties: EvidenceRef,
    pub reactor_design: EvidenceRef,
    pub input_snapshot_id: String,
}

impl PyrolysisEvidence {
    fn validate(&self) -> Result<(), SoilProcessError> {
        self.feedstock.validate("evidence.feedstock")?;
        self.process_parameters.validate("evidence.process_parameters")?;
        self.thermophysical_properties
            .validate("evidence.thermophysical_properties")?;
        self.reactor_design.validate("evidence.reactor_design")?;
        if self.input_snapshot_id.trim().is_empty() {
            return Err(SoilProcessError::new(
                "evidence.input_snapshot_id",
                "input snapshot ID cannot be empty",
            ));
        }
        Ok(())
    }
}

/// Provenance for stream composition, recovery calibration, and period-specific
/// plant-availability fractions. These are separate evidence sources by design.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NutrientRecoveryEvidence {
    pub influent_composition: EvidenceRef,
    pub recovery_parameters: EvidenceRef,
    pub plant_availability_parameters: EvidenceRef,
    pub input_snapshot_id: String,
}

impl NutrientRecoveryEvidence {
    fn validate(&self) -> Result<(), SoilProcessError> {
        self.influent_composition
            .validate("evidence.influent_composition")?;
        self.recovery_parameters
            .validate("evidence.recovery_parameters")?;
        self.plant_availability_parameters
            .validate("evidence.plant_availability_parameters")?;
        if self.input_snapshot_id.trim().is_empty() {
            return Err(SoilProcessError::new(
                "evidence.input_snapshot_id",
                "input snapshot ID cannot be empty",
            ));
        }
        Ok(())
    }
}

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
    /// Provenance for every empirical/property input group and a stable input snapshot.
    pub evidence: PyrolysisEvidence,
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
    pub evidence: PyrolysisEvidence,
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
    input.evidence.validate()?;
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
            return Err(SoilProcessError::new(
                "carbon_balance",
                "positive char carbon from zero feedstock carbon",
            ));
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
        evidence: input.evidence.clone(),
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
    pub evidence: NutrientRecoveryEvidence,
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
    evidence: &NutrientRecoveryEvidence,
) -> Result<RecoveredNutrientMassKg, SoilProcessError> {
    evidence.validate()?;
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
        evidence: evidence.clone(),
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

/// A climate-flow category must be explicit: emissions, removals, and avoided
/// emissions are reported separately and never silently collapsed into a single score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClimateFlowKind {
    Emission,
    Removal,
    AvoidedEmission,
}

/// Positive quantity in kg CO2-equivalent. The flow kind controls whether it contributes
/// to emissions or credits. A flow is only as reliable as its explicit evidence record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClimateFlow {
    pub flow_id: String,
    pub kind: ClimateFlowKind,
    pub kg_co2e: f64,
    pub evidence: EvidenceRef,
}

/// Whether the system has a documented basis for crediting biogenic carbon stored in char.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CharStorageEligibility {
    /// Sustainable biogenic sourcing eligibility has been verified by the caller.
    VerifiedEligible,
    /// Evidence establishes that durable biogenic storage must not be credited.
    VerifiedIneligible,
    /// Eligibility is not established. A complete net result must remain unknown.
    Unknown,
}

/// Completeness is relative to one declared life-cycle boundary, not a claim that all
/// possible impacts in the universe have been measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClimateInventoryStatus {
    CompleteForDeclaredBoundary,
    Partial,
}

/// Explicit char-storage credit assumptions for one time horizon.
///
/// A durable-storage fraction is empirical/modelled, never derived from the carbon mass
/// balance alone. For VerifiedEligible, eligibility evidence must be a measured/verified
/// record. Persistence can be measured, literature-derived or scenario input, and should
/// remain labelled accordingly by the caller.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CharStorageAccounting {
    pub horizon_years: u32,
    pub eligibility: CharStorageEligibility,
    pub eligibility_evidence: Option<EvidenceRef>,
    pub durable_fraction_at_horizon: Option<f64>,
    pub persistence_evidence: Option<EvidenceRef>,
}

impl CharStorageAccounting {
    fn validate(&self) -> Result<(), SoilProcessError> {
        if self.horizon_years == 0 {
            return Err(SoilProcessError::new(
                "char_storage.horizon_years",
                "must be greater than zero",
            ));
        }

        match self.eligibility {
            CharStorageEligibility::VerifiedEligible => {
                let eligibility = self.eligibility_evidence.as_ref().ok_or_else(|| {
                    SoilProcessError::new(
                        "char_storage.eligibility_evidence",
                        "verified eligibility requires an evidence record",
                    )
                })?;
                eligibility.validate("char_storage.eligibility_evidence")?;
                if eligibility.kind != EvidenceKind::Measured {
                    return Err(SoilProcessError::new(
                        "char_storage.eligibility_evidence",
                        "verified eligibility requires measured/verified evidence, not a scenario",
                    ));
                }
                let fraction = self.durable_fraction_at_horizon.ok_or_else(|| {
                    SoilProcessError::new(
                        "char_storage.durable_fraction_at_horizon",
                        "verified storage requires an explicit durability fraction",
                    )
                })?;
                fraction_value(fraction, "char_storage.durable_fraction_at_horizon")?;
                let persistence = self.persistence_evidence.as_ref().ok_or_else(|| {
                    SoilProcessError::new(
                        "char_storage.persistence_evidence",
                        "durability fraction requires a source record",
                    )
                })?;
                persistence.validate("char_storage.persistence_evidence")?;
            }
            CharStorageEligibility::VerifiedIneligible => {
                let eligibility = self.eligibility_evidence.as_ref().ok_or_else(|| {
                    SoilProcessError::new(
                        "char_storage.eligibility_evidence",
                        "verified ineligibility requires an evidence record",
                    )
                })?;
                eligibility.validate("char_storage.eligibility_evidence")?;
                if self.durable_fraction_at_horizon.is_some()
                    || self.persistence_evidence.is_some()
                {
                    return Err(SoilProcessError::new(
                        "char_storage",
                        "ineligible storage cannot also supply a credited durability fraction",
                    ));
                }
            }
            CharStorageEligibility::Unknown => {
                if self.eligibility_evidence.is_some()
                    || self.durable_fraction_at_horizon.is_some()
                    || self.persistence_evidence.is_some()
                {
                    return Err(SoilProcessError::new(
                        "char_storage",
                        "unknown eligibility must not smuggle in an asserted storage credit",
                    ));
                }
            }
        }
        Ok(())
    }
}

fn fraction_value(value: f64, field: &'static str) -> Result<(), SoilProcessError> {
    finite_nonnegative(value, field)?;
    if value > 1.0 {
        return Err(SoilProcessError::new(field, "must be within [0, 1]"));
    }
    Ok(())
}

/// Climate inventory for one explicitly identified biochar batch and declared boundary.
///
/// Examples of separate flows include reactor energy, methane/N2O and other measured
/// process emissions, transport, application, measured removals, and substantiated
/// avoided emissions such as a documented displaced product or waste-fate baseline.
/// The function does not invent counterfactual credits or infer gas emissions from the
/// difference between feedstock carbon and char carbon.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BiocharClimateInput {
    pub boundary_id: String,
    pub inventory_status: ClimateInventoryStatus,
    pub inventory_evidence: EvidenceRef,
    pub flows: Vec<ClimateFlow>,
    pub char_storage: CharStorageAccounting,
}

/// Auditable climate subtotal for a batch. Positive net means net emissions; negative
/// net means net removals/credits within the declared inventory boundary only.
/// net_kg_co2e is absent unless both the inventory and storage eligibility are complete.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BiocharClimateAssessment {
    pub boundary_id: String,
    pub horizon_years: u32,
    pub inventory_status: ClimateInventoryStatus,
    pub gross_emissions_kg_co2e: f64,
    pub other_removals_kg_co2e: f64,
    pub avoided_emissions_kg_co2e: f64,
    /// None means the model is not justified in applying a durable-char-storage credit.
    pub durable_char_storage_kg_co2e: Option<f64>,
    pub net_kg_co2e: Option<f64>,
    pub flows: Vec<ClimateFlow>,
    pub char_storage: CharStorageAccounting,
    pub inventory_evidence: EvidenceRef,
    pub process_evidence: PyrolysisEvidence,
    pub scope_note: String,
}

/// Assess a biochar batch's climate terms without inventing missing lifecycle credits.
///
/// Carbon in char is converted to CO2-equivalent with the molecular-mass ratio 44/12.
/// Only the explicitly supplied durability fraction at the specified horizon is credited.
/// The result is not an ISO-conformant LCA, carbon-credit issuance, or proof of net removal.
pub fn assess_biochar_climate(
    process: &PyrolysisBatchResult,
    input: &BiocharClimateInput,
) -> Result<BiocharClimateAssessment, SoilProcessError> {
    if input.boundary_id.trim().is_empty() {
        return Err(SoilProcessError::new("boundary_id", "cannot be empty"));
    }
    input.inventory_evidence.validate("inventory_evidence")?;
    input.char_storage.validate()?;

    finite_nonnegative(process.char_product_kg, "process.char_product_kg")?;
    finite_nonnegative(process.char_carbon_kg, "process.char_carbon_kg")?;
    if process.char_carbon_kg > process.char_product_kg {
        return Err(SoilProcessError::new(
            "process.char_carbon_kg",
            "must be no greater than total char mass",
        ));
    }
    process.evidence.feedstock.validate("process.evidence.feedstock")?;
    process
        .evidence
        .process_parameters
        .validate("process.evidence.process_parameters")?;
    process
        .evidence
        .thermophysical_properties
        .validate("process.evidence.thermophysical_properties")?;
    process
        .evidence
        .reactor_design
        .validate("process.evidence.reactor_design")?;
    if process.evidence.input_snapshot_id.trim().is_empty() {
        return Err(SoilProcessError::new(
            "process.evidence.input_snapshot_id",
            "cannot be empty",
        ));
    }

    let mut ids = std::collections::HashSet::new();
    let mut emissions = 0.0_f64;
    let mut removals = 0.0_f64;
    let mut avoided = 0.0_f64;

    for flow in &input.flows {
        if flow.flow_id.trim().is_empty() || !ids.insert(flow.flow_id.as_str()) {
            return Err(SoilProcessError::new(
                "flows.flow_id",
                "flow IDs must be non-empty and unique",
            ));
        }
        finite_nonnegative(flow.kg_co2e, "flows.kg_co2e")?;
        flow.evidence.validate("flows.evidence")?;
        let subtotal = match flow.kind {
            ClimateFlowKind::Emission => &mut emissions,
            ClimateFlowKind::Removal => &mut removals,
            ClimateFlowKind::AvoidedEmission => &mut avoided,
        };
        *subtotal += flow.kg_co2e;
        if !subtotal.is_finite() {
            return Err(SoilProcessError::new("flows", "subtotal overflow"));
        }
    }

    let durable_char_storage_kg_co2e = match input.char_storage.eligibility {
        CharStorageEligibility::VerifiedEligible => {
            let fraction = input
                .char_storage
                .durable_fraction_at_horizon
                .ok_or_else(|| {
                    SoilProcessError::new(
                        "char_storage.durable_fraction_at_horizon",
                        "verified storage requires an explicit durability fraction",
                    )
                })?;
            Some(process.char_carbon_kg * fraction * (44.0 / 12.0))
        }
        CharStorageEligibility::VerifiedIneligible => Some(0.0),
        CharStorageEligibility::Unknown => None,
    };

    let net_kg_co2e =
        if input.inventory_status == ClimateInventoryStatus::CompleteForDeclaredBoundary {
        durable_char_storage_kg_co2e.map(|storage| emissions - removals - avoided - storage)
    } else {
        None
    };

    let totals = [
        emissions,
        removals,
        avoided,
        durable_char_storage_kg_co2e.unwrap_or(0.0),
    ];
    if totals.iter().any(|value| !value.is_finite() || *value < 0.0) {
        return Err(SoilProcessError::new("assessment", "invalid or non-finite subtotal"));
    }
    if net_kg_co2e.is_some_and(|value| !value.is_finite()) {
        return Err(SoilProcessError::new("net_kg_co2e", "net balance overflow"));
    }

    Ok(BiocharClimateAssessment {
        boundary_id: input.boundary_id.clone(),
        horizon_years: input.char_storage.horizon_years,
        inventory_status: input.inventory_status,
        gross_emissions_kg_co2e: emissions,
        other_removals_kg_co2e: removals,
        avoided_emissions_kg_co2e: avoided,
        durable_char_storage_kg_co2e,
        net_kg_co2e,
        flows: input.flows.clone(),
        char_storage: input.char_storage.clone(),
        inventory_evidence: input.inventory_evidence.clone(),
        process_evidence: process.evidence.clone(),
        scope_note: "scenario/batch accounting within declared boundary; not an ISO-conformant LCA, carbon-credit issuance, or proof of net removal"
            .into(),
    })
}



/// A measured output stream for an explicit physical process boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservedMassStream {
    pub stream_id: String,
    /// Actual mass crossing the boundary, kg. Do not mix dry-basis and wet-basis values.
    pub mass_kg: f64,
    /// Must reference measurement evidence for a physical batch.
    pub evidence: EvidenceRef,
}

/// Physical-batch mass accounting based on independently measured inlet and outlet streams.
/// This deliberately differs from the algebraically inferred residual in PyrolysisBatchResult.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservedMassBalanceInput {
    pub boundary_id: String,
    /// Sum of all measured mass entering the declared boundary, kg. Include purge gas,
    /// added water/agents, and other inlets when they cross the boundary.
    pub total_input_mass_kg: f64,
    pub input_evidence: EvidenceRef,
    /// Enumerate every material output crossing the same boundary, including captured
    /// solids/liquids and quantified gas streams. Unmeasured streams must not be omitted
    /// and then treated as zero.
    pub output_streams: Vec<ObservedMassStream>,
    /// Relative closure tolerance, justified from the measurement/instrument procedure.
    pub closure_tolerance_fraction: f64,
    pub tolerance_evidence: EvidenceRef,
    pub input_snapshot_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservedMassClosureStatus {
    WithinTolerance,
    OutsideTolerance,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservedMassBalanceAssessment {
    pub boundary_id: String,
    pub input_snapshot_id: String,
    pub total_input_mass_kg: f64,
    pub total_observed_output_mass_kg: f64,
    /// Positive: inlet mass exceeds observed outlet mass. Negative: outlets exceed inlets.
    pub signed_residual_kg: f64,
    pub absolute_closure_error_fraction: f64,
    pub permitted_closure_tolerance_fraction: f64,
    pub status: ObservedMassClosureStatus,
    pub input_evidence: EvidenceRef,
    pub output_streams: Vec<ObservedMassStream>,
    pub tolerance_evidence: EvidenceRef,
    /// Closure is a measurement-quality screen, not product-quality or process-safety approval.
    pub scope_note: String,
}

/// Compare independently measured mass streams over an explicit boundary.
/// Every physical mass measurement must carry EvidenceKind::Measured; a scenario or
/// literature estimate is not accepted as evidence that a physical batch actually closed.
/// The reported closure tolerance must be tied to an evidence record, not chosen to force a pass.
pub fn assess_observed_mass_balance(
    input: &ObservedMassBalanceInput,
) -> Result<ObservedMassBalanceAssessment, SoilProcessError> {
    if input.boundary_id.trim().is_empty() {
        return Err(SoilProcessError::new("boundary_id", "cannot be empty"));
    }
    if input.input_snapshot_id.trim().is_empty() {
        return Err(SoilProcessError::new("input_snapshot_id", "cannot be empty"));
    }
    positive(input.total_input_mass_kg, "total_input_mass_kg")?;
    fraction(
        input.closure_tolerance_fraction,
        "closure_tolerance_fraction",
        true,
    )?;
    if input.closure_tolerance_fraction >= 1.0 {
        return Err(SoilProcessError::new(
            "closure_tolerance_fraction",
            "must be strictly less than 1.0",
        ));
    }
    input.input_evidence.validate("input_evidence")?;
    if input.input_evidence.kind != EvidenceKind::Measured {
        return Err(SoilProcessError::new(
            "input_evidence.kind",
            "physical-batch inlet mass requires measured evidence",
        ));
    }
    input.tolerance_evidence.validate("tolerance_evidence")?;
    if input.tolerance_evidence.kind == EvidenceKind::Scenario {
        return Err(SoilProcessError::new(
            "tolerance_evidence.kind",
            "closure tolerance cannot be justified only by a scenario",
        ));
    }
    if input.output_streams.is_empty() {
        return Err(SoilProcessError::new(
            "output_streams",
            "at least one independently measured output stream is required",
        ));
    }

    let mut ids = std::collections::HashSet::new();
    let mut total_output = 0.0_f64;
    for stream in &input.output_streams {
        if stream.stream_id.trim().is_empty() || !ids.insert(stream.stream_id.as_str()) {
            return Err(SoilProcessError::new(
                "output_streams.stream_id",
                "stream IDs must be non-empty and unique",
            ));
        }
        finite_nonnegative(stream.mass_kg, "output_streams.mass_kg")?;
        stream.evidence.validate("output_streams.evidence")?;
        if stream.evidence.kind != EvidenceKind::Measured {
            return Err(SoilProcessError::new(
                "output_streams.evidence.kind",
                "physical-batch outlet mass requires measured evidence",
            ));
        }
        total_output += stream.mass_kg;
        if !total_output.is_finite() {
            return Err(SoilProcessError::new(
                "output_streams",
                "observed outlet mass total overflow",
            ));
        }
    }

    let signed_residual = input.total_input_mass_kg - total_output;
    if !signed_residual.is_finite() {
        return Err(SoilProcessError::new(
            "signed_residual_kg",
            "mass residual overflow",
        ));
    }
    let absolute_closure_error_fraction = signed_residual.abs() / input.total_input_mass_kg;
    if !absolute_closure_error_fraction.is_finite() {
        return Err(SoilProcessError::new(
            "absolute_closure_error_fraction",
            "relative mass residual overflow",
        ));
    }
    let status = if absolute_closure_error_fraction <= input.closure_tolerance_fraction {
        ObservedMassClosureStatus::WithinTolerance
    } else {
        ObservedMassClosureStatus::OutsideTolerance
    };

    Ok(ObservedMassBalanceAssessment {
        boundary_id: input.boundary_id.clone(),
        input_snapshot_id: input.input_snapshot_id.clone(),
        total_input_mass_kg: input.total_input_mass_kg,
        total_observed_output_mass_kg: total_output,
        signed_residual_kg: signed_residual,
        absolute_closure_error_fraction,
        permitted_closure_tolerance_fraction: input.closure_tolerance_fraction,
        status,
        input_evidence: input.input_evidence.clone(),
        output_streams: input.output_streams.clone(),
        tolerance_evidence: input.tolerance_evidence.clone(),
        scope_note: "observed mass closure only; not product-quality, emissions, safety, nutrient-availability, or agronomic approval"
            .into(),
    })
}

#[cfg(test)]
mod observed_mass_balance_tests {
    use super::*;

    fn measured(id: &str) -> EvidenceRef {
        EvidenceRef {
            evidence_id: id.into(),
            kind: EvidenceKind::Measured,
        }
    }

    fn input() -> ObservedMassBalanceInput {
        ObservedMassBalanceInput {
            boundary_id: "pilot-reactor-batch-001".into(),
            total_input_mass_kg: 1_000.0,
            input_evidence: measured("inlet-weigh-log-001"),
            output_streams: vec![
                ObservedMassStream {
                    stream_id: "dry-char".into(),
                    mass_kg: 240.0,
                    evidence: measured("char-scale-001"),
                },
                ObservedMassStream {
                    stream_id: "condensate".into(),
                    mass_kg: 220.0,
                    evidence: measured("condensate-scale-001"),
                },
                ObservedMassStream {
                    stream_id: "gas".into(),
                    mass_kg: 520.0,
                    evidence: measured("gas-flow-composition-001"),
                },
                ObservedMassStream {
                    stream_id: "captured-fines".into(),
                    mass_kg: 20.0,
                    evidence: measured("fines-scale-001"),
                },
            ],
            closure_tolerance_fraction: 0.02,
            tolerance_evidence: EvidenceRef {
                evidence_id: "measurement-uncertainty-protocol-001".into(),
                kind: EvidenceKind::Literature,
            },
            input_snapshot_id: "pilot-batch-snapshot-001".into(),
        }
    }

    #[test]
    fn observed_streams_close_only_from_measured_outputs() {
        let result = assess_observed_mass_balance(&input()).unwrap();
        assert_eq!(result.status, ObservedMassClosureStatus::WithinTolerance);
        assert_eq!(result.total_observed_output_mass_kg, 1_000.0);
        assert_eq!(result.signed_residual_kg, 0.0);
        assert_eq!(result.absolute_closure_error_fraction, 0.0);
        assert_eq!(result.output_streams.len(), 4);
    }

    #[test]
    fn stream_gap_is_reported_not_algebraically_filled() {
        let mut measurement = input();
        measurement.output_streams[2].mass_kg = 480.0;
        let result = assess_observed_mass_balance(&measurement).unwrap();
        assert_eq!(result.signed_residual_kg, 40.0);
        assert_eq!(result.absolute_closure_error_fraction, 0.04);
        assert_eq!(result.status, ObservedMassClosureStatus::OutsideTolerance);
    }

    #[test]
    fn scenario_and_duplicate_output_streams_are_rejected() {
        let mut measurement = input();
        measurement.output_streams[0].evidence.kind = EvidenceKind::Scenario;
        assert!(assess_observed_mass_balance(&measurement).is_err());

        measurement = input();
        measurement.output_streams[1].stream_id = measurement.output_streams[0].stream_id.clone();
        assert!(assess_observed_mass_balance(&measurement).is_err());
    }

    #[test]
    fn invalid_tolerance_missing_streams_and_overflow_fail_closed() {
        let mut measurement = input();
        measurement.closure_tolerance_fraction = f64::NAN;
        assert!(assess_observed_mass_balance(&measurement).is_err());

        measurement = input();
        measurement.closure_tolerance_fraction = 1.0;
        assert!(assess_observed_mass_balance(&measurement).is_err());

        measurement = input();
        measurement.tolerance_evidence.kind = EvidenceKind::Scenario;
        assert!(assess_observed_mass_balance(&measurement).is_err());

        measurement = input();
        measurement.output_streams.clear();
        assert!(assess_observed_mass_balance(&measurement).is_err());

        measurement = input();
        measurement.output_streams[0].mass_kg = f64::MAX;
        measurement.output_streams[1].mass_kg = f64::MAX;
        assert!(assess_observed_mass_balance(&measurement).is_err());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pyrolysis_input() -> PyrolysisBatchInput {
        PyrolysisBatchInput {
            evidence: PyrolysisEvidence {
                feedstock: EvidenceRef {
                    evidence_id: "scenario-feedstock-001".into(),
                    kind: EvidenceKind::Scenario,
                },
                process_parameters: EvidenceRef {
                    evidence_id: "scenario-pyrolysis-yield-001".into(),
                    kind: EvidenceKind::Scenario,
                },
                thermophysical_properties: EvidenceRef {
                    evidence_id: "reference-properties-001".into(),
                    kind: EvidenceKind::Literature,
                },
                reactor_design: EvidenceRef {
                    evidence_id: "scenario-reactor-001".into(),
                    kind: EvidenceKind::Scenario,
                },
                input_snapshot_id: "scenario-run-001".into(),
            },
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
    fn missing_provenance_is_rejected_and_scenario_status_is_retained() {
        let mut input = pyrolysis_input();
        input.evidence.process_parameters.evidence_id.clear();
        assert!(calculate_pyrolysis_batch(&input).is_err());

        input = pyrolysis_input();
        let result = calculate_pyrolysis_batch(&input).unwrap();
        assert_eq!(result.evidence, input.evidence);
        assert_eq!(result.evidence.process_parameters.kind, EvidenceKind::Scenario);
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
        let evidence = NutrientRecoveryEvidence {
            influent_composition: EvidenceRef {
                evidence_id: "scenario-influent-001".into(),
                kind: EvidenceKind::Scenario,
            },
            recovery_parameters: EvidenceRef {
                evidence_id: "scenario-recovery-001".into(),
                kind: EvidenceKind::Scenario,
            },
            plant_availability_parameters: EvidenceRef {
                evidence_id: "scenario-availability-001".into(),
                kind: EvidenceKind::Scenario,
            },
            input_snapshot_id: "scenario-nutrient-run-001".into(),
        };
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
            &evidence,
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
        let evidence = NutrientRecoveryEvidence {
            influent_composition: EvidenceRef {
                evidence_id: "scenario-influent-001".into(),
                kind: EvidenceKind::Scenario,
            },
            recovery_parameters: EvidenceRef {
                evidence_id: "scenario-recovery-001".into(),
                kind: EvidenceKind::Scenario,
            },
            plant_availability_parameters: EvidenceRef {
                evidence_id: "scenario-availability-001".into(),
                kind: EvidenceKind::Scenario,
            },
            input_snapshot_id: "scenario-nutrient-run-001".into(),
        };
        let mut fractions = NutrientRecoveryFractions {
            nitrogen_recovered: 0.8,
            phosphorus_recovered: 0.8,
            potassium_recovered: 0.8,
            nitrogen_plant_available_in_period: 0.5,
            phosphorus_plant_available_in_period: 0.5,
            potassium_plant_available_in_period: 0.5,
        };
        assert!(calculate_recovered_nutrients(-1.0, concentrations, fractions, &evidence).is_err());
        fractions.nitrogen_recovered = 1.1;
        assert!(calculate_recovered_nutrients(10.0, concentrations, fractions, &evidence).is_err());
        fractions.nitrogen_recovered = 0.8;
        assert!(calculate_recovered_nutrients(
            f64::MAX,
            concentrations,
            fractions,
            &evidence
        )
        .is_err());
    }

    fn climate_evidence(id: &str, kind: EvidenceKind) -> EvidenceRef {
        EvidenceRef {
            evidence_id: id.into(),
            kind,
        }
    }

    fn climate_input(
        status: ClimateInventoryStatus,
        eligibility: CharStorageEligibility,
    ) -> BiocharClimateInput {
        let eligible = eligibility == CharStorageEligibility::VerifiedEligible;
        BiocharClimateInput {
            boundary_id: "batch-boundary-v1".into(),
            inventory_status: status,
            inventory_evidence: climate_evidence("inventory-scenario-v1", EvidenceKind::Scenario),
            flows: vec![
                ClimateFlow {
                    flow_id: "process-emissions".into(),
                    kind: ClimateFlowKind::Emission,
                    kg_co2e: 100.0,
                    evidence: climate_evidence("process-emissions-v1", EvidenceKind::Scenario),
                },
                ClimateFlow {
                    flow_id: "other-removal".into(),
                    kind: ClimateFlowKind::Removal,
                    kg_co2e: 5.0,
                    evidence: climate_evidence("other-removal-v1", EvidenceKind::Scenario),
                },
                ClimateFlow {
                    flow_id: "avoided-fertilizer".into(),
                    kind: ClimateFlowKind::AvoidedEmission,
                    kg_co2e: 10.0,
                    evidence: climate_evidence("avoided-fertilizer-v1", EvidenceKind::Scenario),
                },
            ],
            char_storage: CharStorageAccounting {
                horizon_years: 100,
                eligibility,
                eligibility_evidence: eligible.then(|| {
                    climate_evidence("verified-biogenic-sourcing-v1", EvidenceKind::Measured)
                }),
                durable_fraction_at_horizon: eligible.then_some(0.8),
                persistence_evidence: eligible.then(|| {
                    climate_evidence("persistence-scenario-v1", EvidenceKind::Scenario)
                }),
            },
        }
    }

    #[test]
    fn climate_balance_separates_emissions_credits_and_horizon_storage() {
        let process = calculate_pyrolysis_batch(&pyrolysis_input()).unwrap();
        let input = climate_input(
            ClimateInventoryStatus::CompleteForDeclaredBoundary,
            CharStorageEligibility::VerifiedEligible,
        );
        let assessment = assess_biochar_climate(&process, &input).unwrap();

        // 180 kg char C × 0.8 assumed durable fraction × 44/12 = 528 kg CO2e.
        assert!((assessment.durable_char_storage_kg_co2e.unwrap() - 528.0).abs() < 1e-10);
        assert_eq!(assessment.gross_emissions_kg_co2e, 100.0);
        assert_eq!(assessment.other_removals_kg_co2e, 5.0);
        assert_eq!(assessment.avoided_emissions_kg_co2e, 10.0);
        assert!((assessment.net_kg_co2e.unwrap() - (-443.0)).abs() < 1e-10);
        assert_eq!(assessment.horizon_years, 100);
        assert_eq!(assessment.process_evidence, process.evidence);
    }

    #[test]
    fn incomplete_inventory_or_unknown_storage_never_reports_net_result() {
        let process = calculate_pyrolysis_batch(&pyrolysis_input()).unwrap();

        let incomplete = climate_input(
            ClimateInventoryStatus::Partial,
            CharStorageEligibility::VerifiedEligible,
        );
        let assessment = assess_biochar_climate(&process, &incomplete).unwrap();
        assert!(assessment.durable_char_storage_kg_co2e.is_some());
        assert_eq!(assessment.net_kg_co2e, None);

        let unknown = climate_input(
            ClimateInventoryStatus::CompleteForDeclaredBoundary,
            CharStorageEligibility::Unknown,
        );
        let assessment = assess_biochar_climate(&process, &unknown).unwrap();
        assert_eq!(assessment.durable_char_storage_kg_co2e, None);
        assert_eq!(assessment.net_kg_co2e, None);
    }

    #[test]
    fn climate_ledger_rejects_unsupported_storage_credit_and_duplicate_flows() {
        let process = calculate_pyrolysis_batch(&pyrolysis_input()).unwrap();

        let mut input = climate_input(
            ClimateInventoryStatus::CompleteForDeclaredBoundary,
            CharStorageEligibility::VerifiedEligible,
        );
        input.char_storage.eligibility_evidence =
            Some(climate_evidence("scenario-eligibility", EvidenceKind::Scenario));
        assert!(assess_biochar_climate(&process, &input).is_err());

        input = climate_input(
            ClimateInventoryStatus::CompleteForDeclaredBoundary,
            CharStorageEligibility::VerifiedIneligible,
        );
        input.flows[1].flow_id = input.flows[0].flow_id.clone();
        assert!(assess_biochar_climate(&process, &input).is_err());

        input = climate_input(
            ClimateInventoryStatus::CompleteForDeclaredBoundary,
            CharStorageEligibility::VerifiedEligible,
        );
        input.char_storage.durable_fraction_at_horizon = Some(1.01);
        assert!(assess_biochar_climate(&process, &input).is_err());
    }

    #[test]
    fn serde_roundtrip_preserves_explicit_assumptions() {
        let input = pyrolysis_input();
        let json = serde_json::to_string(&input).unwrap();
        let decoded: PyrolysisBatchInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, input);
    }
}
