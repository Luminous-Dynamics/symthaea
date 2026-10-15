// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Conservative arithmetic screening for persistent-platform buoyancy and external pressure.
//!
//! This module deliberately does not predict structural strength. Buoyancy is calculated from
//! explicit SI inputs; pressure capacity is compared only when a caller supplies a capacity
//! record with non-empty provenance fields. Those fields are carried, not authenticated.
//! A declared ratio is not a safety factor, certification, flight-readiness result, or substitute
//! for imperfection-sensitive nonlinear analysis and physical tests.

use std::error::Error;
use std::fmt;

/// Invalid input or non-finite derived arithmetic in a screening calculation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScreeningError {
    /// An input value was NaN or infinite.
    NonFiniteInput(&'static str),
    /// An input that must exceed zero was zero or negative.
    MustBePositive(&'static str),
    /// A mass or density that must be non-negative was negative.
    MustBeNonNegative(&'static str),
    /// Arithmetic overflowed while deriving a result.
    NonFiniteResult(&'static str),
    /// A fraction was outside the inclusive interval [0, 1].
    FractionOutOfRange(&'static str),
    /// A declared lower bound exceeds its upper bound.
    InvalidInterval(&'static str),
    /// A peak-to-mean load multiplier was less than one.
    FactorBelowOne(&'static str),
    /// A scenario set is empty.
    EmptyScenarioSet,
    /// A required scenario metadata identifier is blank.
    EmptyIdentifier(&'static str),
    /// Scenario identifiers must be unique within one set.
    DuplicateScenarioId,
}

impl fmt::Display for ScreeningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteInput(name) => write!(f, "{name} must be finite"),
            Self::MustBePositive(name) => write!(f, "{name} must be greater than zero"),
            Self::MustBeNonNegative(name) => write!(f, "{name} must be non-negative"),
            Self::NonFiniteResult(name) => write!(f, "derived {name} is not finite"),
            Self::FractionOutOfRange(name) => write!(f, "{name} must be within [0, 1]"),
            Self::InvalidInterval(name) => {
                write!(f, "{name} lower bound must not exceed its upper bound")
            },
            Self::FactorBelowOne(name) => {
                write!(f, "{name} must be greater than or equal to one")
            },
            Self::EmptyScenarioSet => write!(f, "at least one load-path scenario is required"),
            Self::EmptyIdentifier(name) => write!(f, "{name} must not be blank"),
            Self::DuplicateScenarioId => {
                write!(f, "scenario_id values must be unique within the scenario set")
            },
        }
    }
}

impl Error for ScreeningError {}

fn validate_finite(value: f64, name: &'static str) -> Result<(), ScreeningError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ScreeningError::NonFiniteInput(name))
    }
}

fn validate_positive(value: f64, name: &'static str) -> Result<(), ScreeningError> {
    validate_finite(value, name)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(ScreeningError::MustBePositive(name))
    }
}

fn validate_non_negative(value: f64, name: &'static str) -> Result<(), ScreeningError> {
    validate_finite(value, name)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(ScreeningError::MustBeNonNegative(name))
    }
}

fn validate_result(value: f64, name: &'static str) -> Result<(), ScreeningError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ScreeningError::NonFiniteResult(name))
    }
}

/// Conditional weakest-link calculation for a population of manufactured features.
///
/// This models the probability that at least one feature exceeds a defined defect criterion when
/// every feature has the same declared exceedance probability and all feature events are independent.
/// It is not a shell-failure probability, and it is not an empirical estimate unless the supplied
/// per-feature probability is supported by representative manufacturing/inspection data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeatureDefectPopulationResult {
    pub feature_count: u64,
    pub per_feature_exceedance_probability: f64,
    pub probability_at_least_one_feature_exceeds_criterion: f64,
}

/// Screen how a per-feature defect-tail probability scales with the number of manufactured features.
///
/// The identical-independent-feature assumption is strong: common-cause manufacturing drift,
/// spatial correlation, inspection bias, and feature-to-feature heterogeneity are not represented.
/// The probability is conditional on supplied input, not a measured property of the candidate.
pub fn screen_feature_defect_population(
    feature_count: u64,
    per_feature_exceedance_probability: f64,
) -> Result<FeatureDefectPopulationResult, ScreeningError> {
    if feature_count == 0 {
        return Err(ScreeningError::MustBePositive("feature_count"));
    }
    validate_finite(
        per_feature_exceedance_probability,
        "per_feature_exceedance_probability",
    )?;
    if !(0.0..=1.0).contains(&per_feature_exceedance_probability) {
        return Err(ScreeningError::FractionOutOfRange(
            "per_feature_exceedance_probability",
        ));
    }

    // 1 - (1 - p)^N, evaluated with ln_1p/exp_m1 to reduce cancellation for small p.
    let probability_at_least_one_feature_exceeds_criterion = -((feature_count as f64)
        * (-per_feature_exceedance_probability).ln_1p())
    .exp_m1();
    validate_result(
        probability_at_least_one_feature_exceeds_criterion,
        "feature-population exceedance probability",
    )?;

    Ok(FeatureDefectPopulationResult {
        feature_count,
        per_feature_exceedance_probability,
        probability_at_least_one_feature_exceeds_criterion,
    })
}

/// Exact sphere volume for a positive radius, useful for an initial reference geometry.
pub fn sphere_volume_m3(radius_m: f64) -> Result<f64, ScreeningError> {
    validate_positive(radius_m, "radius_m")?;
    let volume = (4.0 / 3.0) * std::f64::consts::PI * radius_m.powi(3);
    validate_result(volume, "sphere volume")?;
    Ok(volume)
}

/// Explicit non-payload mass terms. No mass is implicitly estimated or defaulted.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct VehicleMassBreakdown {
    /// Pressure boundary, sealing skin, coatings, and attached envelope parts.
    pub envelope_and_seals_kg: f64,
    /// Primary frame, lattice members, joints, and bracing.
    pub frame_lattice_and_joints_kg: f64,
    /// Motors, propellers, and propulsion hardware.
    pub propulsion_kg: f64,
    /// Batteries and other stored-energy hardware.
    pub energy_storage_kg: f64,
    /// Avionics, navigation, communications, and control hardware.
    pub avionics_and_communications_kg: f64,
    /// Landing systems, thermal systems, pumps, redundancy, and all other non-payload mass.
    pub other_systems_kg: f64,
}

impl VehicleMassBreakdown {
    /// Validate each explicit mass and return total vehicle mass excluding payload.
    pub fn total_vehicle_mass_kg(self) -> Result<f64, ScreeningError> {
        let terms = [
            ("envelope_and_seals_kg", self.envelope_and_seals_kg),
            ("frame_lattice_and_joints_kg", self.frame_lattice_and_joints_kg),
            ("propulsion_kg", self.propulsion_kg),
            ("energy_storage_kg", self.energy_storage_kg),
            ("avionics_and_communications_kg", self.avionics_and_communications_kg),
            ("other_systems_kg", self.other_systems_kg),
        ];
        for (name, value) in terms {
            validate_non_negative(value, name)?;
        }
        let total = terms.iter().map(|(_, value)| *value).sum::<f64>();
        validate_result(total, "vehicle mass")?;
        Ok(total)
    }
}

/// Fully explicit inputs for an idealized buoyancy mass balance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BuoyancyRequest {
    /// Enclosed displacement volume in cubic metres.
    pub enclosed_volume_m3: f64,
    /// Ambient air density at the declared operating condition, kg/m³.
    pub ambient_air_density_kg_m3: f64,
    /// Interior gas density at the same reference condition, kg/m³; use 0 for ideal vacuum.
    pub internal_gas_density_kg_m3: f64,
    /// Complete non-payload vehicle mass breakdown.
    pub vehicle_mass: VehicleMassBreakdown,
}

/// Arithmetic mass balance. A negative payload allowance means the vehicle is too heavy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BuoyancyResult {
    /// Mass of displaced ambient air, kg.
    pub displaced_air_mass_kg: f64,
    /// Mass of gas inside the envelope, kg.
    pub internal_gas_mass_kg: f64,
    /// Ideal net buoyant mass before vehicle mass is subtracted, kg.
    pub net_buoyant_mass_kg: f64,
    /// Explicit sum of all non-payload vehicle masses, kg.
    pub non_payload_vehicle_mass_kg: f64,
    /// Maximum theoretical payload allowance, kg; signed to preserve negative results.
    pub payload_allowance_kg: f64,
    /// Payload allowance divided by net buoyant mass when that mass is positive.
    /// This can be negative and is an arithmetic ratio, not an operating margin.
    pub payload_fraction_of_net_buoyant_mass: Option<f64>,
}

/// Compute an idealized SI mass balance. No material strength or pressure resistance is inferred.
pub fn screen_buoyancy(request: BuoyancyRequest) -> Result<BuoyancyResult, ScreeningError> {
    validate_positive(request.enclosed_volume_m3, "enclosed_volume_m3")?;
    validate_positive(request.ambient_air_density_kg_m3, "ambient_air_density_kg_m3")?;
    validate_non_negative(request.internal_gas_density_kg_m3, "internal_gas_density_kg_m3")?;

    let displaced_air_mass_kg =
        request.ambient_air_density_kg_m3 * request.enclosed_volume_m3;
    let internal_gas_mass_kg =
        request.internal_gas_density_kg_m3 * request.enclosed_volume_m3;
    validate_result(displaced_air_mass_kg, "displaced air mass")?;
    validate_result(internal_gas_mass_kg, "internal gas mass")?;

    let net_buoyant_mass_kg = displaced_air_mass_kg - internal_gas_mass_kg;
    validate_result(net_buoyant_mass_kg, "net buoyant mass")?;
    let non_payload_vehicle_mass_kg = request.vehicle_mass.total_vehicle_mass_kg()?;
    let payload_allowance_kg = net_buoyant_mass_kg - non_payload_vehicle_mass_kg;
    validate_result(payload_allowance_kg, "payload allowance")?;

    let payload_fraction_of_net_buoyant_mass = if net_buoyant_mass_kg > 0.0 {
        let fraction = payload_allowance_kg / net_buoyant_mass_kg;
        validate_result(fraction, "payload fraction")?;
        Some(fraction)
    } else {
        None
    };

    Ok(BuoyancyResult {
        displaced_air_mass_kg,
        internal_gas_mass_kg,
        net_buoyant_mass_kg,
        non_payload_vehicle_mass_kg,
        payload_allowance_kg,
        payload_fraction_of_net_buoyant_mass,
    })
}

/// Geometry-based mass budget for an idealized spherical envelope.
///
/// The radius is used for both enclosed volume and pressure-boundary surface area. This is a
/// thin-envelope screening approximation: it ignores thickness-induced volume/area differences,
/// local appendages, deformation, and the mass of structural supports unless the caller includes
/// those in `fixed_non_envelope_vehicle_mass_kg`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SphericalEnvelopeBudgetRequest {
    /// Reference radius of the spherical pressure boundary, metres.
    pub radius_m: f64,
    /// Ambient air density at the declared operating condition, kg/m³.
    pub ambient_air_density_kg_m3: f64,
    /// Interior gas density at the same reference condition, kg/m³; use 0 for ideal vacuum.
    pub internal_gas_density_kg_m3: f64,
    /// All non-envelope, non-payload mass, including frame, joints, propulsion and systems, kg.
    pub fixed_non_envelope_vehicle_mass_kg: f64,
    /// Desired payload mass, kg.
    pub target_payload_mass_kg: f64,
}

/// Signed upper bound for the spherical envelope's remaining mass budget.
///
/// A negative `maximum_envelope_mass_kg` or areal density means the target is already impossible
/// before an envelope is added. A non-negative value only passes arithmetic bookkeeping; it does
/// not imply the envelope can resist external pressure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SphericalEnvelopeBudgetResult {
    /// Idealized enclosed volume, m³.
    pub enclosed_volume_m3: f64,
    /// Idealized pressure-boundary area, m².
    pub envelope_surface_area_m2: f64,
    /// Ideal net buoyant mass before vehicle mass is subtracted, kg.
    pub net_buoyant_mass_kg: f64,
    /// Declared mass of all non-envelope vehicle components, kg.
    pub fixed_non_envelope_vehicle_mass_kg: f64,
    /// Declared target payload mass, kg.
    pub target_payload_mass_kg: f64,
    /// Maximum mass left for the entire envelope and its directly attached layers, kg.
    pub maximum_envelope_mass_kg: f64,
    /// Maximum mean envelope areal mass, kg/m²; signed so infeasibility is preserved.
    pub maximum_envelope_areal_density_kg_m2: f64,
}

/// Determine the mass ceiling available to a spherical pressure envelope.
///
/// This is useful for quickly rejecting candidate materials/architectures that exceed the
/// buoyancy budget. It cannot establish a minimum required thickness or pressure capacity.
pub fn screen_spherical_envelope_budget(
    request: SphericalEnvelopeBudgetRequest,
) -> Result<SphericalEnvelopeBudgetResult, ScreeningError> {
    validate_positive(request.radius_m, "radius_m")?;
    validate_positive(
        request.ambient_air_density_kg_m3,
        "ambient_air_density_kg_m3",
    )?;
    validate_non_negative(
        request.internal_gas_density_kg_m3,
        "internal_gas_density_kg_m3",
    )?;
    validate_non_negative(
        request.fixed_non_envelope_vehicle_mass_kg,
        "fixed_non_envelope_vehicle_mass_kg",
    )?;
    validate_non_negative(request.target_payload_mass_kg, "target_payload_mass_kg")?;

    let enclosed_volume_m3 = sphere_volume_m3(request.radius_m)?;
    let envelope_surface_area_m2 =
        4.0 * std::f64::consts::PI * request.radius_m.powi(2);
    validate_result(envelope_surface_area_m2, "envelope surface area")?;

    let net_buoyant_mass_kg = (request.ambient_air_density_kg_m3
        - request.internal_gas_density_kg_m3)
        * enclosed_volume_m3;
    validate_result(net_buoyant_mass_kg, "net buoyant mass")?;

    let maximum_envelope_mass_kg = net_buoyant_mass_kg
        - request.fixed_non_envelope_vehicle_mass_kg
        - request.target_payload_mass_kg;
    validate_result(maximum_envelope_mass_kg, "maximum envelope mass")?;

    let maximum_envelope_areal_density_kg_m2 =
        maximum_envelope_mass_kg / envelope_surface_area_m2;
    validate_result(
        maximum_envelope_areal_density_kg_m2,
        "maximum envelope areal density",
    )?;

    Ok(SphericalEnvelopeBudgetResult {
        enclosed_volume_m3,
        envelope_surface_area_m2,
        net_buoyant_mass_kg,
        fixed_non_envelope_vehicle_mass_kg: request.fixed_non_envelope_vehicle_mass_kg,
        target_payload_mass_kg: request.target_payload_mass_kg,
        maximum_envelope_mass_kg,
        maximum_envelope_areal_density_kg_m2,
    })
}

/// Candidate spherical-vehicle mass balance using an externally supplied structural areal-mass estimate.
///
/// The structural areal mass must be understood as an effective mass per outer envelope area for
/// the selected architecture (pressure boundary, primary frame/lattice, joints and attached
/// structural elements). It is an input assumption, not a structural prediction. A low value does
/// not imply the candidate can withstand atmospheric pressure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SphericalPayloadRequest {
    /// Reference outer radius in metres.
    pub radius_m: f64,
    /// Ambient air density in kg/m³ at the declared operating condition.
    pub ambient_air_density_kg_m3: f64,
    /// Interior gas density in kg/m³ at the same condition; 0 represents the vacuum limit.
    pub internal_gas_density_kg_m3: f64,
    /// Assumed total primary pressure-envelope / structural-system mass per outer area, kg/m².
    pub structural_system_areal_mass_kg_m2: f64,
    /// Other non-payload vehicle mass not already included in structural areal mass, kg.
    pub fixed_non_structural_vehicle_mass_kg: f64,
}

/// Arithmetic payload budget for one explicitly assumed structural architecture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SphericalPayloadResult {
    pub enclosed_volume_m3: f64,
    pub outer_surface_area_m2: f64,
    pub gross_buoyant_mass_kg: f64,
    pub assumed_structural_system_mass_kg: f64,
    pub fixed_non_structural_vehicle_mass_kg: f64,
    /// Signed maximum payload allowance; negative values mean infeasible under the assumptions.
    pub payload_allowance_kg: f64,
    pub payload_allowance_non_negative: bool,
}

/// Estimate payload allowance conditional on a supplied effective structural areal mass.
///
/// This model is intentionally not a buckling or material model. The caller must source the
/// structural areal-mass assumption and separately establish pressure capacity, failure margins,
/// cargo geometry, route capability, and operational reserves.
pub fn screen_spherical_payload(
    request: SphericalPayloadRequest,
) -> Result<SphericalPayloadResult, ScreeningError> {
    validate_positive(request.radius_m, "radius_m")?;
    validate_positive(
        request.ambient_air_density_kg_m3,
        "ambient_air_density_kg_m3",
    )?;
    validate_non_negative(
        request.internal_gas_density_kg_m3,
        "internal_gas_density_kg_m3",
    )?;
    validate_non_negative(
        request.structural_system_areal_mass_kg_m2,
        "structural_system_areal_mass_kg_m2",
    )?;
    validate_non_negative(
        request.fixed_non_structural_vehicle_mass_kg,
        "fixed_non_structural_vehicle_mass_kg",
    )?;

    let enclosed_volume_m3 = sphere_volume_m3(request.radius_m)?;
    let outer_surface_area_m2 =
        4.0 * std::f64::consts::PI * request.radius_m.powi(2);
    validate_result(outer_surface_area_m2, "outer surface area")?;
    let gross_buoyant_mass_kg = (request.ambient_air_density_kg_m3
        - request.internal_gas_density_kg_m3)
        * enclosed_volume_m3;
    let assumed_structural_system_mass_kg =
        request.structural_system_areal_mass_kg_m2 * outer_surface_area_m2;
    validate_result(gross_buoyant_mass_kg, "gross buoyant mass")?;
    validate_result(
        assumed_structural_system_mass_kg,
        "assumed structural system mass",
    )?;
    let payload_allowance_kg = gross_buoyant_mass_kg
        - assumed_structural_system_mass_kg
        - request.fixed_non_structural_vehicle_mass_kg;
    validate_result(payload_allowance_kg, "payload allowance")?;

    Ok(SphericalPayloadResult {
        enclosed_volume_m3,
        outer_surface_area_m2,
        gross_buoyant_mass_kg,
        assumed_structural_system_mass_kg,
        fixed_non_structural_vehicle_mass_kg: request.fixed_non_structural_vehicle_mass_kg,
        payload_allowance_kg,
        payload_allowance_non_negative: payload_allowance_kg >= 0.0,
    })
}

/// Side-by-side ideal lift comparison for vacuum and helium at the same outer displacement volume.
///
/// Vehicle masses are separately declared for each configuration. They should include every
/// non-payload mass item for that lifting-gas architecture; the function does not estimate
/// pressure-vessel mass or assert that either configuration is structurally viable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VacuumHeliumComparisonRequest {
    /// Enclosed external displacement volume, cubic metres.
    pub enclosed_volume_m3: f64,
    /// Ambient air density at the specified operating condition, kg/m³.
    pub ambient_air_density_kg_m3: f64,
    /// Helium density at the same pressure and temperature as ambient density, kg/m³.
    pub helium_density_kg_m3: f64,
    /// Complete vacuum-vehicle non-payload mass, kg.
    pub vacuum_vehicle_mass_kg: f64,
    /// Complete helium-vehicle non-payload mass, kg.
    pub helium_vehicle_mass_kg: f64,
}

/// Relative mass-budget result for two idealized vehicles with identical outer volume.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VacuumHeliumComparisonResult {
    /// Ideal gross buoyant mass for the vacuum configuration before vehicle mass, kg.
    pub vacuum_gross_buoyant_mass_kg: f64,
    /// Ideal gross buoyant mass for helium before vehicle mass, kg.
    pub helium_gross_buoyant_mass_kg: f64,
    /// Additional gross lift available to vacuum solely because it omits helium, kg.
    pub vacuum_gross_lift_advantage_kg: f64,
    /// Payload allowance after declared vacuum vehicle mass, kg; signed.
    pub vacuum_payload_allowance_kg: f64,
    /// Payload allowance after declared helium vehicle mass, kg; signed.
    pub helium_payload_allowance_kg: f64,
    /// Vacuum payload allowance minus helium payload allowance, kg; signed.
    pub vacuum_minus_helium_payload_kg: f64,
    /// Vacuum vehicle mass minus helium vehicle mass, kg; signed.
    pub vacuum_minus_helium_vehicle_mass_kg: f64,
    /// True when vacuum has the higher arithmetic payload allowance, not a viability claim.
    pub vacuum_has_higher_payload_allowance: bool,
}

/// Compare idealized vacuum and helium payload budgets without hiding the vacuum structure penalty.
///
/// Since `gross_lift_vacuum - gross_lift_helium = helium_density * volume`,
/// vacuum only carries more payload when its additional non-payload mass is less than this amount.
/// Densities must refer to the same operating conditions. No cargo bay, range, weather, reserve,
/// dynamic loading, structural strength, reliability, or operating-cost model is included.
pub fn compare_vacuum_and_helium(
    request: VacuumHeliumComparisonRequest,
) -> Result<VacuumHeliumComparisonResult, ScreeningError> {
    validate_positive(request.enclosed_volume_m3, "enclosed_volume_m3")?;
    validate_positive(
        request.ambient_air_density_kg_m3,
        "ambient_air_density_kg_m3",
    )?;
    validate_non_negative(request.helium_density_kg_m3, "helium_density_kg_m3")?;
    validate_non_negative(request.vacuum_vehicle_mass_kg, "vacuum_vehicle_mass_kg")?;
    validate_non_negative(request.helium_vehicle_mass_kg, "helium_vehicle_mass_kg")?;

    let vacuum_gross_buoyant_mass_kg =
        request.ambient_air_density_kg_m3 * request.enclosed_volume_m3;
    let helium_gross_buoyant_mass_kg = (request.ambient_air_density_kg_m3
        - request.helium_density_kg_m3)
        * request.enclosed_volume_m3;
    let vacuum_gross_lift_advantage_kg =
        vacuum_gross_buoyant_mass_kg - helium_gross_buoyant_mass_kg;
    let vacuum_payload_allowance_kg =
        vacuum_gross_buoyant_mass_kg - request.vacuum_vehicle_mass_kg;
    let helium_payload_allowance_kg =
        helium_gross_buoyant_mass_kg - request.helium_vehicle_mass_kg;
    for (name, value) in [
        ("vacuum gross buoyant mass", vacuum_gross_buoyant_mass_kg),
        ("helium gross buoyant mass", helium_gross_buoyant_mass_kg),
        ("vacuum gross lift advantage", vacuum_gross_lift_advantage_kg),
        ("vacuum payload allowance", vacuum_payload_allowance_kg),
        ("helium payload allowance", helium_payload_allowance_kg),
    ] {
        validate_result(value, name)?;
    }

    let vacuum_minus_helium_payload_kg =
        vacuum_payload_allowance_kg - helium_payload_allowance_kg;
    let vacuum_minus_helium_vehicle_mass_kg =
        request.vacuum_vehicle_mass_kg - request.helium_vehicle_mass_kg;
    validate_result(
        vacuum_minus_helium_payload_kg,
        "vacuum-minus-helium payload",
    )?;
    validate_result(
        vacuum_minus_helium_vehicle_mass_kg,
        "vacuum-minus-helium vehicle mass",
    )?;

    Ok(VacuumHeliumComparisonResult {
        vacuum_gross_buoyant_mass_kg,
        helium_gross_buoyant_mass_kg,
        vacuum_gross_lift_advantage_kg,
        vacuum_payload_allowance_kg,
        helium_payload_allowance_kg,
        vacuum_minus_helium_payload_kg,
        vacuum_minus_helium_vehicle_mass_kg,
        vacuum_has_higher_payload_allowance: vacuum_minus_helium_payload_kg > 0.0,
    })
}

/// Explicit cargo payload requirement and declared usable cargo-space envelope.
///
/// Usable cargo volume is intentionally caller-supplied; it must not be inferred from a vacuum
/// hull's internal volume because pressure boundaries, structure, access, restraints, payload
/// enclosures, and clearances consume or restrict usable space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CargoFitRequest {
    /// Vehicle's calculated payload allowance, kg; may be negative.
    pub payload_allowance_kg: f64,
    /// Required cargo mass, kg.
    pub target_cargo_mass_kg: f64,
    /// Required cargo volume, m³.
    pub target_cargo_volume_m3: f64,
    /// Configuration-specific usable cargo volume, m³.
    pub usable_cargo_volume_m3: f64,
}

/// Separate mass/volume requirements. Passing both gates is only a cargo-fit screen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CargoFitResult {
    /// Payload allowance minus target cargo mass, kg; non-negative is a mass fit.
    pub cargo_mass_margin_kg: f64,
    /// Usable cargo volume minus target cargo volume, m³; non-negative is a volume fit.
    pub cargo_volume_margin_m3: f64,
    pub cargo_fits_mass: bool,
    pub cargo_fits_volume: bool,
    /// True only when both the mass and the declared volume gates pass.
    /// It does not imply route, structure, handling, weather, or commercial feasibility.
    pub cargo_fits_declared_mass_and_volume: bool,
}

/// Check cargo mass and package volume against explicitly supplied vehicle capacity.
pub fn screen_cargo_fit(request: CargoFitRequest) -> Result<CargoFitResult, ScreeningError> {
    validate_finite(request.payload_allowance_kg, "payload_allowance_kg")?;
    validate_non_negative(request.target_cargo_mass_kg, "target_cargo_mass_kg")?;
    validate_non_negative(request.target_cargo_volume_m3, "target_cargo_volume_m3")?;
    validate_non_negative(request.usable_cargo_volume_m3, "usable_cargo_volume_m3")?;

    let cargo_mass_margin_kg =
        request.payload_allowance_kg - request.target_cargo_mass_kg;
    let cargo_volume_margin_m3 =
        request.usable_cargo_volume_m3 - request.target_cargo_volume_m3;
    validate_result(cargo_mass_margin_kg, "cargo mass margin")?;
    validate_result(cargo_volume_margin_m3, "cargo volume margin")?;
    let cargo_fits_mass = cargo_mass_margin_kg >= 0.0;
    let cargo_fits_volume = cargo_volume_margin_m3 >= 0.0;

    Ok(CargoFitResult {
        cargo_mass_margin_kg,
        cargo_volume_margin_m3,
        cargo_fits_mass,
        cargo_fits_volume,
        cargo_fits_declared_mass_and_volume: cargo_fits_mass && cargo_fits_volume,
    })
}

/// Registered measurements from a structural recovery cycling test.
///
/// Deformation metrics must be computed from registered full-field shape scans against the same
/// baseline, not from visual inspection or one diameter alone. Stiffness values must use the same
/// declared test method and units before/after cycling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElasticRecoveryObservation {
    /// RMS magnitude of full-field displacement at maximum prescribed load, metres.
    pub peak_deformation_rms_m: f64,
    /// RMS magnitude of residual displacement after unload, metres.
    pub residual_deformation_rms_m: f64,
    /// Maximum measured surface deviation from baseline after unload, metres.
    pub max_residual_surface_deviation_m: f64,
    /// Bound on absolute metrology error for maximum residual surface deviation, metres.
    pub metrology_uncertainty_m: f64,
    /// Pre-cycle tangent stiffness under the declared test method, N/m.
    pub pre_cycle_stiffness_n_per_m: f64,
    /// Post-cycle tangent stiffness under the same test method, N/m.
    pub post_cycle_stiffness_n_per_m: f64,
    /// Number of completed load/unload cycles before post-cycle measurements.
    pub cycle_count: u32,
}

/// Pre-registered geometric and stiffness criteria for a specific recovery test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElasticRecoveryCriteria {
    /// Maximum permissible post-unload surface deviation, metres.
    pub max_residual_surface_deviation_m: f64,
    /// Minimum acceptable retained stiffness as a fraction of pre-cycle stiffness.
    pub min_stiffness_retention_fraction: f64,
}

/// Result for a declared test criterion, with uncertainty represented explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryGate {
    /// The measured uncertainty interval satisfies the criterion.
    Pass,
    /// The measured uncertainty interval clearly violates the criterion.
    Fail,
    /// Measurement uncertainty overlaps the acceptance boundary.
    Indeterminate,
}

/// Metric-only result of an elastic shape-recovery test.
///
/// Passing this gate means only that this specimen met its pre-registered shape/stiffness
/// recovery criteria for the tested load history. It does not demonstrate pressure capacity,
/// fatigue life, damage tolerance, airworthiness, or suitability for flight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElasticRecoveryResult {
    /// One minus residual RMS deformation divided by peak RMS deformation.
    /// Negative values are preserved if permanent residual deformation exceeds peak deformation.
    pub shape_recovery_fraction: f64,
    /// Post-cycle stiffness divided by pre-cycle stiffness.
    pub stiffness_retention_fraction: f64,
    /// Conservative lower bound on maximum residual surface deviation, metres.
    pub residual_deviation_lower_bound_m: f64,
    /// Conservative upper bound on maximum residual surface deviation, metres.
    pub residual_deviation_upper_bound_m: f64,
    pub shape_recovery_gate: RecoveryGate,
    pub stiffness_retention_gate: RecoveryGate,
    /// Combined result for recovery metrics only; it is not a structural safety gate.
    pub overall_recovery_gate: RecoveryGate,
    pub cycle_count: u32,
}

/// Evaluate post-load shape recovery and stiffness retention against frozen acceptance criteria.
pub fn assess_elastic_recovery(
    observation: ElasticRecoveryObservation,
    criteria: ElasticRecoveryCriteria,
) -> Result<ElasticRecoveryResult, ScreeningError> {
    validate_positive(
        observation.peak_deformation_rms_m,
        "peak_deformation_rms_m",
    )?;
    validate_non_negative(
        observation.residual_deformation_rms_m,
        "residual_deformation_rms_m",
    )?;
    validate_non_negative(
        observation.max_residual_surface_deviation_m,
        "max_residual_surface_deviation_m",
    )?;
    validate_non_negative(
        observation.metrology_uncertainty_m,
        "metrology_uncertainty_m",
    )?;
    validate_positive(
        observation.pre_cycle_stiffness_n_per_m,
        "pre_cycle_stiffness_n_per_m",
    )?;
    validate_positive(
        observation.post_cycle_stiffness_n_per_m,
        "post_cycle_stiffness_n_per_m",
    )?;
    if observation.cycle_count == 0 {
        return Err(ScreeningError::MustBePositive("cycle_count"));
    }
    validate_non_negative(
        criteria.max_residual_surface_deviation_m,
        "max_residual_surface_deviation_m",
    )?;
    validate_non_negative(
        criteria.min_stiffness_retention_fraction,
        "min_stiffness_retention_fraction",
    )?;
    if criteria.min_stiffness_retention_fraction > 1.0 {
        return Err(ScreeningError::FractionOutOfRange(
            "min_stiffness_retention_fraction",
        ));
    }

    let shape_recovery_fraction = 1.0
        - observation.residual_deformation_rms_m / observation.peak_deformation_rms_m;
    let stiffness_retention_fraction =
        observation.post_cycle_stiffness_n_per_m / observation.pre_cycle_stiffness_n_per_m;
    let residual_deviation_lower_bound_m = (observation.max_residual_surface_deviation_m
        - observation.metrology_uncertainty_m)
        .max(0.0);
    let residual_deviation_upper_bound_m = observation.max_residual_surface_deviation_m
        + observation.metrology_uncertainty_m;
    for (name, value) in [
        ("shape recovery fraction", shape_recovery_fraction),
        ("stiffness retention fraction", stiffness_retention_fraction),
        ("residual deviation lower bound", residual_deviation_lower_bound_m),
        ("residual deviation upper bound", residual_deviation_upper_bound_m),
    ] {
        validate_result(value, name)?;
    }

    let shape_recovery_gate =
        if residual_deviation_upper_bound_m <= criteria.max_residual_surface_deviation_m {
            RecoveryGate::Pass
        } else if residual_deviation_lower_bound_m
            > criteria.max_residual_surface_deviation_m
        {
            RecoveryGate::Fail
        } else {
            RecoveryGate::Indeterminate
        };
    let stiffness_retention_gate =
        if stiffness_retention_fraction >= criteria.min_stiffness_retention_fraction {
            RecoveryGate::Pass
        } else {
            RecoveryGate::Fail
        };
    let overall_recovery_gate = if shape_recovery_gate == RecoveryGate::Fail
        || stiffness_retention_gate == RecoveryGate::Fail
    {
        RecoveryGate::Fail
    } else if shape_recovery_gate == RecoveryGate::Indeterminate
        || stiffness_retention_gate == RecoveryGate::Indeterminate
    {
        RecoveryGate::Indeterminate
    } else {
        RecoveryGate::Pass
    };

    Ok(ElasticRecoveryResult {
        shape_recovery_fraction,
        stiffness_retention_fraction,
        residual_deviation_lower_bound_m,
        residual_deviation_upper_bound_m,
        shape_recovery_gate,
        stiffness_retention_gate,
        overall_recovery_gate,
        cycle_count: observation.cycle_count,
    })
}

/// External-pressure force resultants for a spherical shell's great-circle cut.
///
/// This applies the ideal equilibrium relation C = Δp · πR² to a perfect sphere. It yields the
/// total compressive resultant transmitted across a great-circle section, not the load on one
/// lattice member and not a buckling prediction. Real member loads depend on topology, joint
/// stiffness, imperfections, boundary conditions and load redistribution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SphericalPressureLoadRequest {
    /// Reference outer radius, metres.
    pub radius_m: f64,
    /// External absolute pressure, Pa.
    pub external_pressure_pa: f64,
    /// Internal absolute pressure, Pa.
    pub internal_pressure_pa: f64,
}

/// Pressure load across a great-circle section under ideal spherical equilibrium.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SphericalPressureLoadResult {
    /// Signed exterior minus interior pressure, Pa.
    pub differential_pressure_pa: f64,
    /// Applied positive external-pressure differential, Pa; zero for non-compressive cases.
    pub positive_external_differential_pa: f64,
    /// Idealized area of the great-circle cut, m².
    pub great_circle_area_m2: f64,
    /// Ideal compressive resultant across that cut, newtons; zero if external pressure does not exceed internal pressure.
    pub great_circle_compressive_resultant_n: f64,
}

/// Calculate the global pressure resultant that the structural topology must transfer.
///
/// This helps convert a pressure requirement into a structural-load scale for subsequent analysis.
/// It does not distribute the resultant across members or prove any material/structure is capable
/// of resisting it.
pub fn screen_spherical_pressure_load(
    request: SphericalPressureLoadRequest,
) -> Result<SphericalPressureLoadResult, ScreeningError> {
    validate_positive(request.radius_m, "radius_m")?;
    validate_non_negative(request.external_pressure_pa, "external_pressure_pa")?;
    validate_non_negative(request.internal_pressure_pa, "internal_pressure_pa")?;

    let differential_pressure_pa =
        request.external_pressure_pa - request.internal_pressure_pa;
    let positive_external_differential_pa = differential_pressure_pa.max(0.0);
    let great_circle_area_m2 = std::f64::consts::PI * request.radius_m.powi(2);
    let great_circle_compressive_resultant_n =
        positive_external_differential_pa * great_circle_area_m2;
    validate_result(differential_pressure_pa, "pressure differential")?;
    validate_result(great_circle_area_m2, "great-circle area")?;
    validate_result(
        great_circle_compressive_resultant_n,
        "great-circle compressive resultant",
    )?;

    Ok(SphericalPressureLoadResult {
        differential_pressure_pa,
        positive_external_differential_pa,
        great_circle_area_m2,
        great_circle_compressive_resultant_n,
    })
}

/// Inputs for translating a global compressive resultant into an explicit load-path estimate.
///
/// The effective path count may be a homogenized/non-integer count from a declared analytical
/// model. It is not inferred from geometry by this function. The peak-to-mean factor is caller
/// supplied; it is not a safety factor and is not automatically conservative.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoadPathDemandRequest {
    /// Total compressive resultant to be distributed, newtons.
    pub total_compressive_resultant_n: f64,
    /// Declared effective number of load paths crossing the cut; must be positive and finite.
    pub effective_load_path_count: f64,
    /// Declared peak-to-mean load concentration multiplier; must be >= 1.
    pub assumed_peak_to_mean_load_factor: f64,
}

/// Arithmetic load-path estimate for a declared effective count and concentration factor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoadPathDemandResult {
    pub total_compressive_resultant_n: f64,
    pub effective_load_path_count: f64,
    /// Equal-share reference load per effective path, newtons.
    pub mean_compressive_load_per_effective_path_n: f64,
    /// Caller-supplied factor. This is not an independently established upper bound.
    pub assumed_peak_to_mean_load_factor: f64,
    /// Mean load multiplied by the caller-supplied concentration factor, newtons.
    pub factor_scaled_peak_load_path_demand_n: f64,
}

/// Convert a global resultant into an equal-share reference and factor-scaled load-path demand.
///
/// This is a bookkeeping bridge to a geometry-specific structural model. Actual load sharing
/// depends on topology, joint stiffness, boundary conditions and spatial imperfections. A supplied
/// factor below the real concentration would understate member demand, so this function never
/// labels its scaled output as a maximum or capacity.
pub fn screen_load_path_demand(
    request: LoadPathDemandRequest,
) -> Result<LoadPathDemandResult, ScreeningError> {
    validate_non_negative(
        request.total_compressive_resultant_n,
        "total_compressive_resultant_n",
    )?;
    validate_positive(
        request.effective_load_path_count,
        "effective_load_path_count",
    )?;
    validate_finite(
        request.assumed_peak_to_mean_load_factor,
        "assumed_peak_to_mean_load_factor",
    )?;
    if request.assumed_peak_to_mean_load_factor < 1.0 {
        return Err(ScreeningError::FactorBelowOne(
            "assumed_peak_to_mean_load_factor",
        ));
    }

    let mean_compressive_load_per_effective_path_n =
        request.total_compressive_resultant_n / request.effective_load_path_count;
    let factor_scaled_peak_load_path_demand_n =
        mean_compressive_load_per_effective_path_n * request.assumed_peak_to_mean_load_factor;
    validate_result(
        mean_compressive_load_per_effective_path_n,
        "mean compressive load per effective path",
    )?;
    validate_result(
        factor_scaled_peak_load_path_demand_n,
        "factor-scaled peak load-path demand",
    )?;

    Ok(LoadPathDemandResult {
        total_compressive_resultant_n: request.total_compressive_resultant_n,
        effective_load_path_count: request.effective_load_path_count,
        mean_compressive_load_per_effective_path_n,
        assumed_peak_to_mean_load_factor: request.assumed_peak_to_mean_load_factor,
        factor_scaled_peak_load_path_demand_n,
    })
}

/// Bounded total load, effective-path count and concentration factor for load-path screening.
///
/// Bounds are deterministic assumptions, not probability distributions. All interval endpoints
/// must be justified independently; the function does not infer paths or concentration from CAD.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundedLoadPathDemandRequest {
    /// Lower/upper total compressive resultant, newtons.
    pub total_compressive_resultant_n: ScalarBounds,
    /// Lower/upper effective number of load paths; both must be positive.
    pub effective_load_path_count: ScalarBounds,
    /// Lower/upper peak-to-mean concentration multiplier; both must be >= 1.
    pub peak_to_mean_load_factor: ScalarBounds,
}

/// Deterministic envelope for mean and factor-scaled load-path demand.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundedLoadPathDemandResult {
    pub mean_compressive_load_per_effective_path_lower_bound_n: f64,
    pub mean_compressive_load_per_effective_path_upper_bound_n: f64,
    /// Lower arithmetic endpoint over all independent declared input intervals.
    pub factor_scaled_load_path_demand_lower_bound_n: f64,
    /// Upper arithmetic endpoint over all independent declared input intervals.
    /// This remains a caller-bound-dependent estimate, not a verified peak member force.
    pub factor_scaled_load_path_demand_upper_bound_n: f64,
}

/// Propagate declared uncertainty bounds through the load-path arithmetic.
///
/// The minimum uses the minimum resultant, maximum path count, and minimum multiplier. The maximum
/// uses the maximum resultant, minimum path count, and maximum multiplier. This is a conservative
/// endpoint calculation only for the supplied independent intervals; if inputs are correlated, the
/// user must supply joint bounds or analyze the correlations separately. The upper endpoint is not
/// a shell capacity or validated maximum force.
pub fn screen_bounded_load_path_demand(
    request: BoundedLoadPathDemandRequest,
) -> Result<BoundedLoadPathDemandResult, ScreeningError> {
    validate_bounds(
        request.total_compressive_resultant_n,
        "total_compressive_resultant_n",
        true,
    )?;
    validate_bounds(
        request.effective_load_path_count,
        "effective_load_path_count",
        false,
    )?;
    validate_bounds(
        request.peak_to_mean_load_factor,
        "peak_to_mean_load_factor",
        true,
    )?;
    if request.peak_to_mean_load_factor.min < 1.0 {
        return Err(ScreeningError::FactorBelowOne("peak_to_mean_load_factor"));
    }

    let mean_compressive_load_per_effective_path_lower_bound_n =
        request.total_compressive_resultant_n.min
            / request.effective_load_path_count.max;
    let mean_compressive_load_per_effective_path_upper_bound_n =
        request.total_compressive_resultant_n.max
            / request.effective_load_path_count.min;
    let factor_scaled_load_path_demand_lower_bound_n =
        mean_compressive_load_per_effective_path_lower_bound_n
            * request.peak_to_mean_load_factor.min;
    let factor_scaled_load_path_demand_upper_bound_n =
        mean_compressive_load_per_effective_path_upper_bound_n
            * request.peak_to_mean_load_factor.max;

    for (name, value) in [
        (
            "mean load-path lower bound",
            mean_compressive_load_per_effective_path_lower_bound_n,
        ),
        (
            "mean load-path upper bound",
            mean_compressive_load_per_effective_path_upper_bound_n,
        ),
        (
            "factor-scaled load-path lower bound",
            factor_scaled_load_path_demand_lower_bound_n,
        ),
        (
            "factor-scaled load-path upper bound",
            factor_scaled_load_path_demand_upper_bound_n,
        ),
    ] {
        validate_result(value, name)?;
    }

    Ok(BoundedLoadPathDemandResult {
        mean_compressive_load_per_effective_path_lower_bound_n,
        mean_compressive_load_per_effective_path_upper_bound_n,
        factor_scaled_load_path_demand_lower_bound_n,
        factor_scaled_load_path_demand_upper_bound_n,
    })
}

/// One coherent, explicitly declared load-path case.
///
/// Keep resultant, path count, and concentration factor from the same geometry/load/imperfection
/// case together here. IDs are carried for auditability but are not authenticated by this module.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadPathDemandScenario {
    /// Unique identifier for this exact scenario within a submitted set.
    pub scenario_id: String,
    /// Immutable analysis or evidence-manifest reference for this scenario.
    pub evidence_record_id: String,
    /// Total compressive resultant for this scenario, newtons.
    pub total_compressive_resultant_n: f64,
    /// Effective load-path count for this same scenario; must be positive.
    pub effective_load_path_count: f64,
    /// Peak-to-mean concentration factor for this same scenario; must be >= 1.
    pub peak_to_mean_load_factor: f64,
}

/// Calculated demand paired with the source identifiers for a single coherent scenario.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadPathScenarioDemandResult {
    pub scenario_id: String,
    pub evidence_record_id: String,
    pub demand: LoadPathDemandResult,
}

/// Demand envelope across only the coherent scenarios supplied by the caller.
///
/// This is not a bound on unmodeled geometries, defect fields, or operating conditions.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadPathScenarioSetResult {
    pub evaluated_scenario_count: usize,
    pub scenario_results: Vec<LoadPathScenarioDemandResult>,
    pub minimum_factor_scaled_demand_among_scenarios_n: f64,
    pub minimum_scenario_id: String,
    pub maximum_factor_scaled_demand_among_scenarios_n: f64,
    pub maximum_scenario_id: String,
}

/// Evaluate each supplied joint scenario without combining marginal extrema from different cases.
///
/// Unlike screen_bounded_load_path_demand, this function preserves dependence between the total
/// resultant, effective path count, and concentration factor by evaluating each coherent tuple.
/// The reported envelope covers the listed scenarios only; it offers no coverage guarantee for
/// cases or imperfections that were not supplied. Scenario IDs must be unique and both identifiers
/// must be non-blank. Identifier checks improve traceability but do not authenticate source records.
pub fn screen_load_path_scenario_set(
    scenarios: &[LoadPathDemandScenario],
) -> Result<LoadPathScenarioSetResult, ScreeningError> {
    if scenarios.is_empty() {
        return Err(ScreeningError::EmptyScenarioSet);
    }

    let mut seen_ids: Vec<&str> = Vec::with_capacity(scenarios.len());
    let mut scenario_results = Vec::with_capacity(scenarios.len());
    for scenario in scenarios {
        if scenario.scenario_id.trim().is_empty() {
            return Err(ScreeningError::EmptyIdentifier("scenario_id"));
        }
        if scenario.evidence_record_id.trim().is_empty() {
            return Err(ScreeningError::EmptyIdentifier("evidence_record_id"));
        }
        if seen_ids.iter().any(|id| *id == scenario.scenario_id.as_str()) {
            return Err(ScreeningError::DuplicateScenarioId);
        }
        seen_ids.push(scenario.scenario_id.as_str());

        let demand = screen_load_path_demand(LoadPathDemandRequest {
            total_compressive_resultant_n: scenario.total_compressive_resultant_n,
            effective_load_path_count: scenario.effective_load_path_count,
            assumed_peak_to_mean_load_factor: scenario.peak_to_mean_load_factor,
        })?;
        scenario_results.push(LoadPathScenarioDemandResult {
            scenario_id: scenario.scenario_id.clone(),
            evidence_record_id: scenario.evidence_record_id.clone(),
            demand,
        });
    }

    let mut minimum_index = 0;
    let mut maximum_index = 0;
    for index in 1..scenario_results.len() {
        let demand = scenario_results[index].demand.factor_scaled_peak_load_path_demand_n;
        if demand
            < scenario_results[minimum_index].demand.factor_scaled_peak_load_path_demand_n
        {
            minimum_index = index;
        }
        if demand
            > scenario_results[maximum_index].demand.factor_scaled_peak_load_path_demand_n
        {
            maximum_index = index;
        }
    }

    let minimum_factor_scaled_demand_among_scenarios_n = scenario_results[minimum_index]
        .demand
        .factor_scaled_peak_load_path_demand_n;
    let minimum_scenario_id = scenario_results[minimum_index].scenario_id.clone();
    let maximum_factor_scaled_demand_among_scenarios_n = scenario_results[maximum_index]
        .demand
        .factor_scaled_peak_load_path_demand_n;
    let maximum_scenario_id = scenario_results[maximum_index].scenario_id.clone();
    let evaluated_scenario_count = scenario_results.len();

    Ok(LoadPathScenarioSetResult {
        evaluated_scenario_count,
        scenario_results,
        minimum_factor_scaled_demand_among_scenarios_n,
        minimum_scenario_id,
        maximum_factor_scaled_demand_among_scenarios_n,
        maximum_scenario_id,
    })
}

/// An externally sourced pressure-capacity declaration plus the references needed to audit it.
///
/// The caller must supply a conservative allowable pressure for the exact applicable design
/// generation, not merely a perfect-geometry linear eigenvalue. This structure checks that
/// references are present; it cannot authenticate a solver, test, or independence claim.
#[derive(Debug, Clone, PartialEq)]
pub struct PressureCapacityRecord {
    /// Declared allowable external differential pressure, Pa.
    pub allowable_external_pressure_pa: f64,
    /// Immutable record identifying the capacity result.
    pub evidence_record_id: String,
    /// Exact model/test profile and configuration reference.
    pub basis_profile_id: String,
    /// Independent check or physical-test record reference.
    pub independent_check_or_test_id: String,
}

/// Explicit external-pressure case. All pressures are absolute SI pascals.
#[derive(Debug, Clone, PartialEq)]
pub struct ExternalPressureRequest {
    pub external_pressure_pa: f64,
    pub internal_pressure_pa: f64,
    pub capacity: Option<PressureCapacityRecord>,
}

/// Pressure comparison status. No variant declares a design safe or flight-ready.
#[derive(Debug, Clone, PartialEq)]
pub enum PressureAssessment {
    /// Interior pressure equals or exceeds exterior pressure: no positive external differential.
    NoExternalPressureDifferential {
        differential_pressure_pa: f64,
    },
    /// No external capacity record was supplied, so resistance is unknown to this screen.
    CapacityUnknown {
        differential_pressure_pa: f64,
    },
    /// Capacity was supplied, but one or more audit references are absent.
    IncompleteProvenance {
        differential_pressure_pa: f64,
        missing_fields: Vec<&'static str>,
    },
    /// Arithmetic comparison only; the references have not been authenticated by this module.
    DeclaredCapacityComparison {
        differential_pressure_pa: f64,
        allowable_external_pressure_pa: f64,
        declared_capacity_ratio: f64,
        evidence_record_id: String,
        basis_profile_id: String,
        independent_check_or_test_id: String,
    },
}

/// Compare an explicit external differential pressure against an externally qualified capacity.
///
/// A ratio above 1.0 means only that the declared capacity exceeds this declared pressure load.
/// It does not account for mismatched model scope, unmodeled damage, reliability, certification,
/// dynamic loading, or whether a purported independent review is actually independent.
pub fn assess_external_pressure(
    request: ExternalPressureRequest,
) -> Result<PressureAssessment, ScreeningError> {
    validate_non_negative(request.external_pressure_pa, "external_pressure_pa")?;
    validate_non_negative(request.internal_pressure_pa, "internal_pressure_pa")?;

    let differential_pressure_pa = request.external_pressure_pa - request.internal_pressure_pa;
    validate_result(differential_pressure_pa, "differential pressure")?;
    if differential_pressure_pa <= 0.0 {
        return Ok(PressureAssessment::NoExternalPressureDifferential {
            differential_pressure_pa,
        });
    }

    let Some(capacity) = request.capacity else {
        return Ok(PressureAssessment::CapacityUnknown {
            differential_pressure_pa,
        });
    };
    validate_positive(
        capacity.allowable_external_pressure_pa,
        "allowable_external_pressure_pa",
    )?;

    let mut missing_fields = Vec::new();
    if capacity.evidence_record_id.trim().is_empty() {
        missing_fields.push("evidence_record_id");
    }
    if capacity.basis_profile_id.trim().is_empty() {
        missing_fields.push("basis_profile_id");
    }
    if capacity.independent_check_or_test_id.trim().is_empty() {
        missing_fields.push("independent_check_or_test_id");
    }
    if !missing_fields.is_empty() {
        return Ok(PressureAssessment::IncompleteProvenance {
            differential_pressure_pa,
            missing_fields,
        });
    }

    let declared_capacity_ratio =
        capacity.allowable_external_pressure_pa / differential_pressure_pa;
    validate_result(declared_capacity_ratio, "declared capacity ratio")?;

    Ok(PressureAssessment::DeclaredCapacityComparison {
        differential_pressure_pa,
        allowable_external_pressure_pa: capacity.allowable_external_pressure_pa,
        declared_capacity_ratio,
        evidence_record_id: capacity.evidence_record_id,
        basis_profile_id: capacity.basis_profile_id,
        independent_check_or_test_id: capacity.independent_check_or_test_id,
    })
}


/// Identifies the exact model scope a pressure-capacity record claims to cover.
///
/// Values should be immutable IDs or content hashes for versioned artifacts, not free-form
/// descriptions. This screen compares identifiers exactly but does not fetch or authenticate
/// those artifacts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PressureCaseScope {
    /// Exact CAD/as-built geometry or measured shape-scan revision.
    pub geometry_revision_id: String,
    /// Exact FE mesh or, for physical evidence, specimen/build revision.
    pub mesh_or_specimen_revision_id: String,
    /// Exact laminate/material model or tested material/batch revision.
    pub material_model_revision_id: String,
    /// Solver build/settings or physical test method/instrumentation procedure.
    pub analysis_or_test_method_id: String,
    pub load_case_id: String,
    pub boundary_condition_profile_id: String,
    /// ID/hash for the spatial imperfection field, mode family and bounded defect set;
    /// a single scalar amplitude is not sufficient to identify this analysis basis.
    pub imperfection_basis_id: String,
    /// Defines the quantity and derivation of the claimed capacity, including knockdown/allowable
    /// method or the tested failure/cycle criterion; eigenvalue and qualified allowable are distinct.
    pub capacity_basis_id: String,
}

/// One field where supplied capacity scope does not match the requested case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PressureScopeMismatch {
    pub field: &'static str,
    pub required_value: String,
    pub supplied_value: String,
}

/// Request the bounded pressure comparison only for evidence matching a declared case scope.
#[derive(Debug, Clone, PartialEq)]
pub struct ScopedExternalPressureBoundsRequest {
    pub pressure: ExternalPressureBoundsRequest,
    /// Scope of the exact model/article and load case being assessed.
    pub required_scope: PressureCaseScope,
    /// Scope claimed by the artifact used to justify the pressure capacity.
    pub supplied_capacity_scope: Option<PressureCaseScope>,
}

/// Scope-aware wrapper around the arithmetic pressure comparison.
///
/// An identifier match is necessary metadata, not proof that the artifacts exist, are authentic,
/// independent, or appropriately qualified. Scope mismatch blocks the capacity comparison.
#[derive(Debug, Clone, PartialEq)]
pub enum ScopedPressureBoundsAssessment {
    ScopeIncomplete {
        missing_fields: Vec<&'static str>,
    },
    ScopeMismatch {
        mismatches: Vec<PressureScopeMismatch>,
    },
    PressureAssessment(PressureBoundsAssessment),
}

/// Compare bounded pressure demand/capacity only after exact case-scope identifiers match.
///
/// Inputs and any declared capacity interval are arithmetically validated first. When no capacity
/// is supplied, the result remains unknown. When there is no positive external-pressure demand,
/// the demand result is returned without requiring an irrelevant capacity-scope record.
pub fn assess_scoped_external_pressure_bounds(
    request: ScopedExternalPressureBoundsRequest,
) -> Result<ScopedPressureBoundsAssessment, ScreeningError> {
    let pressure_assessment = assess_external_pressure_bounds(request.pressure.clone())?;
    if matches!(
        &pressure_assessment,
        PressureBoundsAssessment::NoExternalPressureDemand { .. }
            | PressureBoundsAssessment::CapacityUnknown { .. }
    ) {
        return Ok(ScopedPressureBoundsAssessment::PressureAssessment(
            pressure_assessment,
        ));
    }

    let Some(supplied_scope) = request.supplied_capacity_scope.as_ref() else {
        return Ok(ScopedPressureBoundsAssessment::ScopeIncomplete {
            missing_fields: vec!["supplied_capacity_scope"],
        });
    };

    let scope_fields: [(&'static str, &str, &str); 8] = [
        (
            "geometry_revision_id",
            &request.required_scope.geometry_revision_id,
            &supplied_scope.geometry_revision_id,
        ),
        (
            "mesh_or_specimen_revision_id",
            &request.required_scope.mesh_or_specimen_revision_id,
            &supplied_scope.mesh_or_specimen_revision_id,
        ),
        (
            "material_model_revision_id",
            &request.required_scope.material_model_revision_id,
            &supplied_scope.material_model_revision_id,
        ),
        (
            "analysis_or_test_method_id",
            &request.required_scope.analysis_or_test_method_id,
            &supplied_scope.analysis_or_test_method_id,
        ),
        (
            "load_case_id",
            &request.required_scope.load_case_id,
            &supplied_scope.load_case_id,
        ),
        (
            "boundary_condition_profile_id",
            &request.required_scope.boundary_condition_profile_id,
            &supplied_scope.boundary_condition_profile_id,
        ),
        (
            "imperfection_basis_id",
            &request.required_scope.imperfection_basis_id,
            &supplied_scope.imperfection_basis_id,
        ),
        (
            "capacity_basis_id",
            &request.required_scope.capacity_basis_id,
            &supplied_scope.capacity_basis_id,
        ),
    ];

    let mut missing_fields = Vec::new();
    let mut mismatches = Vec::new();
    for (field, required, supplied) in scope_fields {
        if required.trim().is_empty() {
            missing_fields.push(match field {
                "geometry_revision_id" => "required_scope.geometry_revision_id",
                "mesh_or_specimen_revision_id" => {
                    "required_scope.mesh_or_specimen_revision_id"
                }
                "material_model_revision_id" => "required_scope.material_model_revision_id",
                "analysis_or_test_method_id" => "required_scope.analysis_or_test_method_id",
                "load_case_id" => "required_scope.load_case_id",
                "boundary_condition_profile_id" => {
                    "required_scope.boundary_condition_profile_id"
                }
                "imperfection_basis_id" => "required_scope.imperfection_basis_id",
                "capacity_basis_id" => "required_scope.capacity_basis_id",
                _ => unreachable!("scope field names are a closed set"),
            });
        }
        if supplied.trim().is_empty() {
            missing_fields.push(match field {
                "geometry_revision_id" => "supplied_capacity_scope.geometry_revision_id",
                "mesh_or_specimen_revision_id" => {
                    "supplied_capacity_scope.mesh_or_specimen_revision_id"
                }
                "material_model_revision_id" => {
                    "supplied_capacity_scope.material_model_revision_id"
                }
                "analysis_or_test_method_id" => {
                    "supplied_capacity_scope.analysis_or_test_method_id"
                }
                "load_case_id" => "supplied_capacity_scope.load_case_id",
                "boundary_condition_profile_id" => {
                    "supplied_capacity_scope.boundary_condition_profile_id"
                }
                "imperfection_basis_id" => "supplied_capacity_scope.imperfection_basis_id",
                "capacity_basis_id" => "supplied_capacity_scope.capacity_basis_id",
                _ => unreachable!("scope field names are a closed set"),
            });
        }
        if !required.trim().is_empty()
            && !supplied.trim().is_empty()
            && required != supplied
        {
            mismatches.push(PressureScopeMismatch {
                field,
                required_value: required.to_owned(),
                supplied_value: supplied.to_owned(),
            });
        }
    }

    if !missing_fields.is_empty() {
        return Ok(ScopedPressureBoundsAssessment::ScopeIncomplete { missing_fields });
    }
    if !mismatches.is_empty() {
        return Ok(ScopedPressureBoundsAssessment::ScopeMismatch { mismatches });
    }

    Ok(ScopedPressureBoundsAssessment::PressureAssessment(
        pressure_assessment,
    ))
}

/// Deterministic bounds for a scalar quantity. These are engineering bounds, not statistical
/// confidence intervals; the caller is responsible for choosing justified limits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalarBounds {
    pub min: f64,
    pub max: f64,
}

fn validate_bounds(
    bounds: ScalarBounds,
    name: &'static str,
    allow_zero: bool,
) -> Result<(), ScreeningError> {
    validate_finite(bounds.min, name)?;
    validate_finite(bounds.max, name)?;
    if bounds.min > bounds.max {
        return Err(ScreeningError::InvalidInterval(name));
    }
    if allow_zero {
        validate_non_negative(bounds.min, name)?;
        validate_non_negative(bounds.max, name)?;
    } else {
        validate_positive(bounds.min, name)?;
        validate_positive(bounds.max, name)?;
    }
    Ok(())
}


/// A declared interval for qualified external-pressure capacity plus audit references.
///
/// Non-empty identifiers establish only metadata completeness: this screen cannot authenticate
/// the artifacts, establish that the allowable is qualified, or validate that the evidence
/// applies.
#[derive(Debug, Clone, PartialEq)]
pub struct PressureCapacityBoundsRecord {
    /// Lower/upper independently sourced allowable external differential pressure, Pa.
    pub allowable_external_pressure_pa: ScalarBounds,
    /// Immutable record identifying the capacity result.
    pub evidence_record_id: String,
    /// Exact model/test profile and configuration reference.
    pub basis_profile_id: String,
    /// Independent check or physical-test record reference.
    pub independent_check_or_test_id: String,
}

/// Pressure-demand intervals plus an optional case-specific capacity evidence declaration.
#[derive(Debug, Clone, PartialEq)]
pub struct ExternalPressureBoundsRequest {
    /// Lower/upper external absolute pressure, Pa.
    pub external_pressure_pa: ScalarBounds,
    /// Lower/upper internal absolute pressure, Pa.
    pub internal_pressure_pa: ScalarBounds,
    pub capacity: Option<PressureCapacityBoundsRecord>,
}

/// Arithmetic comparison status for bounded demand/capacity only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapacityBoundsStatus {
    /// Maximum declared demand is no greater than the minimum declared capacity.
    DeclaredCapacityCoversDemandBounds,
    /// Demand and capacity intervals overlap; endpoint data do not settle the comparison.
    BoundsOverlap,
    /// Minimum positive demand exceeds the maximum declared capacity.
    DeclaredCapacityBelowDemandBounds,
}

/// Result of comparing pressure-demand bounds with declared capacity bounds.
///
/// None of these variants establishes structural safety, model qualification, or airworthiness.
#[derive(Debug, Clone, PartialEq)]
pub enum PressureBoundsAssessment {
    /// The upper bound of external pressure does not exceed the lower bound of internal pressure.
    NoExternalPressureDemand {
        minimum_signed_differential_pa: f64,
        maximum_signed_differential_pa: f64,
    },
    /// Positive external-pressure demand exists but no capacity bounds were supplied.
    CapacityUnknown {
        minimum_positive_demand_pa: f64,
        maximum_positive_demand_pa: f64,
    },
    /// Required audit identifiers are missing from the supplied capacity declaration.
    IncompleteProvenance {
        minimum_positive_demand_pa: f64,
        maximum_positive_demand_pa: f64,
        missing_fields: Vec<&'static str>,
    },
    /// Comparison of declared endpoints only; supplied evidence references remain unauthenticated.
    DeclaredCapacityBoundsComparison {
        minimum_positive_demand_pa: f64,
        maximum_positive_demand_pa: f64,
        minimum_declared_capacity_pa: f64,
        maximum_declared_capacity_pa: f64,
        status: CapacityBoundsStatus,
        evidence_record_id: String,
        basis_profile_id: String,
        independent_check_or_test_id: String,
    },
}

/// Propagate absolute-pressure bounds to a positive differential-demand interval.
///
/// For independent bounded inputs, the signed differential lies in
/// [external.min - internal.max, external.max - internal.min]. Positive demand is the maximum
/// of that differential and zero. This function compares bounds only; it performs no shell
/// analysis and must not turn an unqualified prediction into qualified capacity.
pub fn assess_external_pressure_bounds(
    request: ExternalPressureBoundsRequest,
) -> Result<PressureBoundsAssessment, ScreeningError> {
    validate_bounds(request.external_pressure_pa, "external_pressure_pa", true)?;
    validate_bounds(request.internal_pressure_pa, "internal_pressure_pa", true)?;
    // Validate any supplied capacity interval even when the pressure envelope later shows no
    // external demand; malformed declarations should not be silently ignored.
    if let Some(capacity) = request.capacity.as_ref() {
        validate_bounds(
            capacity.allowable_external_pressure_pa,
            "allowable_external_pressure_pa",
            false,
        )?;
    }

    let minimum_signed_differential_pa =
        request.external_pressure_pa.min - request.internal_pressure_pa.max;
    let maximum_signed_differential_pa =
        request.external_pressure_pa.max - request.internal_pressure_pa.min;
    validate_result(minimum_signed_differential_pa, "minimum pressure differential")?;
    validate_result(maximum_signed_differential_pa, "maximum pressure differential")?;

    let minimum_positive_demand_pa = minimum_signed_differential_pa.max(0.0);
    let maximum_positive_demand_pa = maximum_signed_differential_pa.max(0.0);
    validate_result(minimum_positive_demand_pa, "minimum positive pressure demand")?;
    validate_result(maximum_positive_demand_pa, "maximum positive pressure demand")?;

    if maximum_positive_demand_pa <= 0.0 {
        return Ok(PressureBoundsAssessment::NoExternalPressureDemand {
            minimum_signed_differential_pa,
            maximum_signed_differential_pa,
        });
    }

    let Some(capacity) = request.capacity else {
        return Ok(PressureBoundsAssessment::CapacityUnknown {
            minimum_positive_demand_pa,
            maximum_positive_demand_pa,
        });
    };
    let mut missing_fields = Vec::new();
    if capacity.evidence_record_id.trim().is_empty() {
        missing_fields.push("evidence_record_id");
    }
    if capacity.basis_profile_id.trim().is_empty() {
        missing_fields.push("basis_profile_id");
    }
    if capacity.independent_check_or_test_id.trim().is_empty() {
        missing_fields.push("independent_check_or_test_id");
    }
    if !missing_fields.is_empty() {
        return Ok(PressureBoundsAssessment::IncompleteProvenance {
            minimum_positive_demand_pa,
            maximum_positive_demand_pa,
            missing_fields,
        });
    }

    let status = if maximum_positive_demand_pa
        <= capacity.allowable_external_pressure_pa.min
    {
        CapacityBoundsStatus::DeclaredCapacityCoversDemandBounds
    } else if minimum_positive_demand_pa
        > capacity.allowable_external_pressure_pa.max
    {
        CapacityBoundsStatus::DeclaredCapacityBelowDemandBounds
    } else {
        CapacityBoundsStatus::BoundsOverlap
    };

    Ok(PressureBoundsAssessment::DeclaredCapacityBoundsComparison {
        minimum_positive_demand_pa,
        maximum_positive_demand_pa,
        minimum_declared_capacity_pa: capacity.allowable_external_pressure_pa.min,
        maximum_declared_capacity_pa: capacity.allowable_external_pressure_pa.max,
        status,
        evidence_record_id: capacity.evidence_record_id,
        basis_profile_id: capacity.basis_profile_id,
        independent_check_or_test_id: capacity.independent_check_or_test_id,
    })
}

/// Inputs for deterministic payload bounds at a fixed spherical reference radius.
///
/// Only scalar masses/densities vary over intervals. The radius is held fixed so this
/// calculation does not silently assume payload monotonicity in radius; radius tolerance is a
/// separate geometry-sensitivity study. The returned bounds cover all independent combinations
/// inside the declared intervals. They do not establish structural pressure capacity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SphericalPayloadBoundsRequest {
    /// Fixed reference radius in metres.
    pub radius_m: f64,
    /// Lower/upper ambient-air density, kg/m³.
    pub ambient_air_density_kg_m3: ScalarBounds,
    /// Lower/upper internal gas density, kg/m³; use [0, 0] for ideal vacuum.
    pub internal_gas_density_kg_m3: ScalarBounds,
    /// Lower/upper complete structural-system areal mass, kg/m².
    pub structural_system_areal_mass_kg_m2: ScalarBounds,
    /// Lower/upper fixed non-structural vehicle mass, kg.
    pub fixed_non_structural_vehicle_mass_kg: ScalarBounds,
}

/// Deterministic interval result for mass budgeting, never for pressure-strength assessment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SphericalPayloadBoundsResult {
    pub enclosed_volume_m3: f64,
    pub outer_surface_area_m2: f64,
    /// Conservative lower bound under the supplied independent bounds, kg.
    pub payload_allowance_lower_bound_kg: f64,
    /// Conservative upper bound under the supplied independent bounds, kg.
    pub payload_allowance_upper_bound_kg: f64,
    /// Upper minus lower payload allowance bound, kg.
    pub payload_allowance_interval_width_kg: f64,
    /// Contribution of ambient-density bounds to the payload interval width, kg.
    pub ambient_air_density_interval_contribution_kg: f64,
    /// Contribution of internal-gas density bounds to the payload interval width, kg.
    pub internal_gas_density_interval_contribution_kg: f64,
    /// Contribution of structural areal-mass bounds to the payload interval width, kg.
    pub structural_areal_mass_interval_contribution_kg: f64,
    /// Contribution of fixed vehicle-mass bounds to the payload interval width, kg.
    pub fixed_vehicle_mass_interval_contribution_kg: f64,
    pub mass_budget_status: MassBudgetStatus,
}

/// What the declared mass bounds say—and only what they say—about payload arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MassBudgetStatus {
    /// Every combination within the declared intervals has non-negative payload allowance.
    RobustlyNonNegative,
    /// Some combinations are non-negative and some are negative; additional evidence is needed.
    BoundSensitive,
    /// Every combination within the declared intervals has negative payload allowance.
    NegativeThroughout,
}

/// State of the pressure-capacity evidence after checking case-scope identifiers.
///
/// ScopeMatchedUnverified means identifiers match; referenced artifacts are not authenticated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PressureEvidenceGateState {
    MissingCapacity,
    ScopeIncomplete,
    ScopeMismatch,
    ScopeMatchedUnverified,
}

/// Maturity of the structural analysis for a candidate evidence package.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonlinearAnalysisGateState {
    Missing,
    LinearOrPerfectGeometryOnly,
    NonlinearImperfectionAware,
}

/// Whether a competent independent structural review is recorded for the analysis package.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndependentReviewGateState {
    Missing,
    ReviewNotIndependentOrNotRecorded,
    IndependentlyReviewed,
}

/// Physical validation maturity for the candidate shell architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicalValidationGateState {
    Missing,
    CouponsOrJointsOnly,
    SubscaleShell,
    RepresentativeShell,
}

/// Damage-tolerance evidence maturity within a declared load/operating envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DamageToleranceGateState {
    Missing,
    AnalysisOnly,
    DemonstratedByTestWithinDeclaredEnvelope,
}

/// Maturity of the declared operating envelope and its independent review.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatingEnvelopeGateState {
    Missing,
    DraftOnly,
    DefinedAndIndependentlyReviewed,
}

/// Completeness of the evidence manifest references, not a cryptographic verification result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceManifestGateState {
    Missing,
    IdentifiersOnly,
    ContentAddressed,
}

/// Completeness of the component-level mass ledger and reconciliation against reality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VehicleMassLedgerGateState {
    Incomplete,
    ComponentCompleteUnreconciled,
    ReconciledToMeasuredOrBuildEstimate,
}

/// Evidence that scaling from analyzed specimens to the intended vehicle size is justified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleTransferGateState {
    Unassessed,
    AnalyticalScalingOnly,
    ValidatedAcrossRelevantScaleRange,
}

/// Characterization maturity for the tail of the manufacturing-defect population.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefectPopulationGateState {
    Uncharacterized,
    SingleSpecimenOrSmallSampleOnly,
    CharacterizedForTargetManufacturingPopulation,
}

/// Declared maturity profile used to find blockers before a candidate is promoted to review.
///
/// Every state is supplied by the caller. This function does not inspect files, authenticate
/// hashes, certify reviewers, validate experiments, or qualify a solver. It deliberately uses
/// conjunctive gates rather than an averaged score: one missing critical evidence class cannot be
/// compensated for by a strong payload estimate in another class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CandidateEvidenceGateProfile {
    pub mass_budget: MassBudgetStatus,
    pub pressure_capacity: PressureEvidenceGateState,
    pub nonlinear_analysis: NonlinearAnalysisGateState,
    pub independent_structural_review: IndependentReviewGateState,
    pub physical_validation: PhysicalValidationGateState,
    pub damage_tolerance: DamageToleranceGateState,
    pub operating_envelope: OperatingEnvelopeGateState,
    pub evidence_manifest: EvidenceManifestGateState,
    pub vehicle_mass_ledger: VehicleMassLedgerGateState,
    pub scale_transfer: ScaleTransferGateState,
    pub defect_population: DefectPopulationGateState,
}

/// Fail-closed outcome of the candidate evidence-gap assessment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateEvidenceGateStatus {
    /// Even the optimistic declared payload bound is negative.
    MassBudgetNegativeThroughout,
    /// The payload interval crosses zero; the mass claim is assumption-sensitive.
    MassBudgetBoundSensitive,
    /// One or more critical evidence gates remain open.
    EvidenceGapsRemain,
    /// All declared gates are satisfied; external engineering review is still required.
    EvidenceBundleReadyForIndependentAssessment,
}

/// Evidence gaps and a non-certifying promotion decision for a vacuum-airship candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateEvidenceGateResult {
    pub status: CandidateEvidenceGateStatus,
    /// Stable machine-readable gap identifiers; empty only when all declared gates are satisfied.
    pub blockers: Vec<&'static str>,
}

/// Identify missing evidence gates without converting mixed evidence into a scalar readiness score.
///
/// EvidenceBundleReadyForIndependentAssessment is only a bookkeeping milestone. The inputs are
/// self-declared, the referenced records are not authenticated, and this result must never be read
/// as safe, structurally feasible, flight-ready, or certified.
pub fn assess_candidate_evidence_gates(
    profile: CandidateEvidenceGateProfile,
) -> CandidateEvidenceGateResult {
    let mut blockers = Vec::new();
    match profile.mass_budget {
        MassBudgetStatus::RobustlyNonNegative => {}
        MassBudgetStatus::BoundSensitive => blockers.push("mass_budget_not_robustly_nonnegative"),
        MassBudgetStatus::NegativeThroughout => blockers.push("mass_budget_negative_throughout"),
    }
    if profile.pressure_capacity != PressureEvidenceGateState::ScopeMatchedUnverified {
        blockers.push("pressure_capacity_scope_not_matched_or_missing");
    }
    if profile.nonlinear_analysis != NonlinearAnalysisGateState::NonlinearImperfectionAware {
        blockers.push("nonlinear_imperfection_analysis_missing_or_insufficient");
    }
    if profile.independent_structural_review != IndependentReviewGateState::IndependentlyReviewed {
        blockers.push("independent_structural_review_missing");
    }
    if profile.physical_validation != PhysicalValidationGateState::RepresentativeShell {
        blockers.push("representative_shell_validation_missing");
    }
    if profile.damage_tolerance
        != DamageToleranceGateState::DemonstratedByTestWithinDeclaredEnvelope
    {
        blockers.push("damage_tolerance_not_demonstrated_by_test");
    }
    if profile.operating_envelope != OperatingEnvelopeGateState::DefinedAndIndependentlyReviewed {
        blockers.push("operating_envelope_missing_or_unreviewed");
    }
    if profile.evidence_manifest != EvidenceManifestGateState::ContentAddressed {
        blockers.push("evidence_manifest_not_content_addressed");
    }
    if profile.vehicle_mass_ledger
        != VehicleMassLedgerGateState::ReconciledToMeasuredOrBuildEstimate
    {
        blockers.push("vehicle_mass_ledger_not_reconciled");
    }
    if profile.scale_transfer != ScaleTransferGateState::ValidatedAcrossRelevantScaleRange {
        blockers.push("scale_transfer_not_validated");
    }
    if profile.defect_population
        != DefectPopulationGateState::CharacterizedForTargetManufacturingPopulation
    {
        blockers.push("manufacturing_defect_population_not_characterized");
    }

    let status = match profile.mass_budget {
        MassBudgetStatus::NegativeThroughout => {
            CandidateEvidenceGateStatus::MassBudgetNegativeThroughout
        }
        MassBudgetStatus::BoundSensitive => CandidateEvidenceGateStatus::MassBudgetBoundSensitive,
        MassBudgetStatus::RobustlyNonNegative if blockers.is_empty() => {
            CandidateEvidenceGateStatus::EvidenceBundleReadyForIndependentAssessment
        }
        MassBudgetStatus::RobustlyNonNegative => CandidateEvidenceGateStatus::EvidenceGapsRemain,
    };

    CandidateEvidenceGateResult { status, blockers }
}

/// Propagate deterministic input bounds through the spherical payload equation.
///
/// The lower result uses minimum ambient density, maximum interior density, maximum structural
/// areal mass, and maximum fixed mass. The upper result uses the opposite endpoints. These are
/// conservative arithmetic bounds for the fixed radius; they are not a probability distribution,
/// a structural model, a capacity prediction, or a flight-readiness decision.
pub fn screen_spherical_payload_bounds(
    request: SphericalPayloadBoundsRequest,
) -> Result<SphericalPayloadBoundsResult, ScreeningError> {
    validate_positive(request.radius_m, "radius_m")?;
    validate_bounds(
        request.ambient_air_density_kg_m3,
        "ambient_air_density_kg_m3",
        false,
    )?;
    validate_bounds(
        request.internal_gas_density_kg_m3,
        "internal_gas_density_kg_m3",
        true,
    )?;
    validate_bounds(
        request.structural_system_areal_mass_kg_m2,
        "structural_system_areal_mass_kg_m2",
        true,
    )?;
    validate_bounds(
        request.fixed_non_structural_vehicle_mass_kg,
        "fixed_non_structural_vehicle_mass_kg",
        true,
    )?;

    let enclosed_volume_m3 = sphere_volume_m3(request.radius_m)?;
    let outer_surface_area_m2 = 4.0 * std::f64::consts::PI * request.radius_m.powi(2);
    validate_result(outer_surface_area_m2, "outer surface area")?;

    let payload_allowance_lower_bound_kg =
        (request.ambient_air_density_kg_m3.min
            - request.internal_gas_density_kg_m3.max)
            * enclosed_volume_m3
            - request.structural_system_areal_mass_kg_m2.max * outer_surface_area_m2
            - request.fixed_non_structural_vehicle_mass_kg.max;
    let payload_allowance_upper_bound_kg =
        (request.ambient_air_density_kg_m3.max
            - request.internal_gas_density_kg_m3.min)
            * enclosed_volume_m3
            - request.structural_system_areal_mass_kg_m2.min * outer_surface_area_m2
            - request.fixed_non_structural_vehicle_mass_kg.min;
    validate_result(
        payload_allowance_lower_bound_kg,
        "payload allowance lower bound",
    )?;
    validate_result(
        payload_allowance_upper_bound_kg,
        "payload allowance upper bound",
    )?;

    let payload_allowance_interval_width_kg =
        payload_allowance_upper_bound_kg - payload_allowance_lower_bound_kg;
    validate_result(
        payload_allowance_interval_width_kg,
        "payload allowance interval width",
    )?;
    if payload_allowance_interval_width_kg < 0.0 {
        return Err(ScreeningError::NonFiniteResult(
            "payload allowance interval ordering",
        ));
    }

    // The payload equation is affine in these four bounded inputs for a fixed radius.
    // These terms partition the total interval width, helping prioritize measurements.
    let ambient_air_density_interval_contribution_kg =
        (request.ambient_air_density_kg_m3.max - request.ambient_air_density_kg_m3.min)
            * enclosed_volume_m3;
    let internal_gas_density_interval_contribution_kg =
        (request.internal_gas_density_kg_m3.max - request.internal_gas_density_kg_m3.min)
            * enclosed_volume_m3;
    let structural_areal_mass_interval_contribution_kg =
        (request.structural_system_areal_mass_kg_m2.max
            - request.structural_system_areal_mass_kg_m2.min)
            * outer_surface_area_m2;
    let fixed_vehicle_mass_interval_contribution_kg =
        request.fixed_non_structural_vehicle_mass_kg.max
            - request.fixed_non_structural_vehicle_mass_kg.min;

    for (name, value) in [
        (
            "ambient density interval contribution",
            ambient_air_density_interval_contribution_kg,
        ),
        (
            "internal gas density interval contribution",
            internal_gas_density_interval_contribution_kg,
        ),
        (
            "structural areal mass interval contribution",
            structural_areal_mass_interval_contribution_kg,
        ),
        (
            "fixed vehicle mass interval contribution",
            fixed_vehicle_mass_interval_contribution_kg,
        ),
    ] {
        validate_result(value, name)?;
        if value < 0.0 {
            return Err(ScreeningError::NonFiniteResult(name));
        }
    }

    let mass_budget_status = if payload_allowance_lower_bound_kg >= 0.0 {
        MassBudgetStatus::RobustlyNonNegative
    } else if payload_allowance_upper_bound_kg < 0.0 {
        MassBudgetStatus::NegativeThroughout
    } else {
        MassBudgetStatus::BoundSensitive
    };

    Ok(SphericalPayloadBoundsResult {
        enclosed_volume_m3,
        outer_surface_area_m2,
        payload_allowance_lower_bound_kg,
        payload_allowance_upper_bound_kg,
        payload_allowance_interval_width_kg,
        ambient_air_density_interval_contribution_kg,
        internal_gas_density_interval_contribution_kg,
        structural_areal_mass_interval_contribution_kg,
        fixed_vehicle_mass_interval_contribution_kg,
        mass_budget_status,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_mass() -> VehicleMassBreakdown {
        VehicleMassBreakdown::default()
    }

    fn qualified_record(capacity_pa: f64) -> PressureCapacityRecord {
        PressureCapacityRecord {
            allowable_external_pressure_pa: capacity_pa,
            evidence_record_id: "evidence://pressure-result/42".into(),
            basis_profile_id: "profile://nonlinear-imperfect-shell/v1".into(),
            independent_check_or_test_id: "evidence://independent-check/7".into(),
        }
    }

    #[test]
    fn unit_sphere_volume_matches_analytic_reference() {
        let volume = sphere_volume_m3(1.0).unwrap();
        assert!((volume - 4.0 * std::f64::consts::PI / 3.0).abs() < 1e-12);
    }

    #[test]
    fn twenty_metre_radius_vacuum_sphere_has_expected_gross_lift() {
        let volume = sphere_volume_m3(20.0).unwrap();
        let result = screen_buoyancy(BuoyancyRequest {
            enclosed_volume_m3: volume,
            ambient_air_density_kg_m3: 1.225,
            internal_gas_density_kg_m3: 0.0,
            vehicle_mass: zero_mass(),
        })
        .unwrap();
        assert!((result.net_buoyant_mass_kg - 41_050.14400690663).abs() < 1e-6);
        assert_eq!(result.internal_gas_mass_kg, 0.0);
        assert_eq!(result.payload_allowance_kg, result.net_buoyant_mass_kg);
    }

    #[test]
    fn twenty_metre_payload_allowance_subtracts_every_explicit_mass() {
        let volume = sphere_volume_m3(20.0).unwrap();
        let mass = VehicleMassBreakdown {
            envelope_and_seals_kg: 10.0,
            frame_lattice_and_joints_kg: 20.0,
            propulsion_kg: 30.0,
            energy_storage_kg: 40.0,
            avionics_and_communications_kg: 5.0,
            other_systems_kg: 5.0,
        };
        let result = screen_buoyancy(BuoyancyRequest {
            enclosed_volume_m3: volume,
            ambient_air_density_kg_m3: 1.225,
            internal_gas_density_kg_m3: 0.0,
            vehicle_mass: mass,
        })
        .unwrap();
        assert_eq!(result.non_payload_vehicle_mass_kg, 110.0);
        assert!((result.payload_allowance_kg - (41_050.14400690663 - 110.0)).abs() < 1e-6);
    }

    #[test]
    fn heavier_than_buoyancy_vehicle_keeps_negative_payload_allowance() {
        let result = screen_buoyancy(BuoyancyRequest {
            enclosed_volume_m3: 1.0,
            ambient_air_density_kg_m3: 1.0,
            internal_gas_density_kg_m3: 0.0,
            vehicle_mass: VehicleMassBreakdown {
                frame_lattice_and_joints_kg: 2.0,
                ..zero_mass()
            },
        })
        .unwrap();
        assert_eq!(result.payload_allowance_kg, -1.0);
        assert_eq!(result.payload_fraction_of_net_buoyant_mass, Some(-1.0));
    }

    #[test]
    fn nonzero_internal_gas_mass_reduces_net_lift() {
        let vacuum = screen_buoyancy(BuoyancyRequest {
            enclosed_volume_m3: 100.0,
            ambient_air_density_kg_m3: 1.225,
            internal_gas_density_kg_m3: 0.0,
            vehicle_mass: zero_mass(),
        })
        .unwrap();
        let helium_like = screen_buoyancy(BuoyancyRequest {
            enclosed_volume_m3: 100.0,
            ambient_air_density_kg_m3: 1.225,
            internal_gas_density_kg_m3: 0.169_280_218_717_622_25,
            vehicle_mass: zero_mass(),
        })
        .unwrap();
        assert!(helium_like.net_buoyant_mass_kg < vacuum.net_buoyant_mass_kg);
    }

    #[test]
    fn negative_mass_is_rejected() {
        let result = screen_buoyancy(BuoyancyRequest {
            enclosed_volume_m3: 1.0,
            ambient_air_density_kg_m3: 1.0,
            internal_gas_density_kg_m3: 0.0,
            vehicle_mass: VehicleMassBreakdown {
                propulsion_kg: -1.0,
                ..zero_mass()
            },
        });
        assert_eq!(
            result.unwrap_err(),
            ScreeningError::MustBeNonNegative("propulsion_kg")
        );
    }

    #[test]
    fn nan_density_is_rejected() {
        let result = screen_buoyancy(BuoyancyRequest {
            enclosed_volume_m3: 1.0,
            ambient_air_density_kg_m3: f64::NAN,
            internal_gas_density_kg_m3: 0.0,
            vehicle_mass: zero_mass(),
        });
        assert_eq!(
            result.unwrap_err(),
            ScreeningError::NonFiniteInput("ambient_air_density_kg_m3")
        );
    }

    #[test]
    fn twenty_metre_vacuum_sphere_has_8_1667_kg_m2_envelope_ceiling_before_other_mass() {
        let result = screen_spherical_envelope_budget(SphericalEnvelopeBudgetRequest {
            radius_m: 20.0,
            ambient_air_density_kg_m3: 1.225,
            internal_gas_density_kg_m3: 0.0,
            fixed_non_envelope_vehicle_mass_kg: 0.0,
            target_payload_mass_kg: 0.0,
        })
        .unwrap();

        // For a spherical shell with no other mass or payload:
        // (rho * 4/3*pi*R^3) / (4*pi*R^2) = rho*R/3.
        assert!(
            (result.maximum_envelope_areal_density_kg_m2 - (1.225 * 20.0 / 3.0)).abs() < 1e-12
        );
    }

    #[test]
    fn fixed_vehicle_mass_and_target_payload_reduce_envelope_budget() {
        let result = screen_spherical_envelope_budget(SphericalEnvelopeBudgetRequest {
            radius_m: 20.0,
            ambient_air_density_kg_m3: 1.225,
            internal_gas_density_kg_m3: 0.0,
            fixed_non_envelope_vehicle_mass_kg: 10_000.0,
            target_payload_mass_kg: 5_000.0,
        })
        .unwrap();

        assert!(
            (result.maximum_envelope_mass_kg - (41_050.14400690663 - 15_000.0)).abs() < 1e-6
        );
        assert!(result.maximum_envelope_areal_density_kg_m2 < 8.166666666666667);
    }

    #[test]
    fn envelope_budget_preserves_infeasible_negative_mass_ceiling() {
        let result = screen_spherical_envelope_budget(SphericalEnvelopeBudgetRequest {
            radius_m: 1.0,
            ambient_air_density_kg_m3: 1.0,
            internal_gas_density_kg_m3: 0.0,
            fixed_non_envelope_vehicle_mass_kg: 100.0,
            target_payload_mass_kg: 1.0,
        })
        .unwrap();

        assert!(result.maximum_envelope_mass_kg < 0.0);
        assert!(result.maximum_envelope_areal_density_kg_m2 < 0.0);
    }

    #[test]
    fn negative_target_payload_is_rejected() {
        let result = screen_spherical_envelope_budget(SphericalEnvelopeBudgetRequest {
            radius_m: 1.0,
            ambient_air_density_kg_m3: 1.0,
            internal_gas_density_kg_m3: 0.0,
            fixed_non_envelope_vehicle_mass_kg: 0.0,
            target_payload_mass_kg: -1.0,
        });

        assert_eq!(
            result.unwrap_err(),
            ScreeningError::MustBeNonNegative("target_payload_mass_kg")
        );
    }

    #[test]
    fn payload_screen_uses_declared_structural_areal_mass_without_claiming_capacity() {
        let result = screen_spherical_payload(SphericalPayloadRequest {
            radius_m: 20.0,
            ambient_air_density_kg_m3: 1.225,
            internal_gas_density_kg_m3: 0.0,
            structural_system_areal_mass_kg_m2: 4.0,
            fixed_non_structural_vehicle_mass_kg: 10_000.0,
        })
        .unwrap();

        assert!(
            (result.assumed_structural_system_mass_kg
                - 4.0 * 4.0 * std::f64::consts::PI * 20.0_f64.powi(2))
                .abs()
                < 1e-8
        );
        assert!(
            (result.payload_allowance_kg
                - (1.225 * (4.0 / 3.0) * std::f64::consts::PI * 20.0_f64.powi(3)
                    - 4.0 * 4.0 * std::f64::consts::PI * 20.0_f64.powi(2)
                    - 10_000.0))
                .abs()
                < 1e-8
        );
        assert!(result.payload_allowance_non_negative);
    }

    #[test]
    fn payload_screen_preserves_negative_allowance() {
        let result = screen_spherical_payload(SphericalPayloadRequest {
            radius_m: 1.0,
            ambient_air_density_kg_m3: 1.0,
            internal_gas_density_kg_m3: 0.0,
            structural_system_areal_mass_kg_m2: 1.0,
            fixed_non_structural_vehicle_mass_kg: 0.0,
        })
        .unwrap();
        assert!(result.payload_allowance_kg < 0.0);
        assert!(!result.payload_allowance_non_negative);
    }

    #[test]
    fn twenty_metre_sphere_gains_about_5_67_tonnes_gross_lift_over_helium() {
        let volume = sphere_volume_m3(20.0).unwrap();
        let result = compare_vacuum_and_helium(VacuumHeliumComparisonRequest {
            enclosed_volume_m3: volume,
            ambient_air_density_kg_m3: 1.225,
            helium_density_kg_m3: 0.169_280_218_717_622_25,
            vacuum_vehicle_mass_kg: 0.0,
            helium_vehicle_mass_kg: 0.0,
        })
        .unwrap();
        assert!((result.vacuum_gross_lift_advantage_kg - 5_672.6345762277915).abs() < 1e-6);
        assert!(result.vacuum_has_higher_payload_allowance);
    }

    #[test]
    fn vacuum_loses_payload_advantage_when_extra_mass_exceeds_helium_mass() {
        let volume = sphere_volume_m3(20.0).unwrap();
        let helium_density = 0.169_280_218_717_622_25;
        let helium_mass = 20_000.0;
        let vacuum_mass = helium_mass + helium_density * volume + 1.0;
        let result = compare_vacuum_and_helium(VacuumHeliumComparisonRequest {
            enclosed_volume_m3: volume,
            ambient_air_density_kg_m3: 1.225,
            helium_density_kg_m3: helium_density,
            vacuum_vehicle_mass_kg: vacuum_mass,
            helium_vehicle_mass_kg: helium_mass,
        })
        .unwrap();
        assert!((result.vacuum_minus_helium_payload_kg + 1.0).abs() < 1e-6);
        assert!(!result.vacuum_has_higher_payload_allowance);
    }

    #[test]
    fn cargo_fit_keeps_mass_and_volume_gates_separate() {
        let result = screen_cargo_fit(CargoFitRequest {
            payload_allowance_kg: 12_000.0,
            target_cargo_mass_kg: 10_000.0,
            target_cargo_volume_m3: 100.0,
            usable_cargo_volume_m3: 80.0,
        })
        .unwrap();
        assert!(result.cargo_fits_mass);
        assert!(!result.cargo_fits_volume);
        assert!(!result.cargo_fits_declared_mass_and_volume);
        assert_eq!(result.cargo_volume_margin_m3, -20.0);
    }

    #[test]
    fn cargo_mass_fit_rejects_infeasible_negative_payload_allowance() {
        let result = screen_cargo_fit(CargoFitRequest {
            payload_allowance_kg: -1.0,
            target_cargo_mass_kg: 0.0,
            target_cargo_volume_m3: 0.0,
            usable_cargo_volume_m3: 1.0,
        })
        .unwrap();
        assert!(!result.cargo_fits_mass);
        assert!(result.cargo_fits_volume);
        assert!(!result.cargo_fits_declared_mass_and_volume);
    }

    #[test]
    fn invalid_cargo_volume_is_rejected() {
        let result = screen_cargo_fit(CargoFitRequest {
            payload_allowance_kg: 10.0,
            target_cargo_mass_kg: 1.0,
            target_cargo_volume_m3: 2.0,
            usable_cargo_volume_m3: f64::INFINITY,
        });
        assert_eq!(
            result.unwrap_err(),
            ScreeningError::NonFiniteInput("usable_cargo_volume_m3")
        );
    }

    fn recovery_observation() -> ElasticRecoveryObservation {
        ElasticRecoveryObservation {
            peak_deformation_rms_m: 0.010,
            residual_deformation_rms_m: 0.0001,
            max_residual_surface_deviation_m: 0.0002,
            metrology_uncertainty_m: 0.00005,
            pre_cycle_stiffness_n_per_m: 100_000.0,
            post_cycle_stiffness_n_per_m: 98_000.0,
            cycle_count: 100,
        }
    }

    fn recovery_criteria() -> ElasticRecoveryCriteria {
        ElasticRecoveryCriteria {
            max_residual_surface_deviation_m: 0.0005,
            min_stiffness_retention_fraction: 0.95,
        }
    }

    #[test]
    fn elastic_recovery_passes_only_declared_geometry_and_stiffness_gates() {
        let result =
            assess_elastic_recovery(recovery_observation(), recovery_criteria()).unwrap();
        assert!((result.shape_recovery_fraction - 0.99).abs() < 1e-12);
        assert!((result.stiffness_retention_fraction - 0.98).abs() < 1e-12);
        assert_eq!(result.shape_recovery_gate, RecoveryGate::Pass);
        assert_eq!(result.stiffness_retention_gate, RecoveryGate::Pass);
        assert_eq!(result.overall_recovery_gate, RecoveryGate::Pass);
    }

    #[test]
    fn recovery_shape_gate_is_indeterminate_when_metrology_overlaps_tolerance() {
        let observation = ElasticRecoveryObservation {
            max_residual_surface_deviation_m: 0.0005,
            metrology_uncertainty_m: 0.0001,
            ..recovery_observation()
        };
        let result = assess_elastic_recovery(observation, recovery_criteria()).unwrap();
        assert_eq!(result.shape_recovery_gate, RecoveryGate::Indeterminate);
        assert_eq!(result.overall_recovery_gate, RecoveryGate::Indeterminate);
    }

    #[test]
    fn recovered_shape_cannot_hide_stiffness_loss() {
        let observation = ElasticRecoveryObservation {
            post_cycle_stiffness_n_per_m: 80_000.0,
            ..recovery_observation()
        };
        let result = assess_elastic_recovery(observation, recovery_criteria()).unwrap();
        assert_eq!(result.shape_recovery_gate, RecoveryGate::Pass);
        assert_eq!(result.stiffness_retention_gate, RecoveryGate::Fail);
        assert_eq!(result.overall_recovery_gate, RecoveryGate::Fail);
    }

    #[test]
    fn residual_deformation_larger_than_peak_keeps_negative_recovery_metric() {
        let observation = ElasticRecoveryObservation {
            residual_deformation_rms_m: 0.020,
            ..recovery_observation()
        };
        let result = assess_elastic_recovery(observation, recovery_criteria()).unwrap();
        assert!(result.shape_recovery_fraction < 0.0);
    }

    #[test]
    fn recovery_test_rejects_zero_peak_deformation() {
        let observation = ElasticRecoveryObservation {
            peak_deformation_rms_m: 0.0,
            ..recovery_observation()
        };
        assert_eq!(
            assess_elastic_recovery(observation, recovery_criteria()).unwrap_err(),
            ScreeningError::MustBePositive("peak_deformation_rms_m")
        );
    }

    #[test]
    fn recovery_test_rejects_stiffness_retention_threshold_above_one() {
        let criteria = ElasticRecoveryCriteria {
            min_stiffness_retention_fraction: 1.01,
            ..recovery_criteria()
        };
        assert_eq!(
            assess_elastic_recovery(recovery_observation(), criteria).unwrap_err(),
            ScreeningError::FractionOutOfRange("min_stiffness_retention_fraction")
        );
    }

    #[test]
    fn recovery_test_rejects_zero_cycles() {
        let observation = ElasticRecoveryObservation {
            cycle_count: 0,
            ..recovery_observation()
        };
        assert_eq!(
            assess_elastic_recovery(observation, recovery_criteria()).unwrap_err(),
            ScreeningError::MustBePositive("cycle_count")
        );
    }

    #[test]
    fn twenty_metre_radius_sea_level_pressure_cut_load_is_about_127_meganewtons() {
        let result = screen_spherical_pressure_load(SphericalPressureLoadRequest {
            radius_m: 20.0,
            external_pressure_pa: 101_325.0,
            internal_pressure_pa: 0.0,
        })
        .unwrap();
        assert!(
            (result.great_circle_compressive_resultant_n - 127_328_750.249_994_31).abs() < 1e-6
        );
    }

    #[test]
    fn pressure_resultant_scales_with_radius_squared() {
        let small = screen_spherical_pressure_load(SphericalPressureLoadRequest {
            radius_m: 20.0,
            external_pressure_pa: 101_325.0,
            internal_pressure_pa: 0.0,
        })
        .unwrap();
        let large = screen_spherical_pressure_load(SphericalPressureLoadRequest {
            radius_m: 40.0,
            external_pressure_pa: 101_325.0,
            internal_pressure_pa: 0.0,
        })
        .unwrap();
        assert!(
            (large.great_circle_compressive_resultant_n
                / small.great_circle_compressive_resultant_n
                - 4.0)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn positive_internal_overpressure_produces_no_external_compressive_resultant() {
        let result = screen_spherical_pressure_load(SphericalPressureLoadRequest {
            radius_m: 20.0,
            external_pressure_pa: 90_000.0,
            internal_pressure_pa: 101_325.0,
        })
        .unwrap();
        assert_eq!(result.positive_external_differential_pa, 0.0);
        assert_eq!(result.great_circle_compressive_resultant_n, 0.0);
        assert_eq!(result.differential_pressure_pa, -11_325.0);
    }

    #[test]
    fn absent_pressure_capacity_stays_unknown() {
        let assessment = assess_external_pressure(ExternalPressureRequest {
            external_pressure_pa: 101_325.0,
            internal_pressure_pa: 0.0,
            capacity: None,
        })
        .unwrap();
        assert_eq!(
            assessment,
            PressureAssessment::CapacityUnknown {
                differential_pressure_pa: 101_325.0
            }
        );
    }

    #[test]
    fn incomplete_pressure_provenance_blocks_ratio() {
        let assessment = assess_external_pressure(ExternalPressureRequest {
            external_pressure_pa: 101_325.0,
            internal_pressure_pa: 0.0,
            capacity: Some(PressureCapacityRecord {
                allowable_external_pressure_pa: 250_000.0,
                evidence_record_id: String::new(),
                basis_profile_id: "profile/v1".into(),
                independent_check_or_test_id: "check/1".into(),
            }),
        })
        .unwrap();
        assert_eq!(
            assessment,
            PressureAssessment::IncompleteProvenance {
                differential_pressure_pa: 101_325.0,
                missing_fields: vec!["evidence_record_id"]
            }
        );
    }

    #[test]
    fn declared_pressure_capacity_ratio_is_arithmetic_not_a_pass_flag() {
        let assessment = assess_external_pressure(ExternalPressureRequest {
            external_pressure_pa: 101_325.0,
            internal_pressure_pa: 0.0,
            capacity: Some(qualified_record(202_650.0)),
        })
        .unwrap();
        match assessment {
            PressureAssessment::DeclaredCapacityComparison {
                declared_capacity_ratio,
                ..
            } => assert!((declared_capacity_ratio - 2.0).abs() < 1e-12),
            other => panic!("unexpected assessment: {other:?}"),
        }
    }

    #[test]
    fn pressure_case_without_external_differential_does_not_claim_capacity() {
        let assessment = assess_external_pressure(ExternalPressureRequest {
            external_pressure_pa: 90_000.0,
            internal_pressure_pa: 101_325.0,
            capacity: Some(qualified_record(202_650.0)),
        })
        .unwrap();
        assert_eq!(
            assessment,
            PressureAssessment::NoExternalPressureDifferential {
                differential_pressure_pa: -11_325.0
            }
        );
    }

    #[test]
    fn zero_radius_is_rejected() {
        assert_eq!(
            sphere_volume_m3(0.0).unwrap_err(),
            ScreeningError::MustBePositive("radius_m")
        );
    }


    fn bounded_payload_request() -> SphericalPayloadBoundsRequest {
        SphericalPayloadBoundsRequest {
            radius_m: 20.0,
            ambient_air_density_kg_m3: ScalarBounds {
                min: 1.20,
                max: 1.25,
            },
            internal_gas_density_kg_m3: ScalarBounds {
                min: 0.0,
                max: 0.01,
            },
            structural_system_areal_mass_kg_m2: ScalarBounds {
                min: 4.0,
                max: 5.0,
            },
            fixed_non_structural_vehicle_mass_kg: ScalarBounds {
                min: 14_000.0,
                max: 18_000.0,
            },
        }
    }

    #[test]
    fn bounded_payload_screen_preserves_optimistic_and_worst_case_bounds() {
        let result = screen_spherical_payload_bounds(bounded_payload_request()).unwrap();
        assert!(result.payload_allowance_lower_bound_kg < 0.0);
        assert!(result.payload_allowance_upper_bound_kg > 0.0);
        assert!(result.payload_allowance_interval_width_kg > 0.0);
        let contribution_sum = result.ambient_air_density_interval_contribution_kg
            + result.internal_gas_density_interval_contribution_kg
            + result.structural_areal_mass_interval_contribution_kg
            + result.fixed_vehicle_mass_interval_contribution_kg;
        assert!((contribution_sum - result.payload_allowance_interval_width_kg).abs() < 1e-7);
        assert_eq!(result.mass_budget_status, MassBudgetStatus::BoundSensitive);
    }

    #[test]
    fn bounded_payload_screen_identifies_robustly_non_negative_mass_budget() {
        let request = SphericalPayloadBoundsRequest {
            fixed_non_structural_vehicle_mass_kg: ScalarBounds {
                min: 1_000.0,
                max: 2_000.0,
            },
            ..bounded_payload_request()
        };
        let result = screen_spherical_payload_bounds(request).unwrap();
        assert!(result.payload_allowance_lower_bound_kg >= 0.0);
        assert_eq!(result.mass_budget_status, MassBudgetStatus::RobustlyNonNegative);
    }

    #[test]
    fn bounded_payload_screen_identifies_negative_budget_throughout_bounds() {
        let request = SphericalPayloadBoundsRequest {
            fixed_non_structural_vehicle_mass_kg: ScalarBounds {
                min: 100_000.0,
                max: 120_000.0,
            },
            ..bounded_payload_request()
        };
        let result = screen_spherical_payload_bounds(request).unwrap();
        assert!(result.payload_allowance_upper_bound_kg < 0.0);
        assert_eq!(result.mass_budget_status, MassBudgetStatus::NegativeThroughout);
    }

    #[test]
    fn bounded_payload_screen_rejects_reversed_bounds() {
        let request = SphericalPayloadBoundsRequest {
            ambient_air_density_kg_m3: ScalarBounds {
                min: 1.3,
                max: 1.2,
            },
            ..bounded_payload_request()
        };
        assert_eq!(
            screen_spherical_payload_bounds(request).unwrap_err(),
            ScreeningError::InvalidInterval("ambient_air_density_kg_m3")
        );
    }

    #[test]
    fn bounded_payload_screen_rejects_non_finite_interval_endpoints() {
        let request = SphericalPayloadBoundsRequest {
            structural_system_areal_mass_kg_m2: ScalarBounds {
                min: 1.0,
                max: f64::INFINITY,
            },
            ..bounded_payload_request()
        };
        assert_eq!(
            screen_spherical_payload_bounds(request).unwrap_err(),
            ScreeningError::NonFiniteInput("structural_system_areal_mass_kg_m2")
        );
    }


    #[test]
    fn payload_interval_reports_each_input_uncertainty_contribution() {
        let result = screen_spherical_payload_bounds(bounded_payload_request()).unwrap();
        assert!(result.ambient_air_density_interval_contribution_kg > 0.0);
        assert!(result.internal_gas_density_interval_contribution_kg > 0.0);
        assert!(result.structural_areal_mass_interval_contribution_kg > 0.0);
        assert!(result.fixed_vehicle_mass_interval_contribution_kg > 0.0);
        let sum = result.ambient_air_density_interval_contribution_kg
            + result.internal_gas_density_interval_contribution_kg
            + result.structural_areal_mass_interval_contribution_kg
            + result.fixed_vehicle_mass_interval_contribution_kg;
        assert!((sum - result.payload_allowance_interval_width_kg).abs() < 1e-7);
    }


    #[test]
    fn load_path_screen_reproduces_declared_reference_scaling() {
        let pressure = screen_spherical_pressure_load(SphericalPressureLoadRequest {
            radius_m: 20.0,
            external_pressure_pa: 101_325.0,
            internal_pressure_pa: 0.0,
        })
        .unwrap();
        let frequency = 3.0;
        let effective_paths = 30.0 * frequency / std::f64::consts::PI;
        let result = screen_load_path_demand(LoadPathDemandRequest {
            total_compressive_resultant_n: pressure.great_circle_compressive_resultant_n,
            effective_load_path_count: effective_paths,
            assumed_peak_to_mean_load_factor: 1.32,
        })
        .unwrap();
        assert!(
            (result.mean_compressive_load_per_effective_path_n - 4_444_611.848623908).abs()
                < 1e-6
        );
        assert!(
            (result.factor_scaled_peak_load_path_demand_n - 5_866_887.640183559).abs()
                < 1e-6
        );
        assert_eq!(result.assumed_peak_to_mean_load_factor, 1.32);
    }

    #[test]
    fn load_path_screen_equal_share_conserves_total_resultant() {
        let result = screen_load_path_demand(LoadPathDemandRequest {
            total_compressive_resultant_n: 9_000.0,
            effective_load_path_count: 12.0,
            assumed_peak_to_mean_load_factor: 1.0,
        })
        .unwrap();
        assert!((result.mean_compressive_load_per_effective_path_n - 750.0).abs() < 1e-12);
        assert!(
            (result.mean_compressive_load_per_effective_path_n
                * result.effective_load_path_count
                - result.total_compressive_resultant_n)
                .abs()
                < 1e-9
        );
        assert_eq!(result.factor_scaled_peak_load_path_demand_n, 750.0);
    }

    #[test]
    fn load_path_screen_accepts_zero_resultant_without_hiding_path_inputs() {
        let result = screen_load_path_demand(LoadPathDemandRequest {
            total_compressive_resultant_n: 0.0,
            effective_load_path_count: 4.5,
            assumed_peak_to_mean_load_factor: 1.4,
        })
        .unwrap();
        assert_eq!(result.mean_compressive_load_per_effective_path_n, 0.0);
        assert_eq!(result.factor_scaled_peak_load_path_demand_n, 0.0);
    }

    #[test]
    fn load_path_screen_rejects_zero_path_count() {
        let result = screen_load_path_demand(LoadPathDemandRequest {
            total_compressive_resultant_n: 1.0,
            effective_load_path_count: 0.0,
            assumed_peak_to_mean_load_factor: 1.0,
        });
        assert_eq!(
            result.unwrap_err(),
            ScreeningError::MustBePositive("effective_load_path_count")
        );
    }

    #[test]
    fn load_path_screen_rejects_factor_below_one() {
        let result = screen_load_path_demand(LoadPathDemandRequest {
            total_compressive_resultant_n: 1.0,
            effective_load_path_count: 1.0,
            assumed_peak_to_mean_load_factor: 0.99,
        });
        assert_eq!(
            result.unwrap_err(),
            ScreeningError::FactorBelowOne("assumed_peak_to_mean_load_factor")
        );
    }

    #[test]
    fn load_path_screen_rejects_non_finite_multiplier() {
        let result = screen_load_path_demand(LoadPathDemandRequest {
            total_compressive_resultant_n: 1.0,
            effective_load_path_count: 1.0,
            assumed_peak_to_mean_load_factor: f64::INFINITY,
        });
        assert_eq!(
            result.unwrap_err(),
            ScreeningError::NonFiniteInput("assumed_peak_to_mean_load_factor")
        );
    }

    #[test]
    fn load_path_screen_rejects_derived_overflow() {
        let result = screen_load_path_demand(LoadPathDemandRequest {
            total_compressive_resultant_n: f64::MAX,
            effective_load_path_count: 0.5,
            assumed_peak_to_mean_load_factor: 2.0,
        });
        assert_eq!(
            result.unwrap_err(),
            ScreeningError::NonFiniteResult("mean compressive load per effective path")
        );
    }

    #[test]
    fn bounded_load_path_screen_propagates_all_declared_input_bounds() {
        let result = screen_bounded_load_path_demand(BoundedLoadPathDemandRequest {
            total_compressive_resultant_n: ScalarBounds {
                min: 100_000.0,
                max: 120_000.0,
            },
            effective_load_path_count: ScalarBounds { min: 10.0, max: 20.0 },
            peak_to_mean_load_factor: ScalarBounds { min: 1.0, max: 1.5 },
        })
        .unwrap();
        assert_eq!(result.mean_compressive_load_per_effective_path_lower_bound_n, 5_000.0);
        assert_eq!(result.mean_compressive_load_per_effective_path_upper_bound_n, 12_000.0);
        assert_eq!(result.factor_scaled_load_path_demand_lower_bound_n, 5_000.0);
        assert_eq!(result.factor_scaled_load_path_demand_upper_bound_n, 18_000.0);
    }

    #[test]
    fn bounded_load_path_equal_intervals_match_scalar_screen() {
        let scalar = screen_load_path_demand(LoadPathDemandRequest {
            total_compressive_resultant_n: 90_000.0,
            effective_load_path_count: 15.0,
            assumed_peak_to_mean_load_factor: 1.4,
        })
        .unwrap();
        let bounded = screen_bounded_load_path_demand(BoundedLoadPathDemandRequest {
            total_compressive_resultant_n: ScalarBounds { min: 90_000.0, max: 90_000.0 },
            effective_load_path_count: ScalarBounds { min: 15.0, max: 15.0 },
            peak_to_mean_load_factor: ScalarBounds { min: 1.4, max: 1.4 },
        })
        .unwrap();
        assert_eq!(
            bounded.mean_compressive_load_per_effective_path_lower_bound_n,
            scalar.mean_compressive_load_per_effective_path_n
        );
        assert_eq!(
            bounded.factor_scaled_load_path_demand_upper_bound_n,
            scalar.factor_scaled_peak_load_path_demand_n
        );
    }

    #[test]
    fn bounded_load_path_screen_keeps_zero_resultant_zero() {
        let result = screen_bounded_load_path_demand(BoundedLoadPathDemandRequest {
            total_compressive_resultant_n: ScalarBounds { min: 0.0, max: 0.0 },
            effective_load_path_count: ScalarBounds { min: 1.0, max: 100.0 },
            peak_to_mean_load_factor: ScalarBounds { min: 1.0, max: 2.0 },
        })
        .unwrap();
        assert_eq!(result.factor_scaled_load_path_demand_lower_bound_n, 0.0);
        assert_eq!(result.factor_scaled_load_path_demand_upper_bound_n, 0.0);
    }

    #[test]
    fn bounded_load_path_screen_rejects_reversed_resultant_bounds() {
        let result = screen_bounded_load_path_demand(BoundedLoadPathDemandRequest {
            total_compressive_resultant_n: ScalarBounds { min: 10.0, max: 9.0 },
            effective_load_path_count: ScalarBounds { min: 1.0, max: 2.0 },
            peak_to_mean_load_factor: ScalarBounds { min: 1.0, max: 1.0 },
        });
        assert_eq!(
            result.unwrap_err(),
            ScreeningError::InvalidInterval("total_compressive_resultant_n")
        );
    }

    #[test]
    fn bounded_load_path_screen_rejects_zero_minimum_path_count() {
        let result = screen_bounded_load_path_demand(BoundedLoadPathDemandRequest {
            total_compressive_resultant_n: ScalarBounds { min: 1.0, max: 2.0 },
            effective_load_path_count: ScalarBounds { min: 0.0, max: 2.0 },
            peak_to_mean_load_factor: ScalarBounds { min: 1.0, max: 1.0 },
        });
        assert_eq!(
            result.unwrap_err(),
            ScreeningError::MustBePositive("effective_load_path_count")
        );
    }

    #[test]
    fn bounded_load_path_screen_rejects_factor_interval_below_one() {
        let result = screen_bounded_load_path_demand(BoundedLoadPathDemandRequest {
            total_compressive_resultant_n: ScalarBounds { min: 1.0, max: 2.0 },
            effective_load_path_count: ScalarBounds { min: 1.0, max: 2.0 },
            peak_to_mean_load_factor: ScalarBounds { min: 0.9, max: 1.2 },
        });
        assert_eq!(
            result.unwrap_err(),
            ScreeningError::FactorBelowOne("peak_to_mean_load_factor")
        );
    }

    #[test]
    fn bounded_load_path_screen_rejects_non_finite_factor_endpoint() {
        let result = screen_bounded_load_path_demand(BoundedLoadPathDemandRequest {
            total_compressive_resultant_n: ScalarBounds { min: 1.0, max: 2.0 },
            effective_load_path_count: ScalarBounds { min: 1.0, max: 2.0 },
            peak_to_mean_load_factor: ScalarBounds { min: 1.0, max: f64::INFINITY },
        });
        assert_eq!(
            result.unwrap_err(),
            ScreeningError::NonFiniteInput("peak_to_mean_load_factor")
        );
    }

    #[test]
    fn bounded_load_path_screen_rejects_derived_load_overflow() {
        let result = screen_bounded_load_path_demand(BoundedLoadPathDemandRequest {
            total_compressive_resultant_n: ScalarBounds { min: 1.0, max: f64::MAX },
            effective_load_path_count: ScalarBounds { min: 0.5, max: 1.0 },
            peak_to_mean_load_factor: ScalarBounds { min: 1.0, max: 2.0 },
        });
        assert_eq!(
            result.unwrap_err(),
            ScreeningError::NonFiniteResult("mean load-path upper bound")
        );
    }

    #[test]
    fn coherent_load_path_scenarios_preserve_correlations_and_identify_extrema() {
        let scenarios = [
            LoadPathDemandScenario {
                scenario_id: "geometry-A/imperfection-1".into(),
                evidence_record_id: "analysis-record-A".into(),
                total_compressive_resultant_n: 1_000.0,
                effective_load_path_count: 10.0,
                peak_to_mean_load_factor: 1.1,
            },
            LoadPathDemandScenario {
                scenario_id: "geometry-B/imperfection-2".into(),
                evidence_record_id: "analysis-record-B".into(),
                total_compressive_resultant_n: 2_000.0,
                effective_load_path_count: 40.0,
                peak_to_mean_load_factor: 1.3,
            },
        ];
        let result = screen_load_path_scenario_set(&scenarios).unwrap();
        assert_eq!(result.evaluated_scenario_count, 2);
        assert_eq!(result.scenario_results[0].scenario_id, "geometry-A/imperfection-1");
        assert_eq!(result.scenario_results[0].evidence_record_id, "analysis-record-A");
        assert!(
            (result.scenario_results[0].demand.factor_scaled_peak_load_path_demand_n - 110.0)
                .abs()
                < 1e-12
        );
        assert!(
            (result.scenario_results[1].demand.factor_scaled_peak_load_path_demand_n - 65.0)
                .abs()
                < 1e-12
        );
        assert!(
            (result.minimum_factor_scaled_demand_among_scenarios_n - 65.0).abs() < 1e-12
        );
        assert_eq!(result.minimum_scenario_id, "geometry-B/imperfection-2");
        assert!(
            (result.maximum_factor_scaled_demand_among_scenarios_n - 110.0).abs() < 1e-12
        );
        assert_eq!(result.maximum_scenario_id, "geometry-A/imperfection-1");

        let independent_marginals = screen_bounded_load_path_demand(BoundedLoadPathDemandRequest {
            total_compressive_resultant_n: ScalarBounds {
                min: 1_000.0,
                max: 2_000.0,
            },
            effective_load_path_count: ScalarBounds { min: 10.0, max: 40.0 },
            peak_to_mean_load_factor: ScalarBounds { min: 1.1, max: 1.3 },
        })
        .unwrap();
        assert!(
            (independent_marginals.factor_scaled_load_path_demand_upper_bound_n - 260.0).abs()
                < 1e-12
        );
        assert!(
            result.maximum_factor_scaled_demand_among_scenarios_n
                < independent_marginals.factor_scaled_load_path_demand_upper_bound_n
        );
    }

    #[test]
    fn coherent_load_path_scenario_set_rejects_empty_input() {
        assert_eq!(
            screen_load_path_scenario_set(&[]).unwrap_err(),
            ScreeningError::EmptyScenarioSet
        );
    }

    #[test]
    fn coherent_load_path_scenario_set_rejects_blank_identifiers() {
        let blank_scenario = [LoadPathDemandScenario {
            scenario_id: "  ".into(),
            evidence_record_id: "analysis-record".into(),
            total_compressive_resultant_n: 1.0,
            effective_load_path_count: 1.0,
            peak_to_mean_load_factor: 1.0,
        }];
        assert_eq!(
            screen_load_path_scenario_set(&blank_scenario).unwrap_err(),
            ScreeningError::EmptyIdentifier("scenario_id")
        );

        let blank_evidence = [LoadPathDemandScenario {
            scenario_id: "scenario-1".into(),
            evidence_record_id: "".into(),
            total_compressive_resultant_n: 1.0,
            effective_load_path_count: 1.0,
            peak_to_mean_load_factor: 1.0,
        }];
        assert_eq!(
            screen_load_path_scenario_set(&blank_evidence).unwrap_err(),
            ScreeningError::EmptyIdentifier("evidence_record_id")
        );
    }

    #[test]
    fn coherent_load_path_scenario_set_rejects_duplicate_scenario_ids() {
        let scenarios = [
            LoadPathDemandScenario {
                scenario_id: "same-id".into(),
                evidence_record_id: "record-A".into(),
                total_compressive_resultant_n: 1.0,
                effective_load_path_count: 1.0,
                peak_to_mean_load_factor: 1.0,
            },
            LoadPathDemandScenario {
                scenario_id: "same-id".into(),
                evidence_record_id: "record-B".into(),
                total_compressive_resultant_n: 2.0,
                effective_load_path_count: 2.0,
                peak_to_mean_load_factor: 1.0,
            },
        ];
        assert_eq!(
            screen_load_path_scenario_set(&scenarios).unwrap_err(),
            ScreeningError::DuplicateScenarioId
        );
    }

    #[test]
    fn feature_defect_population_probability_increases_with_population_size() {
        let small = screen_feature_defect_population(270, 0.0001).unwrap();
        let large = screen_feature_defect_population(10_000, 0.0001).unwrap();
        assert!(small.probability_at_least_one_feature_exceeds_criterion > 0.0);
        assert!(large.probability_at_least_one_feature_exceeds_criterion
            > small.probability_at_least_one_feature_exceeds_criterion);
        assert!((small.probability_at_least_one_feature_exceeds_criterion - 0.02664).abs() < 0.00001);
        assert!((large.probability_at_least_one_feature_exceeds_criterion - 0.63214).abs() < 0.00001);
    }

    #[test]
    fn feature_defect_population_probability_handles_probability_endpoints() {
        assert_eq!(
            screen_feature_defect_population(10, 0.0).unwrap()
                .probability_at_least_one_feature_exceeds_criterion,
            0.0
        );
        assert_eq!(
            screen_feature_defect_population(10, 1.0).unwrap()
                .probability_at_least_one_feature_exceeds_criterion,
            1.0
        );
    }

    #[test]
    fn feature_defect_population_probability_rejects_invalid_input() {
        assert_eq!(
            screen_feature_defect_population(0, 0.1).unwrap_err(),
            ScreeningError::MustBePositive("feature_count")
        );
        assert_eq!(
            screen_feature_defect_population(10, 1.1).unwrap_err(),
            ScreeningError::FractionOutOfRange("per_feature_exceedance_probability")
        );
        assert_eq!(
            screen_feature_defect_population(10, f64::NAN).unwrap_err(),
            ScreeningError::NonFiniteInput("per_feature_exceedance_probability")
        );
    }

    fn complete_candidate_evidence_profile() -> CandidateEvidenceGateProfile {
        CandidateEvidenceGateProfile {
            mass_budget: MassBudgetStatus::RobustlyNonNegative,
            pressure_capacity: PressureEvidenceGateState::ScopeMatchedUnverified,
            nonlinear_analysis: NonlinearAnalysisGateState::NonlinearImperfectionAware,
            independent_structural_review: IndependentReviewGateState::IndependentlyReviewed,
            physical_validation: PhysicalValidationGateState::RepresentativeShell,
            damage_tolerance: DamageToleranceGateState::DemonstratedByTestWithinDeclaredEnvelope,
            operating_envelope: OperatingEnvelopeGateState::DefinedAndIndependentlyReviewed,
            evidence_manifest: EvidenceManifestGateState::ContentAddressed,
            vehicle_mass_ledger: VehicleMassLedgerGateState::ReconciledToMeasuredOrBuildEstimate,
            scale_transfer: ScaleTransferGateState::ValidatedAcrossRelevantScaleRange,
            defect_population: DefectPopulationGateState::CharacterizedForTargetManufacturingPopulation,
        }
    }

    #[test]
    fn candidate_gate_requires_all_critical_evidence_classes() {
        let mut profile = complete_candidate_evidence_profile();
        profile.pressure_capacity = PressureEvidenceGateState::ScopeMismatch;
        profile.physical_validation = PhysicalValidationGateState::SubscaleShell;
        profile.damage_tolerance = DamageToleranceGateState::AnalysisOnly;
        let result = assess_candidate_evidence_gates(profile);
        assert_eq!(result.status, CandidateEvidenceGateStatus::EvidenceGapsRemain);
        assert!(result.blockers.contains(&"pressure_capacity_scope_not_matched_or_missing"));
        assert!(result.blockers.contains(&"representative_shell_validation_missing"));
        assert!(result.blockers.contains(&"damage_tolerance_not_demonstrated_by_test"));
    }

    #[test]
    fn candidate_gate_never_averages_away_a_bound_sensitive_mass_budget() {
        let mut profile = complete_candidate_evidence_profile();
        profile.mass_budget = MassBudgetStatus::BoundSensitive;
        let result = assess_candidate_evidence_gates(profile);
        assert_eq!(result.status, CandidateEvidenceGateStatus::MassBudgetBoundSensitive);
        assert!(result.blockers.contains(&"mass_budget_not_robustly_nonnegative"));
    }

    #[test]
    fn candidate_gate_negative_mass_budget_takes_precedence() {
        let mut profile = complete_candidate_evidence_profile();
        profile.mass_budget = MassBudgetStatus::NegativeThroughout;
        let result = assess_candidate_evidence_gates(profile);
        assert_eq!(result.status, CandidateEvidenceGateStatus::MassBudgetNegativeThroughout);
        assert_eq!(result.blockers, vec!["mass_budget_negative_throughout"]);
    }

    #[test]
    fn candidate_gate_requires_imperfection_analysis_independent_review_and_manifest() {
        let mut profile = complete_candidate_evidence_profile();
        profile.nonlinear_analysis = NonlinearAnalysisGateState::LinearOrPerfectGeometryOnly;
        profile.independent_structural_review =
            IndependentReviewGateState::ReviewNotIndependentOrNotRecorded;
        profile.evidence_manifest = EvidenceManifestGateState::IdentifiersOnly;
        let result = assess_candidate_evidence_gates(profile);
        assert_eq!(result.status, CandidateEvidenceGateStatus::EvidenceGapsRemain);
        assert!(
            result
                .blockers
                .contains(&"nonlinear_imperfection_analysis_missing_or_insufficient")
        );
        assert!(result.blockers.contains(&"independent_structural_review_missing"));
        assert!(result.blockers.contains(&"evidence_manifest_not_content_addressed"));
    }

    #[test]
    fn candidate_gate_requires_mass_reconciliation_scale_transfer_and_defect_tail() {
        let mut profile = complete_candidate_evidence_profile();
        profile.vehicle_mass_ledger = VehicleMassLedgerGateState::ComponentCompleteUnreconciled;
        profile.scale_transfer = ScaleTransferGateState::AnalyticalScalingOnly;
        profile.defect_population = DefectPopulationGateState::SingleSpecimenOrSmallSampleOnly;
        let result = assess_candidate_evidence_gates(profile);
        assert_eq!(result.status, CandidateEvidenceGateStatus::EvidenceGapsRemain);
        assert!(result.blockers.contains(&"vehicle_mass_ledger_not_reconciled"));
        assert!(result.blockers.contains(&"scale_transfer_not_validated"));
        assert!(result.blockers.contains(&"manufacturing_defect_population_not_characterized"));
    }

    #[test]
    fn candidate_gate_full_declared_profile_only_advances_to_external_assessment() {
        let result = assess_candidate_evidence_gates(complete_candidate_evidence_profile());
        assert_eq!(
            result.status,
            CandidateEvidenceGateStatus::EvidenceBundleReadyForIndependentAssessment
        );
        assert!(result.blockers.is_empty());
    }

    fn pressure_capacity_bounds(min_pa: f64, max_pa: f64) -> PressureCapacityBoundsRecord {
        PressureCapacityBoundsRecord {
            allowable_external_pressure_pa: ScalarBounds { min: min_pa, max: max_pa },
            evidence_record_id: "capacity-record/v1".into(),
            basis_profile_id: "design-profile/v1".into(),
            independent_check_or_test_id: "independent-test/v1".into(),
        }
    }

    fn pressure_bounds_request(
        external: ScalarBounds,
        internal: ScalarBounds,
        capacity: Option<PressureCapacityBoundsRecord>,
    ) -> ExternalPressureBoundsRequest {
        ExternalPressureBoundsRequest {
            external_pressure_pa: external,
            internal_pressure_pa: internal,
            capacity,
        }
    }

    #[test]
    fn bounded_pressure_capacity_covers_all_declared_demands_when_endpoints_allow() {
        let result = assess_external_pressure_bounds(pressure_bounds_request(
            ScalarBounds { min: 100_000.0, max: 102_000.0 },
            ScalarBounds { min: 0.0, max: 2_000.0 },
            Some(pressure_capacity_bounds(105_000.0, 120_000.0)),
        )).unwrap();
        match result {
            PressureBoundsAssessment::DeclaredCapacityBoundsComparison {
                status,
                minimum_positive_demand_pa,
                maximum_positive_demand_pa,
                ..
            } => {
                assert_eq!(status, CapacityBoundsStatus::DeclaredCapacityCoversDemandBounds);
                assert_eq!(minimum_positive_demand_pa, 98_000.0);
                assert_eq!(maximum_positive_demand_pa, 102_000.0);
            }
            other => panic!("unexpected assessment: {other:?}"),
        }
    }

    #[test]
    fn bounded_pressure_capacity_returns_overlap_when_bounds_cross() {
        let result = assess_external_pressure_bounds(pressure_bounds_request(
            ScalarBounds { min: 100_000.0, max: 110_000.0 },
            ScalarBounds { min: 0.0, max: 1_000.0 },
            Some(pressure_capacity_bounds(105_000.0, 115_000.0)),
        )).unwrap();
        assert!(matches!(
            result,
            PressureBoundsAssessment::DeclaredCapacityBoundsComparison {
                status: CapacityBoundsStatus::BoundsOverlap,
                ..
            }
        ));
    }

    #[test]
    fn bounded_pressure_capacity_reports_declared_capacity_below_all_demands() {
        let result = assess_external_pressure_bounds(pressure_bounds_request(
            ScalarBounds { min: 110_000.0, max: 120_000.0 },
            ScalarBounds { min: 0.0, max: 1_000.0 },
            Some(pressure_capacity_bounds(90_000.0, 100_000.0)),
        )).unwrap();
        assert!(matches!(
            result,
            PressureBoundsAssessment::DeclaredCapacityBoundsComparison {
                status: CapacityBoundsStatus::DeclaredCapacityBelowDemandBounds,
                ..
            }
        ));
    }

    #[test]
    fn bounded_pressure_screen_keeps_missing_capacity_unknown() {
        let result = assess_external_pressure_bounds(pressure_bounds_request(
            ScalarBounds { min: 100_000.0, max: 102_000.0 },
            ScalarBounds { min: 0.0, max: 1_000.0 },
            None,
        )).unwrap();
        assert!(matches!(result, PressureBoundsAssessment::CapacityUnknown { .. }));
    }


    #[test]
    fn bounded_pressure_screen_rejects_invalid_capacity_even_when_no_demand() {
        let result = assess_external_pressure_bounds(pressure_bounds_request(
            ScalarBounds { min: 80_000.0, max: 95_000.0 },
            ScalarBounds { min: 100_000.0, max: 110_000.0 },
            Some(pressure_capacity_bounds(130_000.0, 120_000.0)),
        ));
        assert_eq!(
            result.unwrap_err(),
            ScreeningError::InvalidInterval("allowable_external_pressure_pa")
        );
    }

    #[test]
    fn bounded_pressure_screen_reports_no_demand_only_if_entire_interval_has_no_positive_delta() {
        let result = assess_external_pressure_bounds(pressure_bounds_request(
            ScalarBounds { min: 80_000.0, max: 95_000.0 },
            ScalarBounds { min: 100_000.0, max: 110_000.0 },
            None,
        )).unwrap();
        match result {
            PressureBoundsAssessment::NoExternalPressureDemand {
                minimum_signed_differential_pa,
                maximum_signed_differential_pa,
            } => {
                assert!((minimum_signed_differential_pa - -30_000.0).abs() < 1e-9);
                assert!((maximum_signed_differential_pa - -5_000.0).abs() < 1e-9);
            }
            other => panic!("unexpected assessment: {other:?}"),
        }
    }

    #[test]
    fn bounded_pressure_screen_blocks_comparison_on_missing_provenance() {
        let mut capacity = pressure_capacity_bounds(120_000.0, 130_000.0);
        capacity.independent_check_or_test_id.clear();
        let result = assess_external_pressure_bounds(pressure_bounds_request(
            ScalarBounds { min: 100_000.0, max: 102_000.0 },
            ScalarBounds { min: 0.0, max: 1_000.0 },
            Some(capacity),
        )).unwrap();
        assert!(matches!(
            result,
            PressureBoundsAssessment::IncompleteProvenance {
                missing_fields,
                ..
            } if missing_fields == vec!["independent_check_or_test_id"]
        ));
    }

    #[test]
    fn bounded_pressure_screen_rejects_reversed_capacity_bounds() {
        let result = assess_external_pressure_bounds(pressure_bounds_request(
            ScalarBounds { min: 100_000.0, max: 102_000.0 },
            ScalarBounds { min: 0.0, max: 1_000.0 },
            Some(pressure_capacity_bounds(130_000.0, 120_000.0)),
        ));
        assert_eq!(
            result.unwrap_err(),
            ScreeningError::InvalidInterval("allowable_external_pressure_pa")
        );
    }


    fn matching_pressure_scope() -> PressureCaseScope {
        PressureCaseScope {
            geometry_revision_id: "geometry/sha256:abc".into(),
            mesh_or_specimen_revision_id: "mesh-or-specimen/sha256:def".into(),
            material_model_revision_id: "material-model/v3".into(),
            analysis_or_test_method_id:
                "analysis-method/nonlinear-imperfection-sensitive-v1".into(),
            load_case_id: "external-pressure-sea-level/v1".into(),
            boundary_condition_profile_id: "supports-and-joints/v2".into(),
            imperfection_basis_id: "measured-imperfections/bounds-v1".into(),
            capacity_basis_id: "allowable-basis/validated-knockdown-method-v1".into(),
        }
    }

    fn scoped_pressure_request(
        capacity: Option<PressureCapacityBoundsRecord>,
        required_scope: PressureCaseScope,
        supplied_capacity_scope: Option<PressureCaseScope>,
    ) -> ScopedExternalPressureBoundsRequest {
        ScopedExternalPressureBoundsRequest {
            pressure: pressure_bounds_request(
                ScalarBounds { min: 100_000.0, max: 102_000.0 },
                ScalarBounds { min: 0.0, max: 1_000.0 },
                capacity,
            ),
            required_scope,
            supplied_capacity_scope,
        }
    }

    #[test]
    fn scoped_pressure_comparison_allows_exact_scope_match_only() {
        let scope = matching_pressure_scope();
        let result = assess_scoped_external_pressure_bounds(scoped_pressure_request(
            Some(pressure_capacity_bounds(105_000.0, 120_000.0)),
            scope.clone(),
            Some(scope),
        )).unwrap();
        assert!(matches!(
            result,
            ScopedPressureBoundsAssessment::PressureAssessment(
                PressureBoundsAssessment::DeclaredCapacityBoundsComparison {
                    status: CapacityBoundsStatus::DeclaredCapacityCoversDemandBounds,
                    ..
                }
            )
        ));
    }

    #[test]
    fn scoped_pressure_comparison_blocks_geometry_revision_mismatch() {
        let required = matching_pressure_scope();
        let mut supplied = required.clone();
        supplied.geometry_revision_id = "geometry/sha256:old".into();
        let result = assess_scoped_external_pressure_bounds(scoped_pressure_request(
            Some(pressure_capacity_bounds(105_000.0, 120_000.0)),
            required,
            Some(supplied),
        )).unwrap();
        assert!(matches!(
            result,
            ScopedPressureBoundsAssessment::ScopeMismatch { mismatches }
                if mismatches.len() == 1
                    && mismatches[0].field == "geometry_revision_id"
                    && mismatches[0].required_value == "geometry/sha256:abc"
                    && mismatches[0].supplied_value == "geometry/sha256:old"
        ));
    }

    #[test]
    fn scoped_pressure_comparison_blocks_material_load_boundary_and_imperfection_mismatch() {
        let required = matching_pressure_scope();
        let mut supplied = required.clone();
        supplied.material_model_revision_id = "material-model/v2".into();
        supplied.load_case_id = "different-load-case".into();
        supplied.boundary_condition_profile_id = "different-boundaries".into();
        supplied.imperfection_basis_id = "perfect-geometry-only".into();
        let result = assess_scoped_external_pressure_bounds(scoped_pressure_request(
            Some(pressure_capacity_bounds(105_000.0, 120_000.0)),
            required,
            Some(supplied),
        )).unwrap();
        assert!(matches!(
            result,
            ScopedPressureBoundsAssessment::ScopeMismatch { mismatches }
                if mismatches.len() == 4
        ));
    }


    #[test]
    fn scoped_pressure_comparison_blocks_method_and_capacity_basis_mismatch() {
        let required = matching_pressure_scope();
        let mut supplied = required.clone();
        supplied.analysis_or_test_method_id = "linear-eigenvalue-only".into();
        supplied.capacity_basis_id = "perfect-geometry-critical-load".into();
        let result = assess_scoped_external_pressure_bounds(scoped_pressure_request(
            Some(pressure_capacity_bounds(105_000.0, 120_000.0)),
            required,
            Some(supplied),
        )).unwrap();
        assert!(matches!(
            result,
            ScopedPressureBoundsAssessment::ScopeMismatch { mismatches }
                if mismatches.len() == 2
                    && mismatches.iter().any(|m| m.field == "analysis_or_test_method_id")
                    && mismatches.iter().any(|m| m.field == "capacity_basis_id")
        ));
    }

    #[test]
    fn scoped_pressure_comparison_blocks_mesh_or_specimen_revision_mismatch() {
        let required = matching_pressure_scope();
        let mut supplied = required.clone();
        supplied.mesh_or_specimen_revision_id = "mesh-or-specimen/old".into();
        let result = assess_scoped_external_pressure_bounds(scoped_pressure_request(
            Some(pressure_capacity_bounds(105_000.0, 120_000.0)),
            required,
            Some(supplied),
        )).unwrap();
        assert!(matches!(
            result,
            ScopedPressureBoundsAssessment::ScopeMismatch { mismatches }
                if mismatches.len() == 1
                    && mismatches[0].field == "mesh_or_specimen_revision_id"
        ));
    }

    #[test]
    fn scoped_pressure_comparison_rejects_missing_capacity_basis() {
        let mut required = matching_pressure_scope();
        required.capacity_basis_id.clear();
        let result = assess_scoped_external_pressure_bounds(scoped_pressure_request(
            Some(pressure_capacity_bounds(105_000.0, 120_000.0)),
            required,
            Some(matching_pressure_scope()),
        )).unwrap();
        assert!(matches!(
            result,
            ScopedPressureBoundsAssessment::ScopeIncomplete { missing_fields }
                if missing_fields == vec!["required_scope.capacity_basis_id"]
        ));
    }

    #[test]
    fn scoped_pressure_comparison_rejects_missing_scope_identifiers() {
        let mut required = matching_pressure_scope();
        required.imperfection_basis_id.clear();
        let result = assess_scoped_external_pressure_bounds(scoped_pressure_request(
            Some(pressure_capacity_bounds(105_000.0, 120_000.0)),
            required,
            Some(matching_pressure_scope()),
        )).unwrap();
        assert!(matches!(
            result,
            ScopedPressureBoundsAssessment::ScopeIncomplete { missing_fields }
                if missing_fields == vec!["required_scope.imperfection_basis_id"]
        ));
    }


    #[test]
    fn scoped_pressure_comparison_rejects_blank_supplied_scope_field() {
        let required = matching_pressure_scope();
        let mut supplied = required.clone();
        supplied.boundary_condition_profile_id = "   ".into();
        let result = assess_scoped_external_pressure_bounds(scoped_pressure_request(
            Some(pressure_capacity_bounds(105_000.0, 120_000.0)),
            required,
            Some(supplied),
        )).unwrap();
        assert!(matches!(
            result,
            ScopedPressureBoundsAssessment::ScopeIncomplete { missing_fields }
                if missing_fields == vec![
                    "supplied_capacity_scope.boundary_condition_profile_id"
                ]
        ));
    }

    #[test]
    fn scoped_pressure_comparison_preserves_incomplete_evidence_provenance() {
        let scope = matching_pressure_scope();
        let mut capacity = pressure_capacity_bounds(105_000.0, 120_000.0);
        capacity.independent_check_or_test_id.clear();
        let result = assess_scoped_external_pressure_bounds(scoped_pressure_request(
            Some(capacity),
            scope.clone(),
            Some(scope),
        )).unwrap();
        assert!(matches!(
            result,
            ScopedPressureBoundsAssessment::PressureAssessment(
                PressureBoundsAssessment::IncompleteProvenance {
                    missing_fields,
                    ..
                }
            ) if missing_fields == vec!["independent_check_or_test_id"]
        ));
    }

    #[test]
    fn scoped_pressure_comparison_keeps_unknown_capacity_without_scope() {
        let result = assess_scoped_external_pressure_bounds(scoped_pressure_request(
            None,
            matching_pressure_scope(),
            None,
        )).unwrap();
        assert!(matches!(
            result,
            ScopedPressureBoundsAssessment::PressureAssessment(
                PressureBoundsAssessment::CapacityUnknown { .. }
            )
        ));
    }

    #[test]
    fn scoped_pressure_comparison_rejects_missing_capacity_scope() {
        let result = assess_scoped_external_pressure_bounds(scoped_pressure_request(
            Some(pressure_capacity_bounds(105_000.0, 120_000.0)),
            matching_pressure_scope(),
            None,
        )).unwrap();
        assert!(matches!(
            result,
            ScopedPressureBoundsAssessment::ScopeIncomplete { missing_fields }
                if missing_fields == vec!["supplied_capacity_scope"]
        ));
    }

    #[test]
    fn equal_bounds_reduce_to_point_screen_payload_equation() {
        let request = SphericalPayloadBoundsRequest {
            radius_m: 20.0,
            ambient_air_density_kg_m3: ScalarBounds { min: 1.225, max: 1.225 },
            internal_gas_density_kg_m3: ScalarBounds { min: 0.0, max: 0.0 },
            structural_system_areal_mass_kg_m2: ScalarBounds { min: 4.0, max: 4.0 },
            fixed_non_structural_vehicle_mass_kg: ScalarBounds { min: 10_000.0, max: 10_000.0 },
        };
        let bounded = screen_spherical_payload_bounds(request).unwrap();
        let point = screen_spherical_payload(SphericalPayloadRequest {
            radius_m: 20.0,
            ambient_air_density_kg_m3: 1.225,
            internal_gas_density_kg_m3: 0.0,
            structural_system_areal_mass_kg_m2: 4.0,
            fixed_non_structural_vehicle_mass_kg: 10_000.0,
        })
        .unwrap();
        assert!(
            (bounded.payload_allowance_lower_bound_kg - point.payload_allowance_kg).abs() < 1e-8
        );
        assert!(
            (bounded.payload_allowance_upper_bound_kg - point.payload_allowance_kg).abs() < 1e-8
        );
        assert!(bounded.payload_allowance_interval_width_kg.abs() < 1e-8);
    }

}
