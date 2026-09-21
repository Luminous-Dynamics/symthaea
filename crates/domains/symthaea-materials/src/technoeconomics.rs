// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Source-bound materials technoeconomics and criticality accounting.
//!
//! Economic observations are time-dependent evidence, not timeless material
//! constants. Every price therefore carries a date, currency, physical basis,
//! source identifier, and artifact digest. Missing or incompatible price data
//! remain unknown instead of being silently treated as zero.
//!
//! Alloy compositions are supplied as exact atomic fractions but procurement is
//! mass-based. The evaluator therefore requires atomic masses and converts
//! `x_i -> w_i = x_i M_i / sum_j(x_j M_j)` before any $/kg or scarce-mass
//! accounting. Using atomic fraction directly as mass fraction is forbidden.
//!
//! This module also keeps scientific performance separate from economics. It
//! calculates direct material/process burdens and economic Pareto fronts; it does
//! not advance any MAT-001 scientific evidence stage.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Exact integer denominator used for composition fractions.
pub const ECONOMIC_COMPOSITION_PPM_TOTAL: u32 = 1_000_000;

/// Physical basis associated with a quoted price.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PriceBasis {
    /// Refined elemental metal suitable as a direct metallic feedstock basis.
    ElementalMetal,
    /// Oxide price. Not automatically convertible to a metal feedstock price.
    Oxide,
    /// Ore or mineral concentrate price.
    Concentrate,
    /// Alloy or master-alloy feedstock rather than the pure element.
    AlloyFeedstock,
    /// Explicitly named other basis.
    Other(String),
}

/// One exact composition component plus the atomic mass needed for mass costing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EconomicCompositionComponent {
    /// Atomic number.
    pub atomic_number: u16,
    /// Atomic fraction in integer ppm.
    pub fraction_ppm: u32,
    /// Atomic mass in unified atomic mass units from a bound composition basis.
    pub atomic_mass_u: f64,
}

/// Dated price evidence for one element.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElementPriceObservation {
    /// Element atomic number.
    pub atomic_number: u16,
    /// Quoted price per kilogram of the declared physical basis.
    pub price_per_kg: f64,
    /// ISO-style currency code or explicit currency identifier.
    pub currency: String,
    /// Physical form/basis to which the quote applies.
    pub basis: PriceBasis,
    /// Purity, grade, contract, or market-basis note.
    pub grade_or_basis_note: String,
    /// Observation/publication date as an ISO `YYYY-MM-DD` string.
    pub observation_date: String,
    /// Source identifier.
    pub source_id: String,
    /// SHA-256 digest of the exact evidence artifact or captured response.
    pub artifact_sha256: String,
}

/// Classification supplied by one named critical-material scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CriticalityStatus {
    /// Listed as critical under the bound scheme/version.
    Critical,
    /// Explicitly evaluated and not listed as critical.
    NotCritical,
    /// Source does not establish a classification.
    Unknown,
}

/// Source-bound criticality observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElementCriticalityObservation {
    /// Element atomic number.
    pub atomic_number: u16,
    /// Scheme/version/jurisdiction identifier.
    pub scheme_id: String,
    /// Classification under that exact scheme.
    pub status: CriticalityStatus,
    /// Optional import-reliance fraction in [0,1].
    pub import_reliance_fraction: Option<f64>,
    /// Observation/publication date as `YYYY-MM-DD`.
    pub observation_date: String,
    /// Source identifier.
    pub source_id: String,
    /// SHA-256 digest of the source artifact.
    pub artifact_sha256: String,
}

/// Manufacturing scenario applied to a finished material mass.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProcessEconomicScenario {
    /// Stable scenario identifier (for example `vacuum-arc-base-2026q3`).
    pub scenario_id: String,
    /// Human-readable process route.
    pub process_route: String,
    /// Finished-good mass yield from purchased feedstock, in (0,1].
    pub material_yield_fraction: f64,
    /// Process energy per kilogram of finished output.
    pub process_energy_kwh_per_finished_kg: f64,
    /// Non-feedstock direct process cost per kilogram of finished output.
    pub non_material_cost_per_finished_kg: f64,
    /// Currency for process cost; must match element-price observations.
    pub currency: String,
    /// Source or assumption-set identifier.
    pub source_id: String,
    /// SHA-256 digest of the scenario artifact.
    pub artifact_sha256: String,
}

/// Complete technoeconomic evaluation request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaterialTechnoeconomicInput {
    /// Stable candidate/material identity.
    pub subject_id: String,
    /// Finished component/material mass to evaluate.
    pub finished_mass_kg: f64,
    /// Exact atomic composition plus atomic masses.
    pub composition: Vec<EconomicCompositionComponent>,
    /// One price observation per constituent element.
    pub prices: Vec<ElementPriceObservation>,
    /// Zero or one criticality observation per constituent element.
    pub criticality: Vec<ElementCriticalityObservation>,
    /// Manufacturing scenario.
    pub process: ProcessEconomicScenario,
}

/// Element-level mass and cost contribution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElementEconomicContribution {
    /// Atomic number.
    pub atomic_number: u16,
    /// Atomic fraction in ppm.
    pub fraction_ppm: u32,
    /// Mass fraction derived from atomic fraction and atomic mass.
    pub mass_fraction: f64,
    /// Purchased feedstock mass allocated to this element.
    pub allocated_feedstock_mass_kg: f64,
    /// Direct feedstock cost contribution.
    pub feedstock_cost: f64,
    /// Bound price source.
    pub price_source_id: String,
    /// Criticality status, or `Unknown` if no source-bound classification exists.
    pub criticality_status: CriticalityStatus,
}

/// Technoeconomic result for one process scenario.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaterialTechnoeconomicResult {
    /// Subject identity.
    pub subject_id: String,
    /// Scenario identity.
    pub scenario_id: String,
    /// Currency shared by all monetary values.
    pub currency: String,
    /// Finished mass evaluated.
    pub finished_mass_kg: f64,
    /// Purchased feedstock mass after yield loss.
    pub required_feedstock_mass_kg: f64,
    /// Composition-allocated feedstock cost.
    pub raw_material_cost: f64,
    /// Direct non-material processing cost.
    pub process_cost: f64,
    /// Sum of material and process direct cost.
    pub total_direct_cost: f64,
    /// Process energy burden for the evaluated finished mass.
    pub process_energy_kwh: f64,
    /// Mass allocated to constituents explicitly classified as critical.
    pub known_critical_material_mass_kg: f64,
    /// Elements lacking a source-bound criticality classification.
    pub unknown_criticality_elements: Vec<u16>,
    /// Element-level accounting.
    pub elemental_contributions: Vec<ElementEconomicContribution>,
}

/// Economic point used for Pareto-front extraction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EconomicParetoPoint {
    /// Subject/scenario label.
    pub point_id: String,
    /// Direct cost to minimize.
    pub total_direct_cost: f64,
    /// Process energy to minimize.
    pub process_energy_kwh: f64,
    /// Known critical-material mass to minimize.
    pub known_critical_material_mass_kg: f64,
}

/// Evaluate one source-bound material/process scenario.
///
/// Direct feedstock costing currently requires `ElementalMetal` quotes. Oxide,
/// concentrate, and master-alloy prices are rejected rather than pretending they
/// are equivalent to refined metal delivered into an alloy process.
pub fn evaluate_material_technoeconomics(
    input: &MaterialTechnoeconomicInput,
) -> Result<MaterialTechnoeconomicResult, TechnoeconomicError> {
    validate_input(input)?;

    let price_map: HashMap<u16, &ElementPriceObservation> =
        input.prices.iter().map(|price| (price.atomic_number, price)).collect();
    let criticality_map: HashMap<u16, &ElementCriticalityObservation> = input
        .criticality
        .iter()
        .map(|record| (record.atomic_number, record))
        .collect();

    let atomic_mass_denominator: f64 = input
        .composition
        .iter()
        .map(|component| {
            (component.fraction_ppm as f64 / ECONOMIC_COMPOSITION_PPM_TOTAL as f64)
                * component.atomic_mass_u
        })
        .sum();
    if !atomic_mass_denominator.is_finite() || atomic_mass_denominator <= 0.0 {
        return Err(TechnoeconomicError::InvalidAtomicMassBasis);
    }

    let required_feedstock_mass_kg = input.finished_mass_kg / input.process.material_yield_fraction;
    let mut raw_material_cost = 0.0;
    let mut known_critical_material_mass_kg = 0.0;
    let mut unknown_criticality_elements = Vec::new();
    let mut elemental_contributions = Vec::with_capacity(input.composition.len());

    for component in &input.composition {
        let price = price_map
            .get(&component.atomic_number)
            .ok_or(TechnoeconomicError::MissingPrice(component.atomic_number))?;
        if price.basis != PriceBasis::ElementalMetal {
            return Err(TechnoeconomicError::IncompatiblePriceBasis {
                atomic_number: component.atomic_number,
                basis: price.basis.clone(),
            });
        }

        let atomic_fraction =
            component.fraction_ppm as f64 / ECONOMIC_COMPOSITION_PPM_TOTAL as f64;
        let mass_fraction = atomic_fraction * component.atomic_mass_u / atomic_mass_denominator;
        let allocated_mass = required_feedstock_mass_kg * mass_fraction;
        let feedstock_cost = allocated_mass * price.price_per_kg;
        raw_material_cost += feedstock_cost;

        let status = criticality_map
            .get(&component.atomic_number)
            .map(|record| record.status)
            .unwrap_or(CriticalityStatus::Unknown);
        match status {
            CriticalityStatus::Critical => known_critical_material_mass_kg += allocated_mass,
            CriticalityStatus::Unknown => unknown_criticality_elements.push(component.atomic_number),
            CriticalityStatus::NotCritical => {}
        }

        elemental_contributions.push(ElementEconomicContribution {
            atomic_number: component.atomic_number,
            fraction_ppm: component.fraction_ppm,
            mass_fraction,
            allocated_feedstock_mass_kg: allocated_mass,
            feedstock_cost,
            price_source_id: price.source_id.clone(),
            criticality_status: status,
        });
    }

    unknown_criticality_elements.sort_unstable();
    let process_cost = input.finished_mass_kg * input.process.non_material_cost_per_finished_kg;
    let process_energy_kwh =
        input.finished_mass_kg * input.process.process_energy_kwh_per_finished_kg;

    Ok(MaterialTechnoeconomicResult {
        subject_id: input.subject_id.clone(),
        scenario_id: input.process.scenario_id.clone(),
        currency: input.process.currency.clone(),
        finished_mass_kg: input.finished_mass_kg,
        required_feedstock_mass_kg,
        raw_material_cost,
        process_cost,
        total_direct_cost: raw_material_cost + process_cost,
        process_energy_kwh,
        known_critical_material_mass_kg,
        unknown_criticality_elements,
        elemental_contributions,
    })
}

/// Return all nondominated economic points, minimizing cost, process energy,
/// and known critical-material mass simultaneously.
pub fn economic_pareto_front(
    points: &[EconomicParetoPoint],
) -> Result<Vec<EconomicParetoPoint>, TechnoeconomicError> {
    for point in points {
        for (field, value) in [
            ("total_direct_cost", point.total_direct_cost),
            ("process_energy_kwh", point.process_energy_kwh),
            (
                "known_critical_material_mass_kg",
                point.known_critical_material_mass_kg,
            ),
        ] {
            validate_nonnegative_finite(field, value)?;
        }
        if point.point_id.trim().is_empty() {
            return Err(TechnoeconomicError::EmptyParetoPointId);
        }
    }

    let mut front = Vec::new();
    'candidate: for (i, candidate) in points.iter().enumerate() {
        for (j, other) in points.iter().enumerate() {
            if i == j {
                continue;
            }
            if dominates(other, candidate) {
                continue 'candidate;
            }
        }
        front.push(candidate.clone());
    }
    Ok(front)
}

fn dominates(a: &EconomicParetoPoint, b: &EconomicParetoPoint) -> bool {
    let no_worse = a.total_direct_cost <= b.total_direct_cost
        && a.process_energy_kwh <= b.process_energy_kwh
        && a.known_critical_material_mass_kg <= b.known_critical_material_mass_kg;
    let strictly_better = a.total_direct_cost < b.total_direct_cost
        || a.process_energy_kwh < b.process_energy_kwh
        || a.known_critical_material_mass_kg < b.known_critical_material_mass_kg;
    no_worse && strictly_better
}

fn validate_input(input: &MaterialTechnoeconomicInput) -> Result<(), TechnoeconomicError> {
    if input.subject_id.trim().is_empty() {
        return Err(TechnoeconomicError::EmptySubjectId);
    }
    validate_positive_finite("finished_mass_kg", input.finished_mass_kg)?;
    if input.composition.is_empty() {
        return Err(TechnoeconomicError::EmptyComposition);
    }

    let mut elements = HashSet::new();
    let mut sum = 0u32;
    for component in &input.composition {
        if component.fraction_ppm == 0 {
            return Err(TechnoeconomicError::ZeroCompositionFraction(
                component.atomic_number,
            ));
        }
        if !elements.insert(component.atomic_number) {
            return Err(TechnoeconomicError::DuplicateCompositionElement(
                component.atomic_number,
            ));
        }
        validate_positive_finite("atomic_mass_u", component.atomic_mass_u)?;
        sum = sum
            .checked_add(component.fraction_ppm)
            .ok_or(TechnoeconomicError::CompositionFractionOverflow)?;
    }
    if sum != ECONOMIC_COMPOSITION_PPM_TOTAL {
        return Err(TechnoeconomicError::CompositionDoesNotSumToOne { sum_ppm: sum });
    }

    validate_process(&input.process)?;

    let mut price_elements = HashSet::new();
    for price in &input.prices {
        if !price_elements.insert(price.atomic_number) {
            return Err(TechnoeconomicError::DuplicatePrice(price.atomic_number));
        }
        if !elements.contains(&price.atomic_number) {
            return Err(TechnoeconomicError::PriceElementOutsideComposition(
                price.atomic_number,
            ));
        }
        validate_positive_finite("price_per_kg", price.price_per_kg)?;
        validate_currency(&price.currency)?;
        if price.currency != input.process.currency {
            return Err(TechnoeconomicError::MixedCurrency {
                expected: input.process.currency.clone(),
                actual: price.currency.clone(),
            });
        }
        validate_date(&price.observation_date)?;
        validate_source_binding(&price.source_id, &price.artifact_sha256)?;
        if price.grade_or_basis_note.trim().is_empty() {
            return Err(TechnoeconomicError::EmptyPriceBasisNote(price.atomic_number));
        }
    }
    for atomic_number in &elements {
        if !price_elements.contains(atomic_number) {
            return Err(TechnoeconomicError::MissingPrice(*atomic_number));
        }
    }

    let mut criticality_elements = HashSet::new();
    for record in &input.criticality {
        if !criticality_elements.insert(record.atomic_number) {
            return Err(TechnoeconomicError::DuplicateCriticalityRecord(
                record.atomic_number,
            ));
        }
        if !elements.contains(&record.atomic_number) {
            return Err(TechnoeconomicError::CriticalityElementOutsideComposition(
                record.atomic_number,
            ));
        }
        if record.scheme_id.trim().is_empty() {
            return Err(TechnoeconomicError::EmptyCriticalityScheme(
                record.atomic_number,
            ));
        }
        if let Some(reliance) = record.import_reliance_fraction {
            if !reliance.is_finite() || !(0.0..=1.0).contains(&reliance) {
                return Err(TechnoeconomicError::InvalidImportReliance {
                    atomic_number: record.atomic_number,
                    value: reliance,
                });
            }
        }
        validate_date(&record.observation_date)?;
        validate_source_binding(&record.source_id, &record.artifact_sha256)?;
    }
    Ok(())
}

fn validate_process(process: &ProcessEconomicScenario) -> Result<(), TechnoeconomicError> {
    if process.scenario_id.trim().is_empty() || process.process_route.trim().is_empty() {
        return Err(TechnoeconomicError::EmptyProcessIdentity);
    }
    if !process.material_yield_fraction.is_finite()
        || process.material_yield_fraction <= 0.0
        || process.material_yield_fraction > 1.0
    {
        return Err(TechnoeconomicError::InvalidYield(
            process.material_yield_fraction,
        ));
    }
    validate_nonnegative_finite(
        "process_energy_kwh_per_finished_kg",
        process.process_energy_kwh_per_finished_kg,
    )?;
    validate_nonnegative_finite(
        "non_material_cost_per_finished_kg",
        process.non_material_cost_per_finished_kg,
    )?;
    validate_currency(&process.currency)?;
    validate_source_binding(&process.source_id, &process.artifact_sha256)?;
    Ok(())
}

fn validate_currency(currency: &str) -> Result<(), TechnoeconomicError> {
    if currency.trim().len() != 3 || !currency.bytes().all(|b| b.is_ascii_alphabetic()) {
        return Err(TechnoeconomicError::InvalidCurrency(currency.to_string()));
    }
    Ok(())
}

fn validate_date(date: &str) -> Result<(), TechnoeconomicError> {
    let bytes = date.as_bytes();
    let shape_ok = bytes.len() == 10
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && bytes
            .iter()
            .enumerate()
            .all(|(i, b)| i == 4 || i == 7 || b.is_ascii_digit());
    if !shape_ok {
        return Err(TechnoeconomicError::InvalidObservationDate(date.to_string()));
    }
    Ok(())
}

fn validate_source_binding(source_id: &str, artifact_sha256: &str) -> Result<(), TechnoeconomicError> {
    if source_id.trim().is_empty() {
        return Err(TechnoeconomicError::EmptySourceId);
    }
    if artifact_sha256.len() != 64 || !artifact_sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(TechnoeconomicError::InvalidSha256);
    }
    Ok(())
}

fn validate_positive_finite(field: &'static str, value: f64) -> Result<(), TechnoeconomicError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(TechnoeconomicError::InvalidPositiveValue { field, value });
    }
    Ok(())
}

fn validate_nonnegative_finite(
    field: &'static str,
    value: f64,
) -> Result<(), TechnoeconomicError> {
    if !value.is_finite() || value < 0.0 {
        return Err(TechnoeconomicError::InvalidNonnegativeValue { field, value });
    }
    Ok(())
}

/// Technoeconomic validation/evaluation failures.
#[derive(Debug, Clone, PartialEq)]
pub enum TechnoeconomicError {
    /// Subject identity was empty.
    EmptySubjectId,
    /// Composition was empty.
    EmptyComposition,
    /// Composition contained a zero-valued component.
    ZeroCompositionFraction(u16),
    /// Composition repeated an element.
    DuplicateCompositionElement(u16),
    /// Composition sum overflowed.
    CompositionFractionOverflow,
    /// Composition did not sum to exactly one million ppm.
    CompositionDoesNotSumToOne {
        /// Actual sum.
        sum_ppm: u32,
    },
    /// Atomic-mass conversion basis was invalid.
    InvalidAtomicMassBasis,
    /// No price observation exists for a required constituent.
    MissingPrice(u16),
    /// Same element price supplied more than once.
    DuplicatePrice(u16),
    /// Price referenced an element outside the composition.
    PriceElementOutsideComposition(u16),
    /// Price basis cannot be used as direct refined-metal feedstock cost.
    IncompatiblePriceBasis {
        /// Element atomic number.
        atomic_number: u16,
        /// Actual quoted physical basis.
        basis: PriceBasis,
    },
    /// Price grade/basis note was empty.
    EmptyPriceBasisNote(u16),
    /// Price currency differed from the process scenario currency.
    MixedCurrency {
        /// Required currency.
        expected: String,
        /// Observed currency.
        actual: String,
    },
    /// Currency identifier was malformed.
    InvalidCurrency(String),
    /// Criticality record repeated an element.
    DuplicateCriticalityRecord(u16),
    /// Criticality record referenced an element outside the composition.
    CriticalityElementOutsideComposition(u16),
    /// Criticality scheme identifier was empty.
    EmptyCriticalityScheme(u16),
    /// Import reliance was outside [0,1].
    InvalidImportReliance {
        /// Element atomic number.
        atomic_number: u16,
        /// Invalid reliance.
        value: f64,
    },
    /// Process identity or route was empty.
    EmptyProcessIdentity,
    /// Material yield was outside (0,1].
    InvalidYield(f64),
    /// Observation date was not `YYYY-MM-DD` shaped.
    InvalidObservationDate(String),
    /// Source identifier was empty.
    EmptySourceId,
    /// Evidence artifact digest was not hexadecimal SHA-256 shape.
    InvalidSha256,
    /// Required positive numeric input was invalid.
    InvalidPositiveValue {
        /// Field name.
        field: &'static str,
        /// Invalid value.
        value: f64,
    },
    /// Required nonnegative numeric input was invalid.
    InvalidNonnegativeValue {
        /// Field name.
        field: &'static str,
        /// Invalid value.
        value: f64,
    },
    /// Pareto point identity was empty.
    EmptyParetoPointId,
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIGEST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn price(z: u16, value: f64) -> ElementPriceObservation {
        ElementPriceObservation {
            atomic_number: z,
            price_per_kg: value,
            currency: "USD".to_string(),
            basis: PriceBasis::ElementalMetal,
            grade_or_basis_note: "fixture refined metal".to_string(),
            observation_date: "2026-09-19".to_string(),
            source_id: format!("fixture-price-Z{z}"),
            artifact_sha256: DIGEST.to_string(),
        }
    }

    fn critical(z: u16, status: CriticalityStatus) -> ElementCriticalityObservation {
        ElementCriticalityObservation {
            atomic_number: z,
            scheme_id: "fixture-criticality-v1".to_string(),
            status,
            import_reliance_fraction: Some(1.0),
            observation_date: "2026-09-19".to_string(),
            source_id: format!("fixture-critical-Z{z}"),
            artifact_sha256: DIGEST.to_string(),
        }
    }

    fn base_input(finished_mass_kg: f64) -> MaterialTechnoeconomicInput {
        MaterialTechnoeconomicInput {
            subject_id: "fixture:Ti50-Ta50".to_string(),
            finished_mass_kg,
            composition: vec![
                EconomicCompositionComponent {
                    atomic_number: 22,
                    fraction_ppm: 500_000,
                    atomic_mass_u: 47.867,
                },
                EconomicCompositionComponent {
                    atomic_number: 73,
                    fraction_ppm: 500_000,
                    atomic_mass_u: 180.947_88,
                },
            ],
            prices: vec![price(22, 10.0), price(73, 100.0)],
            criticality: vec![
                critical(22, CriticalityStatus::NotCritical),
                critical(73, CriticalityStatus::Critical),
            ],
            process: ProcessEconomicScenario {
                scenario_id: "fixture-process".to_string(),
                process_route: "fixture melt".to_string(),
                material_yield_fraction: 0.8,
                process_energy_kwh_per_finished_kg: 20.0,
                non_material_cost_per_finished_kg: 30.0,
                currency: "USD".to_string(),
                source_id: "fixture-process-source".to_string(),
                artifact_sha256: DIGEST.to_string(),
            },
        }
    }

    #[test]
    fn atomic_fraction_is_converted_to_mass_fraction_before_costing() {
        let result = evaluate_material_technoeconomics(&base_input(1.0)).unwrap();
        let ti = &result.elemental_contributions[0];
        let ta = &result.elemental_contributions[1];
        assert!((ti.mass_fraction - 0.209_195_311_1).abs() < 1.0e-9);
        assert!((ta.mass_fraction - 0.790_804_688_9).abs() < 1.0e-9);
        assert!((ti.mass_fraction + ta.mass_fraction - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn yield_loss_increases_required_feedstock_and_cost() {
        let result = evaluate_material_technoeconomics(&base_input(1.0)).unwrap();
        assert!((result.required_feedstock_mass_kg - 1.25).abs() < 1.0e-12);
        assert!((result.raw_material_cost - 101.465_527_50).abs() < 1.0e-8);
        assert!((result.process_cost - 30.0).abs() < 1.0e-12);
        assert!((result.total_direct_cost - 131.465_527_50).abs() < 1.0e-8);
        assert!((result.known_critical_material_mass_kg - 0.988_505_861_2).abs() < 1.0e-9);
    }

    #[test]
    fn missing_price_is_not_interpreted_as_zero() {
        let mut input = base_input(1.0);
        input.prices.pop();
        assert_eq!(
            evaluate_material_technoeconomics(&input),
            Err(TechnoeconomicError::MissingPrice(73))
        );
    }

    #[test]
    fn oxide_quote_is_not_silently_used_as_metal_feedstock_price() {
        let mut input = base_input(1.0);
        input.prices[1].basis = PriceBasis::Oxide;
        assert_eq!(
            evaluate_material_technoeconomics(&input),
            Err(TechnoeconomicError::IncompatiblePriceBasis {
                atomic_number: 73,
                basis: PriceBasis::Oxide,
            })
        );
    }

    #[test]
    fn thin_functional_layer_scales_down_cost_and_critical_mass() {
        let bulk = evaluate_material_technoeconomics(&base_input(1.0)).unwrap();
        let thin = evaluate_material_technoeconomics(&base_input(0.01)).unwrap();
        assert!((thin.total_direct_cost / bulk.total_direct_cost - 0.01).abs() < 1.0e-12);
        assert!(
            (thin.known_critical_material_mass_kg / bulk.known_critical_material_mass_kg - 0.01)
                .abs()
                < 1.0e-12
        );
    }

    #[test]
    fn absent_criticality_classification_remains_unknown() {
        let mut input = base_input(1.0);
        input.criticality.retain(|record| record.atomic_number != 22);
        let result = evaluate_material_technoeconomics(&input).unwrap();
        assert_eq!(result.unknown_criticality_elements, vec![22]);
    }

    #[test]
    fn mixed_currency_is_rejected() {
        let mut input = base_input(1.0);
        input.prices[0].currency = "EUR".to_string();
        assert!(matches!(
            evaluate_material_technoeconomics(&input),
            Err(TechnoeconomicError::MixedCurrency { .. })
        ));
    }

    #[test]
    fn pareto_front_removes_strictly_dominated_points() {
        let points = vec![
            EconomicParetoPoint {
                point_id: "A".to_string(),
                total_direct_cost: 100.0,
                process_energy_kwh: 50.0,
                known_critical_material_mass_kg: 1.0,
            },
            EconomicParetoPoint {
                point_id: "B".to_string(),
                total_direct_cost: 120.0,
                process_energy_kwh: 55.0,
                known_critical_material_mass_kg: 1.2,
            },
            EconomicParetoPoint {
                point_id: "C".to_string(),
                total_direct_cost: 90.0,
                process_energy_kwh: 70.0,
                known_critical_material_mass_kg: 0.8,
            },
        ];
        let front = economic_pareto_front(&points).unwrap();
        let ids: HashSet<_> = front.iter().map(|point| point.point_id.as_str()).collect();
        assert!(ids.contains("A"));
        assert!(ids.contains("C"));
        assert!(!ids.contains("B"));
    }
}
