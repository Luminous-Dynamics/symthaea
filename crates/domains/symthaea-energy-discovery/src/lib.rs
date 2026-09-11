// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Energy-domain semantics for the generic `symthaea-discovery` contracts.
//!
//! This crate intentionally contains no optimizer, physics solver, experiment
//! runner, procurement path, or deployment authority. It defines energy
//! technology vocabulary and application-dependent objective/constraint sets so
//! downstream discovery remains explicit about *what service* is being optimized.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use symthaea_discovery::{
    Candidate, CandidateId, CandidateOrigin, Constraint, ConstraintBound, DiscoveryError,
    Evaluation, Objective, ObjectiveDirection,
};
use thiserror::Error;

/// Canonical energy metric names used by discovery profiles.
pub mod metric {
    pub const GRAVIMETRIC_ENERGY_DENSITY: &str = "gravimetric_energy_density";
    pub const VOLUMETRIC_ENERGY_DENSITY: &str = "volumetric_energy_density";
    pub const POWER_DENSITY: &str = "power_density";
    pub const ROUND_TRIP_EFFICIENCY: &str = "round_trip_efficiency";
    pub const CYCLE_LIFE: &str = "cycle_life";
    pub const CALENDAR_LIFE: &str = "calendar_life";
    pub const RESPONSE_TIME: &str = "response_time";
    pub const DISCHARGE_DURATION: &str = "discharge_duration";
    pub const CAPEX_PER_ENERGY: &str = "capex_per_energy";
    pub const CAPEX_PER_POWER: &str = "capex_per_power";
    pub const SELF_DISCHARGE_PER_DAY: &str = "self_discharge_per_day";
    pub const LCOE: &str = "lcoe";
    pub const LCOH: &str = "lcoh";
    pub const CAPACITY_FACTOR: &str = "capacity_factor";
    pub const LIFECYCLE_CARBON_INTENSITY: &str = "lifecycle_carbon_intensity";
    pub const WATER_INTENSITY: &str = "water_intensity";
    pub const LAND_INTENSITY: &str = "land_intensity";
    pub const CONVERSION_EFFICIENCY: &str = "conversion_efficiency";
}

/// Canonical unit strings paired with the metrics above.
pub mod unit {
    pub const WH_PER_KG: &str = "Wh/kg";
    pub const WH_PER_L: &str = "Wh/L";
    pub const W_PER_KG: &str = "W/kg";
    pub const FRACTION: &str = "fraction";
    pub const CYCLES: &str = "cycles";
    pub const YEARS: &str = "years";
    pub const SECONDS: &str = "s";
    pub const HOURS: &str = "h";
    pub const USD_PER_KWH: &str = "USD/kWh";
    pub const USD_PER_KW: &str = "USD/kW";
    pub const FRACTION_PER_DAY: &str = "fraction/day";
    pub const USD_PER_MWH: &str = "USD/MWh";
    pub const USD_PER_MWH_THERMAL: &str = "USD/MWh_th";
    pub const KG_CO2E_PER_MWH: &str = "kgCO2e/MWh";
    pub const L_PER_MWH: &str = "L/MWh";
    pub const M2_PER_MWH: &str = "m2/MWh";
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnergyDomain {
    Generation,
    Storage,
    Conversion,
}

impl EnergyDomain {
    pub const fn as_str(self) -> &'static str {
        match self {
            EnergyDomain::Generation => "generation",
            EnergyDomain::Storage => "storage",
            EnergyDomain::Conversion => "conversion",
        }
    }
}

/// Broad technology identity only. Detailed chemistry/device/process
/// specification belongs in candidate attributes and later evidence records.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnergyTechnology {
    Photovoltaic,
    Wind,
    Hydro,
    Geothermal,
    Fission,
    /// Research classification only; this variant carries no engineering or
    /// deployment claim.
    FusionResearch,
    ElectrochemicalBattery,
    FlowBattery,
    PumpedHydro,
    ThermalStorage,
    HydrogenStorage,
    Supercapacitor,
    FuelCell,
    SyntheticFuel,
    Other { name: String, domain: EnergyDomain },
}

impl EnergyTechnology {
    pub const fn domain(&self) -> EnergyDomain {
        match self {
            EnergyTechnology::Photovoltaic
            | EnergyTechnology::Wind
            | EnergyTechnology::Hydro
            | EnergyTechnology::Geothermal
            | EnergyTechnology::Fission
            | EnergyTechnology::FusionResearch => EnergyDomain::Generation,
            EnergyTechnology::ElectrochemicalBattery
            | EnergyTechnology::FlowBattery
            | EnergyTechnology::PumpedHydro
            | EnergyTechnology::ThermalStorage
            | EnergyTechnology::HydrogenStorage
            | EnergyTechnology::Supercapacitor => EnergyDomain::Storage,
            EnergyTechnology::FuelCell | EnergyTechnology::SyntheticFuel => {
                EnergyDomain::Conversion
            }
            EnergyTechnology::Other { domain, .. } => *domain,
        }
    }

    pub fn label(&self) -> &str {
        match self {
            EnergyTechnology::Photovoltaic => "photovoltaic",
            EnergyTechnology::Wind => "wind",
            EnergyTechnology::Hydro => "hydro",
            EnergyTechnology::Geothermal => "geothermal",
            EnergyTechnology::Fission => "fission",
            EnergyTechnology::FusionResearch => "fusion_research",
            EnergyTechnology::ElectrochemicalBattery => "electrochemical_battery",
            EnergyTechnology::FlowBattery => "flow_battery",
            EnergyTechnology::PumpedHydro => "pumped_hydro",
            EnergyTechnology::ThermalStorage => "thermal_storage",
            EnergyTechnology::HydrogenStorage => "hydrogen_storage",
            EnergyTechnology::Supercapacitor => "supercapacitor",
            EnergyTechnology::FuelCell => "fuel_cell",
            EnergyTechnology::SyntheticFuel => "synthetic_fuel",
            EnergyTechnology::Other { name, .. } => name.as_str(),
        }
    }

    fn validate(&self) -> Result<(), EnergyError> {
        if let EnergyTechnology::Other { name, .. } = self {
            if name.trim().is_empty() {
                return Err(EnergyError::InvalidTechnologyName);
            }
        }
        Ok(())
    }
}

/// Service context. Context is separate from technology so the same candidate
/// can be evaluated differently for mobility, long-duration storage, or space.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnergyApplication {
    ElectricVehicle,
    GridFrequencyResponse,
    GridStorage { minimum_duration_hours: f64 },
    Microgrid { minimum_duration_hours: f64 },
    RemotePower { minimum_duration_hours: f64 },
    Spacecraft,
    BulkElectricityGeneration,
    IndustrialHeat,
}

impl EnergyApplication {
    pub fn validate(&self) -> Result<(), EnergyError> {
        match self {
            EnergyApplication::GridStorage {
                minimum_duration_hours,
            }
            | EnergyApplication::Microgrid {
                minimum_duration_hours,
            }
            | EnergyApplication::RemotePower {
                minimum_duration_hours,
            } => {
                if !minimum_duration_hours.is_finite() || *minimum_duration_hours <= 0.0 {
                    return Err(EnergyError::InvalidDuration(*minimum_duration_hours));
                }
            }
            _ => {}
        }
        Ok(())
    }
}

const RESERVED_ATTRIBUTES: [&str; 2] = ["energy_domain", "energy_technology"];

/// Typed energy description that converts into the generic discovery candidate
/// without introducing a competing candidate identity system.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnergyCandidateDescriptor {
    pub id: CandidateId,
    pub technology: EnergyTechnology,
    pub origin: CandidateOrigin,
    #[serde(default)]
    pub attributes: BTreeMap<String, String>,
}

impl EnergyCandidateDescriptor {
    pub fn new(
        id: CandidateId,
        technology: EnergyTechnology,
        origin: CandidateOrigin,
    ) -> Result<Self, EnergyError> {
        CandidateId::new(id.0.clone())?;
        technology.validate()?;
        Ok(Self {
            id,
            technology,
            origin,
            attributes: BTreeMap::new(),
        })
    }

    pub fn with_attribute(
        mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<Self, EnergyError> {
        let key = key.into();
        if key.trim().is_empty() {
            return Err(EnergyError::InvalidAttributeKey);
        }
        if RESERVED_ATTRIBUTES.contains(&key.as_str()) {
            return Err(EnergyError::ReservedAttribute(key));
        }
        self.attributes.insert(key, value.into());
        Ok(self)
    }

    /// Convert into the generic candidate contract. Public fields are
    /// revalidated here so deserialization or direct struct construction cannot
    /// bypass the descriptor's constructor checks.
    pub fn to_discovery_candidate(&self) -> Result<Candidate, EnergyError> {
        let id = CandidateId::new(self.id.0.clone())?;
        self.technology.validate()?;
        for key in self.attributes.keys() {
            if key.trim().is_empty() {
                return Err(EnergyError::InvalidAttributeKey);
            }
            if RESERVED_ATTRIBUTES.contains(&key.as_str()) {
                return Err(EnergyError::ReservedAttribute(key.clone()));
            }
        }

        let mut candidate = Candidate::new(id, "energy_technology", self.origin.clone())?;
        for (key, value) in &self.attributes {
            candidate = candidate.with_spec(key.clone(), value.clone());
        }
        candidate = candidate
            .with_spec("energy_domain", self.technology.domain().as_str())
            .with_spec("energy_technology", self.technology.label());
        Ok(candidate)
    }
}

/// Application-dependent objective/constraint set. No weights are present by
/// design: downstream Pareto methods see the separate objectives explicitly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnergyDiscoveryProfile {
    pub application: EnergyApplication,
    pub objectives: Vec<Objective>,
    pub constraints: Vec<Constraint>,
}

impl EnergyDiscoveryProfile {
    pub fn for_application(application: EnergyApplication) -> Result<Self, EnergyError> {
        application.validate()?;
        let (objectives, constraints) = match &application {
            EnergyApplication::ElectricVehicle => (
                vec![
                    maximize(metric::GRAVIMETRIC_ENERGY_DENSITY, unit::WH_PER_KG),
                    maximize(metric::VOLUMETRIC_ENERGY_DENSITY, unit::WH_PER_L),
                    maximize(metric::POWER_DENSITY, unit::W_PER_KG),
                    maximize(metric::ROUND_TRIP_EFFICIENCY, unit::FRACTION),
                    maximize(metric::CYCLE_LIFE, unit::CYCLES),
                    minimize(metric::CAPEX_PER_ENERGY, unit::USD_PER_KWH),
                ],
                vec![],
            ),
            EnergyApplication::GridFrequencyResponse => (
                vec![
                    minimize(metric::RESPONSE_TIME, unit::SECONDS),
                    maximize(metric::POWER_DENSITY, unit::W_PER_KG),
                    maximize(metric::ROUND_TRIP_EFFICIENCY, unit::FRACTION),
                    maximize(metric::CYCLE_LIFE, unit::CYCLES),
                    minimize(metric::CAPEX_PER_POWER, unit::USD_PER_KW),
                ],
                vec![],
            ),
            EnergyApplication::GridStorage {
                minimum_duration_hours,
            } => (
                vec![
                    minimize(metric::CAPEX_PER_ENERGY, unit::USD_PER_KWH),
                    minimize(metric::CAPEX_PER_POWER, unit::USD_PER_KW),
                    maximize(metric::ROUND_TRIP_EFFICIENCY, unit::FRACTION),
                    maximize(metric::CYCLE_LIFE, unit::CYCLES),
                    maximize(metric::CALENDAR_LIFE, unit::YEARS),
                    minimize(metric::SELF_DISCHARGE_PER_DAY, unit::FRACTION_PER_DAY),
                ],
                vec![at_least(
                    metric::DISCHARGE_DURATION,
                    unit::HOURS,
                    *minimum_duration_hours,
                )],
            ),
            EnergyApplication::Microgrid {
                minimum_duration_hours,
            } => (
                vec![
                    minimize(metric::CAPEX_PER_ENERGY, unit::USD_PER_KWH),
                    maximize(metric::ROUND_TRIP_EFFICIENCY, unit::FRACTION),
                    maximize(metric::CALENDAR_LIFE, unit::YEARS),
                    maximize(metric::CYCLE_LIFE, unit::CYCLES),
                    minimize(metric::RESPONSE_TIME, unit::SECONDS),
                ],
                vec![at_least(
                    metric::DISCHARGE_DURATION,
                    unit::HOURS,
                    *minimum_duration_hours,
                )],
            ),
            EnergyApplication::RemotePower {
                minimum_duration_hours,
            } => (
                vec![
                    maximize(metric::GRAVIMETRIC_ENERGY_DENSITY, unit::WH_PER_KG),
                    maximize(metric::ROUND_TRIP_EFFICIENCY, unit::FRACTION),
                    maximize(metric::CALENDAR_LIFE, unit::YEARS),
                    minimize(metric::SELF_DISCHARGE_PER_DAY, unit::FRACTION_PER_DAY),
                    minimize(metric::CAPEX_PER_ENERGY, unit::USD_PER_KWH),
                ],
                vec![at_least(
                    metric::DISCHARGE_DURATION,
                    unit::HOURS,
                    *minimum_duration_hours,
                )],
            ),
            EnergyApplication::Spacecraft => (
                vec![
                    maximize(metric::GRAVIMETRIC_ENERGY_DENSITY, unit::WH_PER_KG),
                    maximize(metric::VOLUMETRIC_ENERGY_DENSITY, unit::WH_PER_L),
                    maximize(metric::POWER_DENSITY, unit::W_PER_KG),
                    maximize(metric::CALENDAR_LIFE, unit::YEARS),
                    maximize(metric::ROUND_TRIP_EFFICIENCY, unit::FRACTION),
                    minimize(metric::SELF_DISCHARGE_PER_DAY, unit::FRACTION_PER_DAY),
                ],
                vec![],
            ),
            EnergyApplication::BulkElectricityGeneration => (
                vec![
                    minimize(metric::LCOE, unit::USD_PER_MWH),
                    maximize(metric::CAPACITY_FACTOR, unit::FRACTION),
                    minimize(
                        metric::LIFECYCLE_CARBON_INTENSITY,
                        unit::KG_CO2E_PER_MWH,
                    ),
                    minimize(metric::WATER_INTENSITY, unit::L_PER_MWH),
                    minimize(metric::LAND_INTENSITY, unit::M2_PER_MWH),
                ],
                vec![],
            ),
            EnergyApplication::IndustrialHeat => (
                vec![
                    minimize(metric::LCOH, unit::USD_PER_MWH_THERMAL),
                    maximize(metric::CONVERSION_EFFICIENCY, unit::FRACTION),
                    minimize(
                        metric::LIFECYCLE_CARBON_INTENSITY,
                        unit::KG_CO2E_PER_MWH,
                    ),
                    minimize(metric::WATER_INTENSITY, unit::L_PER_MWH),
                ],
                vec![],
            ),
        };

        let profile = Self {
            application,
            objectives,
            constraints,
        };
        profile.validate()?;
        Ok(profile)
    }

    pub fn validate(&self) -> Result<(), EnergyError> {
        self.application.validate()?;
        // Reuse the generic discovery contract's objective/constraint checks
        // rather than creating a second validation implementation here.
        Evaluation {
            candidate_id: CandidateId::new("energy-profile-validation")?,
            objectives: self.objectives.clone(),
            constraints: self.constraints.clone(),
            predictions: Vec::new(),
            pareto_rank: None,
        }
        .validate()?;
        Ok(())
    }
}

fn maximize(metric: &str, unit: &str) -> Objective {
    Objective {
        metric: metric.to_owned(),
        unit: unit.to_owned(),
        direction: ObjectiveDirection::Maximize,
    }
}

fn minimize(metric: &str, unit: &str) -> Objective {
    Objective {
        metric: metric.to_owned(),
        unit: unit.to_owned(),
        direction: ObjectiveDirection::Minimize,
    }
}

fn at_least(metric: &str, unit: &str, value: f64) -> Constraint {
    Constraint {
        metric: metric.to_owned(),
        unit: unit.to_owned(),
        bound: ConstraintBound::AtLeast(value),
    }
}

#[derive(Debug, Error)]
pub enum EnergyError {
    #[error(transparent)]
    Discovery(#[from] DiscoveryError),
    #[error("energy application duration must be finite and greater than zero, got {0}")]
    InvalidDuration(f64),
    #[error("custom energy technology name cannot be empty")]
    InvalidTechnologyName,
    #[error("energy candidate attribute key cannot be empty")]
    InvalidAttributeKey,
    #[error("energy candidate attribute {0:?} is reserved for canonical semantics")]
    ReservedAttribute(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn objective_metrics(profile: &EnergyDiscoveryProfile) -> BTreeSet<&str> {
        profile
            .objectives
            .iter()
            .map(|objective| objective.metric.as_str())
            .collect()
    }

    #[test]
    fn technology_domain_is_explicit_and_not_inferred_from_application() {
        assert_eq!(
            EnergyTechnology::Photovoltaic.domain(),
            EnergyDomain::Generation
        );
        assert_eq!(
            EnergyTechnology::FlowBattery.domain(),
            EnergyDomain::Storage
        );
        assert_eq!(EnergyTechnology::FuelCell.domain(), EnergyDomain::Conversion);
    }

    #[test]
    fn candidate_reuses_generic_identity_and_preserves_energy_semantics() {
        let descriptor = EnergyCandidateDescriptor::new(
            CandidateId::new("candidate:flow-001").unwrap(),
            EnergyTechnology::FlowBattery,
            CandidateOrigin::Generated {
                generator: "screening-study".into(),
                version: Some("v1".into()),
            },
        )
        .unwrap()
        .with_attribute("electrolyte_family", "aqueous")
        .unwrap();

        let candidate = descriptor.to_discovery_candidate().unwrap();
        assert_eq!(candidate.id.0, "candidate:flow-001");
        assert_eq!(candidate.kind, "energy_technology");
        assert_eq!(candidate.specification["energy_domain"], "storage");
        assert_eq!(candidate.specification["energy_technology"], "flow_battery");
        assert_eq!(candidate.specification["electrolyte_family"], "aqueous");
    }

    #[test]
    fn public_attributes_cannot_override_canonical_energy_identity() {
        let mut descriptor = EnergyCandidateDescriptor::new(
            CandidateId::new("candidate:1").unwrap(),
            EnergyTechnology::Wind,
            CandidateOrigin::UserProposed,
        )
        .unwrap();
        descriptor
            .attributes
            .insert("energy_domain".into(), "storage".into());

        assert!(matches!(
            descriptor.to_discovery_candidate(),
            Err(EnergyError::ReservedAttribute(_))
        ));
    }

    #[test]
    fn electric_vehicle_and_long_duration_grid_storage_are_not_one_score() {
        let ev = EnergyDiscoveryProfile::for_application(EnergyApplication::ElectricVehicle)
            .unwrap();
        let grid = EnergyDiscoveryProfile::for_application(EnergyApplication::GridStorage {
            minimum_duration_hours: 12.0,
        })
        .unwrap();

        let ev_metrics = objective_metrics(&ev);
        let grid_metrics = objective_metrics(&grid);
        assert!(ev_metrics.contains(metric::GRAVIMETRIC_ENERGY_DENSITY));
        assert!(ev_metrics.contains(metric::VOLUMETRIC_ENERGY_DENSITY));
        assert!(!grid_metrics.contains(metric::GRAVIMETRIC_ENERGY_DENSITY));
        assert!(grid_metrics.contains(metric::CALENDAR_LIFE));
        assert_ne!(ev_metrics, grid_metrics);
    }

    #[test]
    fn long_duration_requirement_is_a_service_constraint_not_a_weight() {
        let profile = EnergyDiscoveryProfile::for_application(EnergyApplication::GridStorage {
            minimum_duration_hours: 12.0,
        })
        .unwrap();

        assert_eq!(profile.constraints.len(), 1);
        let constraint = &profile.constraints[0];
        assert_eq!(constraint.metric, metric::DISCHARGE_DURATION);
        assert_eq!(constraint.unit, unit::HOURS);
        assert_eq!(constraint.bound, ConstraintBound::AtLeast(12.0));
    }

    #[test]
    fn frequency_response_prioritizes_response_and_power_not_energy_density() {
        let profile = EnergyDiscoveryProfile::for_application(
            EnergyApplication::GridFrequencyResponse,
        )
        .unwrap();
        let metrics = objective_metrics(&profile);
        assert!(metrics.contains(metric::RESPONSE_TIME));
        assert!(metrics.contains(metric::POWER_DENSITY));
        assert!(!metrics.contains(metric::GRAVIMETRIC_ENERGY_DENSITY));
    }

    #[test]
    fn bulk_generation_uses_generation_system_metrics() {
        let profile = EnergyDiscoveryProfile::for_application(
            EnergyApplication::BulkElectricityGeneration,
        )
        .unwrap();
        let metrics = objective_metrics(&profile);
        assert!(metrics.contains(metric::LCOE));
        assert!(metrics.contains(metric::CAPACITY_FACTOR));
        assert!(metrics.contains(metric::LIFECYCLE_CARBON_INTENSITY));
        assert!(!metrics.contains(metric::CYCLE_LIFE));
    }

    #[test]
    fn invalid_duration_fails_closed() {
        for duration in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                EnergyDiscoveryProfile::for_application(EnergyApplication::GridStorage {
                    minimum_duration_hours: duration,
                }),
                Err(EnergyError::InvalidDuration(_))
            ));
        }
    }

    #[test]
    fn direct_struct_construction_is_revalidated_at_conversion_boundary() {
        let descriptor = EnergyCandidateDescriptor {
            id: CandidateId(" ".into()),
            technology: EnergyTechnology::Other {
                name: " ".into(),
                domain: EnergyDomain::Storage,
            },
            origin: CandidateOrigin::UserProposed,
            attributes: BTreeMap::new(),
        };

        assert!(descriptor.to_discovery_candidate().is_err());
    }
}
