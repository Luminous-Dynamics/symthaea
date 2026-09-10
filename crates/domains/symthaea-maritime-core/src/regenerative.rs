// SPDX-License-Identifier: AGPL-3.0-or-later
//! Regenerative industrial-closure contracts for long-lived maritime infrastructure.
//!
//! This module answers a deliberately narrower question than manufacturing control:
//! given a set of recurring dependencies, local production/recycling capacity and
//! stockpiles, how much of an essential capability set is regenerative and what
//! dependency bounds autonomous operation when external supply disappears?
//!
//! The model is descriptive and conservative. It does not prescribe mining,
//! enrichment, fuel fabrication, reactor operation, semiconductor fabrication or
//! any other physical process. Dependencies marked [`DependencyGovernance::SafeguardedExternal`]
//! are structurally forbidden from claiming ordinary local production or recycling.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const MAX_MODEL_ITEMS: usize = 4096;
const MAX_ID_LEN: usize = 256;
const MAX_EVIDENCE_LEN: usize = 1024;

/// Governance boundary applied to a regenerative dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DependencyGovernance {
    /// Ordinary industrial dependency that may be produced or recycled locally.
    Ordinary,
    /// Dependency intentionally serviced only through a separately safeguarded
    /// external process. The ordinary closure model may stockpile it but may not
    /// claim local production or recycling for it.
    SafeguardedExternal,
}

/// Coarse kind of dependency represented by the closure graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RegenerativeDependencyKind {
    /// Physical feedstock or bulk material.
    Material,
    /// Manufactured component or replaceable module.
    Component,
    /// Machine tool or production equipment capacity.
    Tooling,
    /// Calibration, inspection or measurement capacity.
    Metrology,
    /// Consumable or process input required by manufacturing.
    ProcessInput,
    /// Energy or other infrastructure service.
    EnergyService,
    /// Software, design or other reproducible digital artifact.
    SoftwareArtifact,
    /// Externally provided service that does not fit the other categories.
    ExternalService,
}

/// One recurring dependency required by the regenerative system.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeDependency {
    /// Canonical dependency identifier.
    pub dependency_id: String,
    /// Dependency category.
    pub kind: RegenerativeDependencyKind,
    /// Governance boundary for this dependency.
    pub governance: DependencyGovernance,
    /// Units consumed during one model period.
    pub demand_units_per_period: u64,
    /// Units that ordinary local production can create during one period.
    pub local_production_units_per_period: u64,
    /// Units recoverable from recycling during one period.
    pub recycling_units_per_period: u64,
    /// Qualified units already stockpiled when autonomous operation begins.
    pub stockpile_units: u64,
    /// Optional physical mass of one unit, in grams, for mass-flow closure metrics.
    /// Non-physical services and software should use `None`.
    pub unit_mass_grams: Option<u64>,
    /// Opaque binding to the evidence supporting the modeled rates/inventory.
    pub evidence_binding: String,
}

impl RegenerativeDependency {
    /// Validate shape and governance invariants.
    pub fn validate(&self) -> Result<(), RegenerativeClosureError> {
        validate_id("dependency_id", &self.dependency_id)?;
        validate_evidence(&self.evidence_binding)?;

        if self.demand_units_per_period == 0 {
            return Err(RegenerativeClosureError::ZeroDemand {
                dependency_id: self.dependency_id.clone(),
            });
        }
        if self.unit_mass_grams == Some(0) {
            return Err(RegenerativeClosureError::ZeroPhysicalMass {
                dependency_id: self.dependency_id.clone(),
            });
        }
        if self.governance == DependencyGovernance::SafeguardedExternal
            && (self.local_production_units_per_period != 0
                || self.recycling_units_per_period != 0)
        {
            return Err(RegenerativeClosureError::SafeguardedDependencyClaimsLocalSupply {
                dependency_id: self.dependency_id.clone(),
            });
        }
        Ok(())
    }

    /// Locally regenerated units available per model period, excluding stockpile.
    pub fn local_flow_units_per_period(&self) -> u128 {
        u128::from(self.local_production_units_per_period)
            + u128::from(self.recycling_units_per_period)
    }

    /// Autonomous horizon for this dependency under the static rates in the model.
    ///
    /// A result of [`RegenerativeHorizon::IndefiniteUnderStaticModel`] means only
    /// that modeled local production plus recycling meets recurring demand. It is
    /// not a claim that the real physical system can operate forever.
    pub fn autonomous_horizon(&self) -> Result<RegenerativeHorizon, RegenerativeClosureError> {
        self.validate()?;
        let demand = u128::from(self.demand_units_per_period);
        let local_flow = self.local_flow_units_per_period();
        if local_flow >= demand {
            return Ok(RegenerativeHorizon::IndefiniteUnderStaticModel);
        }

        let deficit = demand - local_flow;
        let periods = u128::from(self.stockpile_units) / deficit;
        Ok(RegenerativeHorizon::FinitePeriods(periods as u64))
    }
}

/// Capability whose continued availability depends on one or more dependencies.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeCapability {
    /// Canonical capability identifier.
    pub capability_id: String,
    /// Whether loss of this capability ends the modeled autonomous mission/system.
    pub essential: bool,
    /// Dependencies that must all remain available for this capability.
    pub dependency_ids: BTreeSet<String>,
    /// Opaque binding to the evidence/design definition for this capability.
    pub evidence_binding: String,
}

impl RegenerativeCapability {
    /// Validate the capability's standalone shape.
    pub fn validate(&self) -> Result<(), RegenerativeClosureError> {
        validate_id("capability_id", &self.capability_id)?;
        validate_evidence(&self.evidence_binding)?;
        if self.dependency_ids.is_empty() {
            return Err(RegenerativeClosureError::CapabilityHasNoDependencies {
                capability_id: self.capability_id.clone(),
            });
        }
        for dependency_id in &self.dependency_ids {
            validate_id("dependency_id", dependency_id)?;
        }
        Ok(())
    }
}

/// Version-agnostic static industrial-closure model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeClosureModel {
    /// Canonical model identifier.
    pub model_id: String,
    /// Duration of one demand/production period in milliseconds.
    pub period_duration_ms: u64,
    /// Recurring dependencies considered by the model.
    pub dependencies: Vec<RegenerativeDependency>,
    /// Capabilities whose viability is evaluated from the dependency set.
    pub capabilities: Vec<RegenerativeCapability>,
    /// Opaque evidence/version binding for the complete model definition.
    pub evidence_binding: String,
}

impl RegenerativeClosureModel {
    /// Validate identifiers, references, cardinality and governance invariants.
    pub fn validate(&self) -> Result<(), RegenerativeClosureError> {
        validate_id("model_id", &self.model_id)?;
        validate_evidence(&self.evidence_binding)?;
        if self.period_duration_ms == 0 {
            return Err(RegenerativeClosureError::ZeroPeriodDuration);
        }
        if self.dependencies.is_empty() {
            return Err(RegenerativeClosureError::NoDependencies);
        }
        if self.capabilities.is_empty() {
            return Err(RegenerativeClosureError::NoCapabilities);
        }
        if self.dependencies.len() > MAX_MODEL_ITEMS || self.capabilities.len() > MAX_MODEL_ITEMS {
            return Err(RegenerativeClosureError::ModelTooLarge);
        }

        let mut dependency_ids = BTreeSet::new();
        for dependency in &self.dependencies {
            dependency.validate()?;
            if !dependency_ids.insert(dependency.dependency_id.clone()) {
                return Err(RegenerativeClosureError::DuplicateDependency {
                    dependency_id: dependency.dependency_id.clone(),
                });
            }
        }

        let mut capability_ids = BTreeSet::new();
        let mut has_essential = false;
        for capability in &self.capabilities {
            capability.validate()?;
            has_essential |= capability.essential;
            if !capability_ids.insert(capability.capability_id.clone()) {
                return Err(RegenerativeClosureError::DuplicateCapability {
                    capability_id: capability.capability_id.clone(),
                });
            }
            for dependency_id in &capability.dependency_ids {
                if !dependency_ids.contains(dependency_id) {
                    return Err(RegenerativeClosureError::UnknownDependencyReference {
                        capability_id: capability.capability_id.clone(),
                        dependency_id: dependency_id.clone(),
                    });
                }
            }
        }
        if !has_essential {
            return Err(RegenerativeClosureError::NoEssentialCapabilities);
        }
        Ok(())
    }

    /// Evaluate mass-flow closure and the autonomous horizon of each capability.
    pub fn evaluate(&self) -> Result<RegenerativeClosureReport, RegenerativeClosureError> {
        self.validate()?;

        let dependency_map: BTreeMap<&str, &RegenerativeDependency> = self
            .dependencies
            .iter()
            .map(|dependency| (dependency.dependency_id.as_str(), dependency))
            .collect();

        let mut required_physical_mass = 0_u128;
        let mut regenerated_physical_mass = 0_u128;
        let mut ordinary_count = 0_u128;
        let mut ordinary_closed_count = 0_u128;
        let mut safeguarded_external_dependency_ids = Vec::new();

        for dependency in &self.dependencies {
            if dependency.governance == DependencyGovernance::Ordinary {
                ordinary_count += 1;
                if matches!(
                    dependency.autonomous_horizon()?,
                    RegenerativeHorizon::IndefiniteUnderStaticModel
                ) {
                    ordinary_closed_count += 1;
                }
            } else {
                safeguarded_external_dependency_ids.push(dependency.dependency_id.clone());
            }

            if let Some(unit_mass_grams) = dependency.unit_mass_grams {
                let unit_mass = u128::from(unit_mass_grams);
                let demand_mass = u128::from(dependency.demand_units_per_period)
                    .checked_mul(unit_mass)
                    .ok_or(RegenerativeClosureError::ArithmeticOverflow)?;
                required_physical_mass = required_physical_mass
                    .checked_add(demand_mass)
                    .ok_or(RegenerativeClosureError::ArithmeticOverflow)?;

                let regenerated_units = dependency
                    .local_flow_units_per_period()
                    .min(u128::from(dependency.demand_units_per_period));
                let regenerated_mass = regenerated_units
                    .checked_mul(unit_mass)
                    .ok_or(RegenerativeClosureError::ArithmeticOverflow)?;
                regenerated_physical_mass = regenerated_physical_mass
                    .checked_add(regenerated_mass)
                    .ok_or(RegenerativeClosureError::ArithmeticOverflow)?;
            }
        }

        let physical_mass_flow_closure_basis_points = if required_physical_mass == 0 {
            None
        } else {
            Some(
                ((regenerated_physical_mass * 10_000) / required_physical_mass)
                    .min(10_000) as u16,
            )
        };

        let ordinary_dependency_flow_closure_basis_points = if ordinary_count == 0 {
            None
        } else {
            Some(((ordinary_closed_count * 10_000) / ordinary_count) as u16)
        };

        let mut capability_viability = Vec::with_capacity(self.capabilities.len());
        for capability in &self.capabilities {
            let mut finite_min: Option<u64> = None;
            let mut limiting = Vec::new();

            for dependency_id in &capability.dependency_ids {
                let dependency = dependency_map[dependency_id.as_str()];
                if let RegenerativeHorizon::FinitePeriods(periods) = dependency.autonomous_horizon()?
                {
                    match finite_min {
                        None => {
                            finite_min = Some(periods);
                            limiting.clear();
                            limiting.push(dependency_id.clone());
                        }
                        Some(current) if periods < current => {
                            finite_min = Some(periods);
                            limiting.clear();
                            limiting.push(dependency_id.clone());
                        }
                        Some(current) if periods == current => limiting.push(dependency_id.clone()),
                        Some(_) => {}
                    }
                }
            }

            let horizon = finite_min.map_or(
                RegenerativeHorizon::IndefiniteUnderStaticModel,
                RegenerativeHorizon::FinitePeriods,
            );
            capability_viability.push(CapabilityViability {
                capability_id: capability.capability_id.clone(),
                essential: capability.essential,
                horizon,
                limiting_dependency_ids: limiting,
            });
        }

        let essential_finite_min = capability_viability
            .iter()
            .filter(|capability| capability.essential)
            .filter_map(|capability| match capability.horizon {
                RegenerativeHorizon::FinitePeriods(periods) => Some(periods),
                RegenerativeHorizon::IndefiniteUnderStaticModel => None,
            })
            .min();

        let essential_horizon = essential_finite_min.map_or(
            RegenerativeHorizon::IndefiniteUnderStaticModel,
            RegenerativeHorizon::FinitePeriods,
        );

        let mut limiting_dependency_ids = BTreeSet::new();
        if let Some(limit) = essential_finite_min {
            for capability in capability_viability.iter().filter(|capability| capability.essential) {
                if capability.horizon == RegenerativeHorizon::FinitePeriods(limit) {
                    limiting_dependency_ids.extend(capability.limiting_dependency_ids.iter().cloned());
                }
            }
        }

        Ok(RegenerativeClosureReport {
            model_id: self.model_id.clone(),
            period_duration_ms: self.period_duration_ms,
            physical_mass_flow_closure_basis_points,
            ordinary_dependency_flow_closure_basis_points,
            essential_horizon,
            limiting_dependency_ids: limiting_dependency_ids.into_iter().collect(),
            safeguarded_external_dependency_ids,
            capability_viability,
        })
    }
}

/// Autonomous horizon under the static rates and stockpiles in a closure model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegenerativeHorizon {
    /// Number of complete model periods that can be sustained without external supply.
    FinitePeriods(u64),
    /// Modeled regenerative flow meets recurring demand.
    /// This is not a real-world perpetual-operation claim.
    IndefiniteUnderStaticModel,
}

/// Derived viability result for one capability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapabilityViability {
    /// Capability identifier from the source model.
    pub capability_id: String,
    /// Whether the capability participates in the system-level horizon.
    pub essential: bool,
    /// Autonomous horizon of the capability.
    pub horizon: RegenerativeHorizon,
    /// Dependencies that establish the finite horizon, if any.
    pub limiting_dependency_ids: Vec<String>,
}

/// Derived closure report. This is diagnostic evidence, not authority to manufacture.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeClosureReport {
    /// Source model identifier.
    pub model_id: String,
    /// Duration of each reported period in milliseconds.
    pub period_duration_ms: u64,
    /// Regenerated share of recurring physical mass demand, in basis points.
    /// Stockpiles do not count as regenerative flow.
    pub physical_mass_flow_closure_basis_points: Option<u16>,
    /// Share of ordinary dependency categories whose recurring local flow fully
    /// covers recurring demand, in basis points.
    pub ordinary_dependency_flow_closure_basis_points: Option<u16>,
    /// Earliest modeled loss horizon among essential capabilities.
    pub essential_horizon: RegenerativeHorizon,
    /// Dependencies responsible for the system-level finite horizon.
    pub limiting_dependency_ids: Vec<String>,
    /// Dependencies intentionally outside ordinary local industrial closure.
    pub safeguarded_external_dependency_ids: Vec<String>,
    /// Per-capability viability details.
    pub capability_viability: Vec<CapabilityViability>,
}

/// Validation/evaluation errors for regenerative closure models.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegenerativeClosureError {
    /// Required identifier is empty, padded, contains control characters or is too long.
    InvalidIdentifier { field: &'static str },
    /// Evidence binding is empty, padded, contains control characters or is too long.
    InvalidEvidenceBinding,
    /// Model period has zero duration.
    ZeroPeriodDuration,
    /// Model contains no dependencies.
    NoDependencies,
    /// Model contains no capabilities.
    NoCapabilities,
    /// Model contains no capability marked essential.
    NoEssentialCapabilities,
    /// Model exceeds the bounded dependency/capability cardinality.
    ModelTooLarge,
    /// Dependency has zero recurring demand.
    ZeroDemand { dependency_id: String },
    /// A dependency marked as physical has zero unit mass.
    ZeroPhysicalMass { dependency_id: String },
    /// Safeguarded dependency incorrectly claims ordinary local production/recycling.
    SafeguardedDependencyClaimsLocalSupply { dependency_id: String },
    /// Dependency identifier is duplicated.
    DuplicateDependency { dependency_id: String },
    /// Capability identifier is duplicated.
    DuplicateCapability { capability_id: String },
    /// Capability does not declare any dependencies.
    CapabilityHasNoDependencies { capability_id: String },
    /// Capability refers to a dependency absent from the same model.
    UnknownDependencyReference {
        capability_id: String,
        dependency_id: String,
    },
    /// Bounded integer arithmetic overflowed while deriving aggregate mass metrics.
    ArithmeticOverflow,
}

fn validate_id(field: &'static str, value: &str) -> Result<(), RegenerativeClosureError> {
    if value.is_empty()
        || value.len() > MAX_ID_LEN
        || value.trim() != value
        || value.chars().any(char::is_control)
    {
        return Err(RegenerativeClosureError::InvalidIdentifier { field });
    }
    Ok(())
}

fn validate_evidence(value: &str) -> Result<(), RegenerativeClosureError> {
    if value.is_empty()
        || value.len() > MAX_EVIDENCE_LEN
        || value.trim() != value
        || value.chars().any(char::is_control)
    {
        return Err(RegenerativeClosureError::InvalidEvidenceBinding);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const YEAR_MS: u64 = 31_557_600_000;

    fn dependency(
        id: &str,
        governance: DependencyGovernance,
        demand: u64,
        production: u64,
        recycling: u64,
        stockpile: u64,
        unit_mass_grams: Option<u64>,
    ) -> RegenerativeDependency {
        RegenerativeDependency {
            dependency_id: id.into(),
            kind: RegenerativeDependencyKind::Component,
            governance,
            demand_units_per_period: demand,
            local_production_units_per_period: production,
            recycling_units_per_period: recycling,
            stockpile_units: stockpile,
            unit_mass_grams,
            evidence_binding: format!("evidence:{id}"),
        }
    }

    fn model() -> RegenerativeClosureModel {
        let steel = dependency(
            "structural-steel",
            DependencyGovernance::Ordinary,
            1_000_000,
            950_000,
            50_000,
            0,
            Some(1_000),
        );
        let electronics = dependency(
            "control-electronics",
            DependencyGovernance::Ordinary,
            100,
            90,
            0,
            1_000,
            Some(10),
        );
        let nuclear_service = dependency(
            "qualified-reactor-fuel-service",
            DependencyGovernance::SafeguardedExternal,
            1,
            0,
            0,
            30,
            Some(1),
        );
        RegenerativeClosureModel {
            model_id: "manta-closure-v1".into(),
            period_duration_ms: YEAR_MS,
            dependencies: vec![steel, electronics, nuclear_service],
            capabilities: vec![RegenerativeCapability {
                capability_id: "persistent-ocean-infrastructure".into(),
                essential: true,
                dependency_ids: BTreeSet::from([
                    "structural-steel".into(),
                    "control-electronics".into(),
                    "qualified-reactor-fuel-service".into(),
                ]),
                evidence_binding: "evidence:capability".into(),
            }],
            evidence_binding: "evidence:model".into(),
        }
    }

    #[test]
    fn high_mass_closure_can_still_have_a_finite_critical_horizon() {
        let report = model().evaluate().unwrap();
        assert!(report.physical_mass_flow_closure_basis_points.unwrap() > 9_999);
        assert_eq!(report.essential_horizon, RegenerativeHorizon::FinitePeriods(30));
        assert_eq!(
            report.limiting_dependency_ids,
            vec!["qualified-reactor-fuel-service".to_string()]
        );
        assert_eq!(
            report.safeguarded_external_dependency_ids,
            vec!["qualified-reactor-fuel-service".to_string()]
        );
    }

    #[test]
    fn local_flow_and_recycling_can_close_an_ordinary_dependency() {
        let dep = dependency(
            "steel",
            DependencyGovernance::Ordinary,
            100,
            80,
            20,
            0,
            Some(1_000),
        );
        assert_eq!(
            dep.autonomous_horizon().unwrap(),
            RegenerativeHorizon::IndefiniteUnderStaticModel
        );
    }

    #[test]
    fn stockpile_only_extends_the_finite_horizon_and_not_closure_rate() {
        let dep = dependency(
            "electronics",
            DependencyGovernance::Ordinary,
            100,
            90,
            0,
            250,
            Some(10),
        );
        assert_eq!(
            dep.autonomous_horizon().unwrap(),
            RegenerativeHorizon::FinitePeriods(25)
        );
    }

    #[test]
    fn safeguarded_external_dependency_cannot_claim_ordinary_local_fuel_cycle() {
        let dep = dependency(
            "reactor-fuel",
            DependencyGovernance::SafeguardedExternal,
            1,
            1,
            0,
            10,
            Some(1),
        );
        assert_eq!(
            dep.validate(),
            Err(RegenerativeClosureError::SafeguardedDependencyClaimsLocalSupply {
                dependency_id: "reactor-fuel".into(),
            })
        );
    }

    #[test]
    fn unknown_dependency_reference_fails_closed() {
        let mut model = model();
        model.capabilities[0]
            .dependency_ids
            .insert("imaginary-supplier".into());
        assert_eq!(
            model.validate(),
            Err(RegenerativeClosureError::UnknownDependencyReference {
                capability_id: "persistent-ocean-infrastructure".into(),
                dependency_id: "imaginary-supplier".into(),
            })
        );
    }

    #[test]
    fn duplicate_ids_and_noncanonical_evidence_are_rejected() {
        let mut duplicate = model();
        duplicate.dependencies.push(duplicate.dependencies[0].clone());
        assert_eq!(
            duplicate.validate(),
            Err(RegenerativeClosureError::DuplicateDependency {
                dependency_id: "structural-steel".into(),
            })
        );

        let mut malformed = model();
        malformed.dependencies[0].evidence_binding = " evidence:steel".into();
        assert_eq!(
            malformed.validate(),
            Err(RegenerativeClosureError::InvalidEvidenceBinding)
        );
    }

    #[test]
    fn adding_supply_cannot_shorten_a_dependency_horizon() {
        let baseline = dependency(
            "electronics",
            DependencyGovernance::Ordinary,
            100,
            50,
            0,
            1_000,
            Some(10),
        );
        let improved = dependency(
            "electronics",
            DependencyGovernance::Ordinary,
            100,
            70,
            10,
            1_000,
            Some(10),
        );
        let baseline_horizon = match baseline.autonomous_horizon().unwrap() {
            RegenerativeHorizon::FinitePeriods(periods) => periods,
            RegenerativeHorizon::IndefiniteUnderStaticModel => u64::MAX,
        };
        let improved_horizon = match improved.autonomous_horizon().unwrap() {
            RegenerativeHorizon::FinitePeriods(periods) => periods,
            RegenerativeHorizon::IndefiniteUnderStaticModel => u64::MAX,
        };
        assert!(improved_horizon >= baseline_horizon);
    }
}
