// SPDX-License-Identifier: AGPL-3.0-or-later
//! Regenerative industrial-closure contracts for long-lived maritime infrastructure.
//!
//! This module models recurring dependency demand, ordinary local production,
//! recycling, stockpiles and capability dependencies. It is descriptive rather
//! than a manufacturing controller. In particular, a dependency marked
//! [`DependencyGovernance::SafeguardedExternal`] may be stockpiled but is
//! structurally forbidden from claiming ordinary local production or recycling.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const MAX_MODEL_ITEMS: usize = 4096;
const MAX_ID_LEN: usize = 256;
const MAX_EVIDENCE_LEN: usize = 1024;
const BASIS_POINTS_SCALE: u128 = 10_000;

/// Governance boundary applied to a regenerative dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DependencyGovernance {
    /// Ordinary industrial dependency that may be produced or recycled locally.
    Ordinary,
    /// Dependency intentionally supplied only through a separately safeguarded
    /// external process. Ordinary local production/recycling is prohibited.
    SafeguardedExternal,
}

/// Coarse type of dependency represented by the closure model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RegenerativeDependencyKind {
    /// Physical feedstock or bulk material.
    Material,
    /// Manufactured component or replaceable module.
    Component,
    /// Machine tool or production-equipment capacity.
    Tooling,
    /// Calibration, inspection or measurement capacity.
    Metrology,
    /// Consumable or other process input.
    ProcessInput,
    /// Energy or supporting infrastructure service.
    EnergyService,
    /// Software, design or another reproducible digital artifact.
    SoftwareArtifact,
    /// Other externally provided service.
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
    /// Governance boundary.
    pub governance: DependencyGovernance,
    /// Units consumed during one model period.
    pub demand_units_per_period: u64,
    /// Units ordinary local production can create during one period.
    pub local_production_units_per_period: u64,
    /// Units recoverable by recycling during one period.
    pub recycling_units_per_period: u64,
    /// Qualified units stockpiled when autonomous operation begins.
    pub stockpile_units: u64,
    /// Optional mass of one unit, in grams, for physical-mass closure metrics.
    /// Non-physical services/software use `None`.
    pub unit_mass_grams: Option<u64>,
    /// Opaque binding to evidence supporting the modeled rates/inventory.
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

    /// Locally regenerated units per period, excluding stockpile.
    pub fn local_flow_units_per_period(&self) -> u128 {
        u128::from(self.local_production_units_per_period)
            + u128::from(self.recycling_units_per_period)
    }

    /// Autonomous horizon under the static rates represented by this dependency.
    ///
    /// `IndefiniteUnderStaticModel` means recurring modeled local flow meets
    /// recurring modeled demand; it is not a perpetual-operation claim.
    pub fn autonomous_horizon(&self) -> Result<RegenerativeHorizon, RegenerativeClosureError> {
        self.validate()?;
        let demand = u128::from(self.demand_units_per_period);
        let local = self.local_flow_units_per_period();
        if local >= demand {
            return Ok(RegenerativeHorizon::IndefiniteUnderStaticModel);
        }
        let deficit = demand - local;
        let periods = u128::from(self.stockpile_units) / deficit;
        Ok(RegenerativeHorizon::FinitePeriods(periods as u64))
    }
}

/// Capability whose availability depends on all named dependencies remaining viable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeCapability {
    /// Canonical capability identifier.
    pub capability_id: String,
    /// Whether loss of this capability ends the modeled autonomous system/mission.
    pub essential: bool,
    /// Dependencies required by this capability.
    pub dependency_ids: BTreeSet<String>,
    /// Opaque evidence/design binding for the capability definition.
    pub evidence_binding: String,
}

impl RegenerativeCapability {
    /// Validate standalone capability shape.
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

/// Static industrial-closure model for one infrastructure configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeClosureModel {
    /// Canonical model identifier.
    pub model_id: String,
    /// Duration of one demand/production period in milliseconds.
    pub period_duration_ms: u64,
    /// Recurring dependencies considered by the model.
    pub dependencies: Vec<RegenerativeDependency>,
    /// Capabilities evaluated from those dependencies.
    pub capabilities: Vec<RegenerativeCapability>,
    /// Opaque evidence/version binding for the complete model definition.
    pub evidence_binding: String,
}

impl RegenerativeClosureModel {
    /// Validate identifiers, references, cardinality and governance boundaries.
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

    /// Evaluate regenerative flow metrics and capability survival horizons.
    pub fn evaluate(&self) -> Result<RegenerativeClosureReport, RegenerativeClosureError> {
        self.validate()?;
        let dependencies: BTreeMap<&str, &RegenerativeDependency> = self
            .dependencies
            .iter()
            .map(|dependency| (dependency.dependency_id.as_str(), dependency))
            .collect();

        let mut required_mass = 0_u128;
        let mut regenerated_mass = 0_u128;
        let mut ordinary_count = 0_u128;
        let mut ordinary_closed = 0_u128;
        let mut safeguarded_external_dependency_ids = Vec::new();

        for dependency in &self.dependencies {
            let horizon = dependency.autonomous_horizon()?;
            match dependency.governance {
                DependencyGovernance::Ordinary => {
                    ordinary_count += 1;
                    if horizon == RegenerativeHorizon::IndefiniteUnderStaticModel {
                        ordinary_closed += 1;
                    }
                }
                DependencyGovernance::SafeguardedExternal => {
                    safeguarded_external_dependency_ids.push(dependency.dependency_id.clone());
                }
            }

            if let Some(unit_mass) = dependency.unit_mass_grams {
                let unit_mass = u128::from(unit_mass);
                let demand_mass = u128::from(dependency.demand_units_per_period)
                    .checked_mul(unit_mass)
                    .ok_or(RegenerativeClosureError::ArithmeticOverflow)?;
                required_mass = required_mass
                    .checked_add(demand_mass)
                    .ok_or(RegenerativeClosureError::ArithmeticOverflow)?;

                let regenerated_units = dependency
                    .local_flow_units_per_period()
                    .min(u128::from(dependency.demand_units_per_period));
                let local_mass = regenerated_units
                    .checked_mul(unit_mass)
                    .ok_or(RegenerativeClosureError::ArithmeticOverflow)?;
                regenerated_mass = regenerated_mass
                    .checked_add(local_mass)
                    .ok_or(RegenerativeClosureError::ArithmeticOverflow)?;
            }
        }

        let physical_mass_flow_closure_basis_points =
            basis_points(regenerated_mass, required_mass)?;
        let ordinary_dependency_flow_closure_basis_points =
            basis_points(ordinary_closed, ordinary_count)?;

        let mut capability_viability = Vec::with_capacity(self.capabilities.len());
        for capability in &self.capabilities {
            let mut finite_min = None;
            let mut limiting = Vec::new();
            for dependency_id in &capability.dependency_ids {
                let dependency = dependencies.get(dependency_id.as_str()).ok_or_else(|| {
                    RegenerativeClosureError::UnknownDependencyReference {
                        capability_id: capability.capability_id.clone(),
                        dependency_id: dependency_id.clone(),
                    }
                })?;
                if let RegenerativeHorizon::FinitePeriods(periods) = dependency.autonomous_horizon()?
                {
                    match finite_min {
                        None => {
                            finite_min = Some(periods);
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
            capability_viability.push(CapabilityViability {
                capability_id: capability.capability_id.clone(),
                essential: capability.essential,
                horizon: finite_min.map_or(
                    RegenerativeHorizon::IndefiniteUnderStaticModel,
                    RegenerativeHorizon::FinitePeriods,
                ),
                limiting_dependency_ids: limiting,
            });
        }

        let essential_limit = capability_viability
            .iter()
            .filter(|capability| capability.essential)
            .filter_map(|capability| match capability.horizon {
                RegenerativeHorizon::FinitePeriods(periods) => Some(periods),
                RegenerativeHorizon::IndefiniteUnderStaticModel => None,
            })
            .min();

        let mut system_limiting = BTreeSet::new();
        if let Some(limit) = essential_limit {
            for capability in capability_viability.iter().filter(|capability| capability.essential) {
                if capability.horizon == RegenerativeHorizon::FinitePeriods(limit) {
                    system_limiting.extend(capability.limiting_dependency_ids.iter().cloned());
                }
            }
        }

        Ok(RegenerativeClosureReport {
            model_id: self.model_id.clone(),
            period_duration_ms: self.period_duration_ms,
            physical_mass_flow_closure_basis_points,
            ordinary_dependency_flow_closure_basis_points,
            essential_horizon: essential_limit.map_or(
                RegenerativeHorizon::IndefiniteUnderStaticModel,
                RegenerativeHorizon::FinitePeriods,
            ),
            limiting_dependency_ids: system_limiting.into_iter().collect(),
            safeguarded_external_dependency_ids,
            capability_viability,
        })
    }
}

/// Calculate a ratio in basis points without silently saturating intermediate arithmetic.
fn basis_points(
    numerator: u128,
    denominator: u128,
) -> Result<Option<u16>, RegenerativeClosureError> {
    if denominator == 0 {
        return Ok(None);
    }
    let scaled = numerator
        .checked_mul(BASIS_POINTS_SCALE)
        .ok_or(RegenerativeClosureError::ArithmeticOverflow)?;
    let ratio = (scaled / denominator).min(BASIS_POINTS_SCALE);
    let ratio = u16::try_from(ratio).map_err(|_| RegenerativeClosureError::ArithmeticOverflow)?;
    Ok(Some(ratio))
}

/// Autonomous horizon under the static rates and stockpiles in a closure model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegenerativeHorizon {
    /// Number of complete model periods sustainable without external supply.
    FinitePeriods(u64),
    /// Modeled recurring local flow meets recurring demand.
    IndefiniteUnderStaticModel,
}

/// Derived viability result for one capability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapabilityViability {
    /// Capability identifier from the source model.
    pub capability_id: String,
    /// Whether this capability participates in the system-level horizon.
    pub essential: bool,
    /// Autonomous horizon of this capability.
    pub horizon: RegenerativeHorizon,
    /// Dependencies establishing the finite horizon, if any.
    pub limiting_dependency_ids: Vec<String>,
}

/// Derived closure report. This is diagnostic evidence, not manufacturing authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeClosureReport {
    /// Source model identifier.
    pub model_id: String,
    /// Duration of each reported period in milliseconds.
    pub period_duration_ms: u64,
    /// Regenerated share of recurring physical mass demand, in basis points.
    /// Stockpile inventory is deliberately excluded from regenerative flow.
    pub physical_mass_flow_closure_basis_points: Option<u16>,
    /// Share of ordinary dependency categories fully closed by recurring local flow.
    pub ordinary_dependency_flow_closure_basis_points: Option<u16>,
    /// Earliest finite loss horizon among essential capabilities.
    pub essential_horizon: RegenerativeHorizon,
    /// Dependencies responsible for that system-level finite horizon.
    pub limiting_dependency_ids: Vec<String>,
    /// Dependencies intentionally outside ordinary industrial closure.
    pub safeguarded_external_dependency_ids: Vec<String>,
    /// Per-capability viability results.
    pub capability_viability: Vec<CapabilityViability>,
}

/// Validation/evaluation errors for regenerative closure models.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegenerativeClosureError {
    /// Identifier is empty, padded, contains control characters or is too long.
    InvalidIdentifier { field: &'static str },
    /// Evidence binding is empty, padded, contains control characters or is too long.
    InvalidEvidenceBinding,
    /// Model period has zero duration.
    ZeroPeriodDuration,
    /// Model contains no dependencies.
    NoDependencies,
    /// Model contains no capabilities.
    NoCapabilities,
    /// Model contains no essential capability.
    NoEssentialCapabilities,
    /// Model exceeds bounded dependency/capability cardinality.
    ModelTooLarge,
    /// Dependency has zero recurring demand.
    ZeroDemand { dependency_id: String },
    /// A dependency marked as physical has zero unit mass.
    ZeroPhysicalMass { dependency_id: String },
    /// Safeguarded dependency incorrectly claims ordinary local supply.
    SafeguardedDependencyClaimsLocalSupply { dependency_id: String },
    /// Dependency identifier is duplicated.
    DuplicateDependency { dependency_id: String },
    /// Capability identifier is duplicated.
    DuplicateCapability { capability_id: String },
    /// Capability declares no dependencies.
    CapabilityHasNoDependencies { capability_id: String },
    /// Capability refers to a dependency absent from the same model.
    UnknownDependencyReference {
        capability_id: String,
        dependency_id: String,
    },
    /// Integer arithmetic overflowed while deriving aggregate closure metrics.
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
        RegenerativeClosureModel {
            model_id: "manta-closure-v1".into(),
            period_duration_ms: YEAR_MS,
            dependencies: vec![
                dependency(
                    "structural-steel",
                    DependencyGovernance::Ordinary,
                    1_000_000,
                    950_000,
                    50_000,
                    0,
                    Some(1_000),
                ),
                dependency(
                    "control-electronics",
                    DependencyGovernance::Ordinary,
                    100,
                    90,
                    0,
                    1_000,
                    Some(10),
                ),
                dependency(
                    "qualified-reactor-fuel-service",
                    DependencyGovernance::SafeguardedExternal,
                    1,
                    0,
                    0,
                    30,
                    Some(1),
                ),
            ],
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
        assert_eq!(report.physical_mass_flow_closure_basis_points, Some(9_999));
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
    fn recycling_can_close_an_ordinary_dependency() {
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
    fn stockpile_extends_horizon_but_not_regenerative_flow() {
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
    fn safeguarded_dependency_cannot_claim_ordinary_local_fuel_cycle() {
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
    fn adding_local_supply_cannot_shorten_horizon() {
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
        assert_eq!(baseline.autonomous_horizon().unwrap(), RegenerativeHorizon::FinitePeriods(20));
        assert_eq!(improved.autonomous_horizon().unwrap(), RegenerativeHorizon::FinitePeriods(50));
    }

    #[test]
    fn extreme_ratio_arithmetic_fails_closed_instead_of_saturating() {
        let mut extreme = model();
        extreme.dependencies = vec![dependency(
            "extreme-mass",
            DependencyGovernance::Ordinary,
            u64::MAX,
            u64::MAX,
            0,
            0,
            Some(u64::MAX),
        )];
        extreme.capabilities[0].dependency_ids = BTreeSet::from(["extreme-mass".into()]);
        assert_eq!(
            extreme.evaluate(),
            Err(RegenerativeClosureError::ArithmeticOverflow)
        );
    }
}
