// SPDX-License-Identifier: AGPL-3.0-or-later
//! Recipe-free support evidence for regenerative production and recycling claims.
//!
//! A closure model may claim recurring local production or recycling. This module
//! requires every positive local flow to be backed by explicit capability/evidence
//! support and either modeled prerequisite dependencies or an explicit opaque
//! external-input binding. It contains no process recipes, machine settings, or
//! physical control instructions.

use crate::{DependencyGovernance, RegenerativeClosureModel};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Current regenerative-flow-support schema version.
pub const REGENERATIVE_FLOW_SUPPORT_SCHEMA_V1: u8 = 1;

const MAX_ITEMS: usize = 4096;
const MAX_ID_LEN: usize = 256;
const MAX_BINDING_LEN: usize = 1024;

/// Which recurring local flow a support claim qualifies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegenerativeFlowKindV1 {
    /// Ordinary local production claimed by the closure model.
    Production,
    /// Ordinary local recycling/recovery flow claimed by the closure model.
    Recycling,
}

/// Evidence supporting one positive recurring local flow.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeFlowSupportClaimV1 {
    /// Dependency whose recurring local flow is being supported.
    pub dependency_id: String,
    /// Whether this claim supports production or recycling.
    pub flow_kind: RegenerativeFlowKindV1,
    /// Opaque binding to the production/recycling capability definition.
    pub capability_binding: String,
    /// Closure-model dependencies required to sustain this local flow.
    /// Sorted and unique.
    pub prerequisite_dependency_ids: Vec<String>,
    /// Optional opaque binding to a required environmental/exogenous input not
    /// represented as a closure-model dependency.
    pub external_input_binding: Option<String>,
    /// Opaque metrology/inspection evidence for the claimed output.
    pub metrology_binding: String,
    /// Opaque qualification evidence for the claimed local capability.
    pub qualification_binding: String,
    /// Optional evidence establishing how a cyclic dependency network is bootstrapped.
    pub bootstrap_binding: Option<String>,
}

impl RegenerativeFlowSupportClaimV1 {
    fn validate_shape(&self) -> Result<(), RegenerativeFlowSupportError> {
        validate_id("dependency_id", &self.dependency_id)?;
        validate_binding(&self.capability_binding)?;
        validate_binding(&self.metrology_binding)?;
        validate_binding(&self.qualification_binding)?;
        if let Some(binding) = &self.external_input_binding {
            validate_binding(binding)?;
        }
        if let Some(binding) = &self.bootstrap_binding {
            validate_binding(binding)?;
        }
        if self.prerequisite_dependency_ids.is_empty() && self.external_input_binding.is_none() {
            return Err(RegenerativeFlowSupportError::UngroundedFlowClaim {
                dependency_id: self.dependency_id.clone(),
                flow_kind: self.flow_kind,
            });
        }
        if self.prerequisite_dependency_ids.len() > MAX_ITEMS {
            return Err(RegenerativeFlowSupportError::TooManyPrerequisites {
                dependency_id: self.dependency_id.clone(),
            });
        }
        for prerequisite in &self.prerequisite_dependency_ids {
            validate_id("prerequisite_dependency_id", prerequisite)?;
            if prerequisite == &self.dependency_id {
                return Err(RegenerativeFlowSupportError::DirectSelfDependency {
                    dependency_id: self.dependency_id.clone(),
                    flow_kind: self.flow_kind,
                });
            }
        }
        if self
            .prerequisite_dependency_ids
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            return Err(RegenerativeFlowSupportError::NonCanonicalPrerequisites {
                dependency_id: self.dependency_id.clone(),
                flow_kind: self.flow_kind,
            });
        }
        Ok(())
    }
}

/// Complete support graph for the positive local flows in one closure model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeFlowSupportV1 {
    /// Exact schema version.
    pub schema_version: u8,
    /// Canonical support-graph identifier.
    pub support_id: String,
    /// Exact closure-model identifier.
    pub closure_model_id: String,
    /// Exact closure-model evidence/version binding.
    pub closure_model_evidence_binding: String,
    /// Positive-flow support claims, strictly ordered by `(dependency_id, flow_kind)`.
    pub claims: Vec<RegenerativeFlowSupportClaimV1>,
    /// Opaque evidence/version binding for this complete support graph.
    pub evidence_binding: String,
}

impl RegenerativeFlowSupportV1 {
    /// Validate complete positive-flow coverage and bootstrap-safe support cycles.
    pub fn validate_against_model(
        &self,
        model: &RegenerativeClosureModel,
    ) -> Result<(), RegenerativeFlowSupportError> {
        if self.schema_version != REGENERATIVE_FLOW_SUPPORT_SCHEMA_V1 {
            return Err(RegenerativeFlowSupportError::UnsupportedSchemaVersion {
                schema_version: self.schema_version,
            });
        }
        validate_id("support_id", &self.support_id)?;
        validate_id("closure_model_id", &self.closure_model_id)?;
        validate_binding(&self.closure_model_evidence_binding)?;
        validate_binding(&self.evidence_binding)?;
        if self.claims.len() > MAX_ITEMS {
            return Err(RegenerativeFlowSupportError::TooManyClaims);
        }
        model
            .validate()
            .map_err(|_| RegenerativeFlowSupportError::ClosureModelInvalid)?;
        if self.closure_model_id != model.model_id
            || self.closure_model_evidence_binding != model.evidence_binding
        {
            return Err(RegenerativeFlowSupportError::ClosureModelBindingMismatch);
        }

        let dependencies: BTreeMap<&str, _> = model
            .dependencies
            .iter()
            .map(|dependency| (dependency.dependency_id.as_str(), dependency))
            .collect();
        let mut expected = BTreeSet::new();
        for dependency in &model.dependencies {
            if dependency.local_production_units_per_period > 0 {
                expected.insert((
                    dependency.dependency_id.clone(),
                    RegenerativeFlowKindV1::Production,
                ));
            }
            if dependency.recycling_units_per_period > 0 {
                expected.insert((
                    dependency.dependency_id.clone(),
                    RegenerativeFlowKindV1::Recycling,
                ));
            }
        }

        let mut seen = BTreeSet::new();
        for claim in &self.claims {
            claim.validate_shape()?;
            let dependency = dependencies
                .get(claim.dependency_id.as_str())
                .ok_or_else(|| RegenerativeFlowSupportError::UnknownDependency {
                    dependency_id: claim.dependency_id.clone(),
                })?;
            if dependency.governance == DependencyGovernance::SafeguardedExternal {
                return Err(RegenerativeFlowSupportError::SafeguardedDependencyClaimsOrdinaryFlow {
                    dependency_id: claim.dependency_id.clone(),
                    flow_kind: claim.flow_kind,
                });
            }
            let actual_positive = match claim.flow_kind {
                RegenerativeFlowKindV1::Production => {
                    dependency.local_production_units_per_period > 0
                }
                RegenerativeFlowKindV1::Recycling => dependency.recycling_units_per_period > 0,
            };
            if !actual_positive {
                return Err(RegenerativeFlowSupportError::ClaimForAbsentFlow {
                    dependency_id: claim.dependency_id.clone(),
                    flow_kind: claim.flow_kind,
                });
            }
            for prerequisite in &claim.prerequisite_dependency_ids {
                if !dependencies.contains_key(prerequisite.as_str()) {
                    return Err(RegenerativeFlowSupportError::UnknownPrerequisite {
                        dependency_id: claim.dependency_id.clone(),
                        prerequisite_dependency_id: prerequisite.clone(),
                    });
                }
            }
            let key = (claim.dependency_id.clone(), claim.flow_kind);
            if !seen.insert(key.clone()) {
                return Err(RegenerativeFlowSupportError::DuplicateFlowClaim {
                    dependency_id: key.0,
                    flow_kind: key.1,
                });
            }
        }
        if self.claims.windows(2).any(|pair| {
            (pair[0].dependency_id.as_str(), pair[0].flow_kind)
                >= (pair[1].dependency_id.as_str(), pair[1].flow_kind)
        }) {
            return Err(RegenerativeFlowSupportError::NonCanonicalClaimOrder);
        }
        if seen != expected {
            return Err(RegenerativeFlowSupportError::FlowCoverageMismatch {
                missing: expected.difference(&seen).next().cloned(),
                extra: seen.difference(&expected).next().cloned(),
            });
        }
        self.validate_cycles()
    }

    fn validate_cycles(&self) -> Result<(), RegenerativeFlowSupportError> {
        let supported_dependencies: BTreeSet<&str> = self
            .claims
            .iter()
            .map(|claim| claim.dependency_id.as_str())
            .collect();
        let mut adjacency: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
        for claim in &self.claims {
            let entry = adjacency.entry(claim.dependency_id.as_str()).or_default();
            for prerequisite in &claim.prerequisite_dependency_ids {
                if supported_dependencies.contains(prerequisite.as_str()) {
                    entry.insert(prerequisite.as_str());
                }
            }
        }

        let cyclic_dependencies: BTreeSet<&str> = supported_dependencies
            .iter()
            .copied()
            .filter(|dependency_id| {
                reaches(
                    dependency_id,
                    dependency_id,
                    &adjacency,
                    true,
                    &mut BTreeSet::new(),
                )
            })
            .collect();

        for dependency_id in cyclic_dependencies {
            let has_bootstrap = self.claims.iter().any(|claim| {
                claim.dependency_id == dependency_id && claim.bootstrap_binding.is_some()
            });
            if !has_bootstrap {
                return Err(RegenerativeFlowSupportError::UnbootstrappedCycle {
                    dependency_id: dependency_id.to_string(),
                });
            }
        }
        Ok(())
    }
}

fn reaches<'a>(
    origin: &'a str,
    current: &'a str,
    adjacency: &BTreeMap<&'a str, BTreeSet<&'a str>>,
    skip_zero: bool,
    visited: &mut BTreeSet<&'a str>,
) -> bool {
    if !skip_zero && current == origin {
        return true;
    }
    if !visited.insert(current) {
        return false;
    }
    adjacency.get(current).is_some_and(|nexts| {
        nexts
            .iter()
            .copied()
            .any(|next| reaches(origin, next, adjacency, false, &mut visited.clone()))
    })
}

/// Validation errors for regenerative-flow support graphs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegenerativeFlowSupportError {
    UnsupportedSchemaVersion { schema_version: u8 },
    InvalidIdentifier { field: &'static str },
    InvalidBinding,
    ClosureModelInvalid,
    ClosureModelBindingMismatch,
    TooManyClaims,
    TooManyPrerequisites { dependency_id: String },
    UngroundedFlowClaim {
        dependency_id: String,
        flow_kind: RegenerativeFlowKindV1,
    },
    DirectSelfDependency {
        dependency_id: String,
        flow_kind: RegenerativeFlowKindV1,
    },
    NonCanonicalPrerequisites {
        dependency_id: String,
        flow_kind: RegenerativeFlowKindV1,
    },
    UnknownDependency { dependency_id: String },
    UnknownPrerequisite {
        dependency_id: String,
        prerequisite_dependency_id: String,
    },
    SafeguardedDependencyClaimsOrdinaryFlow {
        dependency_id: String,
        flow_kind: RegenerativeFlowKindV1,
    },
    ClaimForAbsentFlow {
        dependency_id: String,
        flow_kind: RegenerativeFlowKindV1,
    },
    DuplicateFlowClaim {
        dependency_id: String,
        flow_kind: RegenerativeFlowKindV1,
    },
    NonCanonicalClaimOrder,
    FlowCoverageMismatch {
        missing: Option<(String, RegenerativeFlowKindV1)>,
        extra: Option<(String, RegenerativeFlowKindV1)>,
    },
    UnbootstrappedCycle { dependency_id: String },
}

fn validate_id(field: &'static str, value: &str) -> Result<(), RegenerativeFlowSupportError> {
    if value.is_empty()
        || value.len() > MAX_ID_LEN
        || value.trim() != value
        || value.chars().any(char::is_control)
    {
        Err(RegenerativeFlowSupportError::InvalidIdentifier { field })
    } else {
        Ok(())
    }
}

fn validate_binding(value: &str) -> Result<(), RegenerativeFlowSupportError> {
    if value.is_empty()
        || value.len() > MAX_BINDING_LEN
        || value.trim() != value
        || value.chars().any(char::is_control)
    {
        Err(RegenerativeFlowSupportError::InvalidBinding)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RegenerativeCapability, RegenerativeDependency, RegenerativeDependencyKind};

    fn model() -> RegenerativeClosureModel {
        RegenerativeClosureModel {
            model_id: "forge-flow-model-v1".into(),
            period_duration_ms: 1,
            dependencies: vec![
                RegenerativeDependency {
                    dependency_id: "energy-service".into(),
                    kind: RegenerativeDependencyKind::EnergyService,
                    governance: DependencyGovernance::Ordinary,
                    demand_units_per_period: 10,
                    local_production_units_per_period: 10,
                    recycling_units_per_period: 0,
                    stockpile_units: 0,
                    unit_mass_grams: None,
                    evidence_binding: "dep:energy".into(),
                },
                RegenerativeDependency {
                    dependency_id: "metrology".into(),
                    kind: RegenerativeDependencyKind::Metrology,
                    governance: DependencyGovernance::Ordinary,
                    demand_units_per_period: 1,
                    local_production_units_per_period: 1,
                    recycling_units_per_period: 0,
                    stockpile_units: 0,
                    unit_mass_grams: None,
                    evidence_binding: "dep:metrology".into(),
                },
                RegenerativeDependency {
                    dependency_id: "structural-material".into(),
                    kind: RegenerativeDependencyKind::Material,
                    governance: DependencyGovernance::Ordinary,
                    demand_units_per_period: 100,
                    local_production_units_per_period: 80,
                    recycling_units_per_period: 20,
                    stockpile_units: 0,
                    unit_mass_grams: Some(1_000),
                    evidence_binding: "dep:structure".into(),
                },
                RegenerativeDependency {
                    dependency_id: "qualified-reactor-service".into(),
                    kind: RegenerativeDependencyKind::ExternalService,
                    governance: DependencyGovernance::SafeguardedExternal,
                    demand_units_per_period: 1,
                    local_production_units_per_period: 0,
                    recycling_units_per_period: 0,
                    stockpile_units: 30,
                    unit_mass_grams: None,
                    evidence_binding: "dep:reactor".into(),
                },
            ],
            capabilities: vec![RegenerativeCapability {
                capability_id: "persistent-ocean-infrastructure".into(),
                essential: true,
                dependency_ids: BTreeSet::from([
                    "energy-service".into(),
                    "metrology".into(),
                    "qualified-reactor-service".into(),
                    "structural-material".into(),
                ]),
                evidence_binding: "cap:persistent".into(),
            }],
            evidence_binding: "model:forge-flow-v1".into(),
        }
    }

    fn claim(
        dependency_id: &str,
        flow_kind: RegenerativeFlowKindV1,
        prerequisites: &[&str],
    ) -> RegenerativeFlowSupportClaimV1 {
        RegenerativeFlowSupportClaimV1 {
            dependency_id: dependency_id.into(),
            flow_kind,
            capability_binding: format!("capability:{dependency_id}:{flow_kind:?}"),
            prerequisite_dependency_ids: prerequisites.iter().map(|value| (*value).into()).collect(),
            external_input_binding: None,
            metrology_binding: format!("metrology:{dependency_id}:{flow_kind:?}"),
            qualification_binding: format!("qualification:{dependency_id}:{flow_kind:?}"),
            bootstrap_binding: None,
        }
    }

    fn support() -> RegenerativeFlowSupportV1 {
        let mut claims = vec![
            claim("energy-service", RegenerativeFlowKindV1::Production, &["metrology"]),
            claim("metrology", RegenerativeFlowKindV1::Production, &["energy-service"]),
            claim(
                "structural-material",
                RegenerativeFlowKindV1::Production,
                &["energy-service", "metrology"],
            ),
            claim(
                "structural-material",
                RegenerativeFlowKindV1::Recycling,
                &["energy-service", "metrology"],
            ),
        ];
        claims[0].bootstrap_binding = Some("bootstrap:energy-metrology-loop".into());
        claims.sort_by(|a, b| {
            (a.dependency_id.as_str(), a.flow_kind)
                .cmp(&(b.dependency_id.as_str(), b.flow_kind))
        });
        RegenerativeFlowSupportV1 {
            schema_version: REGENERATIVE_FLOW_SUPPORT_SCHEMA_V1,
            support_id: "forge-flow-support-v1".into(),
            closure_model_id: "forge-flow-model-v1".into(),
            closure_model_evidence_binding: "model:forge-flow-v1".into(),
            claims,
            evidence_binding: "flow-support:forge-v1".into(),
        }
    }

    #[test]
    fn complete_flow_support_graph_is_accepted() {
        assert_eq!(support().validate_against_model(&model()), Ok(()));
    }

    #[test]
    fn every_positive_local_flow_requires_exactly_one_support_claim() {
        let mut support = support();
        support.claims.retain(|claim| {
            !(claim.dependency_id == "structural-material"
                && claim.flow_kind == RegenerativeFlowKindV1::Recycling)
        });
        assert!(matches!(
            support.validate_against_model(&model()),
            Err(RegenerativeFlowSupportError::FlowCoverageMismatch { .. })
        ));
    }

    #[test]
    fn ungrounded_flow_claim_is_rejected() {
        let mut support = support();
        let structural = support
            .claims
            .iter_mut()
            .find(|claim| {
                claim.dependency_id == "structural-material"
                    && claim.flow_kind == RegenerativeFlowKindV1::Production
            })
            .unwrap();
        structural.prerequisite_dependency_ids.clear();
        assert!(matches!(
            support.validate_against_model(&model()),
            Err(RegenerativeFlowSupportError::UngroundedFlowClaim { .. })
        ));
        structural.external_input_binding = Some("external-input:qualified-natural-feed".into());
        assert!(support.validate_against_model(&model()).is_ok());
    }

    #[test]
    fn cyclic_dependency_requires_explicit_bootstrap_evidence() {
        let mut support = support();
        for claim in &mut support.claims {
            claim.bootstrap_binding = None;
        }
        assert!(matches!(
            support.validate_against_model(&model()),
            Err(RegenerativeFlowSupportError::UnbootstrappedCycle { .. })
        ));
    }

    #[test]
    fn safeguarded_dependency_cannot_enter_generic_local_flow_support() {
        let mut model = model();
        let reactor = model
            .dependencies
            .iter_mut()
            .find(|dependency| dependency.dependency_id == "qualified-reactor-service")
            .unwrap();
        reactor.local_production_units_per_period = 1;
        assert!(model.validate().is_err());
    }
}
