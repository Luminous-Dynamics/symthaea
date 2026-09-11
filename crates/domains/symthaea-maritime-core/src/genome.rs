// SPDX-License-Identifier: AGPL-3.0-or-later
//! Versioned, recipe-free design/capability manifests for regenerative infrastructure.
//!
//! A genome describes what a successor platform or subsystem must prove in order
//! to preserve required capabilities. It deliberately does not contain toolpaths,
//! process setpoints, actuator commands, or manufacturing recipes.

use crate::{
    DependencyGovernance, RegenerativeClosureError, RegenerativeClosureModel,
    RegenerativeHorizon,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Current regenerative-genome schema version.
pub const REGENERATIVE_GENOME_SCHEMA_V1: u8 = 1;

const MAX_MODEL_ITEMS: usize = 4096;
const MAX_ID_LEN: usize = 256;
const MAX_BINDING_LEN: usize = 1024;
const MAX_SUBSTITUTION_BINDINGS: usize = 64;

/// One design requirement needed to preserve a named capability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeGenomeRequirementV1 {
    /// Stable requirement identifier inside this genome.
    pub requirement_id: String,
    /// Capability in the bound closure model that this requirement serves.
    pub capability_id: String,
    /// Baseline closure dependency satisfying this requirement.
    pub baseline_dependency_id: String,
    /// Opaque binding to the design/BOM/capability definition being qualified.
    pub design_binding: String,
    /// Opaque binding to the metrology/inspection profile required for acceptance.
    pub metrology_profile_binding: String,
    /// Opaque binding to the requalification criteria for return to service.
    pub requalification_profile_binding: String,
    /// Opaque binding to the intended disassembly/decommissioning profile.
    pub disassembly_profile_binding: String,
    /// Opaque binding to material/component recovery expectations.
    pub recovery_profile_binding: String,
    /// Explicitly qualified substitution-evidence bindings, sorted and unique.
    ///
    /// These are references only. They do not cause automatic substitution.
    pub qualified_substitution_bindings: Vec<String>,
}

impl RegenerativeGenomeRequirementV1 {
    fn validate_shape(&self) -> Result<(), RegenerativeGenomeError> {
        validate_id("requirement_id", &self.requirement_id)?;
        validate_id("capability_id", &self.capability_id)?;
        validate_id("baseline_dependency_id", &self.baseline_dependency_id)?;
        for binding in [
            &self.design_binding,
            &self.metrology_profile_binding,
            &self.requalification_profile_binding,
            &self.disassembly_profile_binding,
            &self.recovery_profile_binding,
        ] {
            validate_binding(binding)?;
        }
        if self.qualified_substitution_bindings.len() > MAX_SUBSTITUTION_BINDINGS {
            return Err(RegenerativeGenomeError::TooManySubstitutionBindings {
                requirement_id: self.requirement_id.clone(),
            });
        }
        for binding in &self.qualified_substitution_bindings {
            validate_binding(binding)?;
        }
        if self
            .qualified_substitution_bindings
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            return Err(RegenerativeGenomeError::NonCanonicalSubstitutionBindings {
                requirement_id: self.requirement_id.clone(),
            });
        }
        Ok(())
    }
}

/// Versioned design/capability manifest for one regenerative successor design.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeGenomeV1 {
    /// Exact schema version.
    pub schema_version: u8,
    /// Canonical genome identifier.
    pub genome_id: String,
    /// Optional opaque binding to the parent genome/lineage evidence.
    pub lineage_parent_binding: Option<String>,
    /// Exact closure model identifier this genome was evaluated against.
    pub closure_model_id: String,
    /// Exact evidence/version binding of that closure model.
    pub closure_model_evidence_binding: String,
    /// Requirements that preserve the design's capabilities.
    pub requirements: Vec<RegenerativeGenomeRequirementV1>,
    /// Opaque evidence/version binding for this complete genome definition.
    pub evidence_binding: String,
}

impl RegenerativeGenomeV1 {
    /// Validate this genome against one exact closure model.
    ///
    /// Every dependency of every essential capability must have one and only one
    /// genome requirement. Non-essential capability dependencies may be omitted.
    pub fn validate_against_model(
        &self,
        model: &RegenerativeClosureModel,
    ) -> Result<(), RegenerativeGenomeError> {
        if self.schema_version != REGENERATIVE_GENOME_SCHEMA_V1 {
            return Err(RegenerativeGenomeError::UnsupportedSchemaVersion {
                schema_version: self.schema_version,
            });
        }
        validate_id("genome_id", &self.genome_id)?;
        validate_id("closure_model_id", &self.closure_model_id)?;
        validate_binding(&self.closure_model_evidence_binding)?;
        validate_binding(&self.evidence_binding)?;
        if let Some(parent) = &self.lineage_parent_binding {
            validate_binding(parent)?;
            if parent == &self.evidence_binding {
                return Err(RegenerativeGenomeError::SelfParentGenomeLineage);
            }
        }
        if self.requirements.is_empty() {
            return Err(RegenerativeGenomeError::NoRequirements);
        }
        if self.requirements.len() > MAX_MODEL_ITEMS {
            return Err(RegenerativeGenomeError::GenomeTooLarge);
        }

        model
            .validate()
            .map_err(RegenerativeGenomeError::ClosureModelInvalid)?;
        if self.closure_model_id != model.model_id
            || self.closure_model_evidence_binding != model.evidence_binding
        {
            return Err(RegenerativeGenomeError::ClosureModelBindingMismatch);
        }

        let dependencies: BTreeMap<&str, _> = model
            .dependencies
            .iter()
            .map(|dependency| (dependency.dependency_id.as_str(), dependency))
            .collect();
        let capabilities: BTreeMap<&str, _> = model
            .capabilities
            .iter()
            .map(|capability| (capability.capability_id.as_str(), capability))
            .collect();

        let mut requirement_ids = BTreeSet::new();
        let mut covered_pairs = BTreeSet::new();
        let mut has_substitution_evidence = false;
        for requirement in &self.requirements {
            requirement.validate_shape()?;
            has_substitution_evidence |= !requirement.qualified_substitution_bindings.is_empty();
            if !requirement_ids.insert(requirement.requirement_id.clone()) {
                return Err(RegenerativeGenomeError::DuplicateRequirement {
                    requirement_id: requirement.requirement_id.clone(),
                });
            }
            let capability = capabilities
                .get(requirement.capability_id.as_str())
                .ok_or_else(|| RegenerativeGenomeError::UnknownCapability {
                    capability_id: requirement.capability_id.clone(),
                })?;
            let dependency = dependencies
                .get(requirement.baseline_dependency_id.as_str())
                .ok_or_else(|| RegenerativeGenomeError::UnknownDependency {
                    dependency_id: requirement.baseline_dependency_id.clone(),
                })?;
            if !capability
                .dependency_ids
                .contains(&requirement.baseline_dependency_id)
            {
                return Err(RegenerativeGenomeError::DependencyNotRequiredByCapability {
                    capability_id: requirement.capability_id.clone(),
                    dependency_id: requirement.baseline_dependency_id.clone(),
                });
            }
            if dependency.governance == DependencyGovernance::SafeguardedExternal
                && !requirement.qualified_substitution_bindings.is_empty()
            {
                return Err(RegenerativeGenomeError::SafeguardedRequirementClaimsSubstitution {
                    requirement_id: requirement.requirement_id.clone(),
                });
            }
            let pair = (
                requirement.capability_id.clone(),
                requirement.baseline_dependency_id.clone(),
            );
            if !covered_pairs.insert(pair.clone()) {
                return Err(RegenerativeGenomeError::DuplicateRequirementCoverage {
                    capability_id: pair.0,
                    dependency_id: pair.1,
                });
            }
        }
        if has_substitution_evidence && self.lineage_parent_binding.is_none() {
            return Err(RegenerativeGenomeError::SubstitutionDerivedGenomeRequiresParent);
        }
        if self
            .requirements
            .windows(2)
            .any(|pair| pair[0].requirement_id >= pair[1].requirement_id)
        {
            return Err(RegenerativeGenomeError::NonCanonicalRequirementOrder);
        }

        for capability in model.capabilities.iter().filter(|capability| capability.essential) {
            for dependency_id in &capability.dependency_ids {
                if !covered_pairs.contains(&(capability.capability_id.clone(), dependency_id.clone())) {
                    return Err(RegenerativeGenomeError::MissingEssentialRequirement {
                        capability_id: capability.capability_id.clone(),
                        dependency_id: dependency_id.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Evaluate how long this exact genome remains supportable under the bound model.
    ///
    /// Qualified substitution references are audit metadata only here. A substitute
    /// changes supportability only after an explicit substitution has produced a
    /// separately identified closure model.
    pub fn evaluate_supportability(
        &self,
        model: &RegenerativeClosureModel,
    ) -> Result<RegenerativeGenomeSupportabilityReport, RegenerativeGenomeError> {
        self.validate_against_model(model)?;
        let dependencies: BTreeMap<&str, _> = model
            .dependencies
            .iter()
            .map(|dependency| (dependency.dependency_id.as_str(), dependency))
            .collect();
        let capabilities: BTreeMap<&str, _> = model
            .capabilities
            .iter()
            .map(|capability| (capability.capability_id.as_str(), capability))
            .collect();

        let mut requirements = Vec::with_capacity(self.requirements.len());
        let mut essential_limit: Option<u64> = None;
        for requirement in &self.requirements {
            let dependency = dependencies
                .get(requirement.baseline_dependency_id.as_str())
                .ok_or_else(|| RegenerativeGenomeError::UnknownDependency {
                    dependency_id: requirement.baseline_dependency_id.clone(),
                })?;
            let capability = capabilities
                .get(requirement.capability_id.as_str())
                .ok_or_else(|| RegenerativeGenomeError::UnknownCapability {
                    capability_id: requirement.capability_id.clone(),
                })?;
            let horizon = dependency
                .autonomous_horizon()
                .map_err(RegenerativeGenomeError::ClosureModelInvalid)?;
            if capability.essential {
                if let RegenerativeHorizon::FinitePeriods(periods) = horizon {
                    essential_limit = Some(essential_limit.map_or(periods, |current| current.min(periods)));
                }
            }
            requirements.push(RegenerativeGenomeRequirementSupportability {
                requirement_id: requirement.requirement_id.clone(),
                capability_id: requirement.capability_id.clone(),
                baseline_dependency_id: requirement.baseline_dependency_id.clone(),
                essential: capability.essential,
                horizon,
                qualified_substitution_count: requirement.qualified_substitution_bindings.len() as u16,
            });
        }

        let essential_horizon = essential_limit.map_or(
            RegenerativeHorizon::IndefiniteUnderStaticModel,
            RegenerativeHorizon::FinitePeriods,
        );
        let limiting_requirement_ids = match essential_limit {
            Some(limit) => requirements
                .iter()
                .filter(|requirement| {
                    requirement.essential
                        && requirement.horizon == RegenerativeHorizon::FinitePeriods(limit)
                })
                .map(|requirement| requirement.requirement_id.clone())
                .collect(),
            None => Vec::new(),
        };

        Ok(RegenerativeGenomeSupportabilityReport {
            genome_id: self.genome_id.clone(),
            closure_model_id: self.closure_model_id.clone(),
            essential_horizon,
            limiting_requirement_ids,
            requirements,
        })
    }
}

/// Supportability result for one genome requirement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeGenomeRequirementSupportability {
    /// Requirement identifier.
    pub requirement_id: String,
    /// Capability identifier.
    pub capability_id: String,
    /// Baseline closure dependency.
    pub baseline_dependency_id: String,
    /// Whether the owning capability is essential.
    pub essential: bool,
    /// Static support horizon of the baseline dependency.
    pub horizon: RegenerativeHorizon,
    /// Number of explicitly qualified substitution references carried by the genome.
    pub qualified_substitution_count: u16,
}

/// Diagnostic supportability report for one genome/closure-model pairing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeGenomeSupportabilityReport {
    /// Genome identifier.
    pub genome_id: String,
    /// Closure-model identifier.
    pub closure_model_id: String,
    /// Earliest finite horizon among essential genome requirements.
    pub essential_horizon: RegenerativeHorizon,
    /// Requirements establishing the system-level finite horizon.
    pub limiting_requirement_ids: Vec<String>,
    /// Per-requirement diagnostic supportability.
    pub requirements: Vec<RegenerativeGenomeRequirementSupportability>,
}

/// Validation/evaluation errors for regenerative genomes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegenerativeGenomeError {
    /// Unsupported genome schema version.
    UnsupportedSchemaVersion { schema_version: u8 },
    /// Identifier is empty, padded, contains controls, or is too long.
    InvalidIdentifier { field: &'static str },
    /// Evidence/profile binding is malformed.
    InvalidBinding,
    /// Genome points to its own evidence binding as its parent.
    SelfParentGenomeLineage,
    /// A genome that cites qualified substitution evidence must name a parent lineage.
    SubstitutionDerivedGenomeRequiresParent,
    /// Genome has no requirements.
    NoRequirements,
    /// Genome exceeds bounded cardinality.
    GenomeTooLarge,
    /// Bound closure model itself is invalid.
    ClosureModelInvalid(RegenerativeClosureError),
    /// Genome names a different closure model ID or evidence binding.
    ClosureModelBindingMismatch,
    /// Requirement identifier is duplicated.
    DuplicateRequirement { requirement_id: String },
    /// Requirement vector is not strictly sorted by requirement ID.
    NonCanonicalRequirementOrder,
    /// Requirement names an unknown capability.
    UnknownCapability { capability_id: String },
    /// Requirement names an unknown dependency.
    UnknownDependency { dependency_id: String },
    /// Requirement names a dependency the capability does not require.
    DependencyNotRequiredByCapability {
        capability_id: String,
        dependency_id: String,
    },
    /// More than one requirement claims the same capability/dependency pair.
    DuplicateRequirementCoverage {
        capability_id: String,
        dependency_id: String,
    },
    /// Essential capability dependency is missing from the genome.
    MissingEssentialRequirement {
        capability_id: String,
        dependency_id: String,
    },
    /// Requirement carries too many substitution references.
    TooManySubstitutionBindings { requirement_id: String },
    /// Substitution references are not sorted and unique.
    NonCanonicalSubstitutionBindings { requirement_id: String },
    /// Generic substitution references are forbidden for safeguarded dependencies.
    SafeguardedRequirementClaimsSubstitution { requirement_id: String },
}

fn validate_id(field: &'static str, value: &str) -> Result<(), RegenerativeGenomeError> {
    if value.is_empty()
        || value.len() > MAX_ID_LEN
        || value.trim() != value
        || value.chars().any(char::is_control)
    {
        Err(RegenerativeGenomeError::InvalidIdentifier { field })
    } else {
        Ok(())
    }
}

fn validate_binding(value: &str) -> Result<(), RegenerativeGenomeError> {
    if value.is_empty()
        || value.len() > MAX_BINDING_LEN
        || value.trim() != value
        || value.chars().any(char::is_control)
    {
        Err(RegenerativeGenomeError::InvalidBinding)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        RegenerativeCapability, RegenerativeDependency, RegenerativeDependencyKind,
    };

    fn model() -> RegenerativeClosureModel {
        RegenerativeClosureModel {
            model_id: "manta-forge-v1".into(),
            period_duration_ms: 1,
            dependencies: vec![
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
                    dependency_id: "electronics".into(),
                    kind: RegenerativeDependencyKind::Component,
                    governance: DependencyGovernance::Ordinary,
                    demand_units_per_period: 10,
                    local_production_units_per_period: 9,
                    recycling_units_per_period: 0,
                    stockpile_units: 1_000,
                    unit_mass_grams: Some(10),
                    evidence_binding: "dep:electronics".into(),
                },
                RegenerativeDependency {
                    dependency_id: "qualified-reactor-fuel-service".into(),
                    kind: RegenerativeDependencyKind::ExternalService,
                    governance: DependencyGovernance::SafeguardedExternal,
                    demand_units_per_period: 1,
                    local_production_units_per_period: 0,
                    recycling_units_per_period: 0,
                    stockpile_units: 30,
                    unit_mass_grams: Some(1),
                    evidence_binding: "dep:reactor-service".into(),
                },
            ],
            capabilities: vec![RegenerativeCapability {
                capability_id: "persistent-ocean-infrastructure".into(),
                essential: true,
                dependency_ids: BTreeSet::from([
                    "structural-material".into(),
                    "electronics".into(),
                    "qualified-reactor-fuel-service".into(),
                ]),
                evidence_binding: "cap:persistent-ocean-infrastructure".into(),
            }],
            evidence_binding: "model:manta-forge-v1".into(),
        }
    }

    fn requirement(id: &str, dependency_id: &str) -> RegenerativeGenomeRequirementV1 {
        RegenerativeGenomeRequirementV1 {
            requirement_id: id.into(),
            capability_id: "persistent-ocean-infrastructure".into(),
            baseline_dependency_id: dependency_id.into(),
            design_binding: format!("design:{id}"),
            metrology_profile_binding: format!("metrology:{id}"),
            requalification_profile_binding: format!("requalification:{id}"),
            disassembly_profile_binding: format!("disassembly:{id}"),
            recovery_profile_binding: format!("recovery:{id}"),
            qualified_substitution_bindings: Vec::new(),
        }
    }

    fn genome() -> RegenerativeGenomeV1 {
        RegenerativeGenomeV1 {
            schema_version: REGENERATIVE_GENOME_SCHEMA_V1,
            genome_id: "manta-genome-v1".into(),
            lineage_parent_binding: Some("genome-lineage:root".into()),
            closure_model_id: "manta-forge-v1".into(),
            closure_model_evidence_binding: "model:manta-forge-v1".into(),
            requirements: vec![
                requirement("req-electronics", "electronics"),
                requirement("req-reactor-service", "qualified-reactor-fuel-service"),
                requirement("req-structure", "structural-material"),
            ],
            evidence_binding: "genome:manta-v1".into(),
        }
    }

    #[test]
    fn complete_genome_reports_safeguarded_dependency_as_design_horizon() {
        let report = genome().evaluate_supportability(&model()).unwrap();
        assert_eq!(report.essential_horizon, RegenerativeHorizon::FinitePeriods(30));
        assert_eq!(report.limiting_requirement_ids, vec!["req-reactor-service"]);
    }

    #[test]
    fn every_essential_capability_dependency_requires_genome_coverage() {
        let mut genome = genome();
        genome.requirements.retain(|requirement| requirement.requirement_id != "req-electronics");
        assert_eq!(
            genome.validate_against_model(&model()),
            Err(RegenerativeGenomeError::MissingEssentialRequirement {
                capability_id: "persistent-ocean-infrastructure".into(),
                dependency_id: "electronics".into(),
            })
        );
    }

    #[test]
    fn genome_cannot_name_itself_as_parent() {
        let mut genome = genome();
        genome.lineage_parent_binding = Some(genome.evidence_binding.clone());
        assert_eq!(
            genome.validate_against_model(&model()),
            Err(RegenerativeGenomeError::SelfParentGenomeLineage)
        );
    }

    #[test]
    fn substitution_derived_genome_requires_parent_lineage() {
        let mut genome = genome();
        genome.lineage_parent_binding = None;
        genome.requirements[0].qualified_substitution_bindings =
            vec!["substitution:electronics-local-v1".into()];
        assert_eq!(
            genome.validate_against_model(&model()),
            Err(RegenerativeGenomeError::SubstitutionDerivedGenomeRequiresParent)
        );
    }

    #[test]
    fn requirement_order_is_canonical() {
        let mut genome = genome();
        genome.requirements.swap(0, 1);
        assert_eq!(
            genome.validate_against_model(&model()),
            Err(RegenerativeGenomeError::NonCanonicalRequirementOrder)
        );
    }

    #[test]
    fn safeguarded_requirement_cannot_advertise_generic_substitution() {
        let mut genome = genome();
        genome.requirements[1].qualified_substitution_bindings = vec!["substitution:reactor".into()];
        assert_eq!(
            genome.validate_against_model(&model()),
            Err(RegenerativeGenomeError::SafeguardedRequirementClaimsSubstitution {
                requirement_id: "req-reactor-service".into(),
            })
        );
    }

    #[test]
    fn closure_model_binding_must_match_exactly() {
        let mut genome = genome();
        genome.closure_model_evidence_binding = "model:other".into();
        assert_eq!(
            genome.validate_against_model(&model()),
            Err(RegenerativeGenomeError::ClosureModelBindingMismatch)
        );
    }

    #[test]
    fn duplicate_capability_dependency_coverage_is_rejected() {
        let mut genome = genome();
        genome
            .requirements
            .push(requirement("req-structure-copy", "structural-material"));
        assert_eq!(
            genome.validate_against_model(&model()),
            Err(RegenerativeGenomeError::DuplicateRequirementCoverage {
                capability_id: "persistent-ocean-infrastructure".into(),
                dependency_id: "structural-material".into(),
            })
        );
    }

    #[test]
    fn substitutions_are_references_only_and_do_not_change_baseline_horizon() {
        let mut genome = genome();
        genome.requirements[0].qualified_substitution_bindings = vec![
            "substitution:electronics-a".into(),
            "substitution:electronics-b".into(),
        ];
        let report = genome.evaluate_supportability(&model()).unwrap();
        let electronics = report
            .requirements
            .iter()
            .find(|requirement| requirement.requirement_id == "req-electronics")
            .unwrap();
        assert_eq!(electronics.qualified_substitution_count, 2);
        assert_eq!(electronics.horizon, RegenerativeHorizon::FinitePeriods(1_000));
    }

    #[test]
    fn indefinitely_supportable_genome_has_no_finite_limiter() {
        let mut model = model();
        for dependency in &mut model.dependencies {
            if dependency.governance == DependencyGovernance::Ordinary {
                dependency.local_production_units_per_period = dependency.demand_units_per_period;
                dependency.recycling_units_per_period = 0;
            } else {
                dependency.governance = DependencyGovernance::Ordinary;
                dependency.local_production_units_per_period = dependency.demand_units_per_period;
            }
        }
        let mut genome = genome();
        genome.requirements[1].qualified_substitution_bindings.clear();
        let report = genome.evaluate_supportability(&model).unwrap();
        assert_eq!(report.essential_horizon, RegenerativeHorizon::IndefiniteUnderStaticModel);
        assert!(report.limiting_requirement_ids.is_empty());
    }
}
