// SPDX-License-Identifier: AGPL-3.0-or-later
//! Role-separated regenerative viability for evolving platform lineages.
//!
//! A long-lived platform can remain operational after it has lost the ability to
//! construct or qualify a trustworthy successor. This module keeps those questions
//! separate and derives conservative role horizons from the existing Genome +
//! support-qualified closure evidence. It is diagnostic evidence only: it does not
//! contain manufacturing recipes, process settings, actuator commands, or authority
//! to construct, qualify, operate, or service physical systems.

use crate::{
    evaluate_genome_supported_closure, RegenerativeClosureError, RegenerativeClosureModel,
    RegenerativeFlowSupportV1, RegenerativeGenomeSupportedError,
    RegenerativeGenomeSupportedRequirementReport, RegenerativeGenomeV1, RegenerativeHorizon,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Current lineage-viability profile schema version.
pub const REGENERATIVE_LINEAGE_VIABILITY_SCHEMA_V1: u8 = 1;

const MAX_ID_LEN: usize = 256;
const MAX_BINDING_LEN: usize = 1024;
const MAX_ROLE_CAPABILITIES: usize = 4096;

/// Distinct questions that must remain true for a regenerative lineage to continue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegenerativeLineageRoleV1 {
    /// Capabilities required for the current platform to remain operational.
    Operation,
    /// Capabilities required to construct the physical successor design.
    SuccessorConstruction,
    /// Capabilities required to inspect, calibrate, and qualify that successor.
    SuccessorQualification,
}

/// Evidence-bound declaration of which capabilities answer each lineage question.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeLineageViabilityProfileV1 {
    /// Exact schema version.
    pub schema_version: u8,
    /// Canonical profile identifier.
    pub profile_id: String,
    /// Exact closure model this profile classifies.
    pub closure_model_id: String,
    /// Exact evidence/version binding of the closure model.
    pub closure_model_evidence_binding: String,
    /// Capabilities required to operate the current platform, sorted and unique.
    pub operational_capability_ids: Vec<String>,
    /// Capabilities required to construct a successor, sorted and unique.
    pub successor_construction_capability_ids: Vec<String>,
    /// Capabilities required to qualify a successor, sorted and unique.
    pub successor_qualification_capability_ids: Vec<String>,
    /// Opaque evidence/version binding for this role assignment.
    pub evidence_binding: String,
}

impl RegenerativeLineageViabilityProfileV1 {
    /// Validate exact model binding, canonical ordering, and capability references.
    pub fn validate_against_model(
        &self,
        model: &RegenerativeClosureModel,
    ) -> Result<(), RegenerativeLineageViabilityError> {
        if self.schema_version != REGENERATIVE_LINEAGE_VIABILITY_SCHEMA_V1 {
            return Err(RegenerativeLineageViabilityError::UnsupportedSchemaVersion {
                schema_version: self.schema_version,
            });
        }
        validate_id(&self.profile_id)?;
        validate_id(&self.closure_model_id)?;
        validate_binding(&self.closure_model_evidence_binding)?;
        validate_binding(&self.evidence_binding)?;
        model
            .validate()
            .map_err(RegenerativeLineageViabilityError::ClosureModelInvalid)?;
        if self.closure_model_id != model.model_id
            || self.closure_model_evidence_binding != model.evidence_binding
        {
            return Err(RegenerativeLineageViabilityError::ClosureModelBindingMismatch);
        }

        let known: BTreeSet<&str> = model
            .capabilities
            .iter()
            .map(|capability| capability.capability_id.as_str())
            .collect();
        for role in [
            RegenerativeLineageRoleV1::Operation,
            RegenerativeLineageRoleV1::SuccessorConstruction,
            RegenerativeLineageRoleV1::SuccessorQualification,
        ] {
            let capability_ids = self.capability_ids(role);
            if capability_ids.is_empty() {
                return Err(RegenerativeLineageViabilityError::EmptyRole { role });
            }
            if capability_ids.len() > MAX_ROLE_CAPABILITIES {
                return Err(RegenerativeLineageViabilityError::TooManyRoleCapabilities { role });
            }
            for capability_id in capability_ids {
                validate_id(capability_id)?;
                if !known.contains(capability_id.as_str()) {
                    return Err(RegenerativeLineageViabilityError::UnknownCapability {
                        role,
                        capability_id: capability_id.clone(),
                    });
                }
            }
            if capability_ids.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Err(RegenerativeLineageViabilityError::NonCanonicalRoleCapabilities {
                    role,
                });
            }
        }
        Ok(())
    }

    fn capability_ids(&self, role: RegenerativeLineageRoleV1) -> &[String] {
        match role {
            RegenerativeLineageRoleV1::Operation => &self.operational_capability_ids,
            RegenerativeLineageRoleV1::SuccessorConstruction => {
                &self.successor_construction_capability_ids
            }
            RegenerativeLineageRoleV1::SuccessorQualification => {
                &self.successor_qualification_capability_ids
            }
        }
    }
}

/// Conservative support result for one lineage role.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeLineageRoleReportV1 {
    /// Role being evaluated.
    pub role: RegenerativeLineageRoleV1,
    /// Earliest conservative loss horizon among requirements serving this role.
    pub conservative_horizon: RegenerativeHorizon,
    /// Genome requirements establishing the finite role bound.
    pub limiting_requirement_ids: Vec<String>,
    /// Root modeled dependencies establishing the finite role bound.
    pub root_limiting_dependency_ids: Vec<String>,
    /// True only when no requirement in this role reaches an opaque external input.
    pub fully_modeled_support: bool,
    /// Role requirements whose support paths reach opaque external inputs.
    pub externally_conditioned_requirement_ids: Vec<String>,
}

/// Role-separated Regenerative Viability Horizon for one exact Genome configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeLineageViabilityReportV1 {
    /// Source viability-profile identifier.
    pub profile_id: String,
    /// Genome being evaluated.
    pub genome_id: String,
    /// Bound closure-model identifier.
    pub closure_model_id: String,
    /// Bound flow-support graph identifier.
    pub flow_support_id: String,
    /// Current-platform operational support horizon.
    pub operation: RegenerativeLineageRoleReportV1,
    /// Physical successor-construction support horizon.
    pub successor_construction: RegenerativeLineageRoleReportV1,
    /// Successor metrology/qualification support horizon.
    pub successor_qualification: RegenerativeLineageRoleReportV1,
    /// Latest conservative horizon at which construction and qualification are both supported.
    pub successor_reproduction_horizon: RegenerativeHorizon,
    /// Latest conservative horizon at which operation, construction, and qualification all hold.
    pub regenerative_viability_horizon: RegenerativeHorizon,
    /// Roles establishing the finite regenerative viability bound.
    pub limiting_roles: Vec<RegenerativeLineageRoleV1>,
    /// True only when all three role support paths are fully modeled.
    pub fully_modeled_regenerative_viability: bool,
}

/// Structural errors while deriving lineage viability evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegenerativeLineageViabilityError {
    /// Profile schema version is unsupported.
    UnsupportedSchemaVersion { schema_version: u8 },
    /// Identifier is empty, padded, oversized, or control-bearing.
    InvalidIdentifier,
    /// Evidence binding is empty, padded, oversized, or control-bearing.
    InvalidBinding,
    /// Bound closure model itself is invalid.
    ClosureModelInvalid(RegenerativeClosureError),
    /// Profile names a different closure model ID or evidence binding.
    ClosureModelBindingMismatch,
    /// A role contains no required capabilities.
    EmptyRole { role: RegenerativeLineageRoleV1 },
    /// A role exceeds bounded capability cardinality.
    TooManyRoleCapabilities { role: RegenerativeLineageRoleV1 },
    /// Role capability identifiers are not strictly sorted and unique.
    NonCanonicalRoleCapabilities { role: RegenerativeLineageRoleV1 },
    /// Role names a capability absent from the bound closure model.
    UnknownCapability {
        role: RegenerativeLineageRoleV1,
        capability_id: String,
    },
    /// Genome/support-qualified composition failed.
    GenomeSupported(RegenerativeGenomeSupportedError),
    /// A classified role capability has no requirement in the exact Genome.
    MissingGenomeCoverage {
        role: RegenerativeLineageRoleV1,
        capability_id: String,
    },
    /// Successor does not bind to the exact parent Genome evidence.
    ParentBindingMismatch,
    /// Successor reuses the parent's Genome identity/evidence identity.
    SuccessorIdentityNotDistinct,
    /// Successor changes only identity metadata and not its requirement manifest.
    SuccessorManifestUnchanged,
}

/// Evaluate operation, successor construction, and successor qualification separately.
pub fn evaluate_regenerative_lineage_viability(
    profile: &RegenerativeLineageViabilityProfileV1,
    genome: &RegenerativeGenomeV1,
    model: &RegenerativeClosureModel,
    support: &RegenerativeFlowSupportV1,
) -> Result<RegenerativeLineageViabilityReportV1, RegenerativeLineageViabilityError> {
    profile.validate_against_model(model)?;
    let supported = evaluate_genome_supported_closure(genome, model, support)
        .map_err(RegenerativeLineageViabilityError::GenomeSupported)?;

    let operation = evaluate_role(
        profile,
        RegenerativeLineageRoleV1::Operation,
        &supported.requirements,
    )?;
    let successor_construction = evaluate_role(
        profile,
        RegenerativeLineageRoleV1::SuccessorConstruction,
        &supported.requirements,
    )?;
    let successor_qualification = evaluate_role(
        profile,
        RegenerativeLineageRoleV1::SuccessorQualification,
        &supported.requirements,
    )?;

    let successor_reproduction_horizon = min_horizon(
        successor_construction.conservative_horizon,
        successor_qualification.conservative_horizon,
    );
    let regenerative_viability_horizon =
        min_horizon(operation.conservative_horizon, successor_reproduction_horizon);

    let limiting_roles = match regenerative_viability_horizon {
        RegenerativeHorizon::FinitePeriods(limit) => [
            &operation,
            &successor_construction,
            &successor_qualification,
        ]
        .into_iter()
        .filter(|report| report.conservative_horizon == RegenerativeHorizon::FinitePeriods(limit))
        .map(|report| report.role)
        .collect(),
        RegenerativeHorizon::IndefiniteUnderStaticModel => Vec::new(),
    };

    Ok(RegenerativeLineageViabilityReportV1 {
        profile_id: profile.profile_id.clone(),
        genome_id: supported.genome_id,
        closure_model_id: supported.closure_model_id,
        flow_support_id: supported.flow_support_id,
        fully_modeled_regenerative_viability: operation.fully_modeled_support
            && successor_construction.fully_modeled_support
            && successor_qualification.fully_modeled_support,
        operation,
        successor_construction,
        successor_qualification,
        successor_reproduction_horizon,
        regenerative_viability_horizon,
        limiting_roles,
    })
}

/// Validate that a successor Genome is bound to, and materially evolves from, its exact parent.
///
/// This deliberately rejects a rename-only successor. Requirement-manifest equality is checked
/// independently of `genome_id`, evidence bindings, and parent metadata.
pub fn validate_regenerative_lineage_successor(
    parent: &RegenerativeGenomeV1,
    successor: &RegenerativeGenomeV1,
) -> Result<(), RegenerativeLineageViabilityError> {
    if successor.lineage_parent_binding.as_deref() != Some(parent.evidence_binding.as_str()) {
        return Err(RegenerativeLineageViabilityError::ParentBindingMismatch);
    }
    if successor.genome_id == parent.genome_id || successor.evidence_binding == parent.evidence_binding {
        return Err(RegenerativeLineageViabilityError::SuccessorIdentityNotDistinct);
    }
    if successor.requirements == parent.requirements {
        return Err(RegenerativeLineageViabilityError::SuccessorManifestUnchanged);
    }
    Ok(())
}

fn evaluate_role(
    profile: &RegenerativeLineageViabilityProfileV1,
    role: RegenerativeLineageRoleV1,
    requirements: &[RegenerativeGenomeSupportedRequirementReport],
) -> Result<RegenerativeLineageRoleReportV1, RegenerativeLineageViabilityError> {
    let capability_ids = profile.capability_ids(role);
    let wanted: BTreeSet<&str> = capability_ids.iter().map(String::as_str).collect();

    for capability_id in capability_ids {
        if !requirements
            .iter()
            .any(|requirement| requirement.capability_id == *capability_id)
        {
            return Err(RegenerativeLineageViabilityError::MissingGenomeCoverage {
                role,
                capability_id: capability_id.clone(),
            });
        }
    }

    let selected: Vec<&RegenerativeGenomeSupportedRequirementReport> = requirements
        .iter()
        .filter(|requirement| wanted.contains(requirement.capability_id.as_str()))
        .collect();

    let finite_limit = selected
        .iter()
        .filter_map(|requirement| match requirement.conservative_horizon {
            RegenerativeHorizon::FinitePeriods(periods) => Some(periods),
            RegenerativeHorizon::IndefiniteUnderStaticModel => None,
        })
        .min();
    let conservative_horizon = finite_limit.map_or(
        RegenerativeHorizon::IndefiniteUnderStaticModel,
        RegenerativeHorizon::FinitePeriods,
    );

    let mut limiting_requirement_ids = Vec::new();
    let mut root_limiting_dependency_ids = BTreeSet::new();
    if let Some(limit) = finite_limit {
        for requirement in &selected {
            if requirement.conservative_horizon == RegenerativeHorizon::FinitePeriods(limit) {
                limiting_requirement_ids.push(requirement.requirement_id.clone());
                root_limiting_dependency_ids
                    .extend(requirement.limiting_dependency_ids.iter().cloned());
            }
        }
    }

    let externally_conditioned_requirement_ids: Vec<String> = selected
        .iter()
        .filter(|requirement| requirement.externally_conditioned)
        .map(|requirement| requirement.requirement_id.clone())
        .collect();

    Ok(RegenerativeLineageRoleReportV1 {
        role,
        conservative_horizon,
        limiting_requirement_ids,
        root_limiting_dependency_ids: root_limiting_dependency_ids.into_iter().collect(),
        fully_modeled_support: externally_conditioned_requirement_ids.is_empty(),
        externally_conditioned_requirement_ids,
    })
}

fn min_horizon(left: RegenerativeHorizon, right: RegenerativeHorizon) -> RegenerativeHorizon {
    match (left, right) {
        (RegenerativeHorizon::FinitePeriods(a), RegenerativeHorizon::FinitePeriods(b)) => {
            RegenerativeHorizon::FinitePeriods(a.min(b))
        }
        (RegenerativeHorizon::FinitePeriods(periods), _)
        | (_, RegenerativeHorizon::FinitePeriods(periods)) => {
            RegenerativeHorizon::FinitePeriods(periods)
        }
        (
            RegenerativeHorizon::IndefiniteUnderStaticModel,
            RegenerativeHorizon::IndefiniteUnderStaticModel,
        ) => RegenerativeHorizon::IndefiniteUnderStaticModel,
    }
}

fn validate_id(value: &str) -> Result<(), RegenerativeLineageViabilityError> {
    if value.is_empty()
        || value.len() > MAX_ID_LEN
        || value.trim() != value
        || value.chars().any(char::is_control)
    {
        Err(RegenerativeLineageViabilityError::InvalidIdentifier)
    } else {
        Ok(())
    }
}

fn validate_binding(value: &str) -> Result<(), RegenerativeLineageViabilityError> {
    if value.is_empty()
        || value.len() > MAX_BINDING_LEN
        || value.trim() != value
        || value.chars().any(char::is_control)
    {
        Err(RegenerativeLineageViabilityError::InvalidBinding)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DependencyGovernance, RegenerativeCapability, RegenerativeDependency,
        RegenerativeDependencyKind, RegenerativeFlowKindV1, RegenerativeFlowSupportClaimV1,
        RegenerativeGenomeRequirementV1, REGENERATIVE_FLOW_SUPPORT_SCHEMA_V1,
        REGENERATIVE_GENOME_SCHEMA_V1,
    };

    fn dependency(id: &str, stockpile: u64) -> RegenerativeDependency {
        RegenerativeDependency {
            dependency_id: id.into(),
            kind: RegenerativeDependencyKind::Component,
            governance: DependencyGovernance::Ordinary,
            demand_units_per_period: 1,
            local_production_units_per_period: 0,
            recycling_units_per_period: 0,
            stockpile_units: stockpile,
            unit_mass_grams: Some(1_000),
            evidence_binding: format!("dep:{id}"),
        }
    }

    fn model() -> RegenerativeClosureModel {
        RegenerativeClosureModel {
            model_id: "manta-lineage-v1".into(),
            period_duration_ms: 1,
            dependencies: vec![
                dependency("forge-tooling", 40),
                dependency("metrology", 30),
                dependency("platform-spares", 100),
            ],
            capabilities: vec![
                RegenerativeCapability {
                    capability_id: "platform-operation".into(),
                    essential: true,
                    dependency_ids: BTreeSet::from(["platform-spares".into()]),
                    evidence_binding: "cap:platform-operation".into(),
                },
                RegenerativeCapability {
                    capability_id: "successor-construction".into(),
                    essential: true,
                    dependency_ids: BTreeSet::from(["forge-tooling".into()]),
                    evidence_binding: "cap:successor-construction".into(),
                },
                RegenerativeCapability {
                    capability_id: "successor-qualification".into(),
                    essential: true,
                    dependency_ids: BTreeSet::from(["metrology".into()]),
                    evidence_binding: "cap:successor-qualification".into(),
                },
            ],
            evidence_binding: "model:manta-lineage-v1".into(),
        }
    }

    fn requirement(id: &str, capability: &str, dependency: &str) -> RegenerativeGenomeRequirementV1 {
        RegenerativeGenomeRequirementV1 {
            requirement_id: id.into(),
            capability_id: capability.into(),
            baseline_dependency_id: dependency.into(),
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
            genome_id: "manta-v1".into(),
            lineage_parent_binding: None,
            closure_model_id: "manta-lineage-v1".into(),
            closure_model_evidence_binding: "model:manta-lineage-v1".into(),
            requirements: vec![
                requirement("req-construction", "successor-construction", "forge-tooling"),
                requirement("req-operation", "platform-operation", "platform-spares"),
                requirement("req-qualification", "successor-qualification", "metrology"),
            ],
            evidence_binding: "genome:manta-v1".into(),
        }
    }

    fn support() -> RegenerativeFlowSupportV1 {
        RegenerativeFlowSupportV1 {
            schema_version: REGENERATIVE_FLOW_SUPPORT_SCHEMA_V1,
            support_id: "support:manta-lineage-v1".into(),
            closure_model_id: "manta-lineage-v1".into(),
            closure_model_evidence_binding: "model:manta-lineage-v1".into(),
            claims: Vec::new(),
            evidence_binding: "support:manta-lineage-v1".into(),
        }
    }

    fn profile() -> RegenerativeLineageViabilityProfileV1 {
        RegenerativeLineageViabilityProfileV1 {
            schema_version: REGENERATIVE_LINEAGE_VIABILITY_SCHEMA_V1,
            profile_id: "profile:manta-lineage-v1".into(),
            closure_model_id: "manta-lineage-v1".into(),
            closure_model_evidence_binding: "model:manta-lineage-v1".into(),
            operational_capability_ids: vec!["platform-operation".into()],
            successor_construction_capability_ids: vec!["successor-construction".into()],
            successor_qualification_capability_ids: vec!["successor-qualification".into()],
            evidence_binding: "profile-evidence:manta-lineage-v1".into(),
        }
    }

    #[test]
    fn qualification_can_end_lineage_reproduction_before_platform_operation() {
        let report = evaluate_regenerative_lineage_viability(
            &profile(),
            &genome(),
            &model(),
            &support(),
        )
        .unwrap();

        assert_eq!(
            report.operation.conservative_horizon,
            RegenerativeHorizon::FinitePeriods(100)
        );
        assert_eq!(
            report.successor_construction.conservative_horizon,
            RegenerativeHorizon::FinitePeriods(40)
        );
        assert_eq!(
            report.successor_qualification.conservative_horizon,
            RegenerativeHorizon::FinitePeriods(30)
        );
        assert_eq!(
            report.successor_reproduction_horizon,
            RegenerativeHorizon::FinitePeriods(30)
        );
        assert_eq!(
            report.regenerative_viability_horizon,
            RegenerativeHorizon::FinitePeriods(30)
        );
        assert_eq!(
            report.limiting_roles,
            vec![RegenerativeLineageRoleV1::SuccessorQualification]
        );
        assert_eq!(
            report.successor_qualification.root_limiting_dependency_ids,
            vec!["metrology"]
        );
        assert!(report.fully_modeled_regenerative_viability);
    }

    #[test]
    fn opaque_qualification_input_remains_uncertainty_not_a_fictional_lifetime() {
        let mut model = model();
        let metrology = model
            .dependencies
            .iter_mut()
            .find(|dependency| dependency.dependency_id == "metrology")
            .unwrap();
        metrology.local_production_units_per_period = 1;
        metrology.stockpile_units = 0;

        let mut support = support();
        support.claims.push(RegenerativeFlowSupportClaimV1 {
            dependency_id: "metrology".into(),
            flow_kind: RegenerativeFlowKindV1::Production,
            capability_binding: "capability:metrology-production".into(),
            prerequisite_dependency_ids: Vec::new(),
            external_input_binding: Some("external-input:reference-standard".into()),
            metrology_binding: "metrology:metrology-production".into(),
            qualification_binding: "qualification:metrology-production".into(),
            bootstrap_binding: None,
        });

        let report = evaluate_regenerative_lineage_viability(
            &profile(),
            &genome(),
            &model,
            &support,
        )
        .unwrap();
        assert_eq!(
            report.successor_qualification.conservative_horizon,
            RegenerativeHorizon::IndefiniteUnderStaticModel
        );
        assert!(!report.successor_qualification.fully_modeled_support);
        assert_eq!(
            report.successor_qualification.externally_conditioned_requirement_ids,
            vec!["req-qualification"]
        );
        assert_eq!(
            report.regenerative_viability_horizon,
            RegenerativeHorizon::FinitePeriods(40)
        );
        assert!(!report.fully_modeled_regenerative_viability);
    }

    #[test]
    fn successor_must_bind_exact_parent_and_change_requirement_manifest() {
        let parent = genome();
        let mut renamed = parent.clone();
        renamed.genome_id = "manta-v2".into();
        renamed.evidence_binding = "genome:manta-v2".into();
        renamed.lineage_parent_binding = Some(parent.evidence_binding.clone());
        assert_eq!(
            validate_regenerative_lineage_successor(&parent, &renamed),
            Err(RegenerativeLineageViabilityError::SuccessorManifestUnchanged)
        );

        let mut evolved = renamed;
        evolved.requirements[0].design_binding = "design:req-construction:v2".into();
        assert_eq!(
            validate_regenerative_lineage_successor(&parent, &evolved),
            Ok(())
        );

        let mut wrong_parent = evolved;
        wrong_parent.lineage_parent_binding = Some("genome:other".into());
        assert_eq!(
            validate_regenerative_lineage_successor(&parent, &wrong_parent),
            Err(RegenerativeLineageViabilityError::ParentBindingMismatch)
        );
    }
}
