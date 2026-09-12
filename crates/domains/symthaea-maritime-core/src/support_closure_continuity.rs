// SPDX-License-Identifier: AGPL-3.0-or-later
//! Intergenerational continuity of the support closure serving successor reproduction.
//!
//! A scalar regenerative runway is only portable when the role-relevant support
//! closure itself remains semantically equivalent across the handoff. This module
//! proves a strict V1 isomorphism over successor construction/qualification support
//! and additionally requires every finite root in that closure to be an actual,
//! basis-qualified transfer. It is evidence qualification only, not manufacturing
//! authority or a runtime inventory converter.

use crate::{
    evaluate_basis_qualified_regenerative_policy_sensitivity_surface,
    evaluate_supported_closure, RegenerativeBasisQualifiedPolicySensitivityError,
    RegenerativeClosureModel, RegenerativeEpochHandoffEvidenceV1, RegenerativeFlowKindV1,
    RegenerativeFlowSupportV1, RegenerativeHorizon,
    RegenerativeIntergenerationalSupportBasisReportV1, RegenerativeLineageViabilityProfileV1,
    RegenerativeLineageViabilityReportV1, RegenerativePolicySensitivitySurfaceV1,
    RegenerativeReproductionPolicySpecV1,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const MAX_ITEMS: usize = 4096;
const MAX_ID_LEN: usize = 256;
const MAX_BINDING_LEN: usize = 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeSupportClosureMappingV1 {
    pub source_dependency_id: String,
    pub successor_dependency_id: String,
    /// Evidence that the two dependency identities represent the same role in the
    /// intergenerational support graph. Finite inventory quantity equivalence is
    /// separately owned by the support-basis theorem.
    pub topology_equivalence_binding: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeRoleSupportClosureContinuityPolicyV1 {
    pub policy_id: String,
    pub evidence_binding: String,
    /// Strictly sorted by `(source_dependency_id, successor_dependency_id)`.
    /// The mapping must cover exactly the support closure reachable from successor
    /// construction and successor qualification on both sides.
    pub mappings: Vec<RegenerativeSupportClosureMappingV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeRoleSupportClosureContinuityReportV1 {
    pub policy_id: String,
    pub policy_evidence_binding: String,
    pub source_model_id: String,
    pub successor_model_id: String,
    pub source_support_id: String,
    pub successor_support_id: String,
    pub source_role_support_dependency_ids: Vec<String>,
    pub successor_role_support_dependency_ids: Vec<String>,
    pub source_finite_root_dependency_ids: Vec<String>,
    pub successor_finite_root_dependency_ids: Vec<String>,
    pub mapped_dependency_count: u16,
    pub finite_root_transfer_count: u16,
    /// True only because successful construction of this report has already proven
    /// exact closure coverage, topology/rate equivalence, root-set equivalence,
    /// transfer continuity and support-basis continuity.
    pub scalar_runway_projection_safe: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegenerativeSupportClosureContinuityError {
    InvalidIdentifier,
    InvalidEvidenceBinding,
    NoMappings,
    TooManyMappings,
    NonCanonicalMappingOrder,
    DuplicateSuccessorMapping { dependency_id: String },
    SourceProfileInvalid,
    SuccessorProfileInvalid,
    SourceSupportInvalid,
    SuccessorSupportInvalid,
    PeriodDurationChanged,
    SourceClosureMappingCoverageMismatch,
    SuccessorClosureMappingCoverageMismatch,
    UnknownSourceDependency { dependency_id: String },
    UnknownSuccessorDependency { dependency_id: String },
    DependencySemanticsChanged {
        source_dependency_id: String,
        successor_dependency_id: String,
    },
    MissingSuccessorFlowClaim {
        source_dependency_id: String,
        successor_dependency_id: String,
        flow_kind: RegenerativeFlowKindV1,
    },
    FlowTopologyChanged {
        source_dependency_id: String,
        successor_dependency_id: String,
        flow_kind: RegenerativeFlowKindV1,
    },
    SupportedClosureEvaluationFailed,
    FiniteRootSetChanged,
    HandoffBasisSubjectMismatch,
    FiniteRootNotTransferred {
        source_dependency_id: String,
        successor_dependency_id: String,
    },
    FiniteRootBasisNotQualified {
        source_dependency_id: String,
        successor_dependency_id: String,
    },
    CountOverflow,
}

#[derive(Debug)]
pub enum RegenerativeClosureQualifiedPolicySensitivityError {
    ClosureContinuity(RegenerativeSupportClosureContinuityError),
    SourceModelMismatch,
    Basis(RegenerativeBasisQualifiedPolicySensitivityError),
}

pub fn qualify_regenerative_role_support_closure_continuity(
    policy: &RegenerativeRoleSupportClosureContinuityPolicyV1,
    source_profile: &RegenerativeLineageViabilityProfileV1,
    source_model: &RegenerativeClosureModel,
    source_support: &RegenerativeFlowSupportV1,
    successor_profile: &RegenerativeLineageViabilityProfileV1,
    successor_model: &RegenerativeClosureModel,
    successor_support: &RegenerativeFlowSupportV1,
    handoff_evidence: &RegenerativeEpochHandoffEvidenceV1,
    basis_report: &RegenerativeIntergenerationalSupportBasisReportV1,
) -> Result<RegenerativeRoleSupportClosureContinuityReportV1, RegenerativeSupportClosureContinuityError>
{
    validate_policy(policy)?;
    source_profile
        .validate_against_model(source_model)
        .map_err(|_| RegenerativeSupportClosureContinuityError::SourceProfileInvalid)?;
    successor_profile
        .validate_against_model(successor_model)
        .map_err(|_| RegenerativeSupportClosureContinuityError::SuccessorProfileInvalid)?;
    source_support
        .validate_against_model(source_model)
        .map_err(|_| RegenerativeSupportClosureContinuityError::SourceSupportInvalid)?;
    successor_support
        .validate_against_model(successor_model)
        .map_err(|_| RegenerativeSupportClosureContinuityError::SuccessorSupportInvalid)?;

    if source_model.period_duration_ms != successor_model.period_duration_ms {
        return Err(RegenerativeSupportClosureContinuityError::PeriodDurationChanged);
    }

    let source_closure = role_support_closure(source_profile, source_model, source_support)?;
    let successor_closure = role_support_closure(successor_profile, successor_model, successor_support)?;

    let mut mapping_by_source = BTreeMap::new();
    let mut mapped_successor_ids = BTreeSet::new();
    for mapping in &policy.mappings {
        mapping_by_source.insert(
            mapping.source_dependency_id.as_str(),
            mapping.successor_dependency_id.as_str(),
        );
        if !mapped_successor_ids.insert(mapping.successor_dependency_id.as_str()) {
            return Err(RegenerativeSupportClosureContinuityError::DuplicateSuccessorMapping {
                dependency_id: mapping.successor_dependency_id.clone(),
            });
        }
    }
    let mapped_source_ids: BTreeSet<String> = policy
        .mappings
        .iter()
        .map(|mapping| mapping.source_dependency_id.clone())
        .collect();
    let mapped_successor_ids_owned: BTreeSet<String> = policy
        .mappings
        .iter()
        .map(|mapping| mapping.successor_dependency_id.clone())
        .collect();
    if mapped_source_ids != source_closure {
        return Err(
            RegenerativeSupportClosureContinuityError::SourceClosureMappingCoverageMismatch,
        );
    }
    if mapped_successor_ids_owned != successor_closure {
        return Err(
            RegenerativeSupportClosureContinuityError::SuccessorClosureMappingCoverageMismatch,
        );
    }

    let source_dependencies: BTreeMap<&str, _> = source_model
        .dependencies
        .iter()
        .map(|dependency| (dependency.dependency_id.as_str(), dependency))
        .collect();
    let successor_dependencies: BTreeMap<&str, _> = successor_model
        .dependencies
        .iter()
        .map(|dependency| (dependency.dependency_id.as_str(), dependency))
        .collect();

    for mapping in &policy.mappings {
        let source = source_dependencies
            .get(mapping.source_dependency_id.as_str())
            .ok_or_else(|| RegenerativeSupportClosureContinuityError::UnknownSourceDependency {
                dependency_id: mapping.source_dependency_id.clone(),
            })?;
        let successor = successor_dependencies
            .get(mapping.successor_dependency_id.as_str())
            .ok_or_else(|| RegenerativeSupportClosureContinuityError::UnknownSuccessorDependency {
                dependency_id: mapping.successor_dependency_id.clone(),
            })?;
        if source.kind != successor.kind
            || source.governance != successor.governance
            || source.demand_units_per_period != successor.demand_units_per_period
            || source.local_production_units_per_period
                != successor.local_production_units_per_period
            || source.recycling_units_per_period != successor.recycling_units_per_period
        {
            return Err(
                RegenerativeSupportClosureContinuityError::DependencySemanticsChanged {
                    source_dependency_id: mapping.source_dependency_id.clone(),
                    successor_dependency_id: mapping.successor_dependency_id.clone(),
                },
            );
        }

        let source_claims: Vec<_> = source_support
            .claims
            .iter()
            .filter(|claim| claim.dependency_id == mapping.source_dependency_id)
            .collect();
        let successor_claims: Vec<_> = successor_support
            .claims
            .iter()
            .filter(|claim| claim.dependency_id == mapping.successor_dependency_id)
            .collect();
        if source_claims.len() != successor_claims.len() {
            return Err(RegenerativeSupportClosureContinuityError::FlowTopologyChanged {
                source_dependency_id: mapping.source_dependency_id.clone(),
                successor_dependency_id: mapping.successor_dependency_id.clone(),
                flow_kind: source_claims
                    .first()
                    .map(|claim| claim.flow_kind)
                    .unwrap_or(RegenerativeFlowKindV1::Production),
            });
        }
        for source_claim in source_claims {
            let successor_claim = successor_claims
                .iter()
                .copied()
                .find(|claim| claim.flow_kind == source_claim.flow_kind)
                .ok_or_else(|| {
                    RegenerativeSupportClosureContinuityError::MissingSuccessorFlowClaim {
                        source_dependency_id: mapping.source_dependency_id.clone(),
                        successor_dependency_id: mapping.successor_dependency_id.clone(),
                        flow_kind: source_claim.flow_kind,
                    }
                })?;
            let mapped_prerequisites: Vec<String> = source_claim
                .prerequisite_dependency_ids
                .iter()
                .map(|source_id| {
                    mapping_by_source
                        .get(source_id.as_str())
                        .copied()
                        .map(str::to_string)
                        .ok_or_else(|| {
                            RegenerativeSupportClosureContinuityError::SourceClosureMappingCoverageMismatch
                        })
                })
                .collect::<Result<_, _>>()?;
            if mapped_prerequisites != successor_claim.prerequisite_dependency_ids
                || source_claim.external_input_binding.is_some()
                    != successor_claim.external_input_binding.is_some()
                || source_claim.bootstrap_binding.is_some()
                    != successor_claim.bootstrap_binding.is_some()
            {
                return Err(RegenerativeSupportClosureContinuityError::FlowTopologyChanged {
                    source_dependency_id: mapping.source_dependency_id.clone(),
                    successor_dependency_id: mapping.successor_dependency_id.clone(),
                    flow_kind: source_claim.flow_kind,
                });
            }
        }
    }

    let source_supported = evaluate_supported_closure(source_model, source_support)
        .map_err(|_| RegenerativeSupportClosureContinuityError::SupportedClosureEvaluationFailed)?;
    let successor_supported = evaluate_supported_closure(successor_model, successor_support)
        .map_err(|_| RegenerativeSupportClosureContinuityError::SupportedClosureEvaluationFailed)?;
    let source_roots = finite_roots(&source_closure, &source_supported.dependencies);
    let successor_roots = finite_roots(&successor_closure, &successor_supported.dependencies);
    let mapped_source_roots: BTreeSet<String> = source_roots
        .iter()
        .map(|source_id| {
            mapping_by_source
                .get(source_id.as_str())
                .copied()
                .map(str::to_string)
                .ok_or(RegenerativeSupportClosureContinuityError::SourceClosureMappingCoverageMismatch)
        })
        .collect::<Result<_, _>>()?;
    if mapped_source_roots != successor_roots {
        return Err(RegenerativeSupportClosureContinuityError::FiniteRootSetChanged);
    }

    if basis_report.handoff_id != handoff_evidence.handoff_id
        || basis_report.dynamic_handoff_receipt_binding
            != handoff_evidence.dynamic_handoff_receipt_binding
        || basis_report.source_model_id != source_model.model_id
        || basis_report.successor_model_id != successor_model.model_id
    {
        return Err(RegenerativeSupportClosureContinuityError::HandoffBasisSubjectMismatch);
    }

    for source_root in &source_roots {
        let successor_root = mapping_by_source
            .get(source_root.as_str())
            .copied()
            .ok_or(RegenerativeSupportClosureContinuityError::SourceClosureMappingCoverageMismatch)?;
        let transferred = handoff_evidence.transfer_qualifications.iter().any(|transfer| {
            transfer.source_dependency_id == *source_root
                && transfer.successor_dependency_id == successor_root
        });
        if !transferred {
            return Err(RegenerativeSupportClosureContinuityError::FiniteRootNotTransferred {
                source_dependency_id: source_root.clone(),
                successor_dependency_id: successor_root.to_string(),
            });
        }
        let basis_qualified = basis_report.assessments.iter().any(|assessment| {
            assessment.source_dependency_id == *source_root
                && assessment.successor_dependency_id == successor_root
                && assessment.scalar_period_basis_preserved
        });
        if !basis_qualified {
            return Err(
                RegenerativeSupportClosureContinuityError::FiniteRootBasisNotQualified {
                    source_dependency_id: source_root.clone(),
                    successor_dependency_id: successor_root.to_string(),
                },
            );
        }
    }

    Ok(RegenerativeRoleSupportClosureContinuityReportV1 {
        policy_id: policy.policy_id.clone(),
        policy_evidence_binding: policy.evidence_binding.clone(),
        source_model_id: source_model.model_id.clone(),
        successor_model_id: successor_model.model_id.clone(),
        source_support_id: source_support.support_id.clone(),
        successor_support_id: successor_support.support_id.clone(),
        source_role_support_dependency_ids: source_closure.into_iter().collect(),
        successor_role_support_dependency_ids: successor_closure.into_iter().collect(),
        source_finite_root_dependency_ids: source_roots.into_iter().collect(),
        successor_finite_root_dependency_ids: successor_roots.into_iter().collect(),
        mapped_dependency_count: count(policy.mappings.len())?,
        finite_root_transfer_count: count(mapped_source_roots.len())?,
        scalar_runway_projection_safe: true,
    })
}

pub fn evaluate_closure_qualified_regenerative_policy_sensitivity_surface(
    lineage_report: &RegenerativeLineageViabilityReportV1,
    policies: &[RegenerativeReproductionPolicySpecV1],
    basis_report: &RegenerativeIntergenerationalSupportBasisReportV1,
    continuity_report: &RegenerativeRoleSupportClosureContinuityReportV1,
) -> Result<RegenerativePolicySensitivitySurfaceV1, RegenerativeClosureQualifiedPolicySensitivityError>
{
    if !continuity_report.scalar_runway_projection_safe {
        return Err(RegenerativeClosureQualifiedPolicySensitivityError::ClosureContinuity(
            RegenerativeSupportClosureContinuityError::FiniteRootSetChanged,
        ));
    }
    if lineage_report.closure_model_id != continuity_report.source_model_id {
        return Err(RegenerativeClosureQualifiedPolicySensitivityError::SourceModelMismatch);
    }
    evaluate_basis_qualified_regenerative_policy_sensitivity_surface(
        lineage_report,
        policies,
        basis_report,
    )
    .map_err(RegenerativeClosureQualifiedPolicySensitivityError::Basis)
}

fn role_support_closure(
    profile: &RegenerativeLineageViabilityProfileV1,
    model: &RegenerativeClosureModel,
    support: &RegenerativeFlowSupportV1,
) -> Result<BTreeSet<String>, RegenerativeSupportClosureContinuityError> {
    let selected_capabilities: BTreeSet<&str> = profile
        .successor_construction_capability_ids
        .iter()
        .chain(profile.successor_qualification_capability_ids.iter())
        .map(String::as_str)
        .collect();
    let mut closure = BTreeSet::new();
    for capability in &model.capabilities {
        if selected_capabilities.contains(capability.capability_id.as_str()) {
            closure.extend(capability.dependency_ids.iter().cloned());
        }
    }
    for _ in 0..model.dependencies.len().max(1) {
        let before = closure.len();
        for claim in &support.claims {
            if closure.contains(&claim.dependency_id) {
                closure.extend(claim.prerequisite_dependency_ids.iter().cloned());
            }
        }
        if closure.len() == before {
            break;
        }
    }
    Ok(closure)
}

fn finite_roots(
    closure: &BTreeSet<String>,
    reports: &[crate::RegenerativeSupportedDependencyReport],
) -> BTreeSet<String> {
    let mut roots = BTreeSet::new();
    for report in reports {
        if closure.contains(&report.dependency_id)
            && matches!(report.conservative_horizon, RegenerativeHorizon::FinitePeriods(_))
        {
            roots.extend(report.limiting_dependency_ids.iter().cloned());
        }
    }
    roots
}

fn validate_policy(
    policy: &RegenerativeRoleSupportClosureContinuityPolicyV1,
) -> Result<(), RegenerativeSupportClosureContinuityError> {
    validate_id(&policy.policy_id)?;
    validate_binding(&policy.evidence_binding)?;
    if policy.mappings.is_empty() {
        return Err(RegenerativeSupportClosureContinuityError::NoMappings);
    }
    if policy.mappings.len() > MAX_ITEMS {
        return Err(RegenerativeSupportClosureContinuityError::TooManyMappings);
    }
    for mapping in &policy.mappings {
        validate_id(&mapping.source_dependency_id)?;
        validate_id(&mapping.successor_dependency_id)?;
        validate_binding(&mapping.topology_equivalence_binding)?;
    }
    if policy.mappings.windows(2).any(|pair| {
        (
            pair[0].source_dependency_id.as_str(),
            pair[0].successor_dependency_id.as_str(),
        ) >= (
            pair[1].source_dependency_id.as_str(),
            pair[1].successor_dependency_id.as_str(),
        )
    }) {
        return Err(RegenerativeSupportClosureContinuityError::NonCanonicalMappingOrder);
    }
    Ok(())
}

fn count(value: usize) -> Result<u16, RegenerativeSupportClosureContinuityError> {
    u16::try_from(value).map_err(|_| RegenerativeSupportClosureContinuityError::CountOverflow)
}

fn validate_id(value: &str) -> Result<(), RegenerativeSupportClosureContinuityError> {
    if value.is_empty()
        || value.len() > MAX_ID_LEN
        || value.trim() != value
        || value.chars().any(char::is_whitespace)
        || value.chars().any(char::is_control)
    {
        Err(RegenerativeSupportClosureContinuityError::InvalidIdentifier)
    } else {
        Ok(())
    }
}

fn validate_binding(value: &str) -> Result<(), RegenerativeSupportClosureContinuityError> {
    if value.is_empty()
        || value.len() > MAX_BINDING_LEN
        || value.trim() != value
        || !value.contains(':')
        || value.chars().any(char::is_whitespace)
        || value.chars().any(char::is_control)
    {
        Err(RegenerativeSupportClosureContinuityError::InvalidEvidenceBinding)
    } else {
        Ok(())
    }
}
