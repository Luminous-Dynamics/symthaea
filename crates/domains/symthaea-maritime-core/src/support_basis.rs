// SPDX-License-Identifier: AGPL-3.0-or-later
//! Intergenerational quantity/time-basis qualification for regenerative support.
//!
//! A temporal runway scalar is only portable across an epoch handoff when the
//! transferred support quantity has an explicitly equivalent unit basis and its
//! stockpile draw rate is preserved on the common time basis. This module composes
//! with epoch-handoff qualification; it does not replace conservation accounting,
//! transfer admissibility, manufacturing qualification, or operating authority.

use crate::{
    evaluate_regenerative_policy_sensitivity_surface, RegenerativeClosureError,
    RegenerativeClosureModel, RegenerativeEpochHandoffEvidenceV1,
    RegenerativeEpochHandoffQualificationReportV1, RegenerativeLineageViabilityReportV1,
    RegenerativePolicySensitivityError, RegenerativePolicySensitivitySurfaceV1,
    RegenerativeReproductionPolicySpecV1,
};
use serde::{Deserialize, Serialize};

const MAX_ITEMS: usize = 4096;
const MAX_ID_LEN: usize = 256;
const MAX_BINDING_LEN: usize = 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeSupportBasisRequirementV1 {
    pub source_dependency_id: String,
    pub successor_dependency_id: String,
    /// Evidence that one source inventory unit and one successor inventory unit use
    /// the same quantity basis. This is intentionally separate from generic transfer
    /// admissibility: "may be transferred" is not the same claim as "one unit means
    /// the same quantity on both sides".
    pub quantity_basis_equivalence_binding: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeIntergenerationalSupportBasisPolicyV1 {
    pub policy_id: String,
    pub evidence_binding: String,
    /// Bootstrap-critical transfer pairs, strictly sorted by source then successor ID.
    pub requirements: Vec<RegenerativeSupportBasisRequirementV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeSupportBasisAssessmentV1 {
    pub source_dependency_id: String,
    pub successor_dependency_id: String,
    pub quantity_basis_equivalence_binding: String,
    /// Net stockpile draw after modeled local production/recycling.
    pub source_stockpile_draw_units_per_period: u64,
    pub successor_stockpile_draw_units_per_period: u64,
    pub source_period_duration_ms: u64,
    pub successor_period_duration_ms: u64,
    /// Exact rational-rate equality checked without floating point:
    /// source_draw/source_period == successor_draw/successor_period.
    pub normalized_stockpile_draw_rate_preserved: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeIntergenerationalSupportBasisReportV1 {
    pub policy_id: String,
    pub policy_evidence_binding: String,
    pub handoff_id: String,
    pub dynamic_handoff_receipt_binding: String,
    pub source_model_id: String,
    pub source_model_evidence_binding: String,
    pub successor_model_id: String,
    pub successor_model_evidence_binding: String,
    pub assessments: Vec<RegenerativeSupportBasisAssessmentV1>,
    /// True only when every bootstrap-critical mapping preserves both explicit unit
    /// equivalence evidence and normalized stockpile draw rate.
    pub scalar_runway_projection_safe: bool,
}

#[derive(Debug)]
pub enum RegenerativeSupportBasisError {
    InvalidIdentifier,
    InvalidEvidenceBinding,
    NoRequirements,
    TooManyRequirements,
    NonCanonicalRequirementOrder,
    SourceModelInvalid(RegenerativeClosureError),
    SuccessorModelInvalid(RegenerativeClosureError),
    HandoffQualificationMismatch,
    QualifiedTransferCountMismatch,
    MissingQualifiedTransfer {
        source_dependency_id: String,
        successor_dependency_id: String,
    },
    AmbiguousQualifiedTransfer {
        source_dependency_id: String,
        successor_dependency_id: String,
    },
    UnknownSourceDependency {
        dependency_id: String,
    },
    UnknownSuccessorDependency {
        dependency_id: String,
    },
    QualifiedTransferSemanticDrift {
        source_dependency_id: String,
        successor_dependency_id: String,
    },
    ScalarRunwayProjectionUnsafe,
}

#[derive(Debug)]
pub enum RegenerativeBasisQualifiedPolicySensitivityError {
    SupportBasis(RegenerativeSupportBasisError),
    BasisSourceModelMismatch,
    PolicySensitivity(RegenerativePolicySensitivityError),
}

pub fn assess_intergenerational_support_basis(
    policy: &RegenerativeIntergenerationalSupportBasisPolicyV1,
    handoff_evidence: &RegenerativeEpochHandoffEvidenceV1,
    handoff_report: &RegenerativeEpochHandoffQualificationReportV1,
    source_model: &RegenerativeClosureModel,
    successor_model: &RegenerativeClosureModel,
) -> Result<RegenerativeIntergenerationalSupportBasisReportV1, RegenerativeSupportBasisError> {
    validate_policy(policy)?;
    source_model
        .validate()
        .map_err(RegenerativeSupportBasisError::SourceModelInvalid)?;
    successor_model
        .validate()
        .map_err(RegenerativeSupportBasisError::SuccessorModelInvalid)?;

    if handoff_report.handoff_id != handoff_evidence.handoff_id
        || handoff_report.dynamic_handoff_receipt_binding
            != handoff_evidence.dynamic_handoff_receipt_binding
    {
        return Err(RegenerativeSupportBasisError::HandoffQualificationMismatch);
    }
    if usize::from(handoff_report.qualified_transfer_count)
        != handoff_evidence.transfer_qualifications.len()
    {
        return Err(RegenerativeSupportBasisError::QualifiedTransferCountMismatch);
    }

    let mut assessments = Vec::with_capacity(policy.requirements.len());
    for requirement in &policy.requirements {
        let matching = handoff_evidence
            .transfer_qualifications
            .iter()
            .filter(|transfer| {
                transfer.source_dependency_id == requirement.source_dependency_id
                    && transfer.successor_dependency_id == requirement.successor_dependency_id
            })
            .collect::<Vec<_>>();
        match matching.len() {
            0 => {
                return Err(RegenerativeSupportBasisError::MissingQualifiedTransfer {
                    source_dependency_id: requirement.source_dependency_id.clone(),
                    successor_dependency_id: requirement.successor_dependency_id.clone(),
                });
            }
            1 => {}
            _ => {
                return Err(RegenerativeSupportBasisError::AmbiguousQualifiedTransfer {
                    source_dependency_id: requirement.source_dependency_id.clone(),
                    successor_dependency_id: requirement.successor_dependency_id.clone(),
                });
            }
        }
        // The upstream transfer theorem must at minimum have a real qualification
        // binding for the exact pair used here.
        validate_binding(&matching[0].transfer_qualification_binding)?;

        let source = source_model
            .dependencies
            .iter()
            .find(|dependency| dependency.dependency_id == requirement.source_dependency_id)
            .ok_or_else(|| RegenerativeSupportBasisError::UnknownSourceDependency {
                dependency_id: requirement.source_dependency_id.clone(),
            })?;
        let successor = successor_model
            .dependencies
            .iter()
            .find(|dependency| dependency.dependency_id == requirement.successor_dependency_id)
            .ok_or_else(|| RegenerativeSupportBasisError::UnknownSuccessorDependency {
                dependency_id: requirement.successor_dependency_id.clone(),
            })?;

        // This layer composes with the epoch-handoff theorem, but still refuses to
        // evaluate a pair whose model semantics drifted away from that theorem.
        if source.kind != successor.kind || source.governance != successor.governance {
            return Err(RegenerativeSupportBasisError::QualifiedTransferSemanticDrift {
                source_dependency_id: requirement.source_dependency_id.clone(),
                successor_dependency_id: requirement.successor_dependency_id.clone(),
            });
        }

        let source_draw = stockpile_draw_units_per_period(source);
        let successor_draw = stockpile_draw_units_per_period(successor);
        let preserved = u128::from(source_draw) * u128::from(successor_model.period_duration_ms)
            == u128::from(successor_draw) * u128::from(source_model.period_duration_ms);

        assessments.push(RegenerativeSupportBasisAssessmentV1 {
            source_dependency_id: requirement.source_dependency_id.clone(),
            successor_dependency_id: requirement.successor_dependency_id.clone(),
            quantity_basis_equivalence_binding: requirement
                .quantity_basis_equivalence_binding
                .clone(),
            source_stockpile_draw_units_per_period: source_draw,
            successor_stockpile_draw_units_per_period: successor_draw,
            source_period_duration_ms: source_model.period_duration_ms,
            successor_period_duration_ms: successor_model.period_duration_ms,
            normalized_stockpile_draw_rate_preserved: preserved,
        });
    }

    let scalar_runway_projection_safe = assessments
        .iter()
        .all(|assessment| assessment.normalized_stockpile_draw_rate_preserved);

    Ok(RegenerativeIntergenerationalSupportBasisReportV1 {
        policy_id: policy.policy_id.clone(),
        policy_evidence_binding: policy.evidence_binding.clone(),
        handoff_id: handoff_evidence.handoff_id.clone(),
        dynamic_handoff_receipt_binding: handoff_evidence.dynamic_handoff_receipt_binding.clone(),
        source_model_id: source_model.model_id.clone(),
        source_model_evidence_binding: source_model.evidence_binding.clone(),
        successor_model_id: successor_model.model_id.clone(),
        successor_model_evidence_binding: successor_model.evidence_binding.clone(),
        assessments,
        scalar_runway_projection_safe,
    })
}

pub fn require_scalar_runway_projection_safe(
    report: &RegenerativeIntergenerationalSupportBasisReportV1,
) -> Result<(), RegenerativeSupportBasisError> {
    if report.scalar_runway_projection_safe
        && report
            .assessments
            .iter()
            .all(|assessment| assessment.normalized_stockpile_draw_rate_preserved)
    {
        Ok(())
    } else {
        Err(RegenerativeSupportBasisError::ScalarRunwayProjectionUnsafe)
    }
}

/// Policy-sensitivity projection guarded by an independently assessed
/// intergenerational support basis.
pub fn evaluate_basis_qualified_regenerative_policy_sensitivity_surface(
    lineage_report: &RegenerativeLineageViabilityReportV1,
    policies: &[RegenerativeReproductionPolicySpecV1],
    basis_report: &RegenerativeIntergenerationalSupportBasisReportV1,
) -> Result<RegenerativePolicySensitivitySurfaceV1, RegenerativeBasisQualifiedPolicySensitivityError>
{
    require_scalar_runway_projection_safe(basis_report)
        .map_err(RegenerativeBasisQualifiedPolicySensitivityError::SupportBasis)?;
    if lineage_report.closure_model_id != basis_report.source_model_id {
        return Err(RegenerativeBasisQualifiedPolicySensitivityError::BasisSourceModelMismatch);
    }
    evaluate_regenerative_policy_sensitivity_surface(lineage_report, policies)
        .map_err(RegenerativeBasisQualifiedPolicySensitivityError::PolicySensitivity)
}

fn stockpile_draw_units_per_period(dependency: &crate::RegenerativeDependency) -> u64 {
    let local = u128::from(dependency.local_production_units_per_period)
        + u128::from(dependency.recycling_units_per_period);
    let demand = u128::from(dependency.demand_units_per_period);
    if local >= demand {
        0
    } else {
        (demand - local) as u64
    }
}

fn validate_policy(
    policy: &RegenerativeIntergenerationalSupportBasisPolicyV1,
) -> Result<(), RegenerativeSupportBasisError> {
    validate_id(&policy.policy_id)?;
    validate_binding(&policy.evidence_binding)?;
    if policy.requirements.is_empty() {
        return Err(RegenerativeSupportBasisError::NoRequirements);
    }
    if policy.requirements.len() > MAX_ITEMS {
        return Err(RegenerativeSupportBasisError::TooManyRequirements);
    }
    for requirement in &policy.requirements {
        validate_id(&requirement.source_dependency_id)?;
        validate_id(&requirement.successor_dependency_id)?;
        validate_binding(&requirement.quantity_basis_equivalence_binding)?;
    }
    if policy.requirements.windows(2).any(|pair| {
        (
            pair[0].source_dependency_id.as_str(),
            pair[0].successor_dependency_id.as_str(),
        ) >= (
            pair[1].source_dependency_id.as_str(),
            pair[1].successor_dependency_id.as_str(),
        )
    }) {
        return Err(RegenerativeSupportBasisError::NonCanonicalRequirementOrder);
    }
    Ok(())
}

fn validate_id(value: &str) -> Result<(), RegenerativeSupportBasisError> {
    if value.is_empty()
        || value.len() > MAX_ID_LEN
        || value.trim() != value
        || value.chars().any(char::is_whitespace)
        || value.chars().any(char::is_control)
    {
        Err(RegenerativeSupportBasisError::InvalidIdentifier)
    } else {
        Ok(())
    }
}

fn validate_binding(value: &str) -> Result<(), RegenerativeSupportBasisError> {
    if value.is_empty()
        || value.len() > MAX_BINDING_LEN
        || value.trim() != value
        || !value.contains(':')
        || value.chars().any(char::is_whitespace)
        || value.chars().any(char::is_control)
    {
        Err(RegenerativeSupportBasisError::InvalidEvidenceBinding)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DependencyGovernance, RegenerativeCapability, RegenerativeDependency,
        RegenerativeDependencyKind, RegenerativeEpochTransferQualificationV1,
    };
    use std::collections::BTreeSet;

    fn dependency(id: &str, demand: u64) -> RegenerativeDependency {
        RegenerativeDependency {
            dependency_id: id.into(),
            kind: RegenerativeDependencyKind::Tooling,
            governance: DependencyGovernance::Ordinary,
            demand_units_per_period: demand,
            local_production_units_per_period: 0,
            recycling_units_per_period: 0,
            stockpile_units: 8,
            unit_mass_grams: None,
            evidence_binding: format!("dependency:{id}:evidence"),
        }
    }

    fn model(
        id: &str,
        period_ms: u64,
        dependency_id: &str,
        demand: u64,
    ) -> RegenerativeClosureModel {
        RegenerativeClosureModel {
            model_id: id.into(),
            period_duration_ms: period_ms,
            dependencies: vec![dependency(dependency_id, demand)],
            capabilities: vec![RegenerativeCapability {
                capability_id: format!("capability:{id}"),
                essential: true,
                dependency_ids: BTreeSet::from([dependency_id.into()]),
                evidence_binding: format!("capability:{id}:evidence"),
            }],
            evidence_binding: format!("model:{id}:evidence"),
        }
    }

    fn handoff() -> (
        RegenerativeEpochHandoffEvidenceV1,
        RegenerativeEpochHandoffQualificationReportV1,
    ) {
        let evidence = RegenerativeEpochHandoffEvidenceV1 {
            schema_version: 1,
            handoff_id: "handoff-v3-v4".into(),
            source_epoch_id: "epoch-v3".into(),
            source_epoch_evidence_binding: "epoch:v3:evidence".into(),
            successor_epoch_id: "epoch-v4".into(),
            successor_epoch_evidence_binding: "epoch:v4:evidence".into(),
            source_genome_id: "genome-v3".into(),
            source_genome_evidence_binding: "genome:v3:evidence".into(),
            successor_genome_id: "genome-v4".into(),
            successor_genome_evidence_binding: "genome:v4:evidence".into(),
            dynamic_handoff_receipt_binding: "symtropy:handoff:v3-v4".into(),
            transfer_qualifications: vec![RegenerativeEpochTransferQualificationV1 {
                source_dependency_id: "tooling-v3".into(),
                successor_dependency_id: "tooling-v4".into(),
                transfer_qualification_binding: "transfer:tooling:v3-v4".into(),
                safeguarded_continuity_binding: None,
            }],
            external_admission_qualifications: Vec::new(),
            evidence_binding: "symthaea:handoff:v3-v4".into(),
        };
        let report = RegenerativeEpochHandoffQualificationReportV1 {
            handoff_id: evidence.handoff_id.clone(),
            source_genome_id: evidence.source_genome_id.clone(),
            successor_genome_id: evidence.successor_genome_id.clone(),
            dynamic_handoff_receipt_binding: evidence.dynamic_handoff_receipt_binding.clone(),
            qualified_transfer_count: 1,
            cross_id_transfer_count: 1,
            safeguarded_transfer_count: 0,
            external_admission_count: 0,
            safeguarded_external_admission_count: 0,
        };
        (evidence, report)
    }

    fn policy() -> RegenerativeIntergenerationalSupportBasisPolicyV1 {
        RegenerativeIntergenerationalSupportBasisPolicyV1 {
            policy_id: "basis-policy-v1".into(),
            evidence_binding: "basis-policy:v1:evidence".into(),
            requirements: vec![RegenerativeSupportBasisRequirementV1 {
                source_dependency_id: "tooling-v3".into(),
                successor_dependency_id: "tooling-v4".into(),
                quantity_basis_equivalence_binding: "quantity-basis:tooling:v3-v4".into(),
            }],
        }
    }

    #[test]
    fn changed_units_per_time_basis_blocks_scalar_runway_projection() {
        let (evidence, handoff_report) = handoff();
        let source = model("closure-v3", 1, "tooling-v3", 2);
        let successor = model("closure-v4", 1, "tooling-v4", 1);
        let report = assess_intergenerational_support_basis(
            &policy(),
            &evidence,
            &handoff_report,
            &source,
            &successor,
        )
        .unwrap();
        assert!(!report.scalar_runway_projection_safe);
        assert!(!report.assessments[0].normalized_stockpile_draw_rate_preserved);
        assert!(matches!(
            require_scalar_runway_projection_safe(&report),
            Err(RegenerativeSupportBasisError::ScalarRunwayProjectionUnsafe)
        ));
    }

    #[test]
    fn preserved_units_per_time_basis_authorizes_scalar_projection() {
        let (evidence, handoff_report) = handoff();
        let source = model("closure-v3", 1, "tooling-v3", 2);
        let successor = model("closure-v4", 1, "tooling-v4", 2);
        let report = assess_intergenerational_support_basis(
            &policy(),
            &evidence,
            &handoff_report,
            &source,
            &successor,
        )
        .unwrap();
        assert!(report.scalar_runway_projection_safe);
        assert!(require_scalar_runway_projection_safe(&report).is_ok());
    }

    #[test]
    fn equivalent_rate_can_use_different_period_durations_without_float_math() {
        let (evidence, handoff_report) = handoff();
        let source = model("closure-v3", 2, "tooling-v3", 2);
        let successor = model("closure-v4", 1, "tooling-v4", 1);
        let report = assess_intergenerational_support_basis(
            &policy(),
            &evidence,
            &handoff_report,
            &source,
            &successor,
        )
        .unwrap();
        assert!(report.scalar_runway_projection_safe);
        assert_eq!(
            report.assessments[0].source_stockpile_draw_units_per_period,
            2
        );
        assert_eq!(
            report.assessments[0].successor_stockpile_draw_units_per_period,
            1
        );
    }

    #[test]
    fn missing_exact_transfer_mapping_fails_closed() {
        let (mut evidence, handoff_report) = handoff();
        evidence.transfer_qualifications[0].successor_dependency_id = "other-tooling".into();
        let source = model("closure-v3", 1, "tooling-v3", 1);
        let successor = model("closure-v4", 1, "tooling-v4", 1);
        assert!(matches!(
            assess_intergenerational_support_basis(
                &policy(),
                &evidence,
                &handoff_report,
                &source,
                &successor,
            ),
            Err(RegenerativeSupportBasisError::MissingQualifiedTransfer { .. })
        ));
    }
}
