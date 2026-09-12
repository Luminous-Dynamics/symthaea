// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact evidence envelope for regenerative lineage viability reports.
//!
//! The role-separated viability report is useful as a diagnostic result, but a
//! downstream evidence graph also needs to know exactly which profile, Genome,
//! closure model, and flow-support graph produced it. This module preserves those
//! evidence/version bindings without changing the viability calculation itself.

use crate::{
    RegenerativeClosureModel, RegenerativeFlowSupportV1, RegenerativeGenomeV1,
    RegenerativeLineageViabilityError, RegenerativeLineageViabilityProfileV1,
    RegenerativeLineageViabilityReportV1, evaluate_regenerative_lineage_viability,
};
use serde::{Deserialize, Serialize};

/// Exact evidence identities plus the derived role-separated viability report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeLineageEvidenceEnvelopeV1 {
    pub profile_id: String,
    pub profile_evidence_binding: String,
    pub genome_id: String,
    pub genome_evidence_binding: String,
    pub lineage_parent_binding: Option<String>,
    pub closure_model_id: String,
    pub closure_model_evidence_binding: String,
    pub flow_support_id: String,
    pub flow_support_evidence_binding: String,
    pub viability: RegenerativeLineageViabilityReportV1,
}

/// Derive lineage viability while preserving the exact evidence identities used.
pub fn evaluate_regenerative_lineage_evidence(
    profile: &RegenerativeLineageViabilityProfileV1,
    genome: &RegenerativeGenomeV1,
    model: &RegenerativeClosureModel,
    support: &RegenerativeFlowSupportV1,
) -> Result<RegenerativeLineageEvidenceEnvelopeV1, RegenerativeLineageViabilityError> {
    let viability = evaluate_regenerative_lineage_viability(profile, genome, model, support)?;
    Ok(RegenerativeLineageEvidenceEnvelopeV1 {
        profile_id: profile.profile_id.clone(),
        profile_evidence_binding: profile.evidence_binding.clone(),
        genome_id: genome.genome_id.clone(),
        genome_evidence_binding: genome.evidence_binding.clone(),
        lineage_parent_binding: genome.lineage_parent_binding.clone(),
        closure_model_id: model.model_id.clone(),
        closure_model_evidence_binding: model.evidence_binding.clone(),
        flow_support_id: support.support_id.clone(),
        flow_support_evidence_binding: support.evidence_binding.clone(),
        viability,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DependencyGovernance, RegenerativeCapability, RegenerativeDependency,
        RegenerativeDependencyKind, RegenerativeGenomeRequirementV1, RegenerativeHorizon,
        REGENERATIVE_FLOW_SUPPORT_SCHEMA_V1, REGENERATIVE_GENOME_SCHEMA_V1,
        REGENERATIVE_LINEAGE_VIABILITY_SCHEMA_V1,
    };
    use std::collections::BTreeSet;

    #[test]
    fn envelope_preserves_exact_input_evidence_identities() {
        let model = RegenerativeClosureModel {
            model_id: "lineage-evidence-model".into(),
            period_duration_ms: 1,
            dependencies: vec![RegenerativeDependency {
                dependency_id: "common".into(),
                kind: RegenerativeDependencyKind::Component,
                governance: DependencyGovernance::Ordinary,
                demand_units_per_period: 1,
                local_production_units_per_period: 0,
                recycling_units_per_period: 0,
                stockpile_units: 5,
                unit_mass_grams: Some(1),
                evidence_binding: "dep:common".into(),
            }],
            capabilities: vec![RegenerativeCapability {
                capability_id: "common-capability".into(),
                essential: true,
                dependency_ids: BTreeSet::from(["common".into()]),
                evidence_binding: "cap:common".into(),
            }],
            evidence_binding: "model-evidence:v1".into(),
        };
        let genome = RegenerativeGenomeV1 {
            schema_version: REGENERATIVE_GENOME_SCHEMA_V1,
            genome_id: "genome:v1".into(),
            lineage_parent_binding: Some("genome:parent-evidence".into()),
            closure_model_id: model.model_id.clone(),
            closure_model_evidence_binding: model.evidence_binding.clone(),
            requirements: vec![RegenerativeGenomeRequirementV1 {
                requirement_id: "req-common".into(),
                capability_id: "common-capability".into(),
                baseline_dependency_id: "common".into(),
                design_binding: "design:common".into(),
                metrology_profile_binding: "metrology:common".into(),
                requalification_profile_binding: "requalification:common".into(),
                disassembly_profile_binding: "disassembly:common".into(),
                recovery_profile_binding: "recovery:common".into(),
                qualified_substitution_bindings: Vec::new(),
            }],
            evidence_binding: "genome-evidence:v1".into(),
        };
        let support = RegenerativeFlowSupportV1 {
            schema_version: REGENERATIVE_FLOW_SUPPORT_SCHEMA_V1,
            support_id: "support:v1".into(),
            closure_model_id: model.model_id.clone(),
            closure_model_evidence_binding: model.evidence_binding.clone(),
            claims: Vec::new(),
            evidence_binding: "support-evidence:v1".into(),
        };
        let profile = RegenerativeLineageViabilityProfileV1 {
            schema_version: REGENERATIVE_LINEAGE_VIABILITY_SCHEMA_V1,
            profile_id: "profile:v1".into(),
            closure_model_id: model.model_id.clone(),
            closure_model_evidence_binding: model.evidence_binding.clone(),
            operational_capability_ids: vec!["common-capability".into()],
            successor_construction_capability_ids: vec!["common-capability".into()],
            successor_qualification_capability_ids: vec!["common-capability".into()],
            evidence_binding: "profile-evidence:v1".into(),
        };

        let envelope =
            evaluate_regenerative_lineage_evidence(&profile, &genome, &model, &support).unwrap();
        assert_eq!(envelope.profile_evidence_binding, "profile-evidence:v1");
        assert_eq!(envelope.genome_evidence_binding, "genome-evidence:v1");
        assert_eq!(envelope.closure_model_evidence_binding, "model-evidence:v1");
        assert_eq!(envelope.flow_support_evidence_binding, "support-evidence:v1");
        assert_eq!(
            envelope.lineage_parent_binding,
            Some("genome:parent-evidence".into())
        );
        assert_eq!(
            envelope.viability.regenerative_viability_horizon,
            RegenerativeHorizon::FinitePeriods(5)
        );
    }
}
