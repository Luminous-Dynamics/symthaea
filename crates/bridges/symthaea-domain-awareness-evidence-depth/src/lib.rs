// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Candidate-evidence bindings from evidence-depth qualification into
//! canonical DomainAwareness obligations.

#![deny(unsafe_code)]

use std::collections::BTreeMap;

use symthaea_domain_awareness_evidence::{ArtifactBinding, CandidateEvidence};
use symthaea_evidence_depth_crucible::{
    EvidenceDepthCrucibleReport, EvidenceDepthCrucibleStatus,
};
use symthaea_formal_safety::DomainAwarenessObligation;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvidenceDepthBindingError {
    InvalidArtifactBinding,
    CrucibleDidNotPass,
    DuplicateScenarioId(String),
    MissingRequiredScenario(String),
    RequiredScenarioDidNotPass(String),
}

fn candidate(
    obligation: DomainAwarenessObligation,
    binding: &ArtifactBinding,
    observed_at_ms: u64,
    rationale: impl Into<String>,
) -> Result<CandidateEvidence, EvidenceDepthBindingError> {
    if !binding.validate() {
        return Err(EvidenceDepthBindingError::InvalidArtifactBinding);
    }
    let candidate = CandidateEvidence {
        candidate_id: format!("{}:evidence-depth-crucible:{}", obligation.code(), observed_at_ms),
        obligation,
        evidence_ref: binding.evidence_ref.clone(),
        evidence_digest: binding.evidence_digest.clone(),
        observed_at_ms,
        rationale: rationale.into(),
    };
    if !candidate.validate() {
        return Err(EvidenceDepthBindingError::InvalidArtifactBinding);
    }
    Ok(candidate)
}

fn scenario_index(
    report: &EvidenceDepthCrucibleReport,
) -> Result<BTreeMap<&str, bool>, EvidenceDepthBindingError> {
    if report.status != EvidenceDepthCrucibleStatus::Pass
        || report.scenarios.iter().any(|scenario| !scenario.passed)
    {
        return Err(EvidenceDepthBindingError::CrucibleDidNotPass);
    }

    let mut by_id = BTreeMap::new();
    for scenario in &report.scenarios {
        if by_id.insert(scenario.scenario_id.as_str(), scenario.passed).is_some() {
            return Err(EvidenceDepthBindingError::DuplicateScenarioId(
                scenario.scenario_id.clone(),
            ));
        }
    }
    Ok(by_id)
}

fn require_scenarios(
    by_id: &BTreeMap<&str, bool>,
    required: &[&str],
) -> Result<(), EvidenceDepthBindingError> {
    for scenario_id in required {
        let Some(passed) = by_id.get(scenario_id).copied() else {
            return Err(EvidenceDepthBindingError::MissingRequiredScenario(
                (*scenario_id).to_string(),
            ));
        };
        if !passed {
            return Err(EvidenceDepthBindingError::RequiredScenarioDidNotPass(
                (*scenario_id).to_string(),
            ));
        }
    }
    Ok(())
}

/// Convert a passing evidence-depth crucible into candidate evidence for DA-029
/// and DA-030. The candidates remain subject to independent verification,
/// scoping, lifecycle checks, and explicit workflow discharge.
pub fn evidence_depth_candidates(
    report: &EvidenceDepthCrucibleReport,
    binding: &ArtifactBinding,
    observed_at_ms: u64,
) -> Result<Vec<CandidateEvidence>, EvidenceDepthBindingError> {
    let by_id = scenario_index(report)?;

    require_scenarios(
        &by_id,
        &[
            "verifier_diverse_positive",
            "duplicate_receipts_same_verifier_block",
            "distinct_verifier_ids_same_org_block",
            "missing_verifier_profile_blocks",
            "duplicate_verifier_profile_invalid",
        ],
    )?;
    let verifier_diversity = candidate(
        DomainAwarenessObligation::VerifierCommonCauseDiversityRequired,
        binding,
        observed_at_ms,
        "positive and negative verifier controls demonstrate that receipt multiplicity cannot manufacture policy-required independence across verifier common-cause fault domains",
    )?;

    require_scenarios(
        &by_id,
        &[
            "atomic_explicit_positive",
            "parent_receipt_has_no_implicit_facet_coverage",
            "facet_coverage_does_not_leak_to_sibling",
            "same_subartifact_cannot_fake_distinct_evidence_objects",
            "distinct_subartifacts_can_satisfy_replication_policy",
        ],
    )?;
    let atomic_coverage = candidate(
        DomainAwarenessObligation::CompositeObligationsRequireAtomicCoverage,
        binding,
        observed_at_ms,
        "positive and negative atomic-coverage controls demonstrate that parent receipts do not implicitly cover facets and reviewed replication requirements count explicit facet-level evidence only",
    )?;

    Ok(vec![verifier_diversity, atomic_coverage])
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_evidence_depth_crucible::run_evidence_depth_crucible;

    fn binding() -> ArtifactBinding {
        ArtifactBinding {
            evidence_ref: "artifact:evidence-depth-qualification-bundle".into(),
            evidence_digest: "blake3:evidence-depth-qualification-bundle".into(),
        }
    }

    #[test]
    fn passing_depth_crucible_maps_exactly_to_da_029_and_da_030() {
        let report = run_evidence_depth_crucible();
        let candidates = evidence_depth_candidates(&report, &binding(), 2_000).unwrap();
        assert_eq!(candidates.len(), 2);
        assert_eq!(
            candidates[0].obligation,
            DomainAwarenessObligation::VerifierCommonCauseDiversityRequired
        );
        assert_eq!(
            candidates[1].obligation,
            DomainAwarenessObligation::CompositeObligationsRequireAtomicCoverage
        );
        assert!(candidates.iter().all(|candidate| !candidate.grants_physical_authority()));
    }

    #[test]
    fn missing_required_verifier_control_fails_closed() {
        let mut report = run_evidence_depth_crucible();
        report
            .scenarios
            .retain(|scenario| scenario.scenario_id != "distinct_verifier_ids_same_org_block");
        let result = evidence_depth_candidates(&report, &binding(), 2_000);
        assert!(matches!(
            result,
            Err(EvidenceDepthBindingError::MissingRequiredScenario(id))
                if id == "distinct_verifier_ids_same_org_block"
        ));
    }

    #[test]
    fn missing_required_atomic_control_fails_closed() {
        let mut report = run_evidence_depth_crucible();
        report
            .scenarios
            .retain(|scenario| scenario.scenario_id != "facet_coverage_does_not_leak_to_sibling");
        let result = evidence_depth_candidates(&report, &binding(), 2_000);
        assert!(matches!(
            result,
            Err(EvidenceDepthBindingError::MissingRequiredScenario(id))
                if id == "facet_coverage_does_not_leak_to_sibling"
        ));
    }

    #[test]
    fn failed_crucible_cannot_create_candidate_evidence() {
        let mut report = run_evidence_depth_crucible();
        report.status = EvidenceDepthCrucibleStatus::Fail;
        let result = evidence_depth_candidates(&report, &binding(), 2_000);
        assert_eq!(result, Err(EvidenceDepthBindingError::CrucibleDidNotPass));
    }
}
