// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Candidate-evidence bindings from policy-governance qualification into
//! canonical DomainAwareness obligations.

#![deny(unsafe_code)]

use std::collections::BTreeMap;

use symthaea_domain_awareness_evidence::{ArtifactBinding, CandidateEvidence};
use symthaea_formal_safety::DomainAwarenessObligation;
use symthaea_policy_governance_crucible::{
    PolicyGovernanceCrucibleReport, PolicyGovernanceCrucibleStatus,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyGovernanceBindingError {
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
) -> Result<CandidateEvidence, PolicyGovernanceBindingError> {
    if !binding.validate() {
        return Err(PolicyGovernanceBindingError::InvalidArtifactBinding);
    }
    let candidate = CandidateEvidence {
        candidate_id: format!("{}:policy-governance-crucible:{}", obligation.code(), observed_at_ms),
        obligation,
        evidence_ref: binding.evidence_ref.clone(),
        evidence_digest: binding.evidence_digest.clone(),
        observed_at_ms,
        rationale: rationale.into(),
    };
    if !candidate.validate() {
        return Err(PolicyGovernanceBindingError::InvalidArtifactBinding);
    }
    Ok(candidate)
}

fn scenario_index(
    report: &PolicyGovernanceCrucibleReport,
) -> Result<BTreeMap<&str, bool>, PolicyGovernanceBindingError> {
    if report.status != PolicyGovernanceCrucibleStatus::Pass
        || report.scenarios.iter().any(|scenario| !scenario.passed)
    {
        return Err(PolicyGovernanceBindingError::CrucibleDidNotPass);
    }

    let mut by_id = BTreeMap::new();
    for scenario in &report.scenarios {
        if by_id.insert(scenario.scenario_id.as_str(), scenario.passed).is_some() {
            return Err(PolicyGovernanceBindingError::DuplicateScenarioId(
                scenario.scenario_id.clone(),
            ));
        }
    }
    Ok(by_id)
}

fn require_scenarios(
    by_id: &BTreeMap<&str, bool>,
    required: &[&str],
) -> Result<(), PolicyGovernanceBindingError> {
    for scenario_id in required {
        let Some(passed) = by_id.get(scenario_id).copied() else {
            return Err(PolicyGovernanceBindingError::MissingRequiredScenario(
                (*scenario_id).to_string(),
            ));
        };
        if !passed {
            return Err(PolicyGovernanceBindingError::RequiredScenarioDidNotPass(
                (*scenario_id).to_string(),
            ));
        }
    }
    Ok(())
}

/// Convert a passing policy-governance crucible into candidate evidence for
/// DA-031..DA-033.
///
/// The artifact binding is expected to identify the complete stacked governance
/// qualification bundle. In particular, independent verification should include
/// the lower lineage-gate tests for exact signature-verification-record binding in
/// addition to the scenario matrix checked here.
pub fn policy_governance_candidates(
    report: &PolicyGovernanceCrucibleReport,
    binding: &ArtifactBinding,
    observed_at_ms: u64,
) -> Result<Vec<CandidateEvidence>, PolicyGovernanceBindingError> {
    let by_id = scenario_index(report)?;

    require_scenarios(
        &by_id,
        &[
            "baseline-ready",
            "truncated-lineage-rollback-blocks",
            "deleted-policy-revision-invalid",
            "policy-manifest-substitution-invalid",
        ],
    )?;
    let signed_tip = candidate(
        DomainAwarenessObligation::CurrentPolicyMustBeSignedLineageTip,
        binding,
        observed_at_ms,
        "governance qualification demonstrates that readiness requires the current manifest to be the valid signed lineage tip and rejects lineage truncation, revision deletion, and same-revision manifest substitution; independent verification must also confirm the bound lower-layer signature-record-splicing tests",
    )?;

    require_scenarios(
        &by_id,
        &[
            "baseline-ready",
            "unreviewed-signer-replacement-invalid",
            "reviewed-signer-rotation-ready",
            "signing-governance-substitution-invalid",
        ],
    )?;
    let signing_authority = candidate(
        DomainAwarenessObligation::ManifestSigningAuthorityIsExternallyGoverned,
        binding,
        observed_at_ms,
        "positive and negative signer-governance controls demonstrate that signer/key replacement requires an explicit reviewed transition under the provisioned external governance digest and that a weaker substituted authority policy cannot preserve readiness",
    )?;

    require_scenarios(
        &by_id,
        &[
            "baseline-ready",
            "truncated-lineage-rollback-blocks",
            "old-anchor-substitution-invalid",
            "uncheckpointed-forward-revision-blocks",
            "late-anchor-cannot-retroactively-enable-readiness",
            "deleted-anchor-revision-invalid",
        ],
    )?;
    let rollback_anchor = candidate(
        DomainAwarenessObligation::PolicyLineageRequiresExternalRollbackAnchor,
        binding,
        observed_at_ms,
        "external-checkpoint controls demonstrate that signed history behind the anchored tip is blocked, forward policy revisions remain blocked until checkpointed, old-anchor substitution is invalid, late anchors cannot act retroactively, and anchor-history deletion fails closed",
    )?;

    Ok(vec![signed_tip, signing_authority, rollback_anchor])
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_policy_governance_crucible::run_policy_governance_crucible;

    fn binding() -> ArtifactBinding {
        ArtifactBinding {
            evidence_ref: "artifact:policy-governance-qualification-bundle".into(),
            evidence_digest: "blake3:policy-governance-qualification-bundle".into(),
        }
    }

    #[test]
    fn passing_governance_crucible_maps_exactly_to_da_031_through_da_033() {
        let report = run_policy_governance_crucible();
        let candidates = policy_governance_candidates(&report, &binding(), 2_000).unwrap();
        assert_eq!(candidates.len(), 3);
        assert_eq!(
            candidates[0].obligation,
            DomainAwarenessObligation::CurrentPolicyMustBeSignedLineageTip
        );
        assert_eq!(
            candidates[1].obligation,
            DomainAwarenessObligation::ManifestSigningAuthorityIsExternallyGoverned
        );
        assert_eq!(
            candidates[2].obligation,
            DomainAwarenessObligation::PolicyLineageRequiresExternalRollbackAnchor
        );
        assert!(candidates.iter().all(|candidate| !candidate.grants_physical_authority()));
    }

    #[test]
    fn missing_signed_tip_control_fails_closed() {
        let mut report = run_policy_governance_crucible();
        report
            .scenarios
            .retain(|scenario| scenario.scenario_id != "policy-manifest-substitution-invalid");
        let result = policy_governance_candidates(&report, &binding(), 2_000);
        assert!(matches!(
            result,
            Err(PolicyGovernanceBindingError::MissingRequiredScenario(id))
                if id == "policy-manifest-substitution-invalid"
        ));
    }

    #[test]
    fn missing_signing_authority_control_fails_closed() {
        let mut report = run_policy_governance_crucible();
        report
            .scenarios
            .retain(|scenario| scenario.scenario_id != "reviewed-signer-rotation-ready");
        let result = policy_governance_candidates(&report, &binding(), 2_000);
        assert!(matches!(
            result,
            Err(PolicyGovernanceBindingError::MissingRequiredScenario(id))
                if id == "reviewed-signer-rotation-ready"
        ));
    }

    #[test]
    fn missing_anchor_control_fails_closed() {
        let mut report = run_policy_governance_crucible();
        report
            .scenarios
            .retain(|scenario| scenario.scenario_id != "old-anchor-substitution-invalid");
        let result = policy_governance_candidates(&report, &binding(), 2_000);
        assert!(matches!(
            result,
            Err(PolicyGovernanceBindingError::MissingRequiredScenario(id))
                if id == "old-anchor-substitution-invalid"
        ));
    }

    #[test]
    fn failed_crucible_cannot_create_candidate_evidence() {
        let mut report = run_policy_governance_crucible();
        report.status = PolicyGovernanceCrucibleStatus::Fail;
        let result = policy_governance_candidates(&report, &binding(), 2_000);
        assert_eq!(result, Err(PolicyGovernanceBindingError::CrucibleDidNotPass));
    }
}
