// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Candidate-evidence bindings from trust-root qualification into DA-034..DA-037.

#![deny(unsafe_code)]

use std::collections::BTreeMap;

use symthaea_domain_awareness_evidence::{ArtifactBinding, CandidateEvidence};
use symthaea_formal_safety::DomainAwarenessObligation;
use symthaea_trust_root_crucible::{TrustRootCrucibleReport, TrustRootCrucibleStatus};
use symthaea_trust_root_readiness_crucible::{
    TrustRootReadinessCrucibleReport, TrustRootReadinessCrucibleStatus,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrustRootEvidenceBindingError {
    InvalidArtifactBinding,
    ComponentCrucibleDidNotPass,
    ReadinessCrucibleDidNotPass,
    DuplicateScenarioId(String),
    MissingRequiredScenario(String),
    RequiredScenarioDidNotPass(String),
    InvalidCandidate,
}

fn component_scenarios(
    report: &TrustRootCrucibleReport,
) -> Result<BTreeMap<&str, bool>, TrustRootEvidenceBindingError> {
    if report.status != TrustRootCrucibleStatus::Pass
        || report.scenarios.iter().any(|scenario| !scenario.passed)
    {
        return Err(TrustRootEvidenceBindingError::ComponentCrucibleDidNotPass);
    }
    index_scenarios(
        report
            .scenarios
            .iter()
            .map(|scenario| (scenario.scenario_id.as_str(), scenario.passed)),
    )
}

fn readiness_scenarios(
    report: &TrustRootReadinessCrucibleReport,
) -> Result<BTreeMap<&str, bool>, TrustRootEvidenceBindingError> {
    if report.status != TrustRootReadinessCrucibleStatus::Pass
        || report.scenarios.iter().any(|scenario| !scenario.passed)
    {
        return Err(TrustRootEvidenceBindingError::ReadinessCrucibleDidNotPass);
    }
    index_scenarios(
        report
            .scenarios
            .iter()
            .map(|scenario| (scenario.scenario_id.as_str(), scenario.passed)),
    )
}

fn index_scenarios<'a>(
    scenarios: impl IntoIterator<Item = (&'a str, bool)>,
) -> Result<BTreeMap<&'a str, bool>, TrustRootEvidenceBindingError> {
    let mut by_id = BTreeMap::new();
    for (id, passed) in scenarios {
        if by_id.insert(id, passed).is_some() {
            return Err(TrustRootEvidenceBindingError::DuplicateScenarioId(
                id.to_string(),
            ));
        }
    }
    Ok(by_id)
}

fn require_scenarios(
    by_id: &BTreeMap<&str, bool>,
    required: &[&str],
) -> Result<(), TrustRootEvidenceBindingError> {
    for id in required {
        let Some(passed) = by_id.get(id).copied() else {
            return Err(TrustRootEvidenceBindingError::MissingRequiredScenario(
                (*id).to_string(),
            ));
        };
        if !passed {
            return Err(TrustRootEvidenceBindingError::RequiredScenarioDidNotPass(
                (*id).to_string(),
            ));
        }
    }
    Ok(())
}

fn candidate(
    obligation: DomainAwarenessObligation,
    source_id: &str,
    binding: &ArtifactBinding,
    observed_at_ms: u64,
    rationale: impl Into<String>,
) -> Result<CandidateEvidence, TrustRootEvidenceBindingError> {
    if !binding.validate() {
        return Err(TrustRootEvidenceBindingError::InvalidArtifactBinding);
    }
    let value = CandidateEvidence {
        candidate_id: format!("{}:{}:{}", obligation.code(), source_id, observed_at_ms),
        obligation,
        evidence_ref: binding.evidence_ref.clone(),
        evidence_digest: binding.evidence_digest.clone(),
        observed_at_ms,
        rationale: rationale.into(),
    };
    value
        .validate()
        .then_some(value)
        .ok_or(TrustRootEvidenceBindingError::InvalidCandidate)
}

/// Convert passing component and final-readiness qualification into candidate
/// evidence for DA-034..DA-037.
///
/// DA-035 and DA-036 deliberately receive two candidate objects each because
/// their controlled claims span both component behavior and final readiness
/// composition. None of these candidates creates a verified receipt or mutates
/// obligation workflow state.
pub fn trust_root_candidates(
    component_report: &TrustRootCrucibleReport,
    component_binding: &ArtifactBinding,
    readiness_report: &TrustRootReadinessCrucibleReport,
    readiness_binding: &ArtifactBinding,
    observed_at_ms: u64,
) -> Result<Vec<CandidateEvidence>, TrustRootEvidenceBindingError> {
    if !component_binding.validate() || !readiness_binding.validate() {
        return Err(TrustRootEvidenceBindingError::InvalidArtifactBinding);
    }
    let component = component_scenarios(component_report)?;
    let readiness = readiness_scenarios(readiness_report)?;

    require_scenarios(
        &readiness,
        &[
            "exact_current_trust_root_ready",
            "checkpoint_anchor_substitution_invalid",
            "trust_store_identity_substitution_invalid",
            "current_counter_regression_invalid",
            "accepted_recovered_trust_root_can_retain_readiness",
        ],
    )?;
    let da034 = candidate(
        DomainAwarenessObligation::CurrentPolicyAnchorRequiresMonotonicTrustRoot,
        "readiness",
        readiness_binding,
        observed_at_ms,
        "end-to-end readiness controls show that the current policy anchor must match the current monotonic trust-store state, with counter/store substitution failing closed and reviewed recovered state able to retain readiness",
    )?;

    require_scenarios(
        &component,
        &[
            "external_checkpoint_attestation_positive",
            "trust_store_self_verification_rejected",
            "checkpoint_attestation_substitution_rejected",
        ],
    )?;
    let da035_component = candidate(
        DomainAwarenessObligation::TrustStoreAttestationRequiresIndependentVerification,
        "component-attestation",
        component_binding,
        observed_at_ms,
        "component controls demonstrate exact checkpoint/profile attestation binding, rejection of store self-verification, and rejection of checkpoint-attestation substitution",
    )?;

    require_scenarios(
        &readiness,
        &[
            "exact_current_trust_root_ready",
            "attestation_substitution_invalid",
            "late_attestation_verification_cannot_retroactively_enable_interval",
        ],
    )?;
    let da035_readiness = candidate(
        DomainAwarenessObligation::TrustStoreAttestationRequiresIndependentVerification,
        "readiness-attestation",
        readiness_binding,
        observed_at_ms,
        "end-to-end readiness controls demonstrate that exact independent attestation verification gates readiness and cannot act retroactively across the trusted-time interval",
    )?;

    require_scenarios(
        &component,
        &[
            "old_backup_restore_blocked",
            "forged_backup_snapshot_blocked",
            "reviewed_recovery_continuity_positive",
            "replacement_checkpoint_substitution_blocked",
            "invalid_continuity_cannot_enter_recovery_ledger",
        ],
    )?;
    let da036_component = candidate(
        DomainAwarenessObligation::TrustStoreRecoveryPreservesExactContinuity,
        "component-recovery",
        component_binding,
        observed_at_ms,
        "component controls demonstrate explicit reviewed recovery, exact pre-loss snapshot continuity, stale/forged backup rejection, replacement-checkpoint binding, and refusal to accept invalid continuity",
    )?;

    require_scenarios(
        &readiness,
        &[
            "exact_current_trust_root_ready",
            "accepted_recovered_trust_root_can_retain_readiness",
            "trust_store_identity_substitution_invalid",
        ],
    )?;
    let da036_readiness = candidate(
        DomainAwarenessObligation::TrustStoreRecoveryPreservesExactContinuity,
        "readiness-recovery",
        readiness_binding,
        observed_at_ms,
        "end-to-end controls show that an accepted reviewed recovery can retain readiness while unreviewed trust-store identity substitution cannot",
    )?;

    require_scenarios(
        &component,
        &[
            "reviewed_recovery_continuity_positive",
            "invalid_continuity_cannot_enter_recovery_ledger",
            "authorization_replay_rejected",
            "old_checkpoint_recovery_fork_rejected",
            "accepted_recovered_segment_positive",
            "unaccepted_recovered_segment_rejected",
        ],
    )?;
    let da037 = candidate(
        DomainAwarenessObligation::TrustStoreRecoveryAcceptanceIsOneShotAndForkResistant,
        "component-recovery-ledger",
        component_binding,
        observed_at_ms,
        "recovery-ledger controls demonstrate one-shot authorization, old-checkpoint fork rejection, and exact accepted recovered-segment selection",
    )?;

    Ok(vec![
        da034,
        da035_component,
        da035_readiness,
        da036_component,
        da036_readiness,
        da037,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_trust_root_crucible::run_trust_root_crucible;
    use symthaea_trust_root_readiness_crucible::run_trust_root_readiness_crucible;

    fn component_binding() -> ArtifactBinding {
        ArtifactBinding {
            evidence_ref: "artifact:trust-root-component-qualification".into(),
            evidence_digest: "blake3:trust-root-component-qualification".into(),
        }
    }

    fn readiness_binding() -> ArtifactBinding {
        ArtifactBinding {
            evidence_ref: "artifact:trust-root-readiness-qualification".into(),
            evidence_digest: "blake3:trust-root-readiness-qualification".into(),
        }
    }

    #[test]
    fn passing_qualification_emits_six_atomic_candidates_for_four_obligations() {
        let component = run_trust_root_crucible();
        let readiness = run_trust_root_readiness_crucible();
        let candidates = trust_root_candidates(
            &component,
            &component_binding(),
            &readiness,
            &readiness_binding(),
            10_000,
        )
        .unwrap();
        assert_eq!(candidates.len(), 6);
        assert_eq!(
            candidates
                .iter()
                .filter(|candidate| candidate.obligation
                    == DomainAwarenessObligation::TrustStoreAttestationRequiresIndependentVerification)
                .count(),
            2
        );
        assert_eq!(
            candidates
                .iter()
                .filter(|candidate| candidate.obligation
                    == DomainAwarenessObligation::TrustStoreRecoveryPreservesExactContinuity)
                .count(),
            2
        );
        assert!(candidates.iter().all(CandidateEvidence::validate));
        assert!(candidates
            .iter()
            .all(|candidate| !candidate.grants_physical_authority()));
    }

    #[test]
    fn missing_component_recovery_control_fails_closed() {
        let mut component = run_trust_root_crucible();
        component
            .scenarios
            .retain(|scenario| scenario.scenario_id != "old_backup_restore_blocked");
        let result = trust_root_candidates(
            &component,
            &component_binding(),
            &run_trust_root_readiness_crucible(),
            &readiness_binding(),
            10_000,
        );
        assert!(matches!(
            result,
            Err(TrustRootEvidenceBindingError::MissingRequiredScenario(id))
                if id == "old_backup_restore_blocked"
        ));
    }

    #[test]
    fn missing_readiness_binding_control_fails_closed() {
        let mut readiness = run_trust_root_readiness_crucible();
        readiness
            .scenarios
            .retain(|scenario| scenario.scenario_id != "checkpoint_anchor_substitution_invalid");
        let result = trust_root_candidates(
            &run_trust_root_crucible(),
            &component_binding(),
            &readiness,
            &readiness_binding(),
            10_000,
        );
        assert!(matches!(
            result,
            Err(TrustRootEvidenceBindingError::MissingRequiredScenario(id))
                if id == "checkpoint_anchor_substitution_invalid"
        ));
    }

    #[test]
    fn failed_component_or_readiness_crucible_cannot_create_candidates() {
        let mut component = run_trust_root_crucible();
        component.status = TrustRootCrucibleStatus::Fail;
        let result = trust_root_candidates(
            &component,
            &component_binding(),
            &run_trust_root_readiness_crucible(),
            &readiness_binding(),
            10_000,
        );
        assert_eq!(
            result,
            Err(TrustRootEvidenceBindingError::ComponentCrucibleDidNotPass)
        );
    }
}
