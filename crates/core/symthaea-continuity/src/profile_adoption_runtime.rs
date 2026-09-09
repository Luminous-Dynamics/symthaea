// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Attenuation-aware runtime policy for persisted verifier-profile adoptions.
//!
//! Commit-time currentness is intentionally strict: changing a grant while a write
//! is in flight invalidates that write. Runtime use after a successful historical
//! commit has different semantics. A newer local grant may narrow authority
//! immediately without requiring the verifier to be re-adopted, but it may never
//! widen authority beyond the exact signed adoption.
//!
//! Core theorem:
//!
//! `effective runtime authority = adopted envelope ∩ current local grant`.
//!
//! This module still proves policy currentness only. It does not prove that the
//! historical adoption was cryptographically authenticated or actually committed.

use thiserror::Error;

use crate::contract::{ContinuityRequirementId, ValidatedContinuityContractV1};
use crate::profile_adoption::{VerifierAdoptionScopeV1, VerifierProfileAdoptionError};
use crate::profile_adoption_admission::{
    VerifierProfileAdoptionAdmissionError, VerifierProfileAdoptionAdmissionPolicyV1,
    VerifierProfileAdoptionHeadV1,
};
use crate::profile_adoption_authority::VerifierAdoptionAuthorityGrantV1;
use crate::profile_adoption_commit::VerifierAdoptionAuthorityRootSnapshotV1;
use crate::profile_adoption_currentness::{
    VerifierProfileAdoptionCurrentnessError, VerifierProfileAdoptionRegistryRecordIdV1,
    VerifierProfileAdoptionRegistryRecordV1,
};
use crate::verifier::VerifierProfileV1;
use crate::witness::EvidenceClass;

/// Non-serializable runtime policy envelope for one exact currently observed
/// verifier adoption.
///
/// This is deliberately not named `AuthorizedVerifierProfile`: it contains no
/// cryptographic committed-adoption proof. A future authority type must combine
/// this policy envelope with an authenticated committed-adoption receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyCurrentVerifierRuntimeEnvelopeV1 {
    record_id: VerifierProfileAdoptionRegistryRecordIdV1,
    profile: VerifierProfileV1,
    effective_evidence_class: EvidenceClass,
    effective_scope: VerifierAdoptionScopeV1,
    current_grant_epoch: u64,
    current_root_epoch: u64,
    current_head: VerifierProfileAdoptionHeadV1,
}

impl PolicyCurrentVerifierRuntimeEnvelopeV1 {
    pub fn record_id(&self) -> VerifierProfileAdoptionRegistryRecordIdV1 { self.record_id }
    pub fn profile(&self) -> &VerifierProfileV1 { &self.profile }
    pub fn effective_evidence_class(&self) -> EvidenceClass { self.effective_evidence_class }
    pub fn effective_scope(&self) -> &VerifierAdoptionScopeV1 { &self.effective_scope }
    pub fn current_grant_epoch(&self) -> u64 { self.current_grant_epoch }
    pub fn current_root_epoch(&self) -> u64 { self.current_root_epoch }
    pub fn current_head(&self) -> &VerifierProfileAdoptionHeadV1 { &self.current_head }
}

/// Revalidate one persisted adoption for runtime policy use while allowing only
/// monotone attenuation of the local grant.
///
/// `current_grant == None` is explicit revocation. A newer grant epoch may narrow
/// evidence strength and/or scope; widening the local grant never widens the
/// verifier beyond the exact adopted subject.
#[allow(clippy::too_many_arguments)]
pub fn check_verifier_runtime_policy(
    record: &VerifierProfileAdoptionRegistryRecordV1,
    now_unix_ms: u64,
    current_root: &VerifierAdoptionAuthorityRootSnapshotV1,
    current_grant: Option<&VerifierAdoptionAuthorityGrantV1>,
    current_head: &VerifierProfileAdoptionHeadV1,
    current_profile: &VerifierProfileV1,
    scope_contract: Option<&ValidatedContinuityContractV1>,
) -> Result<PolicyCurrentVerifierRuntimeEnvelopeV1, VerifierProfileRuntimePolicyError> {
    record.validate()?;

    let stored_root = record.authority_root();
    if stored_root.authority_subject() != current_root.authority_subject()
        || stored_root.root_id() != current_root.root_id()
        || stored_root.root_digest() != current_root.root_digest()
        || stored_root.provisioning_epoch() != current_root.epoch()
    {
        return Err(VerifierProfileRuntimePolicyError::AuthorityRootNoLongerCurrent);
    }

    let grant = current_grant.ok_or(VerifierProfileRuntimePolicyError::AuthorityGrantRevoked)?;
    let stored_grant = record.authority_grant();
    if stored_grant.grant_id_text() != grant.grant_id_text()
        || stored_grant.authority_subject() != grant.authority_subject()
        || stored_grant.authority_root_id() != grant.authority_root_id()
        || stored_grant.authority_root_digest() != grant.authority_root_digest()
        || stored_grant.verifier_role_id() != grant.verifier_role_id()
    {
        return Err(VerifierProfileRuntimePolicyError::AuthorityGrantSlotChanged);
    }
    if grant.grant_epoch() < stored_grant.grant_epoch() {
        return Err(VerifierProfileRuntimePolicyError::AuthorityGrantEpochRollback {
            recorded: stored_grant.grant_epoch(),
            observed: grant.grant_epoch(),
        });
    }

    let transition = record.transition();
    let subject = transition.subject();
    if grant.authority_subject() != subject.authority_subject()
        || grant.authority_root_id() != subject.authority_root_id()
        || grant.authority_root_digest() != subject.authority_root_digest()
        || grant.verifier_role_id() != subject.verifier_role_id()
    {
        return Err(VerifierProfileRuntimePolicyError::CurrentGrantAuthorityMismatch);
    }

    let candidate_head = VerifierProfileAdoptionHeadV1::from_transition(transition)?;
    if &candidate_head != current_head {
        return Err(VerifierProfileRuntimePolicyError::AdoptionNoLongerCurrentHead);
    }

    let predecessor_head = match record.predecessor_transition() {
        None => VerifierProfileAdoptionHeadV1::Uninitialized,
        Some(predecessor) => VerifierProfileAdoptionHeadV1::from_transition(predecessor)?,
    };
    let admission = VerifierProfileAdoptionAdmissionPolicyV1::new(
        subject.authority_subject(),
        subject.authority_root_id(),
        subject.authority_root_digest(),
        subject.verifier_role_id(),
        predecessor_head,
    )?;
    // Re-running admission rechecks exact profile identity, current validity, exact
    // predecessor lineage, and adopted scope membership in the supplied contract.
    admission.check(now_unix_ms, transition, current_profile, scope_contract)?;

    ground_scope(grant.allowed_scope(), scope_contract, true)?;
    let effective_scope = intersect_scopes(subject.scope(), grant.allowed_scope())
        .ok_or(VerifierProfileRuntimePolicyError::NoEffectiveScope)?;
    ground_scope(&effective_scope, scope_contract, false)?;

    let effective_evidence_class = weaker_class(
        weaker_class(subject.evidence_class_ceiling(), current_profile.evidence_class()),
        grant.maximum_evidence_class(),
    );

    Ok(PolicyCurrentVerifierRuntimeEnvelopeV1 {
        record_id: record.id(),
        profile: current_profile.clone(),
        effective_evidence_class,
        effective_scope,
        current_grant_epoch: grant.grant_epoch(),
        current_root_epoch: current_root.epoch(),
        current_head: current_head.clone(),
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum VerifierProfileRuntimePolicyError {
    #[error(transparent)]
    Record(#[from] VerifierProfileAdoptionCurrentnessError),
    #[error(transparent)]
    Adoption(#[from] VerifierProfileAdoptionError),
    #[error(transparent)]
    Admission(#[from] VerifierProfileAdoptionAdmissionError),
    #[error("adoption-authority root provisioning state no longer matches the persisted record")]
    AuthorityRootNoLongerCurrent,
    #[error("local verifier-adoption authority grant is revoked")]
    AuthorityGrantRevoked,
    #[error("current authority grant belongs to a different grant slot/authority/role")]
    AuthorityGrantSlotChanged,
    #[error("authority grant epoch rolled back: recorded {recorded}, observed {observed}")]
    AuthorityGrantEpochRollback { recorded: u64, observed: u64 },
    #[error("current authority grant does not belong to the adoption authority/role")]
    CurrentGrantAuthorityMismatch,
    #[error("persisted adoption is no longer the exact current lineage head")]
    AdoptionNoLongerCurrentHead,
    #[error("contract context is required to evaluate current verifier scope")]
    MissingScopeContract,
    #[error("current verifier scope references a different continuity contract")]
    ScopeContractMismatch,
    #[error("current authority grant references requirement outside the exact contract: {requirement:?}")]
    UnknownGrantRequirement { requirement: ContinuityRequirementId },
    #[error("effective verifier scope references requirement outside the exact contract: {requirement:?}")]
    UnknownEffectiveRequirement { requirement: ContinuityRequirementId },
    #[error("current authority grant and adopted scope have no effective overlap")]
    NoEffectiveScope,
}

fn weaker_class(a: EvidenceClass, b: EvidenceClass) -> EvidenceClass {
    if a <= b { a } else { b }
}

fn intersect_scopes(
    adopted: &VerifierAdoptionScopeV1,
    current_grant: &VerifierAdoptionScopeV1,
) -> Option<VerifierAdoptionScopeV1> {
    match (adopted, current_grant) {
        (VerifierAdoptionScopeV1::AllContinuityVerification, other)
        | (other, VerifierAdoptionScopeV1::AllContinuityVerification) => Some(other.clone()),
        (
            VerifierAdoptionScopeV1::Contract { contract_id: a },
            VerifierAdoptionScopeV1::Contract { contract_id: b },
        ) if a == b => Some(VerifierAdoptionScopeV1::Contract { contract_id: *a }),
        (
            VerifierAdoptionScopeV1::Contract { contract_id: a },
            VerifierAdoptionScopeV1::Requirements { contract_id: b, requirement_ids },
        )
        | (
            VerifierAdoptionScopeV1::Requirements { contract_id: b, requirement_ids },
            VerifierAdoptionScopeV1::Contract { contract_id: a },
        ) if a == b => VerifierAdoptionScopeV1::requirements(*a, requirement_ids.clone()).ok(),
        (
            VerifierAdoptionScopeV1::Requirements { contract_id: a, requirement_ids: left },
            VerifierAdoptionScopeV1::Requirements { contract_id: b, requirement_ids: right },
        ) if a == b => {
            let overlap: Vec<_> = left.iter().copied()
                .filter(|id| right.binary_search(id).is_ok())
                .collect();
            if overlap.is_empty() {
                None
            } else {
                VerifierAdoptionScopeV1::requirements(*a, overlap).ok()
            }
        }
        _ => None,
    }
}

fn ground_scope(
    scope: &VerifierAdoptionScopeV1,
    contract: Option<&ValidatedContinuityContractV1>,
    current_grant: bool,
) -> Result<(), VerifierProfileRuntimePolicyError> {
    match scope {
        VerifierAdoptionScopeV1::AllContinuityVerification => Ok(()),
        VerifierAdoptionScopeV1::Contract { contract_id } => {
            let contract = contract.ok_or(VerifierProfileRuntimePolicyError::MissingScopeContract)?;
            if contract.id() != *contract_id {
                return Err(VerifierProfileRuntimePolicyError::ScopeContractMismatch);
            }
            Ok(())
        }
        VerifierAdoptionScopeV1::Requirements { contract_id, requirement_ids } => {
            let contract = contract.ok_or(VerifierProfileRuntimePolicyError::MissingScopeContract)?;
            if contract.id() != *contract_id {
                return Err(VerifierProfileRuntimePolicyError::ScopeContractMismatch);
            }
            for requirement in requirement_ids {
                if !contract.requirements().iter().any(|item| item.id() == *requirement) {
                    return Err(if current_grant {
                        VerifierProfileRuntimePolicyError::UnknownGrantRequirement { requirement: *requirement }
                    } else {
                        VerifierProfileRuntimePolicyError::UnknownEffectiveRequirement { requirement: *requirement }
                    });
                }
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{
        ApprovalBasis, ContinuityContractV1, ContinuityRequirementV1, EquivalencePredicate,
        RequirementCriticality,
    };
    use crate::observation::{
        DependencyBasis, DependencyClaimV1, EvidenceBasis, ObservationCoverage,
        ObservationEnvelopeV1,
    };
    use crate::profile_adoption::{
        VerifierProfileAdoptionSubjectV1, VerifierProfileAdoptionTransitionV1,
    };
    use crate::profile_adoption_authority::bind_policy_checked_adoption_to_authority_grant;
    use crate::profile_adoption_currentness::VerifierProfileAdoptionRegistryRecordV1;

    fn profile() -> VerifierProfileV1 {
        VerifierProfileV1::new(
            "hardware-verifier-v1", [9; 32], 7, EvidenceClass::HardwareVerified,
        ).unwrap()
    }

    fn contract() -> ValidatedContinuityContractV1 {
        let mut requirements = Vec::new();
        for seed in [1u8, 2u8] {
            let observation = ObservationEnvelopeV1::new(
                format!("machine-{seed}"), "workflow.dependency", "fixture", "1",
                1_700_000_000_000 + seed as u64, ObservationCoverage::Complete,
                EvidenceBasis::Tested, [seed; 32], vec![],
            ).unwrap();
            let dependency = DependencyClaimV1::new(
                "role:research", "requires", format!("capability-{seed}"),
                DependencyBasis::Observed, vec![observation.id()], vec![],
            ).unwrap();
            requirements.push(ContinuityRequirementV1::new(
                dependency.id(), format!("workflow-{seed}"), RequirementCriticality::Must,
                EquivalencePredicate::BehavioralScenario { scenario_id: format!("scenario-{seed}") },
                ApprovalBasis::ExplicitPolicy, [seed + 10; 32],
            ).unwrap());
        }
        ContinuityContractV1::new("research-fleet", [33; 32], requirements)
            .unwrap().validate().unwrap()
    }

    fn root(epoch: u64) -> VerifierAdoptionAuthorityRootSnapshotV1 {
        VerifierAdoptionAuthorityRootSnapshotV1::new(
            "organization:test", "adoption-root-1", [0x55; 32], epoch,
        ).unwrap()
    }

    fn grant(
        epoch: u64,
        class: EvidenceClass,
        scope: VerifierAdoptionScopeV1,
    ) -> VerifierAdoptionAuthorityGrantV1 {
        VerifierAdoptionAuthorityGrantV1::new(
            "grant-1", "organization:test", "adoption-root-1", [0x55; 32],
            "hardware-verifier-v1", epoch, class, scope,
        ).unwrap()
    }

    fn fixture() -> (
        ValidatedContinuityContractV1,
        VerifierProfileV1,
        VerifierAdoptionAuthorityGrantV1,
        VerifierAdoptionAuthorityRootSnapshotV1,
        VerifierProfileAdoptionRegistryRecordV1,
        VerifierProfileAdoptionHeadV1,
    ) {
        let contract = contract();
        let profile = profile();
        let initial_grant = grant(
            3,
            EvidenceClass::HardwareVerified,
            VerifierAdoptionScopeV1::Contract { contract_id: contract.id() },
        );
        let subject = VerifierProfileAdoptionSubjectV1::new(
            "adopt-1", "organization:test", "adoption-root-1", [0x55; 32],
            &profile, 1, 1_000, 2_000, EvidenceClass::HardwareVerified,
            VerifierAdoptionScopeV1::Contract { contract_id: contract.id() },
        ).unwrap();
        let transition = VerifierProfileAdoptionTransitionV1::bootstrap(subject).unwrap();
        let admission = VerifierProfileAdoptionAdmissionPolicyV1::new(
            "organization:test", "adoption-root-1", [0x55; 32],
            "hardware-verifier-v1", VerifierProfileAdoptionHeadV1::Uninitialized,
        ).unwrap();
        let checked = admission.check(1_500, &transition, &profile, Some(&contract)).unwrap();
        let granted = bind_policy_checked_adoption_to_authority_grant(
            checked, &initial_grant, Some(&contract),
        ).unwrap();
        let root = root(4);
        let record = VerifierProfileAdoptionRegistryRecordV1::new_uncommitted_projection(
            &granted, &root, None,
        ).unwrap();
        let head = VerifierProfileAdoptionHeadV1::from_transition(&transition).unwrap();
        (contract, profile, initial_grant, root, record, head)
    }

    #[test]
    fn exact_current_grant_preserves_adopted_envelope() {
        let (contract, profile, grant, root, record, head) = fixture();
        let current = check_verifier_runtime_policy(
            &record, 1_500, &root, Some(&grant), &head, &profile, Some(&contract),
        ).unwrap();
        assert_eq!(current.effective_evidence_class(), EvidenceClass::HardwareVerified);
        assert_eq!(current.effective_scope(), &VerifierAdoptionScopeV1::Contract { contract_id: contract.id() });
    }

    #[test]
    fn newer_grant_can_narrow_evidence_strength_without_readoption() {
        let (contract, profile, _, root, record, head) = fixture();
        let narrowed = grant(
            4,
            EvidenceClass::DifferentiallyVerified,
            VerifierAdoptionScopeV1::Contract { contract_id: contract.id() },
        );
        let current = check_verifier_runtime_policy(
            &record, 1_500, &root, Some(&narrowed), &head, &profile, Some(&contract),
        ).unwrap();
        assert_eq!(current.effective_evidence_class(), EvidenceClass::DifferentiallyVerified);
        assert_eq!(current.current_grant_epoch(), 4);
    }

    #[test]
    fn newer_grant_can_narrow_scope_without_readoption() {
        let (contract, profile, _, root, record, head) = fixture();
        let requirement = contract.requirements()[0].id();
        let narrowed_scope = VerifierAdoptionScopeV1::requirements(contract.id(), vec![requirement]).unwrap();
        let narrowed = grant(4, EvidenceClass::HardwareVerified, narrowed_scope.clone());
        let current = check_verifier_runtime_policy(
            &record, 1_500, &root, Some(&narrowed), &head, &profile, Some(&contract),
        ).unwrap();
        assert_eq!(current.effective_scope(), &narrowed_scope);
    }

    #[test]
    fn widened_grant_cannot_widen_beyond_adoption() {
        let contract = contract();
        let profile = profile();
        let requirement = contract.requirements()[0].id();
        let adopted_scope = VerifierAdoptionScopeV1::requirements(contract.id(), vec![requirement]).unwrap();
        let initial_grant = grant(3, EvidenceClass::HardwareVerified, adopted_scope.clone());
        let subject = VerifierProfileAdoptionSubjectV1::new(
            "adopt-1", "organization:test", "adoption-root-1", [0x55; 32],
            &profile, 1, 1_000, 2_000, EvidenceClass::DifferentiallyVerified,
            adopted_scope.clone(),
        ).unwrap();
        let transition = VerifierProfileAdoptionTransitionV1::bootstrap(subject).unwrap();
        let admission = VerifierProfileAdoptionAdmissionPolicyV1::new(
            "organization:test", "adoption-root-1", [0x55; 32],
            "hardware-verifier-v1", VerifierProfileAdoptionHeadV1::Uninitialized,
        ).unwrap();
        let checked = admission.check(1_500, &transition, &profile, Some(&contract)).unwrap();
        let granted = bind_policy_checked_adoption_to_authority_grant(checked, &initial_grant, Some(&contract)).unwrap();
        let root = root(4);
        let record = VerifierProfileAdoptionRegistryRecordV1::new_uncommitted_projection(&granted, &root, None).unwrap();
        let head = VerifierProfileAdoptionHeadV1::from_transition(&transition).unwrap();
        let widened = grant(4, EvidenceClass::HardwareVerified, VerifierAdoptionScopeV1::AllContinuityVerification);
        let current = check_verifier_runtime_policy(
            &record, 1_500, &root, Some(&widened), &head, &profile, Some(&contract),
        ).unwrap();
        assert_eq!(current.effective_evidence_class(), EvidenceClass::DifferentiallyVerified);
        assert_eq!(current.effective_scope(), &adopted_scope);
    }

    #[test]
    fn revoked_or_epoch_rollback_grant_fails_closed() {
        let (contract, profile, _, root, record, head) = fixture();
        assert_eq!(
            check_verifier_runtime_policy(&record, 1_500, &root, None, &head, &profile, Some(&contract)).unwrap_err(),
            VerifierProfileRuntimePolicyError::AuthorityGrantRevoked,
        );
        let replay = grant(2, EvidenceClass::HardwareVerified, VerifierAdoptionScopeV1::Contract { contract_id: contract.id() });
        assert_eq!(
            check_verifier_runtime_policy(&record, 1_500, &root, Some(&replay), &head, &profile, Some(&contract)).unwrap_err(),
            VerifierProfileRuntimePolicyError::AuthorityGrantEpochRollback { recorded: 3, observed: 2 },
        );
    }
}
