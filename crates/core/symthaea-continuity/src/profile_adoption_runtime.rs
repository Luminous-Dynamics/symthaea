// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Current-use policy envelope for one verifier-profile adoption.
//!
//! Historical persistence is not current authority. Runtime use must re-evaluate the
//! exact signed adoption against fresh local root/head/profile/grant/time state while
//! retaining enough historical policy identity to detect rollback and reprovisioning.
//!
//! Core theorem:
//!
//! `current runtime envelope = signed adoption ∩ current local grant`,
//!
//! additionally gated by the exact historical root-provisioning snapshot, exact
//! current adoption head/profile, monotone grant slot, and bounded current clock.
//!
//! This module remains non-cryptographic. A successful value is policy currentness
//! only; it is not proof that the transition was signed or atomically committed.

use thiserror::Error;

use crate::contract::ValidatedContinuityContractV1;
use crate::profile_adoption::{
    VerifierAdoptionScopeV1, VerifierProfileAdoptionError,
    VerifierProfileAdoptionTransitionDigest, VerifierProfileAdoptionTransitionV1,
};
use crate::profile_adoption_admission::{
    VerifierProfileAdoptionAdmissionError, VerifierProfileAdoptionHeadV1,
};
use crate::profile_adoption_authority::{
    ground_scope, intersect_scopes, require_same_authority,
    AuthorityGrantedVerifierProfileAdoptionV1, ScopeOwner,
    VerifierAdoptionAuthorityGrantError, VerifierAdoptionAuthorityGrantIdV1,
    VerifierAdoptionAuthorityGrantV1,
};
use crate::profile_adoption_root::{
    VerifierProfileAdoptionAuthorityRootSnapshotId,
    VerifierProfileAdoptionAuthorityRootSnapshotV1,
};
use crate::profile_adoption_time::{
    VerifierProfileAdoptionClockObservationId, VerifierProfileAdoptionClockObservationV1,
    VerifierProfileAdoptionTimeError,
};
use crate::verifier::VerifierProfileV1;
use crate::witness::EvidenceClass;

/// Historical non-Serde policy baseline for one adoption transaction.
///
/// This value still is not proof of a committed write. A future committed-adoption
/// receipt should be the production source of equivalent baseline facts after a
/// restart. The purpose here is to make the required continuity explicit now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifierProfileRuntimeAuthorityBaselineV1 {
    transition_digest: VerifierProfileAdoptionTransitionDigest,
    root_snapshot_id: VerifierProfileAdoptionAuthorityRootSnapshotId,
    historical_grant: VerifierAdoptionAuthorityGrantV1,
}

impl VerifierProfileRuntimeAuthorityBaselineV1 {
    pub fn from_granted_adoption(
        granted: &AuthorityGrantedVerifierProfileAdoptionV1,
        historical_root: &VerifierProfileAdoptionAuthorityRootSnapshotV1,
    ) -> Result<Self, VerifierProfileRuntimeAuthorityError> {
        let subject = granted.checked().transition().subject();
        if subject.authority_subject() != historical_root.authority_subject()
            || subject.authority_root_id() != historical_root.authority_root_id()
            || subject.authority_root_digest() != historical_root.authority_root_digest()
        {
            return Err(VerifierProfileRuntimeAuthorityError::HistoricalRootMismatch);
        }
        let transition_digest = granted.checked().transition().transition_digest()?;
        Ok(Self {
            transition_digest,
            root_snapshot_id: historical_root.id(),
            historical_grant: granted.authority_grant().clone(),
        })
    }

    pub fn transition_digest(&self) -> VerifierProfileAdoptionTransitionDigest {
        self.transition_digest
    }
    pub fn root_snapshot_id(&self) -> VerifierProfileAdoptionAuthorityRootSnapshotId {
        self.root_snapshot_id
    }
    pub fn historical_grant(&self) -> &VerifierAdoptionAuthorityGrantV1 {
        &self.historical_grant
    }
}

/// Non-Serde policy-current verifier envelope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyCurrentVerifierRuntimeEnvelopeV1 {
    transition_digest: VerifierProfileAdoptionTransitionDigest,
    profile: VerifierProfileV1,
    effective_evidence_class: EvidenceClass,
    effective_scope: VerifierAdoptionScopeV1,
    current_grant_id: VerifierAdoptionAuthorityGrantIdV1,
    current_grant_epoch: u64,
    current_root_snapshot_id: VerifierProfileAdoptionAuthorityRootSnapshotId,
    current_head: VerifierProfileAdoptionHeadV1,
    clock_observation_id: VerifierProfileAdoptionClockObservationId,
}

impl PolicyCurrentVerifierRuntimeEnvelopeV1 {
    pub fn transition_digest(&self) -> VerifierProfileAdoptionTransitionDigest {
        self.transition_digest
    }
    pub fn profile(&self) -> &VerifierProfileV1 {
        &self.profile
    }
    pub fn effective_evidence_class(&self) -> EvidenceClass {
        self.effective_evidence_class
    }
    pub fn effective_scope(&self) -> &VerifierAdoptionScopeV1 {
        &self.effective_scope
    }
    pub fn current_grant_id(&self) -> VerifierAdoptionAuthorityGrantIdV1 {
        self.current_grant_id
    }
    pub fn current_grant_epoch(&self) -> u64 {
        self.current_grant_epoch
    }
    pub fn current_root_snapshot_id(&self) -> VerifierProfileAdoptionAuthorityRootSnapshotId {
        self.current_root_snapshot_id
    }
    pub fn current_head(&self) -> &VerifierProfileAdoptionHeadV1 {
        &self.current_head
    }
    pub fn clock_observation_id(&self) -> VerifierProfileAdoptionClockObservationId {
        self.clock_observation_id
    }

    pub(crate) fn allows_claim(
        &self,
        contract_id: crate::contract::ContinuityContractId,
        requirement_id: crate::contract::ContinuityRequirementId,
    ) -> bool {
        match &self.effective_scope {
            VerifierAdoptionScopeV1::AllContinuityVerification => true,
            VerifierAdoptionScopeV1::Contract { contract_id: allowed } => *allowed == contract_id,
            VerifierAdoptionScopeV1::Requirements { contract_id: allowed, requirement_ids } => {
                *allowed == contract_id && requirement_ids.binary_search(&requirement_id).is_ok()
            }
        }
    }
}

/// Re-evaluate one exact adoption transition for current runtime policy use.
///
/// `current_grant == None` is explicit local revocation. A current grant may narrow
/// authority immediately. A wider current grant cannot widen the signed adoption.
/// Grant epoch rollback and same-epoch semantic equivocation fail closed.
#[allow(clippy::too_many_arguments)]
pub fn check_current_verifier_runtime_policy(
    baseline: &VerifierProfileRuntimeAuthorityBaselineV1,
    transition: &VerifierProfileAdoptionTransitionV1,
    current_head: &VerifierProfileAdoptionHeadV1,
    current_root: &VerifierProfileAdoptionAuthorityRootSnapshotV1,
    current_grant: Option<&VerifierAdoptionAuthorityGrantV1>,
    current_profile: &VerifierProfileV1,
    current_clock: &VerifierProfileAdoptionClockObservationV1,
    max_clock_uncertainty_ms: u64,
    scope_contract: Option<&ValidatedContinuityContractV1>,
) -> Result<PolicyCurrentVerifierRuntimeEnvelopeV1, VerifierProfileRuntimeAuthorityError> {
    let subject = transition.subject();
    subject.validate_against_profile(current_profile)?;
    let transition_digest = transition.transition_digest()?;
    if transition_digest != baseline.transition_digest {
        return Err(VerifierProfileRuntimeAuthorityError::HistoricalTransitionMismatch);
    }

    let expected_head = VerifierProfileAdoptionHeadV1::from_transition(transition)?;
    if &expected_head != current_head {
        return Err(VerifierProfileRuntimeAuthorityError::AdoptionNoLongerCurrentHead);
    }

    if current_root.id() != baseline.root_snapshot_id {
        return Err(VerifierProfileRuntimeAuthorityError::AuthorityRootSnapshotChanged);
    }
    if subject.authority_subject() != current_root.authority_subject()
        || subject.authority_root_id() != current_root.authority_root_id()
        || subject.authority_root_digest() != current_root.authority_root_digest()
    {
        return Err(VerifierProfileRuntimeAuthorityError::AuthorityRootNoLongerCurrent);
    }

    let grant = current_grant.ok_or(VerifierProfileRuntimeAuthorityError::AuthorityGrantRevoked)?;
    grant.validate()?;
    require_same_authority(
        subject.authority_subject(),
        subject.authority_root_id(),
        subject.authority_root_digest(),
        subject.verifier_role_id(),
        grant,
    )?;
    let historical = &baseline.historical_grant;
    if grant.grant_slot_id() != historical.grant_slot_id() {
        return Err(VerifierProfileRuntimeAuthorityError::AuthorityGrantSlotChanged);
    }
    if grant.grant_epoch() < historical.grant_epoch() {
        return Err(VerifierProfileRuntimeAuthorityError::AuthorityGrantEpochRollback {
            recorded: historical.grant_epoch(),
            observed: grant.grant_epoch(),
        });
    }
    if grant.grant_epoch() == historical.grant_epoch() && grant.id() != historical.id() {
        return Err(VerifierProfileRuntimeAuthorityError::SameEpochGrantEquivocation {
            epoch: grant.grant_epoch(),
        });
    }

    current_clock.validate()?;
    let uncertainty_ms = current_clock.uncertainty_ms();
    if uncertainty_ms > max_clock_uncertainty_ms {
        return Err(VerifierProfileRuntimeAuthorityError::ClockUncertaintyExceedsPolicy {
            observed_ms: uncertainty_ms,
            maximum_ms: max_clock_uncertainty_ms,
        });
    }
    if current_clock.earliest_unix_ms() < subject.valid_from_unix_ms()
        || current_clock.latest_unix_ms() >= subject.valid_until_unix_ms()
    {
        return Err(VerifierProfileRuntimeAuthorityError::ClockIntervalOutsideAdoptionValidity {
            earliest_unix_ms: current_clock.earliest_unix_ms(),
            latest_unix_ms: current_clock.latest_unix_ms(),
            valid_from_unix_ms: subject.valid_from_unix_ms(),
            valid_until_unix_ms: subject.valid_until_unix_ms(),
        });
    }

    ground_scope(subject.scope(), scope_contract, ScopeOwner::Candidate)?;
    ground_scope(grant.allowed_scope(), scope_contract, ScopeOwner::Grant)?;
    let effective_scope = intersect_scopes(subject.scope(), grant.allowed_scope())
        .ok_or(VerifierProfileRuntimeAuthorityError::NoEffectiveScope)?;
    ground_scope(&effective_scope, scope_contract, ScopeOwner::Candidate)?;

    let effective_evidence_class = weaker_class(
        weaker_class(current_profile.evidence_class(), subject.evidence_class_ceiling()),
        grant.maximum_evidence_class(),
    );

    Ok(PolicyCurrentVerifierRuntimeEnvelopeV1 {
        transition_digest,
        profile: current_profile.clone(),
        effective_evidence_class,
        effective_scope,
        current_grant_id: grant.id(),
        current_grant_epoch: grant.grant_epoch(),
        current_root_snapshot_id: current_root.id(),
        current_head: current_head.clone(),
        clock_observation_id: current_clock.id(),
    })
}

fn weaker_class(a: EvidenceClass, b: EvidenceClass) -> EvidenceClass {
    if a <= b { a } else { b }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum VerifierProfileRuntimeAuthorityError {
    #[error(transparent)]
    Adoption(#[from] VerifierProfileAdoptionError),
    #[error(transparent)]
    Admission(#[from] VerifierProfileAdoptionAdmissionError),
    #[error(transparent)]
    Grant(#[from] VerifierAdoptionAuthorityGrantError),
    #[error(transparent)]
    Clock(#[from] VerifierProfileAdoptionTimeError),
    #[error("historical runtime baseline belongs to a different adoption-authority root")]
    HistoricalRootMismatch,
    #[error("runtime transition differs from the historical adoption baseline")]
    HistoricalTransitionMismatch,
    #[error("current verifier-adoption head no longer equals this exact transition")]
    AdoptionNoLongerCurrentHead,
    #[error("adoption-authority root provisioning snapshot changed since adoption")]
    AuthorityRootSnapshotChanged,
    #[error("current adoption-authority root no longer matches this transition")]
    AuthorityRootNoLongerCurrent,
    #[error("current local verifier-adoption authority grant is revoked")]
    AuthorityGrantRevoked,
    #[error("current authority grant belongs to a different local grant slot")]
    AuthorityGrantSlotChanged,
    #[error("authority grant epoch rolled back: recorded {recorded}, observed {observed}")]
    AuthorityGrantEpochRollback { recorded: u64, observed: u64 },
    #[error("authority grant semantics changed without advancing grant epoch {epoch}")]
    SameEpochGrantEquivocation { epoch: u64 },
    #[error("clock uncertainty {observed_ms}ms exceeds runtime policy maximum {maximum_ms}ms")]
    ClockUncertaintyExceedsPolicy { observed_ms: u64, maximum_ms: u64 },
    #[error("current clock interval [{earliest_unix_ms}, {latest_unix_ms}] lies outside adoption validity [{valid_from_unix_ms}, {valid_until_unix_ms})")]
    ClockIntervalOutsideAdoptionValidity {
        earliest_unix_ms: u64,
        latest_unix_ms: u64,
        valid_from_unix_ms: u64,
        valid_until_unix_ms: u64,
    },
    #[error("signed adoption and current local grant have no effective verifier scope")]
    NoEffectiveScope,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile_adoption::{
        VerifierAdoptionScopeV1, VerifierProfileAdoptionSubjectV1,
        VerifierProfileAdoptionTransitionV1,
    };
    use crate::profile_adoption_admission::{
        VerifierProfileAdoptionAdmissionPolicyV1, VerifierProfileAdoptionHeadV1,
    };

    fn profile() -> VerifierProfileV1 {
        VerifierProfileV1::new(
            "hardware-verifier-v1", [9; 32], 7, EvidenceClass::HardwareVerified,
        ).unwrap()
    }

    fn make_transition(
        profile: &VerifierProfileV1,
        ceiling: EvidenceClass,
    ) -> VerifierProfileAdoptionTransitionV1 {
        VerifierProfileAdoptionTransitionV1::bootstrap(
            VerifierProfileAdoptionSubjectV1::new(
                "adopt-1", "organization:test", "adoption-root-1", [0x55; 32], profile,
                1, 1_000, 2_000, ceiling,
                VerifierAdoptionScopeV1::AllContinuityVerification,
            ).unwrap(),
        ).unwrap()
    }

    fn root(epoch: u64) -> VerifierProfileAdoptionAuthorityRootSnapshotV1 {
        VerifierProfileAdoptionAuthorityRootSnapshotV1::new(
            "organization:test", "adoption-root-1", [0x55; 32], epoch,
        ).unwrap()
    }

    fn grant(epoch: u64, class: EvidenceClass) -> VerifierAdoptionAuthorityGrantV1 {
        VerifierAdoptionAuthorityGrantV1::new(
            "grant-1", "organization:test", "adoption-root-1", [0x55; 32],
            "hardware-verifier-v1", epoch, class,
            VerifierAdoptionScopeV1::AllContinuityVerification,
        ).unwrap()
    }

    fn clock(earliest: u64, latest: u64) -> VerifierProfileAdoptionClockObservationV1 {
        VerifierProfileAdoptionClockObservationV1::new("trusted-clock", 4, earliest, latest).unwrap()
    }

    fn fixture() -> (
        VerifierProfileV1,
        VerifierProfileAdoptionTransitionV1,
        VerifierProfileRuntimeAuthorityBaselineV1,
        VerifierProfileAdoptionHeadV1,
        VerifierProfileAdoptionAuthorityRootSnapshotV1,
        VerifierAdoptionAuthorityGrantV1,
    ) {
        let profile = profile();
        let transition = make_transition(&profile, EvidenceClass::HardwareVerified);
        let checked = VerifierProfileAdoptionAdmissionPolicyV1::new(
            "organization:test", "adoption-root-1", [0x55; 32], "hardware-verifier-v1",
            VerifierProfileAdoptionHeadV1::Uninitialized,
        ).unwrap().check(1_500, &transition, &profile, None).unwrap();
        let historical_grant = grant(3, EvidenceClass::HardwareVerified);
        let granted = crate::profile_adoption_authority::bind_policy_checked_adoption_to_authority_grant(
            checked, &historical_grant, None,
        ).unwrap();
        let historical_root = root(8);
        let baseline = VerifierProfileRuntimeAuthorityBaselineV1::from_granted_adoption(
            &granted, &historical_root,
        ).unwrap();
        let head = VerifierProfileAdoptionHeadV1::from_transition(&transition).unwrap();
        (profile, transition, baseline, head, historical_root, historical_grant)
    }

    #[test]
    fn newer_grant_can_narrow_but_not_widen_signed_authority() {
        let (profile, initial_transition, baseline, head, root, _) = fixture();
        let narrower = grant(4, EvidenceClass::DifferentiallyVerified);
        let current = check_current_verifier_runtime_policy(
            &baseline, &initial_transition, &head, &root, Some(&narrower), &profile,
            &clock(1_400, 1_600), 250, None,
        ).unwrap();
        assert_eq!(current.effective_evidence_class(), EvidenceClass::DifferentiallyVerified);

        let signed_narrow = make_transition(&profile, EvidenceClass::DifferentiallyVerified);
        let checked = VerifierProfileAdoptionAdmissionPolicyV1::new(
            "organization:test", "adoption-root-1", [0x55; 32], "hardware-verifier-v1",
            VerifierProfileAdoptionHeadV1::Uninitialized,
        ).unwrap().check(1_500, &signed_narrow, &profile, None).unwrap();
        let broad = grant(3, EvidenceClass::HardwareVerified);
        let granted = crate::profile_adoption_authority::bind_policy_checked_adoption_to_authority_grant(
            checked, &broad, None,
        ).unwrap();
        let baseline = VerifierProfileRuntimeAuthorityBaselineV1::from_granted_adoption(&granted, &root).unwrap();
        let head = VerifierProfileAdoptionHeadV1::from_transition(&signed_narrow).unwrap();
        let current = check_current_verifier_runtime_policy(
            &baseline, &signed_narrow, &head, &root, Some(&grant(4, EvidenceClass::HardwareVerified)),
            &profile, &clock(1_400, 1_600), 250, None,
        ).unwrap();
        assert_eq!(current.effective_evidence_class(), EvidenceClass::DifferentiallyVerified);
    }

    #[test]
    fn same_key_reprovisioning_and_grant_rollback_fail_closed() {
        let (profile, transition, baseline, head, root, historical_grant) = fixture();
        let reprovisioned = root(root.provisioning_epoch() + 1);
        assert_eq!(
            check_current_verifier_runtime_policy(
                &baseline, &transition, &head, &reprovisioned, Some(&historical_grant), &profile,
                &clock(1_400, 1_500), 250, None,
            ).unwrap_err(),
            VerifierProfileRuntimeAuthorityError::AuthorityRootSnapshotChanged,
        );

        let rollback = grant(historical_grant.grant_epoch() - 1, EvidenceClass::HardwareVerified);
        assert_eq!(
            check_current_verifier_runtime_policy(
                &baseline, &transition, &head, &root, Some(&rollback), &profile,
                &clock(1_400, 1_500), 250, None,
            ).unwrap_err(),
            VerifierProfileRuntimeAuthorityError::AuthorityGrantEpochRollback {
                recorded: historical_grant.grant_epoch(),
                observed: rollback.grant_epoch(),
            },
        );
    }

    #[test]
    fn revocation_head_change_and_time_boundary_fail_closed() {
        let (profile, transition, baseline, head, root, historical_grant) = fixture();
        assert_eq!(
            check_current_verifier_runtime_policy(
                &baseline, &transition, &head, &root, None, &profile,
                &clock(1_400, 1_500), 250, None,
            ).unwrap_err(),
            VerifierProfileRuntimeAuthorityError::AuthorityGrantRevoked,
        );
        assert!(matches!(
            check_current_verifier_runtime_policy(
                &baseline, &transition, &VerifierProfileAdoptionHeadV1::Uninitialized, &root,
                Some(&historical_grant), &profile, &clock(1_400, 1_500), 250, None,
            ),
            Err(VerifierProfileRuntimeAuthorityError::AdoptionNoLongerCurrentHead)
        ));
        assert!(matches!(
            check_current_verifier_runtime_policy(
                &baseline, &transition, &head, &root, Some(&historical_grant), &profile,
                &clock(1_900, 2_000), 250, None,
            ),
            Err(VerifierProfileRuntimeAuthorityError::ClockIntervalOutsideAdoptionValidity { .. })
        ));
    }
}
