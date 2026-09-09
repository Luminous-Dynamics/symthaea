// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Current-use policy envelope for one verifier-profile adoption.
//!
//! Historical commit state is not current authority. Runtime use must re-evaluate
//! the exact signed adoption against fresh local root/head/profile/grant/time state.
//!
//! Core theorem:
//!
//! `current runtime envelope = signed adoption ∩ current local grant`,
//!
//! additionally gated by the exact #1112 root provisioning snapshot, exact current
//! adoption head/profile, monotone grant slot, and the same #1149 clock lineage and
//! uncertainty policy captured by #1198.
//!
//! This module remains non-cryptographic. A successful value proves policy
//! currentness only; it is not evidence that the historical adoption was signed or
//! durably committed.

use thiserror::Error;

use crate::contract::{ContinuityRequirementId, ValidatedContinuityContractV1};
use crate::profile_adoption::{
    VerifierAdoptionScopeV1, VerifierProfileAdoptionError,
    VerifierProfileAdoptionTransitionDigest, VerifierProfileAdoptionTransitionV1,
};
use crate::profile_adoption_admission::{
    VerifierProfileAdoptionAdmissionError, VerifierProfileAdoptionHeadV1,
};
use crate::profile_adoption_grant::{
    VerifierAdoptionAuthorityGrantIdV1, VerifierAdoptionAuthorityGrantV1,
};
use crate::profile_adoption_grant_commit::GrantBoundVerifierProfileAdoptionCommitPreconditionsV1;
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

/// Historical non-Serde policy baseline derived from the canonical #1198 commit
/// preconditions.
///
/// This is deliberately not proof that a registry write happened. Its purpose is to
/// retain the exact local policy lineage that a later cryptographically authenticated
/// committed-adoption receipt must corroborate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifierProfileRuntimeAuthorityBaselineV1 {
    transition_digest: VerifierProfileAdoptionTransitionDigest,
    root_snapshot_id: VerifierProfileAdoptionAuthorityRootSnapshotId,
    historical_grant: VerifierAdoptionAuthorityGrantV1,
    expected_clock_source_id: String,
    expected_clock_epoch: u64,
    max_clock_uncertainty_ms: u64,
}

impl VerifierProfileRuntimeAuthorityBaselineV1 {
    pub fn from_commit_preconditions(
        preconditions: &GrantBoundVerifierProfileAdoptionCommitPreconditionsV1,
    ) -> Self {
        let grant_bound = preconditions.grant_bound();
        let time_bound = preconditions.time_bound();
        Self {
            transition_digest: grant_bound.root_bound().transition_digest(),
            root_snapshot_id: grant_bound.root_bound().authority_root_snapshot().id(),
            historical_grant: grant_bound.authority_grant().clone(),
            expected_clock_source_id: time_bound.expected_clock_source_id().to_owned(),
            expected_clock_epoch: time_bound.expected_clock_epoch(),
            max_clock_uncertainty_ms: time_bound.max_uncertainty_ms(),
        }
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

    pub fn expected_clock_source_id(&self) -> &str {
        &self.expected_clock_source_id
    }

    pub fn expected_clock_epoch(&self) -> u64 {
        self.expected_clock_epoch
    }

    pub fn max_clock_uncertainty_ms(&self) -> u64 {
        self.max_clock_uncertainty_ms
    }
}

/// Non-Serde current-use policy envelope.
///
/// It is intentionally weaker than a future `CurrentlyAuthorizedVerifierProfile`:
/// no cryptographic proof or committed-adoption receipt is present here.
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
        requirement_id: ContinuityRequirementId,
    ) -> bool {
        match &self.effective_scope {
            VerifierAdoptionScopeV1::AllContinuityVerification => true,
            VerifierAdoptionScopeV1::Contract {
                contract_id: allowed,
            } => *allowed == contract_id,
            VerifierAdoptionScopeV1::Requirements {
                contract_id: allowed,
                requirement_ids,
            } => {
                *allowed == contract_id && requirement_ids.binary_search(&requirement_id).is_ok()
            }
        }
    }
}

/// Re-evaluate one exact adoption transition for current runtime policy use.
///
/// `current_grant == None` is explicit local revocation. A newer grant may narrow
/// authority immediately. A wider grant never widens beyond the signed adoption.
/// Grant rollback, same-epoch equivocation, root reprovisioning, head supersession,
/// clock-lineage changes, and boundary-straddling time uncertainty all fail closed.
#[allow(clippy::too_many_arguments)]
pub fn check_current_verifier_runtime_policy(
    baseline: &VerifierProfileRuntimeAuthorityBaselineV1,
    transition: &VerifierProfileAdoptionTransitionV1,
    current_head: &VerifierProfileAdoptionHeadV1,
    current_root: &VerifierProfileAdoptionAuthorityRootSnapshotV1,
    current_grant: Option<&VerifierAdoptionAuthorityGrantV1>,
    current_profile: &VerifierProfileV1,
    current_clock: &VerifierProfileAdoptionClockObservationV1,
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
    if grant.authority_root_snapshot() != current_root {
        return Err(VerifierProfileRuntimeAuthorityError::AuthorityGrantRootSnapshotChanged);
    }
    if grant.verifier_role_id() != subject.verifier_role_id() {
        return Err(VerifierProfileRuntimeAuthorityError::AuthorityGrantRoleChanged);
    }

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
    if current_clock.source_id() != baseline.expected_clock_source_id
        || current_clock.source_epoch() != baseline.expected_clock_epoch
    {
        return Err(VerifierProfileRuntimeAuthorityError::ClockLineageChanged {
            expected_source_id: baseline.expected_clock_source_id.clone(),
            expected_epoch: baseline.expected_clock_epoch,
            observed_source_id: current_clock.source_id().to_owned(),
            observed_epoch: current_clock.source_epoch(),
        });
    }

    let uncertainty_ms = current_clock.uncertainty_ms();
    if uncertainty_ms > baseline.max_clock_uncertainty_ms {
        return Err(VerifierProfileRuntimeAuthorityError::ClockUncertaintyExceedsPolicy {
            observed_ms: uncertainty_ms,
            maximum_ms: baseline.max_clock_uncertainty_ms,
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

    ground_scope(subject.scope(), scope_contract, ScopeOwner::Adoption)?;
    ground_scope(grant.allowed_scope(), scope_contract, ScopeOwner::Grant)?;
    let effective_scope = intersect_scopes(subject.scope(), grant.allowed_scope())
        .ok_or(VerifierProfileRuntimeAuthorityError::NoEffectiveScope)?;
    ground_scope(&effective_scope, scope_contract, ScopeOwner::Effective)?;

    let effective_evidence_class = weaker_class(
        weaker_class(
            current_profile.evidence_class(),
            subject.evidence_class_ceiling(),
        ),
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

fn intersect_scopes(
    left: &VerifierAdoptionScopeV1,
    right: &VerifierAdoptionScopeV1,
) -> Option<VerifierAdoptionScopeV1> {
    match (left, right) {
        (VerifierAdoptionScopeV1::AllContinuityVerification, other)
        | (other, VerifierAdoptionScopeV1::AllContinuityVerification) => Some(other.clone()),
        (
            VerifierAdoptionScopeV1::Contract { contract_id: a },
            VerifierAdoptionScopeV1::Contract { contract_id: b },
        ) if a == b => Some(VerifierAdoptionScopeV1::Contract { contract_id: *a }),
        (
            VerifierAdoptionScopeV1::Contract { contract_id: a },
            VerifierAdoptionScopeV1::Requirements {
                contract_id: b,
                requirement_ids,
            },
        )
        | (
            VerifierAdoptionScopeV1::Requirements {
                contract_id: b,
                requirement_ids,
            },
            VerifierAdoptionScopeV1::Contract { contract_id: a },
        ) if a == b => VerifierAdoptionScopeV1::requirements(*a, requirement_ids.clone()).ok(),
        (
            VerifierAdoptionScopeV1::Requirements {
                contract_id: a,
                requirement_ids: left,
            },
            VerifierAdoptionScopeV1::Requirements {
                contract_id: b,
                requirement_ids: right,
            },
        ) if a == b => {
            let overlap: Vec<_> = left
                .iter()
                .copied()
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

#[derive(Debug, Clone, Copy)]
enum ScopeOwner {
    Adoption,
    Grant,
    Effective,
}

fn ground_scope(
    scope: &VerifierAdoptionScopeV1,
    contract: Option<&ValidatedContinuityContractV1>,
    owner: ScopeOwner,
) -> Result<(), VerifierProfileRuntimeAuthorityError> {
    match scope {
        VerifierAdoptionScopeV1::AllContinuityVerification => Ok(()),
        VerifierAdoptionScopeV1::Contract { contract_id } => {
            let contract = contract.ok_or(VerifierProfileRuntimeAuthorityError::MissingScopeContract)?;
            if contract.id() != *contract_id {
                return Err(VerifierProfileRuntimeAuthorityError::ScopeContractMismatch);
            }
            Ok(())
        }
        VerifierAdoptionScopeV1::Requirements {
            contract_id,
            requirement_ids,
        } => {
            let contract = contract.ok_or(VerifierProfileRuntimeAuthorityError::MissingScopeContract)?;
            if contract.id() != *contract_id {
                return Err(VerifierProfileRuntimeAuthorityError::ScopeContractMismatch);
            }
            for requirement in requirement_ids {
                if !contract
                    .requirements()
                    .iter()
                    .any(|item| item.id() == *requirement)
                {
                    return Err(VerifierProfileRuntimeAuthorityError::UnknownScopeRequirement {
                        owner,
                        requirement: *requirement,
                    });
                }
            }
            Ok(())
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum VerifierProfileRuntimeAuthorityError {
    #[error(transparent)]
    Adoption(#[from] VerifierProfileAdoptionError),
    #[error(transparent)]
    Admission(#[from] VerifierProfileAdoptionAdmissionError),
    #[error(transparent)]
    Clock(#[from] VerifierProfileAdoptionTimeError),
    #[error("runtime transition differs from the historical adoption baseline")]
    HistoricalTransitionMismatch,
    #[error("current verifier-adoption head no longer equals this exact transition")]
    AdoptionNoLongerCurrentHead,
    #[error("adoption-authority root provisioning snapshot changed since admission")]
    AuthorityRootSnapshotChanged,
    #[error("current adoption-authority root no longer matches this transition")]
    AuthorityRootNoLongerCurrent,
    #[error("current local verifier-adoption authority grant is revoked")]
    AuthorityGrantRevoked,
    #[error("current authority grant is bound to a different root provisioning snapshot")]
    AuthorityGrantRootSnapshotChanged,
    #[error("current authority grant belongs to a different verifier role")]
    AuthorityGrantRoleChanged,
    #[error("current authority grant belongs to a different local grant slot")]
    AuthorityGrantSlotChanged,
    #[error("authority grant epoch rolled back: recorded {recorded}, observed {observed}")]
    AuthorityGrantEpochRollback { recorded: u64, observed: u64 },
    #[error("authority grant semantics changed without advancing grant epoch {epoch}")]
    SameEpochGrantEquivocation { epoch: u64 },
    #[error("verifier runtime clock lineage changed: expected {expected_source_id}@{expected_epoch}, observed {observed_source_id}@{observed_epoch}")]
    ClockLineageChanged {
        expected_source_id: String,
        expected_epoch: u64,
        observed_source_id: String,
        observed_epoch: u64,
    },
    #[error("clock uncertainty {observed_ms}ms exceeds runtime policy maximum {maximum_ms}ms")]
    ClockUncertaintyExceedsPolicy { observed_ms: u64, maximum_ms: u64 },
    #[error("current clock interval [{earliest_unix_ms}, {latest_unix_ms}] lies outside adoption validity [{valid_from_unix_ms}, {valid_until_unix_ms})")]
    ClockIntervalOutsideAdoptionValidity {
        earliest_unix_ms: u64,
        latest_unix_ms: u64,
        valid_from_unix_ms: u64,
        valid_until_unix_ms: u64,
    },
    #[error("scoped verifier authority requires the exact validated contract")]
    MissingScopeContract,
    #[error("runtime verifier scope references a different continuity contract")]
    ScopeContractMismatch,
    #[error("{owner:?} scope references requirement outside the exact contract: {requirement:?}")]
    UnknownScopeRequirement {
        owner: ScopeOwner,
        requirement: ContinuityRequirementId,
    },
    #[error("signed adoption and current local grant have no effective verifier scope")]
    NoEffectiveScope,
}
