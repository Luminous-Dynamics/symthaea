// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact commit eligibility for one continuity transition transaction.
//!
//! This layer composes already-qualified local continuity, fresh distributed safety,
//! fresh local target currentness, and independently authenticated transition
//! authority. It still does not grant physical mutation authority.
//!
//! Core theorem:
//!
//! `CommitEligibleTransitionV1 != ExecutionCapability`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::commit_currentness::{
    QualifiedLocalCommitCurrentnessId, QualifiedLocalCommitCurrentnessV1,
};
use crate::distributed_qualification::{
    DistributedCurrentStateDigest, QualifiedDistributedTransitionWitnessId,
    QualifiedDistributedTransitionWitnessV1,
};
use crate::distributed_state::DistributedStateContextId;
use crate::scope::ContinuitySubjectId;
use crate::subject_witness::{
    SubjectBoundQualifiedContinuityWitnessId, SubjectBoundQualifiedContinuityWitnessV1,
    SubjectWitnessBindingError,
};
use crate::transition_authority::{
    AuthenticatedTransitionAuthorityId, AuthenticatedTransitionAuthorityV1,
    TransitionAuthorityClaimId, TransitionAuthorityPolicyId, TransitionAuthorityProfileId,
};
use crate::verifier::VerifierProfileId;
use crate::witness::TargetRealizationId;

const COMMIT_ELIGIBILITY_DOMAIN: &[u8] = b"symthaea.continuity.commit-eligibility.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CommitEligibleTransitionId([u8; 32]);

impl CommitEligibleTransitionId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Non-Serde proof that the exact transition is eligible to enter the later
/// execution-capability boundary at one exact commit transaction time.
///
/// This value is intentionally not serializable and is not itself a mutation token.
#[derive(Debug, Clone)]
pub struct CommitEligibleTransitionV1 {
    eligibility_id: CommitEligibleTransitionId,
    subject_witness_id: SubjectBoundQualifiedContinuityWitnessId,
    subject_id: ContinuitySubjectId,
    target_realization_id: TargetRealizationId,
    distributed_witness_id: QualifiedDistributedTransitionWitnessId,
    distributed_context_id: DistributedStateContextId,
    distributed_current_state_digest: DistributedCurrentStateDigest,
    local_currentness_id: QualifiedLocalCommitCurrentnessId,
    local_currentness_verifier_profile_id: VerifierProfileId,
    local_currentness_verifier_root_epoch: u64,
    authority_id: AuthenticatedTransitionAuthorityId,
    authority_claim_id: TransitionAuthorityClaimId,
    authority_policy_id: TransitionAuthorityPolicyId,
    authority_policy_generation: u64,
    authority_profile_id: TransitionAuthorityProfileId,
    authority_root_epoch: u64,
    authority_basis_digest: [u8; 32],
    commit_time_unix_ms: u64,
}

impl CommitEligibleTransitionV1 {
    pub fn id(&self) -> CommitEligibleTransitionId {
        self.eligibility_id
    }

    pub fn subject_witness_id(&self) -> SubjectBoundQualifiedContinuityWitnessId {
        self.subject_witness_id
    }

    pub fn subject_id(&self) -> ContinuitySubjectId {
        self.subject_id
    }

    pub fn target_realization_id(&self) -> TargetRealizationId {
        self.target_realization_id
    }

    pub fn distributed_witness_id(&self) -> QualifiedDistributedTransitionWitnessId {
        self.distributed_witness_id
    }

    pub fn distributed_context_id(&self) -> DistributedStateContextId {
        self.distributed_context_id
    }

    pub fn distributed_current_state_digest(&self) -> DistributedCurrentStateDigest {
        self.distributed_current_state_digest
    }

    pub fn local_currentness_id(&self) -> QualifiedLocalCommitCurrentnessId {
        self.local_currentness_id
    }

    pub fn local_currentness_verifier_profile_id(&self) -> VerifierProfileId {
        self.local_currentness_verifier_profile_id
    }

    pub fn local_currentness_verifier_root_epoch(&self) -> u64 {
        self.local_currentness_verifier_root_epoch
    }

    pub fn authority_id(&self) -> AuthenticatedTransitionAuthorityId {
        self.authority_id
    }

    pub fn authority_claim_id(&self) -> TransitionAuthorityClaimId {
        self.authority_claim_id
    }

    pub fn authority_policy_id(&self) -> TransitionAuthorityPolicyId {
        self.authority_policy_id
    }

    pub fn authority_policy_generation(&self) -> u64 {
        self.authority_policy_generation
    }

    pub fn authority_profile_id(&self) -> TransitionAuthorityProfileId {
        self.authority_profile_id
    }

    pub fn authority_root_epoch(&self) -> u64 {
        self.authority_root_epoch
    }

    pub fn authority_basis_digest(&self) -> [u8; 32] {
        self.authority_basis_digest
    }

    pub fn commit_time_unix_ms(&self) -> u64 {
        self.commit_time_unix_ms
    }
}

/// Crate-owned commit gate.
///
/// The exact same transaction timestamp must have been used to evaluate the
/// distributed world and to qualify local target currentness. There is no
/// "recent enough" grace interval in V1: if the transaction moves forward in time,
/// the caller must refresh both currentness proofs before retrying.
pub(crate) fn compose_commit_eligible_transition(
    local_witness: &SubjectBoundQualifiedContinuityWitnessV1,
    distributed_witness: &QualifiedDistributedTransitionWitnessV1,
    local_currentness: &QualifiedLocalCommitCurrentnessV1,
    authority: &AuthenticatedTransitionAuthorityV1,
    commit_time_unix_ms: u64,
) -> Result<CommitEligibleTransitionV1, CommitEligibilityError> {
    if commit_time_unix_ms == 0 {
        return Err(CommitEligibilityError::ZeroCommitTime);
    }
    local_witness.validate()?;
    require_local_candidate(local_witness, distributed_witness)?;
    require_exact_commit_time(
        distributed_witness.evaluated_at_unix_ms(),
        local_currentness.qualified_at_unix_ms(),
        commit_time_unix_ms,
    )?;

    if local_currentness.subject_witness_id() != local_witness.id()
        || local_currentness.subject_id() != local_witness.subject_id()
        || local_currentness.target_realization_id() != local_witness.target_realization_id()
    {
        return Err(CommitEligibilityError::LocalCurrentnessWitnessMismatch);
    }
    if local_currentness.distributed_witness_id() != distributed_witness.id()
        || local_currentness.distributed_context_id() != distributed_witness.context_id()
        || local_currentness.distributed_current_state_digest()
            != distributed_witness.current_state_digest()
    {
        return Err(CommitEligibilityError::LocalCurrentnessDistributedMismatch);
    }

    if authority.subject_witness_id() != local_witness.id()
        || authority.subject_id() != local_witness.subject_id()
        || authority.target_realization_id() != local_witness.target_realization_id()
    {
        return Err(CommitEligibilityError::AuthorityLocalIntentMismatch);
    }
    if authority.distributed_context_id() != distributed_witness.context_id() {
        return Err(CommitEligibilityError::AuthorityDistributedContextMismatch);
    }
    if commit_time_unix_ms < authority.valid_from_unix_ms()
        || commit_time_unix_ms > authority.valid_until_unix_ms()
    {
        return Err(CommitEligibilityError::AuthorityNotValidAtCommit {
            commit_time_unix_ms,
            valid_from_unix_ms: authority.valid_from_unix_ms(),
            valid_until_unix_ms: authority.valid_until_unix_ms(),
        });
    }

    let eligibility_id = CommitEligibleTransitionId(hash_eligibility(
        local_witness,
        distributed_witness,
        local_currentness,
        authority,
        commit_time_unix_ms,
    ));

    Ok(CommitEligibleTransitionV1 {
        eligibility_id,
        subject_witness_id: local_witness.id(),
        subject_id: local_witness.subject_id(),
        target_realization_id: local_witness.target_realization_id(),
        distributed_witness_id: distributed_witness.id(),
        distributed_context_id: distributed_witness.context_id(),
        distributed_current_state_digest: distributed_witness.current_state_digest(),
        local_currentness_id: local_currentness.id(),
        local_currentness_verifier_profile_id: local_currentness.verifier_profile_id(),
        local_currentness_verifier_root_epoch: local_currentness.verifier_root_epoch(),
        authority_id: authority.id(),
        authority_claim_id: authority.claim_id(),
        authority_policy_id: authority.policy_id(),
        authority_policy_generation: authority.policy_generation(),
        authority_profile_id: authority.profile_id(),
        authority_root_epoch: authority.root_epoch(),
        authority_basis_digest: authority.authority_basis_digest(),
        commit_time_unix_ms,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CommitEligibilityError {
    #[error(transparent)]
    SubjectWitness(#[from] SubjectWitnessBindingError),
    #[error("commit transaction time must be non-zero")]
    ZeroCommitTime,
    #[error("local continuity subject is not in the exact distributed candidate set")]
    LocalSubjectNotCandidate,
    #[error(
        "commit currentness must be evaluated at one exact transaction time: distributed={distributed_time_unix_ms}, local={local_time_unix_ms}, commit={commit_time_unix_ms}"
    )]
    CommitTimeMismatch {
        distributed_time_unix_ms: u64,
        local_time_unix_ms: u64,
        commit_time_unix_ms: u64,
    },
    #[error("qualified local currentness does not bind the exact local subject witness/target")]
    LocalCurrentnessWitnessMismatch,
    #[error("qualified local currentness does not bind the exact distributed witness/current state")]
    LocalCurrentnessDistributedMismatch,
    #[error("authenticated transition authority does not bind the exact local subject/target intent")]
    AuthorityLocalIntentMismatch,
    #[error("authenticated transition authority belongs to a different distributed transaction context")]
    AuthorityDistributedContextMismatch,
    #[error(
        "authenticated transition authority is not valid at commit time {commit_time_unix_ms}; valid interval is {valid_from_unix_ms}..={valid_until_unix_ms}"
    )]
    AuthorityNotValidAtCommit {
        commit_time_unix_ms: u64,
        valid_from_unix_ms: u64,
        valid_until_unix_ms: u64,
    },
}

fn require_local_candidate(
    local_witness: &SubjectBoundQualifiedContinuityWitnessV1,
    distributed_witness: &QualifiedDistributedTransitionWitnessV1,
) -> Result<(), CommitEligibilityError> {
    if distributed_witness
        .candidate_subject_ids()
        .binary_search(&local_witness.subject_id())
        .is_err()
    {
        return Err(CommitEligibilityError::LocalSubjectNotCandidate);
    }
    Ok(())
}

fn require_exact_commit_time(
    distributed_time_unix_ms: u64,
    local_time_unix_ms: u64,
    commit_time_unix_ms: u64,
) -> Result<(), CommitEligibilityError> {
    if distributed_time_unix_ms != commit_time_unix_ms
        || local_time_unix_ms != commit_time_unix_ms
    {
        return Err(CommitEligibilityError::CommitTimeMismatch {
            distributed_time_unix_ms,
            local_time_unix_ms,
            commit_time_unix_ms,
        });
    }
    Ok(())
}

fn hash_eligibility(
    local_witness: &SubjectBoundQualifiedContinuityWitnessV1,
    distributed_witness: &QualifiedDistributedTransitionWitnessV1,
    local_currentness: &QualifiedLocalCommitCurrentnessV1,
    authority: &AuthenticatedTransitionAuthorityV1,
    commit_time_unix_ms: u64,
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(COMMIT_ELIGIBILITY_DOMAIN);
    hasher.update(local_witness.id().as_bytes());
    hasher.update(local_witness.subject_id().as_bytes());
    hasher.update(local_witness.target_realization_id().as_bytes());
    hasher.update(distributed_witness.id().as_bytes());
    hasher.update(distributed_witness.context_id().as_bytes());
    hasher.update(distributed_witness.current_state_digest().as_bytes());
    hasher.update(local_currentness.id().as_bytes());
    hasher.update(local_currentness.verifier_profile_id().as_bytes());
    hasher.update(&local_currentness.verifier_root_epoch().to_le_bytes());
    hasher.update(authority.id().as_bytes());
    hasher.update(authority.claim_id().as_bytes());
    hasher.update(authority.policy_id().as_bytes());
    hasher.update(&authority.policy_generation().to_le_bytes());
    hasher.update(authority.profile_id().as_bytes());
    hasher.update(&authority.root_epoch().to_le_bytes());
    hasher.update(&authority.authority_basis_digest());
    hasher.update(&commit_time_unix_ms.to_le_bytes());
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_commit_time_has_no_grace_window() {
        require_exact_commit_time(10_000, 10_000, 10_000).unwrap();
        assert_eq!(
            require_exact_commit_time(9_999, 10_000, 10_000).unwrap_err(),
            CommitEligibilityError::CommitTimeMismatch {
                distributed_time_unix_ms: 9_999,
                local_time_unix_ms: 10_000,
                commit_time_unix_ms: 10_000,
            }
        );
        assert_eq!(
            require_exact_commit_time(10_000, 9_999, 10_000).unwrap_err(),
            CommitEligibilityError::CommitTimeMismatch {
                distributed_time_unix_ms: 10_000,
                local_time_unix_ms: 9_999,
                commit_time_unix_ms: 10_000,
            }
        );
    }

    #[test]
    fn authority_window_is_inclusive_only_at_exact_bounds() {
        let valid_from = 10_000;
        let valid_until = 20_000;
        assert!(valid_from <= 10_000 && 10_000 <= valid_until);
        assert!(valid_from <= 20_000 && 20_000 <= valid_until);
        assert!(!(valid_from <= 20_001 && 20_001 <= valid_until));
    }
}
