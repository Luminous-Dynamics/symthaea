// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Store-facing commit state for verifier-profile adoption.
//!
//! [`crate::profile_adoption_commit`] captures root/head/time currentness while
//! [`crate::profile_adoption_authority`] captures the signer's bounded grant
//! capability. A future registry write must depend on both. This module composes
//! those independent facts into one exact precondition object and rechecks the
//! current authority grant before allowing a write-ready result.
//!
//! This still performs no persistence and proves no signature.

use thiserror::Error;

use crate::profile_adoption_authority::{
    AuthorityGrantedVerifierProfileAdoptionV1, VerifierAdoptionAuthorityGrantV1,
};
use crate::profile_adoption_commit::{
    VerifierAdoptionAuthorityRootSnapshotV1, VerifierProfileAdoptionCommitError,
    VerifierProfileAdoptionCommitPreconditionsV1, VerifierProfileAdoptionCommitStateV1,
};
use crate::profile_adoption_admission::VerifierProfileAdoptionHeadV1;

/// Exact combined state that a future atomic registry adapter must recheck.
///
/// It can only be constructed from an adoption already bounded by #1120's local
/// authority capability plus #1106's exact root/head/time commit preconditions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifierProfileAdoptionRegistryCommitPreconditionsV1 {
    state_preconditions: VerifierProfileAdoptionCommitPreconditionsV1,
    expected_authority_grant: VerifierAdoptionAuthorityGrantV1,
}

impl VerifierProfileAdoptionRegistryCommitPreconditionsV1 {
    /// Compose independently derived authority-grant and store-state preconditions.
    ///
    /// Cross-object substitution fails closed: both objects must refer to the same
    /// exact predecessor head, candidate head, authority root context, verifier
    /// role, and validity interval before one registry precondition object exists.
    pub fn from_authority_granted(
        granted: &AuthorityGrantedVerifierProfileAdoptionV1,
        state_preconditions: VerifierProfileAdoptionCommitPreconditionsV1,
    ) -> Result<Self, VerifierProfileAdoptionRegistryCommitError> {
        let checked = granted.checked();
        let grant = granted.authority_grant();
        let subject = checked.transition().subject();

        if state_preconditions.expected_predecessor_head()
            != checked.expected_predecessor_head()
        {
            return Err(
                VerifierProfileAdoptionRegistryCommitError::ExpectedPredecessorHeadMismatch,
            );
        }
        if state_preconditions.candidate_head() != checked.candidate_head() {
            return Err(VerifierProfileAdoptionRegistryCommitError::CandidateHeadMismatch);
        }
        if state_preconditions.expected_verifier_role_id() != grant.verifier_role_id()
            || subject.verifier_role_id() != grant.verifier_role_id()
        {
            return Err(VerifierProfileAdoptionRegistryCommitError::VerifierRoleMismatch);
        }

        let root = state_preconditions.expected_root();
        if root.authority_subject() != grant.authority_subject()
            || subject.authority_subject() != grant.authority_subject()
        {
            return Err(VerifierProfileAdoptionRegistryCommitError::AuthoritySubjectMismatch);
        }
        if root.root_id() != grant.authority_root_id()
            || subject.authority_root_id() != grant.authority_root_id()
        {
            return Err(VerifierProfileAdoptionRegistryCommitError::AuthorityRootIdMismatch);
        }
        if root.root_digest() != grant.authority_root_digest()
            || subject.authority_root_digest() != grant.authority_root_digest()
        {
            return Err(VerifierProfileAdoptionRegistryCommitError::AuthorityRootDigestMismatch);
        }

        if state_preconditions.valid_from_unix_ms() != subject.valid_from_unix_ms()
            || state_preconditions.valid_until_unix_ms() != subject.valid_until_unix_ms()
        {
            return Err(VerifierProfileAdoptionRegistryCommitError::ValidityIntervalMismatch);
        }

        Ok(Self {
            state_preconditions,
            expected_authority_grant: grant.clone(),
        })
    }

    pub fn state_preconditions(&self) -> &VerifierProfileAdoptionCommitPreconditionsV1 {
        &self.state_preconditions
    }

    pub fn expected_authority_grant(&self) -> &VerifierAdoptionAuthorityGrantV1 {
        &self.expected_authority_grant
    }

    /// Recheck one atomic registry observation immediately before write.
    ///
    /// Exact-candidate idempotent recovery is intentionally delegated first to the
    /// underlying state preconditions. If the candidate already exists, later root,
    /// grant, or time changes do not erase the historical fact that the write
    /// succeeded. For a new write, the exact authority grant must still be current.
    pub fn recheck_registry_observation(
        &self,
        now_unix_ms: u64,
        current_root: &VerifierAdoptionAuthorityRootSnapshotV1,
        current_authority_grant: &VerifierAdoptionAuthorityGrantV1,
        current_head: &VerifierProfileAdoptionHeadV1,
    ) -> Result<VerifierProfileAdoptionCommitStateV1, VerifierProfileAdoptionRegistryCommitError>
    {
        let state = self.state_preconditions.recheck_commit_observation(
            now_unix_ms,
            current_root,
            current_head,
        )?;
        if state == VerifierProfileAdoptionCommitStateV1::AlreadyCommitted {
            return Ok(state);
        }
        if current_authority_grant != &self.expected_authority_grant {
            return Err(VerifierProfileAdoptionRegistryCommitError::AuthorityGrantChanged);
        }
        Ok(VerifierProfileAdoptionCommitStateV1::ReadyToCommit)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum VerifierProfileAdoptionRegistryCommitError {
    #[error(transparent)]
    State(#[from] VerifierProfileAdoptionCommitError),
    #[error("authority-granted adoption and state preconditions disagree on predecessor head")]
    ExpectedPredecessorHeadMismatch,
    #[error("authority-granted adoption and state preconditions disagree on candidate head")]
    CandidateHeadMismatch,
    #[error("authority-granted adoption and state preconditions disagree on verifier role")]
    VerifierRoleMismatch,
    #[error("authority-granted adoption and state preconditions disagree on authority subject")]
    AuthoritySubjectMismatch,
    #[error("authority-granted adoption and state preconditions disagree on authority root id")]
    AuthorityRootIdMismatch,
    #[error("authority-granted adoption and state preconditions disagree on authority root digest")]
    AuthorityRootDigestMismatch,
    #[error("authority-granted adoption and state preconditions disagree on validity interval")]
    ValidityIntervalMismatch,
    #[error("local verifier-adoption authority grant changed before registry commit")]
    AuthorityGrantChanged,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile_adoption::{
        VerifierAdoptionScopeV1, VerifierProfileAdoptionSubjectV1,
        VerifierProfileAdoptionTransitionV1,
    };
    use crate::profile_adoption_admission::{
        PolicyCheckedVerifierProfileAdoptionV1, VerifierProfileAdoptionAdmissionPolicyV1,
        VerifierProfileAdoptionHeadV1,
    };
    use crate::profile_adoption_authority::{
        bind_policy_checked_adoption_to_authority_grant,
        AuthorityGrantedVerifierProfileAdoptionV1, VerifierAdoptionAuthorityGrantV1,
    };
    use crate::profile_adoption_commit::{
        VerifierAdoptionAuthorityRootSnapshotV1, VerifierProfileAdoptionCommitPreconditionsV1,
    };
    use crate::witness::EvidenceClass;
    use crate::VerifierProfileV1;

    fn profile(root: u8, epoch: u64) -> VerifierProfileV1 {
        VerifierProfileV1::new(
            "hardware-verifier-v1",
            [root; 32],
            epoch,
            EvidenceClass::HardwareVerified,
        )
        .unwrap()
    }

    fn checked_for(
        profile: &VerifierProfileV1,
        adoption_id: &str,
    ) -> PolicyCheckedVerifierProfileAdoptionV1 {
        let subject = VerifierProfileAdoptionSubjectV1::new(
            adoption_id,
            "organization:test",
            "adoption-root-1",
            [0x55; 32],
            profile,
            1,
            1_000,
            2_000,
            EvidenceClass::HardwareVerified,
            VerifierAdoptionScopeV1::AllContinuityVerification,
        )
        .unwrap();
        let transition = VerifierProfileAdoptionTransitionV1::bootstrap(subject).unwrap();
        let policy = VerifierProfileAdoptionAdmissionPolicyV1::new(
            "organization:test",
            "adoption-root-1",
            [0x55; 32],
            "hardware-verifier-v1",
            VerifierProfileAdoptionHeadV1::Uninitialized,
        )
        .unwrap();
        policy.check(1_500, &transition, profile, None).unwrap()
    }

    fn grant(epoch: u64, maximum: EvidenceClass) -> VerifierAdoptionAuthorityGrantV1 {
        VerifierAdoptionAuthorityGrantV1::new(
            "grant-1",
            "organization:test",
            "adoption-root-1",
            [0x55; 32],
            "hardware-verifier-v1",
            epoch,
            maximum,
            VerifierAdoptionScopeV1::AllContinuityVerification,
        )
        .unwrap()
    }

    fn fixture() -> (
        AuthorityGrantedVerifierProfileAdoptionV1,
        VerifierProfileAdoptionCommitPreconditionsV1,
        VerifierAdoptionAuthorityRootSnapshotV1,
    ) {
        let profile = profile(9, 7);
        let checked = checked_for(&profile, "adopt-1");
        let root = VerifierAdoptionAuthorityRootSnapshotV1::new(
            "organization:test",
            "adoption-root-1",
            [0x55; 32],
            1,
        )
        .unwrap();
        let state = VerifierProfileAdoptionCommitPreconditionsV1::from_policy_checked(
            &checked,
            root.clone(),
        )
        .unwrap();
        let granted = bind_policy_checked_adoption_to_authority_grant(
            checked,
            &grant(1, EvidenceClass::HardwareVerified),
            None,
        )
        .unwrap();
        (granted, state, root)
    }

    #[test]
    fn exact_root_grant_head_and_time_are_ready_to_commit() {
        let (granted, state, root) = fixture();
        let preconditions =
            VerifierProfileAdoptionRegistryCommitPreconditionsV1::from_authority_granted(
                &granted,
                state,
            )
            .unwrap();

        assert_eq!(
            preconditions
                .recheck_registry_observation(
                    1_500,
                    &root,
                    granted.authority_grant(),
                    granted.checked().expected_predecessor_head(),
                )
                .unwrap(),
            VerifierProfileAdoptionCommitStateV1::ReadyToCommit
        );
    }

    #[test]
    fn grant_epoch_change_invalidates_new_commit_even_when_root_and_head_do_not_change() {
        let (granted, state, root) = fixture();
        let preconditions =
            VerifierProfileAdoptionRegistryCommitPreconditionsV1::from_authority_granted(
                &granted,
                state,
            )
            .unwrap();
        let new_grant = grant(2, EvidenceClass::HardwareVerified);

        assert_eq!(
            preconditions.recheck_registry_observation(
                1_500,
                &root,
                &new_grant,
                granted.checked().expected_predecessor_head(),
            ),
            Err(VerifierProfileAdoptionRegistryCommitError::AuthorityGrantChanged)
        );
    }

    #[test]
    fn grant_strength_change_invalidates_new_commit() {
        let (granted, state, root) = fixture();
        let preconditions =
            VerifierProfileAdoptionRegistryCommitPreconditionsV1::from_authority_granted(
                &granted,
                state,
            )
            .unwrap();
        let weaker = grant(1, EvidenceClass::DifferentiallyVerified);

        assert_eq!(
            preconditions.recheck_registry_observation(
                1_500,
                &root,
                &weaker,
                granted.checked().expected_predecessor_head(),
            ),
            Err(VerifierProfileAdoptionRegistryCommitError::AuthorityGrantChanged)
        );
    }

    #[test]
    fn mixed_candidate_state_cannot_be_composed() {
        let profile = profile(9, 7);
        let checked_a = checked_for(&profile, "adopt-a");
        let checked_b = checked_for(&profile, "adopt-b");
        let root = VerifierAdoptionAuthorityRootSnapshotV1::new(
            "organization:test",
            "adoption-root-1",
            [0x55; 32],
            1,
        )
        .unwrap();
        let state_a = VerifierProfileAdoptionCommitPreconditionsV1::from_policy_checked(
            &checked_a,
            root,
        )
        .unwrap();
        let granted_b = bind_policy_checked_adoption_to_authority_grant(
            checked_b,
            &grant(1, EvidenceClass::HardwareVerified),
            None,
        )
        .unwrap();

        assert_eq!(
            VerifierProfileAdoptionRegistryCommitPreconditionsV1::from_authority_granted(
                &granted_b,
                state_a,
            ),
            Err(VerifierProfileAdoptionRegistryCommitError::CandidateHeadMismatch)
        );
    }

    #[test]
    fn already_committed_remains_historical_recovery_after_root_and_grant_change() {
        let (granted, state, _) = fixture();
        let preconditions =
            VerifierProfileAdoptionRegistryCommitPreconditionsV1::from_authority_granted(
                &granted,
                state,
            )
            .unwrap();
        let replaced_root = VerifierAdoptionAuthorityRootSnapshotV1::new(
            "organization:test",
            "replacement-root",
            [0x66; 32],
            9,
        )
        .unwrap();
        let replaced_grant = VerifierAdoptionAuthorityGrantV1::new(
            "replacement-grant",
            "organization:test",
            "replacement-root",
            [0x66; 32],
            "hardware-verifier-v1",
            9,
            EvidenceClass::Observed,
            VerifierAdoptionScopeV1::AllContinuityVerification,
        )
        .unwrap();

        assert_eq!(
            preconditions
                .recheck_registry_observation(
                    9_999,
                    &replaced_root,
                    &replaced_grant,
                    granted.checked().candidate_head(),
                )
                .unwrap(),
            VerifierProfileAdoptionCommitStateV1::AlreadyCommitted
        );
    }

    #[test]
    fn expiry_and_root_changes_still_fail_through_underlying_state_gate() {
        let (granted, state, root) = fixture();
        let preconditions =
            VerifierProfileAdoptionRegistryCommitPreconditionsV1::from_authority_granted(
                &granted,
                state,
            )
            .unwrap();

        assert!(matches!(
            preconditions.recheck_registry_observation(
                2_000,
                &root,
                granted.authority_grant(),
                granted.checked().expected_predecessor_head(),
            ),
            Err(VerifierProfileAdoptionRegistryCommitError::State(
                VerifierProfileAdoptionCommitError::Expired { .. }
            ))
        ));

        let new_root = VerifierAdoptionAuthorityRootSnapshotV1::new(
            "organization:test",
            "adoption-root-1",
            [0x55; 32],
            2,
        )
        .unwrap();
        assert_eq!(
            preconditions.recheck_registry_observation(
                1_500,
                &new_root,
                granted.authority_grant(),
                granted.checked().expected_predecessor_head(),
            ),
            Err(VerifierProfileAdoptionRegistryCommitError::State(
                VerifierProfileAdoptionCommitError::AuthorityRootSnapshotChanged
            ))
        );
    }
}
