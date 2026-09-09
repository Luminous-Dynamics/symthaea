// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Commit-time currentness for grant-bounded verifier-profile adoption.
//!
//! #1149 already proves that a fresh verifier-adoption write remains bound to the
//! exact root snapshot, predecessor head, validity interval, and trusted clock
//! lineage/uncertainty. #1195 separately proves that the adoption is attenuated by
//! an independent local grant capability.
//!
//! This module composes those facts without creating another root, clock, or head
//! model. For a fresh write the exact grant must remain unchanged as well.
//!
//! `trusted time + root/head currentness != grant currentness != authorization`.

use thiserror::Error;

use crate::profile_adoption_admission::VerifierProfileAdoptionHeadV1;
use crate::profile_adoption_commit::VerifierProfileAdoptionCommitStateV1;
use crate::profile_adoption_grant::{
    GrantBoundRootBoundVerifierProfileAdoptionV1, VerifierAdoptionAuthorityGrantV1,
};
use crate::profile_adoption_root::VerifierProfileAdoptionAuthorityRootSnapshotV1;
use crate::profile_adoption_time::{
    TimeBoundVerifierProfileAdoptionCommitPreconditionsV1,
    VerifierProfileAdoptionClockObservationV1, VerifierProfileAdoptionTimeError,
};

/// Commit preconditions that bind the canonical trusted-time/root/head theorem to
/// the exact local grant used when the adoption was admitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrantBoundVerifierProfileAdoptionCommitPreconditionsV1 {
    grant_bound: GrantBoundRootBoundVerifierProfileAdoptionV1,
    time_bound: TimeBoundVerifierProfileAdoptionCommitPreconditionsV1,
}

impl GrantBoundVerifierProfileAdoptionCommitPreconditionsV1 {
    /// Compose only if both inputs refer to the same exact adoption/root/head.
    pub fn new(
        grant_bound: GrantBoundRootBoundVerifierProfileAdoptionV1,
        time_bound: TimeBoundVerifierProfileAdoptionCommitPreconditionsV1,
    ) -> Result<Self, VerifierProfileAdoptionGrantCommitError> {
        let root_bound = grant_bound.root_bound();
        let time_inner = time_bound.inner();

        if root_bound.transition_digest() != time_inner.transition_digest() {
            return Err(VerifierProfileAdoptionGrantCommitError::TransitionMismatch);
        }
        if root_bound.authority_root_snapshot() != time_inner.expected_root_snapshot() {
            return Err(VerifierProfileAdoptionGrantCommitError::AuthorityRootMismatch);
        }
        if root_bound.expected_predecessor_head() != time_inner.expected_predecessor_head() {
            return Err(VerifierProfileAdoptionGrantCommitError::PredecessorHeadMismatch);
        }
        if root_bound.candidate_head() != time_inner.candidate_head() {
            return Err(VerifierProfileAdoptionGrantCommitError::CandidateHeadMismatch);
        }
        if root_bound.profile().id() != time_inner.profile().id() {
            return Err(VerifierProfileAdoptionGrantCommitError::VerifierProfileMismatch);
        }

        Ok(Self {
            grant_bound,
            time_bound,
        })
    }

    pub fn grant_bound(&self) -> &GrantBoundRootBoundVerifierProfileAdoptionV1 {
        &self.grant_bound
    }

    pub fn time_bound(&self) -> &TimeBoundVerifierProfileAdoptionCommitPreconditionsV1 {
        &self.time_bound
    }

    pub fn expected_grant(&self) -> &VerifierAdoptionAuthorityGrantV1 {
        self.grant_bound.authority_grant()
    }

    pub fn candidate_head(&self) -> &VerifierProfileAdoptionHeadV1 {
        self.grant_bound.root_bound().candidate_head()
    }

    /// Recheck all state immediately before a future atomic registry write.
    ///
    /// Exact-candidate equality is historical uncertain-write recovery and is
    /// recognized first. It does not mean the verifier remains authorized now.
    /// A fresh write requires the exact grant to remain unchanged in addition to
    /// #1149's trusted clock, root snapshot, predecessor head, and validity checks.
    pub fn recheck_commit_observation(
        &self,
        current_clock: &VerifierProfileAdoptionClockObservationV1,
        current_root: &VerifierProfileAdoptionAuthorityRootSnapshotV1,
        current_grant: &VerifierAdoptionAuthorityGrantV1,
        current_head: &VerifierProfileAdoptionHeadV1,
    ) -> Result<VerifierProfileAdoptionCommitStateV1, VerifierProfileAdoptionGrantCommitError> {
        if current_head == self.candidate_head() {
            return Ok(VerifierProfileAdoptionCommitStateV1::AlreadyCommitted);
        }

        if current_grant != self.expected_grant() {
            return Err(VerifierProfileAdoptionGrantCommitError::AuthorityGrantChanged);
        }

        Ok(self
            .time_bound
            .recheck_commit_observation(current_clock, current_root, current_head)?)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum VerifierProfileAdoptionGrantCommitError {
    #[error(transparent)]
    Time(#[from] VerifierProfileAdoptionTimeError),
    #[error("grant-bound adoption and trusted-time commit capsule reference different transitions")]
    TransitionMismatch,
    #[error("grant-bound adoption and trusted-time commit capsule reference different root snapshots")]
    AuthorityRootMismatch,
    #[error("grant-bound adoption and trusted-time commit capsule reference different predecessor heads")]
    PredecessorHeadMismatch,
    #[error("grant-bound adoption and trusted-time commit capsule reference different candidate heads")]
    CandidateHeadMismatch,
    #[error("grant-bound adoption and trusted-time commit capsule reference different verifier profiles")]
    VerifierProfileMismatch,
    #[error("local verifier-adoption grant changed before commit")]
    AuthorityGrantChanged,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        EvidenceClass, RootBoundPolicyCheckedVerifierProfileAdoptionV1,
        VerifierAdoptionAuthorityGrantV1, VerifierAdoptionScopeV1,
        VerifierProfileAdoptionAdmissionPolicyV1,
        VerifierProfileAdoptionAuthorityRootSnapshotV1, VerifierProfileAdoptionCommitPreconditionsV1,
        VerifierProfileAdoptionHeadV1, VerifierProfileAdoptionSubjectV1,
        VerifierProfileAdoptionTransitionV1, VerifierProfileV1,
        bind_root_bound_adoption_to_authority_grant,
    };

    fn profile() -> VerifierProfileV1 {
        VerifierProfileV1::new(
            "hardware-verifier-v1",
            [9; 32],
            7,
            EvidenceClass::HardwareVerified,
        )
        .unwrap()
    }

    fn root(epoch: u64) -> VerifierProfileAdoptionAuthorityRootSnapshotV1 {
        VerifierProfileAdoptionAuthorityRootSnapshotV1::new(
            "organization:test",
            "adoption-root-1",
            [0x55; 32],
            epoch,
        )
        .unwrap()
    }

    fn grant(
        root: VerifierProfileAdoptionAuthorityRootSnapshotV1,
        grant_epoch: u64,
        maximum: EvidenceClass,
    ) -> VerifierAdoptionAuthorityGrantV1 {
        VerifierAdoptionAuthorityGrantV1::new(
            "grant-1",
            root,
            "hardware-verifier-v1",
            grant_epoch,
            maximum,
            VerifierAdoptionScopeV1::AllContinuityVerification,
        )
        .unwrap()
    }

    fn root_bound(
        root: &VerifierProfileAdoptionAuthorityRootSnapshotV1,
        adoption_id: &str,
    ) -> RootBoundPolicyCheckedVerifierProfileAdoptionV1 {
        let profile = profile();
        let subject = VerifierProfileAdoptionSubjectV1::new(
            adoption_id,
            root.authority_subject(),
            root.authority_root_id(),
            root.authority_root_digest(),
            &profile,
            1,
            1_000,
            2_000,
            EvidenceClass::DifferentiallyVerified,
            VerifierAdoptionScopeV1::AllContinuityVerification,
        )
        .unwrap();
        let transition = VerifierProfileAdoptionTransitionV1::bootstrap(subject).unwrap();
        let checked = VerifierProfileAdoptionAdmissionPolicyV1::new(
            root.authority_subject(),
            root.authority_root_id(),
            root.authority_root_digest(),
            profile.profile_name(),
            VerifierProfileAdoptionHeadV1::Uninitialized,
        )
        .unwrap()
        .check(1_500, &transition, &profile, None)
        .unwrap();
        RootBoundPolicyCheckedVerifierProfileAdoptionV1::bind(checked, root.clone()).unwrap()
    }

    fn clock(epoch: u64, earliest: u64, latest: u64) -> VerifierProfileAdoptionClockObservationV1 {
        VerifierProfileAdoptionClockObservationV1::new(
            "trusted-clock-1",
            epoch,
            earliest,
            latest,
        )
        .unwrap()
    }

    fn fixture() -> (
        GrantBoundVerifierProfileAdoptionCommitPreconditionsV1,
        VerifierProfileAdoptionAuthorityRootSnapshotV1,
        VerifierAdoptionAuthorityGrantV1,
        VerifierProfileAdoptionClockObservationV1,
    ) {
        let root = root(9);
        let root_bound = root_bound(&root, "adopt-1");
        let commit = VerifierProfileAdoptionCommitPreconditionsV1::from_root_bound(
            root_bound.clone(),
        )
        .unwrap();
        let checked_clock = clock(4, 1_490, 1_510);
        let time_bound = TimeBoundVerifierProfileAdoptionCommitPreconditionsV1::new(
            commit,
            &checked_clock,
            50,
        )
        .unwrap();
        let grant = grant(root.clone(), 3, EvidenceClass::HardwareVerified);
        let grant_bound = bind_root_bound_adoption_to_authority_grant(
            root_bound,
            &grant,
            None,
        )
        .unwrap();
        let combined = GrantBoundVerifierProfileAdoptionCommitPreconditionsV1::new(
            grant_bound,
            time_bound,
        )
        .unwrap();
        (combined, root, grant, checked_clock)
    }

    #[test]
    fn exact_grant_root_head_and_trusted_time_are_ready() {
        let (combined, root, grant, current_clock) = fixture();
        assert_eq!(
            combined
                .recheck_commit_observation(
                    &current_clock,
                    &root,
                    &grant,
                    combined.grant_bound().root_bound().expected_predecessor_head(),
                )
                .unwrap(),
            VerifierProfileAdoptionCommitStateV1::ReadyToCommit
        );
    }

    #[test]
    fn grant_epoch_change_invalidates_fresh_write() {
        let (combined, root, _grant, current_clock) = fixture();
        let changed = grant(root.clone(), 4, EvidenceClass::HardwareVerified);
        assert_eq!(
            combined
                .recheck_commit_observation(
                    &current_clock,
                    &root,
                    &changed,
                    combined.grant_bound().root_bound().expected_predecessor_head(),
                )
                .unwrap_err(),
            VerifierProfileAdoptionGrantCommitError::AuthorityGrantChanged
        );
    }

    #[test]
    fn grant_ceiling_change_invalidates_fresh_write() {
        let (combined, root, _grant, current_clock) = fixture();
        let changed = grant(root.clone(), 3, EvidenceClass::DifferentiallyVerified);
        assert_eq!(
            combined
                .recheck_commit_observation(
                    &current_clock,
                    &root,
                    &changed,
                    combined.grant_bound().root_bound().expected_predecessor_head(),
                )
                .unwrap_err(),
            VerifierProfileAdoptionGrantCommitError::AuthorityGrantChanged
        );
    }

    #[test]
    fn clock_lineage_change_still_fails_through_canonical_time_gate() {
        let (combined, root, grant, _current_clock) = fixture();
        let changed_clock = clock(5, 1_490, 1_510);
        assert!(matches!(
            combined.recheck_commit_observation(
                &changed_clock,
                &root,
                &grant,
                combined.grant_bound().root_bound().expected_predecessor_head(),
            ),
            Err(VerifierProfileAdoptionGrantCommitError::Time(
                VerifierProfileAdoptionTimeError::ClockLineageChanged { .. }
            ))
        ));
    }

    #[test]
    fn already_committed_is_historical_even_after_grant_change() {
        let (combined, root, _grant, _current_clock) = fixture();
        let changed_grant = grant(root.clone(), 99, EvidenceClass::Declared);
        let changed_clock = clock(99, 9_000, 9_100);
        assert_eq!(
            combined
                .recheck_commit_observation(
                    &changed_clock,
                    &root,
                    &changed_grant,
                    combined.candidate_head(),
                )
                .unwrap(),
            VerifierProfileAdoptionCommitStateV1::AlreadyCommitted
        );
    }

    #[test]
    fn capsules_from_different_adoptions_cannot_be_composed() {
        let root = root(9);
        let a = root_bound(&root, "adopt-a");
        let b = root_bound(&root, "adopt-b");
        let grant = grant(root.clone(), 3, EvidenceClass::HardwareVerified);
        let grant_bound = bind_root_bound_adoption_to_authority_grant(a, &grant, None).unwrap();
        let commit_b = VerifierProfileAdoptionCommitPreconditionsV1::from_root_bound(b).unwrap();
        let time_b = TimeBoundVerifierProfileAdoptionCommitPreconditionsV1::new(
            commit_b,
            &clock(4, 1_490, 1_510),
            50,
        )
        .unwrap();
        assert_eq!(
            GrantBoundVerifierProfileAdoptionCommitPreconditionsV1::new(grant_bound, time_b)
                .unwrap_err(),
            VerifierProfileAdoptionGrantCommitError::TransitionMismatch
        );
    }
}
