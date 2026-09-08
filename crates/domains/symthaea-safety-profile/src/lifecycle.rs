// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Atomic lifecycle state for safety-profile authorization.
//!
//! Authorization head state and revocation state must be observed and compared as
//! one unit. A head-only CAS cannot detect a revocation that races a prepared
//! successor authorization because revoking head N does not change the head itself.
//! This module lifts the existing authorization and revocation commit preconditions
//! onto one exact lifecycle state so that race fails closed.
//!
//! The lifecycle state is local persistence state, not cryptographic proof. Future
//! storage adapters remain responsible for sourcing it from trusted durable state,
//! and every state-changing write must still require verifier-owned proof for the
//! exact authorization or revocation bytes before committing the candidate state.

use crate::admission::SafetyProfileAuthorizationHead;
use crate::commit_preconditions::{
    ProfileAuthorityRootSnapshot, SafetyProfileAuthorizationCommitState,
};
use crate::revocation::SafetyProfileAuthorizationRevocationDigest;
use crate::revocation_commit_preconditions::{
    SafetyProfileAuthorizationRevocationCommitError,
    SafetyProfileAuthorizationRevocationCommitPreconditions,
    SafetyProfileAuthorizationRevocationCommitState,
};
use crate::trusted_time::{
    TimeBoundSafetyProfileAuthorizationCommitPreconditions,
    TrustedAuthorizationClockObservation, TrustedAuthorizationTimeError,
};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SafetyProfileAuthorizationLifecycleDisposition {
    Uninitialized,
    Active,
    Revoked(SafetyProfileAuthorizationRevocationDigest),
}

/// Exact locally persisted lifecycle state for one profile-authorization chain.
///
/// `Active` means the exact current authorization head has no committed revocation
/// marker in this lifecycle observation. `Revoked` binds one exact revocation
/// digest to that exact current head. Constructors reject impossible combinations
/// such as an active or revoked uninitialized head.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetyProfileAuthorizationLifecycleState {
    authorization_head: SafetyProfileAuthorizationHead,
    disposition: SafetyProfileAuthorizationLifecycleDisposition,
}

impl SafetyProfileAuthorizationLifecycleState {
    pub fn uninitialized() -> Self {
        Self {
            authorization_head: SafetyProfileAuthorizationHead::Uninitialized,
            disposition: SafetyProfileAuthorizationLifecycleDisposition::Uninitialized,
        }
    }

    pub fn active(
        authorization_head: SafetyProfileAuthorizationHead,
    ) -> Result<Self, SafetyProfileAuthorizationLifecycleError> {
        if authorization_head.identity().is_none() {
            return Err(SafetyProfileAuthorizationLifecycleError::ActiveHeadUninitialized);
        }
        Ok(Self {
            authorization_head,
            disposition: SafetyProfileAuthorizationLifecycleDisposition::Active,
        })
    }

    pub fn revoked(
        authorization_head: SafetyProfileAuthorizationHead,
        revocation_digest: SafetyProfileAuthorizationRevocationDigest,
    ) -> Result<Self, SafetyProfileAuthorizationLifecycleError> {
        if authorization_head.identity().is_none() {
            return Err(SafetyProfileAuthorizationLifecycleError::RevokedHeadUninitialized);
        }
        Ok(Self {
            authorization_head,
            disposition: SafetyProfileAuthorizationLifecycleDisposition::Revoked(
                revocation_digest,
            ),
        })
    }

    pub fn authorization_head(&self) -> &SafetyProfileAuthorizationHead {
        &self.authorization_head
    }

    pub fn revocation_digest(&self) -> Option<SafetyProfileAuthorizationRevocationDigest> {
        match self.disposition {
            SafetyProfileAuthorizationLifecycleDisposition::Revoked(digest) => Some(digest),
            SafetyProfileAuthorizationLifecycleDisposition::Uninitialized
            | SafetyProfileAuthorizationLifecycleDisposition::Active => None,
        }
    }

    pub fn is_uninitialized(&self) -> bool {
        self.disposition == SafetyProfileAuthorizationLifecycleDisposition::Uninitialized
    }

    pub fn is_active(&self) -> bool {
        self.disposition == SafetyProfileAuthorizationLifecycleDisposition::Active
    }

    pub fn is_revoked(&self) -> bool {
        matches!(
            self.disposition,
            SafetyProfileAuthorizationLifecycleDisposition::Revoked(_)
        )
    }
}

/// Result of rechecking one lifecycle-aware atomic write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyProfileAuthorizationLifecycleCommitState {
    ReadyToCommit,
    /// The underlying authorization/revocation write already happened. This is
    /// historical idempotent acknowledgement only, not current execution authority.
    AlreadyCommitted,
}

/// Lifecycle-aware wrapper for one time-bound authorization commit.
///
/// Bootstrap may be prepared from `Uninitialized`; an ordinary successor may be
/// prepared only from an `Active` predecessor. A committed revocation is sticky:
/// an ordinary successor transition does not authenticate the revocation digest and
/// therefore cannot clear `Revoked(head, digest)`. Restoring eligibility after
/// revocation requires a separate reinstatement transition that explicitly binds
/// the exact revocation being superseded.
///
/// The complete observed predecessor lifecycle state must remain unchanged until
/// the atomic write, so a revocation racing after preparation invalidates this
/// commit even though the predecessor authorization head itself has not changed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetyProfileAuthorizationLifecycleCommitPreconditions {
    inner: TimeBoundSafetyProfileAuthorizationCommitPreconditions,
    expected_lifecycle_state: SafetyProfileAuthorizationLifecycleState,
    candidate_lifecycle_state: SafetyProfileAuthorizationLifecycleState,
}

impl SafetyProfileAuthorizationLifecycleCommitPreconditions {
    pub fn new(
        inner: TimeBoundSafetyProfileAuthorizationCommitPreconditions,
        observed_lifecycle_state: SafetyProfileAuthorizationLifecycleState,
    ) -> Result<Self, SafetyProfileAuthorizationLifecycleError> {
        let expected_head = inner.inner().expected_predecessor_head();
        if observed_lifecycle_state.authorization_head() != expected_head {
            return Err(
                SafetyProfileAuthorizationLifecycleError::ObservedAuthorizationHeadMismatch,
            );
        }
        if observed_lifecycle_state.is_revoked() {
            return Err(
                SafetyProfileAuthorizationLifecycleError::AuthorizationCannotClearRevocation,
            );
        }

        let candidate_lifecycle_state = SafetyProfileAuthorizationLifecycleState::active(
            inner.inner().candidate_head().clone(),
        )?;

        Ok(Self {
            inner,
            expected_lifecycle_state: observed_lifecycle_state,
            candidate_lifecycle_state,
        })
    }

    pub fn inner(&self) -> &TimeBoundSafetyProfileAuthorizationCommitPreconditions {
        &self.inner
    }

    pub fn expected_lifecycle_state(&self) -> &SafetyProfileAuthorizationLifecycleState {
        &self.expected_lifecycle_state
    }

    pub fn candidate_lifecycle_state(&self) -> &SafetyProfileAuthorizationLifecycleState {
        &self.candidate_lifecycle_state
    }

    pub fn recheck_commit_observation(
        &self,
        current_clock: &TrustedAuthorizationClockObservation,
        current_root: &ProfileAuthorityRootSnapshot,
        current_lifecycle_state: &SafetyProfileAuthorizationLifecycleState,
    ) -> Result<
        SafetyProfileAuthorizationLifecycleCommitState,
        SafetyProfileAuthorizationLifecycleError,
    > {
        // If the candidate head is now current, the authorization write already
        // happened. A subsequent revocation of that candidate does not erase the
        // historical fact that the authorization commit succeeded.
        if current_lifecycle_state.authorization_head()
            == self.candidate_lifecycle_state.authorization_head()
        {
            return Ok(SafetyProfileAuthorizationLifecycleCommitState::AlreadyCommitted);
        }

        if current_lifecycle_state != &self.expected_lifecycle_state {
            return Err(SafetyProfileAuthorizationLifecycleError::LifecycleStateChanged);
        }

        let state = self.inner.recheck_commit_observation(
            current_clock,
            current_root,
            current_lifecycle_state.authorization_head(),
        )?;
        Ok(match state {
            SafetyProfileAuthorizationCommitState::ReadyToCommit => {
                SafetyProfileAuthorizationLifecycleCommitState::ReadyToCommit
            }
            SafetyProfileAuthorizationCommitState::AlreadyCommitted => {
                SafetyProfileAuthorizationLifecycleCommitState::AlreadyCommitted
            }
        })
    }
}

/// Lifecycle-aware wrapper for one profile-authorization revocation commit.
///
/// Revocation may only be prepared while the exact target head is active. The
/// candidate state atomically changes that same head to `Revoked(exact_digest)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetyProfileAuthorizationRevocationLifecycleCommitPreconditions {
    inner: SafetyProfileAuthorizationRevocationCommitPreconditions,
    expected_lifecycle_state: SafetyProfileAuthorizationLifecycleState,
    candidate_lifecycle_state: SafetyProfileAuthorizationLifecycleState,
}

impl SafetyProfileAuthorizationRevocationLifecycleCommitPreconditions {
    pub fn new(
        inner: SafetyProfileAuthorizationRevocationCommitPreconditions,
        observed_lifecycle_state: SafetyProfileAuthorizationLifecycleState,
    ) -> Result<Self, SafetyProfileAuthorizationLifecycleError> {
        if !observed_lifecycle_state.is_active() {
            return Err(SafetyProfileAuthorizationLifecycleError::RevocationRequiresActiveHead);
        }
        if observed_lifecycle_state.authorization_head() != inner.expected_current_head() {
            return Err(
                SafetyProfileAuthorizationLifecycleError::ObservedAuthorizationHeadMismatch,
            );
        }

        let candidate_lifecycle_state = SafetyProfileAuthorizationLifecycleState::revoked(
            inner.expected_current_head().clone(),
            inner.revocation_digest(),
        )?;

        Ok(Self {
            inner,
            expected_lifecycle_state: observed_lifecycle_state,
            candidate_lifecycle_state,
        })
    }

    pub fn inner(&self) -> &SafetyProfileAuthorizationRevocationCommitPreconditions {
        &self.inner
    }

    pub fn expected_lifecycle_state(&self) -> &SafetyProfileAuthorizationLifecycleState {
        &self.expected_lifecycle_state
    }

    pub fn candidate_lifecycle_state(&self) -> &SafetyProfileAuthorizationLifecycleState {
        &self.candidate_lifecycle_state
    }

    pub fn recheck_commit_observation(
        &self,
        current_clock: &TrustedAuthorizationClockObservation,
        current_root: &ProfileAuthorityRootSnapshot,
        current_lifecycle_state: &SafetyProfileAuthorizationLifecycleState,
    ) -> Result<
        SafetyProfileAuthorizationLifecycleCommitState,
        SafetyProfileAuthorizationLifecycleError,
    > {
        if current_lifecycle_state == &self.candidate_lifecycle_state {
            return Ok(SafetyProfileAuthorizationLifecycleCommitState::AlreadyCommitted);
        }

        // A successor authorization or other head movement wins the race and makes
        // this revocation stale. Check head movement before forwarding the current
        // revocation marker so a revocation attached to a different head is not
        // misreported as a same-head revocation conflict.
        if current_lifecycle_state.authorization_head()
            != self.expected_lifecycle_state.authorization_head()
        {
            return Err(SafetyProfileAuthorizationLifecycleError::LifecycleStateChanged);
        }

        let state = self.inner.recheck_commit_observation(
            current_clock,
            current_root,
            current_lifecycle_state.authorization_head(),
            current_lifecycle_state.revocation_digest(),
        )?;
        Ok(match state {
            SafetyProfileAuthorizationRevocationCommitState::ReadyToCommit => {
                SafetyProfileAuthorizationLifecycleCommitState::ReadyToCommit
            }
            SafetyProfileAuthorizationRevocationCommitState::AlreadyCommitted => {
                SafetyProfileAuthorizationLifecycleCommitState::AlreadyCommitted
            }
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SafetyProfileAuthorizationLifecycleError {
    #[error("active lifecycle state requires a current authorization head")]
    ActiveHeadUninitialized,
    #[error("revoked lifecycle state requires a current authorization head")]
    RevokedHeadUninitialized,
    #[error("observed lifecycle authorization head does not match commit preconditions")]
    ObservedAuthorizationHeadMismatch,
    #[error("ordinary profile authorization cannot clear an already committed revocation; use an explicit revocation-bound reinstatement transition")]
    AuthorizationCannotClearRevocation,
    #[error("profile-authorization lifecycle state changed before commit")]
    LifecycleStateChanged,
    #[error("profile authorization must be active before a revocation commit is prepared")]
    RevocationRequiresActiveHead,
    #[error(transparent)]
    AuthorizationTime(#[from] TrustedAuthorizationTimeError),
    #[error(transparent)]
    RevocationCommit(#[from] SafetyProfileAuthorizationRevocationCommitError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admission::SafetyProfileAuthorizationAdmissionPolicy;
    use crate::authorization::{ProfileAuthorityRootDigest, SafetyProfileAuthorizationSubject};
    use crate::commit_preconditions::SafetyProfileAuthorizationCommitPreconditions;
    use crate::revocation::SafetyProfileAuthorizationRevocationSubject;
    use crate::revocation_admission::SafetyProfileAuthorizationRevocationAdmissionPolicy;
    use crate::transition::SafetyProfileAuthorizationTransition;
    use crate::{
        ComponentRequirement, SafetyConfigurationProfile,
        SAFETY_CONFIGURATION_PROFILE_SCHEMA_V1,
    };

    fn root(byte: u8) -> ProfileAuthorityRootDigest {
        ProfileAuthorityRootDigest::Blake3_256([byte; 32])
    }

    fn profile() -> SafetyConfigurationProfile {
        SafetyConfigurationProfile {
            schema_version: SAFETY_CONFIGURATION_PROFILE_SCHEMA_V1.to_owned(),
            profile_id: "compute-commons-autonomous-node-v1".to_owned(),
            hardware_inventory: ComponentRequirement::Required,
            firmware: ComponentRequirement::Required,
            software_closure: ComponentRequirement::Required,
            electrical_topology: ComponentRequirement::Required,
            thermal_topology: ComponentRequirement::Required,
            protection_settings: ComponentRequirement::Required,
            sensor_map: ComponentRequirement::Required,
            actuator_map: ComponentRequirement::Required,
            calibration: ComponentRequirement::Required,
            network_topology: ComponentRequirement::Required,
        }
    }

    fn subject(id: &str, generation: u64) -> SafetyProfileAuthorizationSubject {
        SafetyProfileAuthorizationSubject::new(
            id,
            "facility-profile-root-v1",
            root(0x33),
            "compute-campus",
            generation,
            1_000,
            3_000,
            &profile(),
        )
        .unwrap()
    }

    fn bootstrap() -> SafetyProfileAuthorizationTransition {
        SafetyProfileAuthorizationTransition::bootstrap(subject("auth-1", 1)).unwrap()
    }

    fn successor(
        predecessor: &SafetyProfileAuthorizationTransition,
    ) -> SafetyProfileAuthorizationTransition {
        SafetyProfileAuthorizationTransition::successor(subject("auth-2", 2), predecessor).unwrap()
    }

    fn clock(
        source: &str,
        epoch: u64,
        earliest: i64,
        latest: i64,
    ) -> TrustedAuthorizationClockObservation {
        TrustedAuthorizationClockObservation::new(source, epoch, earliest, latest).unwrap()
    }

    fn root_snapshot() -> ProfileAuthorityRootSnapshot {
        ProfileAuthorityRootSnapshot::new("facility-profile-root-v1", root(0x33), 4).unwrap()
    }

    fn time_bound_authorization(
        transition: &SafetyProfileAuthorizationTransition,
        current_head: SafetyProfileAuthorizationHead,
    ) -> TimeBoundSafetyProfileAuthorizationCommitPreconditions {
        let policy = SafetyProfileAuthorizationAdmissionPolicy::new(
            "compute-campus",
            root(0x33),
            current_head,
        )
        .unwrap();
        let checked = policy.check(1_500, transition, &profile()).unwrap();
        let commit = SafetyProfileAuthorizationCommitPreconditions::from_policy_checked(
            &checked,
            root_snapshot(),
        )
        .unwrap();
        TimeBoundSafetyProfileAuthorizationCommitPreconditions::new(
            commit,
            &clock("secure-rtc-v1", 7, 1_500, 1_600),
        )
        .unwrap()
    }

    fn revocation_commit(
        target: &SafetyProfileAuthorizationTransition,
    ) -> SafetyProfileAuthorizationRevocationCommitPreconditions {
        let head = SafetyProfileAuthorizationHead::from_transition(target).unwrap();
        let policy = SafetyProfileAuthorizationRevocationAdmissionPolicy::new(
            "compute-campus",
            root_snapshot(),
            head,
        )
        .unwrap();
        let revocation = SafetyProfileAuthorizationRevocationSubject::for_transition(
            "revoke-1",
            target,
            1_400,
        )
        .unwrap();
        let checked = policy
            .check(&clock("secure-rtc-v1", 7, 1_500, 1_600), &revocation)
            .unwrap();
        SafetyProfileAuthorizationRevocationCommitPreconditions::from_policy_checked(&checked)
            .unwrap()
    }

    fn arbitrary_revocation_digest(byte: u8) -> SafetyProfileAuthorizationRevocationDigest {
        SafetyProfileAuthorizationRevocationDigest::Blake3_256([byte; 32])
    }

    #[test]
    fn impossible_uninitialized_active_and_revoked_states_are_rejected() {
        assert_eq!(
            SafetyProfileAuthorizationLifecycleState::active(
                SafetyProfileAuthorizationHead::Uninitialized,
            ),
            Err(SafetyProfileAuthorizationLifecycleError::ActiveHeadUninitialized)
        );
        assert_eq!(
            SafetyProfileAuthorizationLifecycleState::revoked(
                SafetyProfileAuthorizationHead::Uninitialized,
                arbitrary_revocation_digest(0x44),
            ),
            Err(SafetyProfileAuthorizationLifecycleError::RevokedHeadUninitialized)
        );
    }

    #[test]
    fn bootstrap_moves_uninitialized_lifecycle_to_active() {
        let transition = bootstrap();
        let inner = time_bound_authorization(
            &transition,
            SafetyProfileAuthorizationHead::Uninitialized,
        );
        let preconditions = SafetyProfileAuthorizationLifecycleCommitPreconditions::new(
            inner,
            SafetyProfileAuthorizationLifecycleState::uninitialized(),
        )
        .unwrap();

        assert!(preconditions.candidate_lifecycle_state().is_active());
        assert_eq!(
            preconditions
                .recheck_commit_observation(
                    &clock("secure-rtc-v1", 7, 1_550, 1_650),
                    &root_snapshot(),
                    &SafetyProfileAuthorizationLifecycleState::uninitialized(),
                )
                .unwrap(),
            SafetyProfileAuthorizationLifecycleCommitState::ReadyToCommit
        );
    }

    #[test]
    fn revocation_racing_prepared_successor_invalidates_authorization_commit() {
        let first = bootstrap();
        let second = successor(&first);
        let first_head = SafetyProfileAuthorizationHead::from_transition(&first).unwrap();
        let observed = SafetyProfileAuthorizationLifecycleState::active(first_head.clone()).unwrap();
        let inner = time_bound_authorization(&second, first_head.clone());
        let preconditions = SafetyProfileAuthorizationLifecycleCommitPreconditions::new(
            inner,
            observed,
        )
        .unwrap();
        let raced_revocation = SafetyProfileAuthorizationLifecycleState::revoked(
            first_head,
            arbitrary_revocation_digest(0x99),
        )
        .unwrap();

        assert_eq!(
            preconditions.recheck_commit_observation(
                &clock("secure-rtc-v1", 7, 1_550, 1_650),
                &root_snapshot(),
                &raced_revocation,
            ),
            Err(SafetyProfileAuthorizationLifecycleError::LifecycleStateChanged)
        );
    }

    #[test]
    fn ordinary_successor_cannot_clear_observed_revocation() {
        let first = bootstrap();
        let second = successor(&first);
        let first_head = SafetyProfileAuthorizationHead::from_transition(&first).unwrap();
        let observed = SafetyProfileAuthorizationLifecycleState::revoked(
            first_head.clone(),
            arbitrary_revocation_digest(0x77),
        )
        .unwrap();
        let inner = time_bound_authorization(&second, first_head);

        assert!(matches!(
            SafetyProfileAuthorizationLifecycleCommitPreconditions::new(inner, observed),
            Err(SafetyProfileAuthorizationLifecycleError::AuthorizationCannotClearRevocation)
        ));
    }

    #[test]
    fn authorization_commit_recovers_idempotently_even_if_candidate_was_later_revoked() {
        let first = bootstrap();
        let second = successor(&first);
        let first_head = SafetyProfileAuthorizationHead::from_transition(&first).unwrap();
        let inner = time_bound_authorization(&second, first_head.clone());
        let preconditions = SafetyProfileAuthorizationLifecycleCommitPreconditions::new(
            inner,
            SafetyProfileAuthorizationLifecycleState::active(first_head).unwrap(),
        )
        .unwrap();
        let candidate_head = preconditions
            .candidate_lifecycle_state()
            .authorization_head()
            .clone();
        let later_revoked = SafetyProfileAuthorizationLifecycleState::revoked(
            candidate_head,
            arbitrary_revocation_digest(0xaa),
        )
        .unwrap();

        assert_eq!(
            preconditions
                .recheck_commit_observation(
                    &clock("different-clock", 99, 9_000, 9_100),
                    &ProfileAuthorityRootSnapshot::new(
                        "different-root",
                        root(0xee),
                        99,
                    )
                    .unwrap(),
                    &later_revoked,
                )
                .unwrap(),
            SafetyProfileAuthorizationLifecycleCommitState::AlreadyCommitted
        );
    }

    #[test]
    fn revocation_moves_exact_active_head_to_exact_revoked_state() {
        let target = bootstrap();
        let head = SafetyProfileAuthorizationHead::from_transition(&target).unwrap();
        let observed = SafetyProfileAuthorizationLifecycleState::active(head).unwrap();
        let inner = revocation_commit(&target);
        let preconditions =
            SafetyProfileAuthorizationRevocationLifecycleCommitPreconditions::new(
                inner,
                observed.clone(),
            )
            .unwrap();

        assert!(preconditions.candidate_lifecycle_state().is_revoked());
        assert_eq!(
            preconditions
                .recheck_commit_observation(
                    &clock("secure-rtc-v1", 7, 1_550, 1_650),
                    &root_snapshot(),
                    &observed,
                )
                .unwrap(),
            SafetyProfileAuthorizationLifecycleCommitState::ReadyToCommit
        );
    }

    #[test]
    fn exact_revocation_candidate_recovers_idempotently() {
        let target = bootstrap();
        let head = SafetyProfileAuthorizationHead::from_transition(&target).unwrap();
        let observed = SafetyProfileAuthorizationLifecycleState::active(head).unwrap();
        let inner = revocation_commit(&target);
        let preconditions =
            SafetyProfileAuthorizationRevocationLifecycleCommitPreconditions::new(
                inner,
                observed,
            )
            .unwrap();
        let candidate = preconditions.candidate_lifecycle_state().clone();

        assert_eq!(
            preconditions
                .recheck_commit_observation(
                    &clock("different-clock", 99, 9_000, 9_100),
                    &ProfileAuthorityRootSnapshot::new(
                        "different-root",
                        root(0xee),
                        99,
                    )
                    .unwrap(),
                    &candidate,
                )
                .unwrap(),
            SafetyProfileAuthorizationLifecycleCommitState::AlreadyCommitted
        );
    }

    #[test]
    fn different_same_head_revocation_remains_an_explicit_conflict() {
        let target = bootstrap();
        let head = SafetyProfileAuthorizationHead::from_transition(&target).unwrap();
        let observed = SafetyProfileAuthorizationLifecycleState::active(head.clone()).unwrap();
        let inner = revocation_commit(&target);
        let expected_digest = inner.revocation_digest();
        let preconditions =
            SafetyProfileAuthorizationRevocationLifecycleCommitPreconditions::new(
                inner,
                observed,
            )
            .unwrap();
        let conflicting = SafetyProfileAuthorizationLifecycleState::revoked(
            head,
            arbitrary_revocation_digest(0xbb),
        )
        .unwrap();

        assert!(matches!(
            preconditions.recheck_commit_observation(
                &clock("secure-rtc-v1", 7, 1_550, 1_650),
                &root_snapshot(),
                &conflicting,
            ),
            Err(SafetyProfileAuthorizationLifecycleError::RevocationCommit(
                SafetyProfileAuthorizationRevocationCommitError::DifferentRevocationAlreadyCommitted {
                    expected,
                    observed: _,
                }
            )) if expected == expected_digest
        ));
    }

    #[test]
    fn revocation_cannot_be_prepared_from_already_revoked_lifecycle() {
        let target = bootstrap();
        let head = SafetyProfileAuthorizationHead::from_transition(&target).unwrap();
        let already_revoked = SafetyProfileAuthorizationLifecycleState::revoked(
            head,
            arbitrary_revocation_digest(0xcc),
        )
        .unwrap();
        let inner = revocation_commit(&target);

        assert_eq!(
            SafetyProfileAuthorizationRevocationLifecycleCommitPreconditions::new(
                inner,
                already_revoked,
            ),
            Err(SafetyProfileAuthorizationLifecycleError::RevocationRequiresActiveHead)
        );
    }

    #[test]
    fn successor_head_racing_revocation_makes_revocation_stale() {
        let first = bootstrap();
        let second = successor(&first);
        let first_head = SafetyProfileAuthorizationHead::from_transition(&first).unwrap();
        let observed = SafetyProfileAuthorizationLifecycleState::active(first_head).unwrap();
        let inner = revocation_commit(&first);
        let preconditions =
            SafetyProfileAuthorizationRevocationLifecycleCommitPreconditions::new(
                inner,
                observed,
            )
            .unwrap();
        let successor_state = SafetyProfileAuthorizationLifecycleState::active(
            SafetyProfileAuthorizationHead::from_transition(&second).unwrap(),
        )
        .unwrap();

        assert_eq!(
            preconditions.recheck_commit_observation(
                &clock("secure-rtc-v1", 7, 1_550, 1_650),
                &root_snapshot(),
                &successor_state,
            ),
            Err(SafetyProfileAuthorizationLifecycleError::LifecycleStateChanged)
        );
    }
}
