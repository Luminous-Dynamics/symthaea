// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! TOCTOU-resistant commit preconditions for revocation-bound profile reinstatement.
//!
//! Reinstatement admission proves that one canonical restoration is locally admissible
//! at one observation. Detached cryptographic verification may happen afterwards and
//! may take time. During that interval the provisioned root, revoked lifecycle state,
//! or trusted-clock lineage may change. This module captures the exact admitted state
//! and requires it to remain current immediately before persistence.
//!
//! This module performs no persistence and proves no signature. A store must also
//! require verifier-owned proof over [`SafetyProfileAuthorizationReinstatementCommitPreconditions::canonical_reinstatement_bytes`]
//! under the exact captured root before atomically writing the candidate lifecycle
//! state and exact reinstatement marker.

use crate::commit_preconditions::ProfileAuthorityRootSnapshot;
use crate::lifecycle::SafetyProfileAuthorizationLifecycleState;
use crate::reinstatement::{
    SafetyProfileAuthorizationReinstatementDigest, SafetyProfileAuthorizationReinstatementError,
};
use crate::reinstatement_admission::PolicyCheckedSafetyProfileAuthorizationReinstatement;
use crate::trusted_time::TrustedAuthorizationClockObservation;
use thiserror::Error;

/// Result of rechecking one atomic reinstatement-store observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyProfileAuthorizationReinstatementCommitState {
    /// Root/lifecycle/clock state remains exactly the state checked during admission,
    /// and trusted time still proves the successor authorization is currently valid.
    /// The store may commit only after matching cryptographic proof as well.
    ReadyToCommit,
    /// The exact candidate head and exact reinstatement marker are already persisted.
    /// This acknowledges historical write completion only; it does not claim that
    /// the successor authorization remains active or executable now.
    AlreadyCommitted,
}

/// Exact non-cryptographic state that must remain true while one admitted
/// reinstatement is cryptographically verified and durably committed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetyProfileAuthorizationReinstatementCommitPreconditions {
    canonical_reinstatement_bytes: Vec<u8>,
    reinstatement_digest: SafetyProfileAuthorizationReinstatementDigest,
    expected_root: ProfileAuthorityRootSnapshot,
    expected_lifecycle: SafetyProfileAuthorizationLifecycleState,
    candidate_lifecycle: SafetyProfileAuthorizationLifecycleState,
    expected_clock_source_id: String,
    expected_clock_epoch: u64,
    valid_from_unix_ms: i64,
    valid_until_unix_ms: i64,
}

impl SafetyProfileAuthorizationReinstatementCommitPreconditions {
    /// Capture commit preconditions from one exact policy-checked reinstatement.
    pub fn from_policy_checked(
        checked: &PolicyCheckedSafetyProfileAuthorizationReinstatement,
    ) -> Result<Self, SafetyProfileAuthorizationReinstatementCommitError> {
        if !checked.expected_lifecycle().is_revoked() {
            return Err(
                SafetyProfileAuthorizationReinstatementCommitError::ExpectedLifecycleNotRevoked,
            );
        }
        if !checked.candidate_lifecycle().is_active() {
            return Err(
                SafetyProfileAuthorizationReinstatementCommitError::CandidateLifecycleNotActive,
            );
        }

        let expected_identity = checked
            .expected_lifecycle()
            .authorization_head()
            .identity()
            .ok_or(
                SafetyProfileAuthorizationReinstatementCommitError::ExpectedLifecycleHeadNotCurrent,
            )?;
        let candidate_identity = checked
            .candidate_lifecycle()
            .authorization_head()
            .identity()
            .ok_or(
                SafetyProfileAuthorizationReinstatementCommitError::CandidateLifecycleHeadNotCurrent,
            )?;

        if expected_identity.subject_node_id() != candidate_identity.subject_node_id() {
            return Err(
                SafetyProfileAuthorizationReinstatementCommitError::LifecycleNodeChanged,
            );
        }
        if expected_identity.authority_root_id() != checked.expected_root().root_id()
            || expected_identity.authority_root_digest() != checked.expected_root().root_digest()
        {
            return Err(
                SafetyProfileAuthorizationReinstatementCommitError::ExpectedLifecycleRootMismatch,
            );
        }
        if candidate_identity.authority_root_id() != checked.expected_root().root_id()
            || candidate_identity.authority_root_digest() != checked.expected_root().root_digest()
        {
            return Err(
                SafetyProfileAuthorizationReinstatementCommitError::CandidateLifecycleRootMismatch,
            );
        }

        let expected_revocation_digest = checked
            .expected_lifecycle()
            .revocation_digest()
            .ok_or(
                SafetyProfileAuthorizationReinstatementCommitError::ExpectedLifecycleNotRevoked,
            )?;
        let observed_revocation_digest = checked.reinstatement().revocation_digest()?;
        if observed_revocation_digest != expected_revocation_digest {
            return Err(
                SafetyProfileAuthorizationReinstatementCommitError::RevocationDigestMismatch,
            );
        }

        let successor = checked.reinstatement().successor().subject();
        let reinstatement_digest = checked.reinstatement().reinstatement_digest()?;

        Ok(Self {
            canonical_reinstatement_bytes: checked.canonical_reinstatement_bytes().to_vec(),
            reinstatement_digest,
            expected_root: checked.expected_root().clone(),
            expected_lifecycle: checked.expected_lifecycle().clone(),
            candidate_lifecycle: checked.candidate_lifecycle().clone(),
            expected_clock_source_id: checked.expected_clock_source_id().to_owned(),
            expected_clock_epoch: checked.expected_clock_epoch(),
            valid_from_unix_ms: successor.valid_from_unix_ms(),
            valid_until_unix_ms: successor.valid_until_unix_ms(),
        })
    }

    pub fn canonical_reinstatement_bytes(&self) -> &[u8] {
        &self.canonical_reinstatement_bytes
    }

    pub fn reinstatement_digest(&self) -> SafetyProfileAuthorizationReinstatementDigest {
        self.reinstatement_digest
    }

    pub fn expected_root(&self) -> &ProfileAuthorityRootSnapshot {
        &self.expected_root
    }

    pub fn expected_lifecycle(&self) -> &SafetyProfileAuthorizationLifecycleState {
        &self.expected_lifecycle
    }

    pub fn candidate_lifecycle(&self) -> &SafetyProfileAuthorizationLifecycleState {
        &self.candidate_lifecycle
    }

    pub fn expected_clock_source_id(&self) -> &str {
        &self.expected_clock_source_id
    }

    pub fn expected_clock_epoch(&self) -> u64 {
        self.expected_clock_epoch
    }

    pub fn valid_from_unix_ms(&self) -> i64 {
        self.valid_from_unix_ms
    }

    pub fn valid_until_unix_ms(&self) -> i64 {
        self.valid_until_unix_ms
    }

    /// Recheck one atomic lifecycle + reinstatement-marker store observation.
    ///
    /// `current_candidate_reinstatement_digest` is the marker stored for the exact
    /// candidate successor head, if any. A successful commit must atomically write
    /// both the candidate lifecycle and this exact digest. This prevents a retry
    /// from treating a different reinstatement artifact that leads to the same
    /// successor head as idempotently equivalent.
    pub fn recheck_commit_observation(
        &self,
        current_clock: &TrustedAuthorizationClockObservation,
        current_root: &ProfileAuthorityRootSnapshot,
        current_lifecycle: &SafetyProfileAuthorizationLifecycleState,
        current_candidate_reinstatement_digest: Option<
            SafetyProfileAuthorizationReinstatementDigest,
        >,
    ) -> Result<
        SafetyProfileAuthorizationReinstatementCommitState,
        SafetyProfileAuthorizationReinstatementCommitError,
    > {
        if current_lifecycle.authorization_head()
            == self.candidate_lifecycle.authorization_head()
        {
            return match current_candidate_reinstatement_digest {
                Some(observed) if observed == self.reinstatement_digest => {
                    Ok(SafetyProfileAuthorizationReinstatementCommitState::AlreadyCommitted)
                }
                Some(observed) => Err(
                    SafetyProfileAuthorizationReinstatementCommitError::DifferentReinstatementAlreadyCommitted {
                        expected: self.reinstatement_digest,
                        observed,
                    },
                ),
                None => Err(
                    SafetyProfileAuthorizationReinstatementCommitError::CandidateHeadMissingReinstatementMarker,
                ),
            };
        }

        if let Some(observed) = current_candidate_reinstatement_digest {
            return Err(
                SafetyProfileAuthorizationReinstatementCommitError::OrphanReinstatementMarker {
                    observed,
                },
            );
        }

        if current_root != &self.expected_root {
            return Err(
                SafetyProfileAuthorizationReinstatementCommitError::AuthorityRootSnapshotChanged,
            );
        }
        if current_lifecycle != &self.expected_lifecycle {
            return Err(
                SafetyProfileAuthorizationReinstatementCommitError::LifecycleStateChanged,
            );
        }
        if current_clock.source_id() != self.expected_clock_source_id
            || current_clock.epoch() != self.expected_clock_epoch
        {
            return Err(
                SafetyProfileAuthorizationReinstatementCommitError::ClockLineageChanged {
                    expected_source_id: self.expected_clock_source_id.clone(),
                    expected_epoch: self.expected_clock_epoch,
                    observed_source_id: current_clock.source_id().to_owned(),
                    observed_epoch: current_clock.epoch(),
                },
            );
        }

        validate_entire_interval(
            self.valid_from_unix_ms,
            self.valid_until_unix_ms,
            current_clock,
        )?;

        Ok(SafetyProfileAuthorizationReinstatementCommitState::ReadyToCommit)
    }
}

fn validate_entire_interval(
    valid_from_unix_ms: i64,
    valid_until_unix_ms: i64,
    clock: &TrustedAuthorizationClockObservation,
) -> Result<(), SafetyProfileAuthorizationReinstatementCommitError> {
    if clock.latest_unix_ms() < valid_from_unix_ms {
        return Err(
            SafetyProfileAuthorizationReinstatementCommitError::DefinitelyNotYetValid {
                latest_unix_ms: clock.latest_unix_ms(),
                valid_from_unix_ms,
            },
        );
    }
    if clock.earliest_unix_ms() >= valid_until_unix_ms {
        return Err(
            SafetyProfileAuthorizationReinstatementCommitError::DefinitelyExpired {
                earliest_unix_ms: clock.earliest_unix_ms(),
                valid_until_unix_ms,
            },
        );
    }
    if clock.earliest_unix_ms() < valid_from_unix_ms
        || clock.latest_unix_ms() >= valid_until_unix_ms
    {
        return Err(
            SafetyProfileAuthorizationReinstatementCommitError::UncertaintyCrossesValidityBoundary {
                earliest_unix_ms: clock.earliest_unix_ms(),
                latest_unix_ms: clock.latest_unix_ms(),
                valid_from_unix_ms,
                valid_until_unix_ms,
            },
        );
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SafetyProfileAuthorizationReinstatementCommitError {
    #[error(transparent)]
    Reinstatement(#[from] SafetyProfileAuthorizationReinstatementError),
    #[error("policy-checked reinstatement expected lifecycle is not revoked")]
    ExpectedLifecycleNotRevoked,
    #[error("policy-checked reinstatement candidate lifecycle is not active")]
    CandidateLifecycleNotActive,
    #[error("policy-checked reinstatement expected lifecycle has no current authorization head")]
    ExpectedLifecycleHeadNotCurrent,
    #[error("policy-checked reinstatement candidate lifecycle has no current authorization head")]
    CandidateLifecycleHeadNotCurrent,
    #[error("profile-authorization node changed across reinstatement lifecycle")]
    LifecycleNodeChanged,
    #[error("revoked predecessor lifecycle does not belong to the expected root snapshot")]
    ExpectedLifecycleRootMismatch,
    #[error("candidate lifecycle does not belong to the expected root snapshot")]
    CandidateLifecycleRootMismatch,
    #[error("reinstatement revocation digest does not match the expected revoked lifecycle")]
    RevocationDigestMismatch,
    #[error("profile-authority root snapshot changed before reinstatement commit")]
    AuthorityRootSnapshotChanged,
    #[error("profile-authorization lifecycle state changed before reinstatement commit")]
    LifecycleStateChanged,
    #[error("trusted clock lineage changed before reinstatement commit: expected {expected_source_id}@{expected_epoch}, observed {observed_source_id}@{observed_epoch}")]
    ClockLineageChanged {
        expected_source_id: String,
        expected_epoch: u64,
        observed_source_id: String,
        observed_epoch: u64,
    },
    #[error("reinstatement successor is definitely not yet valid: latest possible time {latest_unix_ms} < valid-from {valid_from_unix_ms}")]
    DefinitelyNotYetValid {
        latest_unix_ms: i64,
        valid_from_unix_ms: i64,
    },
    #[error("reinstatement successor is definitely expired: earliest possible time {earliest_unix_ms} >= valid-until {valid_until_unix_ms}")]
    DefinitelyExpired {
        earliest_unix_ms: i64,
        valid_until_unix_ms: i64,
    },
    #[error("trusted clock uncertainty [{earliest_unix_ms}, {latest_unix_ms}] crosses reinstatement successor validity [{valid_from_unix_ms}, {valid_until_unix_ms})")]
    UncertaintyCrossesValidityBoundary {
        earliest_unix_ms: i64,
        latest_unix_ms: i64,
        valid_from_unix_ms: i64,
        valid_until_unix_ms: i64,
    },
    #[error("candidate successor head is persisted without an exact reinstatement marker")]
    CandidateHeadMissingReinstatementMarker,
    #[error("a different reinstatement marker is already committed for the candidate successor head")]
    DifferentReinstatementAlreadyCommitted {
        expected: SafetyProfileAuthorizationReinstatementDigest,
        observed: SafetyProfileAuthorizationReinstatementDigest,
    },
    #[error("a reinstatement marker exists for the candidate successor head while that head is not current")]
    OrphanReinstatementMarker {
        observed: SafetyProfileAuthorizationReinstatementDigest,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::authorization::{ProfileAuthorityRootDigest, SafetyProfileAuthorizationSubject};
    use crate::reinstatement::SafetyProfileAuthorizationReinstatementTransition;
    use crate::reinstatement_admission::SafetyProfileAuthorizationReinstatementAdmissionPolicy;
    use crate::revocation::SafetyProfileAuthorizationRevocationSubject;
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

    fn subject(id: &str, generation: u64, valid_from: i64, valid_until: i64) -> SafetyProfileAuthorizationSubject {
        SafetyProfileAuthorizationSubject::new(
            id,
            "facility-profile-root-v1",
            root(0x33),
            "compute-campus",
            generation,
            valid_from,
            valid_until,
            &profile(),
        )
        .unwrap()
    }

    fn clock(source: &str, epoch: u64, earliest: i64, latest: i64) -> TrustedAuthorizationClockObservation {
        TrustedAuthorizationClockObservation::new(source, epoch, earliest, latest).unwrap()
    }

    fn root_snapshot() -> ProfileAuthorityRootSnapshot {
        ProfileAuthorityRootSnapshot::new("facility-profile-root-v1", root(0x33), 4).unwrap()
    }

    fn checked() -> PolicyCheckedSafetyProfileAuthorizationReinstatement {
        let predecessor =
            SafetyProfileAuthorizationTransition::bootstrap(subject("auth-1", 1, 1_000, 3_000))
                .unwrap();
        let revocation = SafetyProfileAuthorizationRevocationSubject::for_transition(
            "revoke-1",
            &predecessor,
            1_400,
        )
        .unwrap();
        let successor = SafetyProfileAuthorizationTransition::successor(
            subject("auth-2", 2, 1_400, 3_000),
            &predecessor,
        )
        .unwrap();
        let reinstatement = SafetyProfileAuthorizationReinstatementTransition::new(
            "reinstate-1",
            &predecessor,
            &revocation,
            &successor,
        )
        .unwrap();
        let revoked_lifecycle = SafetyProfileAuthorizationLifecycleState::revoked(
            crate::admission::SafetyProfileAuthorizationHead::from_transition(&predecessor)
                .unwrap(),
            revocation.revocation_digest().unwrap(),
        )
        .unwrap();
        let policy = SafetyProfileAuthorizationReinstatementAdmissionPolicy::new(
            "compute-campus",
            root_snapshot(),
            revoked_lifecycle,
        )
        .unwrap();
        policy
            .check(
                &clock("secure-rtc-v1", 7, 1_500, 1_600),
                &reinstatement,
                &profile(),
            )
            .unwrap()
    }

    fn other_reinstatement_digest(byte: u8) -> SafetyProfileAuthorizationReinstatementDigest {
        SafetyProfileAuthorizationReinstatementDigest::Blake3_256([byte; 32])
    }

    #[test]
    fn unchanged_atomic_observation_is_ready_to_commit() {
        let checked = checked();
        let preconditions =
            SafetyProfileAuthorizationReinstatementCommitPreconditions::from_policy_checked(
                &checked,
            )
            .unwrap();

        assert_eq!(
            preconditions
                .recheck_commit_observation(
                    &clock("secure-rtc-v1", 7, 1_550, 1_650),
                    checked.expected_root(),
                    checked.expected_lifecycle(),
                    None,
                )
                .unwrap(),
            SafetyProfileAuthorizationReinstatementCommitState::ReadyToCommit
        );
        assert_eq!(
            preconditions.canonical_reinstatement_bytes(),
            checked.canonical_reinstatement_bytes()
        );
    }

    #[test]
    fn exact_prior_write_recovers_only_with_exact_marker() {
        let checked = checked();
        let preconditions =
            SafetyProfileAuthorizationReinstatementCommitPreconditions::from_policy_checked(
                &checked,
            )
            .unwrap();

        assert_eq!(
            preconditions
                .recheck_commit_observation(
                    &clock("different-clock", 99, 9_000, 9_100),
                    &ProfileAuthorityRootSnapshot::new(
                        "different-root",
                        root(0xaa),
                        99,
                    )
                    .unwrap(),
                    checked.candidate_lifecycle(),
                    Some(preconditions.reinstatement_digest()),
                )
                .unwrap(),
            SafetyProfileAuthorizationReinstatementCommitState::AlreadyCommitted
        );

        assert_eq!(
            preconditions.recheck_commit_observation(
                &clock("secure-rtc-v1", 7, 1_500, 1_600),
                checked.expected_root(),
                checked.candidate_lifecycle(),
                None,
            ),
            Err(
                SafetyProfileAuthorizationReinstatementCommitError::CandidateHeadMissingReinstatementMarker
            )
        );
    }

    #[test]
    fn different_marker_for_candidate_head_is_not_idempotent() {
        let checked = checked();
        let preconditions =
            SafetyProfileAuthorizationReinstatementCommitPreconditions::from_policy_checked(
                &checked,
            )
            .unwrap();
        let observed = other_reinstatement_digest(0xee);

        assert_eq!(
            preconditions.recheck_commit_observation(
                &clock("secure-rtc-v1", 7, 1_500, 1_600),
                checked.expected_root(),
                checked.candidate_lifecycle(),
                Some(observed),
            ),
            Err(
                SafetyProfileAuthorizationReinstatementCommitError::DifferentReinstatementAlreadyCommitted {
                    expected: preconditions.reinstatement_digest(),
                    observed,
                }
            )
        );
    }

    #[test]
    fn lifecycle_root_and_clock_races_fail_closed() {
        let checked = checked();
        let preconditions =
            SafetyProfileAuthorizationReinstatementCommitPreconditions::from_policy_checked(
                &checked,
            )
            .unwrap();

        assert_eq!(
            preconditions.recheck_commit_observation(
                &clock("secure-rtc-v1", 7, 1_500, 1_600),
                &ProfileAuthorityRootSnapshot::new(
                    "facility-profile-root-v1",
                    root(0x33),
                    5,
                )
                .unwrap(),
                checked.expected_lifecycle(),
                None,
            ),
            Err(
                SafetyProfileAuthorizationReinstatementCommitError::AuthorityRootSnapshotChanged
            )
        );

        let active_predecessor = SafetyProfileAuthorizationLifecycleState::active(
            checked.expected_lifecycle().authorization_head().clone(),
        )
        .unwrap();
        assert_eq!(
            preconditions.recheck_commit_observation(
                &clock("secure-rtc-v1", 7, 1_500, 1_600),
                checked.expected_root(),
                &active_predecessor,
                None,
            ),
            Err(SafetyProfileAuthorizationReinstatementCommitError::LifecycleStateChanged)
        );

        assert!(matches!(
            preconditions.recheck_commit_observation(
                &clock("secure-rtc-v2", 8, 1_500, 1_600),
                checked.expected_root(),
                checked.expected_lifecycle(),
                None,
            ),
            Err(SafetyProfileAuthorizationReinstatementCommitError::ClockLineageChanged { .. })
        ));
    }

    #[test]
    fn uncertainty_crossing_successor_validity_fails_closed() {
        let checked = checked();
        let preconditions =
            SafetyProfileAuthorizationReinstatementCommitPreconditions::from_policy_checked(
                &checked,
            )
            .unwrap();

        assert!(matches!(
            preconditions.recheck_commit_observation(
                &clock("secure-rtc-v1", 7, 1_300, 1_500),
                checked.expected_root(),
                checked.expected_lifecycle(),
                None,
            ),
            Err(
                SafetyProfileAuthorizationReinstatementCommitError::UncertaintyCrossesValidityBoundary { .. }
            )
        ));
    }

    #[test]
    fn marker_without_candidate_head_is_rejected_as_orphaned() {
        let checked = checked();
        let preconditions =
            SafetyProfileAuthorizationReinstatementCommitPreconditions::from_policy_checked(
                &checked,
            )
            .unwrap();
        let observed = preconditions.reinstatement_digest();

        assert_eq!(
            preconditions.recheck_commit_observation(
                &clock("secure-rtc-v1", 7, 1_500, 1_600),
                checked.expected_root(),
                checked.expected_lifecycle(),
                Some(observed),
            ),
            Err(
                SafetyProfileAuthorizationReinstatementCommitError::OrphanReinstatementMarker {
                    observed,
                }
            )
        );
    }
}
