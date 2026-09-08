// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! TOCTOU-resistant commit preconditions for profile-authorization revocation.
//!
//! Revocation admission proves that one raw revocation is locally admissible at one
//! observation. Cryptographic verification may happen afterwards and may take time.
//! A persistent revoked-head marker must therefore recheck the exact root snapshot,
//! exact authorization head, trusted-clock lineage, and effective-time boundary at
//! the eventual write.
//!
//! This module performs no persistence and claims no cryptographic verification. A
//! storage adapter must additionally require verifier-owned proof over
//! [`SafetyProfileAuthorizationRevocationCommitPreconditions::canonical_revocation_bytes`]
//! under the exact root captured here before writing a revocation marker.

use crate::admission::SafetyProfileAuthorizationHead;
use crate::commit_preconditions::ProfileAuthorityRootSnapshot;
use crate::revocation::SafetyProfileAuthorizationRevocationDigest;
use crate::revocation_admission::PolicyCheckedSafetyProfileAuthorizationRevocation;
use crate::trusted_time::TrustedAuthorizationClockObservation;
use thiserror::Error;

/// Result of rechecking one atomic revocation-store observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyProfileAuthorizationRevocationCommitState {
    /// Root/head/clock state remains exactly the state checked during admission and
    /// trusted time proves the revocation is definitely effective. A store may
    /// write the exact revocation marker only after cryptographic proof is also
    /// matched to the canonical revocation bytes.
    ReadyToCommit,
    /// The exact revocation digest is already persisted. This acknowledges an
    /// uncertain prior write only; it is not a fresh statement about current
    /// profile authority or execution eligibility.
    AlreadyCommitted,
}

/// Exact non-cryptographic state that must remain true while one admitted profile
/// revocation is cryptographically verified and durably committed.
///
/// The object is non-serializable and derives every field from a policy-checked
/// revocation. It therefore cannot be assembled by independently pairing a target
/// head from one admission with root/time metadata from another.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetyProfileAuthorizationRevocationCommitPreconditions {
    canonical_revocation_bytes: Vec<u8>,
    revocation_digest: SafetyProfileAuthorizationRevocationDigest,
    expected_root: ProfileAuthorityRootSnapshot,
    expected_current_head: SafetyProfileAuthorizationHead,
    expected_clock_source_id: String,
    expected_clock_epoch: u64,
    effective_from_unix_ms: i64,
}

impl SafetyProfileAuthorizationRevocationCommitPreconditions {
    /// Capture commit preconditions from one exact policy-checked revocation.
    pub fn from_policy_checked(
        checked: &PolicyCheckedSafetyProfileAuthorizationRevocation,
    ) -> Result<Self, SafetyProfileAuthorizationRevocationCommitError> {
        let identity = checked.expected_current_head().identity().ok_or(
            SafetyProfileAuthorizationRevocationCommitError::ExpectedHeadNotCurrent,
        )?;
        let revocation = checked.revocation();

        // These relationships were already established by revocation admission,
        // but reassert them when producing the commit capsule so an inconsistent
        // future policy result cannot cross this boundary silently.
        if identity.subject_node_id() != revocation.subject_node_id() {
            return Err(
                SafetyProfileAuthorizationRevocationCommitError::ExpectedHeadNodeMismatch {
                    expected: revocation.subject_node_id().to_owned(),
                    observed: identity.subject_node_id().to_owned(),
                },
            );
        }
        if identity.authority_root_id() != checked.expected_root().root_id()
            || identity.authority_root_digest() != checked.expected_root().root_digest()
        {
            return Err(
                SafetyProfileAuthorizationRevocationCommitError::ExpectedHeadRootMismatch,
            );
        }
        if revocation.authority_root_id() != checked.expected_root().root_id()
            || revocation.authority_root_digest() != checked.expected_root().root_digest()
        {
            return Err(
                SafetyProfileAuthorizationRevocationCommitError::RevocationRootMismatch,
            );
        }

        Ok(Self {
            canonical_revocation_bytes: checked.canonical_revocation_bytes().to_vec(),
            revocation_digest: checked.revocation_digest(),
            expected_root: checked.expected_root().clone(),
            expected_current_head: checked.expected_current_head().clone(),
            expected_clock_source_id: checked.expected_clock_source_id().to_owned(),
            expected_clock_epoch: checked.expected_clock_epoch(),
            effective_from_unix_ms: revocation.effective_from_unix_ms(),
        })
    }

    pub fn canonical_revocation_bytes(&self) -> &[u8] {
        &self.canonical_revocation_bytes
    }

    pub fn revocation_digest(&self) -> SafetyProfileAuthorizationRevocationDigest {
        self.revocation_digest
    }

    pub fn expected_root(&self) -> &ProfileAuthorityRootSnapshot {
        &self.expected_root
    }

    pub fn expected_current_head(&self) -> &SafetyProfileAuthorizationHead {
        &self.expected_current_head
    }

    pub fn expected_clock_source_id(&self) -> &str {
        &self.expected_clock_source_id
    }

    pub fn expected_clock_epoch(&self) -> u64 {
        self.expected_clock_epoch
    }

    pub fn effective_from_unix_ms(&self) -> i64 {
        self.effective_from_unix_ms
    }

    /// Recheck the exact root/head/clock observation immediately before one
    /// persistent revocation write.
    ///
    /// `current_revocation_digest` is the store's current revoked-head marker for
    /// the target head, if any. Exact replay is idempotent. A different marker is
    /// surfaced as conflict rather than silently overwritten.
    pub fn recheck_commit_observation(
        &self,
        current_clock: &TrustedAuthorizationClockObservation,
        current_root: &ProfileAuthorityRootSnapshot,
        current_head: &SafetyProfileAuthorizationHead,
        current_revocation_digest: Option<SafetyProfileAuthorizationRevocationDigest>,
    ) -> Result<
        SafetyProfileAuthorizationRevocationCommitState,
        SafetyProfileAuthorizationRevocationCommitError,
    > {
        if current_revocation_digest == Some(self.revocation_digest) {
            return Ok(SafetyProfileAuthorizationRevocationCommitState::AlreadyCommitted);
        }
        if let Some(observed) = current_revocation_digest {
            return Err(
                SafetyProfileAuthorizationRevocationCommitError::DifferentRevocationAlreadyCommitted {
                    expected: self.revocation_digest,
                    observed,
                },
            );
        }

        if current_root != &self.expected_root {
            return Err(
                SafetyProfileAuthorizationRevocationCommitError::AuthorityRootSnapshotChanged,
            );
        }
        if current_head != &self.expected_current_head {
            return Err(
                SafetyProfileAuthorizationRevocationCommitError::AuthorizationHeadChanged,
            );
        }
        if current_clock.source_id() != self.expected_clock_source_id
            || current_clock.epoch() != self.expected_clock_epoch
        {
            return Err(
                SafetyProfileAuthorizationRevocationCommitError::ClockLineageChanged {
                    expected_source_id: self.expected_clock_source_id.clone(),
                    expected_epoch: self.expected_clock_epoch,
                    observed_source_id: current_clock.source_id().to_owned(),
                    observed_epoch: current_clock.epoch(),
                },
            );
        }

        // Revocation is effective only when the complete uncertainty interval lies
        // at or after the authenticated effective-from boundary.
        if current_clock.latest_unix_ms() < self.effective_from_unix_ms {
            return Err(
                SafetyProfileAuthorizationRevocationCommitError::DefinitelyNotYetEffective {
                    latest_unix_ms: current_clock.latest_unix_ms(),
                    effective_from_unix_ms: self.effective_from_unix_ms,
                },
            );
        }
        if current_clock.earliest_unix_ms() < self.effective_from_unix_ms {
            return Err(
                SafetyProfileAuthorizationRevocationCommitError::UncertaintyCrossesEffectiveBoundary {
                    earliest_unix_ms: current_clock.earliest_unix_ms(),
                    latest_unix_ms: current_clock.latest_unix_ms(),
                    effective_from_unix_ms: self.effective_from_unix_ms,
                },
            );
        }

        Ok(SafetyProfileAuthorizationRevocationCommitState::ReadyToCommit)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SafetyProfileAuthorizationRevocationCommitError {
    #[error("policy-checked revocation unexpectedly has no current authorization head")]
    ExpectedHeadNotCurrent,
    #[error("policy-checked revocation head belongs to node {observed}, expected {expected}")]
    ExpectedHeadNodeMismatch { expected: String, observed: String },
    #[error("policy-checked revocation head does not belong to the expected root snapshot")]
    ExpectedHeadRootMismatch,
    #[error("policy-checked revocation does not belong to the expected root snapshot")]
    RevocationRootMismatch,
    #[error("profile-authority root snapshot changed before revocation commit")]
    AuthorityRootSnapshotChanged,
    #[error("profile-authorization head changed before revocation commit")]
    AuthorizationHeadChanged,
    #[error("trusted clock lineage changed before revocation commit: expected {expected_source_id}@{expected_epoch}, observed {observed_source_id}@{observed_epoch}")]
    ClockLineageChanged {
        expected_source_id: String,
        expected_epoch: u64,
        observed_source_id: String,
        observed_epoch: u64,
    },
    #[error("revocation is definitely not yet effective at commit: latest possible time {latest_unix_ms} < effective-from {effective_from_unix_ms}")]
    DefinitelyNotYetEffective {
        latest_unix_ms: i64,
        effective_from_unix_ms: i64,
    },
    #[error("trusted clock uncertainty [{earliest_unix_ms}, {latest_unix_ms}] crosses revocation effective-from {effective_from_unix_ms}")]
    UncertaintyCrossesEffectiveBoundary {
        earliest_unix_ms: i64,
        latest_unix_ms: i64,
        effective_from_unix_ms: i64,
    },
    #[error("a different revocation marker is already committed for this authorization head")]
    DifferentRevocationAlreadyCommitted {
        expected: SafetyProfileAuthorizationRevocationDigest,
        observed: SafetyProfileAuthorizationRevocationDigest,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admission::SafetyProfileAuthorizationHead;
    use crate::authorization::{ProfileAuthorityRootDigest, SafetyProfileAuthorizationSubject};
    use crate::commit_preconditions::ProfileAuthorityRootSnapshot;
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

    fn transition(id: &str) -> SafetyProfileAuthorizationTransition {
        let profile = profile();
        let subject = SafetyProfileAuthorizationSubject::new(
            id,
            "facility-profile-root-v1",
            root(0x33),
            "compute-campus",
            1,
            1_000,
            2_000,
            &profile,
        )
        .unwrap();
        SafetyProfileAuthorizationTransition::bootstrap(subject).unwrap()
    }

    fn clock(
        source: &str,
        epoch: u64,
        earliest: i64,
        latest: i64,
    ) -> TrustedAuthorizationClockObservation {
        TrustedAuthorizationClockObservation::new(source, epoch, earliest, latest).unwrap()
    }

    fn checked_revocation(
        target: &SafetyProfileAuthorizationTransition,
        effective_from: i64,
    ) -> PolicyCheckedSafetyProfileAuthorizationRevocation {
        let head = SafetyProfileAuthorizationHead::from_transition(target).unwrap();
        let root_snapshot =
            ProfileAuthorityRootSnapshot::new("facility-profile-root-v1", root(0x33), 4).unwrap();
        let policy = SafetyProfileAuthorizationRevocationAdmissionPolicy::new(
            "compute-campus",
            root_snapshot,
            head,
        )
        .unwrap();
        let revocation = SafetyProfileAuthorizationRevocationSubject::for_transition(
            "revoke-a",
            target,
            effective_from,
        )
        .unwrap();
        policy
            .check(&clock("secure-rtc-v1", 7, 1_500, 1_600), &revocation)
            .unwrap()
    }

    #[test]
    fn unchanged_atomic_observation_is_ready_to_commit() {
        let target = transition("auth-a");
        let checked = checked_revocation(&target, 1_400);
        let preconditions =
            SafetyProfileAuthorizationRevocationCommitPreconditions::from_policy_checked(&checked)
                .unwrap();

        assert_eq!(
            preconditions
                .recheck_commit_observation(
                    &clock("secure-rtc-v1", 7, 1_550, 1_650),
                    checked.expected_root(),
                    checked.expected_current_head(),
                    None,
                )
                .unwrap(),
            SafetyProfileAuthorizationRevocationCommitState::ReadyToCommit
        );
        assert_eq!(
            preconditions.canonical_revocation_bytes(),
            checked.canonical_revocation_bytes()
        );
    }

    #[test]
    fn exact_prior_write_recovers_idempotently_even_after_later_state_change() {
        let target = transition("auth-a");
        let checked = checked_revocation(&target, 1_400);
        let preconditions =
            SafetyProfileAuthorizationRevocationCommitPreconditions::from_policy_checked(&checked)
                .unwrap();
        let different_root =
            ProfileAuthorityRootSnapshot::new("new-root", root(0x44), 9).unwrap();

        assert_eq!(
            preconditions
                .recheck_commit_observation(
                    &clock("different-clock", 99, 0, 1),
                    &different_root,
                    &SafetyProfileAuthorizationHead::Uninitialized,
                    Some(preconditions.revocation_digest()),
                )
                .unwrap(),
            SafetyProfileAuthorizationRevocationCommitState::AlreadyCommitted
        );
    }

    #[test]
    fn different_existing_revocation_marker_is_not_overwritten() {
        let target = transition("auth-a");
        let checked = checked_revocation(&target, 1_400);
        let preconditions =
            SafetyProfileAuthorizationRevocationCommitPreconditions::from_policy_checked(&checked)
                .unwrap();
        let other = SafetyProfileAuthorizationRevocationDigest::Blake3_256([0xee; 32]);

        assert!(matches!(
            preconditions.recheck_commit_observation(
                &clock("secure-rtc-v1", 7, 1_550, 1_650),
                checked.expected_root(),
                checked.expected_current_head(),
                Some(other),
            ),
            Err(SafetyProfileAuthorizationRevocationCommitError::DifferentRevocationAlreadyCommitted { .. })
        ));
    }

    #[test]
    fn root_change_invalidates_in_flight_revocation() {
        let target = transition("auth-a");
        let checked = checked_revocation(&target, 1_400);
        let preconditions =
            SafetyProfileAuthorizationRevocationCommitPreconditions::from_policy_checked(&checked)
                .unwrap();
        let changed_root =
            ProfileAuthorityRootSnapshot::new("facility-profile-root-v1", root(0x33), 5).unwrap();

        assert_eq!(
            preconditions.recheck_commit_observation(
                &clock("secure-rtc-v1", 7, 1_550, 1_650),
                &changed_root,
                checked.expected_current_head(),
                None,
            ),
            Err(SafetyProfileAuthorizationRevocationCommitError::AuthorityRootSnapshotChanged)
        );
    }

    #[test]
    fn head_change_invalidates_in_flight_revocation() {
        let target = transition("auth-a");
        let other = transition("auth-b");
        let checked = checked_revocation(&target, 1_400);
        let preconditions =
            SafetyProfileAuthorizationRevocationCommitPreconditions::from_policy_checked(&checked)
                .unwrap();
        let other_head = SafetyProfileAuthorizationHead::from_transition(&other).unwrap();

        assert_eq!(
            preconditions.recheck_commit_observation(
                &clock("secure-rtc-v1", 7, 1_550, 1_650),
                checked.expected_root(),
                &other_head,
                None,
            ),
            Err(SafetyProfileAuthorizationRevocationCommitError::AuthorizationHeadChanged)
        );
    }

    #[test]
    fn clock_lineage_change_invalidates_in_flight_revocation() {
        let target = transition("auth-a");
        let checked = checked_revocation(&target, 1_400);
        let preconditions =
            SafetyProfileAuthorizationRevocationCommitPreconditions::from_policy_checked(&checked)
                .unwrap();

        assert!(matches!(
            preconditions.recheck_commit_observation(
                &clock("secure-rtc-v2", 8, 1_550, 1_650),
                checked.expected_root(),
                checked.expected_current_head(),
                None,
            ),
            Err(SafetyProfileAuthorizationRevocationCommitError::ClockLineageChanged { .. })
        ));
    }

    #[test]
    fn uncertainty_crossing_effective_boundary_fails_closed_at_commit() {
        let target = transition("auth-a");
        let checked = checked_revocation(&target, 1_400);
        let preconditions =
            SafetyProfileAuthorizationRevocationCommitPreconditions::from_policy_checked(&checked)
                .unwrap();

        assert!(matches!(
            preconditions.recheck_commit_observation(
                &clock("secure-rtc-v1", 7, 1_350, 1_450),
                checked.expected_root(),
                checked.expected_current_head(),
                None,
            ),
            Err(SafetyProfileAuthorizationRevocationCommitError::UncertaintyCrossesEffectiveBoundary { .. })
        ));
    }
}
