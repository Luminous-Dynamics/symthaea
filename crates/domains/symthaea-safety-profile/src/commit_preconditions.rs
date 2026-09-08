// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! TOCTOU-resistant commit preconditions for safety-profile authorization.
//!
//! Policy admission and cryptographic verification may take time. Neither should
//! authorize a stale write if the trusted authority root, persisted authorization
//! head, or validity window changes before persistence. This module captures the
//! exact state that a future atomic store must compare at commit time.
//!
//! Nothing in this module performs persistence or claims cryptographic verification.

use crate::admission::{
    PolicyCheckedSafetyProfileAuthorization, SafetyProfileAuthorizationHead,
};
use crate::authorization::ProfileAuthorityRootDigest;
use thiserror::Error;

/// Exact externally provisioned profile-authority root state observed locally.
///
/// `epoch` is a local monotone provisioning generation. It is deliberately
/// separate from profile-authorization generations: reprovisioning the same key
/// under a new root epoch must still invalidate an in-flight authorization commit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileAuthorityRootSnapshot {
    root_id: String,
    root_digest: ProfileAuthorityRootDigest,
    epoch: u64,
}

impl ProfileAuthorityRootSnapshot {
    pub fn new(
        root_id: impl Into<String>,
        root_digest: ProfileAuthorityRootDigest,
        epoch: u64,
    ) -> Result<Self, SafetyProfileAuthorizationCommitError> {
        let root_id = root_id.into();
        if root_id.trim().is_empty() {
            return Err(SafetyProfileAuthorizationCommitError::EmptyAuthorityRootId);
        }
        if epoch == 0 {
            return Err(SafetyProfileAuthorizationCommitError::ZeroAuthorityRootEpoch);
        }
        Ok(Self {
            root_id,
            root_digest,
            epoch,
        })
    }

    pub fn root_id(&self) -> &str {
        &self.root_id
    }

    pub fn root_digest(&self) -> ProfileAuthorityRootDigest {
        self.root_digest
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }
}

/// Result of comparing commit preconditions with one atomic store observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyProfileAuthorizationCommitState {
    /// The expected predecessor/root are still current and the candidate remains
    /// inside its validity interval. A store may atomically replace the expected
    /// predecessor with the candidate, provided cryptographic proof has separately
    /// been established for the exact transition bytes.
    ReadyToCommit,
    /// The exact candidate head is already persisted. This is an idempotent
    /// recovery result for an uncertain prior write, not a claim that the
    /// authorization remains active or executable at the observation time.
    AlreadyCommitted,
}

/// Exact non-cryptographic state that must remain true when a checked profile
/// authorization is persisted.
///
/// A future persistence adapter should use this object inside one transaction/CAS
/// with its root and head reads. A separate verifier-owned proof must still match
/// the exact transition bytes before the store is allowed to write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetyProfileAuthorizationCommitPreconditions {
    expected_subject_node_id: String,
    expected_root: ProfileAuthorityRootSnapshot,
    expected_predecessor_head: SafetyProfileAuthorizationHead,
    candidate_head: SafetyProfileAuthorizationHead,
    valid_from_unix_ms: i64,
    valid_until_unix_ms: i64,
}

impl SafetyProfileAuthorizationCommitPreconditions {
    /// Capture commit preconditions from one policy-checked transition and one
    /// externally trusted root snapshot.
    ///
    /// This closes a gap deliberately left outside pure admission: the subject's
    /// signed `authority_root_id` must identify the same provisioned root as the
    /// digest, and an existing persisted head must belong to that same root ID.
    pub fn from_policy_checked(
        checked: &PolicyCheckedSafetyProfileAuthorization,
        expected_root: ProfileAuthorityRootSnapshot,
    ) -> Result<Self, SafetyProfileAuthorizationCommitError> {
        let subject = checked.transition().subject();

        if subject.authority_root_id() != expected_root.root_id() {
            return Err(SafetyProfileAuthorizationCommitError::AuthorityRootIdMismatch {
                expected: expected_root.root_id().to_owned(),
                observed: subject.authority_root_id().to_owned(),
            });
        }
        if subject.authority_root_digest() != expected_root.root_digest() {
            return Err(SafetyProfileAuthorizationCommitError::AuthorityRootDigestMismatch);
        }

        if let Some(identity) = checked.expected_predecessor_head().identity() {
            if identity.subject_node_id() != subject.subject_node_id() {
                return Err(
                    SafetyProfileAuthorizationCommitError::PersistedHeadNodeMismatch {
                        expected: subject.subject_node_id().to_owned(),
                        observed: identity.subject_node_id().to_owned(),
                    },
                );
            }
            if identity.authority_root_id() != expected_root.root_id()
                || identity.authority_root_digest() != expected_root.root_digest()
            {
                return Err(
                    SafetyProfileAuthorizationCommitError::PersistedHeadAuthorityRootMismatch,
                );
            }
        }

        let candidate_identity = checked
            .candidate_head()
            .identity()
            .ok_or(SafetyProfileAuthorizationCommitError::CandidateHeadNotCurrent)?;
        if candidate_identity.subject_node_id() != subject.subject_node_id() {
            return Err(SafetyProfileAuthorizationCommitError::CandidateHeadNodeMismatch);
        }
        if candidate_identity.authority_root_id() != expected_root.root_id()
            || candidate_identity.authority_root_digest() != expected_root.root_digest()
        {
            return Err(SafetyProfileAuthorizationCommitError::CandidateHeadAuthorityRootMismatch);
        }

        Ok(Self {
            expected_subject_node_id: subject.subject_node_id().to_owned(),
            expected_root,
            expected_predecessor_head: checked.expected_predecessor_head().clone(),
            candidate_head: checked.candidate_head().clone(),
            valid_from_unix_ms: subject.valid_from_unix_ms(),
            valid_until_unix_ms: subject.valid_until_unix_ms(),
        })
    }

    pub fn expected_subject_node_id(&self) -> &str {
        &self.expected_subject_node_id
    }

    pub fn expected_root(&self) -> &ProfileAuthorityRootSnapshot {
        &self.expected_root
    }

    pub fn expected_predecessor_head(&self) -> &SafetyProfileAuthorizationHead {
        &self.expected_predecessor_head
    }

    pub fn candidate_head(&self) -> &SafetyProfileAuthorizationHead {
        &self.candidate_head
    }

    pub fn valid_from_unix_ms(&self) -> i64 {
        self.valid_from_unix_ms
    }

    pub fn valid_until_unix_ms(&self) -> i64 {
        self.valid_until_unix_ms
    }

    /// Recheck one atomic store observation immediately before persistence.
    ///
    /// Ordering is deliberate: seeing the exact candidate head means a previous
    /// identical commit already won, so retries can recover idempotently even if
    /// the authorization later expired or the root was subsequently rotated. That
    /// result is historical write acknowledgement only; it is never an execution
    /// validity result.
    pub fn recheck_commit_observation(
        &self,
        now_unix_ms: i64,
        current_root: &ProfileAuthorityRootSnapshot,
        current_head: &SafetyProfileAuthorizationHead,
    ) -> Result<SafetyProfileAuthorizationCommitState, SafetyProfileAuthorizationCommitError> {
        if current_head == &self.candidate_head {
            return Ok(SafetyProfileAuthorizationCommitState::AlreadyCommitted);
        }

        if current_root != &self.expected_root {
            return Err(SafetyProfileAuthorizationCommitError::AuthorityRootSnapshotChanged);
        }
        if current_head != &self.expected_predecessor_head {
            return Err(SafetyProfileAuthorizationCommitError::AuthorizationHeadChanged);
        }
        if now_unix_ms < self.valid_from_unix_ms {
            return Err(SafetyProfileAuthorizationCommitError::NotYetValid {
                now_unix_ms,
                valid_from_unix_ms: self.valid_from_unix_ms,
            });
        }
        if now_unix_ms >= self.valid_until_unix_ms {
            return Err(SafetyProfileAuthorizationCommitError::Expired {
                now_unix_ms,
                valid_until_unix_ms: self.valid_until_unix_ms,
            });
        }

        Ok(SafetyProfileAuthorizationCommitState::ReadyToCommit)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SafetyProfileAuthorizationCommitError {
    #[error("profile-authority root id must not be empty")]
    EmptyAuthorityRootId,
    #[error("profile-authority root epoch must be greater than zero")]
    ZeroAuthorityRootEpoch,
    #[error("authorization root id mismatch: expected {expected}, observed {observed}")]
    AuthorityRootIdMismatch { expected: String, observed: String },
    #[error("authorization root digest does not match the provisioned root snapshot")]
    AuthorityRootDigestMismatch,
    #[error("persisted authorization head belongs to node {observed}, expected {expected}")]
    PersistedHeadNodeMismatch { expected: String, observed: String },
    #[error("persisted authorization head does not belong to the provisioned root snapshot")]
    PersistedHeadAuthorityRootMismatch,
    #[error("policy-checked authorization candidate head is unexpectedly uninitialized")]
    CandidateHeadNotCurrent,
    #[error("candidate authorization head node does not match its checked transition")]
    CandidateHeadNodeMismatch,
    #[error("candidate authorization head does not belong to the provisioned root snapshot")]
    CandidateHeadAuthorityRootMismatch,
    #[error("profile-authority root snapshot changed before commit")]
    AuthorityRootSnapshotChanged,
    #[error("persisted profile-authorization head changed before commit")]
    AuthorizationHeadChanged,
    #[error("profile authorization is not yet valid at commit: now {now_unix_ms}, valid from {valid_from_unix_ms}")]
    NotYetValid {
        now_unix_ms: i64,
        valid_from_unix_ms: i64,
    },
    #[error("profile authorization expired before commit: now {now_unix_ms}, valid until {valid_until_unix_ms}")]
    Expired {
        now_unix_ms: i64,
        valid_until_unix_ms: i64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admission::SafetyProfileAuthorizationAdmissionPolicy;
    use crate::authorization::{ProfileAuthorityRootDigest, SafetyProfileAuthorizationSubject};
    use crate::transition::SafetyProfileAuthorizationTransition;
    use crate::{ComponentRequirement, SafetyConfigurationProfile, SAFETY_CONFIGURATION_PROFILE_SCHEMA_V1};

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

    fn subject(
        id: &str,
        generation: u64,
        root_id: &str,
        profile: &SafetyConfigurationProfile,
    ) -> SafetyProfileAuthorizationSubject {
        SafetyProfileAuthorizationSubject::new(
            id,
            root_id,
            root(0x33),
            "compute-campus",
            generation,
            1_000,
            2_000,
            profile,
        )
        .unwrap()
    }

    fn bootstrap(
        id: &str,
        root_id: &str,
        profile: &SafetyConfigurationProfile,
    ) -> SafetyProfileAuthorizationTransition {
        SafetyProfileAuthorizationTransition::bootstrap(subject(id, 1, root_id, profile)).unwrap()
    }

    fn checked_bootstrap(
        root_id: &str,
    ) -> (
        PolicyCheckedSafetyProfileAuthorization,
        ProfileAuthorityRootSnapshot,
    ) {
        let profile = profile();
        let transition = bootstrap("auth-1", root_id, &profile);
        let policy = SafetyProfileAuthorizationAdmissionPolicy::new(
            "compute-campus",
            root(0x33),
            SafetyProfileAuthorizationHead::Uninitialized,
        )
        .unwrap();
        let checked = policy.check(1_500, &transition, &profile).unwrap();
        let root_snapshot = ProfileAuthorityRootSnapshot::new(root_id, root(0x33), 1).unwrap();
        (checked, root_snapshot)
    }

    #[test]
    fn exact_snapshot_and_head_are_ready_to_commit() {
        let (checked, root_snapshot) = checked_bootstrap("facility-profile-root-v1");
        let preconditions = SafetyProfileAuthorizationCommitPreconditions::from_policy_checked(
            &checked,
            root_snapshot.clone(),
        )
        .unwrap();

        assert_eq!(
            preconditions
                .recheck_commit_observation(
                    1_500,
                    &root_snapshot,
                    checked.expected_predecessor_head(),
                )
                .unwrap(),
            SafetyProfileAuthorizationCommitState::ReadyToCommit
        );
    }

    #[test]
    fn bootstrap_root_id_must_match_provisioned_root_not_only_key_digest() {
        let (checked, _) = checked_bootstrap("alias-root-id");
        let provisioned =
            ProfileAuthorityRootSnapshot::new("facility-profile-root-v1", root(0x33), 1).unwrap();
        assert!(matches!(
            SafetyProfileAuthorizationCommitPreconditions::from_policy_checked(
                &checked,
                provisioned,
            ),
            Err(SafetyProfileAuthorizationCommitError::AuthorityRootIdMismatch { .. })
        ));
    }

    #[test]
    fn reprovisioning_same_key_under_new_epoch_invalidates_in_flight_commit() {
        let (checked, root_snapshot) = checked_bootstrap("facility-profile-root-v1");
        let preconditions = SafetyProfileAuthorizationCommitPreconditions::from_policy_checked(
            &checked,
            root_snapshot.clone(),
        )
        .unwrap();
        let rotated_epoch = ProfileAuthorityRootSnapshot::new(
            "facility-profile-root-v1",
            root(0x33),
            root_snapshot.epoch() + 1,
        )
        .unwrap();

        assert_eq!(
            preconditions.recheck_commit_observation(
                1_500,
                &rotated_epoch,
                checked.expected_predecessor_head(),
            ),
            Err(SafetyProfileAuthorizationCommitError::AuthorityRootSnapshotChanged)
        );
    }

    #[test]
    fn head_race_invalidates_in_flight_commit() {
        let (checked, root_snapshot) = checked_bootstrap("facility-profile-root-v1");
        let preconditions = SafetyProfileAuthorizationCommitPreconditions::from_policy_checked(
            &checked,
            root_snapshot.clone(),
        )
        .unwrap();

        let profile = profile();
        let conflicting = bootstrap("auth-conflict", "facility-profile-root-v1", &profile);
        let conflicting_head = SafetyProfileAuthorizationHead::from_transition(&conflicting).unwrap();

        assert_eq!(
            preconditions.recheck_commit_observation(1_500, &root_snapshot, &conflicting_head),
            Err(SafetyProfileAuthorizationCommitError::AuthorizationHeadChanged)
        );
    }

    #[test]
    fn expiry_between_check_and_commit_fails_closed() {
        let (checked, root_snapshot) = checked_bootstrap("facility-profile-root-v1");
        let preconditions = SafetyProfileAuthorizationCommitPreconditions::from_policy_checked(
            &checked,
            root_snapshot.clone(),
        )
        .unwrap();

        assert_eq!(
            preconditions.recheck_commit_observation(
                2_000,
                &root_snapshot,
                checked.expected_predecessor_head(),
            ),
            Err(SafetyProfileAuthorizationCommitError::Expired {
                now_unix_ms: 2_000,
                valid_until_unix_ms: 2_000,
            })
        );
    }

    #[test]
    fn exact_candidate_head_is_idempotent_recovery_not_fresh_authority() {
        let (checked, root_snapshot) = checked_bootstrap("facility-profile-root-v1");
        let preconditions = SafetyProfileAuthorizationCommitPreconditions::from_policy_checked(
            &checked,
            root_snapshot,
        )
        .unwrap();

        // The write may have succeeded even if the caller lost the acknowledgement.
        // Recovery identifies that exact persisted head even after the original
        // authorization validity window has elapsed. This does not make it active.
        let subsequently_reprovisioned =
            ProfileAuthorityRootSnapshot::new("new-root", root(0x44), 9).unwrap();
        assert_eq!(
            preconditions
                .recheck_commit_observation(
                    9_999,
                    &subsequently_reprovisioned,
                    checked.candidate_head(),
                )
                .unwrap(),
            SafetyProfileAuthorizationCommitState::AlreadyCommitted
        );
    }

    #[test]
    fn root_snapshot_rejects_empty_id_or_zero_epoch() {
        assert_eq!(
            ProfileAuthorityRootSnapshot::new("", root(0x33), 1),
            Err(SafetyProfileAuthorizationCommitError::EmptyAuthorityRootId)
        );
        assert_eq!(
            ProfileAuthorityRootSnapshot::new("root", root(0x33), 0),
            Err(SafetyProfileAuthorizationCommitError::ZeroAuthorityRootEpoch)
        );
    }
}
