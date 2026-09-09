// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! TOCTOU-resistant commit preconditions for verifier-profile adoption.
//!
//! Local admission and cryptographic authentication may both succeed against state
//! that changes before persistence. Neither should authorize a stale registry
//! write. This module captures the exact provisioning/root state, exact predecessor
//! head, candidate head, and validity interval that a future atomic store must
//! compare immediately before committing verifier authority.
//!
//! Nothing here performs persistence, proves a signature, or creates an authorized
//! verifier profile.

use thiserror::Error;

use crate::profile_adoption_admission::{
    PolicyCheckedVerifierProfileAdoptionV1, VerifierProfileAdoptionHeadV1,
};

/// Exact externally provisioned adoption-authority root state observed locally.
///
/// `epoch` is a monotone local provisioning generation and is intentionally
/// separate from verifier-adoption generations. Reprovisioning the same key under
/// the same root ID but a new epoch invalidates every in-flight commit that was
/// checked against the previous provisioning state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifierAdoptionAuthorityRootSnapshotV1 {
    authority_subject: String,
    root_id: String,
    root_digest: [u8; 32],
    epoch: u64,
}

impl VerifierAdoptionAuthorityRootSnapshotV1 {
    pub fn new(
        authority_subject: impl Into<String>,
        root_id: impl Into<String>,
        root_digest: [u8; 32],
        epoch: u64,
    ) -> Result<Self, VerifierProfileAdoptionCommitError> {
        let authority_subject = checked_text("authority_subject", authority_subject.into())?;
        let root_id = checked_text("root_id", root_id.into())?;
        if root_digest == [0; 32] {
            return Err(VerifierProfileAdoptionCommitError::ZeroAuthorityRootDigest);
        }
        if epoch == 0 {
            return Err(VerifierProfileAdoptionCommitError::ZeroAuthorityRootEpoch);
        }
        Ok(Self {
            authority_subject,
            root_id,
            root_digest,
            epoch,
        })
    }

    pub fn authority_subject(&self) -> &str {
        &self.authority_subject
    }

    pub fn root_id(&self) -> &str {
        &self.root_id
    }

    pub fn root_digest(&self) -> [u8; 32] {
        self.root_digest
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }
}

/// Result of comparing captured commit preconditions with one atomic store read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifierProfileAdoptionCommitStateV1 {
    /// Provisioning state, predecessor head, and validity are still exactly current.
    /// A store may commit the candidate only after cryptographic proof for the exact
    /// transition bytes has independently succeeded.
    ReadyToCommit,
    /// The exact candidate head is already persisted. This supports idempotent
    /// recovery after an uncertain write acknowledgement. It is historical write
    /// acknowledgement only, not evidence that the adoption is still active.
    AlreadyCommitted,
}

/// Exact non-cryptographic state that must remain true at adoption-registry commit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifierProfileAdoptionCommitPreconditionsV1 {
    expected_root: VerifierAdoptionAuthorityRootSnapshotV1,
    expected_verifier_role_id: String,
    expected_predecessor_head: VerifierProfileAdoptionHeadV1,
    candidate_head: VerifierProfileAdoptionHeadV1,
    valid_from_unix_ms: u64,
    valid_until_unix_ms: u64,
}

impl VerifierProfileAdoptionCommitPreconditionsV1 {
    /// Capture one checked adoption against an exact provisioned authority snapshot.
    ///
    /// This closes state intentionally left outside local admission: the authority
    /// subject, root ID, root digest, and provisioning epoch used at commit are one
    /// explicit snapshot rather than independently looked-up values.
    pub fn from_policy_checked(
        checked: &PolicyCheckedVerifierProfileAdoptionV1,
        expected_root: VerifierAdoptionAuthorityRootSnapshotV1,
    ) -> Result<Self, VerifierProfileAdoptionCommitError> {
        let subject = checked.transition().subject();

        if subject.authority_subject() != expected_root.authority_subject() {
            return Err(VerifierProfileAdoptionCommitError::AuthoritySubjectMismatch {
                expected: expected_root.authority_subject().to_owned(),
                observed: subject.authority_subject().to_owned(),
            });
        }
        if subject.authority_root_id() != expected_root.root_id() {
            return Err(VerifierProfileAdoptionCommitError::AuthorityRootIdMismatch {
                expected: expected_root.root_id().to_owned(),
                observed: subject.authority_root_id().to_owned(),
            });
        }
        if subject.authority_root_digest() != expected_root.root_digest() {
            return Err(VerifierProfileAdoptionCommitError::AuthorityRootDigestMismatch);
        }

        if let Some(identity) = checked.expected_predecessor_head().identity() {
            if identity.authority_subject() != expected_root.authority_subject() {
                return Err(
                    VerifierProfileAdoptionCommitError::PersistedHeadAuthoritySubjectMismatch,
                );
            }
            if identity.authority_root_id() != expected_root.root_id()
                || identity.authority_root_digest() != expected_root.root_digest()
            {
                return Err(VerifierProfileAdoptionCommitError::PersistedHeadAuthorityRootMismatch);
            }
            if identity.verifier_role_id() != subject.verifier_role_id() {
                return Err(VerifierProfileAdoptionCommitError::PersistedHeadVerifierRoleMismatch);
            }
        }

        let candidate_identity = checked
            .candidate_head()
            .identity()
            .ok_or(VerifierProfileAdoptionCommitError::CandidateHeadNotCurrent)?;
        if candidate_identity.authority_subject() != expected_root.authority_subject() {
            return Err(VerifierProfileAdoptionCommitError::CandidateHeadAuthoritySubjectMismatch);
        }
        if candidate_identity.authority_root_id() != expected_root.root_id()
            || candidate_identity.authority_root_digest() != expected_root.root_digest()
        {
            return Err(VerifierProfileAdoptionCommitError::CandidateHeadAuthorityRootMismatch);
        }
        if candidate_identity.verifier_role_id() != subject.verifier_role_id() {
            return Err(VerifierProfileAdoptionCommitError::CandidateHeadVerifierRoleMismatch);
        }
        if candidate_identity.verifier_profile_id() != checked.profile().id() {
            return Err(VerifierProfileAdoptionCommitError::CandidateHeadVerifierProfileMismatch);
        }

        Ok(Self {
            expected_root,
            expected_verifier_role_id: subject.verifier_role_id().to_owned(),
            expected_predecessor_head: checked.expected_predecessor_head().clone(),
            candidate_head: checked.candidate_head().clone(),
            valid_from_unix_ms: subject.valid_from_unix_ms(),
            valid_until_unix_ms: subject.valid_until_unix_ms(),
        })
    }

    pub fn expected_root(&self) -> &VerifierAdoptionAuthorityRootSnapshotV1 {
        &self.expected_root
    }

    pub fn expected_verifier_role_id(&self) -> &str {
        &self.expected_verifier_role_id
    }

    pub fn expected_predecessor_head(&self) -> &VerifierProfileAdoptionHeadV1 {
        &self.expected_predecessor_head
    }

    pub fn candidate_head(&self) -> &VerifierProfileAdoptionHeadV1 {
        &self.candidate_head
    }

    pub fn valid_from_unix_ms(&self) -> u64 {
        self.valid_from_unix_ms
    }

    pub fn valid_until_unix_ms(&self) -> u64 {
        self.valid_until_unix_ms
    }

    /// Recheck one atomic registry observation immediately before persistence.
    ///
    /// Ordering is deliberate: if the exact candidate is already persisted, a
    /// retry may recover idempotently even if the adoption later expired or the
    /// authority was reprovisioned. `AlreadyCommitted` never means currently valid.
    pub fn recheck_commit_observation(
        &self,
        now_unix_ms: u64,
        current_root: &VerifierAdoptionAuthorityRootSnapshotV1,
        current_head: &VerifierProfileAdoptionHeadV1,
    ) -> Result<VerifierProfileAdoptionCommitStateV1, VerifierProfileAdoptionCommitError> {
        if current_head == &self.candidate_head {
            return Ok(VerifierProfileAdoptionCommitStateV1::AlreadyCommitted);
        }
        if current_root != &self.expected_root {
            return Err(VerifierProfileAdoptionCommitError::AuthorityRootSnapshotChanged);
        }
        if current_head != &self.expected_predecessor_head {
            return Err(VerifierProfileAdoptionCommitError::AdoptionHeadChanged);
        }
        if now_unix_ms < self.valid_from_unix_ms {
            return Err(VerifierProfileAdoptionCommitError::NotYetValid {
                now_unix_ms,
                valid_from_unix_ms: self.valid_from_unix_ms,
            });
        }
        if now_unix_ms >= self.valid_until_unix_ms {
            return Err(VerifierProfileAdoptionCommitError::Expired {
                now_unix_ms,
                valid_until_unix_ms: self.valid_until_unix_ms,
            });
        }
        Ok(VerifierProfileAdoptionCommitStateV1::ReadyToCommit)
    }
}

fn checked_text(
    field: &'static str,
    value: String,
) -> Result<String, VerifierProfileAdoptionCommitError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(VerifierProfileAdoptionCommitError::BlankText { field });
    }
    if trimmed.len() > 1024 {
        return Err(VerifierProfileAdoptionCommitError::TextTooLong { field });
    }
    if trimmed.chars().any(char::is_control) {
        return Err(VerifierProfileAdoptionCommitError::ControlCharacters { field });
    }
    Ok(trimmed.to_owned())
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum VerifierProfileAdoptionCommitError {
    #[error("{field} must not be blank")]
    BlankText { field: &'static str },
    #[error("{field} exceeds 1024 bytes")]
    TextTooLong { field: &'static str },
    #[error("{field} contains control characters")]
    ControlCharacters { field: &'static str },
    #[error("verifier-adoption authority root digest must be non-zero")]
    ZeroAuthorityRootDigest,
    #[error("verifier-adoption authority root epoch must be greater than zero")]
    ZeroAuthorityRootEpoch,
    #[error("adoption authority subject mismatch: expected {expected}, observed {observed}")]
    AuthoritySubjectMismatch { expected: String, observed: String },
    #[error("adoption authority root id mismatch: expected {expected}, observed {observed}")]
    AuthorityRootIdMismatch { expected: String, observed: String },
    #[error("adoption authority root digest does not match the provisioned snapshot")]
    AuthorityRootDigestMismatch,
    #[error("persisted adoption head belongs to a different authority subject")]
    PersistedHeadAuthoritySubjectMismatch,
    #[error("persisted adoption head does not belong to the provisioned root snapshot")]
    PersistedHeadAuthorityRootMismatch,
    #[error("persisted adoption head belongs to a different logical verifier role")]
    PersistedHeadVerifierRoleMismatch,
    #[error("policy-checked adoption candidate head is unexpectedly uninitialized")]
    CandidateHeadNotCurrent,
    #[error("candidate adoption head belongs to a different authority subject")]
    CandidateHeadAuthoritySubjectMismatch,
    #[error("candidate adoption head does not belong to the provisioned root snapshot")]
    CandidateHeadAuthorityRootMismatch,
    #[error("candidate adoption head belongs to a different logical verifier role")]
    CandidateHeadVerifierRoleMismatch,
    #[error("candidate adoption head does not bind the exact checked verifier profile")]
    CandidateHeadVerifierProfileMismatch,
    #[error("verifier-adoption authority provisioning snapshot changed before commit")]
    AuthorityRootSnapshotChanged,
    #[error("persisted verifier-adoption head changed before commit")]
    AdoptionHeadChanged,
    #[error("verifier-profile adoption is not yet valid at commit: now {now_unix_ms}, valid from {valid_from_unix_ms}")]
    NotYetValid {
        now_unix_ms: u64,
        valid_from_unix_ms: u64,
    },
    #[error("verifier-profile adoption expired before commit: now {now_unix_ms}, valid until {valid_until_unix_ms}")]
    Expired {
        now_unix_ms: u64,
        valid_until_unix_ms: u64,
    },
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

    fn subject(
        profile: &VerifierProfileV1,
        adoption_id: &str,
        generation: u64,
        authority_subject: &str,
        root_id: &str,
    ) -> VerifierProfileAdoptionSubjectV1 {
        VerifierProfileAdoptionSubjectV1::new(
            adoption_id,
            authority_subject,
            root_id,
            [0x55; 32],
            profile,
            generation,
            1_000,
            2_000,
            EvidenceClass::HardwareVerified,
            VerifierAdoptionScopeV1::AllContinuityVerification,
        )
        .unwrap()
    }

    fn bootstrap(
        profile: &VerifierProfileV1,
        authority_subject: &str,
        root_id: &str,
    ) -> VerifierProfileAdoptionTransitionV1 {
        VerifierProfileAdoptionTransitionV1::bootstrap(subject(
            profile,
            "adopt-1",
            1,
            authority_subject,
            root_id,
        ))
        .unwrap()
    }

    fn checked_bootstrap(
        authority_subject: &str,
        root_id: &str,
    ) -> (
        PolicyCheckedVerifierProfileAdoptionV1,
        VerifierAdoptionAuthorityRootSnapshotV1,
    ) {
        let profile = profile(9, 7);
        let transition = bootstrap(&profile, authority_subject, root_id);
        let policy = VerifierProfileAdoptionAdmissionPolicyV1::new(
            authority_subject,
            root_id,
            [0x55; 32],
            "hardware-verifier-v1",
            VerifierProfileAdoptionHeadV1::Uninitialized,
        )
        .unwrap();
        let checked = policy.check(1_500, &transition, &profile, None).unwrap();
        let root_snapshot = VerifierAdoptionAuthorityRootSnapshotV1::new(
            authority_subject,
            root_id,
            [0x55; 32],
            1,
        )
        .unwrap();
        (checked, root_snapshot)
    }

    #[test]
    fn exact_snapshot_and_head_are_ready_to_commit() {
        let (checked, snapshot) = checked_bootstrap("organization:test", "adoption-root-1");
        let preconditions = VerifierProfileAdoptionCommitPreconditionsV1::from_policy_checked(
            &checked,
            snapshot.clone(),
        )
        .unwrap();

        assert_eq!(
            preconditions
                .recheck_commit_observation(
                    1_500,
                    &snapshot,
                    checked.expected_predecessor_head(),
                )
                .unwrap(),
            VerifierProfileAdoptionCommitStateV1::ReadyToCommit
        );
    }

    #[test]
    fn authority_subject_and_root_id_must_match_exact_provisioning_snapshot() {
        let (checked, _) = checked_bootstrap("organization:test", "alias-root");
        let wrong_id = VerifierAdoptionAuthorityRootSnapshotV1::new(
            "organization:test",
            "adoption-root-1",
            [0x55; 32],
            1,
        )
        .unwrap();
        assert!(matches!(
            VerifierProfileAdoptionCommitPreconditionsV1::from_policy_checked(
                &checked,
                wrong_id,
            ),
            Err(VerifierProfileAdoptionCommitError::AuthorityRootIdMismatch { .. })
        ));

        let (checked, _) = checked_bootstrap("organization:alias", "adoption-root-1");
        let wrong_subject = VerifierAdoptionAuthorityRootSnapshotV1::new(
            "organization:test",
            "adoption-root-1",
            [0x55; 32],
            1,
        )
        .unwrap();
        assert!(matches!(
            VerifierProfileAdoptionCommitPreconditionsV1::from_policy_checked(
                &checked,
                wrong_subject,
            ),
            Err(VerifierProfileAdoptionCommitError::AuthoritySubjectMismatch { .. })
        ));
    }

    #[test]
    fn same_key_reprovisioning_under_new_epoch_invalidates_in_flight_commit() {
        let (checked, snapshot) = checked_bootstrap("organization:test", "adoption-root-1");
        let preconditions = VerifierProfileAdoptionCommitPreconditionsV1::from_policy_checked(
            &checked,
            snapshot.clone(),
        )
        .unwrap();
        let new_epoch = VerifierAdoptionAuthorityRootSnapshotV1::new(
            snapshot.authority_subject(),
            snapshot.root_id(),
            snapshot.root_digest(),
            snapshot.epoch() + 1,
        )
        .unwrap();

        assert_eq!(
            preconditions.recheck_commit_observation(
                1_500,
                &new_epoch,
                checked.expected_predecessor_head(),
            ),
            Err(VerifierProfileAdoptionCommitError::AuthorityRootSnapshotChanged)
        );
    }

    #[test]
    fn persisted_head_race_invalidates_in_flight_commit() {
        let (checked, snapshot) = checked_bootstrap("organization:test", "adoption-root-1");
        let preconditions = VerifierProfileAdoptionCommitPreconditionsV1::from_policy_checked(
            &checked,
            snapshot.clone(),
        )
        .unwrap();
        let conflicting_profile = profile(10, 8);
        let conflicting = VerifierProfileAdoptionTransitionV1::bootstrap(
            VerifierProfileAdoptionSubjectV1::new(
                "adopt-conflict",
                "organization:test",
                "adoption-root-1",
                [0x55; 32],
                &conflicting_profile,
                1,
                1_000,
                2_000,
                EvidenceClass::HardwareVerified,
                VerifierAdoptionScopeV1::AllContinuityVerification,
            )
            .unwrap(),
        )
        .unwrap();
        let conflicting_head = VerifierProfileAdoptionHeadV1::from_transition(&conflicting).unwrap();

        assert_eq!(
            preconditions.recheck_commit_observation(1_500, &snapshot, &conflicting_head),
            Err(VerifierProfileAdoptionCommitError::AdoptionHeadChanged)
        );
    }

    #[test]
    fn expiry_between_admission_and_commit_fails_closed() {
        let (checked, snapshot) = checked_bootstrap("organization:test", "adoption-root-1");
        let preconditions = VerifierProfileAdoptionCommitPreconditionsV1::from_policy_checked(
            &checked,
            snapshot.clone(),
        )
        .unwrap();

        assert_eq!(
            preconditions.recheck_commit_observation(
                2_000,
                &snapshot,
                checked.expected_predecessor_head(),
            ),
            Err(VerifierProfileAdoptionCommitError::Expired {
                now_unix_ms: 2_000,
                valid_until_unix_ms: 2_000,
            })
        );
    }

    #[test]
    fn exact_candidate_is_idempotent_recovery_not_current_authority() {
        let (checked, snapshot) = checked_bootstrap("organization:test", "adoption-root-1");
        let preconditions = VerifierProfileAdoptionCommitPreconditionsV1::from_policy_checked(
            &checked,
            snapshot,
        )
        .unwrap();
        let later_root = VerifierAdoptionAuthorityRootSnapshotV1::new(
            "organization:test",
            "new-root",
            [0x66; 32],
            9,
        )
        .unwrap();

        assert_eq!(
            preconditions
                .recheck_commit_observation(9_999, &later_root, checked.candidate_head())
                .unwrap(),
            VerifierProfileAdoptionCommitStateV1::AlreadyCommitted
        );
    }

    #[test]
    fn verifier_rotation_remains_commit_eligible_under_same_authority_lineage() {
        let profile_a = profile(9, 7);
        let profile_b = profile(10, 8);
        let first = bootstrap(&profile_a, "organization:test", "adoption-root-1");
        let current_head = VerifierProfileAdoptionHeadV1::from_transition(&first).unwrap();
        let second = VerifierProfileAdoptionTransitionV1::successor(
            subject(
                &profile_b,
                "adopt-2",
                2,
                "organization:test",
                "adoption-root-1",
            ),
            &first,
        )
        .unwrap();
        let policy = VerifierProfileAdoptionAdmissionPolicyV1::new(
            "organization:test",
            "adoption-root-1",
            [0x55; 32],
            "hardware-verifier-v1",
            current_head,
        )
        .unwrap();
        let checked = policy.check(1_500, &second, &profile_b, None).unwrap();
        let snapshot = VerifierAdoptionAuthorityRootSnapshotV1::new(
            "organization:test",
            "adoption-root-1",
            [0x55; 32],
            3,
        )
        .unwrap();
        let preconditions = VerifierProfileAdoptionCommitPreconditionsV1::from_policy_checked(
            &checked,
            snapshot.clone(),
        )
        .unwrap();

        assert_eq!(
            preconditions
                .recheck_commit_observation(
                    1_500,
                    &snapshot,
                    checked.expected_predecessor_head(),
                )
                .unwrap(),
            VerifierProfileAdoptionCommitStateV1::ReadyToCommit
        );
        assert_eq!(
            checked.candidate_head().identity().unwrap().verifier_profile_id(),
            profile_b.id()
        );
    }

    #[test]
    fn root_snapshot_rejects_blank_subject_blank_id_zero_digest_or_zero_epoch() {
        assert!(matches!(
            VerifierAdoptionAuthorityRootSnapshotV1::new("", "root", [0x55; 32], 1),
            Err(VerifierProfileAdoptionCommitError::BlankText {
                field: "authority_subject"
            })
        ));
        assert!(matches!(
            VerifierAdoptionAuthorityRootSnapshotV1::new("org", "", [0x55; 32], 1),
            Err(VerifierProfileAdoptionCommitError::BlankText { field: "root_id" })
        ));
        assert_eq!(
            VerifierAdoptionAuthorityRootSnapshotV1::new("org", "root", [0; 32], 1),
            Err(VerifierProfileAdoptionCommitError::ZeroAuthorityRootDigest)
        );
        assert_eq!(
            VerifierAdoptionAuthorityRootSnapshotV1::new("org", "root", [0x55; 32], 0),
            Err(VerifierProfileAdoptionCommitError::ZeroAuthorityRootEpoch)
        );
    }
}
