// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! TOCTOU-resistant commit preconditions for root-bound verifier-profile adoption.
//!
//! #1104 proves local adoption policy against an exact transition/profile/head/time/scope.
//! #1112 strengthens that result with the exact local adoption-authority provisioning
//! snapshot, including its monotone provisioning epoch. This module composes those
//! already-proven facts into the state that must still be true immediately before a
//! future atomic registry write.
//!
//! Nothing here verifies a signature, writes persistence, or creates an authorized
//! verifier profile. A production write additionally requires Xenia-owned proof for
//! the exact canonical transition bytes retained by the root-bound witness.

use thiserror::Error;

use crate::profile_adoption::{
    VerifierProfileAdoptionTransitionDigest, VerifierProfileAdoptionTransitionV1,
};
use crate::profile_adoption_admission::VerifierProfileAdoptionHeadV1;
use crate::profile_adoption_root::{
    RootBoundPolicyCheckedVerifierProfileAdoptionV1,
    VerifierProfileAdoptionAuthorityRootSnapshotV1, VerifierProfileAdoptionRootBindingError,
};
use crate::verifier::VerifierProfileV1;

/// Result of comparing captured adoption commit preconditions with one atomic
/// registry observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifierProfileAdoptionCommitStateV1 {
    /// Root provisioning state, predecessor head, and validity are still exact.
    /// Cryptographic proof for the exact transition must independently exist before
    /// the store may install the candidate.
    ReadyToCommit,
    /// The exact candidate head is already the current registry head. This is only
    /// historical acknowledgement for uncertain-write recovery; it does not imply
    /// that the adoption is presently valid after later expiry or root rotation.
    AlreadyCommitted,
}

/// Non-cryptographic state that must remain true at verifier-adoption commit time.
///
/// The complete stronger #1112 witness is retained rather than decomposed into a
/// second parallel root/head schema. This keeps one meaning for root provisioning
/// epoch and one canonical transition identity across admission, crypto matching,
/// and persistence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifierProfileAdoptionCommitPreconditionsV1 {
    root_bound: RootBoundPolicyCheckedVerifierProfileAdoptionV1,
    valid_from_unix_ms: u64,
    valid_until_unix_ms: u64,
}

impl VerifierProfileAdoptionCommitPreconditionsV1 {
    /// Capture commit preconditions only from the stronger root-bound policy witness.
    ///
    /// There is intentionally no constructor from a bare #1104 policy result plus
    /// loose root fields; doing so would recreate the duplicate root-currentness
    /// seam that #1112 exists to remove.
    pub fn from_root_bound(
        root_bound: RootBoundPolicyCheckedVerifierProfileAdoptionV1,
    ) -> Result<Self, VerifierProfileAdoptionCommitError> {
        if root_bound.candidate_head().identity().is_none() {
            return Err(VerifierProfileAdoptionCommitError::CandidateHeadNotCurrent);
        }
        let subject = root_bound.transition().subject();
        Ok(Self {
            valid_from_unix_ms: subject.valid_from_unix_ms(),
            valid_until_unix_ms: subject.valid_until_unix_ms(),
            root_bound,
        })
    }

    pub fn root_bound(&self) -> &RootBoundPolicyCheckedVerifierProfileAdoptionV1 {
        &self.root_bound
    }

    pub fn transition(&self) -> &VerifierProfileAdoptionTransitionV1 {
        self.root_bound.transition()
    }

    pub fn profile(&self) -> &VerifierProfileV1 {
        self.root_bound.profile()
    }

    pub fn canonical_transition_bytes(&self) -> &[u8] {
        self.root_bound.canonical_transition_bytes()
    }

    pub fn transition_digest(&self) -> VerifierProfileAdoptionTransitionDigest {
        self.root_bound.transition_digest()
    }

    pub fn expected_root_snapshot(&self) -> &VerifierProfileAdoptionAuthorityRootSnapshotV1 {
        self.root_bound.authority_root_snapshot()
    }

    pub fn expected_predecessor_head(&self) -> &VerifierProfileAdoptionHeadV1 {
        self.root_bound.expected_predecessor_head()
    }

    pub fn candidate_head(&self) -> &VerifierProfileAdoptionHeadV1 {
        self.root_bound.candidate_head()
    }

    pub fn valid_from_unix_ms(&self) -> u64 {
        self.valid_from_unix_ms
    }

    pub fn valid_until_unix_ms(&self) -> u64 {
        self.valid_until_unix_ms
    }

    /// Recheck one atomic registry observation immediately before persistence.
    ///
    /// Ordering is deliberate. Exact candidate-head equality acknowledges a write
    /// that already happened even if the adoption has since expired or the local
    /// authority root has since been reprovisioned. That historical result does not
    /// establish current authority. A *fresh* write, by contrast, requires the exact
    /// #1112 root snapshot/epoch, predecessor head, and validity interval to remain
    /// current together.
    pub fn recheck_commit_observation(
        &self,
        now_unix_ms: u64,
        current_root: &VerifierProfileAdoptionAuthorityRootSnapshotV1,
        current_head: &VerifierProfileAdoptionHeadV1,
    ) -> Result<VerifierProfileAdoptionCommitStateV1, VerifierProfileAdoptionCommitError> {
        if current_head == self.candidate_head() {
            return Ok(VerifierProfileAdoptionCommitStateV1::AlreadyCommitted);
        }

        self.root_bound
            .require_current_authority_root(current_root)?;

        if current_head != self.expected_predecessor_head() {
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

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum VerifierProfileAdoptionCommitError {
    #[error(transparent)]
    RootBinding(#[from] VerifierProfileAdoptionRootBindingError),
    #[error("policy-checked adoption candidate head is unexpectedly uninitialized")]
    CandidateHeadNotCurrent,
    #[error("persisted verifier-adoption head changed before commit")]
    AdoptionHeadChanged,
    #[error(
        "verifier-profile adoption is not yet valid at commit: now {now_unix_ms}, valid from {valid_from_unix_ms}"
    )]
    NotYetValid {
        now_unix_ms: u64,
        valid_from_unix_ms: u64,
    },
    #[error(
        "verifier-profile adoption expired before commit: now {now_unix_ms}, valid until {valid_until_unix_ms}"
    )]
    Expired {
        now_unix_ms: u64,
        valid_until_unix_ms: u64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        EvidenceClass, PolicyCheckedVerifierProfileAdoptionV1, VerifierAdoptionScopeV1,
        VerifierProfileAdoptionAdmissionPolicyV1, VerifierProfileAdoptionAuthorityRootSnapshotV1,
        VerifierProfileAdoptionHeadV1, VerifierProfileAdoptionSubjectV1,
        VerifierProfileAdoptionTransitionV1, VerifierProfileV1,
    };

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
    ) -> VerifierProfileAdoptionSubjectV1 {
        VerifierProfileAdoptionSubjectV1::new(
            adoption_id,
            "organization:test",
            "adoption-root-1",
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
        adoption_id: &str,
    ) -> VerifierProfileAdoptionTransitionV1 {
        VerifierProfileAdoptionTransitionV1::bootstrap(subject(profile, adoption_id, 1)).unwrap()
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

    fn checked_bootstrap() -> PolicyCheckedVerifierProfileAdoptionV1 {
        let profile = profile(0x22, 7);
        let transition = bootstrap(&profile, "adopt-1");
        VerifierProfileAdoptionAdmissionPolicyV1::new(
            "organization:test",
            "adoption-root-1",
            [0x55; 32],
            "hardware-verifier-v1",
            VerifierProfileAdoptionHeadV1::Uninitialized,
        )
        .unwrap()
        .check(1_500, &transition, &profile, None)
        .unwrap()
    }

    fn preconditions() -> VerifierProfileAdoptionCommitPreconditionsV1 {
        let bound =
            RootBoundPolicyCheckedVerifierProfileAdoptionV1::bind(checked_bootstrap(), root(9))
                .unwrap();
        VerifierProfileAdoptionCommitPreconditionsV1::from_root_bound(bound).unwrap()
    }

    #[test]
    fn exact_root_epoch_predecessor_and_validity_are_ready_to_commit() {
        let preconditions = preconditions();
        assert_eq!(
            preconditions
                .recheck_commit_observation(
                    1_500,
                    &root(9),
                    preconditions.expected_predecessor_head(),
                )
                .unwrap(),
            VerifierProfileAdoptionCommitStateV1::ReadyToCommit
        );
    }

    #[test]
    fn same_key_reprovisioned_under_new_epoch_blocks_fresh_commit() {
        let preconditions = preconditions();
        assert_eq!(
            preconditions.recheck_commit_observation(
                1_500,
                &root(10),
                preconditions.expected_predecessor_head(),
            ),
            Err(VerifierProfileAdoptionCommitError::RootBinding(
                VerifierProfileAdoptionRootBindingError::ProvisioningEpochChanged {
                    expected: 9,
                    observed: 10,
                }
            ))
        );
    }

    #[test]
    fn changed_root_digest_blocks_fresh_commit() {
        let preconditions = preconditions();
        let changed = VerifierProfileAdoptionAuthorityRootSnapshotV1::new(
            "organization:test",
            "adoption-root-1",
            [0x66; 32],
            9,
        )
        .unwrap();
        assert_eq!(
            preconditions.recheck_commit_observation(
                1_500,
                &changed,
                preconditions.expected_predecessor_head(),
            ),
            Err(VerifierProfileAdoptionCommitError::RootBinding(
                VerifierProfileAdoptionRootBindingError::AuthorityRootDigestChanged
            ))
        );
    }

    #[test]
    fn concurrent_head_change_blocks_fresh_commit() {
        let preconditions = preconditions();
        let profile = profile(0x22, 7);
        let competing = bootstrap(&profile, "competing-adoption");
        let competing_head = VerifierProfileAdoptionHeadV1::from_transition(&competing).unwrap();
        assert_ne!(&competing_head, preconditions.expected_predecessor_head());
        assert_ne!(&competing_head, preconditions.candidate_head());

        assert_eq!(
            preconditions.recheck_commit_observation(1_500, &root(9), &competing_head),
            Err(VerifierProfileAdoptionCommitError::AdoptionHeadChanged)
        );
    }

    #[test]
    fn expiry_between_admission_and_commit_blocks_fresh_write() {
        let preconditions = preconditions();
        assert_eq!(
            preconditions.recheck_commit_observation(
                2_000,
                &root(9),
                preconditions.expected_predecessor_head(),
            ),
            Err(VerifierProfileAdoptionCommitError::Expired {
                now_unix_ms: 2_000,
                valid_until_unix_ms: 2_000,
            })
        );
    }

    #[test]
    fn not_yet_valid_time_blocks_fresh_write() {
        let preconditions = preconditions();
        assert_eq!(
            preconditions.recheck_commit_observation(
                999,
                &root(9),
                preconditions.expected_predecessor_head(),
            ),
            Err(VerifierProfileAdoptionCommitError::NotYetValid {
                now_unix_ms: 999,
                valid_from_unix_ms: 1_000,
            })
        );
    }

    #[test]
    fn exact_candidate_head_is_historical_idempotent_even_after_expiry_and_root_rotation() {
        let preconditions = preconditions();
        let replacement_root = VerifierProfileAdoptionAuthorityRootSnapshotV1::new(
            "organization:test",
            "replacement-root",
            [0x77; 32],
            42,
        )
        .unwrap();

        assert_eq!(
            preconditions
                .recheck_commit_observation(
                    9_999,
                    &replacement_root,
                    preconditions.candidate_head(),
                )
                .unwrap(),
            VerifierProfileAdoptionCommitStateV1::AlreadyCommitted
        );
    }

    #[test]
    fn preconditions_retain_exact_transition_crypto_and_registry_identity() {
        let preconditions = preconditions();
        assert_eq!(
            preconditions.canonical_transition_bytes(),
            preconditions
                .transition()
                .canonical_signing_bytes()
                .unwrap()
        );
        assert_eq!(
            preconditions.transition_digest(),
            preconditions.transition().transition_digest().unwrap()
        );
        assert_eq!(
            preconditions.expected_root_snapshot().provisioning_epoch(),
            9
        );
        assert!(preconditions.candidate_head().identity().is_some());
    }
}
