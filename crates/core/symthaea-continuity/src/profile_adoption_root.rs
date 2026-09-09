// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Local provisioning-epoch binding for policy-checked verifier-profile adoption.
//!
//! `PolicyCheckedVerifierProfileAdoptionV1` proves that one raw adoption transition
//! satisfies the local subject/root/role/profile/head/time/scope policy observed at
//! admission. The admission policy intentionally stores only the adoption root ID
//! and digest. That is insufficient to carry *local provisioning currentness* into
//! later cryptographic and commit-time composition: the same key bytes can be
//! deliberately reprovisioned under a newer local trust epoch.
//!
//! This module adds that missing local-state identity without changing the #1078
//! adoption wire contract. The remote adoption authority does not get to assert its
//! own local provisioning epoch; the epoch is supplied by the local trust registry
//! and must be rechecked before persistence.
//!
//! Core theorem:
//!
//! `PolicyCheckedAdoption != RootBoundPolicyCheckedAdoption != AuthorizedProfile`.
//!
//! Nothing here verifies a signature, persists a head, or grants evidence/migration
//! authority.

use thiserror::Error;

use crate::profile_adoption::{
    VerifierProfileAdoptionTransitionDigest, VerifierProfileAdoptionTransitionV1,
};
use crate::profile_adoption_admission::{
    PolicyCheckedVerifierProfileAdoptionV1, VerifierProfileAdoptionHeadV1,
};
use crate::verifier::VerifierProfileV1;

const ROOT_SNAPSHOT_DOMAIN: &[u8] =
    b"symthaea.continuity.verifier-profile-adoption.root-snapshot.v1\0";

/// Content identity of one exact locally provisioned adoption-authority root state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VerifierProfileAdoptionAuthorityRootSnapshotId([u8; 32]);

impl VerifierProfileAdoptionAuthorityRootSnapshotId {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Exact local adoption-authority root state observed from trusted provisioning.
///
/// This type is intentionally non-Serde. Its public constructor validates and binds
/// a tuple but does not prove where the tuple came from; production code must obtain
/// it from the local trust registry and compare it again at commit time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifierProfileAdoptionAuthorityRootSnapshotV1 {
    authority_subject: String,
    authority_root_id: String,
    authority_root_digest: [u8; 32],
    provisioning_epoch: u64,
    snapshot_id: VerifierProfileAdoptionAuthorityRootSnapshotId,
}

impl VerifierProfileAdoptionAuthorityRootSnapshotV1 {
    pub fn new(
        authority_subject: impl Into<String>,
        authority_root_id: impl Into<String>,
        authority_root_digest: [u8; 32],
        provisioning_epoch: u64,
    ) -> Result<Self, VerifierProfileAdoptionRootBindingError> {
        let authority_subject = checked_text("authority_subject", authority_subject.into())?;
        let authority_root_id = checked_text("authority_root_id", authority_root_id.into())?;
        if authority_root_digest == [0; 32] {
            return Err(VerifierProfileAdoptionRootBindingError::ZeroAuthorityRootDigest);
        }
        if provisioning_epoch == 0 {
            return Err(VerifierProfileAdoptionRootBindingError::ZeroProvisioningEpoch);
        }
        let snapshot_id = VerifierProfileAdoptionAuthorityRootSnapshotId(hash_snapshot(
            &authority_subject,
            &authority_root_id,
            authority_root_digest,
            provisioning_epoch,
        ));
        Ok(Self {
            authority_subject,
            authority_root_id,
            authority_root_digest,
            provisioning_epoch,
            snapshot_id,
        })
    }

    pub fn authority_subject(&self) -> &str {
        &self.authority_subject
    }

    pub fn authority_root_id(&self) -> &str {
        &self.authority_root_id
    }

    pub fn authority_root_digest(&self) -> [u8; 32] {
        self.authority_root_digest
    }

    pub fn provisioning_epoch(&self) -> u64 {
        self.provisioning_epoch
    }

    pub fn id(&self) -> VerifierProfileAdoptionAuthorityRootSnapshotId {
        self.snapshot_id
    }
}

/// Stronger non-Serde admission witness that additionally retains the exact local
/// adoption-authority provisioning snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootBoundPolicyCheckedVerifierProfileAdoptionV1 {
    checked: PolicyCheckedVerifierProfileAdoptionV1,
    authority_root_snapshot: VerifierProfileAdoptionAuthorityRootSnapshotV1,
    transition_digest: VerifierProfileAdoptionTransitionDigest,
}

impl RootBoundPolicyCheckedVerifierProfileAdoptionV1 {
    /// Bind an already policy-checked transition to the exact local root snapshot
    /// observed for the same admission transaction.
    pub fn bind(
        checked: PolicyCheckedVerifierProfileAdoptionV1,
        authority_root_snapshot: VerifierProfileAdoptionAuthorityRootSnapshotV1,
    ) -> Result<Self, VerifierProfileAdoptionRootBindingError> {
        let subject = checked.transition().subject();
        if subject.authority_subject() != authority_root_snapshot.authority_subject() {
            return Err(VerifierProfileAdoptionRootBindingError::AuthoritySubjectMismatch);
        }
        if subject.authority_root_id() != authority_root_snapshot.authority_root_id() {
            return Err(VerifierProfileAdoptionRootBindingError::AuthorityRootIdMismatch);
        }
        if subject.authority_root_digest() != authority_root_snapshot.authority_root_digest() {
            return Err(VerifierProfileAdoptionRootBindingError::AuthorityRootDigestMismatch);
        }

        if checked.profile().root_digest() == authority_root_snapshot.authority_root_digest() {
            return Err(VerifierProfileAdoptionRootBindingError::VerifierCannotSelfAdopt);
        }

        if let VerifierProfileAdoptionHeadV1::Current(identity) =
            checked.expected_predecessor_head()
            && (identity.authority_subject() != authority_root_snapshot.authority_subject()
                || identity.authority_root_id() != authority_root_snapshot.authority_root_id()
                || identity.authority_root_digest()
                    != authority_root_snapshot.authority_root_digest())
        {
            return Err(VerifierProfileAdoptionRootBindingError::PredecessorRootSnapshotMismatch);
        }

        let transition_digest = checked.transition().transition_digest()?;
        Ok(Self {
            checked,
            authority_root_snapshot,
            transition_digest,
        })
    }

    pub fn checked(&self) -> &PolicyCheckedVerifierProfileAdoptionV1 {
        &self.checked
    }

    pub fn transition(&self) -> &VerifierProfileAdoptionTransitionV1 {
        self.checked.transition()
    }

    pub fn profile(&self) -> &VerifierProfileV1 {
        self.checked.profile()
    }

    pub fn canonical_transition_bytes(&self) -> &[u8] {
        self.checked.canonical_transition_bytes()
    }

    pub fn transition_digest(&self) -> VerifierProfileAdoptionTransitionDigest {
        self.transition_digest
    }

    pub fn expected_predecessor_head(&self) -> &VerifierProfileAdoptionHeadV1 {
        self.checked.expected_predecessor_head()
    }

    pub fn candidate_head(&self) -> &VerifierProfileAdoptionHeadV1 {
        self.checked.candidate_head()
    }

    pub fn authority_root_snapshot(&self) -> &VerifierProfileAdoptionAuthorityRootSnapshotV1 {
        &self.authority_root_snapshot
    }

    /// Recheck only the local root-provisioning part of currentness.
    ///
    /// A later commit capsule must combine this with exact predecessor-head and
    /// validity rechecks. Distinguishing an epoch-only change is important: the same
    /// key bytes under a new local provisioning epoch do not inherit old authority.
    pub fn require_current_authority_root(
        &self,
        current: &VerifierProfileAdoptionAuthorityRootSnapshotV1,
    ) -> Result<(), VerifierProfileAdoptionRootBindingError> {
        if current.authority_subject() != self.authority_root_snapshot.authority_subject() {
            return Err(VerifierProfileAdoptionRootBindingError::AuthoritySubjectChanged);
        }
        if current.authority_root_id() != self.authority_root_snapshot.authority_root_id() {
            return Err(VerifierProfileAdoptionRootBindingError::AuthorityRootIdChanged);
        }
        if current.authority_root_digest() != self.authority_root_snapshot.authority_root_digest() {
            return Err(VerifierProfileAdoptionRootBindingError::AuthorityRootDigestChanged);
        }
        if current.provisioning_epoch() != self.authority_root_snapshot.provisioning_epoch() {
            return Err(
                VerifierProfileAdoptionRootBindingError::ProvisioningEpochChanged {
                    expected: self.authority_root_snapshot.provisioning_epoch(),
                    observed: current.provisioning_epoch(),
                },
            );
        }
        if current.id() != self.authority_root_snapshot.id() {
            return Err(VerifierProfileAdoptionRootBindingError::RootSnapshotIdentityChanged);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum VerifierProfileAdoptionRootBindingError {
    #[error(transparent)]
    Adoption(#[from] crate::profile_adoption::VerifierProfileAdoptionError),
    #[error("{field} must not be blank")]
    BlankText { field: &'static str },
    #[error("{field} exceeds 1024 bytes")]
    TextTooLong { field: &'static str },
    #[error("{field} contains control characters")]
    ControlCharacters { field: &'static str },
    #[error("adoption-authority root digest must be non-zero")]
    ZeroAuthorityRootDigest,
    #[error("adoption-authority provisioning epoch must be non-zero")]
    ZeroProvisioningEpoch,
    #[error("policy-checked adoption authority subject does not match local root snapshot")]
    AuthoritySubjectMismatch,
    #[error("policy-checked adoption root id does not match local root snapshot")]
    AuthorityRootIdMismatch,
    #[error("policy-checked adoption root digest does not match local root snapshot")]
    AuthorityRootDigestMismatch,
    #[error("verifier root cannot also be the adoption-authority root")]
    VerifierCannotSelfAdopt,
    #[error("policy-checked predecessor lineage does not match local root snapshot")]
    PredecessorRootSnapshotMismatch,
    #[error("adoption-authority subject changed since policy admission")]
    AuthoritySubjectChanged,
    #[error("adoption-authority root id changed since policy admission")]
    AuthorityRootIdChanged,
    #[error("adoption-authority root digest changed since policy admission")]
    AuthorityRootDigestChanged,
    #[error(
        "adoption-authority provisioning epoch changed: expected {expected}, observed {observed}"
    )]
    ProvisioningEpochChanged { expected: u64, observed: u64 },
    #[error("adoption-authority root snapshot identity changed")]
    RootSnapshotIdentityChanged,
}

fn checked_text(
    field: &'static str,
    value: String,
) -> Result<String, VerifierProfileAdoptionRootBindingError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(VerifierProfileAdoptionRootBindingError::BlankText { field });
    }
    if trimmed.len() > 1024 {
        return Err(VerifierProfileAdoptionRootBindingError::TextTooLong { field });
    }
    if trimmed.chars().any(char::is_control) {
        return Err(VerifierProfileAdoptionRootBindingError::ControlCharacters { field });
    }
    Ok(trimmed.to_owned())
}

fn hash_snapshot(
    authority_subject: &str,
    authority_root_id: &str,
    authority_root_digest: [u8; 32],
    provisioning_epoch: u64,
) -> [u8; 32] {
    let mut bytes = Vec::new();
    put_str(&mut bytes, authority_subject);
    put_str(&mut bytes, authority_root_id);
    bytes.extend_from_slice(&authority_root_digest);
    bytes.extend_from_slice(&provisioning_epoch.to_le_bytes());
    let mut hasher = blake3::Hasher::new();
    hasher.update(ROOT_SNAPSHOT_DOMAIN);
    hasher.update(&bytes);
    *hasher.finalize().as_bytes()
}

fn put_str(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        EvidenceClass, VerifierAdoptionScopeV1, VerifierProfileAdoptionAdmissionPolicyV1,
        VerifierProfileAdoptionHeadV1, VerifierProfileAdoptionSubjectV1,
        VerifierProfileAdoptionTransitionV1, VerifierProfileV1,
    };

    fn profile() -> VerifierProfileV1 {
        VerifierProfileV1::new(
            "hardware-verifier-v1",
            [0x22; 32],
            7,
            EvidenceClass::HardwareVerified,
        )
        .unwrap()
    }

    fn transition(profile: &VerifierProfileV1) -> VerifierProfileAdoptionTransitionV1 {
        VerifierProfileAdoptionTransitionV1::bootstrap(
            VerifierProfileAdoptionSubjectV1::new(
                "adopt-1",
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
            .unwrap(),
        )
        .unwrap()
    }

    fn checked() -> PolicyCheckedVerifierProfileAdoptionV1 {
        let profile = profile();
        let transition = transition(&profile);
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

    fn root(epoch: u64) -> VerifierProfileAdoptionAuthorityRootSnapshotV1 {
        VerifierProfileAdoptionAuthorityRootSnapshotV1::new(
            "organization:test",
            "adoption-root-1",
            [0x55; 32],
            epoch,
        )
        .unwrap()
    }

    #[test]
    fn exact_policy_checked_adoption_binds_to_exact_local_root_epoch() {
        let checked = checked();
        let expected_bytes = checked.canonical_transition_bytes().to_vec();
        let expected_digest = checked.transition().transition_digest().unwrap();
        let bound =
            RootBoundPolicyCheckedVerifierProfileAdoptionV1::bind(checked, root(9)).unwrap();

        assert_eq!(bound.authority_root_snapshot().provisioning_epoch(), 9);
        assert_eq!(bound.canonical_transition_bytes(), expected_bytes);
        assert_eq!(bound.transition_digest(), expected_digest);
        assert_eq!(
            bound.authority_root_snapshot().authority_root_digest(),
            [0x55; 32]
        );
        bound.require_current_authority_root(&root(9)).unwrap();
    }

    #[test]
    fn same_key_reprovisioned_under_new_epoch_fails_currentness() {
        let first = root(9);
        let second = root(10);
        assert_eq!(
            first.authority_root_digest(),
            second.authority_root_digest()
        );
        assert_ne!(first.id(), second.id());

        let bound =
            RootBoundPolicyCheckedVerifierProfileAdoptionV1::bind(checked(), first).unwrap();
        assert_eq!(
            bound.require_current_authority_root(&second),
            Err(
                VerifierProfileAdoptionRootBindingError::ProvisioningEpochChanged {
                    expected: 9,
                    observed: 10,
                }
            )
        );
    }

    #[test]
    fn different_local_root_cannot_be_bound_to_policy_checked_transition() {
        let wrong = VerifierProfileAdoptionAuthorityRootSnapshotV1::new(
            "organization:test",
            "adoption-root-1",
            [0x66; 32],
            9,
        )
        .unwrap();
        assert_eq!(
            RootBoundPolicyCheckedVerifierProfileAdoptionV1::bind(checked(), wrong),
            Err(VerifierProfileAdoptionRootBindingError::AuthorityRootDigestMismatch)
        );
    }

    #[test]
    fn different_authority_subject_or_root_id_cannot_be_bound() {
        let wrong_subject = VerifierProfileAdoptionAuthorityRootSnapshotV1::new(
            "organization:other",
            "adoption-root-1",
            [0x55; 32],
            9,
        )
        .unwrap();
        assert_eq!(
            RootBoundPolicyCheckedVerifierProfileAdoptionV1::bind(checked(), wrong_subject),
            Err(VerifierProfileAdoptionRootBindingError::AuthoritySubjectMismatch)
        );

        let wrong_id = VerifierProfileAdoptionAuthorityRootSnapshotV1::new(
            "organization:test",
            "adoption-root-other",
            [0x55; 32],
            9,
        )
        .unwrap();
        assert_eq!(
            RootBoundPolicyCheckedVerifierProfileAdoptionV1::bind(checked(), wrong_id),
            Err(VerifierProfileAdoptionRootBindingError::AuthorityRootIdMismatch)
        );
    }

    #[test]
    fn zero_epoch_or_zero_root_never_forms_snapshot() {
        assert_eq!(
            VerifierProfileAdoptionAuthorityRootSnapshotV1::new(
                "organization:test",
                "adoption-root-1",
                [0x55; 32],
                0,
            ),
            Err(VerifierProfileAdoptionRootBindingError::ZeroProvisioningEpoch)
        );
        assert_eq!(
            VerifierProfileAdoptionAuthorityRootSnapshotV1::new(
                "organization:test",
                "adoption-root-1",
                [0; 32],
                9,
            ),
            Err(VerifierProfileAdoptionRootBindingError::ZeroAuthorityRootDigest)
        );
    }
}
