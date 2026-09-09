// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical claim describing one intended verifier-adoption registry CAS write.
//!
//! This module gives persistence/deployment code an exact object to attest after a
//! successful atomic write. The value remains serializable and therefore untrusted:
//! callers can describe a write, but construction/deserialization does not prove the
//! write occurred.
//!
//! Core theorem:
//!
//! `CommitEvidence != RegistryWriteClaim != CommittedReceipt != CurrentAuthority`.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::profile_adoption::{
    VerifierProfileAdoptionError, VerifierProfileAdoptionPredecessorV1,
    VerifierProfileAdoptionTransitionDigest,
};
use crate::profile_adoption_commit_evidence::{
    VerifierProfileAdoptionCommitEvidenceError, VerifierProfileAdoptionCommitEvidenceIdV1,
    VerifierProfileAdoptionCommitEvidenceV1,
};

pub const VERIFIER_PROFILE_ADOPTION_REGISTRY_WRITE_CLAIM_SCHEMA_V1: &str =
    "symthaea-continuity-verifier-profile-adoption-registry-write-claim-v1";
const SLOT_DOMAIN: &[u8] = b"symthaea.continuity.verifier-profile-adoption.registry-slot.v1\0";
const CLAIM_DOMAIN: &[u8] =
    b"symthaea.continuity.verifier-profile-adoption.registry-write-claim.v1\0";

/// Stable logical registry slot for one authority subject + verifier role.
///
/// Verifier profile/root rotation intentionally does not change this slot. Adoption
/// authority root rotation is also not encoded here: root trust/currentness remains
/// a separate local theorem rather than part of logical role identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct VerifierProfileAdoptionRegistrySlotIdV1([u8; 32]);

impl VerifierProfileAdoptionRegistrySlotIdV1 {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn derive(authority_subject: &str, verifier_role_id: &str) -> Result<Self, VerifierProfileAdoptionRegistryWriteError> {
        checked_text("authority_subject", authority_subject)?;
        checked_text("verifier_role_id", verifier_role_id)?;
        let mut bytes = Vec::new();
        put_str(&mut bytes, authority_subject);
        put_str(&mut bytes, verifier_role_id);
        let mut hasher = blake3::Hasher::new();
        hasher.update(SLOT_DOMAIN);
        hasher.update(&bytes);
        Ok(Self(*hasher.finalize().as_bytes()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct VerifierProfileAdoptionRegistryWriteClaimIdV1([u8; 32]);

impl VerifierProfileAdoptionRegistryWriteClaimIdV1 {
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Serializable description of one intended atomic registry write.
///
/// This is **not** a receipt. A future persistence adapter must compare the exact
/// store state, execute the CAS, and bind successful durable evidence to this claim
/// before any committed-receipt type may exist.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifierProfileAdoptionRegistryWriteClaimV1 {
    schema_version: String,
    registry_id: String,
    registry_epoch: u64,
    slot_id: VerifierProfileAdoptionRegistrySlotIdV1,
    commit_evidence_id: VerifierProfileAdoptionCommitEvidenceIdV1,
    predecessor_transition_digest: Option<VerifierProfileAdoptionTransitionDigest>,
    candidate_transition_digest: VerifierProfileAdoptionTransitionDigest,
    candidate_generation: u64,
    store_revision_before: u64,
    store_revision_after: u64,
    transaction_challenge: [u8; 32],
    claim_id: VerifierProfileAdoptionRegistryWriteClaimIdV1,
}

impl VerifierProfileAdoptionRegistryWriteClaimV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        registry_id: impl Into<String>,
        registry_epoch: u64,
        commit_evidence: &VerifierProfileAdoptionCommitEvidenceV1,
        store_revision_before: u64,
        store_revision_after: u64,
        transaction_challenge: [u8; 32],
    ) -> Result<Self, VerifierProfileAdoptionRegistryWriteError> {
        commit_evidence.validate()?;
        let registry_id = checked_text("registry_id", &registry_id.into())?.to_owned();
        if registry_epoch == 0 {
            return Err(VerifierProfileAdoptionRegistryWriteError::ZeroRegistryEpoch);
        }
        if transaction_challenge == [0; 32] {
            return Err(VerifierProfileAdoptionRegistryWriteError::ZeroTransactionChallenge);
        }
        let expected_after = store_revision_before
            .checked_add(1)
            .ok_or(VerifierProfileAdoptionRegistryWriteError::StoreRevisionExhausted {
                current: store_revision_before,
            })?;
        if store_revision_after != expected_after {
            return Err(VerifierProfileAdoptionRegistryWriteError::StoreRevisionNotSuccessor {
                before: store_revision_before,
                expected_after,
                observed_after: store_revision_after,
            });
        }

        let transition = commit_evidence.transition();
        let subject = transition.subject();
        let slot_id = VerifierProfileAdoptionRegistrySlotIdV1::derive(
            subject.authority_subject(),
            subject.verifier_role_id(),
        )?;
        let predecessor_transition_digest = match transition.predecessor() {
            VerifierProfileAdoptionPredecessorV1::Bootstrap => None,
            VerifierProfileAdoptionPredecessorV1::Previous(digest) => Some(digest),
        };
        let candidate_transition_digest = transition.transition_digest()?;
        let candidate_generation = transition.generation();

        let mut claim = Self {
            schema_version: VERIFIER_PROFILE_ADOPTION_REGISTRY_WRITE_CLAIM_SCHEMA_V1.to_owned(),
            registry_id,
            registry_epoch,
            slot_id,
            commit_evidence_id: commit_evidence.id(),
            predecessor_transition_digest,
            candidate_transition_digest,
            candidate_generation,
            store_revision_before,
            store_revision_after,
            transaction_challenge,
            claim_id: VerifierProfileAdoptionRegistryWriteClaimIdV1([0; 32]),
        };
        claim.claim_id = VerifierProfileAdoptionRegistryWriteClaimIdV1(claim.hash_claim());
        claim.validate_against_commit_evidence(commit_evidence)?;
        Ok(claim)
    }

    /// Intrinsic validation of the write claim itself.
    ///
    /// This cannot prove that `commit_evidence_id` names a real/approved record or
    /// that a store performed the write. Use `validate_against_commit_evidence()` to
    /// additionally bind the claim to the exact durable evidence object.
    pub fn validate(&self) -> Result<(), VerifierProfileAdoptionRegistryWriteError> {
        if self.schema_version != VERIFIER_PROFILE_ADOPTION_REGISTRY_WRITE_CLAIM_SCHEMA_V1 {
            return Err(VerifierProfileAdoptionRegistryWriteError::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        checked_text("registry_id", &self.registry_id)?;
        if self.registry_epoch == 0 {
            return Err(VerifierProfileAdoptionRegistryWriteError::ZeroRegistryEpoch);
        }
        if self.commit_evidence_id.as_bytes() == &[0; 32] {
            return Err(VerifierProfileAdoptionRegistryWriteError::ZeroCommitEvidenceId);
        }
        if self.candidate_transition_digest.as_bytes() == &[0; 32] {
            return Err(VerifierProfileAdoptionRegistryWriteError::ZeroCandidateTransitionDigest);
        }
        if self.candidate_generation == 0 {
            return Err(VerifierProfileAdoptionRegistryWriteError::ZeroCandidateGeneration);
        }
        if self.transaction_challenge == [0; 32] {
            return Err(VerifierProfileAdoptionRegistryWriteError::ZeroTransactionChallenge);
        }
        let expected_after = self
            .store_revision_before
            .checked_add(1)
            .ok_or(VerifierProfileAdoptionRegistryWriteError::StoreRevisionExhausted {
                current: self.store_revision_before,
            })?;
        if self.store_revision_after != expected_after {
            return Err(VerifierProfileAdoptionRegistryWriteError::StoreRevisionNotSuccessor {
                before: self.store_revision_before,
                expected_after,
                observed_after: self.store_revision_after,
            });
        }
        if self.candidate_generation == 1 && self.predecessor_transition_digest.is_some() {
            return Err(VerifierProfileAdoptionRegistryWriteError::GenerationOneHasPredecessor);
        }
        if self.candidate_generation > 1 && self.predecessor_transition_digest.is_none() {
            return Err(VerifierProfileAdoptionRegistryWriteError::MissingPredecessorDigest);
        }
        let expected = VerifierProfileAdoptionRegistryWriteClaimIdV1(self.hash_claim());
        if expected != self.claim_id {
            return Err(VerifierProfileAdoptionRegistryWriteError::ClaimIdentityMismatch);
        }
        Ok(())
    }

    pub fn validate_against_commit_evidence(
        &self,
        commit_evidence: &VerifierProfileAdoptionCommitEvidenceV1,
    ) -> Result<(), VerifierProfileAdoptionRegistryWriteError> {
        self.validate()?;
        commit_evidence.validate()?;
        if self.commit_evidence_id != commit_evidence.id() {
            return Err(VerifierProfileAdoptionRegistryWriteError::CommitEvidenceMismatch);
        }
        let transition = commit_evidence.transition();
        let subject = transition.subject();
        let expected_slot = VerifierProfileAdoptionRegistrySlotIdV1::derive(
            subject.authority_subject(),
            subject.verifier_role_id(),
        )?;
        if self.slot_id != expected_slot {
            return Err(VerifierProfileAdoptionRegistryWriteError::RegistrySlotMismatch);
        }
        if self.candidate_transition_digest != transition.transition_digest()? {
            return Err(VerifierProfileAdoptionRegistryWriteError::CandidateTransitionMismatch);
        }
        if self.candidate_generation != transition.generation() {
            return Err(VerifierProfileAdoptionRegistryWriteError::CandidateGenerationMismatch);
        }
        let expected_predecessor = match transition.predecessor() {
            VerifierProfileAdoptionPredecessorV1::Bootstrap => None,
            VerifierProfileAdoptionPredecessorV1::Previous(digest) => Some(digest),
        };
        if self.predecessor_transition_digest != expected_predecessor {
            return Err(VerifierProfileAdoptionRegistryWriteError::PredecessorTransitionMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> VerifierProfileAdoptionRegistryWriteClaimIdV1 {
        self.claim_id
    }
    pub fn registry_id(&self) -> &str {
        &self.registry_id
    }
    pub fn registry_epoch(&self) -> u64 {
        self.registry_epoch
    }
    pub fn slot_id(&self) -> VerifierProfileAdoptionRegistrySlotIdV1 {
        self.slot_id
    }
    pub fn commit_evidence_id(&self) -> VerifierProfileAdoptionCommitEvidenceIdV1 {
        self.commit_evidence_id
    }
    pub fn predecessor_transition_digest(&self) -> Option<VerifierProfileAdoptionTransitionDigest> {
        self.predecessor_transition_digest
    }
    pub fn candidate_transition_digest(&self) -> VerifierProfileAdoptionTransitionDigest {
        self.candidate_transition_digest
    }
    pub fn candidate_generation(&self) -> u64 {
        self.candidate_generation
    }
    pub fn store_revision_before(&self) -> u64 {
        self.store_revision_before
    }
    pub fn store_revision_after(&self) -> u64 {
        self.store_revision_after
    }
    pub fn transaction_challenge(&self) -> [u8; 32] {
        self.transaction_challenge
    }

    /// Canonical bytes a storage/receipt layer may attest after a successful CAS.
    ///
    /// These bytes describe the claim; authenticating them does not retroactively
    /// prove the write unless the attesting component is itself trusted to observe
    /// and report the atomic store result.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, VerifierProfileAdoptionRegistryWriteError> {
        self.validate()?;
        Ok(self.encode_without_claim_id())
    }

    fn hash_claim(&self) -> [u8; 32] {
        *blake3::hash(&self.encode_without_claim_id()).as_bytes()
    }

    fn encode_without_claim_id(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(CLAIM_DOMAIN);
        put_str(&mut out, VERIFIER_PROFILE_ADOPTION_REGISTRY_WRITE_CLAIM_SCHEMA_V1);
        put_str(&mut out, &self.registry_id);
        out.extend_from_slice(&self.registry_epoch.to_le_bytes());
        out.extend_from_slice(self.slot_id.as_bytes());
        out.extend_from_slice(self.commit_evidence_id.as_bytes());
        match self.predecessor_transition_digest {
            None => out.push(0),
            Some(digest) => {
                out.push(1);
                out.extend_from_slice(digest.as_bytes());
            }
        }
        out.extend_from_slice(self.candidate_transition_digest.as_bytes());
        out.extend_from_slice(&self.candidate_generation.to_le_bytes());
        out.extend_from_slice(&self.store_revision_before.to_le_bytes());
        out.extend_from_slice(&self.store_revision_after.to_le_bytes());
        out.extend_from_slice(&self.transaction_challenge);
        out
    }
}

fn checked_text<'a>(
    field: &'static str,
    value: &'a str,
) -> Result<&'a str, VerifierProfileAdoptionRegistryWriteError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(VerifierProfileAdoptionRegistryWriteError::BlankText { field });
    }
    if trimmed.len() > 1024 {
        return Err(VerifierProfileAdoptionRegistryWriteError::TextTooLong { field });
    }
    if trimmed.chars().any(char::is_control) {
        return Err(VerifierProfileAdoptionRegistryWriteError::ControlCharacters { field });
    }
    Ok(trimmed)
}

fn put_str(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum VerifierProfileAdoptionRegistryWriteError {
    #[error(transparent)]
    CommitEvidence(#[from] VerifierProfileAdoptionCommitEvidenceError),
    #[error(transparent)]
    Adoption(#[from] VerifierProfileAdoptionError),
    #[error("unsupported verifier-profile adoption registry-write schema {0}")]
    UnsupportedSchema(String),
    #[error("{field} must not be blank")]
    BlankText { field: &'static str },
    #[error("{field} exceeds 1024 bytes")]
    TextTooLong { field: &'static str },
    #[error("{field} contains control characters")]
    ControlCharacters { field: &'static str },
    #[error("registry epoch must be greater than zero")]
    ZeroRegistryEpoch,
    #[error("commit-evidence identity must be non-zero")]
    ZeroCommitEvidenceId,
    #[error("candidate transition digest must be non-zero")]
    ZeroCandidateTransitionDigest,
    #[error("candidate adoption generation must be greater than zero")]
    ZeroCandidateGeneration,
    #[error("registry-write transaction challenge must be non-zero")]
    ZeroTransactionChallenge,
    #[error("store revision space exhausted at {current}")]
    StoreRevisionExhausted { current: u64 },
    #[error("store revision must advance by exactly one: before {before}, expected {expected_after}, observed {observed_after}")]
    StoreRevisionNotSuccessor {
        before: u64,
        expected_after: u64,
        observed_after: u64,
    },
    #[error("generation-1 registry write must not name a predecessor transition")]
    GenerationOneHasPredecessor,
    #[error("non-bootstrap registry write must name its exact predecessor transition digest")]
    MissingPredecessorDigest,
    #[error("stored verifier-adoption registry-write claim identity is not canonical")]
    ClaimIdentityMismatch,
    #[error("registry-write claim refers to a different commit-evidence record")]
    CommitEvidenceMismatch,
    #[error("registry-write claim is addressed to a different logical verifier slot")]
    RegistrySlotMismatch,
    #[error("registry-write claim refers to a different candidate transition")]
    CandidateTransitionMismatch,
    #[error("registry-write claim refers to a different candidate generation")]
    CandidateGenerationMismatch,
    #[error("registry-write claim names a different predecessor transition")]
    PredecessorTransitionMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        EvidenceClass, GrantBoundVerifierProfileAdoptionCommitPreconditionsV1,
        RootBoundPolicyCheckedVerifierProfileAdoptionV1,
        TimeBoundVerifierProfileAdoptionCommitPreconditionsV1,
        VerifierAdoptionAuthorityGrantV1, VerifierAdoptionScopeV1,
        VerifierProfileAdoptionAdmissionPolicyV1,
        VerifierProfileAdoptionAuthorityRootSnapshotV1,
        VerifierProfileAdoptionClockObservationV1,
        VerifierProfileAdoptionCommitEvidenceV1, VerifierProfileAdoptionCommitPreconditionsV1,
        VerifierProfileAdoptionHeadV1, VerifierProfileAdoptionSubjectV1,
        VerifierProfileAdoptionTransitionV1, VerifierProfileV1,
        bind_root_bound_adoption_to_authority_grant,
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

    fn commit_evidence(profile: &VerifierProfileV1) -> VerifierProfileAdoptionCommitEvidenceV1 {
        let root = VerifierProfileAdoptionAuthorityRootSnapshotV1::new(
            "organization:test",
            "adoption-root-1",
            [0x55; 32],
            9,
        )
        .unwrap();
        let subject = VerifierProfileAdoptionSubjectV1::new(
            "adopt-1",
            root.authority_subject(),
            root.authority_root_id(),
            root.authority_root_digest(),
            profile,
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
        .check(1_500, &transition, profile, None)
        .unwrap();
        let root_bound =
            RootBoundPolicyCheckedVerifierProfileAdoptionV1::bind(checked, root.clone()).unwrap();
        let grant = VerifierAdoptionAuthorityGrantV1::new(
            "grant-slot-1",
            root,
            profile.profile_name(),
            3,
            EvidenceClass::DifferentiallyVerified,
            VerifierAdoptionScopeV1::AllContinuityVerification,
        )
        .unwrap();
        let grant_bound =
            bind_root_bound_adoption_to_authority_grant(root_bound.clone(), &grant, None).unwrap();
        let commit = VerifierProfileAdoptionCommitPreconditionsV1::from_root_bound(root_bound).unwrap();
        let checked_clock = VerifierProfileAdoptionClockObservationV1::new(
            "trusted-clock-1",
            4,
            1_490,
            1_510,
        )
        .unwrap();
        let time_bound = TimeBoundVerifierProfileAdoptionCommitPreconditionsV1::new(
            commit,
            &checked_clock,
            50,
        )
        .unwrap();
        let preconditions = GrantBoundVerifierProfileAdoptionCommitPreconditionsV1::new(
            grant_bound,
            time_bound,
        )
        .unwrap();
        VerifierProfileAdoptionCommitEvidenceV1::project_from_commit_preconditions(
            &preconditions,
            None,
        )
        .unwrap()
    }

    #[test]
    fn write_claim_binds_exact_commit_evidence_and_revision_transition() {
        let evidence = commit_evidence(&profile(9, 7));
        let claim = VerifierProfileAdoptionRegistryWriteClaimV1::new(
            "continuity-registry-1",
            5,
            &evidence,
            40,
            41,
            [0x77; 32],
        )
        .unwrap();
        claim.validate().unwrap();
        claim.validate_against_commit_evidence(&evidence).unwrap();
        assert_eq!(claim.store_revision_before(), 40);
        assert_eq!(claim.store_revision_after(), 41);
    }

    #[test]
    fn verifier_profile_rotation_preserves_logical_registry_slot() {
        let a = commit_evidence(&profile(9, 7));
        let b = commit_evidence(&profile(10, 8));
        let a = VerifierProfileAdoptionRegistryWriteClaimV1::new(
            "continuity-registry-1", 5, &a, 40, 41, [0x71; 32],
        )
        .unwrap();
        let b = VerifierProfileAdoptionRegistryWriteClaimV1::new(
            "continuity-registry-1", 5, &b, 41, 42, [0x72; 32],
        )
        .unwrap();
        assert_eq!(a.slot_id(), b.slot_id());
        assert_ne!(a.candidate_transition_digest(), b.candidate_transition_digest());
    }

    #[test]
    fn registry_epoch_and_transaction_challenge_change_claim_identity() {
        let evidence = commit_evidence(&profile(9, 7));
        let a = VerifierProfileAdoptionRegistryWriteClaimV1::new(
            "continuity-registry-1", 5, &evidence, 40, 41, [0x71; 32],
        )
        .unwrap();
        let b = VerifierProfileAdoptionRegistryWriteClaimV1::new(
            "continuity-registry-1", 6, &evidence, 40, 41, [0x71; 32],
        )
        .unwrap();
        let c = VerifierProfileAdoptionRegistryWriteClaimV1::new(
            "continuity-registry-1", 5, &evidence, 40, 41, [0x72; 32],
        )
        .unwrap();
        assert_ne!(a.id(), b.id());
        assert_ne!(a.id(), c.id());
    }

    #[test]
    fn store_revision_must_advance_exactly_once() {
        let evidence = commit_evidence(&profile(9, 7));
        let err = VerifierProfileAdoptionRegistryWriteClaimV1::new(
            "continuity-registry-1", 5, &evidence, 40, 42, [0x71; 32],
        )
        .unwrap_err();
        assert_eq!(
            err,
            VerifierProfileAdoptionRegistryWriteError::StoreRevisionNotSuccessor {
                before: 40,
                expected_after: 41,
                observed_after: 42,
            }
        );
    }

    #[test]
    fn same_record_under_different_store_revision_has_different_claim_identity() {
        let evidence = commit_evidence(&profile(9, 7));
        let a = VerifierProfileAdoptionRegistryWriteClaimV1::new(
            "continuity-registry-1", 5, &evidence, 40, 41, [0x71; 32],
        )
        .unwrap();
        let b = VerifierProfileAdoptionRegistryWriteClaimV1::new(
            "continuity-registry-1", 5, &evidence, 41, 42, [0x71; 32],
        )
        .unwrap();
        assert_ne!(a.id(), b.id());
    }

    #[test]
    fn zero_transaction_challenge_is_rejected() {
        let evidence = commit_evidence(&profile(9, 7));
        assert_eq!(
            VerifierProfileAdoptionRegistryWriteClaimV1::new(
                "continuity-registry-1", 5, &evidence, 40, 41, [0; 32],
            )
            .unwrap_err(),
            VerifierProfileAdoptionRegistryWriteError::ZeroTransactionChallenge
        );
    }
}
