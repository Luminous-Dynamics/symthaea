// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! QUAL-ANDROIDPOLICYPERSIST-534: two-plane durable currentness for Android policy.
//!
//! QUAL-ANDROIDPOLICYCODEC-533 gives 532 journals/checkpoints one canonical byte
//! representation. Durable currentness still does not follow from a successful
//! `write(2)` or equivalent API call. This module separates two evidence planes:
//!
//! 1. a journal durability plane that proves the exact canonical journal bytes
//!    were durably prepared, atomically published, metadata-durable, and read back;
//! 2. an independently identified checkpoint-anchor plane that monotonically
//!    advances to the exact resulting 532 checkpoint and is read back there.
//!
//! The evidence identifiers consumed here are opaque outputs of qualified storage
//! adapters. This module does not manufacture fsync, rename, CAS, StrongBox, TPM,
//! secure-element, or OS guarantees. If those guarantees are unavailable, their
//! evidence IDs cannot honestly be produced and this theorem fails closed.
//!
//! A retained prior `AndroidTouchPolicyDurableState` is the recovery barrier. A
//! candidate journal must extend its exact 532 checkpoint. Genesis is deliberately
//! narrow: the first durable state must contain exactly one Install record.

use core::fmt;

use crate::assurance_android_policy_codec::{
    decode_android_policy_checkpoint, decode_android_policy_journal,
    AndroidTouchPolicyCodecError, AndroidTouchPolicyDecodeLimits,
};
use crate::assurance_android_policy_journal::{
    AndroidTouchPolicyCheckpointId, AndroidTouchPolicyJournalCheckpoint,
    AndroidTouchPolicyJournalError,
};
use crate::assurance_android_policy_session::AndroidTouchPolicySessionStatus;

const JOURNAL_BYTES_DOMAIN: &[u8] =
    b"symthaea.soma.presentation.v1/android-policy-journal-bytes\0";
const CHECKPOINT_BYTES_DOMAIN: &[u8] =
    b"symthaea.soma.presentation.v1/android-policy-checkpoint-bytes\0";
const JOURNAL_COMMIT_DOMAIN: &[u8] =
    b"symthaea.soma.presentation.v1/android-policy-journal-durable-commit\0";
const ANCHOR_COMMIT_DOMAIN: &[u8] =
    b"symthaea.soma.presentation.v1/android-policy-anchor-durable-commit\0";
const DURABLE_STATE_DOMAIN: &[u8] =
    b"symthaea.soma.presentation.v1/android-policy-durable-current\0";

macro_rules! digest_id {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(pub [u8; 32]);

        impl $name {
            pub const ZERO: Self = Self([0; 32]);
            pub const fn as_bytes(&self) -> &[u8; 32] {
                &self.0
            }
            pub fn is_zero(&self) -> bool {
                self.0 == [0; 32]
            }
        }
    };
}

digest_id!(AndroidTouchPolicyStorageDomainId);
digest_id!(AndroidTouchPolicyStorageTransactionId);
digest_id!(AndroidTouchPolicyStorageEvidenceId);
digest_id!(AndroidTouchPolicyJournalBytesDigest);
digest_id!(AndroidTouchPolicyCheckpointBytesDigest);
digest_id!(AndroidTouchPolicyJournalDurableCommitId);
digest_id!(AndroidTouchPolicyAnchorDurableCommitId);
digest_id!(AndroidTouchPolicyDurableStateId);

/// Evidence from the adapter that owns the canonical journal blob.
///
/// These fields are assertions by a *separately qualified adapter*. Nonzero IDs
/// make omissions/replay visible; they are not a substitute for qualifying the
/// adapter's actual durability primitives.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AndroidTouchPolicyJournalDurabilityObservation {
    pub storage_domain_id: AndroidTouchPolicyStorageDomainId,
    pub transaction_id: AndroidTouchPolicyStorageTransactionId,
    pub previous_storage_generation: u64,
    pub next_storage_generation: u64,
    pub journal_bytes_digest: AndroidTouchPolicyJournalBytesDigest,
    pub readback_bytes_digest: AndroidTouchPolicyJournalBytesDigest,
    pub durable_prepare_evidence_id: AndroidTouchPolicyStorageEvidenceId,
    pub atomic_publish_evidence_id: AndroidTouchPolicyStorageEvidenceId,
    pub metadata_durability_evidence_id: AndroidTouchPolicyStorageEvidenceId,
    pub exact_readback_evidence_id: AndroidTouchPolicyStorageEvidenceId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AndroidTouchPolicyJournalDurableCommit {
    pub commit_id: AndroidTouchPolicyJournalDurableCommitId,
    pub storage_domain_id: AndroidTouchPolicyStorageDomainId,
    pub transaction_id: AndroidTouchPolicyStorageTransactionId,
    pub storage_generation: u64,
    pub journal_bytes_digest: AndroidTouchPolicyJournalBytesDigest,
    pub checkpoint_id: AndroidTouchPolicyCheckpointId,
}

/// Evidence from an anchor domain intended to survive journal rollback.
///
/// `journal_commit_id` forces ordering at the evidence layer: the anchor may only
/// certify the exact journal durability commit it names.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AndroidTouchPolicyAnchorObservation {
    pub anchor_domain_id: AndroidTouchPolicyStorageDomainId,
    pub previous_anchor_generation: u64,
    pub next_anchor_generation: u64,
    pub previous_checkpoint_id: AndroidTouchPolicyCheckpointId,
    pub next_checkpoint_id: AndroidTouchPolicyCheckpointId,
    pub checkpoint_bytes_digest: AndroidTouchPolicyCheckpointBytesDigest,
    pub readback_checkpoint_bytes_digest: AndroidTouchPolicyCheckpointBytesDigest,
    pub journal_commit_id: AndroidTouchPolicyJournalDurableCommitId,
    pub compare_exchange_evidence_id: AndroidTouchPolicyStorageEvidenceId,
    pub monotonic_advance_evidence_id: AndroidTouchPolicyStorageEvidenceId,
    pub exact_readback_evidence_id: AndroidTouchPolicyStorageEvidenceId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AndroidTouchPolicyAnchorDurableCommit {
    pub commit_id: AndroidTouchPolicyAnchorDurableCommitId,
    pub anchor_domain_id: AndroidTouchPolicyStorageDomainId,
    pub anchor_generation: u64,
    pub checkpoint_id: AndroidTouchPolicyCheckpointId,
    pub checkpoint_bytes_digest: AndroidTouchPolicyCheckpointBytesDigest,
    pub journal_commit_id: AndroidTouchPolicyJournalDurableCommitId,
}

/// Complete restart-authoritative state. The full checkpoint is retained so the
/// next candidate journal can prove exact extension, not merely compare a counter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AndroidTouchPolicyDurableState {
    pub state_id: AndroidTouchPolicyDurableStateId,
    pub journal_storage_domain_id: AndroidTouchPolicyStorageDomainId,
    pub journal_storage_generation: u64,
    pub journal_bytes_digest: AndroidTouchPolicyJournalBytesDigest,
    pub anchor_domain_id: AndroidTouchPolicyStorageDomainId,
    pub anchor_generation: u64,
    pub checkpoint: AndroidTouchPolicyJournalCheckpoint,
    pub checkpoint_id: AndroidTouchPolicyCheckpointId,
    pub checkpoint_bytes_digest: AndroidTouchPolicyCheckpointBytesDigest,
    pub journal_commit_id: AndroidTouchPolicyJournalDurableCommitId,
    pub anchor_commit_id: AndroidTouchPolicyAnchorDurableCommitId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AndroidTouchPolicyDurabilityError {
    Codec(AndroidTouchPolicyCodecError),
    Journal(AndroidTouchPolicyJournalError),
    ZeroStorageDomain,
    SameStorageAndAnchorDomain,
    ZeroTransactionId,
    ZeroEvidenceId,
    JournalDigestMismatch,
    JournalReadbackMismatch,
    CheckpointDigestMismatch,
    CheckpointReadbackMismatch,
    CheckpointDoesNotMatchJournal,
    JournalCommitMismatch,
    PreviousStorageGenerationMismatch,
    StorageGenerationNotSuccessor,
    PreviousAnchorGenerationMismatch,
    AnchorGenerationNotSuccessor,
    PreviousCheckpointMismatch,
    GenesisMustBeSingleInstall,
    PriorDurableStateInvalid,
    JournalDomainChanged,
    AnchorDomainChanged,
}

impl fmt::Display for AndroidTouchPolicyDurabilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for AndroidTouchPolicyDurabilityError {}

impl From<AndroidTouchPolicyCodecError> for AndroidTouchPolicyDurabilityError {
    fn from(value: AndroidTouchPolicyCodecError) -> Self {
        Self::Codec(value)
    }
}

impl From<AndroidTouchPolicyJournalError> for AndroidTouchPolicyDurabilityError {
    fn from(value: AndroidTouchPolicyJournalError) -> Self {
        Self::Journal(value)
    }
}

fn hash_bytes(domain: &[u8], bytes: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    *hasher.finalize().as_bytes()
}

pub fn android_policy_journal_bytes_digest(bytes: &[u8]) -> AndroidTouchPolicyJournalBytesDigest {
    AndroidTouchPolicyJournalBytesDigest(hash_bytes(JOURNAL_BYTES_DOMAIN, bytes))
}

pub fn android_policy_checkpoint_bytes_digest(
    bytes: &[u8],
) -> AndroidTouchPolicyCheckpointBytesDigest {
    AndroidTouchPolicyCheckpointBytesDigest(hash_bytes(CHECKPOINT_BYTES_DOMAIN, bytes))
}

fn require_evidence(
    ids: &[AndroidTouchPolicyStorageEvidenceId],
) -> Result<(), AndroidTouchPolicyDurabilityError> {
    if ids.iter().any(AndroidTouchPolicyStorageEvidenceId::is_zero) {
        return Err(AndroidTouchPolicyDurabilityError::ZeroEvidenceId);
    }
    Ok(())
}

fn successor(previous: u64, next: u64) -> bool {
    previous.checked_add(1) == Some(next)
}

impl AndroidTouchPolicyJournalDurabilityObservation {
    fn certify(
        &self,
        expected_digest: AndroidTouchPolicyJournalBytesDigest,
        checkpoint_id: AndroidTouchPolicyCheckpointId,
        previous: Option<&AndroidTouchPolicyDurableState>,
    ) -> Result<AndroidTouchPolicyJournalDurableCommit, AndroidTouchPolicyDurabilityError> {
        if self.storage_domain_id.is_zero() {
            return Err(AndroidTouchPolicyDurabilityError::ZeroStorageDomain);
        }
        if self.transaction_id.is_zero() {
            return Err(AndroidTouchPolicyDurabilityError::ZeroTransactionId);
        }
        require_evidence(&[
            self.durable_prepare_evidence_id,
            self.atomic_publish_evidence_id,
            self.metadata_durability_evidence_id,
            self.exact_readback_evidence_id,
        ])?;
        if self.journal_bytes_digest != expected_digest {
            return Err(AndroidTouchPolicyDurabilityError::JournalDigestMismatch);
        }
        if self.readback_bytes_digest != expected_digest {
            return Err(AndroidTouchPolicyDurabilityError::JournalReadbackMismatch);
        }

        let expected_previous_generation = previous.map_or(0, |p| p.journal_storage_generation);
        if self.previous_storage_generation != expected_previous_generation {
            return Err(AndroidTouchPolicyDurabilityError::PreviousStorageGenerationMismatch);
        }
        if !successor(self.previous_storage_generation, self.next_storage_generation) {
            return Err(AndroidTouchPolicyDurabilityError::StorageGenerationNotSuccessor);
        }
        if let Some(previous) = previous {
            if self.storage_domain_id != previous.journal_storage_domain_id {
                return Err(AndroidTouchPolicyDurabilityError::JournalDomainChanged);
            }
        }

        let mut hasher = blake3::Hasher::new();
        hasher.update(JOURNAL_COMMIT_DOMAIN);
        hasher.update(self.storage_domain_id.as_bytes());
        hasher.update(self.transaction_id.as_bytes());
        hasher.update(&self.previous_storage_generation.to_le_bytes());
        hasher.update(&self.next_storage_generation.to_le_bytes());
        hasher.update(expected_digest.as_bytes());
        hasher.update(checkpoint_id.as_bytes());
        hasher.update(self.durable_prepare_evidence_id.as_bytes());
        hasher.update(self.atomic_publish_evidence_id.as_bytes());
        hasher.update(self.metadata_durability_evidence_id.as_bytes());
        hasher.update(self.exact_readback_evidence_id.as_bytes());
        let commit_id = AndroidTouchPolicyJournalDurableCommitId(*hasher.finalize().as_bytes());

        Ok(AndroidTouchPolicyJournalDurableCommit {
            commit_id,
            storage_domain_id: self.storage_domain_id,
            transaction_id: self.transaction_id,
            storage_generation: self.next_storage_generation,
            journal_bytes_digest: expected_digest,
            checkpoint_id,
        })
    }
}

impl AndroidTouchPolicyAnchorObservation {
    fn certify(
        &self,
        expected_checkpoint_id: AndroidTouchPolicyCheckpointId,
        expected_checkpoint_digest: AndroidTouchPolicyCheckpointBytesDigest,
        journal_commit: &AndroidTouchPolicyJournalDurableCommit,
        previous: Option<&AndroidTouchPolicyDurableState>,
    ) -> Result<AndroidTouchPolicyAnchorDurableCommit, AndroidTouchPolicyDurabilityError> {
        if self.anchor_domain_id.is_zero() {
            return Err(AndroidTouchPolicyDurabilityError::ZeroStorageDomain);
        }
        if self.anchor_domain_id == journal_commit.storage_domain_id {
            return Err(AndroidTouchPolicyDurabilityError::SameStorageAndAnchorDomain);
        }
        require_evidence(&[
            self.compare_exchange_evidence_id,
            self.monotonic_advance_evidence_id,
            self.exact_readback_evidence_id,
        ])?;
        if self.next_checkpoint_id != expected_checkpoint_id {
            return Err(AndroidTouchPolicyDurabilityError::CheckpointDoesNotMatchJournal);
        }
        if self.checkpoint_bytes_digest != expected_checkpoint_digest {
            return Err(AndroidTouchPolicyDurabilityError::CheckpointDigestMismatch);
        }
        if self.readback_checkpoint_bytes_digest != expected_checkpoint_digest {
            return Err(AndroidTouchPolicyDurabilityError::CheckpointReadbackMismatch);
        }
        if self.journal_commit_id != journal_commit.commit_id {
            return Err(AndroidTouchPolicyDurabilityError::JournalCommitMismatch);
        }

        let (expected_previous_generation, expected_previous_checkpoint) = previous.map_or(
            (0, AndroidTouchPolicyCheckpointId::ZERO),
            |p| (p.anchor_generation, p.checkpoint_id),
        );
        if self.previous_anchor_generation != expected_previous_generation {
            return Err(AndroidTouchPolicyDurabilityError::PreviousAnchorGenerationMismatch);
        }
        if !successor(self.previous_anchor_generation, self.next_anchor_generation) {
            return Err(AndroidTouchPolicyDurabilityError::AnchorGenerationNotSuccessor);
        }
        if self.previous_checkpoint_id != expected_previous_checkpoint {
            return Err(AndroidTouchPolicyDurabilityError::PreviousCheckpointMismatch);
        }
        if let Some(previous) = previous {
            if self.anchor_domain_id != previous.anchor_domain_id {
                return Err(AndroidTouchPolicyDurabilityError::AnchorDomainChanged);
            }
        }

        let mut hasher = blake3::Hasher::new();
        hasher.update(ANCHOR_COMMIT_DOMAIN);
        hasher.update(self.anchor_domain_id.as_bytes());
        hasher.update(&self.previous_anchor_generation.to_le_bytes());
        hasher.update(&self.next_anchor_generation.to_le_bytes());
        hasher.update(self.previous_checkpoint_id.as_bytes());
        hasher.update(self.next_checkpoint_id.as_bytes());
        hasher.update(expected_checkpoint_digest.as_bytes());
        hasher.update(journal_commit.commit_id.as_bytes());
        hasher.update(self.compare_exchange_evidence_id.as_bytes());
        hasher.update(self.monotonic_advance_evidence_id.as_bytes());
        hasher.update(self.exact_readback_evidence_id.as_bytes());
        let commit_id = AndroidTouchPolicyAnchorDurableCommitId(*hasher.finalize().as_bytes());

        Ok(AndroidTouchPolicyAnchorDurableCommit {
            commit_id,
            anchor_domain_id: self.anchor_domain_id,
            anchor_generation: self.next_anchor_generation,
            checkpoint_id: expected_checkpoint_id,
            checkpoint_bytes_digest: expected_checkpoint_digest,
            journal_commit_id: journal_commit.commit_id,
        })
    }
}

impl AndroidTouchPolicyDurableState {
    pub fn validate(&self) -> Result<(), AndroidTouchPolicyDurabilityError> {
        if self.state_id.is_zero()
            || self.journal_storage_domain_id.is_zero()
            || self.anchor_domain_id.is_zero()
            || self.journal_storage_generation == 0
            || self.anchor_generation == 0
            || self.journal_bytes_digest.is_zero()
            || self.checkpoint_id.is_zero()
            || self.checkpoint_bytes_digest.is_zero()
            || self.journal_commit_id.is_zero()
            || self.anchor_commit_id.is_zero()
            || self.journal_storage_domain_id == self.anchor_domain_id
        {
            return Err(AndroidTouchPolicyDurabilityError::PriorDurableStateInvalid);
        }
        if self.checkpoint.checkpoint_id()? != self.checkpoint_id {
            return Err(AndroidTouchPolicyDurabilityError::PriorDurableStateInvalid);
        }
        let expected = compute_durable_state_id(
            self.journal_storage_domain_id,
            self.journal_storage_generation,
            self.journal_bytes_digest,
            self.anchor_domain_id,
            self.anchor_generation,
            self.checkpoint_id,
            self.checkpoint_bytes_digest,
            self.journal_commit_id,
            self.anchor_commit_id,
        );
        if self.state_id != expected {
            return Err(AndroidTouchPolicyDurabilityError::PriorDurableStateInvalid);
        }
        Ok(())
    }
}

fn compute_durable_state_id(
    journal_storage_domain_id: AndroidTouchPolicyStorageDomainId,
    journal_storage_generation: u64,
    journal_bytes_digest: AndroidTouchPolicyJournalBytesDigest,
    anchor_domain_id: AndroidTouchPolicyStorageDomainId,
    anchor_generation: u64,
    checkpoint_id: AndroidTouchPolicyCheckpointId,
    checkpoint_bytes_digest: AndroidTouchPolicyCheckpointBytesDigest,
    journal_commit_id: AndroidTouchPolicyJournalDurableCommitId,
    anchor_commit_id: AndroidTouchPolicyAnchorDurableCommitId,
) -> AndroidTouchPolicyDurableStateId {
    let mut hasher = blake3::Hasher::new();
    hasher.update(DURABLE_STATE_DOMAIN);
    hasher.update(journal_storage_domain_id.as_bytes());
    hasher.update(&journal_storage_generation.to_le_bytes());
    hasher.update(journal_bytes_digest.as_bytes());
    hasher.update(anchor_domain_id.as_bytes());
    hasher.update(&anchor_generation.to_le_bytes());
    hasher.update(checkpoint_id.as_bytes());
    hasher.update(checkpoint_bytes_digest.as_bytes());
    hasher.update(journal_commit_id.as_bytes());
    hasher.update(anchor_commit_id.as_bytes());
    AndroidTouchPolicyDurableStateId(*hasher.finalize().as_bytes())
}

/// Certify one new restart-authoritative policy state.
///
/// `previous` must come from a trust path independent of candidate journal bytes.
/// When it exists, the new canonical journal must extend its exact 532 checkpoint.
pub fn certify_android_policy_durable_state(
    previous: Option<&AndroidTouchPolicyDurableState>,
    journal_bytes: &[u8],
    checkpoint_bytes: &[u8],
    decode_limits: AndroidTouchPolicyDecodeLimits,
    journal_observation: &AndroidTouchPolicyJournalDurabilityObservation,
    anchor_observation: &AndroidTouchPolicyAnchorObservation,
) -> Result<AndroidTouchPolicyDurableState, AndroidTouchPolicyDurabilityError> {
    if let Some(previous) = previous {
        previous.validate()?;
    }

    let journal = decode_android_policy_journal(journal_bytes, decode_limits)?;
    let checkpoint = decode_android_policy_checkpoint(checkpoint_bytes, decode_limits.max_bytes)?;
    let expected_checkpoint = journal.checkpoint()?;
    if checkpoint != expected_checkpoint {
        return Err(AndroidTouchPolicyDurabilityError::CheckpointDoesNotMatchJournal);
    }
    let checkpoint_id = checkpoint.checkpoint_id()?;

    match previous {
        Some(previous) => journal.verify_extends(&previous.checkpoint)?,
        None => {
            if checkpoint.record_count != 1
                || checkpoint.generation != 1
                || checkpoint.status != AndroidTouchPolicySessionStatus::Active
            {
                return Err(AndroidTouchPolicyDurabilityError::GenesisMustBeSingleInstall);
            }
        }
    }

    let journal_digest = android_policy_journal_bytes_digest(journal_bytes);
    let checkpoint_digest = android_policy_checkpoint_bytes_digest(checkpoint_bytes);
    let journal_commit = journal_observation.certify(journal_digest, checkpoint_id, previous)?;
    let anchor_commit = anchor_observation.certify(
        checkpoint_id,
        checkpoint_digest,
        &journal_commit,
        previous,
    )?;

    let state_id = compute_durable_state_id(
        journal_commit.storage_domain_id,
        journal_commit.storage_generation,
        journal_digest,
        anchor_commit.anchor_domain_id,
        anchor_commit.anchor_generation,
        checkpoint_id,
        checkpoint_digest,
        journal_commit.commit_id,
        anchor_commit.commit_id,
    );

    let state = AndroidTouchPolicyDurableState {
        state_id,
        journal_storage_domain_id: journal_commit.storage_domain_id,
        journal_storage_generation: journal_commit.storage_generation,
        journal_bytes_digest: journal_digest,
        anchor_domain_id: anchor_commit.anchor_domain_id,
        anchor_generation: anchor_commit.anchor_generation,
        checkpoint,
        checkpoint_id,
        checkpoint_bytes_digest: checkpoint_digest,
        journal_commit_id: journal_commit.commit_id,
        anchor_commit_id: anchor_commit.commit_id,
    };
    state.validate()?;
    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assurance_android_attention::{
        AndroidAttentionPolicyContextId, AndroidAttentionPolicyStateRoot,
        AndroidAttentionRequirement,
    };
    use crate::assurance_android_attention_policy::AndroidAttentionPolicyAdmission;
    use crate::assurance_android_motion_event::AndroidMotionEventRequirement;
    use crate::assurance_android_motion_policy::AndroidMotionPolicyAdmission;
    use crate::assurance_android_policy_bundle::AndroidTouchPolicyBundle;
    use crate::assurance_android_policy_codec::{
        encode_android_policy_checkpoint, encode_android_policy_journal,
    };
    use crate::assurance_android_policy_journal::AndroidTouchPolicyJournaledStore;
    use crate::assurance_android_policy_session::{
        AndroidTouchPolicySessionAuthorizationId, AndroidTouchPolicySessionNonce,
    };
    use crate::assurance_ingress_policy::{
        IngressAdmissionPolicy, IngressPolicyContextId, IngressPolicyStateRoot,
    };
    use crate::assurance_platform_ingress::{PlatformIngressKind, PlatformIngressProfile};
    use crate::assurance_soma_interaction::{ScreenCaptureProfileId, TouchInputProfileId};
    use symthaea_core::assurance_interaction_continuity::InputAttesterId;
    use symthaea_core::assurance_trusted_surface::TrustedSurfaceId;

    fn d(value: u8) -> [u8; 32] {
        [value; 32]
    }
    fn auth(value: u8) -> AndroidTouchPolicySessionAuthorizationId {
        AndroidTouchPolicySessionAuthorizationId(d(value))
    }
    fn nonce(value: u8) -> AndroidTouchPolicySessionNonce {
        AndroidTouchPolicySessionNonce(d(value))
    }
    fn evidence(value: u8) -> AndroidTouchPolicyStorageEvidenceId {
        AndroidTouchPolicyStorageEvidenceId(d(value))
    }
    fn domain(value: u8) -> AndroidTouchPolicyStorageDomainId {
        AndroidTouchPolicyStorageDomainId(d(value))
    }
    fn tx(value: u8) -> AndroidTouchPolicyStorageTransactionId {
        AndroidTouchPolicyStorageTransactionId(d(value))
    }

    fn bundle(profile_seed: u8) -> AndroidTouchPolicyBundle {
        let attention_requirement = AndroidAttentionRequirement {
            policy_context_id: AndroidAttentionPolicyContextId(d(1)),
            policy_generation: 4,
            policy_state_root: AndroidAttentionPolicyStateRoot(d(2)),
            trusted_surface_id: TrustedSurfaceId(d(3)),
            minimum_sdk_int: 31,
            require_window_focus: true,
            require_flag_secure: true,
            require_hide_application_overlays: true,
            require_filter_touches_when_obscured: true,
            require_view_attached: true,
            require_view_shown: true,
            require_top_resumed: true,
            forbid_multi_window: true,
        };
        let attention_admission = AndroidAttentionPolicyAdmission {
            policy_context_id: attention_requirement.policy_context_id,
            policy_generation: attention_requirement.policy_generation,
            policy_state_root: attention_requirement.policy_state_root,
            authorized_requirement_id: attention_requirement.requirement_id().unwrap(),
        };
        let motion_requirement = AndroidMotionEventRequirement {
            attention_requirement_id: attention_requirement.requirement_id().unwrap(),
            reject_fully_obscured: true,
            reject_partially_obscured: true,
            require_single_pointer: true,
        };
        let motion_admission = AndroidMotionPolicyAdmission {
            policy_context_id: attention_requirement.policy_context_id,
            policy_generation: attention_requirement.policy_generation,
            policy_state_root: attention_requirement.policy_state_root,
            attention_policy_admission_id: attention_admission.admission_id().unwrap(),
            authorized_motion_requirement_id: motion_requirement.requirement_id().unwrap(),
        };
        let ingress_profile = PlatformIngressProfile {
            platform: PlatformIngressKind::AndroidJni,
            abi_version: 1,
            capture_profile_id: ScreenCaptureProfileId(d(profile_seed)),
            touch_input_profile_id: TouchInputProfileId(d(profile_seed.wrapping_add(1))),
            max_frame_bytes: 1024,
        };
        let ingress_policy = IngressAdmissionPolicy {
            policy_context_id: IngressPolicyContextId(d(22)),
            policy_generation: 7,
            policy_state_root: IngressPolicyStateRoot(d(23)),
            platform: PlatformIngressKind::AndroidJni,
            ingress_profile_id: ingress_profile.profile_id().unwrap(),
            trusted_surface_id: attention_requirement.trusted_surface_id,
            input_attester_id: InputAttesterId(d(8)),
            allow_frame: false,
            allow_touch: true,
        };
        AndroidTouchPolicyBundle {
            ingress_policy,
            ingress_profile,
            attention_policy_admission: attention_admission,
            attention_requirement,
            motion_policy_admission: motion_admission,
            motion_requirement,
        }
    }

    fn observations(
        store: &AndroidTouchPolicyJournaledStore,
        previous: Option<&AndroidTouchPolicyDurableState>,
        seed: u8,
    ) -> (
        Vec<u8>,
        Vec<u8>,
        AndroidTouchPolicyJournalDurabilityObservation,
        AndroidTouchPolicyAnchorObservation,
    ) {
        let journal_bytes = encode_android_policy_journal(store.journal()).unwrap();
        let checkpoint = store.checkpoint().unwrap();
        let checkpoint_bytes = encode_android_policy_checkpoint(&checkpoint).unwrap();
        let checkpoint_id = checkpoint.checkpoint_id().unwrap();
        let journal_digest = android_policy_journal_bytes_digest(&journal_bytes);
        let checkpoint_digest = android_policy_checkpoint_bytes_digest(&checkpoint_bytes);
        let previous_storage_generation = previous.map_or(0, |p| p.journal_storage_generation);
        let previous_anchor_generation = previous.map_or(0, |p| p.anchor_generation);
        let previous_checkpoint_id = previous.map_or(AndroidTouchPolicyCheckpointId::ZERO, |p| p.checkpoint_id);

        let journal_observation = AndroidTouchPolicyJournalDurabilityObservation {
            storage_domain_id: domain(90),
            transaction_id: tx(seed),
            previous_storage_generation,
            next_storage_generation: previous_storage_generation + 1,
            journal_bytes_digest: journal_digest,
            readback_bytes_digest: journal_digest,
            durable_prepare_evidence_id: evidence(seed.wrapping_add(1)),
            atomic_publish_evidence_id: evidence(seed.wrapping_add(2)),
            metadata_durability_evidence_id: evidence(seed.wrapping_add(3)),
            exact_readback_evidence_id: evidence(seed.wrapping_add(4)),
        };

        // Compute the journal commit ID using the exact same certified observation;
        // the anchor must explicitly name that commit.
        let journal_commit = journal_observation
            .certify(journal_digest, checkpoint_id, previous)
            .unwrap();
        let anchor_observation = AndroidTouchPolicyAnchorObservation {
            anchor_domain_id: domain(91),
            previous_anchor_generation,
            next_anchor_generation: previous_anchor_generation + 1,
            previous_checkpoint_id,
            next_checkpoint_id: checkpoint_id,
            checkpoint_bytes_digest: checkpoint_digest,
            readback_checkpoint_bytes_digest: checkpoint_digest,
            journal_commit_id: journal_commit.commit_id,
            compare_exchange_evidence_id: evidence(seed.wrapping_add(5)),
            monotonic_advance_evidence_id: evidence(seed.wrapping_add(6)),
            exact_readback_evidence_id: evidence(seed.wrapping_add(7)),
        };
        (
            journal_bytes,
            checkpoint_bytes,
            journal_observation,
            anchor_observation,
        )
    }

    #[test]
    fn genesis_requires_two_distinct_durability_planes_and_exact_readback() {
        let mut store = AndroidTouchPolicyJournaledStore::new();
        store.install(&bundle(20), auth(40), nonce(41)).unwrap();
        let (journal_bytes, checkpoint_bytes, journal_obs, anchor_obs) =
            observations(&store, None, 100);
        let state = certify_android_policy_durable_state(
            None,
            &journal_bytes,
            &checkpoint_bytes,
            AndroidTouchPolicyDecodeLimits::default(),
            &journal_obs,
            &anchor_obs,
        )
        .unwrap();
        assert_eq!(state.journal_storage_generation, 1);
        assert_eq!(state.anchor_generation, 1);
        assert_ne!(state.journal_storage_domain_id, state.anchor_domain_id);
        assert!(!state.state_id.is_zero());
    }

    #[test]
    fn successor_must_extend_retained_checkpoint_and_advance_both_generations() {
        let mut store = AndroidTouchPolicyJournaledStore::new();
        store.install(&bundle(20), auth(40), nonce(41)).unwrap();
        let (j1, c1, jo1, ao1) = observations(&store, None, 100);
        let first = certify_android_policy_durable_state(
            None,
            &j1,
            &c1,
            AndroidTouchPolicyDecodeLimits::default(),
            &jo1,
            &ao1,
        )
        .unwrap();

        let current = store.snapshot().unwrap();
        store.replace(current.state_id, &bundle(30), auth(42)).unwrap();
        let (j2, c2, jo2, ao2) = observations(&store, Some(&first), 110);
        let second = certify_android_policy_durable_state(
            Some(&first),
            &j2,
            &c2,
            AndroidTouchPolicyDecodeLimits::default(),
            &jo2,
            &ao2,
        )
        .unwrap();
        assert_eq!(second.journal_storage_generation, 2);
        assert_eq!(second.anchor_generation, 2);
        assert_eq!(second.checkpoint.record_count, 2);
    }

    #[test]
    fn same_domain_for_journal_and_anchor_is_rejected() {
        let mut store = AndroidTouchPolicyJournaledStore::new();
        store.install(&bundle(20), auth(40), nonce(41)).unwrap();
        let (journal_bytes, checkpoint_bytes, journal_obs, mut anchor_obs) =
            observations(&store, None, 100);
        anchor_obs.anchor_domain_id = journal_obs.storage_domain_id;
        assert_eq!(
            certify_android_policy_durable_state(
                None,
                &journal_bytes,
                &checkpoint_bytes,
                AndroidTouchPolicyDecodeLimits::default(),
                &journal_obs,
                &anchor_obs,
            ),
            Err(AndroidTouchPolicyDurabilityError::SameStorageAndAnchorDomain)
        );
    }

    #[test]
    fn missing_durability_primitive_fails_closed() {
        let mut store = AndroidTouchPolicyJournaledStore::new();
        store.install(&bundle(20), auth(40), nonce(41)).unwrap();
        let (journal_bytes, checkpoint_bytes, mut journal_obs, anchor_obs) =
            observations(&store, None, 100);
        journal_obs.metadata_durability_evidence_id = AndroidTouchPolicyStorageEvidenceId::ZERO;
        assert_eq!(
            certify_android_policy_durable_state(
                None,
                &journal_bytes,
                &checkpoint_bytes,
                AndroidTouchPolicyDecodeLimits::default(),
                &journal_obs,
                &anchor_obs,
            ),
            Err(AndroidTouchPolicyDurabilityError::ZeroEvidenceId)
        );
    }

    #[test]
    fn readback_mismatch_fails_closed() {
        let mut store = AndroidTouchPolicyJournaledStore::new();
        store.install(&bundle(20), auth(40), nonce(41)).unwrap();
        let (journal_bytes, checkpoint_bytes, mut journal_obs, anchor_obs) =
            observations(&store, None, 100);
        journal_obs.readback_bytes_digest = AndroidTouchPolicyJournalBytesDigest(d(7));
        assert_eq!(
            certify_android_policy_durable_state(
                None,
                &journal_bytes,
                &checkpoint_bytes,
                AndroidTouchPolicyDecodeLimits::default(),
                &journal_obs,
                &anchor_obs,
            ),
            Err(AndroidTouchPolicyDurabilityError::JournalReadbackMismatch)
        );
    }

    #[test]
    fn rolled_back_candidate_cannot_replace_retained_durable_state() {
        let mut store = AndroidTouchPolicyJournaledStore::new();
        store.install(&bundle(20), auth(40), nonce(41)).unwrap();
        let (j1, c1, jo1, ao1) = observations(&store, None, 100);
        let first = certify_android_policy_durable_state(
            None,
            &j1,
            &c1,
            AndroidTouchPolicyDecodeLimits::default(),
            &jo1,
            &ao1,
        )
        .unwrap();
        let current = store.snapshot().unwrap();
        store.replace(current.state_id, &bundle(30), auth(42)).unwrap();
        let (j2, c2, jo2, ao2) = observations(&store, Some(&first), 110);
        let second = certify_android_policy_durable_state(
            Some(&first),
            &j2,
            &c2,
            AndroidTouchPolicyDecodeLimits::default(),
            &jo2,
            &ao2,
        )
        .unwrap();

        // Present the old one-record journal while claiming to advance from the
        // two-record retained state. 532 extension verification must reject it.
        let (old_journal, old_checkpoint, mut old_jo, mut old_ao) =
            observations(&AndroidTouchPolicyJournaledStore::recover(
                decode_android_policy_journal(&j1, AndroidTouchPolicyDecodeLimits::default()).unwrap(),
                None,
            ).unwrap(), Some(&second), 120);
        old_jo.previous_storage_generation = second.journal_storage_generation;
        old_jo.next_storage_generation = second.journal_storage_generation + 1;
        old_ao.previous_anchor_generation = second.anchor_generation;
        old_ao.next_anchor_generation = second.anchor_generation + 1;
        old_ao.previous_checkpoint_id = second.checkpoint_id;
        assert!(certify_android_policy_durable_state(
            Some(&second),
            &old_journal,
            &old_checkpoint,
            AndroidTouchPolicyDecodeLimits::default(),
            &old_jo,
            &old_ao,
        )
        .is_err());
    }
}
