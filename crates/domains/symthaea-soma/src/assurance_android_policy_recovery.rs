// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! QUAL-ANDROIDPOLICYRECOVERY-536: bind restart recovery to durable currentness.
//!
//! A valid 532 journal can be replayed into a valid 531 store without proving it is
//! the history selected by 534's durability/anchor planes. This module closes that
//! restart gap. Recovery accepts only the canonical 533 journal/checkpoint bytes
//! whose exact digests and checkpoint are named by one valid 534 durable state,
//! semantically replays them through 532/531, and binds the resulting store snapshot
//! into an immutable recovery receipt.
//!
//! Exact recovery and active currentness remain distinct. A durably revoked session
//! must recover successfully so restart cannot erase the revocation, but it can
//! never produce the active-recovery certificate used by event-time code.

use core::fmt;

use crate::assurance_android_policy_codec::{
    decode_android_policy_checkpoint, decode_android_policy_journal,
    AndroidTouchPolicyCodecError, AndroidTouchPolicyDecodeLimits,
};
use crate::assurance_android_policy_durability::{
    android_policy_checkpoint_bytes_digest, android_policy_journal_bytes_digest,
    AndroidTouchPolicyDurabilityError, AndroidTouchPolicyDurableState,
    AndroidTouchPolicyDurableStateId,
};
use crate::assurance_android_policy_journal::{
    AndroidTouchPolicyCheckpointId, AndroidTouchPolicyJournalError,
    AndroidTouchPolicyJournaledStore,
};
use crate::assurance_android_policy_session::{
    AndroidTouchPolicyCurrentCertificateId, AndroidTouchPolicySessionError,
    AndroidTouchPolicySessionId, AndroidTouchPolicySessionStateId,
    AndroidTouchPolicySessionStatus,
};
use crate::assurance_android_policy_store::{
    AndroidTouchPolicyStoreError, AndroidTouchPolicyStoreSnapshot,
};
use crate::assurance_android_policy_bundle::AndroidTouchPolicyBundleId;

const RECOVERY_DOMAIN: &[u8] =
    b"symthaea.soma.presentation.v1/android-policy-recovery\0";
const ACTIVE_RECOVERY_DOMAIN: &[u8] =
    b"symthaea.soma.presentation.v1/android-policy-active-recovery\0";

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

digest_id!(AndroidTouchPolicyRecoveryReceiptId);
digest_id!(AndroidTouchPolicyActiveRecoveryCertificateId);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AndroidTouchPolicyRecoveryReceipt {
    pub durable_state_id: AndroidTouchPolicyDurableStateId,
    pub checkpoint_id: AndroidTouchPolicyCheckpointId,
    pub session_id: AndroidTouchPolicySessionId,
    pub generation: u64,
    pub state_id: AndroidTouchPolicySessionStateId,
    pub bundle_id: AndroidTouchPolicyBundleId,
    pub status: AndroidTouchPolicySessionStatus,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AndroidTouchPolicyActiveRecoveryCertificate {
    pub certificate_id: AndroidTouchPolicyActiveRecoveryCertificateId,
    pub recovery_receipt_id: AndroidTouchPolicyRecoveryReceiptId,
    pub durable_state_id: AndroidTouchPolicyDurableStateId,
    pub current_certificate_id: AndroidTouchPolicyCurrentCertificateId,
    pub session_id: AndroidTouchPolicySessionId,
    pub generation: u64,
    pub state_id: AndroidTouchPolicySessionStateId,
    pub bundle_id: AndroidTouchPolicyBundleId,
}

#[derive(Clone, Debug)]
pub struct AndroidTouchPolicyRecoveredStore {
    store: AndroidTouchPolicyJournaledStore,
    receipt: AndroidTouchPolicyRecoveryReceipt,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AndroidTouchPolicyRecoveryError {
    Codec(AndroidTouchPolicyCodecError),
    Durability(AndroidTouchPolicyDurabilityError),
    Journal(AndroidTouchPolicyJournalError),
    Store(AndroidTouchPolicyStoreError),
    Session(AndroidTouchPolicySessionError),
    JournalDigestMismatch,
    CheckpointDigestMismatch,
    CheckpointMismatch,
    RecoveredSnapshotMismatch,
    RecoveredRevoked,
    RecoveryReceiptMismatch,
}

impl fmt::Display for AndroidTouchPolicyRecoveryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for AndroidTouchPolicyRecoveryError {}

impl From<AndroidTouchPolicyCodecError> for AndroidTouchPolicyRecoveryError {
    fn from(value: AndroidTouchPolicyCodecError) -> Self {
        Self::Codec(value)
    }
}

impl From<AndroidTouchPolicyDurabilityError> for AndroidTouchPolicyRecoveryError {
    fn from(value: AndroidTouchPolicyDurabilityError) -> Self {
        Self::Durability(value)
    }
}

impl From<AndroidTouchPolicyJournalError> for AndroidTouchPolicyRecoveryError {
    fn from(value: AndroidTouchPolicyJournalError) -> Self {
        Self::Journal(value)
    }
}

impl From<AndroidTouchPolicyStoreError> for AndroidTouchPolicyRecoveryError {
    fn from(value: AndroidTouchPolicyStoreError) -> Self {
        Self::Store(value)
    }
}

impl From<AndroidTouchPolicySessionError> for AndroidTouchPolicyRecoveryError {
    fn from(value: AndroidTouchPolicySessionError) -> Self {
        Self::Session(value)
    }
}

fn snapshot_matches_durable(
    snapshot: &AndroidTouchPolicyStoreSnapshot,
    durable: &AndroidTouchPolicyDurableState,
) -> bool {
    snapshot.session_id == durable.checkpoint.session_id
        && snapshot.generation == durable.checkpoint.generation
        && snapshot.state_id == durable.checkpoint.state_id
        && snapshot.bundle_id == durable.checkpoint.bundle_id
        && snapshot.status == durable.checkpoint.status
}

impl AndroidTouchPolicyRecoveryReceipt {
    pub fn receipt_id(
        &self,
    ) -> Result<AndroidTouchPolicyRecoveryReceiptId, AndroidTouchPolicyRecoveryError> {
        if self.durable_state_id.is_zero()
            || self.checkpoint_id.is_zero()
            || self.session_id.is_zero()
            || self.generation == 0
            || self.state_id.is_zero()
            || self.bundle_id.is_zero()
        {
            return Err(AndroidTouchPolicyRecoveryError::RecoveryReceiptMismatch);
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(RECOVERY_DOMAIN);
        hasher.update(self.durable_state_id.as_bytes());
        hasher.update(self.checkpoint_id.as_bytes());
        hasher.update(self.session_id.as_bytes());
        hasher.update(&self.generation.to_le_bytes());
        hasher.update(self.state_id.as_bytes());
        hasher.update(self.bundle_id.as_bytes());
        hasher.update(&[self.status as u8]);
        Ok(AndroidTouchPolicyRecoveryReceiptId(*hasher.finalize().as_bytes()))
    }
}

/// Recover exactly the policy state named by a retained 534 durable state.
///
/// `checkpoint_bytes` are separately supplied/read so the recovery path proves
/// both canonical persisted objects agree with the retained durable identity.
pub fn recover_android_policy_durable_state(
    durable: &AndroidTouchPolicyDurableState,
    journal_bytes: &[u8],
    checkpoint_bytes: &[u8],
    limits: AndroidTouchPolicyDecodeLimits,
) -> Result<AndroidTouchPolicyRecoveredStore, AndroidTouchPolicyRecoveryError> {
    durable.validate()?;

    if android_policy_journal_bytes_digest(journal_bytes) != durable.journal_bytes_digest {
        return Err(AndroidTouchPolicyRecoveryError::JournalDigestMismatch);
    }
    if android_policy_checkpoint_bytes_digest(checkpoint_bytes) != durable.checkpoint_bytes_digest {
        return Err(AndroidTouchPolicyRecoveryError::CheckpointDigestMismatch);
    }

    let journal = decode_android_policy_journal(journal_bytes, limits)?;
    let checkpoint = decode_android_policy_checkpoint(checkpoint_bytes, limits.max_bytes)?;
    if checkpoint != durable.checkpoint
        || checkpoint.checkpoint_id()? != durable.checkpoint_id
        || journal.checkpoint()? != durable.checkpoint
    {
        return Err(AndroidTouchPolicyRecoveryError::CheckpointMismatch);
    }

    // Requiring extension of the retained checkpoint when the checkpoint is the
    // final prefix also proves exact history equality at the durable boundary.
    let store = AndroidTouchPolicyJournaledStore::recover(journal, Some(&durable.checkpoint))?;
    let snapshot = store.snapshot()?;
    if !snapshot_matches_durable(&snapshot, durable) {
        return Err(AndroidTouchPolicyRecoveryError::RecoveredSnapshotMismatch);
    }

    let receipt = AndroidTouchPolicyRecoveryReceipt {
        durable_state_id: durable.state_id,
        checkpoint_id: durable.checkpoint_id,
        session_id: snapshot.session_id,
        generation: snapshot.generation,
        state_id: snapshot.state_id,
        bundle_id: snapshot.bundle_id,
        status: snapshot.status,
    };
    let _ = receipt.receipt_id()?;
    Ok(AndroidTouchPolicyRecoveredStore { store, receipt })
}

impl AndroidTouchPolicyRecoveredStore {
    pub fn recovery_receipt(&self) -> &AndroidTouchPolicyRecoveryReceipt {
        &self.receipt
    }

    pub fn snapshot(&self) -> Result<AndroidTouchPolicyStoreSnapshot, AndroidTouchPolicyRecoveryError> {
        Ok(self.store.snapshot()?)
    }

    /// Event-time use requires a second theorem after recovery. Revoked state is
    /// faithfully recovered but cannot cross this active boundary.
    pub fn certify_active(
        &self,
    ) -> Result<AndroidTouchPolicyActiveRecoveryCertificate, AndroidTouchPolicyRecoveryError> {
        let snapshot = self.store.snapshot()?;
        if snapshot.status == AndroidTouchPolicySessionStatus::Revoked {
            return Err(AndroidTouchPolicyRecoveryError::RecoveredRevoked);
        }
        if snapshot.session_id != self.receipt.session_id
            || snapshot.generation != self.receipt.generation
            || snapshot.state_id != self.receipt.state_id
            || snapshot.bundle_id != self.receipt.bundle_id
            || snapshot.status != self.receipt.status
        {
            return Err(AndroidTouchPolicyRecoveryError::RecoveryReceiptMismatch);
        }
        let expectation = self.store.expectation()?;
        let current = self.store.verify_current(&expectation)?;
        let current_certificate_id = current.certificate_id()?;
        let recovery_receipt_id = self.receipt.receipt_id()?;

        let mut hasher = blake3::Hasher::new();
        hasher.update(ACTIVE_RECOVERY_DOMAIN);
        hasher.update(recovery_receipt_id.as_bytes());
        hasher.update(self.receipt.durable_state_id.as_bytes());
        hasher.update(current_certificate_id.as_bytes());
        hasher.update(snapshot.session_id.as_bytes());
        hasher.update(&snapshot.generation.to_le_bytes());
        hasher.update(snapshot.state_id.as_bytes());
        hasher.update(snapshot.bundle_id.as_bytes());
        let certificate_id =
            AndroidTouchPolicyActiveRecoveryCertificateId(*hasher.finalize().as_bytes());

        Ok(AndroidTouchPolicyActiveRecoveryCertificate {
            certificate_id,
            recovery_receipt_id,
            durable_state_id: self.receipt.durable_state_id,
            current_certificate_id,
            session_id: snapshot.session_id,
            generation: snapshot.generation,
            state_id: snapshot.state_id,
            bundle_id: snapshot.bundle_id,
        })
    }
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
    use crate::assurance_android_policy_durability::{
        certify_android_policy_durable_state, AndroidTouchPolicyAnchorObservation,
        AndroidTouchPolicyJournalDurabilityObservation, AndroidTouchPolicyStorageDomainId,
        AndroidTouchPolicyStorageEvidenceId, AndroidTouchPolicyStorageTransactionId,
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

    fn durable(
        store: &AndroidTouchPolicyJournaledStore,
        previous: Option<&AndroidTouchPolicyDurableState>,
        seed: u8,
    ) -> (AndroidTouchPolicyDurableState, Vec<u8>, Vec<u8>) {
        let journal_bytes = encode_android_policy_journal(store.journal()).unwrap();
        let checkpoint = store.checkpoint().unwrap();
        let checkpoint_bytes = encode_android_policy_checkpoint(&checkpoint).unwrap();
        let journal_digest = android_policy_journal_bytes_digest(&journal_bytes);
        let checkpoint_digest = android_policy_checkpoint_bytes_digest(&checkpoint_bytes);
        let previous_journal_generation = previous.map_or(0, |p| p.journal_storage_generation);
        let previous_anchor_generation = previous.map_or(0, |p| p.anchor_generation);
        let previous_checkpoint = previous.map_or(AndroidTouchPolicyCheckpointId::ZERO, |p| p.checkpoint_id);
        let journal_observation = AndroidTouchPolicyJournalDurabilityObservation {
            storage_domain_id: domain(90),
            transaction_id: AndroidTouchPolicyStorageTransactionId(d(seed)),
            previous_storage_generation: previous_journal_generation,
            next_storage_generation: previous_journal_generation + 1,
            journal_bytes_digest: journal_digest,
            readback_bytes_digest: journal_digest,
            durable_prepare_evidence_id: evidence(seed.wrapping_add(1)),
            atomic_publish_evidence_id: evidence(seed.wrapping_add(2)),
            metadata_durability_evidence_id: evidence(seed.wrapping_add(3)),
            exact_readback_evidence_id: evidence(seed.wrapping_add(4)),
        };
        let journal_commit = journal_observation
            .certify_for_test(journal_digest, checkpoint.checkpoint_id().unwrap(), previous)
            .unwrap();
        let anchor_observation = AndroidTouchPolicyAnchorObservation {
            anchor_domain_id: domain(91),
            previous_anchor_generation,
            next_anchor_generation: previous_anchor_generation + 1,
            previous_checkpoint_id: previous_checkpoint,
            next_checkpoint_id: checkpoint.checkpoint_id().unwrap(),
            checkpoint_bytes_digest: checkpoint_digest,
            readback_checkpoint_bytes_digest: checkpoint_digest,
            journal_commit_id: journal_commit,
            compare_exchange_evidence_id: evidence(seed.wrapping_add(5)),
            monotonic_advance_evidence_id: evidence(seed.wrapping_add(6)),
            exact_readback_evidence_id: evidence(seed.wrapping_add(7)),
        };
        let durable = certify_android_policy_durable_state(
            previous,
            &journal_bytes,
            &checkpoint_bytes,
            AndroidTouchPolicyDecodeLimits::default(),
            &journal_observation,
            &anchor_observation,
        )
        .unwrap();
        (durable, journal_bytes, checkpoint_bytes)
    }

    #[test]
    fn exact_durable_active_state_recovers_and_reestablishes_currentness() {
        let mut store = AndroidTouchPolicyJournaledStore::new();
        store.install(&bundle(20), auth(40), nonce(41)).unwrap();
        let (durable, journal_bytes, checkpoint_bytes) = durable(&store, None, 100);
        let recovered = recover_android_policy_durable_state(
            &durable,
            &journal_bytes,
            &checkpoint_bytes,
            AndroidTouchPolicyDecodeLimits::default(),
        )
        .unwrap();
        assert_eq!(recovered.snapshot().unwrap(), store.snapshot().unwrap());
        let active = recovered.certify_active().unwrap();
        assert_eq!(active.durable_state_id, durable.state_id);
        assert!(!active.certificate_id.is_zero());
    }

    #[test]
    fn journal_bytes_not_named_by_durable_state_are_rejected() {
        let mut store = AndroidTouchPolicyJournaledStore::new();
        store.install(&bundle(20), auth(40), nonce(41)).unwrap();
        let (durable, mut journal_bytes, checkpoint_bytes) = durable(&store, None, 100);
        journal_bytes[0] ^= 1;
        assert_eq!(
            recover_android_policy_durable_state(
                &durable,
                &journal_bytes,
                &checkpoint_bytes,
                AndroidTouchPolicyDecodeLimits::default(),
            ),
            Err(AndroidTouchPolicyRecoveryError::JournalDigestMismatch)
        );
    }

    #[test]
    fn checkpoint_bytes_not_named_by_durable_state_are_rejected() {
        let mut store = AndroidTouchPolicyJournaledStore::new();
        store.install(&bundle(20), auth(40), nonce(41)).unwrap();
        let (durable, journal_bytes, mut checkpoint_bytes) = durable(&store, None, 100);
        checkpoint_bytes[0] ^= 1;
        assert_eq!(
            recover_android_policy_durable_state(
                &durable,
                &journal_bytes,
                &checkpoint_bytes,
                AndroidTouchPolicyDecodeLimits::default(),
            ),
            Err(AndroidTouchPolicyRecoveryError::CheckpointDigestMismatch)
        );
    }

    #[test]
    fn durably_revoked_state_recovers_but_cannot_become_active() {
        let mut store = AndroidTouchPolicyJournaledStore::new();
        store.install(&bundle(20), auth(40), nonce(41)).unwrap();
        let (first, _, _) = durable(&store, None, 100);
        let current = store.snapshot().unwrap();
        store.revoke(current.state_id, auth(42)).unwrap();
        let (revoked, journal_bytes, checkpoint_bytes) = durable(&store, Some(&first), 110);
        assert_eq!(revoked.checkpoint.status, AndroidTouchPolicySessionStatus::Revoked);

        let recovered = recover_android_policy_durable_state(
            &revoked,
            &journal_bytes,
            &checkpoint_bytes,
            AndroidTouchPolicyDecodeLimits::default(),
        )
        .unwrap();
        assert_eq!(
            recovered.certify_active(),
            Err(AndroidTouchPolicyRecoveryError::RecoveredRevoked)
        );
    }
}
