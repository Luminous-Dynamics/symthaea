// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! QUAL-ANDROIDPOLICYJOURNAL-532: append-only Android policy recovery history.
//!
//! QUAL-ANDROIDPOLICYSTORE-531 owns the active bundle and 530 session state in one
//! Rust object, but that ownership is process-local. A restart must not silently
//! resurrect an older policy, forget a revocation, accept a forked history, or
//! allow an old transition authorization to be reused after intervening changes.
//!
//! This tranche wraps 531 with an append-only semantic journal. Every entry carries
//! the exact 531 transition, the complete 529 bundle needed to replay that step,
//! and (for installation only) the session nonce. Recovery replays every record
//! through the real 531 APIs and requires byte-independent semantic identity of the
//! resulting transition. Authorization identities are unique across the full
//! journal, not merely the immediately preceding state.
//!
//! The journal/checkpoint objects define recovery semantics, not durable storage.
//! A later platform adapter must persist them atomically and anchor checkpoints in
//! rollback-resistant storage. A checkpoint is only a rollback/fork barrier when
//! the caller retains it in a trust domain the candidate journal cannot rewrite.

use core::fmt;
use std::collections::BTreeSet;

use crate::assurance_android_policy_bundle::{
    AndroidTouchPolicyBundle, AndroidTouchPolicyBundleError, AndroidTouchPolicyBundleId,
};
use crate::assurance_android_policy_session::{
    AndroidTouchPolicyCurrentCertificate, AndroidTouchPolicySessionAuthorizationId,
    AndroidTouchPolicySessionError, AndroidTouchPolicySessionExpectation,
    AndroidTouchPolicySessionId, AndroidTouchPolicySessionNonce,
    AndroidTouchPolicySessionStateId, AndroidTouchPolicySessionStatus,
    AndroidTouchPolicySessionTransitionKind,
};
use crate::assurance_android_policy_store::{
    AndroidTouchPolicySessionStore, AndroidTouchPolicyStoreError, AndroidTouchPolicyStoreSnapshot,
    AndroidTouchPolicyStoreTransition,
};

const ENTRY_DOMAIN: &[u8] =
    b"symthaea.soma.presentation.v1/android-touch-policy-journal-entry\0";
const JOURNAL_DOMAIN: &[u8] =
    b"symthaea.soma.presentation.v1/android-touch-policy-journal-root\0";
const AUTH_SET_DOMAIN: &[u8] =
    b"symthaea.soma.presentation.v1/android-touch-policy-journal-auth-set\0";
const CHECKPOINT_DOMAIN: &[u8] =
    b"symthaea.soma.presentation.v1/android-touch-policy-journal-checkpoint\0";

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

digest_id!(AndroidTouchPolicyJournalEntryId);
digest_id!(AndroidTouchPolicyJournalRoot);
digest_id!(AndroidTouchPolicyAuthorizationSetRoot);
digest_id!(AndroidTouchPolicyCheckpointId);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AndroidTouchPolicyJournalEntry {
    pub sequence: u64,
    pub predecessor_entry_id: AndroidTouchPolicyJournalEntryId,
    pub transition: AndroidTouchPolicyStoreTransition,
    pub bundle: AndroidTouchPolicyBundle,
    /// Nonzero only for the initial Install record. Later transitions must use ZERO.
    pub install_nonce: AndroidTouchPolicySessionNonce,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AndroidTouchPolicyJournalCheckpoint {
    pub record_count: u64,
    pub last_entry_id: AndroidTouchPolicyJournalEntryId,
    pub journal_root: AndroidTouchPolicyJournalRoot,
    pub authorization_set_root: AndroidTouchPolicyAuthorizationSetRoot,
    pub session_id: AndroidTouchPolicySessionId,
    pub generation: u64,
    pub state_id: AndroidTouchPolicySessionStateId,
    pub bundle_id: AndroidTouchPolicyBundleId,
    pub status: AndroidTouchPolicySessionStatus,
}

#[derive(Clone, Debug, Default)]
pub struct AndroidTouchPolicyJournal {
    entries: Vec<AndroidTouchPolicyJournalEntry>,
}

#[derive(Clone, Debug)]
pub struct AndroidTouchPolicyJournaledStore {
    store: AndroidTouchPolicySessionStore,
    journal: AndroidTouchPolicyJournal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AndroidTouchPolicyJournalError {
    EmptyJournal,
    RecordCountOverflow,
    SequenceMismatch,
    InstallMustBeFirst,
    MissingInstall,
    UnexpectedInstallNonce,
    MissingInstallNonce,
    PredecessorEntryMismatch,
    SessionMismatch,
    PriorStateMismatch,
    PriorGenerationMismatch,
    PriorBundleMismatch,
    SnapshotMismatch,
    BundleMismatch,
    StatusMismatch,
    AuthorizationReplay,
    AppendAfterRevocation,
    ReceiptIdMismatch,
    ReplayTransitionMismatch,
    CheckpointRecordCountInvalid,
    CheckpointMismatch,
    RollbackDetected,
    ForkDetected,
    StoreStateMismatch,
    Session(AndroidTouchPolicySessionError),
    Store(AndroidTouchPolicyStoreError),
    Bundle(AndroidTouchPolicyBundleError),
}

impl fmt::Display for AndroidTouchPolicyJournalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for AndroidTouchPolicyJournalError {}

impl From<AndroidTouchPolicySessionError> for AndroidTouchPolicyJournalError {
    fn from(value: AndroidTouchPolicySessionError) -> Self {
        Self::Session(value)
    }
}

impl From<AndroidTouchPolicyStoreError> for AndroidTouchPolicyJournalError {
    fn from(value: AndroidTouchPolicyStoreError) -> Self {
        Self::Store(value)
    }
}

impl From<AndroidTouchPolicyBundleError> for AndroidTouchPolicyJournalError {
    fn from(value: AndroidTouchPolicyBundleError) -> Self {
        Self::Bundle(value)
    }
}

fn status_for(kind: AndroidTouchPolicySessionTransitionKind) -> AndroidTouchPolicySessionStatus {
    match kind {
        AndroidTouchPolicySessionTransitionKind::Install
        | AndroidTouchPolicySessionTransitionKind::Replace => {
            AndroidTouchPolicySessionStatus::Active
        }
        AndroidTouchPolicySessionTransitionKind::Revoke => {
            AndroidTouchPolicySessionStatus::Revoked
        }
    }
}

impl AndroidTouchPolicyJournalEntry {
    pub fn entry_id(&self) -> Result<AndroidTouchPolicyJournalEntryId, AndroidTouchPolicyJournalError> {
        self.transition.receipt.validate()?;
        let receipt_id = self.transition.receipt.receipt_id()?;
        if receipt_id != self.transition.receipt_id {
            return Err(AndroidTouchPolicyJournalError::ReceiptIdMismatch);
        }
        let bundle_id = self.bundle.bundle_id()?;
        if bundle_id != self.transition.snapshot.bundle_id
            || bundle_id != self.transition.receipt.next_bundle_id
        {
            return Err(AndroidTouchPolicyJournalError::BundleMismatch);
        }

        let mut hasher = blake3::Hasher::new();
        hasher.update(ENTRY_DOMAIN);
        hasher.update(&self.sequence.to_le_bytes());
        hasher.update(self.predecessor_entry_id.as_bytes());
        hasher.update(receipt_id.as_bytes());
        hasher.update(bundle_id.as_bytes());
        hasher.update(self.install_nonce.as_bytes());
        Ok(AndroidTouchPolicyJournalEntryId(*hasher.finalize().as_bytes()))
    }
}

impl AndroidTouchPolicyJournalCheckpoint {
    pub fn checkpoint_id(
        &self,
    ) -> Result<AndroidTouchPolicyCheckpointId, AndroidTouchPolicyJournalError> {
        if self.record_count == 0
            || self.last_entry_id.is_zero()
            || self.journal_root.is_zero()
            || self.authorization_set_root.is_zero()
            || self.session_id.is_zero()
            || self.generation == 0
            || self.state_id.is_zero()
            || self.bundle_id.is_zero()
        {
            return Err(AndroidTouchPolicyJournalError::CheckpointMismatch);
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(CHECKPOINT_DOMAIN);
        hasher.update(&self.record_count.to_le_bytes());
        hasher.update(self.last_entry_id.as_bytes());
        hasher.update(self.journal_root.as_bytes());
        hasher.update(self.authorization_set_root.as_bytes());
        hasher.update(self.session_id.as_bytes());
        hasher.update(&self.generation.to_le_bytes());
        hasher.update(self.state_id.as_bytes());
        hasher.update(self.bundle_id.as_bytes());
        hasher.update(&[self.status as u8]);
        Ok(AndroidTouchPolicyCheckpointId(*hasher.finalize().as_bytes()))
    }
}

impl AndroidTouchPolicyJournal {
    pub const fn new() -> Self {
        Self { entries: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &[AndroidTouchPolicyJournalEntry] {
        &self.entries
    }

    pub fn contains_authorization(
        &self,
        authorization_id: AndroidTouchPolicySessionAuthorizationId,
    ) -> bool {
        self.entries
            .iter()
            .any(|entry| entry.transition.receipt.authorization_id == authorization_id)
    }

    pub fn append(
        &mut self,
        transition: AndroidTouchPolicyStoreTransition,
        bundle: AndroidTouchPolicyBundle,
        install_nonce: AndroidTouchPolicySessionNonce,
    ) -> Result<AndroidTouchPolicyJournalEntryId, AndroidTouchPolicyJournalError> {
        if self.contains_authorization(transition.receipt.authorization_id) {
            return Err(AndroidTouchPolicyJournalError::AuthorizationReplay);
        }

        let sequence = u64::try_from(self.entries.len())
            .map_err(|_| AndroidTouchPolicyJournalError::RecordCountOverflow)?
            .checked_add(1)
            .ok_or(AndroidTouchPolicyJournalError::RecordCountOverflow)?;
        let predecessor_entry_id = match self.entries.last() {
            Some(previous) => previous.entry_id()?,
            None => AndroidTouchPolicyJournalEntryId::ZERO,
        };
        let entry = AndroidTouchPolicyJournalEntry {
            sequence,
            predecessor_entry_id,
            transition,
            bundle,
            install_nonce,
        };
        self.validate_candidate(&entry)?;
        let entry_id = entry.entry_id()?;
        self.entries.push(entry);
        Ok(entry_id)
    }

    pub fn validate(&self) -> Result<(), AndroidTouchPolicyJournalError> {
        if self.entries.is_empty() {
            return Ok(());
        }
        let mut prefix = AndroidTouchPolicyJournal::new();
        for entry in &self.entries {
            if prefix.contains_authorization(entry.transition.receipt.authorization_id) {
                return Err(AndroidTouchPolicyJournalError::AuthorizationReplay);
            }
            prefix.validate_candidate(entry)?;
            prefix.entries.push(*entry);
        }
        // Semantic replay is deliberately part of journal validity.
        let _ = self.recover_store_unchecked()?;
        Ok(())
    }

    pub fn recover_store(
        &self,
    ) -> Result<AndroidTouchPolicySessionStore, AndroidTouchPolicyJournalError> {
        self.validate()?;
        self.recover_store_unchecked()
    }

    pub fn checkpoint(
        &self,
    ) -> Result<AndroidTouchPolicyJournalCheckpoint, AndroidTouchPolicyJournalError> {
        self.checkpoint_at(self.entries.len())
    }

    pub fn checkpoint_at(
        &self,
        count: usize,
    ) -> Result<AndroidTouchPolicyJournalCheckpoint, AndroidTouchPolicyJournalError> {
        if count == 0 || count > self.entries.len() {
            return Err(AndroidTouchPolicyJournalError::CheckpointRecordCountInvalid);
        }
        let prefix = AndroidTouchPolicyJournal {
            entries: self.entries[..count].to_vec(),
        };
        prefix.validate()?;
        let store = prefix.recover_store_unchecked()?;
        let snapshot = store.snapshot()?;
        let record_count = u64::try_from(count)
            .map_err(|_| AndroidTouchPolicyJournalError::RecordCountOverflow)?;
        let last_entry_id = prefix
            .entries
            .last()
            .ok_or(AndroidTouchPolicyJournalError::EmptyJournal)?
            .entry_id()?;
        Ok(AndroidTouchPolicyJournalCheckpoint {
            record_count,
            last_entry_id,
            journal_root: prefix.journal_root()?,
            authorization_set_root: prefix.authorization_set_root()?,
            session_id: snapshot.session_id,
            generation: snapshot.generation,
            state_id: snapshot.state_id,
            bundle_id: snapshot.bundle_id,
            status: snapshot.status,
        })
    }

    /// Prove that this candidate history extends an independently retained
    /// checkpoint. A shorter candidate is rollback; a different prefix is a fork.
    pub fn verify_extends(
        &self,
        anchor: &AndroidTouchPolicyJournalCheckpoint,
    ) -> Result<(), AndroidTouchPolicyJournalError> {
        let anchor_count = usize::try_from(anchor.record_count)
            .map_err(|_| AndroidTouchPolicyJournalError::CheckpointRecordCountInvalid)?;
        if self.entries.len() < anchor_count {
            return Err(AndroidTouchPolicyJournalError::RollbackDetected);
        }
        if anchor_count == 0 {
            return Err(AndroidTouchPolicyJournalError::CheckpointRecordCountInvalid);
        }
        let candidate_prefix = self.checkpoint_at(anchor_count)?;
        if candidate_prefix.last_entry_id != anchor.last_entry_id {
            return Err(AndroidTouchPolicyJournalError::ForkDetected);
        }
        if candidate_prefix != *anchor
            || candidate_prefix.checkpoint_id()? != anchor.checkpoint_id()?
        {
            return Err(AndroidTouchPolicyJournalError::CheckpointMismatch);
        }
        Ok(())
    }

    pub fn journal_root(&self) -> Result<AndroidTouchPolicyJournalRoot, AndroidTouchPolicyJournalError> {
        if self.entries.is_empty() {
            return Err(AndroidTouchPolicyJournalError::EmptyJournal);
        }
        let count = u64::try_from(self.entries.len())
            .map_err(|_| AndroidTouchPolicyJournalError::RecordCountOverflow)?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(JOURNAL_DOMAIN);
        hasher.update(&count.to_le_bytes());
        for entry in &self.entries {
            hasher.update(entry.entry_id()?.as_bytes());
        }
        Ok(AndroidTouchPolicyJournalRoot(*hasher.finalize().as_bytes()))
    }

    pub fn authorization_set_root(
        &self,
    ) -> Result<AndroidTouchPolicyAuthorizationSetRoot, AndroidTouchPolicyJournalError> {
        if self.entries.is_empty() {
            return Err(AndroidTouchPolicyJournalError::EmptyJournal);
        }
        let ids: BTreeSet<_> = self
            .entries
            .iter()
            .map(|entry| entry.transition.receipt.authorization_id)
            .collect();
        if ids.len() != self.entries.len() {
            return Err(AndroidTouchPolicyJournalError::AuthorizationReplay);
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(AUTH_SET_DOMAIN);
        hasher.update(&(ids.len() as u64).to_le_bytes());
        for id in ids {
            hasher.update(id.as_bytes());
        }
        Ok(AndroidTouchPolicyAuthorizationSetRoot(*hasher.finalize().as_bytes()))
    }

    fn validate_candidate(
        &self,
        entry: &AndroidTouchPolicyJournalEntry,
    ) -> Result<(), AndroidTouchPolicyJournalError> {
        let expected_sequence = u64::try_from(self.entries.len())
            .map_err(|_| AndroidTouchPolicyJournalError::RecordCountOverflow)?
            .checked_add(1)
            .ok_or(AndroidTouchPolicyJournalError::RecordCountOverflow)?;
        if entry.sequence != expected_sequence {
            return Err(AndroidTouchPolicyJournalError::SequenceMismatch);
        }

        entry.transition.receipt.validate()?;
        if entry.transition.receipt.receipt_id()? != entry.transition.receipt_id {
            return Err(AndroidTouchPolicyJournalError::ReceiptIdMismatch);
        }
        let receipt = &entry.transition.receipt;
        let snapshot = &entry.transition.snapshot;
        if snapshot.session_id != receipt.session_id
            || snapshot.generation != receipt.next_generation
            || snapshot.state_id != receipt.next_state_id
            || snapshot.bundle_id != receipt.next_bundle_id
        {
            return Err(AndroidTouchPolicyJournalError::SnapshotMismatch);
        }
        if snapshot.status != status_for(receipt.kind) {
            return Err(AndroidTouchPolicyJournalError::StatusMismatch);
        }
        if entry.bundle.bundle_id()? != receipt.next_bundle_id {
            return Err(AndroidTouchPolicyJournalError::BundleMismatch);
        }

        match self.entries.last() {
            None => {
                if receipt.kind != AndroidTouchPolicySessionTransitionKind::Install {
                    return Err(AndroidTouchPolicyJournalError::MissingInstall);
                }
                if !entry.predecessor_entry_id.is_zero() {
                    return Err(AndroidTouchPolicyJournalError::PredecessorEntryMismatch);
                }
                if entry.install_nonce.is_zero() {
                    return Err(AndroidTouchPolicyJournalError::MissingInstallNonce);
                }
            }
            Some(previous) => {
                if receipt.kind == AndroidTouchPolicySessionTransitionKind::Install {
                    return Err(AndroidTouchPolicyJournalError::InstallMustBeFirst);
                }
                if !entry.install_nonce.is_zero() {
                    return Err(AndroidTouchPolicyJournalError::UnexpectedInstallNonce);
                }
                if previous.transition.snapshot.status == AndroidTouchPolicySessionStatus::Revoked {
                    return Err(AndroidTouchPolicyJournalError::AppendAfterRevocation);
                }
                if entry.predecessor_entry_id != previous.entry_id()? {
                    return Err(AndroidTouchPolicyJournalError::PredecessorEntryMismatch);
                }
                let prior = &previous.transition.snapshot;
                if receipt.session_id != prior.session_id {
                    return Err(AndroidTouchPolicyJournalError::SessionMismatch);
                }
                if receipt.prior_state_id != prior.state_id {
                    return Err(AndroidTouchPolicyJournalError::PriorStateMismatch);
                }
                if receipt.prior_generation != prior.generation {
                    return Err(AndroidTouchPolicyJournalError::PriorGenerationMismatch);
                }
                if receipt.prior_bundle_id != prior.bundle_id {
                    return Err(AndroidTouchPolicyJournalError::PriorBundleMismatch);
                }
            }
        }
        Ok(())
    }

    fn recover_store_unchecked(
        &self,
    ) -> Result<AndroidTouchPolicySessionStore, AndroidTouchPolicyJournalError> {
        if self.entries.is_empty() {
            return Err(AndroidTouchPolicyJournalError::EmptyJournal);
        }
        let mut store = AndroidTouchPolicySessionStore::new();
        let mut used = BTreeSet::new();

        for entry in &self.entries {
            let authorization_id = entry.transition.receipt.authorization_id;
            if !used.insert(authorization_id) {
                return Err(AndroidTouchPolicyJournalError::AuthorizationReplay);
            }
            let replayed = match entry.transition.receipt.kind {
                AndroidTouchPolicySessionTransitionKind::Install => store.install(
                    &entry.bundle,
                    authorization_id,
                    entry.install_nonce,
                )?,
                AndroidTouchPolicySessionTransitionKind::Replace => store.replace(
                    entry.transition.receipt.prior_state_id,
                    &entry.bundle,
                    authorization_id,
                )?,
                AndroidTouchPolicySessionTransitionKind::Revoke => store.revoke(
                    entry.transition.receipt.prior_state_id,
                    authorization_id,
                )?,
            };
            if replayed != entry.transition {
                return Err(AndroidTouchPolicyJournalError::ReplayTransitionMismatch);
            }
        }
        store.validate()?;
        Ok(store)
    }
}

impl Default for AndroidTouchPolicyJournaledStore {
    fn default() -> Self {
        Self::new()
    }
}

impl AndroidTouchPolicyJournaledStore {
    pub const fn new() -> Self {
        Self {
            store: AndroidTouchPolicySessionStore::new(),
            journal: AndroidTouchPolicyJournal::new(),
        }
    }

    pub fn recover(
        journal: AndroidTouchPolicyJournal,
        anchor: Option<&AndroidTouchPolicyJournalCheckpoint>,
    ) -> Result<Self, AndroidTouchPolicyJournalError> {
        journal.validate()?;
        if let Some(anchor) = anchor {
            journal.verify_extends(anchor)?;
        }
        let store = journal.recover_store_unchecked()?;
        let recovered = Self { store, journal };
        recovered.validate()?;
        Ok(recovered)
    }

    pub fn validate(&self) -> Result<(), AndroidTouchPolicyJournalError> {
        self.store.validate()?;
        if self.store.is_empty() {
            if self.journal.is_empty() {
                return Ok(());
            }
            return Err(AndroidTouchPolicyJournalError::StoreStateMismatch);
        }
        if self.journal.is_empty() {
            return Err(AndroidTouchPolicyJournalError::StoreStateMismatch);
        }
        let recovered = self.journal.recover_store()?;
        if recovered.snapshot()? != self.store.snapshot()? {
            return Err(AndroidTouchPolicyJournalError::StoreStateMismatch);
        }
        Ok(())
    }

    pub fn snapshot(&self) -> Result<AndroidTouchPolicyStoreSnapshot, AndroidTouchPolicyJournalError> {
        self.validate()?;
        Ok(self.store.snapshot()?)
    }

    pub fn expectation(
        &self,
    ) -> Result<AndroidTouchPolicySessionExpectation, AndroidTouchPolicyJournalError> {
        self.validate()?;
        Ok(self.store.expectation()?)
    }

    pub fn verify_current(
        &self,
        expected: &AndroidTouchPolicySessionExpectation,
    ) -> Result<AndroidTouchPolicyCurrentCertificate, AndroidTouchPolicyJournalError> {
        self.validate()?;
        Ok(self.store.verify_current(expected)?)
    }

    pub fn checkpoint(
        &self,
    ) -> Result<AndroidTouchPolicyJournalCheckpoint, AndroidTouchPolicyJournalError> {
        self.validate()?;
        self.journal.checkpoint()
    }

    pub fn journal(&self) -> &AndroidTouchPolicyJournal {
        &self.journal
    }

    pub fn install(
        &mut self,
        bundle: &AndroidTouchPolicyBundle,
        authorization_id: AndroidTouchPolicySessionAuthorizationId,
        nonce: AndroidTouchPolicySessionNonce,
    ) -> Result<AndroidTouchPolicyStoreTransition, AndroidTouchPolicyJournalError> {
        self.validate()?;
        if self.journal.contains_authorization(authorization_id) {
            return Err(AndroidTouchPolicyJournalError::AuthorizationReplay);
        }

        let mut candidate_store = self.store.clone();
        let transition = candidate_store.install(bundle, authorization_id, nonce)?;
        let mut candidate_journal = self.journal.clone();
        candidate_journal.append(transition, *bundle, nonce)?;
        let candidate = Self {
            store: candidate_store,
            journal: candidate_journal,
        };
        candidate.validate()?;
        *self = candidate;
        Ok(transition)
    }

    pub fn replace(
        &mut self,
        expected_state_id: AndroidTouchPolicySessionStateId,
        bundle: &AndroidTouchPolicyBundle,
        authorization_id: AndroidTouchPolicySessionAuthorizationId,
    ) -> Result<AndroidTouchPolicyStoreTransition, AndroidTouchPolicyJournalError> {
        self.validate()?;
        if self.journal.contains_authorization(authorization_id) {
            return Err(AndroidTouchPolicyJournalError::AuthorizationReplay);
        }

        let mut candidate_store = self.store.clone();
        let transition = candidate_store.replace(expected_state_id, bundle, authorization_id)?;
        let mut candidate_journal = self.journal.clone();
        candidate_journal.append(
            transition,
            *bundle,
            AndroidTouchPolicySessionNonce::ZERO,
        )?;
        let candidate = Self {
            store: candidate_store,
            journal: candidate_journal,
        };
        candidate.validate()?;
        *self = candidate;
        Ok(transition)
    }

    pub fn revoke(
        &mut self,
        expected_state_id: AndroidTouchPolicySessionStateId,
        authorization_id: AndroidTouchPolicySessionAuthorizationId,
    ) -> Result<AndroidTouchPolicyStoreTransition, AndroidTouchPolicyJournalError> {
        self.validate()?;
        if self.journal.contains_authorization(authorization_id) {
            return Err(AndroidTouchPolicyJournalError::AuthorizationReplay);
        }

        // The last journal entry owns the exact current bundle needed to replay
        // revocation after restart; event-time callers never supply it.
        let current_bundle = self
            .journal
            .entries
            .last()
            .ok_or(AndroidTouchPolicyJournalError::EmptyJournal)?
            .bundle;
        let mut candidate_store = self.store.clone();
        let transition = candidate_store.revoke(expected_state_id, authorization_id)?;
        let mut candidate_journal = self.journal.clone();
        candidate_journal.append(
            transition,
            current_bundle,
            AndroidTouchPolicySessionNonce::ZERO,
        )?;
        let candidate = Self {
            store: candidate_store,
            journal: candidate_journal,
        };
        candidate.validate()?;
        *self = candidate;
        Ok(transition)
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

    #[test]
    fn journaled_store_replays_exact_install_replace_and_revoke() {
        let mut store = AndroidTouchPolicyJournaledStore::new();
        store.install(&bundle(20), auth(40), nonce(41)).unwrap();
        let first = store.snapshot().unwrap();
        store.replace(first.state_id, &bundle(30), auth(42)).unwrap();
        let second = store.snapshot().unwrap();
        store.revoke(second.state_id, auth(43)).unwrap();

        let checkpoint = store.checkpoint().unwrap();
        assert_eq!(checkpoint.record_count, 3);
        assert_eq!(checkpoint.status, AndroidTouchPolicySessionStatus::Revoked);
        assert!(!checkpoint.checkpoint_id().unwrap().is_zero());

        let recovered = AndroidTouchPolicyJournaledStore::recover(
            store.journal().clone(),
            Some(&checkpoint),
        )
        .unwrap();
        assert_eq!(recovered.snapshot().unwrap(), store.snapshot().unwrap());
    }

    #[test]
    fn non_adjacent_authorization_replay_is_rejected() {
        let mut store = AndroidTouchPolicyJournaledStore::new();
        store.install(&bundle(20), auth(40), nonce(41)).unwrap();
        let s1 = store.snapshot().unwrap();
        store.replace(s1.state_id, &bundle(30), auth(42)).unwrap();
        let s2 = store.snapshot().unwrap();
        store.replace(s2.state_id, &bundle(35), auth(43)).unwrap();
        let s3 = store.snapshot().unwrap();

        assert_eq!(
            store.replace(s3.state_id, &bundle(50), auth(42)),
            Err(AndroidTouchPolicyJournalError::AuthorizationReplay)
        );
    }

    #[test]
    fn stale_transition_leaves_store_and_journal_unchanged() {
        let mut store = AndroidTouchPolicyJournaledStore::new();
        store.install(&bundle(20), auth(40), nonce(41)).unwrap();
        let before_snapshot = store.snapshot().unwrap();
        let before_root = store.journal().journal_root().unwrap();

        assert!(store
            .replace(AndroidTouchPolicySessionStateId(d(99)), &bundle(30), auth(42))
            .is_err());
        assert_eq!(store.snapshot().unwrap(), before_snapshot);
        assert_eq!(store.journal().journal_root().unwrap(), before_root);
    }

    #[test]
    fn retained_checkpoint_rejects_rollback_and_fork() {
        let mut canonical = AndroidTouchPolicyJournaledStore::new();
        canonical.install(&bundle(20), auth(40), nonce(41)).unwrap();
        let first_anchor = canonical.checkpoint().unwrap();
        let first = canonical.snapshot().unwrap();
        canonical.replace(first.state_id, &bundle(30), auth(42)).unwrap();
        let second_anchor = canonical.checkpoint().unwrap();

        let rollback = AndroidTouchPolicyJournal {
            entries: canonical.journal().entries[..1].to_vec(),
        };
        assert_eq!(
            rollback.verify_extends(&second_anchor),
            Err(AndroidTouchPolicyJournalError::RollbackDetected)
        );

        let mut fork = AndroidTouchPolicyJournaledStore::recover(
            AndroidTouchPolicyJournal {
                entries: canonical.journal().entries[..1].to_vec(),
            },
            Some(&first_anchor),
        )
        .unwrap();
        let fork_state = fork.snapshot().unwrap();
        fork.replace(fork_state.state_id, &bundle(55), auth(52)).unwrap();
        assert_eq!(
            fork.journal().verify_extends(&second_anchor),
            Err(AndroidTouchPolicyJournalError::ForkDetected)
        );
    }

    #[test]
    fn tampered_record_cannot_replay_into_current_state() {
        let mut store = AndroidTouchPolicyJournaledStore::new();
        store.install(&bundle(20), auth(40), nonce(41)).unwrap();
        let first = store.snapshot().unwrap();
        store.replace(first.state_id, &bundle(30), auth(42)).unwrap();

        let mut tampered = store.journal().clone();
        tampered.entries[1].transition.snapshot.generation += 1;
        assert!(matches!(
            tampered.recover_store(),
            Err(AndroidTouchPolicyJournalError::SnapshotMismatch)
                | Err(AndroidTouchPolicyJournalError::ReplayTransitionMismatch)
        ));
    }

    #[test]
    fn revoked_history_cannot_be_extended() {
        let mut store = AndroidTouchPolicyJournaledStore::new();
        store.install(&bundle(20), auth(40), nonce(41)).unwrap();
        let current = store.snapshot().unwrap();
        store.revoke(current.state_id, auth(42)).unwrap();
        let before = store.journal().journal_root().unwrap();

        let revoked = store.snapshot().unwrap();
        assert!(store
            .replace(revoked.state_id, &bundle(30), auth(43))
            .is_err());
        assert_eq!(store.journal().journal_root().unwrap(), before);
    }
}
