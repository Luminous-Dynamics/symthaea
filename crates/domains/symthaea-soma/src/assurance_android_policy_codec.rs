// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! QUAL-ANDROIDPOLICYCODEC-533: canonical persistence bytes for Android policy state.
//!
//! QUAL-ANDROIDPOLICYJOURNAL-532 defines replay-safe recovery semantics, but a
//! persistence theorem also needs one unambiguous byte representation. This module
//! intentionally does not use serde, bincode, JSON, platform parceling, or defaults.
//! Authority-bearing bytes use fixed widths, little-endian integers, strict 0/1
//! booleans, closed one-byte enums, exact 32-byte identities, explicit magic/version,
//! hard decode ceilings, and exact re-encode equality after semantic replay.
//!
//! Decoding a journal does not merely reconstruct structs: every decoded record is
//! appended through 532 and the resulting journal is semantically replayed through
//! 531 before the bytes are accepted. Alternate encodings are rejected rather than
//! normalized.

use core::fmt;

use symthaea_core::assurance_interaction_continuity::InputAttesterId;
use symthaea_core::assurance_trusted_surface::TrustedSurfaceId;

use crate::assurance_android_attention::{
    AndroidAttentionPolicyContextId, AndroidAttentionPolicyStateRoot,
    AndroidAttentionRequirement,
};
use crate::assurance_android_attention_policy::AndroidAttentionPolicyAdmission;
use crate::assurance_android_motion_event::AndroidMotionEventRequirement;
use crate::assurance_android_motion_policy::AndroidMotionPolicyAdmission;
use crate::assurance_android_policy_bundle::{
    AndroidTouchPolicyBundle, AndroidTouchPolicyBundleError,
};
use crate::assurance_android_policy_journal::{
    AndroidTouchPolicyAuthorizationSetRoot, AndroidTouchPolicyJournal,
    AndroidTouchPolicyJournalCheckpoint, AndroidTouchPolicyJournalEntry,
    AndroidTouchPolicyJournalEntryId, AndroidTouchPolicyJournalError,
    AndroidTouchPolicyJournalRoot,
};
use crate::assurance_android_policy_session::{
    AndroidTouchPolicySessionAuthorizationId, AndroidTouchPolicySessionId,
    AndroidTouchPolicySessionNonce, AndroidTouchPolicySessionStateId,
    AndroidTouchPolicySessionStatus, AndroidTouchPolicySessionTransitionKind,
    AndroidTouchPolicySessionTransitionReceipt, AndroidTouchPolicySessionTransitionReceiptId,
};
use crate::assurance_android_policy_store::{
    AndroidTouchPolicyStoreSnapshot, AndroidTouchPolicyStoreTransition,
};
use crate::assurance_ingress_policy::{
    IngressAdmissionPolicy, IngressPolicyContextId, IngressPolicyStateRoot,
};
use crate::assurance_platform_ingress::{PlatformIngressKind, PlatformIngressProfile};
use crate::assurance_soma_interaction::{ScreenCaptureProfileId, TouchInputProfileId};

const JOURNAL_MAGIC: [u8; 8] = *b"STPJNL01";
const CHECKPOINT_MAGIC: [u8; 8] = *b"STPCHK01";
pub const ANDROID_POLICY_CODEC_VERSION: u32 = 1;
pub const DEFAULT_MAX_JOURNAL_RECORDS: u64 = 65_536;
pub const DEFAULT_MAX_JOURNAL_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AndroidTouchPolicyDecodeLimits {
    pub max_records: u64,
    pub max_bytes: usize,
}

impl Default for AndroidTouchPolicyDecodeLimits {
    fn default() -> Self {
        Self {
            max_records: DEFAULT_MAX_JOURNAL_RECORDS,
            max_bytes: DEFAULT_MAX_JOURNAL_BYTES,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AndroidTouchPolicyCodecError {
    ZeroDecodeRecordLimit,
    ZeroDecodeByteLimit,
    InputTooLarge,
    Truncated,
    BadMagic,
    UnsupportedVersion,
    RecordCountTooLarge,
    RecordCountNotRepresentable,
    InvalidBoolean(u8),
    InvalidPlatform(u8),
    InvalidTransitionKind(u8),
    InvalidSessionStatus(u8),
    TrailingBytes,
    NonCanonical,
    Journal(AndroidTouchPolicyJournalError),
    Bundle(AndroidTouchPolicyBundleError),
}

impl fmt::Display for AndroidTouchPolicyCodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for AndroidTouchPolicyCodecError {}

impl From<AndroidTouchPolicyJournalError> for AndroidTouchPolicyCodecError {
    fn from(value: AndroidTouchPolicyJournalError) -> Self {
        Self::Journal(value)
    }
}

impl From<AndroidTouchPolicyBundleError> for AndroidTouchPolicyCodecError {
    fn from(value: AndroidTouchPolicyBundleError) -> Self {
        Self::Bundle(value)
    }
}

struct Writer {
    bytes: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    fn raw(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(bytes);
    }

    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn bool(&mut self, value: bool) {
        self.u8(u8::from(value));
    }

    fn u32(&mut self, value: u32) {
        self.raw(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.raw(&value.to_le_bytes());
    }

    fn id(&mut self, value: &[u8; 32]) {
        self.raw(value);
    }

    fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], AndroidTouchPolicyCodecError> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or(AndroidTouchPolicyCodecError::Truncated)?;
        let out = self
            .bytes
            .get(self.pos..end)
            .ok_or(AndroidTouchPolicyCodecError::Truncated)?;
        self.pos = end;
        Ok(out)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], AndroidTouchPolicyCodecError> {
        let mut out = [0u8; N];
        out.copy_from_slice(self.take(N)?);
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8, AndroidTouchPolicyCodecError> {
        Ok(self.take(1)?[0])
    }

    fn bool(&mut self) -> Result<bool, AndroidTouchPolicyCodecError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            other => Err(AndroidTouchPolicyCodecError::InvalidBoolean(other)),
        }
    }

    fn u32(&mut self) -> Result<u32, AndroidTouchPolicyCodecError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64, AndroidTouchPolicyCodecError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn id(&mut self) -> Result<[u8; 32], AndroidTouchPolicyCodecError> {
        self.array()
    }

    fn finish(&self) -> Result<(), AndroidTouchPolicyCodecError> {
        if self.pos != self.bytes.len() {
            return Err(AndroidTouchPolicyCodecError::TrailingBytes);
        }
        Ok(())
    }
}

fn validate_limits(
    bytes: &[u8],
    limits: AndroidTouchPolicyDecodeLimits,
) -> Result<(), AndroidTouchPolicyCodecError> {
    if limits.max_records == 0 {
        return Err(AndroidTouchPolicyCodecError::ZeroDecodeRecordLimit);
    }
    if limits.max_bytes == 0 {
        return Err(AndroidTouchPolicyCodecError::ZeroDecodeByteLimit);
    }
    if bytes.len() > limits.max_bytes {
        return Err(AndroidTouchPolicyCodecError::InputTooLarge);
    }
    Ok(())
}

fn write_magic_version(writer: &mut Writer, magic: &[u8; 8]) {
    writer.raw(magic);
    writer.u32(ANDROID_POLICY_CODEC_VERSION);
}

fn read_magic_version(
    reader: &mut Reader<'_>,
    magic: &[u8; 8],
) -> Result<(), AndroidTouchPolicyCodecError> {
    if reader.array::<8>()? != *magic {
        return Err(AndroidTouchPolicyCodecError::BadMagic);
    }
    if reader.u32()? != ANDROID_POLICY_CODEC_VERSION {
        return Err(AndroidTouchPolicyCodecError::UnsupportedVersion);
    }
    Ok(())
}

fn write_platform(writer: &mut Writer, platform: PlatformIngressKind) {
    writer.u8(platform as u8);
}

fn read_platform(reader: &mut Reader<'_>) -> Result<PlatformIngressKind, AndroidTouchPolicyCodecError> {
    match reader.u8()? {
        1 => Ok(PlatformIngressKind::AndroidJni),
        2 => Ok(PlatformIngressKind::IosCAbi),
        other => Err(AndroidTouchPolicyCodecError::InvalidPlatform(other)),
    }
}

fn write_transition_kind(writer: &mut Writer, kind: AndroidTouchPolicySessionTransitionKind) {
    writer.u8(kind as u8);
}

fn read_transition_kind(
    reader: &mut Reader<'_>,
) -> Result<AndroidTouchPolicySessionTransitionKind, AndroidTouchPolicyCodecError> {
    match reader.u8()? {
        1 => Ok(AndroidTouchPolicySessionTransitionKind::Install),
        2 => Ok(AndroidTouchPolicySessionTransitionKind::Replace),
        3 => Ok(AndroidTouchPolicySessionTransitionKind::Revoke),
        other => Err(AndroidTouchPolicyCodecError::InvalidTransitionKind(other)),
    }
}

fn write_status(writer: &mut Writer, status: AndroidTouchPolicySessionStatus) {
    writer.u8(status as u8);
}

fn read_status(
    reader: &mut Reader<'_>,
) -> Result<AndroidTouchPolicySessionStatus, AndroidTouchPolicyCodecError> {
    match reader.u8()? {
        1 => Ok(AndroidTouchPolicySessionStatus::Active),
        2 => Ok(AndroidTouchPolicySessionStatus::Revoked),
        other => Err(AndroidTouchPolicyCodecError::InvalidSessionStatus(other)),
    }
}

fn write_bundle(writer: &mut Writer, bundle: &AndroidTouchPolicyBundle) {
    let ingress = &bundle.ingress_policy;
    writer.id(ingress.policy_context_id.as_bytes());
    writer.u64(ingress.policy_generation);
    writer.id(ingress.policy_state_root.as_bytes());
    write_platform(writer, ingress.platform);
    writer.id(ingress.ingress_profile_id.as_bytes());
    writer.id(ingress.trusted_surface_id.as_bytes());
    writer.id(ingress.input_attester_id.as_bytes());
    writer.bool(ingress.allow_frame);
    writer.bool(ingress.allow_touch);

    let profile = &bundle.ingress_profile;
    write_platform(writer, profile.platform);
    writer.u32(profile.abi_version);
    writer.id(profile.capture_profile_id.as_bytes());
    writer.id(profile.touch_input_profile_id.as_bytes());
    writer.u64(profile.max_frame_bytes);

    let attention_admission = &bundle.attention_policy_admission;
    writer.id(attention_admission.policy_context_id.as_bytes());
    writer.u64(attention_admission.policy_generation);
    writer.id(attention_admission.policy_state_root.as_bytes());
    writer.id(attention_admission.authorized_requirement_id.as_bytes());

    let attention = &bundle.attention_requirement;
    writer.id(attention.policy_context_id.as_bytes());
    writer.u64(attention.policy_generation);
    writer.id(attention.policy_state_root.as_bytes());
    writer.id(attention.trusted_surface_id.as_bytes());
    writer.u32(attention.minimum_sdk_int);
    writer.bool(attention.require_window_focus);
    writer.bool(attention.require_flag_secure);
    writer.bool(attention.require_hide_application_overlays);
    writer.bool(attention.require_filter_touches_when_obscured);
    writer.bool(attention.require_view_attached);
    writer.bool(attention.require_view_shown);
    writer.bool(attention.require_top_resumed);
    writer.bool(attention.forbid_multi_window);

    let motion_admission = &bundle.motion_policy_admission;
    writer.id(motion_admission.policy_context_id.as_bytes());
    writer.u64(motion_admission.policy_generation);
    writer.id(motion_admission.policy_state_root.as_bytes());
    writer.id(motion_admission.attention_policy_admission_id.as_bytes());
    writer.id(motion_admission.authorized_motion_requirement_id.as_bytes());

    let motion = &bundle.motion_requirement;
    writer.id(motion.attention_requirement_id.as_bytes());
    writer.bool(motion.reject_fully_obscured);
    writer.bool(motion.reject_partially_obscured);
    writer.bool(motion.require_single_pointer);
}

fn read_bundle(reader: &mut Reader<'_>) -> Result<AndroidTouchPolicyBundle, AndroidTouchPolicyCodecError> {
    let ingress_policy = IngressAdmissionPolicy {
        policy_context_id: IngressPolicyContextId(reader.id()?),
        policy_generation: reader.u64()?,
        policy_state_root: IngressPolicyStateRoot(reader.id()?),
        platform: read_platform(reader)?,
        ingress_profile_id: crate::assurance_platform_ingress::PlatformIngressProfileId(reader.id()?),
        trusted_surface_id: TrustedSurfaceId(reader.id()?),
        input_attester_id: InputAttesterId(reader.id()?),
        allow_frame: reader.bool()?,
        allow_touch: reader.bool()?,
    };
    let ingress_profile = PlatformIngressProfile {
        platform: read_platform(reader)?,
        abi_version: reader.u32()?,
        capture_profile_id: ScreenCaptureProfileId(reader.id()?),
        touch_input_profile_id: TouchInputProfileId(reader.id()?),
        max_frame_bytes: reader.u64()?,
    };
    let attention_policy_admission = AndroidAttentionPolicyAdmission {
        policy_context_id: AndroidAttentionPolicyContextId(reader.id()?),
        policy_generation: reader.u64()?,
        policy_state_root: AndroidAttentionPolicyStateRoot(reader.id()?),
        authorized_requirement_id: crate::assurance_android_attention::AndroidAttentionRequirementId(
            reader.id()?,
        ),
    };
    let attention_requirement = AndroidAttentionRequirement {
        policy_context_id: AndroidAttentionPolicyContextId(reader.id()?),
        policy_generation: reader.u64()?,
        policy_state_root: AndroidAttentionPolicyStateRoot(reader.id()?),
        trusted_surface_id: TrustedSurfaceId(reader.id()?),
        minimum_sdk_int: reader.u32()?,
        require_window_focus: reader.bool()?,
        require_flag_secure: reader.bool()?,
        require_hide_application_overlays: reader.bool()?,
        require_filter_touches_when_obscured: reader.bool()?,
        require_view_attached: reader.bool()?,
        require_view_shown: reader.bool()?,
        require_top_resumed: reader.bool()?,
        forbid_multi_window: reader.bool()?,
    };
    let motion_policy_admission = AndroidMotionPolicyAdmission {
        policy_context_id: AndroidAttentionPolicyContextId(reader.id()?),
        policy_generation: reader.u64()?,
        policy_state_root: AndroidAttentionPolicyStateRoot(reader.id()?),
        attention_policy_admission_id:
            crate::assurance_android_attention_policy::AndroidAttentionPolicyAdmissionId(
                reader.id()?,
            ),
        authorized_motion_requirement_id:
            crate::assurance_android_motion_event::AndroidMotionEventRequirementId(reader.id()?),
    };
    let motion_requirement = AndroidMotionEventRequirement {
        attention_requirement_id:
            crate::assurance_android_attention::AndroidAttentionRequirementId(reader.id()?),
        reject_fully_obscured: reader.bool()?,
        reject_partially_obscured: reader.bool()?,
        require_single_pointer: reader.bool()?,
    };

    let bundle = AndroidTouchPolicyBundle {
        ingress_policy,
        ingress_profile,
        attention_policy_admission,
        attention_requirement,
        motion_policy_admission,
        motion_requirement,
    };
    bundle.validate()?;
    Ok(bundle)
}

fn write_transition(writer: &mut Writer, transition: &AndroidTouchPolicyStoreTransition) {
    let receipt = &transition.receipt;
    write_transition_kind(writer, receipt.kind);
    writer.id(receipt.authorization_id.as_bytes());
    writer.id(receipt.session_id.as_bytes());
    writer.id(receipt.prior_state_id.as_bytes());
    writer.id(receipt.next_state_id.as_bytes());
    writer.u64(receipt.prior_generation);
    writer.u64(receipt.next_generation);
    writer.id(receipt.prior_bundle_id.as_bytes());
    writer.id(receipt.next_bundle_id.as_bytes());
    writer.id(transition.receipt_id.as_bytes());

    let snapshot = &transition.snapshot;
    writer.id(snapshot.session_id.as_bytes());
    writer.u64(snapshot.generation);
    writer.id(snapshot.state_id.as_bytes());
    writer.id(snapshot.bundle_id.as_bytes());
    write_status(writer, snapshot.status);
}

fn read_transition(
    reader: &mut Reader<'_>,
) -> Result<AndroidTouchPolicyStoreTransition, AndroidTouchPolicyCodecError> {
    let receipt = AndroidTouchPolicySessionTransitionReceipt {
        kind: read_transition_kind(reader)?,
        authorization_id: AndroidTouchPolicySessionAuthorizationId(reader.id()?),
        session_id: AndroidTouchPolicySessionId(reader.id()?),
        prior_state_id: AndroidTouchPolicySessionStateId(reader.id()?),
        next_state_id: AndroidTouchPolicySessionStateId(reader.id()?),
        prior_generation: reader.u64()?,
        next_generation: reader.u64()?,
        prior_bundle_id: crate::assurance_android_policy_bundle::AndroidTouchPolicyBundleId(
            reader.id()?,
        ),
        next_bundle_id: crate::assurance_android_policy_bundle::AndroidTouchPolicyBundleId(
            reader.id()?,
        ),
    };
    let transition = AndroidTouchPolicyStoreTransition {
        receipt,
        receipt_id: AndroidTouchPolicySessionTransitionReceiptId(reader.id()?),
        snapshot: AndroidTouchPolicyStoreSnapshot {
            session_id: AndroidTouchPolicySessionId(reader.id()?),
            generation: reader.u64()?,
            state_id: AndroidTouchPolicySessionStateId(reader.id()?),
            bundle_id: crate::assurance_android_policy_bundle::AndroidTouchPolicyBundleId(
                reader.id()?,
            ),
            status: read_status(reader)?,
        },
    };
    transition.receipt.validate()?;
    if transition.receipt.receipt_id()? != transition.receipt_id {
        return Err(AndroidTouchPolicyCodecError::NonCanonical);
    }
    Ok(transition)
}

fn write_entry(writer: &mut Writer, entry: &AndroidTouchPolicyJournalEntry) {
    writer.u64(entry.sequence);
    writer.id(entry.predecessor_entry_id.as_bytes());
    write_transition(writer, &entry.transition);
    write_bundle(writer, &entry.bundle);
    writer.id(entry.install_nonce.as_bytes());
}

fn read_entry(reader: &mut Reader<'_>) -> Result<AndroidTouchPolicyJournalEntry, AndroidTouchPolicyCodecError> {
    Ok(AndroidTouchPolicyJournalEntry {
        sequence: reader.u64()?,
        predecessor_entry_id: AndroidTouchPolicyJournalEntryId(reader.id()?),
        transition: read_transition(reader)?,
        bundle: read_bundle(reader)?,
        install_nonce: AndroidTouchPolicySessionNonce(reader.id()?),
    })
}

pub fn encode_android_policy_journal(
    journal: &AndroidTouchPolicyJournal,
) -> Result<Vec<u8>, AndroidTouchPolicyCodecError> {
    journal.validate()?;
    let count = u64::try_from(journal.len())
        .map_err(|_| AndroidTouchPolicyCodecError::RecordCountNotRepresentable)?;
    let mut writer = Writer::new();
    write_magic_version(&mut writer, &JOURNAL_MAGIC);
    writer.u64(count);
    for entry in journal.entries() {
        write_entry(&mut writer, entry);
    }
    Ok(writer.finish())
}

pub fn decode_android_policy_journal(
    bytes: &[u8],
    limits: AndroidTouchPolicyDecodeLimits,
) -> Result<AndroidTouchPolicyJournal, AndroidTouchPolicyCodecError> {
    validate_limits(bytes, limits)?;
    let mut reader = Reader::new(bytes);
    read_magic_version(&mut reader, &JOURNAL_MAGIC)?;
    let count = reader.u64()?;
    if count > limits.max_records {
        return Err(AndroidTouchPolicyCodecError::RecordCountTooLarge);
    }
    let count_usize = usize::try_from(count)
        .map_err(|_| AndroidTouchPolicyCodecError::RecordCountNotRepresentable)?;

    let mut journal = AndroidTouchPolicyJournal::new();
    for _ in 0..count_usize {
        let parsed = read_entry(&mut reader)?;
        journal.append(parsed.transition, parsed.bundle, parsed.install_nonce)?;
        if journal.entries().last().copied() != Some(parsed) {
            return Err(AndroidTouchPolicyCodecError::NonCanonical);
        }
    }
    reader.finish()?;
    journal.validate()?;
    if encode_android_policy_journal(&journal)?.as_slice() != bytes {
        return Err(AndroidTouchPolicyCodecError::NonCanonical);
    }
    Ok(journal)
}

pub fn encode_android_policy_checkpoint(
    checkpoint: &AndroidTouchPolicyJournalCheckpoint,
) -> Result<Vec<u8>, AndroidTouchPolicyCodecError> {
    let _ = checkpoint.checkpoint_id()?;
    let mut writer = Writer::new();
    write_magic_version(&mut writer, &CHECKPOINT_MAGIC);
    writer.u64(checkpoint.record_count);
    writer.id(checkpoint.last_entry_id.as_bytes());
    writer.id(checkpoint.journal_root.as_bytes());
    writer.id(checkpoint.authorization_set_root.as_bytes());
    writer.id(checkpoint.session_id.as_bytes());
    writer.u64(checkpoint.generation);
    writer.id(checkpoint.state_id.as_bytes());
    writer.id(checkpoint.bundle_id.as_bytes());
    write_status(&mut writer, checkpoint.status);
    Ok(writer.finish())
}

pub fn decode_android_policy_checkpoint(
    bytes: &[u8],
    max_bytes: usize,
) -> Result<AndroidTouchPolicyJournalCheckpoint, AndroidTouchPolicyCodecError> {
    validate_limits(
        bytes,
        AndroidTouchPolicyDecodeLimits {
            max_records: 1,
            max_bytes,
        },
    )?;
    let mut reader = Reader::new(bytes);
    read_magic_version(&mut reader, &CHECKPOINT_MAGIC)?;
    let checkpoint = AndroidTouchPolicyJournalCheckpoint {
        record_count: reader.u64()?,
        last_entry_id: AndroidTouchPolicyJournalEntryId(reader.id()?),
        journal_root: AndroidTouchPolicyJournalRoot(reader.id()?),
        authorization_set_root: AndroidTouchPolicyAuthorizationSetRoot(reader.id()?),
        session_id: AndroidTouchPolicySessionId(reader.id()?),
        generation: reader.u64()?,
        state_id: AndroidTouchPolicySessionStateId(reader.id()?),
        bundle_id: crate::assurance_android_policy_bundle::AndroidTouchPolicyBundleId(
            reader.id()?,
        ),
        status: read_status(&mut reader)?,
    };
    reader.finish()?;
    let _ = checkpoint.checkpoint_id()?;
    if encode_android_policy_checkpoint(&checkpoint)?.as_slice() != bytes {
        return Err(AndroidTouchPolicyCodecError::NonCanonical);
    }
    Ok(checkpoint)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assurance_android_policy_journal::AndroidTouchPolicyJournaledStore;

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
        let attention_policy_admission = AndroidAttentionPolicyAdmission {
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
        let motion_policy_admission = AndroidMotionPolicyAdmission {
            policy_context_id: attention_requirement.policy_context_id,
            policy_generation: attention_requirement.policy_generation,
            policy_state_root: attention_requirement.policy_state_root,
            attention_policy_admission_id: attention_policy_admission.admission_id().unwrap(),
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
            attention_policy_admission,
            attention_requirement,
            motion_policy_admission,
            motion_requirement,
        }
    }

    fn journal_with_three_transitions() -> AndroidTouchPolicyJournaledStore {
        let mut store = AndroidTouchPolicyJournaledStore::new();
        store.install(&bundle(20), auth(40), nonce(41)).unwrap();
        let s1 = store.snapshot().unwrap();
        store.replace(s1.state_id, &bundle(30), auth(42)).unwrap();
        let s2 = store.snapshot().unwrap();
        store.revoke(s2.state_id, auth(43)).unwrap();
        store
    }

    #[test]
    fn journal_roundtrip_is_exact_and_semantically_replayed() {
        let store = journal_with_three_transitions();
        let bytes = encode_android_policy_journal(store.journal()).unwrap();
        let decoded = decode_android_policy_journal(
            &bytes,
            AndroidTouchPolicyDecodeLimits::default(),
        )
        .unwrap();
        assert_eq!(decoded.entries(), store.journal().entries());
        assert_eq!(encode_android_policy_journal(&decoded).unwrap(), bytes);
        assert_eq!(
            decoded.recover_store().unwrap().snapshot().unwrap(),
            store.snapshot().unwrap()
        );
    }

    #[test]
    fn checkpoint_roundtrip_is_exact() {
        let store = journal_with_three_transitions();
        let checkpoint = store.checkpoint().unwrap();
        let bytes = encode_android_policy_checkpoint(&checkpoint).unwrap();
        let decoded = decode_android_policy_checkpoint(&bytes, 4096).unwrap();
        assert_eq!(decoded, checkpoint);
        assert_eq!(encode_android_policy_checkpoint(&decoded).unwrap(), bytes);
    }

    #[test]
    fn trailing_bytes_are_rejected_not_ignored() {
        let store = journal_with_three_transitions();
        let mut bytes = encode_android_policy_journal(store.journal()).unwrap();
        bytes.push(0);
        assert_eq!(
            decode_android_policy_journal(&bytes, AndroidTouchPolicyDecodeLimits::default()),
            Err(AndroidTouchPolicyCodecError::TrailingBytes)
        );
    }

    #[test]
    fn strict_boolean_parser_rejects_non_boolean_byte() {
        let mut reader = Reader::new(&[2]);
        assert_eq!(
            reader.bool(),
            Err(AndroidTouchPolicyCodecError::InvalidBoolean(2))
        );
    }

    #[test]
    fn closed_enum_parser_rejects_unknown_platform() {
        let mut reader = Reader::new(&[7]);
        assert_eq!(
            read_platform(&mut reader),
            Err(AndroidTouchPolicyCodecError::InvalidPlatform(7))
        );
    }

    #[test]
    fn truncation_and_decode_limits_fail_closed() {
        let store = journal_with_three_transitions();
        let bytes = encode_android_policy_journal(store.journal()).unwrap();
        assert_eq!(
            decode_android_policy_journal(
                &bytes[..bytes.len() - 1],
                AndroidTouchPolicyDecodeLimits::default(),
            ),
            Err(AndroidTouchPolicyCodecError::Truncated)
        );
        assert_eq!(
            decode_android_policy_journal(
                &bytes,
                AndroidTouchPolicyDecodeLimits {
                    max_records: 2,
                    max_bytes: bytes.len(),
                },
            ),
            Err(AndroidTouchPolicyCodecError::RecordCountTooLarge)
        );
        assert_eq!(
            decode_android_policy_journal(
                &bytes,
                AndroidTouchPolicyDecodeLimits {
                    max_records: 10,
                    max_bytes: bytes.len() - 1,
                },
            ),
            Err(AndroidTouchPolicyCodecError::InputTooLarge)
        );
    }
}
