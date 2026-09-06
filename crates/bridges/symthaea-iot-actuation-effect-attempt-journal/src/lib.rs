// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Crash-durable, authority-free physical-effect attempt ambiguity journal.
//!
//! This crate answers one narrow question: after semantic replay state and global actuation
//! currentness have already been established, what can restart safely know about whether the final
//! privileged physical boundary may have been entered?
//!
//! It deliberately does **not** authorize an effect, verify transport/device/controller trust, mint
//! permits, perform HAL I/O or claim physical realization. It only persists one exact attempt
//! correlation before the privileged call and advances that durable state after an in-process
//! result. A crash while the state remains `Prepared` is therefore conservatively unresolved.
//!
//! State machine v0.1:
//!
//! ```text
//! genesis / AbandonedBeforePort
//!          |
//!          v
//!       Prepared  ---- final time check fails ----> AbandonedBeforePort
//!          |
//!          +------ adapter acknowledges ---------> AdapterAcknowledged
//!          |
//!          +------ adapter errors ----------------> AdapterIndeterminate
//! ```
//!
//! `Prepared`, `AdapterAcknowledged` and `AdapterIndeterminate` all block a later attempt for the
//! same device. This crate intentionally has no generic "realized" transition: protocol acceptance
//! and arbitrary telemetry are not universal proof of a physical-world state change. A successor
//! reconciliation layer must consume fresh, device-class-specific verified evidence to close those
//! states.

#![deny(unsafe_code)]

#[cfg(not(target_os = "linux"))]
compile_error!(
    "symthaea-iot-actuation-effect-attempt-journal is Linux-only and relies on a pinned directory capability"
);

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use symthaea_authority::{Digest32, ResourceRef};
use symthaea_iot_actuation_effect_dispatch::{
    DurablePhysicalEffectAttemptJournal, DurablePreparedPhysicalEffectAttempt,
    PhysicalEffectAttemptCorrelation, PhysicalEffectAttemptJournalHead,
};
use thiserror::Error;

pub const EFFECT_ATTEMPT_JOURNAL_SCHEMA_VERSION: u16 = 1;
pub const MAX_EFFECT_ATTEMPT_STATE_BYTES: u64 = 64 * 1024;
pub const MAX_EFFECT_ATTEMPT_ID_BYTES: usize = 1024;

const STATE_FILE_NAME: &str = "effect-attempt.state";
const LOCK_FILE_NAME: &str = ".effect-attempt.lock";
const CHECKPOINT_DOMAIN: &[u8] = b"symthaea-iot-effect-attempt-checkpoint-v1\0";
const CORRELATION_DOMAIN: &[u8] = b"symthaea-iot-effect-attempt-correlation-v1\0";
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Independently retainable anti-rollback head for the durable attempt journal.
///
/// Generation zero is the deterministic genesis checkpoint. Nonzero generations represent actual
/// journal transitions. As with the other durable guard stores, the surrounding deployment should
/// retain this head independently from the state file when adversarial rollback matters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DurableEffectAttemptJournalHeadV1 {
    generation: u64,
    digest: Digest32,
}

impl DurableEffectAttemptJournalHeadV1 {
    pub const fn generation(self) -> u64 {
        self.generation
    }

    pub const fn digest(self) -> Digest32 {
        self.digest
    }

    pub fn from_dispatch_head(head: PhysicalEffectAttemptJournalHead) -> Self {
        Self {
            generation: head.generation(),
            digest: head.digest(),
        }
    }

    fn as_dispatch_head(self) -> Result<PhysicalEffectAttemptJournalHead, EffectAttemptJournalError> {
        PhysicalEffectAttemptJournalHead::new(self.generation, self.digest)
            .map_err(|_| EffectAttemptJournalError::InvalidJournalHead)
    }
}

/// Serializable authority-free projection of the exact correlation created by the terminal
/// dispatcher. No command parameters or caller-supplied replacement command are accepted here.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DurableEffectAttemptCorrelationV1 {
    command_digest: Digest32,
    envelope_digest: Digest32,
    composition_digest: Digest32,
    device: String,
    operation: String,
    executor: String,
    sequence: u64,
    adapter_id: String,
    common_fenced_at_unix_ms: u64,
    wall_valid_until_unix_ms: u64,
}

impl DurableEffectAttemptCorrelationV1 {
    fn from_dispatch(
        correlation: &PhysicalEffectAttemptCorrelation,
    ) -> Result<Self, EffectAttemptJournalError> {
        let value = Self {
            command_digest: correlation.command_digest(),
            envelope_digest: correlation.envelope_digest(),
            composition_digest: correlation.composition_digest(),
            device: correlation.device().0.clone(),
            operation: correlation.operation().0.clone(),
            executor: correlation.executor().0.clone(),
            sequence: correlation.sequence(),
            adapter_id: correlation.adapter_id().to_owned(),
            common_fenced_at_unix_ms: correlation.common_fenced_at_unix_ms(),
            wall_valid_until_unix_ms: correlation.wall_valid_until_unix_ms(),
        };
        value.validate()?;
        Ok(value)
    }

    pub const fn command_digest(&self) -> Digest32 {
        self.command_digest
    }

    pub const fn envelope_digest(&self) -> Digest32 {
        self.envelope_digest
    }

    pub const fn composition_digest(&self) -> Digest32 {
        self.composition_digest
    }

    pub fn device(&self) -> &str {
        &self.device
    }

    pub fn operation(&self) -> &str {
        &self.operation
    }

    pub fn executor(&self) -> &str {
        &self.executor
    }

    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn adapter_id(&self) -> &str {
        &self.adapter_id
    }

    pub const fn common_fenced_at_unix_ms(&self) -> u64 {
        self.common_fenced_at_unix_ms
    }

    pub const fn wall_valid_until_unix_ms(&self) -> u64 {
        self.wall_valid_until_unix_ms
    }

    pub fn digest(&self) -> Result<Digest32, EffectAttemptJournalError> {
        self.validate()?;
        digest_canonical(CORRELATION_DOMAIN, self)
    }

    fn validate(&self) -> Result<(), EffectAttemptJournalError> {
        for digest in [
            self.command_digest,
            self.envelope_digest,
            self.composition_digest,
        ] {
            if digest == Digest32([0; 32]) {
                return Err(EffectAttemptJournalError::ZeroSecurityCommitment);
            }
        }
        for value in [
            self.device.as_str(),
            self.operation.as_str(),
            self.executor.as_str(),
            self.adapter_id.as_str(),
        ] {
            if !valid_id(value) {
                return Err(EffectAttemptJournalError::InvalidIdentity);
            }
        }
        if self.sequence == 0 {
            return Err(EffectAttemptJournalError::SequenceZero);
        }
        if self.wall_valid_until_unix_ms <= self.common_fenced_at_unix_ms {
            return Err(EffectAttemptJournalError::InvalidAttemptWindow);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DurableEffectAttemptStateV1 {
    Prepared {
        correlation: DurableEffectAttemptCorrelationV1,
    },
    AbandonedBeforePort {
        correlation: DurableEffectAttemptCorrelationV1,
    },
    AdapterAcknowledged {
        correlation: DurableEffectAttemptCorrelationV1,
        adapter_evidence_digest: Digest32,
    },
    AdapterIndeterminate {
        correlation: DurableEffectAttemptCorrelationV1,
    },
}

impl DurableEffectAttemptStateV1 {
    pub fn correlation(&self) -> &DurableEffectAttemptCorrelationV1 {
        match self {
            Self::Prepared { correlation }
            | Self::AbandonedBeforePort { correlation }
            | Self::AdapterAcknowledged { correlation, .. }
            | Self::AdapterIndeterminate { correlation } => correlation,
        }
    }

    pub fn requires_reconciliation(&self) -> bool {
        matches!(
            self,
            Self::Prepared { .. }
                | Self::AdapterAcknowledged { .. }
                | Self::AdapterIndeterminate { .. }
        )
    }

    pub fn adapter_evidence_digest(&self) -> Option<Digest32> {
        match self {
            Self::AdapterAcknowledged {
                adapter_evidence_digest,
                ..
            } => Some(*adapter_evidence_digest),
            _ => None,
        }
    }

    fn validate(&self) -> Result<(), EffectAttemptJournalError> {
        self.correlation().validate()?;
        if let Self::AdapterAcknowledged {
            adapter_evidence_digest,
            ..
        } = self
        {
            if *adapter_evidence_digest == Digest32([0; 32]) {
                return Err(EffectAttemptJournalError::ZeroAdapterEvidenceDigest);
            }
        }
        Ok(())
    }
}

/// Complete single-device attempt journal checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DurableEffectAttemptJournalCheckpointV1 {
    schema_version: u16,
    generation: u64,
    previous_checkpoint_digest: Option<Digest32>,
    device: String,
    latest: Option<DurableEffectAttemptStateV1>,
}

impl DurableEffectAttemptJournalCheckpointV1 {
    pub fn genesis(device: &ResourceRef) -> Result<Self, EffectAttemptJournalError> {
        if !valid_id(&device.0) {
            return Err(EffectAttemptJournalError::InvalidIdentity);
        }
        Ok(Self {
            schema_version: EFFECT_ATTEMPT_JOURNAL_SCHEMA_VERSION,
            generation: 0,
            previous_checkpoint_digest: None,
            device: device.0.clone(),
            latest: None,
        })
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub fn device(&self) -> &str {
        &self.device
    }

    pub fn latest(&self) -> Option<&DurableEffectAttemptStateV1> {
        self.latest.as_ref()
    }

    pub fn validate(&self) -> Result<(), EffectAttemptJournalError> {
        if self.schema_version != EFFECT_ATTEMPT_JOURNAL_SCHEMA_VERSION {
            return Err(EffectAttemptJournalError::UnsupportedSchema);
        }
        if !valid_id(&self.device) {
            return Err(EffectAttemptJournalError::InvalidIdentity);
        }
        match self.generation {
            0 => {
                if self.previous_checkpoint_digest.is_some() || self.latest.is_some() {
                    return Err(EffectAttemptJournalError::MalformedGenesis);
                }
            }
            _ => {
                if self.previous_checkpoint_digest.is_none() || self.latest.is_none() {
                    return Err(EffectAttemptJournalError::IncompleteCheckpoint);
                }
                if self.previous_checkpoint_digest == Some(Digest32([0; 32])) {
                    return Err(EffectAttemptJournalError::ZeroSecurityCommitment);
                }
                let latest = self
                    .latest
                    .as_ref()
                    .ok_or(EffectAttemptJournalError::IncompleteCheckpoint)?;
                latest.validate()?;
                if latest.correlation().device != self.device {
                    return Err(EffectAttemptJournalError::CheckpointDeviceMismatch);
                }
            }
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<Digest32, EffectAttemptJournalError> {
        self.validate()?;
        digest_canonical(CHECKPOINT_DOMAIN, self)
    }

    pub fn head(&self) -> Result<DurableEffectAttemptJournalHeadV1, EffectAttemptJournalError> {
        Ok(DurableEffectAttemptJournalHeadV1 {
            generation: self.generation,
            digest: self.digest()?,
        })
    }
}

/// Opaque proof that one exact correlation reached crash-durable `Prepared` state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreparedDurableEffectAttemptV1 {
    journal_head: PhysicalEffectAttemptJournalHead,
    correlation_digest: Digest32,
}

impl DurablePreparedPhysicalEffectAttempt for PreparedDurableEffectAttemptV1 {
    fn journal_head(&self) -> PhysicalEffectAttemptJournalHead {
        self.journal_head
    }
}

/// Crash-durable single-device attempt journal opened against an independently retained head.
pub struct DurableEffectAttemptJournalStore {
    root: PathBuf,
    device: String,
    trusted_current_head: DurableEffectAttemptJournalHeadV1,
    local_lock: Mutex<()>,
    pinned_root: Mutex<Option<Arc<File>>>,
}

impl DurableEffectAttemptJournalStore {
    pub fn open(
        root: impl Into<PathBuf>,
        device: &ResourceRef,
        trusted_current_head: DurableEffectAttemptJournalHeadV1,
    ) -> Result<Self, EffectAttemptJournalError> {
        if !valid_id(&device.0) {
            return Err(EffectAttemptJournalError::InvalidIdentity);
        }
        let store = Self {
            root: root.into(),
            device: device.0.clone(),
            trusted_current_head,
            local_lock: Mutex::new(()),
            pinned_root: Mutex::new(None),
        };
        store.ensure_root()?;
        {
            let _local = store
                .local_lock
                .lock()
                .map_err(|_| EffectAttemptJournalError::LocalLockPoisoned)?;
            let kernel = store.open_lock_file()?;
            kernel.lock().map_err(EffectAttemptJournalError::Io)?;
            let checkpoint = store.read_state_locked()?;
            store.verify_loaded_checkpoint(&checkpoint)?;
        }
        Ok(store)
    }

    pub const fn trusted_current_head(&self) -> DurableEffectAttemptJournalHeadV1 {
        self.trusted_current_head
    }

    pub fn current_checkpoint(
        &self,
    ) -> Result<DurableEffectAttemptJournalCheckpointV1, EffectAttemptJournalError> {
        let _local = self
            .local_lock
            .lock()
            .map_err(|_| EffectAttemptJournalError::LocalLockPoisoned)?;
        let kernel = self.open_lock_file()?;
        kernel.lock().map_err(EffectAttemptJournalError::Io)?;
        let checkpoint = self.read_state_locked()?;
        self.verify_loaded_checkpoint(&checkpoint)?;
        Ok(checkpoint)
    }

    fn persist_prepared_inner(
        &mut self,
        correlation: &PhysicalEffectAttemptCorrelation,
    ) -> Result<PreparedDurableEffectAttemptV1, EffectAttemptJournalError> {
        let projected = DurableEffectAttemptCorrelationV1::from_dispatch(correlation)?;
        if projected.device != self.device {
            return Err(EffectAttemptJournalError::CheckpointDeviceMismatch);
        }

        let _local = self
            .local_lock
            .lock()
            .map_err(|_| EffectAttemptJournalError::LocalLockPoisoned)?;
        let kernel = self.open_lock_file()?;
        kernel.lock().map_err(EffectAttemptJournalError::Io)?;
        let current = self.read_state_locked()?;
        self.verify_loaded_checkpoint(&current)?;

        if let Some(latest) = current.latest() {
            if latest.requires_reconciliation() {
                return Err(EffectAttemptJournalError::UnresolvedAttemptExists);
            }
            if projected.sequence <= latest.correlation().sequence {
                return Err(EffectAttemptJournalError::SequenceNotMonotonic);
            }
        }

        let successor = self.successor(
            &current,
            DurableEffectAttemptStateV1::Prepared {
                correlation: projected.clone(),
            },
        )?;
        let head = self.persist_successor_locked(&successor)?;
        let dispatch_head = head.as_dispatch_head()?;
        self.trusted_current_head = head;
        Ok(PreparedDurableEffectAttemptV1 {
            journal_head: dispatch_head,
            correlation_digest: projected.digest()?,
        })
    }

    fn transition_prepared(
        &mut self,
        prepared: &PreparedDurableEffectAttemptV1,
        transition: impl FnOnce(DurableEffectAttemptCorrelationV1) -> DurableEffectAttemptStateV1,
    ) -> Result<PhysicalEffectAttemptJournalHead, EffectAttemptJournalError> {
        let _local = self
            .local_lock
            .lock()
            .map_err(|_| EffectAttemptJournalError::LocalLockPoisoned)?;
        let kernel = self.open_lock_file()?;
        kernel.lock().map_err(EffectAttemptJournalError::Io)?;
        let current = self.read_state_locked()?;
        self.verify_loaded_checkpoint(&current)?;
        if current.head()? != DurableEffectAttemptJournalHeadV1::from_dispatch_head(prepared.journal_head)
        {
            return Err(EffectAttemptJournalError::PreparedHeadMismatch);
        }
        let correlation = match current.latest() {
            Some(DurableEffectAttemptStateV1::Prepared { correlation }) => correlation.clone(),
            _ => return Err(EffectAttemptJournalError::PreparedStateMissing),
        };
        if correlation.digest()? != prepared.correlation_digest {
            return Err(EffectAttemptJournalError::PreparedCorrelationMismatch);
        }

        let successor = self.successor(&current, transition(correlation))?;
        let head = self.persist_successor_locked(&successor)?;
        self.trusted_current_head = head;
        head.as_dispatch_head()
    }

    fn successor(
        &self,
        current: &DurableEffectAttemptJournalCheckpointV1,
        state: DurableEffectAttemptStateV1,
    ) -> Result<DurableEffectAttemptJournalCheckpointV1, EffectAttemptJournalError> {
        state.validate()?;
        if state.correlation().device != self.device {
            return Err(EffectAttemptJournalError::CheckpointDeviceMismatch);
        }
        let generation = current
            .generation
            .checked_add(1)
            .ok_or(EffectAttemptJournalError::GenerationOverflow)?;
        let successor = DurableEffectAttemptJournalCheckpointV1 {
            schema_version: EFFECT_ATTEMPT_JOURNAL_SCHEMA_VERSION,
            generation,
            previous_checkpoint_digest: Some(current.digest()?),
            device: self.device.clone(),
            latest: Some(state),
        };
        successor.validate()?;
        Ok(successor)
    }

    fn persist_successor_locked(
        &self,
        successor: &DurableEffectAttemptJournalCheckpointV1,
    ) -> Result<DurableEffectAttemptJournalHeadV1, EffectAttemptJournalError> {
        let expected_head = successor.head()?;
        self.write_state_locked(successor)?;
        let persisted = self.read_state_locked()?;
        if persisted != *successor || persisted.head()? != expected_head {
            return Err(EffectAttemptJournalError::PersistedStateReadbackMismatch);
        }
        Ok(expected_head)
    }

    fn verify_loaded_checkpoint(
        &self,
        checkpoint: &DurableEffectAttemptJournalCheckpointV1,
    ) -> Result<(), EffectAttemptJournalError> {
        checkpoint.validate()?;
        if checkpoint.device != self.device {
            return Err(EffectAttemptJournalError::CheckpointDeviceMismatch);
        }
        if checkpoint.head()? != self.trusted_current_head {
            return Err(EffectAttemptJournalError::TrustedJournalHeadMismatch);
        }
        Ok(())
    }

    fn read_state_locked(
        &self,
    ) -> Result<DurableEffectAttemptJournalCheckpointV1, EffectAttemptJournalError> {
        let path = self.operation_root_path()?.join(STATE_FILE_NAME);
        let file = match open_regular_file(&path, false, false) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return DurableEffectAttemptJournalCheckpointV1::genesis(&ResourceRef(
                    self.device.clone(),
                ));
            }
            Err(error) => return Err(EffectAttemptJournalError::Io(error)),
        };
        let metadata = file.metadata().map_err(EffectAttemptJournalError::Io)?;
        if metadata.len() == 0 || metadata.len() > MAX_EFFECT_ATTEMPT_STATE_BYTES {
            return Err(EffectAttemptJournalError::StateSizeOutOfBounds);
        }
        let mut bytes = Vec::with_capacity(metadata.len() as usize);
        file.take(MAX_EFFECT_ATTEMPT_STATE_BYTES.saturating_add(1))
            .read_to_end(&mut bytes)
            .map_err(EffectAttemptJournalError::Io)?;
        if bytes.is_empty() || bytes.len() as u64 > MAX_EFFECT_ATTEMPT_STATE_BYTES {
            return Err(EffectAttemptJournalError::StateSizeOutOfBounds);
        }
        let checkpoint: DurableEffectAttemptJournalCheckpointV1 =
            bincode::deserialize(&bytes).map_err(|_| EffectAttemptJournalError::StateEncoding)?;
        checkpoint.validate()?;
        let canonical =
            bincode::serialize(&checkpoint).map_err(|_| EffectAttemptJournalError::StateEncoding)?;
        if canonical != bytes {
            return Err(EffectAttemptJournalError::NonCanonicalStateEncoding);
        }
        Ok(checkpoint)
    }

    fn write_state_locked(
        &self,
        checkpoint: &DurableEffectAttemptJournalCheckpointV1,
    ) -> Result<(), EffectAttemptJournalError> {
        checkpoint.validate()?;
        if checkpoint.device != self.device {
            return Err(EffectAttemptJournalError::CheckpointDeviceMismatch);
        }
        let encoded =
            bincode::serialize(checkpoint).map_err(|_| EffectAttemptJournalError::StateEncoding)?;
        if encoded.is_empty() || encoded.len() as u64 > MAX_EFFECT_ATTEMPT_STATE_BYTES {
            return Err(EffectAttemptJournalError::StateSizeOutOfBounds);
        }

        let root = self.ensure_root()?;
        let operation_root = self.operation_root_path()?;
        let suffix = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let temp = operation_root.join(format!(
            ".effect-attempt-{}-{suffix}.tmp",
            std::process::id()
        ));
        let target = operation_root.join(STATE_FILE_NAME);
        let result = (|| {
            let mut file = open_regular_file(&temp, true, true)?;
            file.write_all(&encoded)?;
            file.sync_all()?;
            fs::rename(&temp, &target)?;
            root.sync_all()?;
            Ok::<(), std::io::Error>(())
        })();
        let _ = fs::remove_file(&temp);
        result.map_err(EffectAttemptJournalError::Io)
    }

    fn open_lock_file(&self) -> Result<File, EffectAttemptJournalError> {
        let path = self.operation_root_path()?.join(LOCK_FILE_NAME);
        open_regular_file(&path, true, false).map_err(EffectAttemptJournalError::Io)
    }

    fn ensure_root(&self) -> Result<Arc<File>, EffectAttemptJournalError> {
        let mut pinned = self
            .pinned_root
            .lock()
            .map_err(|_| EffectAttemptJournalError::RootLockPoisoned)?;
        if let Some(root) = pinned.as_ref() {
            return Ok(Arc::clone(root));
        }

        match fs::symlink_metadata(&self.root) {
            Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
                return Err(EffectAttemptJournalError::InvalidRootDirectory);
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                fs::create_dir_all(&self.root).map_err(EffectAttemptJournalError::Io)?;
            }
            Err(error) => return Err(EffectAttemptJournalError::Io(error)),
        }
        fs::set_permissions(&self.root, fs::Permissions::from_mode(0o700))
            .map_err(EffectAttemptJournalError::Io)?;

        let mut options = OpenOptions::new();
        options
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW);
        let root = Arc::new(options.open(&self.root).map_err(EffectAttemptJournalError::Io)?);
        if !root
            .metadata()
            .map_err(EffectAttemptJournalError::Io)?
            .is_dir()
        {
            return Err(EffectAttemptJournalError::InvalidRootDirectory);
        }
        *pinned = Some(Arc::clone(&root));
        Ok(root)
    }

    fn operation_root_path(&self) -> Result<PathBuf, EffectAttemptJournalError> {
        let root = self.ensure_root()?;
        let path = PathBuf::from(format!("/proc/self/fd/{}", root.as_raw_fd()));
        if !path.is_dir() {
            return Err(EffectAttemptJournalError::PinnedRootUnavailable);
        }
        Ok(path)
    }
}

impl DurablePhysicalEffectAttemptJournal for DurableEffectAttemptJournalStore {
    type Error = EffectAttemptJournalError;
    type Prepared = PreparedDurableEffectAttemptV1;

    fn persist_prepared(
        &mut self,
        correlation: &PhysicalEffectAttemptCorrelation,
    ) -> Result<Self::Prepared, Self::Error> {
        self.persist_prepared_inner(correlation)
    }

    fn persist_abandoned_before_port(
        &mut self,
        prepared: &Self::Prepared,
    ) -> Result<PhysicalEffectAttemptJournalHead, Self::Error> {
        self.transition_prepared(prepared, |correlation| {
            DurableEffectAttemptStateV1::AbandonedBeforePort { correlation }
        })
    }

    fn persist_adapter_acknowledged(
        &mut self,
        prepared: &Self::Prepared,
        adapter_evidence_digest: Digest32,
    ) -> Result<PhysicalEffectAttemptJournalHead, Self::Error> {
        if adapter_evidence_digest == Digest32([0; 32]) {
            return Err(EffectAttemptJournalError::ZeroAdapterEvidenceDigest);
        }
        self.transition_prepared(prepared, |correlation| {
            DurableEffectAttemptStateV1::AdapterAcknowledged {
                correlation,
                adapter_evidence_digest,
            }
        })
    }

    fn persist_adapter_indeterminate(
        &mut self,
        prepared: &Self::Prepared,
    ) -> Result<PhysicalEffectAttemptJournalHead, Self::Error> {
        self.transition_prepared(prepared, |correlation| {
            DurableEffectAttemptStateV1::AdapterIndeterminate { correlation }
        })
    }
}

fn digest_canonical<T: Serialize>(
    domain: &[u8],
    value: &T,
) -> Result<Digest32, EffectAttemptJournalError> {
    let bytes = bincode::serialize(value).map_err(|_| EffectAttemptJournalError::StateEncoding)?;
    let mut h = blake3::Hasher::new();
    h.update(domain);
    h.update(&(bytes.len() as u64).to_be_bytes());
    h.update(&bytes);
    Ok(Digest32(*h.finalize().as_bytes()))
}

fn valid_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_EFFECT_ATTEMPT_ID_BYTES
        && value.trim() == value
        && !value.chars().any(char::is_control)
}

fn open_regular_file(path: &Path, create: bool, create_new: bool) -> std::io::Result<File> {
    let mut options = OpenOptions::new();
    options
        .read(true)
        .write(true)
        .create(create)
        .create_new(create_new)
        .mode(0o600)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
    let file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "effect-attempt persistence object is not a regular file",
        ));
    }
    Ok(file)
}

#[derive(Debug, Error)]
pub enum EffectAttemptJournalError {
    #[error("unsupported durable effect-attempt journal schema")]
    UnsupportedSchema,
    #[error("effect-attempt journal genesis is malformed")]
    MalformedGenesis,
    #[error("effect-attempt journal checkpoint is incomplete")]
    IncompleteCheckpoint,
    #[error("effect-attempt journal contains a zero required commitment")]
    ZeroSecurityCommitment,
    #[error("effect-attempt adapter acknowledgement digest is zero")]
    ZeroAdapterEvidenceDigest,
    #[error("effect-attempt identity is empty, oversized, padded or contains control characters")]
    InvalidIdentity,
    #[error("effect-attempt command sequence is zero")]
    SequenceZero,
    #[error("effect-attempt wall-clock window is empty or reversed")]
    InvalidAttemptWindow,
    #[error("effect-attempt checkpoint targets another device")]
    CheckpointDeviceMismatch,
    #[error("effect-attempt journal generation overflow")]
    GenerationOverflow,
    #[error("effect-attempt sequence did not increase after the prior terminal state")]
    SequenceNotMonotonic,
    #[error("a prior physical effect remains unresolved and blocks another attempt")]
    UnresolvedAttemptExists,
    #[error("persisted effect-attempt checkpoint does not match the independently retained head")]
    TrustedJournalHeadMismatch,
    #[error("prepared effect-attempt head does not match current durable state")]
    PreparedHeadMismatch,
    #[error("current durable effect-attempt state is not Prepared")]
    PreparedStateMissing,
    #[error("prepared effect-attempt correlation differs from current durable state")]
    PreparedCorrelationMismatch,
    #[error("effect-attempt journal head is invalid")]
    InvalidJournalHead,
    #[error("persisted effect-attempt bytes/read-back do not match intended successor")]
    PersistedStateReadbackMismatch,
    #[error("effect-attempt persistence root is invalid or symlinked")]
    InvalidRootDirectory,
    #[error("pinned effect-attempt persistence root is unavailable")]
    PinnedRootUnavailable,
    #[error("effect-attempt persistence state exceeds its bounded size")]
    StateSizeOutOfBounds,
    #[error("effect-attempt persistence state encoding is invalid")]
    StateEncoding,
    #[error("effect-attempt persistence state is not canonically encoded")]
    NonCanonicalStateEncoding,
    #[error("effect-attempt persistence process-local lock is poisoned")]
    LocalLockPoisoned,
    #[error("effect-attempt persistence root lock is poisoned")]
    RootLockPoisoned,
    #[error("effect-attempt persistence I/O failed: {0}")]
    Io(#[source] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn genesis_is_deterministic_and_nonzero() {
        let device = ResourceRef("iot:valve:72".into());
        let a = DurableEffectAttemptJournalCheckpointV1::genesis(&device).unwrap();
        let b = DurableEffectAttemptJournalCheckpointV1::genesis(&device).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.generation(), 0);
        assert_ne!(a.digest().unwrap(), Digest32([0; 32]));
        assert_eq!(a.head().unwrap(), b.head().unwrap());
    }

    #[test]
    fn unresolved_states_are_explicit() {
        let correlation = DurableEffectAttemptCorrelationV1 {
            command_digest: Digest32([1; 32]),
            envelope_digest: Digest32([2; 32]),
            composition_digest: Digest32([3; 32]),
            device: "iot:valve:72".into(),
            operation: "open".into(),
            executor: "gateway:a".into(),
            sequence: 7,
            adapter_id: "hal:valve-72".into(),
            common_fenced_at_unix_ms: 1_000,
            wall_valid_until_unix_ms: 2_000,
        };
        assert!(
            DurableEffectAttemptStateV1::Prepared {
                correlation: correlation.clone()
            }
            .requires_reconciliation()
        );
        assert!(
            DurableEffectAttemptStateV1::AdapterAcknowledged {
                correlation: correlation.clone(),
                adapter_evidence_digest: Digest32([4; 32]),
            }
            .requires_reconciliation()
        );
        assert!(
            DurableEffectAttemptStateV1::AdapterIndeterminate {
                correlation: correlation.clone()
            }
            .requires_reconciliation()
        );
        assert!(
            !DurableEffectAttemptStateV1::AbandonedBeforePort { correlation }
                .requires_reconciliation()
        );
    }
}
