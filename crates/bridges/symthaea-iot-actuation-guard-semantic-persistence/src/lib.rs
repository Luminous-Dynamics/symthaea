// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Crash-durable semantic acceptance after fixed authenticated device reality.
//!
//! The admission journal burns a command sequence before device reality is known. This
//! crate performs the stronger second durable transition only after an opaque
//! `VerifiedAdmissionDeviceReality` exists and the exact admission reservation, envelope,
//! configuration and transport lineage all agree.
//!
//! Successful persistence is still not physical authority. Controller interlock evidence,
//! final composition, JIT revocation fencing and HAL/device I/O remain later stages.

#![deny(unsafe_code)]

#[cfg(not(target_os = "linux"))]
compile_error!(
    "symthaea-iot-actuation-guard-semantic-persistence is Linux-only and relies on a pinned directory capability"
);

mod current;
pub use current::{CurrentSemanticHeadFence, CurrentSemanticHeadFenceError};

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use symthaea_authority::Digest32;
use symthaea_iot_actuation_guard_admission_reservation::PersistedAdmissionReservation;
use symthaea_iot_actuation_guard_device_reality::VerifiedAdmissionDeviceReality;
use symthaea_iot_device_protocol::{
    DeviceEnforcementConfigV1, DeviceProtocolError, DeviceSemanticCheckpointV1,
    DeviceSemanticHead, SemanticallyAcceptedEffect, prepare_semantic_acceptance,
};
use thiserror::Error;

/// Maximum canonical semantic checkpoint accepted from persistent storage.
pub const MAX_SEMANTIC_ACCEPTANCE_STATE_BYTES: u64 = 64 * 1024;

const STATE_FILE_NAME: &str = "semantic-acceptance.state";
const LOCK_FILE_NAME: &str = ".semantic-acceptance.lock";
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Opaque proof that semantic acceptance reached crash-durable storage after the exact
/// authenticated admission reservation and exact trusted device-reality appraisal.
///
/// The contained `DeviceSemanticHead` should be retained in an independently protected
/// anchor before opening the store for a later command. This crate verifies such an
/// anchor on open but does not itself claim TPM/NVRAM anti-rollback protection.
#[derive(Debug)]
pub struct PersistedSemanticAcceptance {
    admission: PersistedAdmissionReservation,
    device_reality: VerifiedAdmissionDeviceReality,
    semantic_effect: SemanticallyAcceptedEffect,
    checkpoint: DeviceSemanticCheckpointV1,
    device_head: DeviceSemanticHead,
    semantic_persisted_at_unix_ms: u64,
}

impl PersistedSemanticAcceptance {
    /// Exact crash-durable semantic head that must be independently retained for restart.
    pub const fn device_head(&self) -> DeviceSemanticHead {
        self.device_head
    }

    /// Guard-local wall time recorded only after fsync, rename, directory fsync and read-back.
    pub const fn semantic_persisted_at_unix_ms(&self) -> u64 {
        self.semantic_persisted_at_unix_ms
    }

    /// Exact durable semantic checkpoint bytes represented by `device_head`.
    pub fn checkpoint(&self) -> &DeviceSemanticCheckpointV1 {
        &self.checkpoint
    }

    /// Exact upstream crash-durable admission reservation retained for later correlation.
    pub fn admission_reservation(&self) -> &PersistedAdmissionReservation {
        &self.admission
    }

    /// Exact fixed-key authenticated device-reality proof retained for later correlation.
    pub fn device_reality(&self) -> &VerifiedAdmissionDeviceReality {
        &self.device_reality
    }

    /// Existing opaque semantic acceptance produced only after the persisted read-back.
    pub fn semantic_effect(&self) -> &SemanticallyAcceptedEffect {
        &self.semantic_effect
    }

    /// Exact physical-envelope commitment shared by all upstream proof stages.
    pub const fn envelope_digest(&self) -> Digest32 {
        self.device_reality.envelope_digest()
    }

    /// Exact whole signed device-attestation object commitment.
    pub const fn device_attestation_object_digest(&self) -> Digest32 {
        self.device_reality.attestation_object_digest()
    }
}

/// Single-operation crash-durable semantic store opened against an independently retained
/// current `DeviceSemanticHead`.
///
/// The handle is consumed by `persist_semantic_acceptance`. After a successful write, the
/// returned new head must be anchored independently before the next store instance is opened.
/// If the process crashes after disk persistence but before that external anchor is advanced,
/// restart fails closed because the old anchor no longer matches the newer disk checkpoint.
pub struct DurableSemanticAcceptanceStore {
    root: PathBuf,
    config: DeviceEnforcementConfigV1,
    config_digest: Digest32,
    trusted_current_head: DeviceSemanticHead,
    local_lock: Mutex<()>,
    pinned_root: Mutex<Option<Arc<File>>>,
}

impl DurableSemanticAcceptanceStore {
    /// Open one semantic-persistence operation against an independently retained current head.
    pub fn open(
        root: impl Into<PathBuf>,
        config: DeviceEnforcementConfigV1,
        trusted_current_head: DeviceSemanticHead,
    ) -> Result<Self, SemanticPersistenceError> {
        config
            .validate()
            .map_err(SemanticPersistenceError::SemanticPolicy)?;
        let config_digest = config
            .digest()
            .map_err(SemanticPersistenceError::SemanticPolicy)?;
        let store = Self {
            root: root.into(),
            config,
            config_digest,
            trusted_current_head,
            local_lock: Mutex::new(()),
            pinned_root: Mutex::new(None),
        };
        store.ensure_root()?;
        {
            let _local = store
                .local_lock
                .lock()
                .map_err(|_| SemanticPersistenceError::LocalLockPoisoned)?;
            let kernel = store.open_lock_file()?;
            kernel.lock().map_err(SemanticPersistenceError::Io)?;
            let checkpoint = store.read_state_locked()?;
            store.verify_loaded_checkpoint(&checkpoint)?;
        }
        Ok(store)
    }

    /// Configured exact device-enforcement policy used by semantic evaluation.
    pub fn config(&self) -> &DeviceEnforcementConfigV1 {
        &self.config
    }

    /// Exact configured policy commitment.
    pub const fn config_digest(&self) -> Digest32 {
        self.config_digest
    }

    /// Independently retained head against which this single operation was opened.
    pub const fn trusted_current_head(&self) -> DeviceSemanticHead {
        self.trusted_current_head
    }

    /// Persist semantic acceptance for one exact admission/reality pair.
    ///
    /// This consumes the store handle so a second command cannot be accepted under a head
    /// that has not first been independently retained by the surrounding guard deployment.
    pub fn persist_semantic_acceptance(
        self,
        admission: PersistedAdmissionReservation,
        device_reality: VerifiedAdmissionDeviceReality,
    ) -> Result<PersistedSemanticAcceptance, SemanticPersistenceError> {
        let _local = self
            .local_lock
            .lock()
            .map_err(|_| SemanticPersistenceError::LocalLockPoisoned)?;
        let kernel = self.open_lock_file()?;
        kernel.lock().map_err(SemanticPersistenceError::Io)?;

        // Re-read only after acquiring both local and cross-process locks. All freshness,
        // lineage and semantic policy checks below are therefore evaluated against the exact
        // state that will be replaced, with no pre-lock device-reality TOCTOU window.
        let current = self.read_state_locked()?;
        self.verify_loaded_checkpoint(&current)?;
        let validation_started_at_unix_ms = system_unix_ms()?;
        validate_lineage(
            &self.config,
            self.config_digest,
            &admission,
            &device_reality,
            validation_started_at_unix_ms,
        )?;

        let now_unix_s = validation_started_at_unix_ms / 1_000;
        let pending = prepare_semantic_acceptance(
            admission.envelope().clone(),
            &self.config,
            device_reality.runtime_state(),
            &current,
            self.trusted_current_head,
            now_unix_s,
        )?;
        if pending.envelope_digest() != admission.envelope_digest() {
            return Err(SemanticPersistenceError::SemanticEnvelopeDigestMismatch);
        }
        let successor = pending.checkpoint().clone();
        let expected_head = pending.expected_head();
        if successor.head()? != expected_head {
            return Err(SemanticPersistenceError::SemanticSuccessorHeadMismatch);
        }

        self.write_state_locked(&successor)?;
        let persisted = self.read_state_locked()?;
        if persisted != successor || persisted.head()? != expected_head {
            return Err(SemanticPersistenceError::PersistedStateReadbackMismatch);
        }

        let semantic_effect = pending
            .confirm_persisted(expected_head)
            .map_err(|_| SemanticPersistenceError::SemanticConfirmationMismatch)?;
        if semantic_effect.device_head() != expected_head
            || semantic_effect.envelope_digest() != admission.envelope_digest()
        {
            return Err(SemanticPersistenceError::SemanticConfirmationMismatch);
        }

        let semantic_persisted_at_unix_ms = system_unix_ms()?;
        if semantic_persisted_at_unix_ms < validation_started_at_unix_ms
            || semantic_persisted_at_unix_ms < device_reality.verified_at_unix_ms()
        {
            // Disk may already contain the successor. Returning no proof is deliberately
            // fail-closed; restart must reconcile the independently retained semantic head.
            return Err(SemanticPersistenceError::SystemClockRegressedDuringPersistence);
        }

        Ok(PersistedSemanticAcceptance {
            admission,
            device_reality,
            semantic_effect,
            checkpoint: persisted,
            device_head: expected_head,
            semantic_persisted_at_unix_ms,
        })
    }

    fn verify_loaded_checkpoint(
        &self,
        checkpoint: &DeviceSemanticCheckpointV1,
    ) -> Result<(), SemanticPersistenceError> {
        checkpoint.validate()?;
        if checkpoint.device != self.config.device || checkpoint.config_digest != self.config_digest {
            return Err(SemanticPersistenceError::CheckpointConfigMismatch);
        }
        if checkpoint.head()? != self.trusted_current_head {
            return Err(SemanticPersistenceError::TrustedSemanticHeadMismatch);
        }
        Ok(())
    }

    fn read_state_locked(
        &self,
    ) -> Result<DeviceSemanticCheckpointV1, SemanticPersistenceError> {
        let path = self.operation_root_path()?.join(STATE_FILE_NAME);
        let file = match open_regular_file(&path, false, false) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return DeviceSemanticCheckpointV1::genesis(&self.config)
                    .map_err(SemanticPersistenceError::SemanticPolicy);
            }
            Err(error) => return Err(SemanticPersistenceError::Io(error)),
        };
        let metadata = file.metadata().map_err(SemanticPersistenceError::Io)?;
        if metadata.len() == 0 || metadata.len() > MAX_SEMANTIC_ACCEPTANCE_STATE_BYTES {
            return Err(SemanticPersistenceError::StateSizeOutOfBounds);
        }
        let mut bytes = Vec::with_capacity(metadata.len() as usize);
        file.take(MAX_SEMANTIC_ACCEPTANCE_STATE_BYTES.saturating_add(1))
            .read_to_end(&mut bytes)
            .map_err(SemanticPersistenceError::Io)?;
        if bytes.is_empty() || bytes.len() as u64 > MAX_SEMANTIC_ACCEPTANCE_STATE_BYTES {
            return Err(SemanticPersistenceError::StateSizeOutOfBounds);
        }
        let checkpoint: DeviceSemanticCheckpointV1 =
            bincode::deserialize(&bytes).map_err(|_| SemanticPersistenceError::StateEncoding)?;
        checkpoint.validate()?;
        let canonical =
            bincode::serialize(&checkpoint).map_err(|_| SemanticPersistenceError::StateEncoding)?;
        if canonical != bytes {
            return Err(SemanticPersistenceError::NonCanonicalStateEncoding);
        }
        Ok(checkpoint)
    }

    fn write_state_locked(
        &self,
        checkpoint: &DeviceSemanticCheckpointV1,
    ) -> Result<(), SemanticPersistenceError> {
        checkpoint.validate()?;
        if checkpoint.device != self.config.device || checkpoint.config_digest != self.config_digest {
            return Err(SemanticPersistenceError::CheckpointConfigMismatch);
        }
        let encoded =
            bincode::serialize(checkpoint).map_err(|_| SemanticPersistenceError::StateEncoding)?;
        if encoded.is_empty() || encoded.len() as u64 > MAX_SEMANTIC_ACCEPTANCE_STATE_BYTES {
            return Err(SemanticPersistenceError::StateSizeOutOfBounds);
        }

        let root = self.ensure_root()?;
        let operation_root = self.operation_root_path()?;
        let suffix = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let temp = operation_root.join(format!(
            ".semantic-acceptance-{}-{suffix}.tmp",
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
        result.map_err(SemanticPersistenceError::Io)
    }

    fn open_lock_file(&self) -> Result<File, SemanticPersistenceError> {
        let path = self.operation_root_path()?.join(LOCK_FILE_NAME);
        open_regular_file(&path, true, false).map_err(SemanticPersistenceError::Io)
    }

    fn ensure_root(&self) -> Result<Arc<File>, SemanticPersistenceError> {
        let mut pinned = self
            .pinned_root
            .lock()
            .map_err(|_| SemanticPersistenceError::RootLockPoisoned)?;
        if let Some(root) = pinned.as_ref() {
            return Ok(Arc::clone(root));
        }

        match fs::symlink_metadata(&self.root) {
            Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
                return Err(SemanticPersistenceError::InvalidRootDirectory);
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                fs::create_dir_all(&self.root).map_err(SemanticPersistenceError::Io)?;
            }
            Err(error) => return Err(SemanticPersistenceError::Io(error)),
        }
        fs::set_permissions(&self.root, fs::Permissions::from_mode(0o700))
            .map_err(SemanticPersistenceError::Io)?;

        let mut options = OpenOptions::new();
        options
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW);
        let root = Arc::new(options.open(&self.root).map_err(SemanticPersistenceError::Io)?);
        if !root
            .metadata()
            .map_err(SemanticPersistenceError::Io)?
            .is_dir()
        {
            return Err(SemanticPersistenceError::InvalidRootDirectory);
        }
        *pinned = Some(Arc::clone(&root));
        Ok(root)
    }

    fn operation_root_path(&self) -> Result<PathBuf, SemanticPersistenceError> {
        let root = self.ensure_root()?;
        let path = PathBuf::from(format!("/proc/self/fd/{}", root.as_raw_fd()));
        if !path.is_dir() {
            return Err(SemanticPersistenceError::PinnedRootUnavailable);
        }
        Ok(path)
    }
}

fn validate_lineage(
    config: &DeviceEnforcementConfigV1,
    config_digest: Digest32,
    admission: &PersistedAdmissionReservation,
    device_reality: &VerifiedAdmissionDeviceReality,
    now_unix_ms: u64,
) -> Result<(), SemanticPersistenceError> {
    if admission.checkpoint().config_digest != config_digest
        || device_reality.config_digest() != config_digest
        || admission.envelope().command.device != config.device
        || device_reality.attestation_result().body.device != config.device
    {
        return Err(SemanticPersistenceError::AdmissionRealityConfigMismatch);
    }
    if device_reality.reservation_head() != admission.head() {
        return Err(SemanticPersistenceError::AdmissionReservationHeadMismatch);
    }
    if device_reality.envelope_digest() != admission.envelope_digest() {
        return Err(SemanticPersistenceError::AdmissionRealityEnvelopeMismatch);
    }
    if device_reality.transport_receipt_digest() != admission.transport_receipt_digest()
        || device_reality.transport_trust_head() != admission.transport_trust_head()
    {
        return Err(SemanticPersistenceError::AdmissionRealityTransportMismatch);
    }

    let sequence = admission.envelope().command.sequence;
    if admission.checkpoint().highest_reserved_sequence != Some(sequence)
        || admission.checkpoint().last_envelope_digest != Some(admission.envelope_digest())
        || admission.checkpoint().last_transport_receipt_digest
            != Some(admission.transport_receipt_digest())
        || admission.checkpoint().last_transport_trust_head
            != Some(admission.transport_trust_head())
    {
        return Err(SemanticPersistenceError::AdmissionReservationStateMismatch);
    }

    if device_reality.challenge_digest() == Digest32([0; 32])
        || device_reality.attestation_object_digest() == Digest32([0; 32])
        || device_reality.result_digest() == Digest32([0; 32])
    {
        return Err(SemanticPersistenceError::ZeroDeviceRealityCommitment);
    }
    if device_reality.verified_at_unix_ms() < admission.persisted_at_unix_ms()
        || device_reality.verified_at_unix_ms() > now_unix_ms
    {
        return Err(SemanticPersistenceError::DeviceRealityCausalityMismatch);
    }

    let attestation = device_reality.attestation_result();
    let appraised_at_unix_ms = attestation
        .body
        .appraised_at_unix_s
        .checked_mul(1_000)
        .ok_or(SemanticPersistenceError::TimeOverflow)?;
    let expires_at_unix_ms = attestation
        .body
        .expires_at_unix_s
        .checked_mul(1_000)
        .ok_or(SemanticPersistenceError::TimeOverflow)?;
    if now_unix_ms < appraised_at_unix_ms || now_unix_ms >= expires_at_unix_ms {
        return Err(SemanticPersistenceError::DeviceRealityNotFreshForSemanticCommit);
    }
    Ok(())
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
            "semantic persistence object is not a regular file",
        ));
    }
    Ok(file)
}

fn system_unix_ms() -> Result<u64, SemanticPersistenceError> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| SemanticPersistenceError::SystemClockBeforeUnixEpoch)?;
    u64::try_from(elapsed.as_millis()).map_err(|_| SemanticPersistenceError::TimeOverflow)
}

/// Fail-closed semantic-persistence failure.
#[derive(Debug, Error)]
pub enum SemanticPersistenceError {
    #[error("device semantic protocol/policy rejected the operation: {0}")]
    SemanticPolicy(#[from] DeviceProtocolError),
    #[error("admission reservation and authenticated device reality do not share config/device")]
    AdmissionRealityConfigMismatch,
    #[error("authenticated device reality does not bind the exact admission reservation head")]
    AdmissionReservationHeadMismatch,
    #[error("authenticated device reality does not bind the exact reserved envelope")]
    AdmissionRealityEnvelopeMismatch,
    #[error("authenticated device reality does not bind the exact Xenia transport lineage")]
    AdmissionRealityTransportMismatch,
    #[error("persisted admission checkpoint does not describe the exact reserved command")]
    AdmissionReservationStateMismatch,
    #[error("authenticated device reality contains a zero required commitment")]
    ZeroDeviceRealityCommitment,
    #[error("authenticated device reality violates admission/verification causal order")]
    DeviceRealityCausalityMismatch,
    #[error("authenticated device reality is not fresh at semantic persistence time")]
    DeviceRealityNotFreshForSemanticCommit,
    #[error("semantic checkpoint does not match configured device/configuration")]
    CheckpointConfigMismatch,
    #[error("persisted semantic checkpoint does not match independently retained current head")]
    TrustedSemanticHeadMismatch,
    #[error("pending semantic envelope digest differs from durable admission envelope")]
    SemanticEnvelopeDigestMismatch,
    #[error("semantic successor checkpoint does not produce its expected head")]
    SemanticSuccessorHeadMismatch,
    #[error("persisted semantic bytes/read-back do not match intended successor")]
    PersistedStateReadbackMismatch,
    #[error("semantic confirmation after durable read-back failed")]
    SemanticConfirmationMismatch,
    #[error("semantic persistence root is invalid or symlinked")]
    InvalidRootDirectory,
    #[error("pinned semantic persistence root is unavailable")]
    PinnedRootUnavailable,
    #[error("semantic persistence state exceeds its bounded size")]
    StateSizeOutOfBounds,
    #[error("semantic persistence state encoding is invalid")]
    StateEncoding,
    #[error("semantic persistence state is not canonically encoded")]
    NonCanonicalStateEncoding,
    #[error("semantic persistence process-local lock is poisoned")]
    LocalLockPoisoned,
    #[error("semantic persistence root lock is poisoned")]
    RootLockPoisoned,
    #[error("system wall clock is before Unix epoch")]
    SystemClockBeforeUnixEpoch,
    #[error("semantic persistence time conversion overflow")]
    TimeOverflow,
    #[error("system wall clock regressed during semantic persistence")]
    SystemClockRegressedDuringPersistence,
    #[error("semantic persistence I/O failed: {0}")]
    Io(#[source] std::io::Error),
}
