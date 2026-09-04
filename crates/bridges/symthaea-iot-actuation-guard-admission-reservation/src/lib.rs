// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Crash-durable pre-semantic reservation for privileged cyber-physical actuation.
//!
//! This crate consumes an already verified Xenia transport envelope, performs only
//! static device-policy checks, burns the command sequence in a restart-durable journal,
//! re-reads the exact persisted bytes, and returns an opaque persistence proof.
//!
//! It deliberately does **not** claim runtime firmware/observation safety, semantic
//! acceptance, controller-interlock verification, final authority, or physical I/O.
//! The filesystem journal is crash/restart durable; it is not a hardware anti-rollback
//! anchor against a privileged attacker restoring old disk state.

#![deny(unsafe_code)]

#[cfg(not(target_os = "linux"))]
compile_error!("the privileged IoT admission reservation store is Linux-only");

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use symthaea_authority::{Digest32, ResourceRef};
use symthaea_iot_device_protocol::{DeviceEnforcementConfigV1, PhysicalEffectEnvelopeV1};
use symthaea_iot_transport_receipt::{TransportTrustHead, VerifiedTransportEnvelope};
use thiserror::Error;

pub const ADMISSION_RESERVATION_CHECKPOINT_SCHEMA_VERSION: u16 = 1;
pub const MAX_ADMISSION_RESERVATION_STATE_BYTES: u64 = 64 * 1024;

const CHECKPOINT_DOMAIN: &[u8] = b"symthaea-iot-admission-reservation-checkpoint-v1\0";
const STATE_FILE_NAME: &str = "admission-reservation.state";
const LOCK_FILE_NAME: &str = ".admission-reservation.lock";
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdmissionReservationHead {
    pub generation: u64,
    pub digest: Digest32,
}

/// Durable replay tombstone. `highest_reserved_sequence` does not mean the effect was
/// semantically safe, dispatched, or applied.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdmissionReservationCheckpointV1 {
    pub schema_version: u16,
    pub generation: u64,
    pub previous_checkpoint_digest: Option<Digest32>,
    pub device: ResourceRef,
    pub config_digest: Digest32,
    pub highest_reserved_sequence: Option<u64>,
    pub last_envelope_digest: Option<Digest32>,
    pub last_transport_receipt_digest: Option<Digest32>,
    pub last_transport_trust_head: Option<TransportTrustHead>,
}

impl AdmissionReservationCheckpointV1 {
    pub fn genesis(config: &DeviceEnforcementConfigV1) -> Result<Self, AdmissionReservationError> {
        config
            .validate()
            .map_err(|_| AdmissionReservationError::InvalidDeviceConfig)?;
        Ok(Self {
            schema_version: ADMISSION_RESERVATION_CHECKPOINT_SCHEMA_VERSION,
            generation: 0,
            previous_checkpoint_digest: None,
            device: config.device.clone(),
            config_digest: config
                .digest()
                .map_err(|_| AdmissionReservationError::InvalidDeviceConfig)?,
            highest_reserved_sequence: None,
            last_envelope_digest: None,
            last_transport_receipt_digest: None,
            last_transport_trust_head: None,
        })
    }

    pub fn validate(&self) -> Result<(), AdmissionReservationError> {
        if self.schema_version != ADMISSION_RESERVATION_CHECKPOINT_SCHEMA_VERSION {
            return Err(AdmissionReservationError::UnsupportedCheckpointSchema);
        }
        if self.device.0.is_empty() || self.config_digest == Digest32([0; 32]) {
            return Err(AdmissionReservationError::InvalidCheckpoint);
        }
        if self.generation == 0 {
            if self.previous_checkpoint_digest.is_some()
                || self.highest_reserved_sequence.is_some()
                || self.last_envelope_digest.is_some()
                || self.last_transport_receipt_digest.is_some()
                || self.last_transport_trust_head.is_some()
            {
                return Err(AdmissionReservationError::MalformedGenesis);
            }
            return Ok(());
        }

        let sequence = self
            .highest_reserved_sequence
            .ok_or(AdmissionReservationError::IncompleteCheckpoint)?;
        let envelope_digest = self
            .last_envelope_digest
            .ok_or(AdmissionReservationError::IncompleteCheckpoint)?;
        let receipt_digest = self
            .last_transport_receipt_digest
            .ok_or(AdmissionReservationError::IncompleteCheckpoint)?;
        let trust_head = self
            .last_transport_trust_head
            .ok_or(AdmissionReservationError::IncompleteCheckpoint)?;
        if self.previous_checkpoint_digest.is_none()
            || sequence == 0
            || envelope_digest == Digest32([0; 32])
            || receipt_digest == Digest32([0; 32])
            || trust_head.sequence == 0
            || trust_head.digest == Digest32([0; 32])
        {
            return Err(AdmissionReservationError::IncompleteCheckpoint);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<Digest32, AdmissionReservationError> {
        self.validate()?;
        let mut h = blake3::Hasher::new();
        h.update(CHECKPOINT_DOMAIN);
        h.update(&self.schema_version.to_be_bytes());
        h.update(&self.generation.to_be_bytes());
        optional_digest(&mut h, self.previous_checkpoint_digest);
        update_string(&mut h, &self.device.0);
        update_digest(&mut h, self.config_digest);
        optional_u64(&mut h, self.highest_reserved_sequence);
        optional_digest(&mut h, self.last_envelope_digest);
        optional_digest(&mut h, self.last_transport_receipt_digest);
        match self.last_transport_trust_head {
            Some(head) => {
                h.update(&[1]);
                h.update(&head.sequence.to_be_bytes());
                update_digest(&mut h, head.digest);
            }
            None => {
                h.update(&[0]);
            }
        }
        Ok(Digest32(*h.finalize().as_bytes()))
    }

    pub fn head(&self) -> Result<AdmissionReservationHead, AdmissionReservationError> {
        Ok(AdmissionReservationHead {
            generation: self.generation,
            digest: self.digest()?,
        })
    }

    fn successor(
        &self,
        config: &DeviceEnforcementConfigV1,
        transport: &VerifiedTransportEnvelope,
    ) -> Result<Self, AdmissionReservationError> {
        self.validate()?;
        let config_digest = config
            .digest()
            .map_err(|_| AdmissionReservationError::InvalidDeviceConfig)?;
        if self.device != config.device || self.config_digest != config_digest {
            return Err(AdmissionReservationError::CheckpointConfigMismatch);
        }
        Ok(Self {
            schema_version: ADMISSION_RESERVATION_CHECKPOINT_SCHEMA_VERSION,
            generation: self
                .generation
                .checked_add(1)
                .ok_or(AdmissionReservationError::GenerationOverflow)?,
            previous_checkpoint_digest: Some(self.digest()?),
            device: self.device.clone(),
            config_digest,
            highest_reserved_sequence: Some(transport.envelope().command.sequence),
            last_envelope_digest: Some(transport.envelope_digest()),
            last_transport_receipt_digest: Some(transport.receipt_digest()),
            last_transport_trust_head: Some(transport.trust_head()),
        })
    }
}

#[derive(Debug)]
pub struct PersistedAdmissionReservation {
    transport: VerifiedTransportEnvelope,
    checkpoint: AdmissionReservationCheckpointV1,
    head: AdmissionReservationHead,
    persisted_at_unix_ms: u64,
}

impl PersistedAdmissionReservation {
    pub const fn head(&self) -> AdmissionReservationHead {
        self.head
    }

    pub const fn persisted_at_unix_ms(&self) -> u64 {
        self.persisted_at_unix_ms
    }

    pub fn checkpoint(&self) -> &AdmissionReservationCheckpointV1 {
        &self.checkpoint
    }

    pub fn envelope(&self) -> &PhysicalEffectEnvelopeV1 {
        self.transport.envelope()
    }

    pub const fn envelope_digest(&self) -> Digest32 {
        self.transport.envelope_digest()
    }

    pub const fn transport_receipt_digest(&self) -> Digest32 {
        self.transport.receipt_digest()
    }

    pub const fn transport_trust_head(&self) -> TransportTrustHead {
        self.transport.trust_head()
    }

    pub fn into_transport(self) -> VerifiedTransportEnvelope {
        self.transport
    }
}

/// Restart-durable, single-device reservation journal rooted at a pinned Linux directory.
pub struct DurableAdmissionReservationStore {
    root: PathBuf,
    config: DeviceEnforcementConfigV1,
    config_digest: Digest32,
    local_lock: Mutex<()>,
    pinned_root: Mutex<Option<Arc<File>>>,
}

impl DurableAdmissionReservationStore {
    pub fn open(
        root: impl Into<PathBuf>,
        config: DeviceEnforcementConfigV1,
    ) -> Result<Self, AdmissionReservationError> {
        config
            .validate()
            .map_err(|_| AdmissionReservationError::InvalidDeviceConfig)?;
        let config_digest = config
            .digest()
            .map_err(|_| AdmissionReservationError::InvalidDeviceConfig)?;
        let store = Self {
            root: root.into(),
            config,
            config_digest,
            local_lock: Mutex::new(()),
            pinned_root: Mutex::new(None),
        };
        store.ensure_root()?;
        {
            let _local = store
                .local_lock
                .lock()
                .map_err(|_| AdmissionReservationError::LocalLockPoisoned)?;
            let kernel = store.open_lock_file()?;
            kernel.lock().map_err(AdmissionReservationError::Io)?;
            let state = store.read_state_locked()?;
            store.validate_checkpoint_for_store(&state)?;
        }
        Ok(store)
    }

    pub fn config(&self) -> &DeviceEnforcementConfigV1 {
        &self.config
    }

    pub const fn config_digest(&self) -> Digest32 {
        self.config_digest
    }

    pub fn current_checkpoint(
        &self,
    ) -> Result<AdmissionReservationCheckpointV1, AdmissionReservationError> {
        let _local = self
            .local_lock
            .lock()
            .map_err(|_| AdmissionReservationError::LocalLockPoisoned)?;
        let kernel = self.open_lock_file()?;
        kernel.lock().map_err(AdmissionReservationError::Io)?;
        self.read_state_locked()
    }

    pub fn current_head(&self) -> Result<AdmissionReservationHead, AdmissionReservationError> {
        self.current_checkpoint()?.head()
    }

    /// Persist one already-authenticated envelope. The production path always uses the
    /// guard's own wall clock; callers cannot choose the reservation time.
    pub fn reserve_verified_transport(
        &self,
        transport: VerifiedTransportEnvelope,
    ) -> Result<PersistedAdmissionReservation, AdmissionReservationError> {
        self.reserve_verified_transport_at(transport, system_unix_ms()?)
    }

    fn reserve_verified_transport_at(
        &self,
        transport: VerifiedTransportEnvelope,
        now_unix_ms: u64,
    ) -> Result<PersistedAdmissionReservation, AdmissionReservationError> {
        let _local = self
            .local_lock
            .lock()
            .map_err(|_| AdmissionReservationError::LocalLockPoisoned)?;
        let kernel = self.open_lock_file()?;
        kernel.lock().map_err(AdmissionReservationError::Io)?;

        let current = self.read_state_locked()?;
        self.validate_checkpoint_for_store(&current)?;
        validate_static_reservation(
            transport.envelope(),
            transport.opened_at_unix_ms(),
            &self.config,
            &current,
            now_unix_ms,
        )?;

        let successor = current.successor(&self.config, &transport)?;
        let expected_head = successor.head()?;
        self.write_state_locked(&successor)?;

        // A write intent is not a persistence proof. Re-open and validate exact bytes.
        let persisted = self.read_state_locked()?;
        if persisted != successor || persisted.head()? != expected_head {
            return Err(AdmissionReservationError::PersistedStateReadbackMismatch);
        }

        Ok(PersistedAdmissionReservation {
            transport,
            checkpoint: persisted,
            head: expected_head,
            persisted_at_unix_ms: system_unix_ms()?,
        })
    }

    fn validate_checkpoint_for_store(
        &self,
        checkpoint: &AdmissionReservationCheckpointV1,
    ) -> Result<(), AdmissionReservationError> {
        checkpoint.validate()?;
        if checkpoint.device != self.config.device || checkpoint.config_digest != self.config_digest {
            return Err(AdmissionReservationError::CheckpointConfigMismatch);
        }
        Ok(())
    }

    fn read_state_locked(
        &self,
    ) -> Result<AdmissionReservationCheckpointV1, AdmissionReservationError> {
        let path = self.operation_root_path()?.join(STATE_FILE_NAME);
        let file = match open_regular_file(&path, false, false) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return AdmissionReservationCheckpointV1::genesis(&self.config);
            }
            Err(error) => return Err(AdmissionReservationError::Io(error)),
        };
        let metadata = file.metadata().map_err(AdmissionReservationError::Io)?;
        if metadata.len() == 0 || metadata.len() > MAX_ADMISSION_RESERVATION_STATE_BYTES {
            return Err(AdmissionReservationError::StateSizeOutOfBounds);
        }
        let mut bytes = Vec::with_capacity(metadata.len() as usize);
        file.take(MAX_ADMISSION_RESERVATION_STATE_BYTES.saturating_add(1))
            .read_to_end(&mut bytes)
            .map_err(AdmissionReservationError::Io)?;
        if bytes.is_empty() || bytes.len() as u64 > MAX_ADMISSION_RESERVATION_STATE_BYTES {
            return Err(AdmissionReservationError::StateSizeOutOfBounds);
        }
        let checkpoint: AdmissionReservationCheckpointV1 =
            bincode::deserialize(&bytes).map_err(|_| AdmissionReservationError::StateEncoding)?;
        checkpoint.validate()?;
        let canonical =
            bincode::serialize(&checkpoint).map_err(|_| AdmissionReservationError::StateEncoding)?;
        if canonical != bytes {
            return Err(AdmissionReservationError::NonCanonicalStateEncoding);
        }
        self.validate_checkpoint_for_store(&checkpoint)?;
        Ok(checkpoint)
    }

    fn write_state_locked(
        &self,
        checkpoint: &AdmissionReservationCheckpointV1,
    ) -> Result<(), AdmissionReservationError> {
        self.validate_checkpoint_for_store(checkpoint)?;
        let encoded =
            bincode::serialize(checkpoint).map_err(|_| AdmissionReservationError::StateEncoding)?;
        if encoded.is_empty() || encoded.len() as u64 > MAX_ADMISSION_RESERVATION_STATE_BYTES {
            return Err(AdmissionReservationError::StateSizeOutOfBounds);
        }

        let root = self.ensure_root()?;
        let operation_root = self.operation_root_path()?;
        let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| AdmissionReservationError::SystemClockBeforeUnixEpoch)?
            .as_nanos();
        let temp = operation_root.join(format!(
            ".admission-reservation-{}-{counter}-{nanos}.tmp",
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
        result.map_err(AdmissionReservationError::Io)
    }

    fn open_lock_file(&self) -> Result<File, AdmissionReservationError> {
        let path = self.operation_root_path()?.join(LOCK_FILE_NAME);
        open_regular_file(&path, true, false).map_err(AdmissionReservationError::Io)
    }

    fn ensure_root(&self) -> Result<Arc<File>, AdmissionReservationError> {
        let mut pinned = self
            .pinned_root
            .lock()
            .map_err(|_| AdmissionReservationError::RootLockPoisoned)?;
        if let Some(root) = pinned.as_ref() {
            return Ok(Arc::clone(root));
        }

        match fs::symlink_metadata(&self.root) {
            Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
                return Err(AdmissionReservationError::InvalidRootDirectory);
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                fs::create_dir_all(&self.root).map_err(AdmissionReservationError::Io)?;
            }
            Err(error) => return Err(AdmissionReservationError::Io(error)),
        }
        fs::set_permissions(&self.root, fs::Permissions::from_mode(0o700))
            .map_err(AdmissionReservationError::Io)?;

        let mut options = OpenOptions::new();
        options
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW);
        let root = Arc::new(options.open(&self.root).map_err(AdmissionReservationError::Io)?);
        if !root
            .metadata()
            .map_err(AdmissionReservationError::Io)?
            .is_dir()
        {
            return Err(AdmissionReservationError::InvalidRootDirectory);
        }
        *pinned = Some(Arc::clone(&root));
        Ok(root)
    }

    fn operation_root_path(&self) -> Result<PathBuf, AdmissionReservationError> {
        let root = self.ensure_root()?;
        let path = PathBuf::from(format!("/proc/self/fd/{}", root.as_raw_fd()));
        if !path.is_dir() {
            return Err(AdmissionReservationError::PinnedRootUnavailable);
        }
        Ok(path)
    }
}

fn validate_static_reservation(
    envelope: &PhysicalEffectEnvelopeV1,
    transport_opened_at_unix_ms: u64,
    config: &DeviceEnforcementConfigV1,
    current: &AdmissionReservationCheckpointV1,
    now_unix_ms: u64,
) -> Result<(), AdmissionReservationError> {
    envelope
        .validate_structure()
        .map_err(|_| AdmissionReservationError::InvalidPhysicalEnvelope)?;
    config
        .validate()
        .map_err(|_| AdmissionReservationError::InvalidDeviceConfig)?;
    current.validate()?;

    if envelope.command.device != config.device || current.device != config.device {
        return Err(AdmissionReservationError::DeviceMismatch);
    }
    if envelope.command.operation != config.operation {
        return Err(AdmissionReservationError::OperationMismatch);
    }
    if envelope.policy_digest != config.exact_policy_digest {
        return Err(AdmissionReservationError::PolicyDigestMismatch);
    }
    if envelope.policy_registry_head.sequence < config.minimum_policy_registry_sequence {
        return Err(AdmissionReservationError::PolicyGenerationTooOld);
    }
    let envelope_lifetime_s = envelope
        .send_not_after_unix_s
        .checked_sub(envelope.host_preflight_at_unix_s)
        .ok_or(AdmissionReservationError::InvalidEnvelopeLifetime)?;
    if envelope_lifetime_s > config.maximum_envelope_lifetime_s {
        return Err(AdmissionReservationError::EnvelopeLifetimeExceedsDevicePolicy);
    }
    if !config
        .safety
        .allowed_firmware
        .contains(&envelope.command.expected_firmware)
    {
        return Err(AdmissionReservationError::ExpectedFirmwareNotAllowed);
    }
    if config
        .safety
        .parameter_ranges
        .keys()
        .any(|name| !envelope.command.parameters.contains_key(name))
    {
        return Err(AdmissionReservationError::MissingCommandParameter);
    }
    for (name, value) in &envelope.command.parameters {
        let Some(range) = config.safety.parameter_ranges.get(name) else {
            return Err(AdmissionReservationError::UnknownCommandParameter);
        };
        if !range.contains(*value) {
            return Err(AdmissionReservationError::CommandParameterOutOfRange);
        }
    }
    if current
        .highest_reserved_sequence
        .is_some_and(|highest| envelope.command.sequence <= highest)
    {
        return Err(AdmissionReservationError::SequenceAlreadyReserved {
            proposed: envelope.command.sequence,
            highest: current.highest_reserved_sequence.unwrap_or_default(),
        });
    }

    let send_not_after_unix_ms = envelope
        .send_not_after_unix_s
        .checked_mul(1_000)
        .ok_or(AdmissionReservationError::TimeOverflow)?;
    if now_unix_ms < transport_opened_at_unix_ms || now_unix_ms > send_not_after_unix_ms {
        return Err(AdmissionReservationError::EnvelopeNotFreshForReservation);
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
            "admission reservation object is not a regular file",
        ));
    }
    Ok(file)
}

fn system_unix_ms() -> Result<u64, AdmissionReservationError> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| AdmissionReservationError::SystemClockBeforeUnixEpoch)?;
    u64::try_from(elapsed.as_millis()).map_err(|_| AdmissionReservationError::TimeOverflow)
}

fn update_string(h: &mut blake3::Hasher, value: &str) {
    h.update(&(value.len() as u32).to_be_bytes());
    h.update(value.as_bytes());
}

fn update_digest(h: &mut blake3::Hasher, Digest32(value): Digest32) {
    h.update(&value);
}

fn optional_digest(h: &mut blake3::Hasher, value: Option<Digest32>) {
    match value {
        Some(value) => {
            h.update(&[1]);
            update_digest(h, value);
        }
        None => {
            h.update(&[0]);
        }
    }
}

fn optional_u64(h: &mut blake3::Hasher, value: Option<u64>) {
    match value {
        Some(value) => {
            h.update(&[1]);
            h.update(&value.to_be_bytes());
        }
        None => {
            h.update(&[0]);
        }
    }
}

#[derive(Debug, Error)]
pub enum AdmissionReservationError {
    #[error("unsupported admission-reservation checkpoint schema")]
    UnsupportedCheckpointSchema,
    #[error("admission-reservation checkpoint is invalid")]
    InvalidCheckpoint,
    #[error("admission-reservation genesis is malformed")]
    MalformedGenesis,
    #[error("admission-reservation checkpoint is incomplete")]
    IncompleteCheckpoint,
    #[error("device enforcement configuration is invalid")]
    InvalidDeviceConfig,
    #[error("persisted reservation belongs to another device/configuration")]
    CheckpointConfigMismatch,
    #[error("admission-reservation generation overflow")]
    GenerationOverflow,
    #[error("physical-effect envelope is invalid")]
    InvalidPhysicalEnvelope,
    #[error("physical-effect device differs from guard reservation policy")]
    DeviceMismatch,
    #[error("physical-effect operation differs from guard reservation policy")]
    OperationMismatch,
    #[error("physical-effect policy digest differs from guard reservation policy")]
    PolicyDigestMismatch,
    #[error("physical-effect policy generation is below the guard minimum")]
    PolicyGenerationTooOld,
    #[error("physical-effect envelope has an invalid local lifetime")]
    InvalidEnvelopeLifetime,
    #[error("physical-effect envelope lifetime exceeds the device-local ceiling")]
    EnvelopeLifetimeExceedsDevicePolicy,
    #[error("command expected firmware is outside the configured allowed firmware set")]
    ExpectedFirmwareNotAllowed,
    #[error("command omitted a parameter required by local static policy")]
    MissingCommandParameter,
    #[error("command supplied an unknown parameter")]
    UnknownCommandParameter,
    #[error("command parameter is outside the local static range")]
    CommandParameterOutOfRange,
    #[error("command sequence {proposed} is not newer than durable reservation {highest}")]
    SequenceAlreadyReserved { proposed: u64, highest: u64 },
    #[error("physical-effect envelope is not fresh enough to reserve")]
    EnvelopeNotFreshForReservation,
    #[error("reservation state size is outside accepted bounds")]
    StateSizeOutOfBounds,
    #[error("reservation state encoding/decoding failed")]
    StateEncoding,
    #[error("reservation state is not canonically encoded")]
    NonCanonicalStateEncoding,
    #[error("persisted reservation did not read back as the exact successor state")]
    PersistedStateReadbackMismatch,
    #[error("reservation root is not a real directory")]
    InvalidRootDirectory,
    #[error("pinned descriptor-relative reservation root is unavailable")]
    PinnedRootUnavailable,
    #[error("reservation local lock is poisoned")]
    LocalLockPoisoned,
    #[error("reservation root lock is poisoned")]
    RootLockPoisoned,
    #[error("system clock predates Unix epoch")]
    SystemClockBeforeUnixEpoch,
    #[error("time conversion overflow")]
    TimeOverflow,
    #[error("admission-reservation I/O failed: {0}")]
    Io(#[source] std::io::Error),
}
