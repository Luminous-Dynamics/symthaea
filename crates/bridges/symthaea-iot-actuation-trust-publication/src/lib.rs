// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Crash-durable atomic publication of the trust/policy roots that are current for actuation.
//!
//! Transport trust, device-reality trust/policy and controller/interlock trust/policy may be
//! prepared and verified independently, but a physical-actuation boundary needs one authoritative
//! answer to which combination is current. This crate provides that publication meta-root.
//!
//! Publishing a successor and fencing the current publication use the same cross-process file
//! lock. A later actuation linearization boundary can therefore hold [`CurrentActuationTrustFence`]
//! while it obtains the three owner-local cryptographic/currentness fences, preventing a trust or
//! policy publication from becoming authoritative halfway through the attempt.
//!
//! This crate contains no cryptographic verifier, final permit, HAL capability or actuator I/O.

#![deny(unsafe_code)]

#[cfg(not(target_os = "linux"))]
compile_error!("the privileged IoT actuation trust publication store is Linux-only");

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use symthaea_authority::{Digest32, ResourceRef};
use symthaea_iot_actuation_guard_device_reality::DeviceRealityTrustHead;
use symthaea_iot_interlock_trust::InterlockTrustHead;
use symthaea_iot_transport_receipt::TransportTrustHead;
use thiserror::Error;

pub const ACTUATION_TRUST_PUBLICATION_SCHEMA_VERSION: u16 = 1;
pub const MAX_ACTUATION_TRUST_STATE_BYTES: u64 = 64 * 1024;

const PUBLICATION_DOMAIN: &[u8] = b"symthaea-iot-actuation-trust-publication-v1\0";
const STATE_FILE_NAME: &str = "actuation-trust-publication.state";
const LOCK_FILE_NAME: &str = ".actuation-trust-publication.lock";
const MAX_DEVICE_ID_BYTES: usize = 512;
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Monotonic external anchor for one guard-owned policy commitment.
///
/// The policy objects themselves remain in their owning crates. This publication root only records
/// which policy generation/digest is authoritative for actuation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActuationPolicyAnchorV1 {
    pub generation: u64,
    pub digest: Digest32,
}

impl ActuationPolicyAnchorV1 {
    pub fn validate(&self) -> Result<(), ActuationTrustPublicationError> {
        if self.generation == 0 {
            return Err(ActuationTrustPublicationError::PolicyGenerationZero);
        }
        if self.digest == Digest32([0; 32]) {
            return Err(ActuationTrustPublicationError::PolicyDigestZero);
        }
        Ok(())
    }
}

/// Exact trust/policy roots proposed as one authoritative actuation view.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActuationTrustRootsV1 {
    pub device: ResourceRef,
    pub transport_trust_head: TransportTrustHead,
    pub device_reality_trust_head: DeviceRealityTrustHead,
    pub device_reality_policy: ActuationPolicyAnchorV1,
    pub interlock_trust_head: InterlockTrustHead,
    pub interlock_policy: ActuationPolicyAnchorV1,
}

impl ActuationTrustRootsV1 {
    pub fn validate(&self) -> Result<(), ActuationTrustPublicationError> {
        if self.device.0.is_empty()
            || self.device.0.len() > MAX_DEVICE_ID_BYTES
            || self.device.0.trim() != self.device.0
            || self.device.0.chars().any(char::is_control)
        {
            return Err(ActuationTrustPublicationError::InvalidDeviceIdentity);
        }
        validate_transport_head(self.transport_trust_head)?;
        validate_device_reality_head(self.device_reality_trust_head)?;
        validate_interlock_head(self.interlock_trust_head)?;
        self.device_reality_policy.validate()?;
        self.interlock_policy.validate()?;
        Ok(())
    }
}

/// Externally retained anti-rollback head for the complete authoritative trust publication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActuationTrustPublicationHead {
    pub generation: u64,
    pub digest: Digest32,
}

/// Crash-durable publication of the complete actuation trust/policy root set.
///
/// Generation, predecessor and publication time are assigned only by the durable store; callers
/// supply only the candidate root set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActuationTrustPublicationV1 {
    schema_version: u16,
    generation: u64,
    previous_publication_digest: Option<Digest32>,
    roots: ActuationTrustRootsV1,
    published_at_unix_ms: u64,
}

impl ActuationTrustPublicationV1 {
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn previous_publication_digest(&self) -> Option<Digest32> {
        self.previous_publication_digest
    }

    pub fn roots(&self) -> &ActuationTrustRootsV1 {
        &self.roots
    }

    pub const fn published_at_unix_ms(&self) -> u64 {
        self.published_at_unix_ms
    }

    pub fn validate(&self) -> Result<(), ActuationTrustPublicationError> {
        if self.schema_version != ACTUATION_TRUST_PUBLICATION_SCHEMA_VERSION {
            return Err(ActuationTrustPublicationError::UnsupportedPublicationSchema);
        }
        if self.generation == 0 {
            return Err(ActuationTrustPublicationError::PublicationGenerationZero);
        }
        if self.generation == 1 && self.previous_publication_digest.is_some() {
            return Err(ActuationTrustPublicationError::GenesisHasPredecessor);
        }
        if self.generation > 1 && self.previous_publication_digest.is_none() {
            return Err(ActuationTrustPublicationError::SuccessorMissingPredecessor);
        }
        if self.published_at_unix_ms == 0 {
            return Err(ActuationTrustPublicationError::PublicationTimeZero);
        }
        self.roots.validate()?;
        Ok(())
    }

    pub fn digest(&self) -> Result<Digest32, ActuationTrustPublicationError> {
        self.validate()?;
        let mut h = blake3::Hasher::new();
        h.update(PUBLICATION_DOMAIN);
        h.update(&self.schema_version.to_be_bytes());
        h.update(&self.generation.to_be_bytes());
        optional_digest(&mut h, self.previous_publication_digest);
        update_string(&mut h, &self.roots.device.0);
        update_transport_head(&mut h, self.roots.transport_trust_head);
        update_device_reality_head(&mut h, self.roots.device_reality_trust_head);
        update_policy_anchor(&mut h, self.roots.device_reality_policy);
        update_interlock_head(&mut h, self.roots.interlock_trust_head);
        update_policy_anchor(&mut h, self.roots.interlock_policy);
        h.update(&self.published_at_unix_ms.to_be_bytes());
        Ok(Digest32(*h.finalize().as_bytes()))
    }

    pub fn head(&self) -> Result<ActuationTrustPublicationHead, ActuationTrustPublicationError> {
        Ok(ActuationTrustPublicationHead {
            generation: self.generation,
            digest: self.digest()?,
        })
    }

    fn genesis(
        roots: ActuationTrustRootsV1,
        published_at_unix_ms: u64,
    ) -> Result<Self, ActuationTrustPublicationError> {
        roots.validate()?;
        let publication = Self {
            schema_version: ACTUATION_TRUST_PUBLICATION_SCHEMA_VERSION,
            generation: 1,
            previous_publication_digest: None,
            roots,
            published_at_unix_ms,
        };
        publication.validate()?;
        Ok(publication)
    }

    fn successor(
        &self,
        roots: ActuationTrustRootsV1,
        published_at_unix_ms: u64,
    ) -> Result<Self, ActuationTrustPublicationError> {
        self.validate()?;
        roots.validate()?;
        validate_root_successor(&self.roots, &roots)?;
        if roots == self.roots {
            return Err(ActuationTrustPublicationError::NoOpSuccessor);
        }
        if published_at_unix_ms < self.published_at_unix_ms {
            return Err(ActuationTrustPublicationError::PublicationTimeRegressed);
        }
        let generation = self
            .generation
            .checked_add(1)
            .ok_or(ActuationTrustPublicationError::PublicationGenerationOverflow)?;
        let publication = Self {
            schema_version: ACTUATION_TRUST_PUBLICATION_SCHEMA_VERSION,
            generation,
            previous_publication_digest: Some(self.digest()?),
            roots,
            published_at_unix_ms,
        };
        publication.validate()?;
        Ok(publication)
    }
}

/// Durable proof that an actuation-trust publication was fsynced and read back exactly.
#[derive(Debug)]
pub struct PersistedActuationTrustPublication {
    publication: ActuationTrustPublicationV1,
    head: ActuationTrustPublicationHead,
    persisted_at_unix_ms: u64,
}

impl PersistedActuationTrustPublication {
    pub fn publication(&self) -> &ActuationTrustPublicationV1 {
        &self.publication
    }

    pub const fn head(&self) -> ActuationTrustPublicationHead {
        self.head
    }

    pub const fn persisted_at_unix_ms(&self) -> u64 {
        self.persisted_at_unix_ms
    }
}

/// Crash-durable single-device trust-publication journal opened against an independently retained
/// current publication head.
pub struct DurableActuationTrustPublicationStore {
    root: PathBuf,
    trusted_current_head: ActuationTrustPublicationHead,
    local_lock: Mutex<()>,
    pinned_root: Mutex<Option<Arc<File>>>,
}

impl DurableActuationTrustPublicationStore {
    /// Create generation one. Existing state is never overwritten.
    pub fn initialize(
        root: impl Into<PathBuf>,
        roots: ActuationTrustRootsV1,
    ) -> Result<PersistedActuationTrustPublication, ActuationTrustPublicationError> {
        roots.validate()?;
        let published_at_unix_ms = system_unix_ms()?;
        let publication = ActuationTrustPublicationV1::genesis(roots, published_at_unix_ms)?;
        let expected_head = publication.head()?;
        let store = Self {
            root: root.into(),
            trusted_current_head: expected_head,
            local_lock: Mutex::new(()),
            pinned_root: Mutex::new(None),
        };
        store.ensure_root()?;
        let _local = store
            .local_lock
            .lock()
            .map_err(|_| ActuationTrustPublicationError::LocalLockPoisoned)?;
        let kernel = store.open_lock_file()?;
        kernel.lock().map_err(ActuationTrustPublicationError::Io)?;
        let state_path = store.operation_root_path()?.join(STATE_FILE_NAME);
        match fs::symlink_metadata(&state_path) {
            Ok(_) => return Err(ActuationTrustPublicationError::AlreadyInitialized),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(ActuationTrustPublicationError::Io(error)),
        }
        store.write_state_locked(&publication)?;
        let persisted = store.read_state_locked()?;
        if persisted != publication || persisted.head()? != expected_head {
            return Err(ActuationTrustPublicationError::PersistedStateReadbackMismatch);
        }
        let persisted_at_unix_ms = system_unix_ms()?;
        if persisted_at_unix_ms < published_at_unix_ms {
            return Err(ActuationTrustPublicationError::SystemClockRegressedDuringPersistence);
        }
        Ok(PersistedActuationTrustPublication {
            publication: persisted,
            head: expected_head,
            persisted_at_unix_ms,
        })
    }

    /// Open existing publication state only when it matches the separately retained anti-rollback
    /// head. The caller cannot replace that head through later fence/publish operations.
    pub fn open(
        root: impl Into<PathBuf>,
        trusted_current_head: ActuationTrustPublicationHead,
    ) -> Result<Self, ActuationTrustPublicationError> {
        if trusted_current_head.generation == 0 || trusted_current_head.digest == Digest32([0; 32]) {
            return Err(ActuationTrustPublicationError::InvalidTrustedHead);
        }
        let store = Self {
            root: root.into(),
            trusted_current_head,
            local_lock: Mutex::new(()),
            pinned_root: Mutex::new(None),
        };
        store.ensure_root()?;
        {
            let _local = store
                .local_lock
                .lock()
                .map_err(|_| ActuationTrustPublicationError::LocalLockPoisoned)?;
            let kernel = store.open_lock_file()?;
            kernel.lock().map_err(ActuationTrustPublicationError::Io)?;
            let current = store.read_state_locked()?;
            store.verify_current(&current)?;
        }
        Ok(store)
    }

    pub const fn trusted_current_head(&self) -> ActuationTrustPublicationHead {
        self.trusted_current_head
    }

    /// Publish one successor atomically. The handle is consumed so the new head must be retained
    /// independently before another publication store can be opened.
    pub fn publish_successor(
        self,
        roots: ActuationTrustRootsV1,
    ) -> Result<PersistedActuationTrustPublication, ActuationTrustPublicationError> {
        roots.validate()?;
        let _local = self
            .local_lock
            .lock()
            .map_err(|_| ActuationTrustPublicationError::LocalLockPoisoned)?;
        let kernel = self.open_lock_file()?;
        kernel.lock().map_err(ActuationTrustPublicationError::Io)?;
        let current = self.read_state_locked()?;
        self.verify_current(&current)?;
        let published_at_unix_ms = system_unix_ms()?;
        let successor = current.successor(roots, published_at_unix_ms)?;
        let expected_head = successor.head()?;
        self.write_state_locked(&successor)?;
        let persisted = self.read_state_locked()?;
        if persisted != successor || persisted.head()? != expected_head {
            return Err(ActuationTrustPublicationError::PersistedStateReadbackMismatch);
        }
        let persisted_at_unix_ms = system_unix_ms()?;
        if persisted_at_unix_ms < published_at_unix_ms {
            return Err(ActuationTrustPublicationError::SystemClockRegressedDuringPersistence);
        }
        Ok(PersistedActuationTrustPublication {
            publication: persisted,
            head: expected_head,
            persisted_at_unix_ms,
        })
    }

    /// Hold the authoritative trust publication stable across a later multi-root actuation check.
    ///
    /// The returned fence owns the kernel lock file and borrows the store-local mutex guard for its
    /// lifetime. A concurrent publisher cannot make a successor authoritative until this value is
    /// dropped.
    pub fn fence_current(
        &self,
    ) -> Result<CurrentActuationTrustFence<'_>, ActuationTrustPublicationError> {
        let local = self
            .local_lock
            .lock()
            .map_err(|_| ActuationTrustPublicationError::LocalLockPoisoned)?;
        let kernel = self.open_lock_file()?;
        kernel.lock().map_err(ActuationTrustPublicationError::Io)?;
        let publication = self.read_state_locked()?;
        self.verify_current(&publication)?;
        let head = publication.head()?;
        Ok(CurrentActuationTrustFence {
            _local: local,
            _kernel: kernel,
            publication,
            head,
        })
    }

    fn verify_current(
        &self,
        publication: &ActuationTrustPublicationV1,
    ) -> Result<(), ActuationTrustPublicationError> {
        publication.validate()?;
        if publication.head()? != self.trusted_current_head {
            return Err(ActuationTrustPublicationError::TrustedHeadMismatch);
        }
        Ok(())
    }

    fn read_state_locked(
        &self,
    ) -> Result<ActuationTrustPublicationV1, ActuationTrustPublicationError> {
        let path = self.operation_root_path()?.join(STATE_FILE_NAME);
        let file = match open_regular_file(&path, false, false) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(ActuationTrustPublicationError::Uninitialized);
            }
            Err(error) => return Err(ActuationTrustPublicationError::Io(error)),
        };
        let metadata = file.metadata().map_err(ActuationTrustPublicationError::Io)?;
        if metadata.len() == 0 || metadata.len() > MAX_ACTUATION_TRUST_STATE_BYTES {
            return Err(ActuationTrustPublicationError::StateSizeOutOfBounds);
        }
        let mut bytes = Vec::with_capacity(metadata.len() as usize);
        file.take(MAX_ACTUATION_TRUST_STATE_BYTES.saturating_add(1))
            .read_to_end(&mut bytes)
            .map_err(ActuationTrustPublicationError::Io)?;
        if bytes.is_empty() || bytes.len() as u64 > MAX_ACTUATION_TRUST_STATE_BYTES {
            return Err(ActuationTrustPublicationError::StateSizeOutOfBounds);
        }
        let publication: ActuationTrustPublicationV1 = bincode::deserialize(&bytes)
            .map_err(|_| ActuationTrustPublicationError::StateEncoding)?;
        publication.validate()?;
        let canonical = bincode::serialize(&publication)
            .map_err(|_| ActuationTrustPublicationError::StateEncoding)?;
        if canonical != bytes {
            return Err(ActuationTrustPublicationError::NonCanonicalStateEncoding);
        }
        Ok(publication)
    }

    fn write_state_locked(
        &self,
        publication: &ActuationTrustPublicationV1,
    ) -> Result<(), ActuationTrustPublicationError> {
        publication.validate()?;
        let encoded = bincode::serialize(publication)
            .map_err(|_| ActuationTrustPublicationError::StateEncoding)?;
        if encoded.is_empty() || encoded.len() as u64 > MAX_ACTUATION_TRUST_STATE_BYTES {
            return Err(ActuationTrustPublicationError::StateSizeOutOfBounds);
        }
        let root = self.ensure_root()?;
        let operation_root = self.operation_root_path()?;
        let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| ActuationTrustPublicationError::SystemClockBeforeUnixEpoch)?
            .as_nanos();
        let temp = operation_root.join(format!(
            ".actuation-trust-publication-{}-{counter}-{nanos}.tmp",
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
        result.map_err(ActuationTrustPublicationError::Io)
    }

    fn open_lock_file(&self) -> Result<File, ActuationTrustPublicationError> {
        let path = self.operation_root_path()?.join(LOCK_FILE_NAME);
        open_regular_file(&path, true, false).map_err(ActuationTrustPublicationError::Io)
    }

    fn ensure_root(&self) -> Result<Arc<File>, ActuationTrustPublicationError> {
        let mut pinned = self
            .pinned_root
            .lock()
            .map_err(|_| ActuationTrustPublicationError::RootLockPoisoned)?;
        if let Some(root) = pinned.as_ref() {
            return Ok(Arc::clone(root));
        }
        match fs::symlink_metadata(&self.root) {
            Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
                return Err(ActuationTrustPublicationError::InvalidRootDirectory);
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                fs::create_dir_all(&self.root).map_err(ActuationTrustPublicationError::Io)?;
            }
            Err(error) => return Err(ActuationTrustPublicationError::Io(error)),
        }
        fs::set_permissions(&self.root, fs::Permissions::from_mode(0o700))
            .map_err(ActuationTrustPublicationError::Io)?;
        let mut options = OpenOptions::new();
        options
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW);
        let root = Arc::new(options.open(&self.root).map_err(ActuationTrustPublicationError::Io)?);
        if !root
            .metadata()
            .map_err(ActuationTrustPublicationError::Io)?
            .is_dir()
        {
            return Err(ActuationTrustPublicationError::InvalidRootDirectory);
        }
        *pinned = Some(Arc::clone(&root));
        Ok(root)
    }

    fn operation_root_path(&self) -> Result<PathBuf, ActuationTrustPublicationError> {
        let root = self.ensure_root()?;
        let path = PathBuf::from(format!("/proc/self/fd/{}", root.as_raw_fd()));
        if !path.is_dir() {
            return Err(ActuationTrustPublicationError::PinnedRootUnavailable);
        }
        Ok(path)
    }
}

/// Borrowed current publication with the cross-process publication lock held.
#[derive(Debug)]
pub struct CurrentActuationTrustFence<'a> {
    _local: MutexGuard<'a, ()>,
    _kernel: File,
    publication: ActuationTrustPublicationV1,
    head: ActuationTrustPublicationHead,
}

impl CurrentActuationTrustFence<'_> {
    pub fn publication(&self) -> &ActuationTrustPublicationV1 {
        &self.publication
    }

    pub fn roots(&self) -> &ActuationTrustRootsV1 {
        self.publication.roots()
    }

    pub const fn head(&self) -> ActuationTrustPublicationHead {
        self.head
    }
}

fn validate_root_successor(
    previous: &ActuationTrustRootsV1,
    next: &ActuationTrustRootsV1,
) -> Result<(), ActuationTrustPublicationError> {
    if previous.device != next.device {
        return Err(ActuationTrustPublicationError::DeviceChanged);
    }
    validate_transport_transition(previous.transport_trust_head, next.transport_trust_head)?;
    validate_device_reality_transition(
        previous.device_reality_trust_head,
        next.device_reality_trust_head,
    )?;
    validate_interlock_transition(previous.interlock_trust_head, next.interlock_trust_head)?;
    validate_policy_transition(previous.device_reality_policy, next.device_reality_policy)?;
    validate_policy_transition(previous.interlock_policy, next.interlock_policy)?;
    Ok(())
}

fn validate_transport_transition(
    previous: TransportTrustHead,
    next: TransportTrustHead,
) -> Result<(), ActuationTrustPublicationError> {
    if next.sequence < previous.sequence {
        return Err(ActuationTrustPublicationError::TransportTrustRollback);
    }
    if next.sequence == previous.sequence && next.digest != previous.digest {
        return Err(ActuationTrustPublicationError::TransportTrustSameGenerationMutation);
    }
    if next.sequence > previous.sequence && next.digest == previous.digest {
        return Err(ActuationTrustPublicationError::TransportTrustGenerationWithoutDigestChange);
    }
    Ok(())
}

fn validate_device_reality_transition(
    previous: DeviceRealityTrustHead,
    next: DeviceRealityTrustHead,
) -> Result<(), ActuationTrustPublicationError> {
    if next.sequence < previous.sequence {
        return Err(ActuationTrustPublicationError::DeviceRealityTrustRollback);
    }
    if next.sequence == previous.sequence && next.digest != previous.digest {
        return Err(ActuationTrustPublicationError::DeviceRealityTrustSameGenerationMutation);
    }
    if next.sequence > previous.sequence && next.digest == previous.digest {
        return Err(ActuationTrustPublicationError::DeviceRealityTrustGenerationWithoutDigestChange);
    }
    Ok(())
}

fn validate_interlock_transition(
    previous: InterlockTrustHead,
    next: InterlockTrustHead,
) -> Result<(), ActuationTrustPublicationError> {
    if next.sequence < previous.sequence {
        return Err(ActuationTrustPublicationError::InterlockTrustRollback);
    }
    if next.sequence == previous.sequence && next.digest != previous.digest {
        return Err(ActuationTrustPublicationError::InterlockTrustSameGenerationMutation);
    }
    if next.sequence > previous.sequence && next.digest == previous.digest {
        return Err(ActuationTrustPublicationError::InterlockTrustGenerationWithoutDigestChange);
    }
    Ok(())
}

fn validate_policy_transition(
    previous: ActuationPolicyAnchorV1,
    next: ActuationPolicyAnchorV1,
) -> Result<(), ActuationTrustPublicationError> {
    if next.generation < previous.generation {
        return Err(ActuationTrustPublicationError::PolicyRollback);
    }
    if next.generation == previous.generation && next.digest != previous.digest {
        return Err(ActuationTrustPublicationError::PolicySameGenerationMutation);
    }
    if next.generation > previous.generation && next.digest == previous.digest {
        return Err(ActuationTrustPublicationError::PolicyGenerationWithoutDigestChange);
    }
    Ok(())
}

fn validate_transport_head(head: TransportTrustHead) -> Result<(), ActuationTrustPublicationError> {
    if head.sequence == 0 || head.digest == Digest32([0; 32]) {
        return Err(ActuationTrustPublicationError::InvalidTransportTrustHead);
    }
    Ok(())
}

fn validate_device_reality_head(
    head: DeviceRealityTrustHead,
) -> Result<(), ActuationTrustPublicationError> {
    if head.sequence == 0 || head.digest == Digest32([0; 32]) {
        return Err(ActuationTrustPublicationError::InvalidDeviceRealityTrustHead);
    }
    Ok(())
}

fn validate_interlock_head(head: InterlockTrustHead) -> Result<(), ActuationTrustPublicationError> {
    if head.sequence == 0 || head.digest == Digest32([0; 32]) {
        return Err(ActuationTrustPublicationError::InvalidInterlockTrustHead);
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
            "actuation trust publication object is not a regular file",
        ));
    }
    Ok(file)
}

fn system_unix_ms() -> Result<u64, ActuationTrustPublicationError> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| ActuationTrustPublicationError::SystemClockBeforeUnixEpoch)?;
    u64::try_from(elapsed.as_millis()).map_err(|_| ActuationTrustPublicationError::TimeOverflow)
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

fn update_transport_head(h: &mut blake3::Hasher, head: TransportTrustHead) {
    h.update(&head.sequence.to_be_bytes());
    update_digest(h, head.digest);
}

fn update_device_reality_head(h: &mut blake3::Hasher, head: DeviceRealityTrustHead) {
    h.update(&head.sequence.to_be_bytes());
    update_digest(h, head.digest);
}

fn update_interlock_head(h: &mut blake3::Hasher, head: InterlockTrustHead) {
    h.update(&head.sequence.to_be_bytes());
    update_digest(h, head.digest);
}

fn update_policy_anchor(h: &mut blake3::Hasher, anchor: ActuationPolicyAnchorV1) {
    h.update(&anchor.generation.to_be_bytes());
    update_digest(h, anchor.digest);
}

#[derive(Debug, Error)]
pub enum ActuationTrustPublicationError {
    #[error("unsupported actuation trust publication schema")]
    UnsupportedPublicationSchema,
    #[error("actuation trust publication generation is zero")]
    PublicationGenerationZero,
    #[error("actuation trust publication generation overflow")]
    PublicationGenerationOverflow,
    #[error("actuation trust publication genesis unexpectedly has a predecessor")]
    GenesisHasPredecessor,
    #[error("actuation trust publication successor is missing a predecessor")]
    SuccessorMissingPredecessor,
    #[error("actuation trust publication time is zero")]
    PublicationTimeZero,
    #[error("actuation trust publication time regressed")]
    PublicationTimeRegressed,
    #[error("actuation trust policy generation is zero")]
    PolicyGenerationZero,
    #[error("actuation trust policy digest is zero")]
    PolicyDigestZero,
    #[error("actuation trust publication device identity is invalid")]
    InvalidDeviceIdentity,
    #[error("transport trust head is invalid")]
    InvalidTransportTrustHead,
    #[error("device-reality trust head is invalid")]
    InvalidDeviceRealityTrustHead,
    #[error("interlock trust head is invalid")]
    InvalidInterlockTrustHead,
    #[error("actuation trust publication changed device")]
    DeviceChanged,
    #[error("transport trust generation rolled back")]
    TransportTrustRollback,
    #[error("transport trust digest changed without advancing its generation")]
    TransportTrustSameGenerationMutation,
    #[error("transport trust generation advanced without changing its digest")]
    TransportTrustGenerationWithoutDigestChange,
    #[error("device-reality trust generation rolled back")]
    DeviceRealityTrustRollback,
    #[error("device-reality trust digest changed without advancing its generation")]
    DeviceRealityTrustSameGenerationMutation,
    #[error("device-reality trust generation advanced without changing its digest")]
    DeviceRealityTrustGenerationWithoutDigestChange,
    #[error("interlock trust generation rolled back")]
    InterlockTrustRollback,
    #[error("interlock trust digest changed without advancing its generation")]
    InterlockTrustSameGenerationMutation,
    #[error("interlock trust generation advanced without changing its digest")]
    InterlockTrustGenerationWithoutDigestChange,
    #[error("actuation policy generation rolled back")]
    PolicyRollback,
    #[error("actuation policy digest changed without advancing its generation")]
    PolicySameGenerationMutation,
    #[error("actuation policy generation advanced without changing its digest")]
    PolicyGenerationWithoutDigestChange,
    #[error("actuation trust successor changes no authoritative root")]
    NoOpSuccessor,
    #[error("actuation trust publication store is already initialized")]
    AlreadyInitialized,
    #[error("actuation trust publication store is uninitialized")]
    Uninitialized,
    #[error("trusted actuation trust publication head is invalid")]
    InvalidTrustedHead,
    #[error("persisted actuation trust publication differs from independently retained head")]
    TrustedHeadMismatch,
    #[error("actuation trust state size is outside accepted bounds")]
    StateSizeOutOfBounds,
    #[error("actuation trust state encoding/decoding failed")]
    StateEncoding,
    #[error("actuation trust state is not canonically encoded")]
    NonCanonicalStateEncoding,
    #[error("persisted actuation trust publication did not read back exactly")]
    PersistedStateReadbackMismatch,
    #[error("actuation trust publication root is not a real directory")]
    InvalidRootDirectory,
    #[error("pinned descriptor-relative actuation trust root is unavailable")]
    PinnedRootUnavailable,
    #[error("actuation trust publication local lock is poisoned")]
    LocalLockPoisoned,
    #[error("actuation trust publication root lock is poisoned")]
    RootLockPoisoned,
    #[error("system clock predates Unix epoch")]
    SystemClockBeforeUnixEpoch,
    #[error("system clock regressed during persistence")]
    SystemClockRegressedDuringPersistence,
    #[error("time conversion overflow")]
    TimeOverflow,
    #[error("actuation trust publication I/O failed: {0}")]
    Io(#[source] std::io::Error),
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    use super::*;

    fn d(byte: u8) -> Digest32 {
        Digest32([byte; 32])
    }

    fn temp_root(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "symthaea-actuation-trust-{label}-{}-{nanos}",
            std::process::id()
        ))
    }

    fn roots() -> ActuationTrustRootsV1 {
        ActuationTrustRootsV1 {
            device: ResourceRef("iot:valve:72".into()),
            transport_trust_head: TransportTrustHead {
                sequence: 1,
                digest: d(1),
            },
            device_reality_trust_head: DeviceRealityTrustHead {
                sequence: 1,
                digest: d(2),
            },
            device_reality_policy: ActuationPolicyAnchorV1 {
                generation: 2,
                digest: d(3),
            },
            interlock_trust_head: InterlockTrustHead {
                sequence: 1,
                digest: d(4),
            },
            interlock_policy: ActuationPolicyAnchorV1 {
                generation: 2,
                digest: d(5),
            },
        }
    }

    #[test]
    fn current_fence_blocks_authoritative_successor_publication() {
        let root = temp_root("fence-blocks-publish");
        let initial = DurableActuationTrustPublicationStore::initialize(&root, roots()).unwrap();
        let head = initial.head();
        let fence_store = DurableActuationTrustPublicationStore::open(&root, head).unwrap();
        let mutation_store = DurableActuationTrustPublicationStore::open(&root, head).unwrap();
        let fence = fence_store.fence_current().unwrap();
        assert_eq!(fence.head(), head);
        assert_eq!(fence.roots(), &roots());

        let mut next = roots();
        next.transport_trust_head = TransportTrustHead {
            sequence: 2,
            digest: d(6),
        };
        let (started_tx, started_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let worker = thread::spawn(move || {
            started_tx.send(()).unwrap();
            let result = mutation_store.publish_successor(next).map(|proof| proof.head());
            done_tx.send(result).unwrap();
        });
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(done_rx.recv_timeout(Duration::from_millis(150)).is_err());

        drop(fence);
        let successor_head = done_rx
            .recv_timeout(Duration::from_secs(2))
            .unwrap()
            .unwrap();
        assert_eq!(successor_head.generation, 2);
        worker.join().unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn same_generation_trust_substitution_is_rejected() {
        let root = temp_root("same-generation-substitution");
        let initial = DurableActuationTrustPublicationStore::initialize(&root, roots()).unwrap();
        let mut next = roots();
        next.transport_trust_head.digest = d(9);
        let store = DurableActuationTrustPublicationStore::open(&root, initial.head()).unwrap();
        assert!(matches!(
            store.publish_successor(next),
            Err(ActuationTrustPublicationError::TransportTrustSameGenerationMutation)
        ));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn policy_rollback_and_same_generation_substitution_are_rejected() {
        let rollback_root = temp_root("policy-rollback");
        let initial =
            DurableActuationTrustPublicationStore::initialize(&rollback_root, roots()).unwrap();
        let mut rollback = roots();
        rollback.device_reality_policy = ActuationPolicyAnchorV1 {
            generation: 1,
            digest: d(7),
        };
        let store =
            DurableActuationTrustPublicationStore::open(&rollback_root, initial.head()).unwrap();
        assert!(matches!(
            store.publish_successor(rollback),
            Err(ActuationTrustPublicationError::PolicyRollback)
        ));
        std::fs::remove_dir_all(rollback_root).unwrap();

        let mutation_root = temp_root("policy-substitution");
        let initial =
            DurableActuationTrustPublicationStore::initialize(&mutation_root, roots()).unwrap();
        let mut mutation = roots();
        mutation.interlock_policy.digest = d(8);
        let store =
            DurableActuationTrustPublicationStore::open(&mutation_root, initial.head()).unwrap();
        assert!(matches!(
            store.publish_successor(mutation),
            Err(ActuationTrustPublicationError::PolicySameGenerationMutation)
        ));
        std::fs::remove_dir_all(mutation_root).unwrap();
    }

    #[test]
    fn no_op_successor_is_rejected() {
        let root = temp_root("no-op");
        let initial = DurableActuationTrustPublicationStore::initialize(&root, roots()).unwrap();
        let store = DurableActuationTrustPublicationStore::open(&root, initial.head()).unwrap();
        assert!(matches!(
            store.publish_successor(roots()),
            Err(ActuationTrustPublicationError::NoOpSuccessor)
        ));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn persisted_state_requires_exact_external_head() {
        let root = temp_root("external-head");
        let initial = DurableActuationTrustPublicationStore::initialize(&root, roots()).unwrap();
        let wrong = ActuationTrustPublicationHead {
            generation: initial.head().generation,
            digest: d(0xFE),
        };
        assert!(matches!(
            DurableActuationTrustPublicationStore::open(&root, wrong),
            Err(ActuationTrustPublicationError::TrustedHeadMismatch)
        ));
        std::fs::remove_dir_all(root).unwrap();
    }
}
