// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Crash-durable atomic publication of the policy/trust roots current for effect reconciliation.
//!
//! Outcome-verifier policy and trust may be prepared independently, but terminal reconciliation
//! needs one authoritative answer to which combination is current. Publishing a successor and
//! fencing the current publication use the same cross-process kernel lock, so a later journal closer
//! can hold [`CurrentEffectReconciliationTrustFence`] while it constructs the owner-local current
//! outcome-verifier fence and mutates the unresolved effect-attempt journal.
//!
//! This crate contains no outcome cryptographic verification, physical-effect dispatch, terminal
//! journal transition, actuator permit or HAL/device I/O.

#![deny(unsafe_code)]

#[cfg(not(target_os = "linux"))]
compile_error!("the privileged effect-reconciliation trust publication store is Linux-only");

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
use symthaea_iot_actuation_effect_outcome_verifier::EffectOutcomeTrustHead;
use thiserror::Error;

pub const EFFECT_RECONCILIATION_TRUST_PUBLICATION_SCHEMA_VERSION: u16 = 1;
pub const MAX_EFFECT_RECONCILIATION_TRUST_STATE_BYTES: u64 = 64 * 1024;

const PUBLICATION_DOMAIN: &[u8] = b"symthaea-iot-effect-reconciliation-trust-publication-v1\0";
const STATE_FILE_NAME: &str = "effect-reconciliation-trust-publication.state";
const LOCK_FILE_NAME: &str = ".effect-reconciliation-trust-publication.lock";
const MAX_DEVICE_ID_BYTES: usize = 512;
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Independently retained generation/digest of one immutable outcome-verifier policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectReconciliationPolicyAnchorV1 {
    pub generation: u64,
    pub digest: Digest32,
}

impl EffectReconciliationPolicyAnchorV1 {
    pub fn validate(&self) -> Result<(), EffectReconciliationTrustPublicationError> {
        if self.generation == 0 {
            return Err(EffectReconciliationTrustPublicationError::PolicyGenerationZero);
        }
        if self.digest == Digest32([0; 32]) {
            return Err(EffectReconciliationTrustPublicationError::PolicyDigestZero);
        }
        Ok(())
    }
}

/// Persistence-only mirror of the owner-local outcome-verifier trust head.
///
/// The cryptographic verifier crate deliberately remains serialization-free. This type stores only
/// the exact sequence/digest necessary to correlate a held publication back to its owner-local
/// `EffectOutcomeTrustHead`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublishedEffectOutcomeTrustHeadV1 {
    pub sequence: u64,
    pub digest: Digest32,
}

impl PublishedEffectOutcomeTrustHeadV1 {
    pub const fn from_outcome_trust_head(head: EffectOutcomeTrustHead) -> Self {
        Self {
            sequence: head.sequence,
            digest: head.digest,
        }
    }

    pub const fn matches_outcome_trust_head(self, head: EffectOutcomeTrustHead) -> bool {
        self.sequence == head.sequence && self.digest == head.digest
    }

    pub fn validate(&self) -> Result<(), EffectReconciliationTrustPublicationError> {
        if self.sequence == 0 || self.digest == Digest32([0; 32]) {
            return Err(EffectReconciliationTrustPublicationError::InvalidOutcomeTrustHead);
        }
        Ok(())
    }
}

/// Exact device-scoped policy/trust combination authoritative for terminal reconciliation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectReconciliationTrustRootsV1 {
    pub device: ResourceRef,
    pub outcome_policy: EffectReconciliationPolicyAnchorV1,
    pub outcome_trust_head: PublishedEffectOutcomeTrustHeadV1,
}

impl EffectReconciliationTrustRootsV1 {
    pub fn validate(&self) -> Result<(), EffectReconciliationTrustPublicationError> {
        if self.device.0.is_empty()
            || self.device.0.len() > MAX_DEVICE_ID_BYTES
            || self.device.0.trim() != self.device.0
            || self.device.0.chars().any(char::is_control)
        {
            return Err(EffectReconciliationTrustPublicationError::InvalidDeviceIdentity);
        }
        self.outcome_policy.validate()?;
        self.outcome_trust_head.validate()?;
        Ok(())
    }
}

/// Independently retained anti-rollback head for the authoritative reconciliation publication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectReconciliationTrustPublicationHead {
    pub generation: u64,
    pub digest: Digest32,
}

/// Crash-durable publication of the complete reconciliation policy/trust root pair.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectReconciliationTrustPublicationV1 {
    schema_version: u16,
    generation: u64,
    previous_publication_digest: Option<Digest32>,
    roots: EffectReconciliationTrustRootsV1,
    published_at_unix_ms: u64,
}

impl EffectReconciliationTrustPublicationV1 {
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn previous_publication_digest(&self) -> Option<Digest32> {
        self.previous_publication_digest
    }

    pub fn roots(&self) -> &EffectReconciliationTrustRootsV1 {
        &self.roots
    }

    pub const fn published_at_unix_ms(&self) -> u64 {
        self.published_at_unix_ms
    }

    pub fn validate(&self) -> Result<(), EffectReconciliationTrustPublicationError> {
        if self.schema_version != EFFECT_RECONCILIATION_TRUST_PUBLICATION_SCHEMA_VERSION {
            return Err(EffectReconciliationTrustPublicationError::UnsupportedPublicationSchema);
        }
        if self.generation == 0 {
            return Err(EffectReconciliationTrustPublicationError::PublicationGenerationZero);
        }
        if self.generation == 1 && self.previous_publication_digest.is_some() {
            return Err(EffectReconciliationTrustPublicationError::GenesisHasPredecessor);
        }
        if self.generation > 1 && self.previous_publication_digest.is_none() {
            return Err(EffectReconciliationTrustPublicationError::SuccessorMissingPredecessor);
        }
        if self.published_at_unix_ms == 0 {
            return Err(EffectReconciliationTrustPublicationError::PublicationTimeZero);
        }
        self.roots.validate()?;
        Ok(())
    }

    pub fn digest(&self) -> Result<Digest32, EffectReconciliationTrustPublicationError> {
        self.validate()?;
        let mut h = blake3::Hasher::new();
        h.update(PUBLICATION_DOMAIN);
        h.update(&self.schema_version.to_be_bytes());
        h.update(&self.generation.to_be_bytes());
        optional_digest(&mut h, self.previous_publication_digest);
        update_string(&mut h, &self.roots.device.0);
        update_policy_anchor(&mut h, self.roots.outcome_policy);
        update_outcome_trust_head(&mut h, self.roots.outcome_trust_head);
        h.update(&self.published_at_unix_ms.to_be_bytes());
        Ok(Digest32(*h.finalize().as_bytes()))
    }

    pub fn head(
        &self,
    ) -> Result<EffectReconciliationTrustPublicationHead, EffectReconciliationTrustPublicationError>
    {
        Ok(EffectReconciliationTrustPublicationHead {
            generation: self.generation,
            digest: self.digest()?,
        })
    }

    fn genesis(
        roots: EffectReconciliationTrustRootsV1,
        published_at_unix_ms: u64,
    ) -> Result<Self, EffectReconciliationTrustPublicationError> {
        roots.validate()?;
        let publication = Self {
            schema_version: EFFECT_RECONCILIATION_TRUST_PUBLICATION_SCHEMA_VERSION,
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
        roots: EffectReconciliationTrustRootsV1,
        published_at_unix_ms: u64,
    ) -> Result<Self, EffectReconciliationTrustPublicationError> {
        self.validate()?;
        roots.validate()?;
        validate_root_successor(&self.roots, &roots)?;
        if roots == self.roots {
            return Err(EffectReconciliationTrustPublicationError::NoOpSuccessor);
        }
        if published_at_unix_ms < self.published_at_unix_ms {
            return Err(EffectReconciliationTrustPublicationError::PublicationTimeRegressed);
        }
        let generation = self
            .generation
            .checked_add(1)
            .ok_or(EffectReconciliationTrustPublicationError::PublicationGenerationOverflow)?;
        let publication = Self {
            schema_version: EFFECT_RECONCILIATION_TRUST_PUBLICATION_SCHEMA_VERSION,
            generation,
            previous_publication_digest: Some(self.digest()?),
            roots,
            published_at_unix_ms,
        };
        publication.validate()?;
        Ok(publication)
    }
}

/// Durable proof that one publication was fsynced and read back exactly.
#[derive(Debug)]
pub struct PersistedEffectReconciliationTrustPublication {
    publication: EffectReconciliationTrustPublicationV1,
    head: EffectReconciliationTrustPublicationHead,
    persisted_at_unix_ms: u64,
}

impl PersistedEffectReconciliationTrustPublication {
    pub fn publication(&self) -> &EffectReconciliationTrustPublicationV1 {
        &self.publication
    }

    pub const fn head(&self) -> EffectReconciliationTrustPublicationHead {
        self.head
    }

    pub const fn persisted_at_unix_ms(&self) -> u64 {
        self.persisted_at_unix_ms
    }
}

/// Crash-durable single-device publication store opened against an independently retained head.
pub struct DurableEffectReconciliationTrustPublicationStore {
    root: PathBuf,
    trusted_current_head: EffectReconciliationTrustPublicationHead,
    local_lock: Mutex<()>,
    pinned_root: Mutex<Option<Arc<File>>>,
}

impl DurableEffectReconciliationTrustPublicationStore {
    /// Create generation one. Existing state is never overwritten.
    pub fn initialize(
        root: impl Into<PathBuf>,
        roots: EffectReconciliationTrustRootsV1,
    ) -> Result<PersistedEffectReconciliationTrustPublication, EffectReconciliationTrustPublicationError>
    {
        roots.validate()?;
        let published_at_unix_ms = system_unix_ms()?;
        let publication = EffectReconciliationTrustPublicationV1::genesis(
            roots,
            published_at_unix_ms,
        )?;
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
            .map_err(|_| EffectReconciliationTrustPublicationError::LocalLockPoisoned)?;
        let kernel = store.open_lock_file()?;
        kernel
            .lock()
            .map_err(EffectReconciliationTrustPublicationError::Io)?;
        let state_path = store.operation_root_path()?.join(STATE_FILE_NAME);
        match fs::symlink_metadata(&state_path) {
            Ok(_) => return Err(EffectReconciliationTrustPublicationError::AlreadyInitialized),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(EffectReconciliationTrustPublicationError::Io(error)),
        }
        store.write_state_locked(&publication)?;
        let persisted = store.read_state_locked()?;
        if persisted != publication || persisted.head()? != expected_head {
            return Err(
                EffectReconciliationTrustPublicationError::PersistedStateReadbackMismatch,
            );
        }
        let persisted_at_unix_ms = system_unix_ms()?;
        if persisted_at_unix_ms < published_at_unix_ms {
            return Err(
                EffectReconciliationTrustPublicationError::SystemClockRegressedDuringPersistence,
            );
        }
        Ok(PersistedEffectReconciliationTrustPublication {
            publication: persisted,
            head: expected_head,
            persisted_at_unix_ms,
        })
    }

    /// Open only if the serialized publication matches the separately retained anti-rollback head.
    pub fn open(
        root: impl Into<PathBuf>,
        trusted_current_head: EffectReconciliationTrustPublicationHead,
    ) -> Result<Self, EffectReconciliationTrustPublicationError> {
        if trusted_current_head.generation == 0 || trusted_current_head.digest == Digest32([0; 32]) {
            return Err(EffectReconciliationTrustPublicationError::InvalidTrustedHead);
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
                .map_err(|_| EffectReconciliationTrustPublicationError::LocalLockPoisoned)?;
            let kernel = store.open_lock_file()?;
            kernel
                .lock()
                .map_err(EffectReconciliationTrustPublicationError::Io)?;
            let current = store.read_state_locked()?;
            store.verify_current(&current)?;
        }
        Ok(store)
    }

    pub const fn trusted_current_head(&self) -> EffectReconciliationTrustPublicationHead {
        self.trusted_current_head
    }

    /// Publish one successor. The handle is consumed so the new head must be retained externally
    /// before another authoritative store can be opened.
    pub fn publish_successor(
        self,
        roots: EffectReconciliationTrustRootsV1,
    ) -> Result<PersistedEffectReconciliationTrustPublication, EffectReconciliationTrustPublicationError>
    {
        roots.validate()?;
        let _local = self
            .local_lock
            .lock()
            .map_err(|_| EffectReconciliationTrustPublicationError::LocalLockPoisoned)?;
        let kernel = self.open_lock_file()?;
        kernel
            .lock()
            .map_err(EffectReconciliationTrustPublicationError::Io)?;
        let current = self.read_state_locked()?;
        self.verify_current(&current)?;
        let published_at_unix_ms = system_unix_ms()?;
        let successor = current.successor(roots, published_at_unix_ms)?;
        let expected_head = successor.head()?;
        self.write_state_locked(&successor)?;
        let persisted = self.read_state_locked()?;
        if persisted != successor || persisted.head()? != expected_head {
            return Err(
                EffectReconciliationTrustPublicationError::PersistedStateReadbackMismatch,
            );
        }
        let persisted_at_unix_ms = system_unix_ms()?;
        if persisted_at_unix_ms < published_at_unix_ms {
            return Err(
                EffectReconciliationTrustPublicationError::SystemClockRegressedDuringPersistence,
            );
        }
        Ok(PersistedEffectReconciliationTrustPublication {
            publication: persisted,
            head: expected_head,
            persisted_at_unix_ms,
        })
    }

    /// Hold the authoritative reconciliation trust publication stable for terminal reconciliation.
    ///
    /// The same kernel lock is held by `publish_successor`, so another process cannot make a newer
    /// policy/trust pair authoritative until this fence is dropped.
    pub fn fence_current(
        &self,
    ) -> Result<CurrentEffectReconciliationTrustFence<'_>, EffectReconciliationTrustPublicationError>
    {
        let local = self
            .local_lock
            .lock()
            .map_err(|_| EffectReconciliationTrustPublicationError::LocalLockPoisoned)?;
        let kernel = self.open_lock_file()?;
        kernel
            .lock()
            .map_err(EffectReconciliationTrustPublicationError::Io)?;
        let publication = self.read_state_locked()?;
        self.verify_current(&publication)?;
        let head = publication.head()?;
        Ok(CurrentEffectReconciliationTrustFence {
            _local: local,
            _kernel: kernel,
            publication,
            head,
        })
    }

    fn verify_current(
        &self,
        publication: &EffectReconciliationTrustPublicationV1,
    ) -> Result<(), EffectReconciliationTrustPublicationError> {
        publication.validate()?;
        if publication.head()? != self.trusted_current_head {
            return Err(EffectReconciliationTrustPublicationError::TrustedHeadMismatch);
        }
        Ok(())
    }

    fn read_state_locked(
        &self,
    ) -> Result<EffectReconciliationTrustPublicationV1, EffectReconciliationTrustPublicationError>
    {
        let path = self.operation_root_path()?.join(STATE_FILE_NAME);
        let file = match open_regular_file(&path, false, false) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(EffectReconciliationTrustPublicationError::Uninitialized);
            }
            Err(error) => return Err(EffectReconciliationTrustPublicationError::Io(error)),
        };
        let metadata = file
            .metadata()
            .map_err(EffectReconciliationTrustPublicationError::Io)?;
        if metadata.len() == 0 || metadata.len() > MAX_EFFECT_RECONCILIATION_TRUST_STATE_BYTES {
            return Err(EffectReconciliationTrustPublicationError::StateSizeOutOfBounds);
        }
        let mut bytes = Vec::with_capacity(metadata.len() as usize);
        file.take(MAX_EFFECT_RECONCILIATION_TRUST_STATE_BYTES.saturating_add(1))
            .read_to_end(&mut bytes)
            .map_err(EffectReconciliationTrustPublicationError::Io)?;
        if bytes.is_empty() || bytes.len() as u64 > MAX_EFFECT_RECONCILIATION_TRUST_STATE_BYTES {
            return Err(EffectReconciliationTrustPublicationError::StateSizeOutOfBounds);
        }
        let publication: EffectReconciliationTrustPublicationV1 = bincode::deserialize(&bytes)
            .map_err(|_| EffectReconciliationTrustPublicationError::StateEncoding)?;
        publication.validate()?;
        let canonical = bincode::serialize(&publication)
            .map_err(|_| EffectReconciliationTrustPublicationError::StateEncoding)?;
        if canonical != bytes {
            return Err(EffectReconciliationTrustPublicationError::NonCanonicalStateEncoding);
        }
        Ok(publication)
    }

    fn write_state_locked(
        &self,
        publication: &EffectReconciliationTrustPublicationV1,
    ) -> Result<(), EffectReconciliationTrustPublicationError> {
        publication.validate()?;
        let encoded = bincode::serialize(publication)
            .map_err(|_| EffectReconciliationTrustPublicationError::StateEncoding)?;
        if encoded.is_empty() || encoded.len() as u64 > MAX_EFFECT_RECONCILIATION_TRUST_STATE_BYTES {
            return Err(EffectReconciliationTrustPublicationError::StateSizeOutOfBounds);
        }
        let root = self.ensure_root()?;
        let operation_root = self.operation_root_path()?;
        let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| EffectReconciliationTrustPublicationError::SystemClockBeforeUnixEpoch)?
            .as_nanos();
        let temp = operation_root.join(format!(
            ".effect-reconciliation-trust-publication-{}-{counter}-{nanos}.tmp",
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
        result.map_err(EffectReconciliationTrustPublicationError::Io)
    }

    fn open_lock_file(&self) -> Result<File, EffectReconciliationTrustPublicationError> {
        let path = self.operation_root_path()?.join(LOCK_FILE_NAME);
        open_regular_file(&path, true, false).map_err(EffectReconciliationTrustPublicationError::Io)
    }

    fn ensure_root(&self) -> Result<Arc<File>, EffectReconciliationTrustPublicationError> {
        let mut pinned = self
            .pinned_root
            .lock()
            .map_err(|_| EffectReconciliationTrustPublicationError::RootLockPoisoned)?;
        if let Some(root) = pinned.as_ref() {
            return Ok(Arc::clone(root));
        }
        match fs::symlink_metadata(&self.root) {
            Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
                return Err(EffectReconciliationTrustPublicationError::InvalidRootDirectory);
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                fs::create_dir_all(&self.root)
                    .map_err(EffectReconciliationTrustPublicationError::Io)?;
            }
            Err(error) => return Err(EffectReconciliationTrustPublicationError::Io(error)),
        }
        fs::set_permissions(&self.root, fs::Permissions::from_mode(0o700))
            .map_err(EffectReconciliationTrustPublicationError::Io)?;
        let mut options = OpenOptions::new();
        options
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW);
        let root = Arc::new(
            options
                .open(&self.root)
                .map_err(EffectReconciliationTrustPublicationError::Io)?,
        );
        if !root
            .metadata()
            .map_err(EffectReconciliationTrustPublicationError::Io)?
            .is_dir()
        {
            return Err(EffectReconciliationTrustPublicationError::InvalidRootDirectory);
        }
        *pinned = Some(Arc::clone(&root));
        Ok(root)
    }

    fn operation_root_path(&self) -> Result<PathBuf, EffectReconciliationTrustPublicationError> {
        let root = self.ensure_root()?;
        let path = PathBuf::from(format!("/proc/self/fd/{}", root.as_raw_fd()));
        if !path.is_dir() {
            return Err(EffectReconciliationTrustPublicationError::PinnedRootUnavailable);
        }
        Ok(path)
    }
}

/// Borrowed current publication with the cross-process publication lock held.
#[derive(Debug)]
pub struct CurrentEffectReconciliationTrustFence<'a> {
    _local: MutexGuard<'a, ()>,
    _kernel: File,
    publication: EffectReconciliationTrustPublicationV1,
    head: EffectReconciliationTrustPublicationHead,
}

impl CurrentEffectReconciliationTrustFence<'_> {
    pub fn publication(&self) -> &EffectReconciliationTrustPublicationV1 {
        &self.publication
    }

    pub fn roots(&self) -> &EffectReconciliationTrustRootsV1 {
        self.publication.roots()
    }

    pub const fn head(&self) -> EffectReconciliationTrustPublicationHead {
        self.head
    }
}

fn validate_root_successor(
    previous: &EffectReconciliationTrustRootsV1,
    next: &EffectReconciliationTrustRootsV1,
) -> Result<(), EffectReconciliationTrustPublicationError> {
    if previous.device != next.device {
        return Err(EffectReconciliationTrustPublicationError::DeviceChanged);
    }
    validate_policy_transition(previous.outcome_policy, next.outcome_policy)?;
    validate_outcome_trust_transition(previous.outcome_trust_head, next.outcome_trust_head)?;
    Ok(())
}

fn validate_policy_transition(
    previous: EffectReconciliationPolicyAnchorV1,
    next: EffectReconciliationPolicyAnchorV1,
) -> Result<(), EffectReconciliationTrustPublicationError> {
    if next.generation < previous.generation {
        return Err(EffectReconciliationTrustPublicationError::PolicyRollback);
    }
    if next.generation == previous.generation && next.digest != previous.digest {
        return Err(EffectReconciliationTrustPublicationError::PolicySameGenerationMutation);
    }
    if next.generation > previous.generation && next.digest == previous.digest {
        return Err(EffectReconciliationTrustPublicationError::PolicyGenerationWithoutDigestChange);
    }
    Ok(())
}

fn validate_outcome_trust_transition(
    previous: PublishedEffectOutcomeTrustHeadV1,
    next: PublishedEffectOutcomeTrustHeadV1,
) -> Result<(), EffectReconciliationTrustPublicationError> {
    if next.sequence < previous.sequence {
        return Err(EffectReconciliationTrustPublicationError::OutcomeTrustRollback);
    }
    if next.sequence == previous.sequence && next.digest != previous.digest {
        return Err(EffectReconciliationTrustPublicationError::OutcomeTrustSameGenerationMutation);
    }
    if next.sequence > previous.sequence && next.digest == previous.digest {
        return Err(
            EffectReconciliationTrustPublicationError::OutcomeTrustGenerationWithoutDigestChange,
        );
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
            "effect-reconciliation trust publication object is not a regular file",
        ));
    }
    Ok(file)
}

fn system_unix_ms() -> Result<u64, EffectReconciliationTrustPublicationError> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| EffectReconciliationTrustPublicationError::SystemClockBeforeUnixEpoch)?;
    u64::try_from(elapsed.as_millis()).map_err(|_| EffectReconciliationTrustPublicationError::TimeOverflow)
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

fn update_policy_anchor(h: &mut blake3::Hasher, anchor: EffectReconciliationPolicyAnchorV1) {
    h.update(&anchor.generation.to_be_bytes());
    update_digest(h, anchor.digest);
}

fn update_outcome_trust_head(h: &mut blake3::Hasher, head: PublishedEffectOutcomeTrustHeadV1) {
    h.update(&head.sequence.to_be_bytes());
    update_digest(h, head.digest);
}

#[derive(Debug, Error)]
pub enum EffectReconciliationTrustPublicationError {
    #[error("unsupported effect-reconciliation trust publication schema")]
    UnsupportedPublicationSchema,
    #[error("effect-reconciliation trust publication generation is zero")]
    PublicationGenerationZero,
    #[error("effect-reconciliation trust publication generation overflow")]
    PublicationGenerationOverflow,
    #[error("effect-reconciliation trust publication genesis unexpectedly has a predecessor")]
    GenesisHasPredecessor,
    #[error("effect-reconciliation trust publication successor is missing a predecessor")]
    SuccessorMissingPredecessor,
    #[error("effect-reconciliation trust publication time is zero")]
    PublicationTimeZero,
    #[error("effect-reconciliation trust publication time regressed")]
    PublicationTimeRegressed,
    #[error("effect-reconciliation policy generation is zero")]
    PolicyGenerationZero,
    #[error("effect-reconciliation policy digest is zero")]
    PolicyDigestZero,
    #[error("effect-reconciliation publication device identity is invalid")]
    InvalidDeviceIdentity,
    #[error("published outcome-verifier trust head is invalid")]
    InvalidOutcomeTrustHead,
    #[error("effect-reconciliation publication changed device")]
    DeviceChanged,
    #[error("outcome-verifier trust generation rolled back")]
    OutcomeTrustRollback,
    #[error("outcome-verifier trust digest changed without advancing its generation")]
    OutcomeTrustSameGenerationMutation,
    #[error("outcome-verifier trust generation advanced without changing its digest")]
    OutcomeTrustGenerationWithoutDigestChange,
    #[error("outcome policy generation rolled back")]
    PolicyRollback,
    #[error("outcome policy digest changed without advancing its generation")]
    PolicySameGenerationMutation,
    #[error("outcome policy generation advanced without changing its digest")]
    PolicyGenerationWithoutDigestChange,
    #[error("effect-reconciliation trust successor changes no authoritative root")]
    NoOpSuccessor,
    #[error("effect-reconciliation trust publication store is already initialized")]
    AlreadyInitialized,
    #[error("effect-reconciliation trust publication store is uninitialized")]
    Uninitialized,
    #[error("trusted effect-reconciliation publication head is invalid")]
    InvalidTrustedHead,
    #[error("persisted effect-reconciliation publication differs from independently retained head")]
    TrustedHeadMismatch,
    #[error("effect-reconciliation publication state size is outside accepted bounds")]
    StateSizeOutOfBounds,
    #[error("effect-reconciliation publication state encoding/decoding failed")]
    StateEncoding,
    #[error("effect-reconciliation publication state is not canonically encoded")]
    NonCanonicalStateEncoding,
    #[error("persisted effect-reconciliation publication did not read back exactly")]
    PersistedStateReadbackMismatch,
    #[error("effect-reconciliation publication root is not a real directory")]
    InvalidRootDirectory,
    #[error("pinned descriptor-relative effect-reconciliation root is unavailable")]
    PinnedRootUnavailable,
    #[error("effect-reconciliation publication local lock is poisoned")]
    LocalLockPoisoned,
    #[error("effect-reconciliation publication root lock is poisoned")]
    RootLockPoisoned,
    #[error("system clock predates Unix epoch")]
    SystemClockBeforeUnixEpoch,
    #[error("system clock regressed during persistence")]
    SystemClockRegressedDuringPersistence,
    #[error("time conversion overflow")]
    TimeOverflow,
    #[error("effect-reconciliation trust publication I/O failed: {0}")]
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
            "symthaea-effect-reconciliation-trust-{label}-{}-{nanos}",
            std::process::id()
        ))
    }

    fn roots() -> EffectReconciliationTrustRootsV1 {
        EffectReconciliationTrustRootsV1 {
            device: ResourceRef("iot:valve:72".into()),
            outcome_policy: EffectReconciliationPolicyAnchorV1 {
                generation: 3,
                digest: d(1),
            },
            outcome_trust_head: PublishedEffectOutcomeTrustHeadV1 {
                sequence: 4,
                digest: d(2),
            },
        }
    }

    #[test]
    fn current_fence_blocks_authoritative_successor_publication() {
        let root = temp_root("fence-blocks-publish");
        let initial = DurableEffectReconciliationTrustPublicationStore::initialize(
            &root,
            roots(),
        )
        .unwrap();
        let head = initial.head();
        let fence_store =
            DurableEffectReconciliationTrustPublicationStore::open(&root, head).unwrap();
        let mutation_store =
            DurableEffectReconciliationTrustPublicationStore::open(&root, head).unwrap();
        let fence = fence_store.fence_current().unwrap();
        assert_eq!(fence.head(), head);
        assert_eq!(fence.roots(), &roots());

        let mut next = roots();
        next.outcome_trust_head = PublishedEffectOutcomeTrustHeadV1 {
            sequence: 5,
            digest: d(3),
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
    fn trust_and_policy_rollback_or_same_generation_substitution_fail_closed() {
        let trust_root = temp_root("trust-substitution");
        let initial =
            DurableEffectReconciliationTrustPublicationStore::initialize(&trust_root, roots())
                .unwrap();
        let mut changed = roots();
        changed.outcome_trust_head.digest = d(9);
        let store =
            DurableEffectReconciliationTrustPublicationStore::open(&trust_root, initial.head())
                .unwrap();
        assert!(matches!(
            store.publish_successor(changed),
            Err(EffectReconciliationTrustPublicationError::OutcomeTrustSameGenerationMutation)
        ));
        std::fs::remove_dir_all(trust_root).unwrap();

        let policy_root = temp_root("policy-rollback");
        let initial =
            DurableEffectReconciliationTrustPublicationStore::initialize(&policy_root, roots())
                .unwrap();
        let mut rollback = roots();
        rollback.outcome_policy.generation = 2;
        rollback.outcome_policy.digest = d(7);
        let store =
            DurableEffectReconciliationTrustPublicationStore::open(&policy_root, initial.head())
                .unwrap();
        assert!(matches!(
            store.publish_successor(rollback),
            Err(EffectReconciliationTrustPublicationError::PolicyRollback)
        ));
        std::fs::remove_dir_all(policy_root).unwrap();

        let policy_root = temp_root("policy-substitution");
        let initial =
            DurableEffectReconciliationTrustPublicationStore::initialize(&policy_root, roots())
                .unwrap();
        let mut mutation = roots();
        mutation.outcome_policy.digest = d(8);
        let store =
            DurableEffectReconciliationTrustPublicationStore::open(&policy_root, initial.head())
                .unwrap();
        assert!(matches!(
            store.publish_successor(mutation),
            Err(EffectReconciliationTrustPublicationError::PolicySameGenerationMutation)
        ));
        std::fs::remove_dir_all(policy_root).unwrap();
    }

    #[test]
    fn no_op_and_stale_external_head_are_rejected() {
        let root = temp_root("no-op");
        let initial =
            DurableEffectReconciliationTrustPublicationStore::initialize(&root, roots()).unwrap();
        let store =
            DurableEffectReconciliationTrustPublicationStore::open(&root, initial.head()).unwrap();
        assert!(matches!(
            store.publish_successor(roots()),
            Err(EffectReconciliationTrustPublicationError::NoOpSuccessor)
        ));

        let wrong = EffectReconciliationTrustPublicationHead {
            generation: initial.head().generation,
            digest: d(0xFE),
        };
        assert!(matches!(
            DurableEffectReconciliationTrustPublicationStore::open(&root, wrong),
            Err(EffectReconciliationTrustPublicationError::TrustedHeadMismatch)
        ));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn published_trust_head_round_trips_owner_local_identity() {
        let owner = EffectOutcomeTrustHead {
            sequence: 7,
            digest: d(0xA7),
        };
        let published = PublishedEffectOutcomeTrustHeadV1::from_outcome_trust_head(owner);
        assert!(published.matches_outcome_trust_head(owner));
        assert!(!published.matches_outcome_trust_head(EffectOutcomeTrustHead {
            sequence: 8,
            digest: d(0xA8),
        }));
    }
}
