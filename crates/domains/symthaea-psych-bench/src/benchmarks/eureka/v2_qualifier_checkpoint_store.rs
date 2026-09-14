// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Crash-conscious durable storage mechanics for the EUREKA-002 V2 qualifier
//! anti-rollback checkpoint.
//!
//! The anti-rollback checkpoint in `v2_qualifier_anti_rollback` is only a
//! high-water theorem while its bytes survive process and machine restart. This
//! module qualifies one deliberately conservative Unix persistence sequence:
//!
//! 1. acquire a same-directory `create_new` writer lock;
//! 2. under that lock, verify the durable target is the exact expected state;
//! 3. write the successor to a same-directory temporary regular file;
//! 4. `sync_all` the temporary file;
//! 5. reparse/recompare the exact temporary bytes;
//! 6. atomically rename the temporary file over the target;
//! 7. `sync_all` the parent directory;
//! 8. re-read/reparse the durable target;
//! 9. re-read the lock and require exact ownership before removal;
//! 10. remove the writer lock and `sync_all` the directory again;
//! 11. only then return a durable-commit receipt.
//!
//! The lock has no automatic stale-age deletion. A crash or ambiguous write
//! failure therefore tends to leave an observable, self-bound lock record
//! containing the expected predecessor and intended successor commitments.
//! Recovery is an explicit reconciliation ceremony, not a clock-based guess.
//!
//! This remains test-only mechanics. Filesystem/kernel/hardware durability and
//! host compromise remain in the TCB. The claimed write ordering is restricted
//! to Unix, where opening and syncing the containing directory is supported.

#![allow(dead_code)]

use super::v2_qualifier_anti_rollback::V2QualifierAntiRollbackCheckpoint;
use std::fs::{self, File};
use std::io::{ErrorKind, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(unix)]
use std::fs::OpenOptions;
#[cfg(unix)]
use std::io::Write;
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

pub(super) const V2_QUALIFIER_CHECKPOINT_WRITE_LOCK_SCHEMA: &str =
    "EUREKA.002.V2.QUALIFIER_CHECKPOINT_WRITE_LOCK.v1";
pub(super) const V2_QUALIFIER_CHECKPOINT_WRITE_LOCK_COMMITMENT_REVISION: &str =
    "EUREKA.002.V2.QUALIFIER_CHECKPOINT_WRITE_LOCK_COMMITMENT.v1";
pub(super) const V2_DURABLE_CHECKPOINT_COMMIT_RECEIPT_REVISION: &str =
    "EUREKA.002.V2.DURABLE_CHECKPOINT_COMMIT_RECEIPT.v1";

const MAX_CHECKPOINT_BYTES: u64 = 4096;
const MAX_LOCK_BYTES: u64 = 1024;
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct V2CheckpointWriteLockRecord {
    expected_checkpoint_commitment: Option<[u8; 32]>,
    intended_checkpoint_commitment: [u8; 32],
    commitment: [u8; 32],
}

impl V2CheckpointWriteLockRecord {
    fn new(
        expected_checkpoint_commitment: Option<[u8; 32]>,
        intended_checkpoint_commitment: [u8; 32],
    ) -> Self {
        let mut record = Self {
            expected_checkpoint_commitment,
            intended_checkpoint_commitment,
            commitment: [0_u8; 32],
        };
        record.commitment = lock_record_commitment(&record);
        record
    }

    pub(super) const fn expected_checkpoint_commitment(&self) -> Option<[u8; 32]> {
        self.expected_checkpoint_commitment
    }

    pub(super) const fn intended_checkpoint_commitment(&self) -> [u8; 32] {
        self.intended_checkpoint_commitment
    }

    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }

    fn canonical_body_bytes(&self) -> Vec<u8> {
        format!(
            "lock_schema_revision={V2_QUALIFIER_CHECKPOINT_WRITE_LOCK_SCHEMA}\n\
expected_checkpoint_commitment={}\n\
intended_checkpoint_commitment={}\n",
            optional_hex(self.expected_checkpoint_commitment),
            hex32(self.intended_checkpoint_commitment),
        )
        .into_bytes()
    }

    fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = self.canonical_body_bytes();
        bytes.extend_from_slice(
            format!("lock_commitment={}\n", hex32(self.commitment)).as_bytes(),
        );
        bytes
    }

    fn parse(input: &[u8]) -> Result<Self, V2CheckpointStoreError> {
        let text = std::str::from_utf8(input).map_err(|_| V2CheckpointStoreError::CorruptLock)?;
        if !text.ends_with('\n') {
            return Err(V2CheckpointStoreError::CorruptLock);
        }
        let body = &text[..text.len() - 1];
        let lines: Vec<&str> = body.split('\n').collect();
        if lines.len() != 4 {
            return Err(V2CheckpointStoreError::CorruptLock);
        }
        if exact_field(lines[0], "lock_schema_revision")?
            != V2_QUALIFIER_CHECKPOINT_WRITE_LOCK_SCHEMA
        {
            return Err(V2CheckpointStoreError::CorruptLock);
        }
        let expected = optional_commitment(exact_field(
            lines[1],
            "expected_checkpoint_commitment",
        )?)?;
        let intended = decode_hex32(exact_field(
            lines[2],
            "intended_checkpoint_commitment",
        )?)?;
        let claimed_commitment = decode_hex32(exact_field(lines[3], "lock_commitment")?)?;
        let record = Self::new(expected, intended);
        if record.commitment != claimed_commitment || record.canonical_bytes() != input {
            return Err(V2CheckpointStoreError::CorruptLock);
        }
        Ok(record)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct V2DurableCheckpointCommitReceipt {
    checkpoint_commitment: [u8; 32],
    predecessor_checkpoint_commitment: Option<[u8; 32]>,
    persisted_byte_len: u64,
    commitment: [u8; 32],
}

impl V2DurableCheckpointCommitReceipt {
    fn new(checkpoint: &V2QualifierAntiRollbackCheckpoint, persisted_byte_len: u64) -> Self {
        let mut receipt = Self {
            checkpoint_commitment: checkpoint.commitment(),
            predecessor_checkpoint_commitment: checkpoint.predecessor_checkpoint_commitment(),
            persisted_byte_len,
            commitment: [0_u8; 32],
        };
        receipt.commitment = durable_receipt_commitment(&receipt);
        receipt
    }

    pub(super) const fn checkpoint_commitment(&self) -> [u8; 32] {
        self.checkpoint_commitment
    }

    pub(super) const fn predecessor_checkpoint_commitment(&self) -> Option<[u8; 32]> {
        self.predecessor_checkpoint_commitment
    }

    pub(super) const fn persisted_byte_len(&self) -> u64 {
        self.persisted_byte_len
    }

    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
}

#[derive(Debug, Clone)]
pub(super) struct V2QualifierCheckpointStore {
    path: PathBuf,
}

impl V2QualifierCheckpointStore {
    pub(super) fn new(path: impl Into<PathBuf>) -> Result<Self, V2CheckpointStoreError> {
        let path = path.into();
        let Some(parent) = path.parent() else {
            return Err(V2CheckpointStoreError::InvalidPath);
        };
        if path.file_name().and_then(|name| name.to_str()).is_none() {
            return Err(V2CheckpointStoreError::InvalidPath);
        }
        let parent_meta = fs::metadata(parent).map_err(|_| V2CheckpointStoreError::MissingParent)?;
        if !parent_meta.is_dir() {
            return Err(V2CheckpointStoreError::MissingParent);
        }
        Ok(Self { path })
    }

    pub(super) fn path(&self) -> &Path {
        &self.path
    }

    pub(super) fn load(
        &self,
    ) -> Result<Option<V2QualifierAntiRollbackCheckpoint>, V2CheckpointStoreError> {
        match fs::symlink_metadata(&self.path) {
            Ok(metadata) => {
                if !metadata.file_type().is_file() || metadata.len() > MAX_CHECKPOINT_BYTES {
                    return Err(V2CheckpointStoreError::UnsafeTarget);
                }
                let bytes = read_bounded(&self.path, MAX_CHECKPOINT_BYTES)
                    .map_err(|_| V2CheckpointStoreError::CheckpointReadFailed)?;
                let checkpoint = V2QualifierAntiRollbackCheckpoint::parse_persisted(&bytes)
                    .map_err(|_| V2CheckpointStoreError::CorruptCheckpoint)?;
                Ok(Some(checkpoint))
            }
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(None),
            Err(_) => Err(V2CheckpointStoreError::CheckpointReadFailed),
        }
    }

    pub(super) fn inspect_writer_lock(
        &self,
    ) -> Result<Option<V2CheckpointWriteLockRecord>, V2CheckpointStoreError> {
        let lock_path = self.lock_path()?;
        match fs::symlink_metadata(&lock_path) {
            Ok(metadata) => {
                if !metadata.file_type().is_file() || metadata.len() > MAX_LOCK_BYTES {
                    return Err(V2CheckpointStoreError::CorruptLock);
                }
                let bytes = read_bounded(&lock_path, MAX_LOCK_BYTES)
                    .map_err(|_| V2CheckpointStoreError::CorruptLock)?;
                Ok(Some(V2CheckpointWriteLockRecord::parse(&bytes)?))
            }
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(None),
            Err(_) => Err(V2CheckpointStoreError::CorruptLock),
        }
    }

    pub(super) fn persist_genesis(
        &self,
        checkpoint: &V2QualifierAntiRollbackCheckpoint,
    ) -> Result<V2DurableCheckpointCommitReceipt, V2CheckpointStoreError> {
        if checkpoint.predecessor_checkpoint_commitment().is_some()
            || checkpoint.authority_sequence() != 1
            || checkpoint.signer_policy_sequence() != 1
        {
            return Err(V2CheckpointStoreError::WrongTransitionShape);
        }
        self.persist_inner(None, checkpoint, V2CheckpointStoreFailpoint::None)
    }

    pub(super) fn persist_successor(
        &self,
        expected_current: &V2QualifierAntiRollbackCheckpoint,
        successor: &V2QualifierAntiRollbackCheckpoint,
    ) -> Result<V2DurableCheckpointCommitReceipt, V2CheckpointStoreError> {
        if successor.predecessor_checkpoint_commitment() != Some(expected_current.commitment())
            || successor.commitment() == expected_current.commitment()
        {
            return Err(V2CheckpointStoreError::WrongTransitionShape);
        }
        self.persist_inner(
            Some(expected_current),
            successor,
            V2CheckpointStoreFailpoint::None,
        )
    }

    fn persist_inner(
        &self,
        expected_current: Option<&V2QualifierAntiRollbackCheckpoint>,
        successor: &V2QualifierAntiRollbackCheckpoint,
        failpoint: V2CheckpointStoreFailpoint,
    ) -> Result<V2DurableCheckpointCommitReceipt, V2CheckpointStoreError> {
        #[cfg(not(unix))]
        {
            let _ = (expected_current, successor, failpoint);
            return Err(V2CheckpointStoreError::UnsupportedPlatform);
        }

        #[cfg(unix)]
        {
            let parent = self.parent()?;
            let lock_path = self.lock_path()?;
            let lock_record = V2CheckpointWriteLockRecord::new(
                expected_current.map(V2QualifierAntiRollbackCheckpoint::commitment),
                successor.commitment(),
            );
            self.acquire_lock(&lock_path, &lock_record)?;

            if failpoint == V2CheckpointStoreFailpoint::AfterLockSync {
                return Err(V2CheckpointStoreError::InjectedFailure);
            }

            self.verify_expected_current(expected_current)?;

            let successor_bytes = successor.persisted_bytes();
            if successor_bytes.len() as u64 > MAX_CHECKPOINT_BYTES {
                return Err(V2CheckpointStoreError::CheckpointTooLarge);
            }
            let temp_path = self.temp_path()?;
            self.write_temp_file(&temp_path, &successor_bytes)?;

            if failpoint == V2CheckpointStoreFailpoint::AfterTempSync {
                return Err(V2CheckpointStoreError::InjectedFailure);
            }

            let temp_bytes = read_bounded(&temp_path, MAX_CHECKPOINT_BYTES)
                .map_err(|_| V2CheckpointStoreError::CheckpointReadFailed)?;
            let reparsed = V2QualifierAntiRollbackCheckpoint::parse_persisted(&temp_bytes)
                .map_err(|_| V2CheckpointStoreError::CorruptCheckpoint)?;
            if &reparsed != successor || temp_bytes != successor_bytes {
                return Err(V2CheckpointStoreError::CheckpointMismatch);
            }

            fs::rename(&temp_path, &self.path)
                .map_err(|_| V2CheckpointStoreError::RenameFailed)?;

            if failpoint == V2CheckpointStoreFailpoint::AfterRenameBeforeDirectorySync {
                return Err(V2CheckpointStoreError::InjectedFailure);
            }

            sync_directory(parent)?;

            if failpoint == V2CheckpointStoreFailpoint::AfterDirectorySyncBeforeUnlock {
                return Err(V2CheckpointStoreError::InjectedFailure);
            }

            let durable = self
                .load()?
                .ok_or(V2CheckpointStoreError::CheckpointMismatch)?;
            if &durable != successor {
                return Err(V2CheckpointStoreError::CheckpointMismatch);
            }

            let observed_lock = self
                .inspect_writer_lock()?
                .ok_or(V2CheckpointStoreError::LockOwnershipChanged)?;
            if observed_lock != lock_record {
                return Err(V2CheckpointStoreError::LockOwnershipChanged);
            }

            fs::remove_file(&lock_path).map_err(|_| V2CheckpointStoreError::UnlockFailed)?;
            sync_directory(parent)?;

            Ok(V2DurableCheckpointCommitReceipt::new(
                successor,
                successor_bytes.len() as u64,
            ))
        }
    }

    fn persist_with_failpoint(
        &self,
        expected_current: Option<&V2QualifierAntiRollbackCheckpoint>,
        successor: &V2QualifierAntiRollbackCheckpoint,
        failpoint: V2CheckpointStoreFailpoint,
    ) -> Result<V2DurableCheckpointCommitReceipt, V2CheckpointStoreError> {
        self.persist_inner(expected_current, successor, failpoint)
    }

    fn verify_expected_current(
        &self,
        expected_current: Option<&V2QualifierAntiRollbackCheckpoint>,
    ) -> Result<(), V2CheckpointStoreError> {
        let durable = self.load()?;
        match (expected_current, durable.as_ref()) {
            (None, None) => Ok(()),
            (None, Some(_)) => Err(V2CheckpointStoreError::CheckpointAlreadyExists),
            (Some(_), None) => Err(V2CheckpointStoreError::MissingCheckpoint),
            (Some(expected), Some(actual)) if expected == actual => Ok(()),
            (Some(_), Some(_)) => Err(V2CheckpointStoreError::HighWaterMismatch),
        }
    }

    #[cfg(unix)]
    fn acquire_lock(
        &self,
        lock_path: &Path,
        record: &V2CheckpointWriteLockRecord,
    ) -> Result<(), V2CheckpointStoreError> {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true).mode(0o600);
        let mut file = match options.open(lock_path) {
            Ok(file) => file,
            Err(error) if error.kind() == ErrorKind::AlreadyExists => {
                return Err(V2CheckpointStoreError::WriterLockPresent);
            }
            Err(_) => return Err(V2CheckpointStoreError::LockCreateFailed),
        };
        file.write_all(&record.canonical_bytes())
            .map_err(|_| V2CheckpointStoreError::LockWriteFailed)?;
        file.sync_all()
            .map_err(|_| V2CheckpointStoreError::LockSyncFailed)?;
        sync_directory(self.parent()?)
    }

    #[cfg(unix)]
    fn write_temp_file(
        &self,
        temp_path: &Path,
        bytes: &[u8],
    ) -> Result<(), V2CheckpointStoreError> {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true).mode(0o600);
        let mut file = options
            .open(temp_path)
            .map_err(|_| V2CheckpointStoreError::TempCreateFailed)?;
        file.write_all(bytes)
            .map_err(|_| V2CheckpointStoreError::TempWriteFailed)?;
        file.sync_all()
            .map_err(|_| V2CheckpointStoreError::TempSyncFailed)?;
        drop(file);
        Ok(())
    }

    fn parent(&self) -> Result<&Path, V2CheckpointStoreError> {
        self.path
            .parent()
            .ok_or(V2CheckpointStoreError::InvalidPath)
    }

    fn lock_path(&self) -> Result<PathBuf, V2CheckpointStoreError> {
        let parent = self.parent()?;
        let name = self
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or(V2CheckpointStoreError::InvalidPath)?;
        Ok(parent.join(format!(".{name}.eureka-v2.lock")))
    }

    fn temp_path(&self) -> Result<PathBuf, V2CheckpointStoreError> {
        let parent = self.parent()?;
        let name = self
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or(V2CheckpointStoreError::InvalidPath)?;
        let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        Ok(parent.join(format!(
            ".{name}.eureka-v2.tmp.{}.{}",
            std::process::id(),
            counter
        )))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum V2CheckpointStoreFailpoint {
    None,
    AfterLockSync,
    AfterTempSync,
    AfterRenameBeforeDirectorySync,
    AfterDirectorySyncBeforeUnlock,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2CheckpointStoreError {
    UnsupportedPlatform,
    InvalidPath,
    MissingParent,
    UnsafeTarget,
    CheckpointReadFailed,
    CorruptCheckpoint,
    CheckpointTooLarge,
    CheckpointMismatch,
    CheckpointAlreadyExists,
    MissingCheckpoint,
    HighWaterMismatch,
    WrongTransitionShape,
    WriterLockPresent,
    LockCreateFailed,
    LockWriteFailed,
    LockSyncFailed,
    CorruptLock,
    LockOwnershipChanged,
    TempCreateFailed,
    TempWriteFailed,
    TempSyncFailed,
    RenameFailed,
    DirectorySyncFailed,
    UnlockFailed,
    InjectedFailure,
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> Result<(), V2CheckpointStoreError> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|_| V2CheckpointStoreError::DirectorySyncFailed)
}

fn read_bounded(path: &Path, max: u64) -> std::io::Result<Vec<u8>> {
    let mut file = File::open(path)?;
    let metadata = file.metadata()?;
    if metadata.len() > max {
        return Err(std::io::Error::new(
            ErrorKind::InvalidData,
            "file exceeds bounded checkpoint size",
        ));
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn lock_record_commitment(record: &V2CheckpointWriteLockRecord) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(
        &mut bytes,
        V2_QUALIFIER_CHECKPOINT_WRITE_LOCK_COMMITMENT_REVISION.as_bytes(),
    );
    encode_bytes(&mut bytes, &record.canonical_body_bytes());
    *blake3::hash(&bytes).as_bytes()
}

fn durable_receipt_commitment(receipt: &V2DurableCheckpointCommitReceipt) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(
        &mut bytes,
        V2_DURABLE_CHECKPOINT_COMMIT_RECEIPT_REVISION.as_bytes(),
    );
    bytes.extend_from_slice(&receipt.checkpoint_commitment);
    match receipt.predecessor_checkpoint_commitment {
        Some(predecessor) => {
            bytes.push(1);
            bytes.extend_from_slice(&predecessor);
        }
        None => bytes.push(0),
    }
    bytes.extend_from_slice(&receipt.persisted_byte_len.to_le_bytes());
    *blake3::hash(&bytes).as_bytes()
}

fn exact_field<'a>(line: &'a str, key: &str) -> Result<&'a str, V2CheckpointStoreError> {
    let Some((actual_key, value)) = line.split_once('=') else {
        return Err(V2CheckpointStoreError::CorruptLock);
    };
    if actual_key != key || value.is_empty() || value.contains('=') {
        return Err(V2CheckpointStoreError::CorruptLock);
    }
    Ok(value)
}

fn optional_commitment(value: &str) -> Result<Option<[u8; 32]>, V2CheckpointStoreError> {
    if value == "none" {
        Ok(None)
    } else {
        decode_hex32(value).map(Some)
    }
}

fn decode_hex32(value: &str) -> Result<[u8; 32], V2CheckpointStoreError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(V2CheckpointStoreError::CorruptLock);
    }
    let mut output = [0_u8; 32];
    for (index, chunk) in value.as_bytes().chunks_exact(2).enumerate() {
        let high = hex_nibble(chunk[0]).ok_or(V2CheckpointStoreError::CorruptLock)?;
        let low = hex_nibble(chunk[1]).ok_or(V2CheckpointStoreError::CorruptLock)?;
        output[index] = (high << 4) | low;
    }
    Ok(output)
}

const fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

fn optional_hex(value: Option<[u8; 32]>) -> String {
    value.map(hex32).unwrap_or_else(|| "none".to_string())
}

fn hex32(value: [u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in value {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

fn encode_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmarks::eureka::v2_qualifier_anti_rollback::V2QualifierAntiRollbackGuard;
    use crate::benchmarks::eureka::v2_qualifier_authority::{
        V2QualifierAuthorityProfile, V2QualifierAuthorityRecord,
    };
    use crate::benchmarks::eureka::v2_qualifier_currentness::V2QualifierAuthorityLineage;
    use crate::benchmarks::eureka::v2_qualifier_signer_policy::{
        V2GovernanceSigner, V2QualifierSignerPolicy,
    };
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;

    static TEST_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn profile(workflow: char, contract: char) -> V2QualifierAuthorityProfile {
        V2QualifierAuthorityProfile::current_from_hex(
            &workflow.to_string().repeat(64),
            &contract.to_string().repeat(64),
        )
        .unwrap()
    }

    fn signer(principal: &str, key: char) -> V2GovernanceSigner {
        V2GovernanceSigner::from_hex(
            principal,
            "ssh-ed25519",
            &key.to_string().repeat(64),
        )
        .unwrap()
    }

    fn policy() -> V2QualifierSignerPolicy {
        V2QualifierSignerPolicy::genesis(
            1,
            1,
            vec![signer("governance@example.invalid", 'a')],
        )
        .unwrap()
    }

    fn successor_state() -> (
        V2QualifierAntiRollbackCheckpoint,
        V2QualifierAntiRollbackCheckpoint,
    ) {
        let authority = V2QualifierAuthorityRecord::genesis(1, profile('b', 'c')).unwrap();
        let successor = V2QualifierAuthorityRecord::rotate(
            &authority,
            authority.commitment(),
            2,
            profile('d', 'c'),
        )
        .unwrap();
        let lineage = V2QualifierAuthorityLineage::activate_genesis(authority).unwrap();
        let policy = policy();
        let guard = V2QualifierAntiRollbackGuard::activate_genesis(&lineage, &policy).unwrap();
        let genesis = guard.checkpoint().clone();
        let (guard, _) = guard.advance_authority(lineage, successor, &policy).unwrap();
        (genesis, guard.checkpoint().clone())
    }

    fn different_genesis() -> V2QualifierAntiRollbackCheckpoint {
        let authority = V2QualifierAuthorityRecord::genesis(1, profile('e', 'f')).unwrap();
        let lineage = V2QualifierAuthorityLineage::activate_genesis(authority).unwrap();
        let policy = policy();
        V2QualifierAntiRollbackGuard::activate_genesis(&lineage, &policy)
            .unwrap()
            .checkpoint()
            .clone()
    }

    fn temp_store() -> (PathBuf, V2QualifierCheckpointStore) {
        let id = TEST_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "symthaea-eureka-v2-checkpoint-store-{}-{id}",
            std::process::id()
        ));
        fs::create_dir(&dir).unwrap();
        let store = V2QualifierCheckpointStore::new(dir.join("high-water.checkpoint")).unwrap();
        (dir, store)
    }

    fn cleanup(dir: &Path) {
        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(unix)]
    #[test]
    fn genesis_and_successor_commit_only_after_full_durable_sequence() {
        let (dir, store) = temp_store();
        let (genesis, successor) = successor_state();

        let genesis_receipt = store.persist_genesis(&genesis).unwrap();
        assert_eq!(genesis_receipt.checkpoint_commitment(), genesis.commitment());
        assert_eq!(genesis_receipt.predecessor_checkpoint_commitment(), None);
        assert_ne!(genesis_receipt.commitment(), [0_u8; 32]);
        assert_eq!(store.load().unwrap().as_ref(), Some(&genesis));
        assert!(store.inspect_writer_lock().unwrap().is_none());

        let successor_receipt = store.persist_successor(&genesis, &successor).unwrap();
        assert_eq!(successor_receipt.checkpoint_commitment(), successor.commitment());
        assert_eq!(
            successor_receipt.predecessor_checkpoint_commitment(),
            Some(genesis.commitment())
        );
        assert_eq!(store.load().unwrap().as_ref(), Some(&successor));
        assert!(store.inspect_writer_lock().unwrap().is_none());

        let mode = fs::metadata(store.path()).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
        cleanup(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn high_water_mismatch_fails_closed_under_observable_lock() {
        let (dir, store) = temp_store();
        let (genesis, successor) = successor_state();
        let durable_different = different_genesis();
        store.persist_genesis(&durable_different).unwrap();

        assert_eq!(
            store.persist_successor(&genesis, &successor).unwrap_err(),
            V2CheckpointStoreError::HighWaterMismatch
        );
        let lock = store.inspect_writer_lock().unwrap().unwrap();
        assert_eq!(lock.expected_checkpoint_commitment(), Some(genesis.commitment()));
        assert_eq!(lock.intended_checkpoint_commitment(), successor.commitment());
        assert_ne!(lock.commitment(), [0_u8; 32]);
        assert_eq!(store.load().unwrap().as_ref(), Some(&durable_different));
        cleanup(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn second_writer_is_rejected_while_fail_closed_lock_exists() {
        let (dir, store) = temp_store();
        let (genesis, _) = successor_state();
        assert_eq!(
            store
                .persist_with_failpoint(
                    None,
                    &genesis,
                    V2CheckpointStoreFailpoint::AfterLockSync,
                )
                .unwrap_err(),
            V2CheckpointStoreError::InjectedFailure
        );
        assert!(store.inspect_writer_lock().unwrap().is_some());
        assert_eq!(
            store.persist_genesis(&genesis).unwrap_err(),
            V2CheckpointStoreError::WriterLockPresent
        );
        assert!(store.load().unwrap().is_none());
        cleanup(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn self_bound_lock_detects_valid_looking_commitment_mutation() {
        let (dir, store) = temp_store();
        let (genesis, _) = successor_state();
        assert_eq!(
            store
                .persist_with_failpoint(
                    None,
                    &genesis,
                    V2CheckpointStoreFailpoint::AfterLockSync,
                )
                .unwrap_err(),
            V2CheckpointStoreError::InjectedFailure
        );
        let lock_path = store.lock_path().unwrap();
        let original = fs::read_to_string(&lock_path).unwrap();
        let mutated = original.replacen(
            &format!("intended_checkpoint_commitment={}", hex32(genesis.commitment())),
            &format!("intended_checkpoint_commitment={}", "f".repeat(64)),
            1,
        );
        fs::write(&lock_path, mutated).unwrap();
        assert_eq!(
            store.inspect_writer_lock().unwrap_err(),
            V2CheckpointStoreError::CorruptLock
        );
        cleanup(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn crash_before_rename_leaves_old_target_and_lock() {
        let (dir, store) = temp_store();
        let (genesis, successor) = successor_state();
        store.persist_genesis(&genesis).unwrap();

        assert_eq!(
            store
                .persist_with_failpoint(
                    Some(&genesis),
                    &successor,
                    V2CheckpointStoreFailpoint::AfterTempSync,
                )
                .unwrap_err(),
            V2CheckpointStoreError::InjectedFailure
        );
        assert_eq!(store.load().unwrap().as_ref(), Some(&genesis));
        let lock = store.inspect_writer_lock().unwrap().unwrap();
        assert_eq!(lock.expected_checkpoint_commitment(), Some(genesis.commitment()));
        assert_eq!(lock.intended_checkpoint_commitment(), successor.commitment());
        cleanup(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn crash_after_rename_is_explicitly_ambiguous_and_remains_locked() {
        let (dir, store) = temp_store();
        let (genesis, successor) = successor_state();
        store.persist_genesis(&genesis).unwrap();

        assert_eq!(
            store
                .persist_with_failpoint(
                    Some(&genesis),
                    &successor,
                    V2CheckpointStoreFailpoint::AfterRenameBeforeDirectorySync,
                )
                .unwrap_err(),
            V2CheckpointStoreError::InjectedFailure
        );
        assert_eq!(store.load().unwrap().as_ref(), Some(&successor));
        let lock = store.inspect_writer_lock().unwrap().unwrap();
        assert_eq!(lock.expected_checkpoint_commitment(), Some(genesis.commitment()));
        assert_eq!(lock.intended_checkpoint_commitment(), successor.commitment());
        assert_eq!(
            store.persist_successor(&genesis, &successor).unwrap_err(),
            V2CheckpointStoreError::WriterLockPresent
        );
        cleanup(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn crash_after_directory_sync_before_unlock_keeps_durable_successor_locked() {
        let (dir, store) = temp_store();
        let (genesis, successor) = successor_state();
        store.persist_genesis(&genesis).unwrap();

        assert_eq!(
            store
                .persist_with_failpoint(
                    Some(&genesis),
                    &successor,
                    V2CheckpointStoreFailpoint::AfterDirectorySyncBeforeUnlock,
                )
                .unwrap_err(),
            V2CheckpointStoreError::InjectedFailure
        );
        assert_eq!(store.load().unwrap().as_ref(), Some(&successor));
        assert!(store.inspect_writer_lock().unwrap().is_some());
        cleanup(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn symlink_target_is_rejected_instead_of_followed() {
        use std::os::unix::fs::symlink;

        let (dir, store) = temp_store();
        let outside = dir.join("outside");
        fs::write(&outside, b"not a checkpoint").unwrap();
        symlink(&outside, store.path()).unwrap();

        assert_eq!(store.load().unwrap_err(), V2CheckpointStoreError::UnsafeTarget);
        cleanup(&dir);
    }

    #[test]
    fn store_source_has_no_clock_based_stale_lock_or_execution_authority() {
        let production = include_str!("v2_qualifier_checkpoint_store.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        for forbidden in [
            "SystemTime",
            "UNIX_EPOCH",
            "stale_after",
            "V2AdmittedQualifierRoot",
            "V2CanaryAuthorization",
            "predict_ticket",
            "reveal(",
            "score_consequence",
        ] {
            assert!(
                !production.contains(forbidden),
                "forbidden durable-store surface: {forbidden}"
            );
        }
        assert!(production.contains("create_new(true)"));
        assert!(production.contains("sync_all"));
        assert!(production.contains("fs::rename"));
        assert!(production.contains("LockOwnershipChanged"));
        assert!(production.contains("fs::remove_file(&lock_path)"));
    }
}
