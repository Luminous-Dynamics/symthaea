// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Append-only filesystem reference adapter for the MACE monotonic witness.
//!
//! This adapter provides restart persistence and atomic create-if-absent
//! checkpoints within one filesystem namespace. It is NOT a production
//! independent trust anchor: restoring a filesystem snapshot that contains
//! both this directory and the memory store can roll both back together.
//! Production deployments must place the witness in a separately administered
//! failure/rollback domain and authenticate its trust anchor out of band.

use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

use super::{
    LifecycleError, MemorySnapshot, MonotonicWitnessStore, WitnessCheckpoint,
};

const MAGIC: &[u8; 8] = b"MACEWIT1";
const FILE_EXTENSION: &str = "macew";
const MAX_CHECKPOINT_BYTES: u64 = 16 * 1024 * 1024;
const MAX_TOMBSTONES: usize = 100_000;
const MAX_CHECKPOINTS: usize = 1_000_000;
const MAX_FIELD_BYTES: usize = 4096;
const GENERATION_WIDTH: usize = 20;

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Restart-persistent reference witness backed by immutable checkpoint files.
///
/// A checkpoint is first written and fsynced to a unique temporary file, then
/// linked to its final generation name using an atomic create-if-absent
/// operation. Competing writers cannot overwrite an existing generation.
/// The directory is fsynced on Unix after publication.
///
/// This provides local append-only behavior against ordinary process crashes
/// and concurrent writers. It does NOT prevent rollback of the entire
/// filesystem namespace, offline replacement of both data and witness files,
/// or compromise of the host/root account. It is a conformance adapter, not an
/// independently administered production witness service.
#[derive(Clone, Debug)]
pub struct FileBackedWitnessStore {
    directory: PathBuf,
}

impl FileBackedWitnessStore {
    /// Open a witness directory, creating and publishing the genesis checkpoint
    /// when the directory contains no checkpoint history.
    ///
    /// Concurrent first-open attempts are safe: only one writer can publish
    /// generation zero; the other validates the winner's checkpoint chain.
    pub fn open(directory: impl AsRef<Path>) -> Result<Self, LifecycleError> {
        fs::create_dir_all(directory.as_ref())
            .map_err(|_| LifecycleError::WitnessUnavailable)?;
        let directory = fs::canonicalize(directory.as_ref())
            .map_err(|_| LifecycleError::WitnessUnavailable)?;
        if !directory.is_dir() {
            return Err(LifecycleError::WitnessUnavailable);
        }
        #[cfg(unix)]
        fs::set_permissions(&directory, fs::Permissions::from_mode(0o700))
            .map_err(|_| LifecycleError::WitnessUnavailable)?;

        let store = Self { directory };
        match store.read_latest_checkpoint() {
            Ok(_) => Ok(store),
            Err(LifecycleError::WitnessUnavailable) => {
                let genesis = genesis_checkpoint();
                match store.publish_checkpoint(&genesis) {
                    Ok(()) => {
                        store.read_latest_checkpoint()?;
                        Ok(store)
                    }
                    Err(LifecycleError::WitnessConflict) => {
                        store.read_latest_checkpoint()?;
                        Ok(store)
                    }
                    Err(error) => Err(error),
                }
            }
            Err(error) => Err(error),
        }
    }

    /// Canonical directory used by this reference adapter.
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    fn checkpoint_path(&self, generation: u64) -> PathBuf {
        self.directory.join(format!(
            "checkpoint-{generation:0width$}.{extension}",
            width = GENERATION_WIDTH,
            extension = FILE_EXTENSION
        ))
    }

    fn read_latest_checkpoint(&self) -> Result<WitnessCheckpoint, LifecycleError> {
        let mut entries = Vec::<(u64, PathBuf)>::new();
        let iterator = fs::read_dir(&self.directory)
            .map_err(|_| LifecycleError::WitnessUnavailable)?;

        for entry in iterator {
            let entry = entry.map_err(|_| LifecycleError::WitnessUnavailable)?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            if !name.starts_with("checkpoint-") {
                // Interrupted temporary writes are non-authoritative. They
                // never use the final checkpoint namespace.
                continue;
            }

            let suffix = name
                .strip_prefix("checkpoint-")
                .ok_or(LifecycleError::WitnessCheckpointInvalid)?;
            let (digits, extension) = suffix
                .split_once('.')
                .ok_or(LifecycleError::WitnessCheckpointInvalid)?;
            if extension != FILE_EXTENSION
                || digits.len() != GENERATION_WIDTH
                || !digits.bytes().all(|byte| byte.is_ascii_digit())
            {
                return Err(LifecycleError::WitnessCheckpointInvalid);
            }
            let generation = digits
                .parse::<u64>()
                .map_err(|_| LifecycleError::WitnessCheckpointInvalid)?;
            let file_type = entry
                .file_type()
                .map_err(|_| LifecycleError::WitnessUnavailable)?;
            if !file_type.is_file() {
                return Err(LifecycleError::WitnessCheckpointInvalid);
            }
            entries.push((generation, entry.path()));
        }

        if entries.is_empty() {
            return Err(LifecycleError::WitnessUnavailable);
        }
        if entries.len() > MAX_CHECKPOINTS {
            return Err(LifecycleError::WitnessCheckpointInvalid);
        }
        entries.sort_by_key(|(generation, _)| *generation);
        if entries[0].0 != 0 {
            return Err(LifecycleError::WitnessHistoryRegression);
        }

        let mut previous: Option<WitnessCheckpoint> = None;
        for (expected_generation, (generation, path)) in entries.iter().enumerate() {
            let expected_generation = u64::try_from(expected_generation)
                .map_err(|_| LifecycleError::GenerationOverflow)?;
            if *generation != expected_generation {
                return Err(LifecycleError::WitnessHistoryRegression);
            }
            let checkpoint = read_checkpoint(path)?;
            validate_checkpoint(&checkpoint)?;

            match &previous {
                None => {
                    if checkpoint != genesis_checkpoint() {
                        return Err(LifecycleError::WitnessHistoryRegression);
                    }
                }
                Some(current) => {
                    let expected_next = current
                        .generation
                        .checked_add(1)
                        .ok_or(LifecycleError::GenerationOverflow)?;
                    if checkpoint.generation != expected_next
                        || checkpoint.previous_checkpoint_sha256.as_deref()
                            != Some(current.checkpoint_sha256.as_str())
                        || !checkpoint.revoked_ids.is_superset(&current.revoked_ids)
                        || !checkpoint.erased_ids.is_superset(&current.erased_ids)
                    {
                        return Err(LifecycleError::WitnessHistoryRegression);
                    }
                }
            }
            previous = Some(checkpoint);
        }

        previous.ok_or(LifecycleError::WitnessUnavailable)
    }

    fn publish_checkpoint(
        &self,
        checkpoint: &WitnessCheckpoint,
    ) -> Result<(), LifecycleError> {
        validate_checkpoint(checkpoint)?;
        let final_path = self.checkpoint_path(checkpoint.generation);
        let encoded = encode_checkpoint(checkpoint)?;

        let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let temporary_path = self.directory.join(format!(
            ".pending-{}-{counter:016x}",
            std::process::id()
        ));

        let publish_result = (|| {
            let mut options = OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            options.mode(0o600);
            let mut temporary = options
                .open(&temporary_path)
                .map_err(|error| {
                    if error.kind() == std::io::ErrorKind::AlreadyExists {
                        LifecycleError::WitnessConflict
                    } else {
                        LifecycleError::WitnessUnavailable
                    }
                })?;
            temporary
                .write_all(&encoded)
                .map_err(|_| LifecycleError::WitnessUnavailable)?;
            temporary
                .sync_all()
                .map_err(|_| LifecycleError::WitnessUnavailable)?;
            drop(temporary);

            fs::hard_link(&temporary_path, &final_path).map_err(|error| {
                if error.kind() == std::io::ErrorKind::AlreadyExists {
                    LifecycleError::WitnessConflict
                } else {
                    LifecycleError::WitnessUnavailable
                }
            })?;
            fs::remove_file(&temporary_path)
                .map_err(|_| LifecycleError::WitnessUnavailable)?;
            sync_directory(&self.directory)
                .map_err(|_| LifecycleError::WitnessUnavailable)?;

            let persisted = read_checkpoint(&final_path)?;
            if persisted != *checkpoint {
                return Err(LifecycleError::WitnessCheckpointInvalid);
            }
            Ok(())
        })();

        // Best-effort cleanup only. If publication succeeded but a later
        // fsync/read failed, the caller must reopen and reconcile because the
        // final generation may already be visible.
        let _ = fs::remove_file(&temporary_path);
        publish_result
    }
}

impl MonotonicWitnessStore for FileBackedWitnessStore {
    fn latest_checkpoint(&self) -> Result<WitnessCheckpoint, LifecycleError> {
        self.read_latest_checkpoint()
    }

    fn append_checkpoint(
        &mut self,
        expected_checkpoint_sha256: &str,
        next: WitnessCheckpoint,
    ) -> Result<(), LifecycleError> {
        let current = self.read_latest_checkpoint()?;
        if current.checkpoint_sha256 != expected_checkpoint_sha256
            || next.previous_checkpoint_sha256.as_deref()
                != Some(expected_checkpoint_sha256)
        {
            return Err(LifecycleError::WitnessConflict);
        }

        let expected_generation = current
            .generation
            .checked_add(1)
            .ok_or(LifecycleError::GenerationOverflow)?;
        if next.generation != expected_generation
            || !next.revoked_ids.is_superset(&current.revoked_ids)
            || !next.erased_ids.is_superset(&current.erased_ids)
        {
            return Err(LifecycleError::WitnessHistoryRegression);
        }
        validate_checkpoint(&next)?;
        self.publish_checkpoint(&next)
    }
}

fn genesis_checkpoint() -> WitnessCheckpoint {
    WitnessCheckpoint::new(
        0,
        MemorySnapshot::default().head_sha256(),
        BTreeSet::new(),
        BTreeSet::new(),
        None,
    )
}

fn validate_checkpoint(checkpoint: &WitnessCheckpoint) -> Result<(), LifecycleError> {
    if !checkpoint.verify_digest()
        || !valid_sha256(&checkpoint.store_head_sha256)
        || checkpoint
            .previous_checkpoint_sha256
            .as_deref()
            .is_some_and(|digest| !valid_sha256(digest))
        || !valid_sha256(&checkpoint.checkpoint_sha256)
        || checkpoint.revoked_ids.len() > MAX_TOMBSTONES
        || checkpoint.erased_ids.len() > MAX_TOMBSTONES
        || checkpoint
            .revoked_ids
            .iter()
            .chain(checkpoint.erased_ids.iter())
            .any(|value| !valid_field(value))
    {
        return Err(LifecycleError::WitnessCheckpointInvalid);
    }
    Ok(())
}

fn valid_field(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_FIELD_BYTES
        && !value.chars().any(char::is_control)
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn encode_checkpoint(
    checkpoint: &WitnessCheckpoint,
) -> Result<Vec<u8>, LifecycleError> {
    validate_checkpoint(checkpoint)?;
    let mut output = Vec::new();
    output.extend_from_slice(MAGIC);
    output.extend_from_slice(&checkpoint.generation.to_be_bytes());
    write_field(&mut output, &checkpoint.store_head_sha256)?;
    match &checkpoint.previous_checkpoint_sha256 {
        Some(previous) => {
            output.push(1);
            write_field(&mut output, previous)?;
        }
        None => output.push(0),
    }
    write_set(&mut output, &checkpoint.revoked_ids)?;
    write_set(&mut output, &checkpoint.erased_ids)?;
    write_field(&mut output, &checkpoint.checkpoint_sha256)?;
    if output.len() as u64 > MAX_CHECKPOINT_BYTES {
        return Err(LifecycleError::WitnessCheckpointInvalid);
    }
    Ok(output)
}

fn write_field(output: &mut Vec<u8>, value: &str) -> Result<(), LifecycleError> {
    if !valid_field(value) {
        return Err(LifecycleError::WitnessCheckpointInvalid);
    }
    let length = u32::try_from(value.len())
        .map_err(|_| LifecycleError::WitnessCheckpointInvalid)?;
    output.extend_from_slice(&length.to_be_bytes());
    output.extend_from_slice(value.as_bytes());
    Ok(())
}

fn write_set(output: &mut Vec<u8>, values: &BTreeSet<String>) -> Result<(), LifecycleError> {
    if values.len() > MAX_TOMBSTONES {
        return Err(LifecycleError::WitnessCheckpointInvalid);
    }
    let count = u32::try_from(values.len())
        .map_err(|_| LifecycleError::WitnessCheckpointInvalid)?;
    output.extend_from_slice(&count.to_be_bytes());
    for value in values {
        write_field(output, value)?;
    }
    Ok(())
}

fn read_checkpoint(path: &Path) -> Result<WitnessCheckpoint, LifecycleError> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|_| LifecycleError::WitnessUnavailable)?;
    if !metadata.file_type().is_file()
        || metadata.len() > MAX_CHECKPOINT_BYTES
        || metadata.len() < MAGIC.len() as u64
    {
        return Err(LifecycleError::WitnessCheckpointInvalid);
    }
    let file = File::open(path).map_err(|_| LifecycleError::WitnessUnavailable)?;
    let mut limited = file.take(MAX_CHECKPOINT_BYTES + 1);
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    limited
        .read_to_end(&mut bytes)
        .map_err(|_| LifecycleError::WitnessUnavailable)?;
    if bytes.len() as u64 > MAX_CHECKPOINT_BYTES {
        return Err(LifecycleError::WitnessCheckpointInvalid);
    }
    decode_checkpoint(&bytes)
}

fn decode_checkpoint(bytes: &[u8]) -> Result<WitnessCheckpoint, LifecycleError> {
    if bytes.len() as u64 > MAX_CHECKPOINT_BYTES || !bytes.starts_with(MAGIC) {
        return Err(LifecycleError::WitnessCheckpointInvalid);
    }
    let mut cursor = MAGIC.len();
    let generation = take_u64(bytes, &mut cursor)?;
    let store_head_sha256 = take_field(bytes, &mut cursor)?;
    let previous_checkpoint_sha256 = match take_byte(bytes, &mut cursor)? {
        0 => None,
        1 => Some(take_field(bytes, &mut cursor)?),
        _ => return Err(LifecycleError::WitnessCheckpointInvalid),
    };
    let revoked_ids = take_set(bytes, &mut cursor)?;
    let erased_ids = take_set(bytes, &mut cursor)?;
    let checkpoint_sha256 = take_field(bytes, &mut cursor)?;
    if cursor != bytes.len() {
        return Err(LifecycleError::WitnessCheckpointInvalid);
    }

    let checkpoint = WitnessCheckpoint {
        generation,
        store_head_sha256,
        revoked_ids,
        erased_ids,
        previous_checkpoint_sha256,
        checkpoint_sha256,
    };
    validate_checkpoint(&checkpoint)?;
    Ok(checkpoint)
}

fn take_byte(bytes: &[u8], cursor: &mut usize) -> Result<u8, LifecycleError> {
    let value = *bytes
        .get(*cursor)
        .ok_or(LifecycleError::WitnessCheckpointInvalid)?;
    *cursor += 1;
    Ok(value)
}

fn take_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32, LifecycleError> {
    let end = cursor
        .checked_add(4)
        .ok_or(LifecycleError::WitnessCheckpointInvalid)?;
    let slice = bytes
        .get(*cursor..end)
        .ok_or(LifecycleError::WitnessCheckpointInvalid)?;
    *cursor = end;
    Ok(u32::from_be_bytes(
        slice
            .try_into()
            .map_err(|_| LifecycleError::WitnessCheckpointInvalid)?,
    ))
}

fn take_u64(bytes: &[u8], cursor: &mut usize) -> Result<u64, LifecycleError> {
    let end = cursor
        .checked_add(8)
        .ok_or(LifecycleError::WitnessCheckpointInvalid)?;
    let slice = bytes
        .get(*cursor..end)
        .ok_or(LifecycleError::WitnessCheckpointInvalid)?;
    *cursor = end;
    Ok(u64::from_be_bytes(
        slice
            .try_into()
            .map_err(|_| LifecycleError::WitnessCheckpointInvalid)?,
    ))
}

fn take_field(bytes: &[u8], cursor: &mut usize) -> Result<String, LifecycleError> {
    let length = take_u32(bytes, cursor)? as usize;
    if length == 0 || length > MAX_FIELD_BYTES {
        return Err(LifecycleError::WitnessCheckpointInvalid);
    }
    let end = cursor
        .checked_add(length)
        .ok_or(LifecycleError::WitnessCheckpointInvalid)?;
    let slice = bytes
        .get(*cursor..end)
        .ok_or(LifecycleError::WitnessCheckpointInvalid)?;
    *cursor = end;
    let value = std::str::from_utf8(slice)
        .map_err(|_| LifecycleError::WitnessCheckpointInvalid)?
        .to_owned();
    if !valid_field(&value) {
        return Err(LifecycleError::WitnessCheckpointInvalid);
    }
    Ok(value)
}

fn take_set(bytes: &[u8], cursor: &mut usize) -> Result<BTreeSet<String>, LifecycleError> {
    let count = take_u32(bytes, cursor)? as usize;
    if count > MAX_TOMBSTONES {
        return Err(LifecycleError::WitnessCheckpointInvalid);
    }
    let mut values = BTreeSet::new();
    let mut previous: Option<String> = None;
    for _ in 0..count {
        let value = take_field(bytes, cursor)?;
        if previous.as_ref().is_some_and(|prior| prior >= &value) {
            return Err(LifecycleError::WitnessCheckpointInvalid);
        }
        previous = Some(value.clone());
        values.insert(value);
    }
    Ok(values)
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> std::io::Result<()> {
    File::open(path)?.sync_all()
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> std::io::Result<()> {
    // The reference adapter cannot promise directory-entry durability on every
    // platform. A production implementation must use the platform's documented
    // durable-rename/metadata-flush primitive and verify crash semantics.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temporary_directory(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "mace-file-witness-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("create fixture directory");
        path
    }

    fn next_checkpoint(
        current: &WitnessCheckpoint,
        store_head: &str,
        revoked: BTreeSet<String>,
        erased: BTreeSet<String>,
    ) -> WitnessCheckpoint {
        WitnessCheckpoint::new(
            current.generation + 1,
            store_head.to_owned(),
            revoked,
            erased,
            Some(current.checkpoint_sha256.clone()),
        )
    }

    #[derive(Default)]
    struct FixtureAuthority;

    impl crate::IdentityAuthority for FixtureAuthority {
        fn profile_for(&self, issuer_id: &str) -> Option<crate::AuthorityProfile> {
            (issuer_id == "issuer").then(|| crate::AuthorityProfile {
                allowed_scopes: BTreeSet::from(["project-x".to_owned()]),
                allowed_purposes: BTreeSet::from(["planning".to_owned()]),
                may_assert: true,
                may_derive: true,
                may_revoke: true,
                may_erase: true,
            })
        }
    }

    fn copy_directory(source: &Path, destination: &Path) {
        fs::create_dir_all(destination).expect("create destination directory");
        for entry in fs::read_dir(source).expect("read source directory") {
            let entry = entry.expect("source entry");
            let source_path = entry.path();
            let destination_path = destination.join(entry.file_name());
            if entry.file_type().expect("source entry type").is_dir() {
                copy_directory(&source_path, &destination_path);
            } else {
                fs::copy(&source_path, &destination_path).expect("copy checkpoint file");
            }
        }
    }

    #[test]
    fn file_witness_reconciles_t10_when_only_memory_backup_is_stale_and_demonstrates_joint_rollback_limit() {
        let root = temporary_directory("t10-reconciliation");
        let witness_directory = root.join("witness");
        let mut witness =
            FileBackedWitnessStore::open(&witness_directory).expect("open witness");
        let authority = FixtureAuthority;
        let mut current = MemorySnapshot::default();

        crate::commit_record(
            &mut current,
            &mut witness,
            crate::MemoryRecord::new(
                "root",
                "protected source record",
                "issuer",
                "project-x",
                "planning",
                Vec::new(),
            ),
            &authority,
        )
        .expect("commit protected record");

        let pre_erasure_backup = current.clone();
        let rolled_back_witness_directory = root.join("witness-backup-before-erasure");
        copy_directory(&witness_directory, &rolled_back_witness_directory);

        let erased = crate::erase_record(
            &mut current,
            &mut witness,
            &authority,
            "issuer",
            "root",
        )
        .expect("erase protected record");
        assert_eq!(erased.invalidated_ids, vec!["root".to_owned()]);
        assert!(!current.contains("root"));

        drop(witness);
        let current_witness =
            FileBackedWitnessStore::open(&witness_directory).expect("reopen current witness");
        let (recovered, report) = crate::reconcile_snapshot(
            pre_erasure_backup.clone(),
            &current_witness,
        )
        .expect("T10 stale data backup reconciles against current witness");
        assert!(!recovered.contains("root"));
        assert_eq!(report.removed_resurrected_ids, vec!["root".to_owned()]);
        assert_eq!(recovered.head_sha256(), current.head_sha256());
        assert_eq!(report.recovered_head_sha256, current.head_sha256());

        // This is the explicit limit of a same-domain local witness: if both
        // data and witness are restored to the pre-erasure point, no local
        // history remains that can prove the erasure happened.
        let rolled_back_witness = FileBackedWitnessStore::open(&rolled_back_witness_directory)
            .expect("open jointly rolled-back witness");
        let (jointly_rolled_back, _) = crate::reconcile_snapshot(
            pre_erasure_backup,
            &rolled_back_witness,
        )
        .expect("old data and old local witness are mutually consistent");
        assert!(jointly_rolled_back.contains("root"));

        fs::remove_dir_all(root).expect("cleanup");
    }


    #[cfg(unix)]
    #[test]
    fn witness_directory_and_checkpoint_files_are_owner_only() {
        use std::os::unix::fs::PermissionsExt;

        let root = temporary_directory("permissions");
        let store = FileBackedWitnessStore::open(&root).expect("open");
        let directory_mode = fs::metadata(store.directory())
            .expect("directory metadata")
            .permissions()
            .mode()
            & 0o777;
        let checkpoint_mode = fs::metadata(store.checkpoint_path(0))
            .expect("checkpoint metadata")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(directory_mode, 0o700);
        assert_eq!(checkpoint_mode, 0o600);
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn checkpoint_history_survives_reopen_and_preserves_tombstones() {
        let root = temporary_directory("reopen");
        let mut store = FileBackedWitnessStore::open(&root).expect("open witness");
        let genesis = store.latest_checkpoint().expect("genesis");
        let revoked = BTreeSet::from(["record-root".to_owned()]);
        let erased = BTreeSet::from(["record-derived".to_owned()]);
        let next = next_checkpoint(
            &genesis,
            &"a".repeat(64),
            revoked.clone(),
            erased.clone(),
        );
        store
            .append_checkpoint(&genesis.checkpoint_sha256, next.clone())
            .expect("append checkpoint");

        let reopened = FileBackedWitnessStore::open(&root).expect("reopen witness");
        assert_eq!(reopened.latest_checkpoint().expect("latest"), next);
        assert_eq!(reopened.latest_checkpoint().expect("latest").revoked_ids, revoked);
        assert_eq!(reopened.latest_checkpoint().expect("latest").erased_ids, erased);
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn stale_writer_cannot_overwrite_a_published_generation() {
        let root = temporary_directory("stale-writer");
        let mut first = FileBackedWitnessStore::open(&root).expect("first open");
        let mut second = FileBackedWitnessStore::open(&root).expect("second open");
        let first_genesis = first.latest_checkpoint().expect("first genesis");
        let second_genesis = second.latest_checkpoint().expect("second genesis");
        let first_next = next_checkpoint(
            &first_genesis,
            &"b".repeat(64),
            BTreeSet::new(),
            BTreeSet::new(),
        );
        let second_next = next_checkpoint(
            &second_genesis,
            &"c".repeat(64),
            BTreeSet::new(),
            BTreeSet::new(),
        );
        first
            .append_checkpoint(&first_genesis.checkpoint_sha256, first_next.clone())
            .expect("first append");
        assert_eq!(
            second.append_checkpoint(&second_genesis.checkpoint_sha256, second_next),
            Err(LifecycleError::WitnessConflict)
        );
        assert_eq!(first.latest_checkpoint().expect("latest"), first_next);
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn concurrent_writers_publish_at_most_one_checkpoint_per_generation() {
        let root = temporary_directory("concurrent-writers");
        let genesis = FileBackedWitnessStore::open(&root)
            .expect("genesis open")
            .latest_checkpoint()
            .expect("genesis");
        let barrier = Arc::new(Barrier::new(2));
        let mut handles = Vec::new();

        for marker in ["e", "f"] {
            let root = root.clone();
            let barrier = Arc::clone(&barrier);
            let expected = genesis.checkpoint_sha256.clone();
            let previous = genesis.checkpoint_sha256.clone();
            handles.push(thread::spawn(move || {
                let mut store = FileBackedWitnessStore::open(&root).expect("thread open");
                let next = WitnessCheckpoint::new(
                    1,
                    marker.repeat(64),
                    BTreeSet::new(),
                    BTreeSet::new(),
                    Some(previous),
                );
                barrier.wait();
                store.append_checkpoint(&expected, next)
            }));
        }

        let outcomes = handles
            .into_iter()
            .map(|handle| handle.join().expect("writer thread"))
            .collect::<Vec<_>>();
        assert_eq!(outcomes.iter().filter(|result| result.is_ok()).count(), 1);
        assert_eq!(
            outcomes
                .iter()
                .filter(|result| matches!(result, &&Err(LifecycleError::WitnessConflict)))
                .count(),
            1
        );
        let latest = FileBackedWitnessStore::open(&root)
            .expect("reopen")
            .latest_checkpoint()
            .expect("latest");
        assert_eq!(latest.generation, 1);
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn append_cannot_remove_previously_committed_tombstones() {
        let root = temporary_directory("tombstone-regression");
        let mut store = FileBackedWitnessStore::open(&root).expect("open");
        let genesis = store.latest_checkpoint().expect("genesis");
        let revoked = BTreeSet::from(["revoked-root".to_owned()]);
        let first = next_checkpoint(
            &genesis,
            &"1".repeat(64),
            revoked,
            BTreeSet::new(),
        );
        store
            .append_checkpoint(&genesis.checkpoint_sha256, first.clone())
            .expect("first append");

        let regression = next_checkpoint(
            &first,
            &"2".repeat(64),
            BTreeSet::new(),
            BTreeSet::new(),
        );
        assert_eq!(
            store.append_checkpoint(&first.checkpoint_sha256, regression),
            Err(LifecycleError::WitnessHistoryRegression)
        );
        assert_eq!(store.latest_checkpoint().expect("latest"), first);
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn malformed_final_checkpoint_name_fails_closed() {
        let root = temporary_directory("malformed-name");
        let store = FileBackedWitnessStore::open(&root).expect("open");
        fs::write(root.join("checkpoint-2.macew"), b"not a checkpoint")
            .expect("malformed final file");
        assert_eq!(
            store.latest_checkpoint(),
            Err(LifecycleError::WitnessCheckpointInvalid)
        );
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn tampered_checkpoint_is_rejected_on_reopen() {
        let root = temporary_directory("tamper");
        let store = FileBackedWitnessStore::open(&root).expect("open witness");
        let genesis_path = store.checkpoint_path(0);
        let mut bytes = fs::read(&genesis_path).expect("read genesis");
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;
        fs::write(&genesis_path, bytes).expect("tamper genesis");
        assert_eq!(
            FileBackedWitnessStore::open(&root).map(|_| ()),
            Err(LifecycleError::WitnessCheckpointInvalid)
        );
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn missing_generation_is_a_history_regression_not_a_clean_store() {
        let root = temporary_directory("gap");
        let mut store = FileBackedWitnessStore::open(&root).expect("open witness");
        let genesis = store.latest_checkpoint().expect("genesis");
        let next = next_checkpoint(
            &genesis,
            &"d".repeat(64),
            BTreeSet::new(),
            BTreeSet::new(),
        );
        store
            .append_checkpoint(&genesis.checkpoint_sha256, next)
            .expect("append");
        fs::remove_file(store.checkpoint_path(0)).expect("remove genesis");
        assert_eq!(
            store.latest_checkpoint(),
            Err(LifecycleError::WitnessHistoryRegression)
        );
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn malformed_pending_file_is_not_a_published_checkpoint() {
        let root = temporary_directory("pending");
        let store = FileBackedWitnessStore::open(&root).expect("open");
        fs::write(root.join(".pending-interrupted"), b"partial").expect("pending");
        assert!(store.latest_checkpoint().is_ok());
        fs::remove_dir_all(root).expect("cleanup");
    }
}
