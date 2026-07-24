// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Checksummed write-intent journal used by multi-record batches.

use std::fs::{File, OpenOptions};
use std::io::{ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use memmap2::MmapMut;

use crate::HdcStoreError;
use crate::batch::{BatchRecoveryDisposition, BatchRecoveryReport};
use crate::checksum::crc64_ecma;
use crate::header::{ENTRY_SIZE, STATUS_LIVE, STATUS_TOMBSTONE, StoreHeader};

const JOURNAL_MAGIC: [u8; 8] = *b"HDCBTX01";
const JOURNAL_VERSION: u32 = 1;
const JOURNAL_HEADER_SIZE: usize = 96;
const JOURNAL_RECORD_SIZE: usize = 24;
const HEADER_CHECKSUM_OFFSET: usize = 80;
const HEADER_CHECKSUM_END: usize = 88;
const RECORD_APPEND: u8 = 1;
const RECORD_DELETE: u8 = 2;
static JOURNAL_TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JournalRecordKind {
    Append,
    Delete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct JournalRecord {
    pub kind: JournalRecordKind,
    pub id: u64,
    pub index: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BatchJournal {
    pub base_generation: u64,
    pub target_generation: u64,
    pub base_vector_count: u64,
    pub base_live_count: u64,
    pub base_tombstone_count: u64,
    pub append_count: u64,
    pub delete_count: u64,
    pub records: Vec<JournalRecord>,
}

impl BatchJournal {
    pub fn expected_target_counts(&self) -> Result<(u64, u64, u64), HdcStoreError> {
        let vector_count = self
            .base_vector_count
            .checked_add(self.append_count)
            .ok_or(HdcStoreError::ArithmeticOverflow {
                context: "batch target vector_count",
            })?;
        let live_with_appends = self.base_live_count.checked_add(self.append_count).ok_or(
            HdcStoreError::ArithmeticOverflow {
                context: "batch target live_count append",
            },
        )?;
        let live_count = live_with_appends
            .checked_sub(self.delete_count)
            .ok_or_else(|| {
                journal_error(
                    Path::new(""),
                    "delete_count exceeds base live entries plus appends",
                )
            })?;
        let tombstone_count = self
            .base_tombstone_count
            .checked_add(self.delete_count)
            .ok_or(HdcStoreError::ArithmeticOverflow {
                context: "batch target tombstone_count",
            })?;
        Ok((vector_count, live_count, tombstone_count))
    }

    fn validate(&self, path: &Path) -> Result<(), HdcStoreError> {
        if self.target_generation
            != self
                .base_generation
                .checked_add(1)
                .ok_or(HdcStoreError::ArithmeticOverflow {
                    context: "batch target generation",
                })?
        {
            return Err(journal_error(path, "target generation is not base + 1"));
        }
        let total_records = self.append_count.checked_add(self.delete_count).ok_or(
            HdcStoreError::ArithmeticOverflow {
                context: "batch journal record count",
            },
        )?;
        if usize::try_from(total_records).ok() != Some(self.records.len()) {
            return Err(journal_error(
                path,
                format!(
                    "record count mismatch: declared {total_records}, decoded {}",
                    self.records.len()
                ),
            ));
        }
        let committed = self
            .base_live_count
            .checked_add(self.base_tombstone_count)
            .ok_or(HdcStoreError::ArithmeticOverflow {
                context: "batch base count invariant",
            })?;
        if committed != self.base_vector_count {
            return Err(journal_error(path, "base count invariant is invalid"));
        }
        self.expected_target_counts()
            .map_err(|error| journal_error(path, error.to_string()))?;

        let mut append_seen = 0u64;
        let mut delete_seen = 0u64;
        let mut previous_append_index = None;
        let mut previous_delete_index = None;
        let mut ids = std::collections::HashSet::new();
        for record in &self.records {
            if !ids.insert(record.id) {
                return Err(journal_error(
                    path,
                    format!("duplicate id {} in journal", record.id),
                ));
            }
            match record.kind {
                JournalRecordKind::Append => {
                    let expected = self.base_vector_count.checked_add(append_seen).ok_or(
                        HdcStoreError::ArithmeticOverflow {
                            context: "batch append journal index",
                        },
                    )?;
                    if record.index != expected {
                        return Err(journal_error(
                            path,
                            format!(
                                "append index {} is not expected contiguous index {expected}",
                                record.index
                            ),
                        ));
                    }
                    if let Some(previous) = previous_append_index {
                        if record.index <= previous {
                            return Err(journal_error(path, "append indexes are not increasing"));
                        }
                    }
                    previous_append_index = Some(record.index);
                    append_seen += 1;
                }
                JournalRecordKind::Delete => {
                    if record.index >= self.base_vector_count {
                        return Err(journal_error(
                            path,
                            format!(
                                "delete index {} exceeds base committed region",
                                record.index
                            ),
                        ));
                    }
                    if let Some(previous) = previous_delete_index {
                        if record.index <= previous {
                            return Err(journal_error(path, "delete indexes are not increasing"));
                        }
                    }
                    previous_delete_index = Some(record.index);
                    delete_seen += 1;
                }
            }
        }
        if append_seen != self.append_count || delete_seen != self.delete_count {
            return Err(journal_error(
                path,
                "record-kind counts do not match header",
            ));
        }
        Ok(())
    }
}

/// Deterministic sidecar path for an in-progress write batch.
pub fn batch_journal_path(store_path: impl AsRef<Path>) -> PathBuf {
    let mut path = store_path.as_ref().as_os_str().to_os_string();
    path.push(".txn");
    PathBuf::from(path)
}

pub(crate) fn write_batch_journal(
    store_path: &Path,
    journal: &BatchJournal,
) -> Result<PathBuf, HdcStoreError> {
    let final_path = batch_journal_path(store_path);
    if final_path.exists() {
        return Err(HdcStoreError::PendingBatchTransaction { path: final_path });
    }
    journal.validate(&final_path)?;
    let payload = encode_records(&journal.records);
    let payload_checksum = crc64_ecma(&payload);
    let header = encode_header(journal, payload_checksum)?;
    let (temp_path, mut temp_file) = create_temp_file(store_path)?;
    let mut cleanup = TempPathGuard::new(temp_path.clone());
    temp_file.write_all(&header)?;
    temp_file.write_all(&payload)?;
    temp_file.sync_all()?;
    drop(temp_file);
    atomic_replace_without_overwrite(&temp_path, &final_path)?;
    cleanup.disarm();
    sync_parent_directory(&final_path)?;
    Ok(final_path)
}

pub(crate) fn load_batch_journal(store_path: &Path) -> Result<Option<BatchJournal>, HdcStoreError> {
    let path = batch_journal_path(store_path);
    let mut file = match File::open(&path) {
        Ok(file) => file,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let file_len =
        usize::try_from(file.metadata()?.len()).map_err(|_| HdcStoreError::ArithmeticOverflow {
            context: "batch journal file length",
        })?;
    if file_len < JOURNAL_HEADER_SIZE {
        return Err(journal_error(
            &path,
            format!("journal is only {file_len} bytes"),
        ));
    }
    let mut header = [0u8; JOURNAL_HEADER_SIZE];
    file.read_exact(&mut header)?;
    let (mut journal, payload_checksum) = decode_header(&path, &header)?;
    let record_count = journal
        .append_count
        .checked_add(journal.delete_count)
        .ok_or(HdcStoreError::ArithmeticOverflow {
            context: "batch journal decoded record count",
        })?;
    let payload_len = usize::try_from(record_count)
        .map_err(|_| HdcStoreError::ArithmeticOverflow {
            context: "batch journal record count conversion",
        })?
        .checked_mul(JOURNAL_RECORD_SIZE)
        .ok_or(HdcStoreError::ArithmeticOverflow {
            context: "batch journal payload length",
        })?;
    let expected_len =
        JOURNAL_HEADER_SIZE
            .checked_add(payload_len)
            .ok_or(HdcStoreError::ArithmeticOverflow {
                context: "batch journal total length",
            })?;
    if file_len != expected_len {
        return Err(journal_error(
            &path,
            format!("expected {expected_len} bytes, found {file_len}"),
        ));
    }
    let mut payload = vec![0u8; payload_len];
    file.read_exact(&mut payload)?;
    let found_payload_checksum = crc64_ecma(&payload);
    if found_payload_checksum != payload_checksum {
        return Err(journal_error(
            &path,
            format!(
                "payload checksum mismatch: expected {payload_checksum:#018x}, found {found_payload_checksum:#018x}"
            ),
        ));
    }
    journal.records = decode_records(&path, &payload)?;
    journal.validate(&path)?;
    Ok(Some(journal))
}

pub(crate) fn recover_batch_journal(
    store_path: &Path,
    file: &File,
    mmap: &mut MmapMut,
    selected_header: &StoreHeader,
) -> Result<Option<BatchRecoveryReport>, HdcStoreError> {
    let Some(journal) = load_batch_journal(store_path)? else {
        return Ok(None);
    };
    let path = batch_journal_path(store_path);
    let (target_vectors, target_live, target_tombstones) = journal.expected_target_counts()?;

    let disposition = if selected_header.generation == journal.base_generation {
        if selected_header.vector_count != journal.base_vector_count
            || selected_header.live_count != journal.base_live_count
            || selected_header.tombstone_count != journal.base_tombstone_count
        {
            return Err(journal_error(
                &path,
                "selected base-generation header does not match journal base counts",
            ));
        }
        rollback_records(mmap, &journal, &path)?;
        mmap.flush()?;
        file.sync_data()?;
        BatchRecoveryDisposition::RolledBack
    } else if selected_header.generation == journal.target_generation {
        if selected_header.vector_count != target_vectors
            || selected_header.live_count != target_live
            || selected_header.tombstone_count != target_tombstones
        {
            return Err(journal_error(
                &path,
                "selected target-generation header does not match journal target counts",
            ));
        }
        validate_committed_records(mmap, &journal, &path)?;
        BatchRecoveryDisposition::FinalizedCommitted
    } else {
        return Err(journal_error(
            &path,
            format!(
                "selected generation {} is neither journal base {} nor target {}",
                selected_header.generation, journal.base_generation, journal.target_generation
            ),
        ));
    };

    remove_batch_journal(store_path)?;
    Ok(Some(BatchRecoveryReport {
        base_generation: journal.base_generation,
        target_generation: journal.target_generation,
        appended: journal.append_count,
        deleted: journal.delete_count,
        disposition,
    }))
}

pub(crate) fn remove_batch_journal(store_path: &Path) -> Result<(), HdcStoreError> {
    let path = batch_journal_path(store_path);
    match std::fs::remove_file(&path) {
        Ok(()) => sync_parent_directory(&path),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn rollback_records(
    mmap: &mut MmapMut,
    journal: &BatchJournal,
    path: &Path,
) -> Result<(), HdcStoreError> {
    for record in &journal.records {
        let offset = checked_entry_offset(record.index)?;
        let end = offset
            .checked_add(ENTRY_SIZE)
            .ok_or(HdcStoreError::ArithmeticOverflow {
                context: "batch rollback entry end",
            })?;
        if end > mmap.len() {
            return Err(journal_error(
                path,
                "journal record extends beyond store file",
            ));
        }
        match record.kind {
            JournalRecordKind::Append => {
                if mmap[offset] != 0 {
                    validate_record_id(mmap, offset, record, path)?;
                }
                mmap[offset..end].fill(0);
            }
            JournalRecordKind::Delete => {
                validate_record_id(mmap, offset, record, path)?;
                match mmap[offset] {
                    STATUS_LIVE | STATUS_TOMBSTONE => mmap[offset] = STATUS_LIVE,
                    status => {
                        return Err(journal_error(
                            path,
                            format!(
                                "delete rollback index {} has invalid status {status}",
                                record.index
                            ),
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

fn validate_committed_records(
    mmap: &MmapMut,
    journal: &BatchJournal,
    path: &Path,
) -> Result<(), HdcStoreError> {
    for record in &journal.records {
        let offset = checked_entry_offset(record.index)?;
        let end = offset
            .checked_add(ENTRY_SIZE)
            .ok_or(HdcStoreError::ArithmeticOverflow {
                context: "batch committed entry end",
            })?;
        if end > mmap.len() {
            return Err(journal_error(path, "committed journal record exceeds file"));
        }
        validate_record_id(mmap, offset, record, path)?;
        let expected = match record.kind {
            JournalRecordKind::Append => STATUS_LIVE,
            JournalRecordKind::Delete => STATUS_TOMBSTONE,
        };
        if mmap[offset] != expected {
            return Err(journal_error(
                path,
                format!(
                    "record for id {} at index {} expected status {expected}, found {}",
                    record.id, record.index, mmap[offset]
                ),
            ));
        }
    }
    Ok(())
}

fn validate_record_id(
    mmap: &[u8],
    offset: usize,
    record: &JournalRecord,
    path: &Path,
) -> Result<(), HdcStoreError> {
    let end = offset
        .checked_add(9)
        .ok_or(HdcStoreError::ArithmeticOverflow {
            context: "batch journal id end",
        })?;
    let bytes: [u8; 8] = mmap
        .get(offset + 1..end)
        .ok_or_else(|| journal_error(path, "journal id read extends beyond file"))?
        .try_into()
        .expect("journal id slice is eight bytes");
    let found = u64::from_le_bytes(bytes);
    if found != record.id {
        return Err(journal_error(
            path,
            format!(
                "journal record id {} disagrees with on-disk id {found} at index {}",
                record.id, record.index
            ),
        ));
    }
    Ok(())
}

fn checked_entry_offset(index: u64) -> Result<usize, HdcStoreError> {
    crate::header::checked_entry_offset(crate::header::DATA_OFFSET, index)
}

fn encode_records(records: &[JournalRecord]) -> Vec<u8> {
    let mut payload = Vec::with_capacity(records.len().saturating_mul(JOURNAL_RECORD_SIZE));
    for record in records {
        let mut bytes = [0u8; JOURNAL_RECORD_SIZE];
        bytes[0] = match record.kind {
            JournalRecordKind::Append => RECORD_APPEND,
            JournalRecordKind::Delete => RECORD_DELETE,
        };
        bytes[8..16].copy_from_slice(&record.id.to_le_bytes());
        bytes[16..24].copy_from_slice(&record.index.to_le_bytes());
        payload.extend_from_slice(&bytes);
    }
    payload
}

fn decode_records(path: &Path, payload: &[u8]) -> Result<Vec<JournalRecord>, HdcStoreError> {
    let mut records = Vec::with_capacity(payload.len() / JOURNAL_RECORD_SIZE);
    for bytes in payload.chunks_exact(JOURNAL_RECORD_SIZE) {
        if bytes[1..8].iter().any(|byte| *byte != 0) {
            return Err(journal_error(
                path,
                "journal record reserved bytes are non-zero",
            ));
        }
        let kind = match bytes[0] {
            RECORD_APPEND => JournalRecordKind::Append,
            RECORD_DELETE => JournalRecordKind::Delete,
            tag => {
                return Err(journal_error(
                    path,
                    format!("unknown journal record tag {tag}"),
                ));
            }
        };
        records.push(JournalRecord {
            kind,
            id: u64::from_le_bytes(bytes[8..16].try_into().expect("fixed record id")),
            index: u64::from_le_bytes(bytes[16..24].try_into().expect("fixed record index")),
        });
    }
    Ok(records)
}

fn encode_header(
    journal: &BatchJournal,
    payload_checksum: u64,
) -> Result<[u8; JOURNAL_HEADER_SIZE], HdcStoreError> {
    let mut bytes = [0u8; JOURNAL_HEADER_SIZE];
    bytes[0..8].copy_from_slice(&JOURNAL_MAGIC);
    bytes[8..12].copy_from_slice(&JOURNAL_VERSION.to_le_bytes());
    bytes[16..24].copy_from_slice(&journal.base_generation.to_le_bytes());
    bytes[24..32].copy_from_slice(&journal.target_generation.to_le_bytes());
    bytes[32..40].copy_from_slice(&journal.base_vector_count.to_le_bytes());
    bytes[40..48].copy_from_slice(&journal.base_live_count.to_le_bytes());
    bytes[48..56].copy_from_slice(&journal.base_tombstone_count.to_le_bytes());
    bytes[56..64].copy_from_slice(&journal.append_count.to_le_bytes());
    bytes[64..72].copy_from_slice(&journal.delete_count.to_le_bytes());
    bytes[72..80].copy_from_slice(&payload_checksum.to_le_bytes());
    let checksum = crc64_ecma(&bytes);
    bytes[HEADER_CHECKSUM_OFFSET..HEADER_CHECKSUM_END].copy_from_slice(&checksum.to_le_bytes());
    Ok(bytes)
}

fn decode_header(
    path: &Path,
    bytes: &[u8; JOURNAL_HEADER_SIZE],
) -> Result<(BatchJournal, u64), HdcStoreError> {
    if bytes[0..8] != JOURNAL_MAGIC {
        return Err(journal_error(path, "bad journal magic bytes"));
    }
    let version = read_u32(bytes, 8);
    if version != JOURNAL_VERSION {
        return Err(journal_error(
            path,
            format!("journal version mismatch: expected {JOURNAL_VERSION}, found {version}"),
        ));
    }
    if read_u32(bytes, 12) != 0 || bytes[88..96].iter().any(|byte| *byte != 0) {
        return Err(journal_error(path, "journal reserved fields are non-zero"));
    }
    let found_checksum = read_u64(bytes, HEADER_CHECKSUM_OFFSET);
    let mut checksum_bytes = *bytes;
    checksum_bytes[HEADER_CHECKSUM_OFFSET..HEADER_CHECKSUM_END].fill(0);
    let expected_checksum = crc64_ecma(&checksum_bytes);
    if found_checksum != expected_checksum {
        return Err(journal_error(
            path,
            format!(
                "header checksum mismatch: expected {expected_checksum:#018x}, found {found_checksum:#018x}"
            ),
        ));
    }
    Ok((
        BatchJournal {
            base_generation: read_u64(bytes, 16),
            target_generation: read_u64(bytes, 24),
            base_vector_count: read_u64(bytes, 32),
            base_live_count: read_u64(bytes, 40),
            base_tombstone_count: read_u64(bytes, 48),
            append_count: read_u64(bytes, 56),
            delete_count: read_u64(bytes, 64),
            records: Vec::new(),
        },
        read_u64(bytes, 72),
    ))
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("fixed u32 field"),
    )
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("fixed u64 field"),
    )
}

fn create_temp_file(store_path: &Path) -> Result<(PathBuf, File), HdcStoreError> {
    let parent = store_path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = store_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("hdc-store");
    for _ in 0..128 {
        let sequence = JOURNAL_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = parent.join(format!(
            ".{file_name}.txn-{}-{sequence}.tmp",
            std::process::id()
        ));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => return Ok((path, file)),
            Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Err(HdcStoreError::InvalidBatchJournal {
        path: batch_journal_path(store_path),
        reason: "could not allocate a unique journal staging path".into(),
    })
}

fn atomic_replace_without_overwrite(
    source: &Path,
    destination: &Path,
) -> Result<(), HdcStoreError> {
    if destination.exists() {
        return Err(HdcStoreError::PendingBatchTransaction {
            path: destination.to_path_buf(),
        });
    }
    std::fs::rename(source, destination)?;
    Ok(())
}

fn sync_parent_directory(path: &Path) -> Result<(), HdcStoreError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    File::open(parent)?.sync_all()?;
    Ok(())
}

fn journal_error(path: &Path, reason: impl Into<String>) -> HdcStoreError {
    HdcStoreError::InvalidBatchJournal {
        path: path.to_path_buf(),
        reason: reason.into(),
    }
}

struct TempPathGuard {
    path: PathBuf,
    armed: bool,
}

impl TempPathGuard {
    fn new(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for TempPathGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}
