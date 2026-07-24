// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Compact, deterministic, live-only export and restore archives.

use std::fs::{File, OpenOptions};
use std::io::{ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use symthaea_core::hdc::BinaryHV;

use crate::checksum::{Crc64Ecma, crc64_ecma};
use crate::content_checksum::ContentChecksumBuilder;
use crate::locking::{StoreLock, store_lock_path};
use crate::{
    HdcStore, HdcStoreError, HdcStoreReader, StoreConfig, StoreContentChecksum, WriteBatch,
    batch_journal_path, lsh_snapshot_path,
};

const ARCHIVE_MAGIC: [u8; 8] = *b"HDCARCH1";
const ARCHIVE_VERSION: u32 = 1;
const ARCHIVE_HEADER_SIZE: usize = 128;
const ARCHIVE_RECORD_SIZE: usize = 8 + 2048;
const HEADER_CHECKSUM_OFFSET: usize = 64;
const HEADER_CHECKSUM_END: usize = 72;
static ARCHIVE_TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);
const RESTORE_BATCH_RECORDS: usize = 1024;

/// Resource bounds applied before archive payload processing or restore allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PortableArchiveLimits {
    pub max_records: u64,
    pub max_archive_bytes: u64,
}

impl Default for PortableArchiveLimits {
    fn default() -> Self {
        Self {
            max_records: 100_000_000,
            max_archive_bytes: 256 * 1024 * 1024 * 1024,
        }
    }
}

impl PortableArchiveLimits {
    fn check(
        self,
        path: &Path,
        record_count: u64,
        archive_bytes: u64,
    ) -> Result<(), HdcStoreError> {
        if record_count > self.max_records {
            return Err(HdcStoreError::ArchiveLimitExceeded {
                path: path.to_path_buf(),
                reason: format!(
                    "record count {record_count} exceeds configured maximum {}",
                    self.max_records
                ),
            });
        }
        if archive_bytes > self.max_archive_bytes {
            return Err(HdcStoreError::ArchiveLimitExceeded {
                path: path.to_path_buf(),
                reason: format!(
                    "archive length {archive_bytes} exceeds configured maximum {}",
                    self.max_archive_bytes
                ),
            });
        }
        Ok(())
    }
}

/// Validated portable archive metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PortableArchiveMetadata {
    pub version: u32,
    pub source_generation: u64,
    pub record_count: u64,
    pub lsh_bands: u32,
    pub lsh_rows: u32,
    pub content_checksum: StoreContentChecksum,
    pub payload_crc64_ecma: u64,
}

/// Result of atomically exporting one committed store generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortableExportReport {
    pub source: PathBuf,
    pub destination: PathBuf,
    pub metadata: PortableArchiveMetadata,
    pub bytes_written: u64,
}

/// Result of validating and restoring a portable archive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortableRestoreReport {
    pub source: PathBuf,
    pub destination: PathBuf,
    pub metadata: PortableArchiveMetadata,
    pub destination_generation: u64,
    pub destination_checksum: StoreContentChecksum,
}

/// Export a compact live-only archive without replacing an existing path.
pub fn export_portable_archive(
    store_path: impl AsRef<Path>,
    archive_path: impl AsRef<Path>,
) -> Result<PortableExportReport, HdcStoreError> {
    export_portable_archive_with_limits(store_path, archive_path, PortableArchiveLimits::default())
}

/// Export with explicit record-count and archive-size bounds.
pub fn export_portable_archive_with_limits(
    store_path: impl AsRef<Path>,
    archive_path: impl AsRef<Path>,
    limits: PortableArchiveLimits,
) -> Result<PortableExportReport, HdcStoreError> {
    let source = store_path.as_ref().to_path_buf();
    let destination = archive_path.as_ref().to_path_buf();
    if destination.exists() {
        return Err(HdcStoreError::ArchiveDestinationExists { path: destination });
    }

    let reader = HdcStoreReader::open(&source)?;
    let expected_bytes = expected_archive_len(reader.live_count())?;
    limits.check(&source, reader.live_count(), expected_bytes)?;
    let content_checksum = reader.content_checksum();
    let (temp_path, mut file) = create_temp_file(&destination, "export")?;
    let mut cleanup = TempPathGuard::new(temp_path.clone());
    file.write_all(&[0u8; ARCHIVE_HEADER_SIZE])?;

    let mut payload_crc = Crc64Ecma::new();
    for (id, hv) in reader.iter() {
        let id_bytes = id.to_le_bytes();
        file.write_all(&id_bytes)?;
        file.write_all(&hv.0)?;
        payload_crc.update(&id_bytes);
        payload_crc.update(&hv.0);
    }
    let payload_crc64_ecma = payload_crc.finalize();
    let metadata = PortableArchiveMetadata {
        version: ARCHIVE_VERSION,
        source_generation: reader.generation(),
        record_count: reader.live_count(),
        lsh_bands: reader.lsh_bands(),
        lsh_rows: reader.lsh_rows(),
        content_checksum,
        payload_crc64_ecma,
    };
    let header = encode_header(metadata);
    file.seek(SeekFrom::Start(0))?;
    file.write_all(&header)?;
    file.sync_all()?;
    let bytes_written = file.metadata()?.len();
    drop(file);

    install_without_overwrite(&temp_path, &destination)?;
    cleanup.disarm();
    sync_parent_directory(&destination)?;
    Ok(PortableExportReport {
        source,
        destination,
        metadata,
        bytes_written,
    })
}

/// Validate archive structure, ordering, checksums, and logical content.
pub fn inspect_portable_archive(
    archive_path: impl AsRef<Path>,
) -> Result<PortableArchiveMetadata, HdcStoreError> {
    inspect_portable_archive_with_limits(archive_path, PortableArchiveLimits::default())
}

/// Validate an archive under explicit resource limits.
pub fn inspect_portable_archive_with_limits(
    archive_path: impl AsRef<Path>,
    limits: PortableArchiveLimits,
) -> Result<PortableArchiveMetadata, HdcStoreError> {
    validate_archive(archive_path.as_ref(), limits)
}

/// Restore a validated archive through a synchronized same-directory staging store.
///
/// The destination is never overwritten. The complete archive is validated
/// before staging begins, and the reconstructed logical checksum is verified
/// again before the destination path is published.
pub fn restore_portable_archive(
    archive_path: impl AsRef<Path>,
    store_path: impl AsRef<Path>,
) -> Result<(HdcStore, PortableRestoreReport), HdcStoreError> {
    restore_portable_archive_with_limits(archive_path, store_path, PortableArchiveLimits::default())
}

/// Restore under explicit record-count and archive-size bounds.
pub fn restore_portable_archive_with_limits(
    archive_path: impl AsRef<Path>,
    store_path: impl AsRef<Path>,
    limits: PortableArchiveLimits,
) -> Result<(HdcStore, PortableRestoreReport), HdcStoreError> {
    let source = archive_path.as_ref().to_path_buf();
    let destination = store_path.as_ref().to_path_buf();
    let metadata = validate_archive(&source, limits)?;
    let coordination_lock = StoreLock::exclusive(&destination)?;
    if destination.exists() {
        return Err(HdcStoreError::ArchiveDestinationExists { path: destination });
    }

    let initial_capacity = metadata.record_count.max(64);
    let config = StoreConfig {
        initial_capacity,
        lsh_bands: metadata.lsh_bands,
        lsh_rows: metadata.lsh_rows,
    };
    let (temp_path, mut staged) = create_staging_store(&destination, config)?;
    let mut cleanup = TempPathGuard::new(temp_path.clone());
    let mut archive = File::open(&source)?;
    archive.seek(SeekFrom::Start(ARCHIVE_HEADER_SIZE as u64))?;
    let mut batch = WriteBatch::new();
    for _ in 0..metadata.record_count {
        let (id, hv) = read_record(&mut archive, &source)?;
        batch.push_append(id, hv);
        if batch.len() == RESTORE_BATCH_RECORDS {
            staged.apply_batch(std::mem::take(&mut batch))?;
        }
    }
    if !batch.is_empty() {
        staged.apply_batch(batch)?;
    }
    staged.sync_all()?;
    let staged_checksum = staged.content_checksum();
    if staged_checksum != metadata.content_checksum {
        return Err(archive_error(
            &source,
            format!(
                "restored content checksum {} disagrees with archive {}",
                staged_checksum, metadata.content_checksum
            ),
        ));
    }
    drop(staged);

    install_without_overwrite(&temp_path, &destination)?;
    cleanup.disarm();
    sync_parent_directory(&destination)?;
    let store = HdcStore::open_after_replacement(&destination, coordination_lock)?;
    let destination_checksum = store.content_checksum();
    if destination_checksum != metadata.content_checksum {
        return Err(archive_error(
            &source,
            "published destination checksum disagrees with validated archive",
        ));
    }
    let report = PortableRestoreReport {
        source,
        destination,
        metadata,
        destination_generation: store.header_generation(),
        destination_checksum,
    };
    Ok((store, report))
}

fn validate_archive(
    path: &Path,
    limits: PortableArchiveLimits,
) -> Result<PortableArchiveMetadata, HdcStoreError> {
    let mut file = File::open(path)?;
    let file_len_u64 = file.metadata()?.len();
    if file_len_u64 < ARCHIVE_HEADER_SIZE as u64 {
        return Err(archive_error(
            path,
            format!("archive is only {file_len_u64} bytes"),
        ));
    }
    let mut header_bytes = [0u8; ARCHIVE_HEADER_SIZE];
    file.read_exact(&mut header_bytes)?;
    let metadata = decode_header(path, &header_bytes)?;
    limits.check(path, metadata.record_count, file_len_u64)?;
    let expected_len = expected_archive_len(metadata.record_count)?;
    if file_len_u64 != expected_len {
        return Err(archive_error(
            path,
            format!("expected {expected_len} bytes, found {file_len_u64}"),
        ));
    }

    let mut payload_crc = Crc64Ecma::new();
    let mut logical = ContentChecksumBuilder::new(metadata.record_count);
    let mut previous_id = None;
    for _ in 0..metadata.record_count {
        let mut id_bytes = [0u8; 8];
        let mut hv_bytes = [0u8; 2048];
        file.read_exact(&mut id_bytes)?;
        file.read_exact(&mut hv_bytes)?;
        let id = u64::from_le_bytes(id_bytes);
        if previous_id.is_some_and(|previous| id <= previous) {
            return Err(archive_error(
                path,
                format!("record IDs are not strictly increasing at id {id}"),
            ));
        }
        previous_id = Some(id);
        payload_crc.update(&id_bytes);
        payload_crc.update(&hv_bytes);
        logical.update(id, &hv_bytes);
    }
    let found_payload = payload_crc.finalize();
    if found_payload != metadata.payload_crc64_ecma {
        return Err(archive_error(
            path,
            format!(
                "payload checksum mismatch: expected {:#018x}, found {found_payload:#018x}",
                metadata.payload_crc64_ecma
            ),
        ));
    }
    let found_content = logical.finalize().ok_or_else(|| {
        archive_error(path, "decoded record count disagrees with archive metadata")
    })?;
    if found_content != metadata.content_checksum {
        return Err(archive_error(
            path,
            format!(
                "logical content checksum mismatch: expected {}, found {found_content}",
                metadata.content_checksum
            ),
        ));
    }
    Ok(metadata)
}

fn encode_header(metadata: PortableArchiveMetadata) -> [u8; ARCHIVE_HEADER_SIZE] {
    let mut bytes = [0u8; ARCHIVE_HEADER_SIZE];
    bytes[0..8].copy_from_slice(&ARCHIVE_MAGIC);
    bytes[8..12].copy_from_slice(&ARCHIVE_VERSION.to_le_bytes());
    bytes[16..24].copy_from_slice(&metadata.source_generation.to_le_bytes());
    bytes[24..32].copy_from_slice(&metadata.record_count.to_le_bytes());
    bytes[32..36].copy_from_slice(&metadata.lsh_bands.to_le_bytes());
    bytes[36..40].copy_from_slice(&metadata.lsh_rows.to_le_bytes());
    bytes[40..44].copy_from_slice(&metadata.content_checksum.version.to_le_bytes());
    bytes[44..48].copy_from_slice(&(ARCHIVE_RECORD_SIZE as u32).to_le_bytes());
    bytes[48..56].copy_from_slice(&metadata.content_checksum.crc64_ecma.to_le_bytes());
    bytes[56..64].copy_from_slice(&metadata.payload_crc64_ecma.to_le_bytes());
    let checksum = crc64_ecma(&bytes);
    bytes[HEADER_CHECKSUM_OFFSET..HEADER_CHECKSUM_END].copy_from_slice(&checksum.to_le_bytes());
    bytes
}

fn decode_header(
    path: &Path,
    bytes: &[u8; ARCHIVE_HEADER_SIZE],
) -> Result<PortableArchiveMetadata, HdcStoreError> {
    if bytes[0..8] != ARCHIVE_MAGIC {
        return Err(archive_error(path, "bad portable archive magic bytes"));
    }
    let version = read_u32(bytes, 8);
    if version != ARCHIVE_VERSION {
        return Err(archive_error(
            path,
            format!("archive version mismatch: expected {ARCHIVE_VERSION}, found {version}"),
        ));
    }
    if read_u32(bytes, 12) != 0 || bytes[72..].iter().any(|byte| *byte != 0) {
        return Err(archive_error(
            path,
            "portable archive reserved fields are non-zero",
        ));
    }
    let found_header_checksum = read_u64(bytes, HEADER_CHECKSUM_OFFSET);
    let mut checksum_bytes = *bytes;
    checksum_bytes[HEADER_CHECKSUM_OFFSET..HEADER_CHECKSUM_END].fill(0);
    let expected_header_checksum = crc64_ecma(&checksum_bytes);
    if found_header_checksum != expected_header_checksum {
        return Err(archive_error(
            path,
            format!(
                "header checksum mismatch: expected {expected_header_checksum:#018x}, found {found_header_checksum:#018x}"
            ),
        ));
    }
    let record_size = read_u32(bytes, 44);
    if record_size != ARCHIVE_RECORD_SIZE as u32 {
        return Err(archive_error(
            path,
            format!("record size mismatch: expected {ARCHIVE_RECORD_SIZE}, found {record_size}"),
        ));
    }
    crate::lsh_persistent::validate_lsh_config(
        read_u32(bytes, 32) as usize,
        read_u32(bytes, 36) as usize,
    )
    .map_err(|error| archive_error(path, format!("invalid LSH configuration: {error}")))?;
    let content_version = read_u32(bytes, 40);
    if content_version != StoreContentChecksum::VERSION {
        return Err(archive_error(
            path,
            format!(
                "content checksum version mismatch: expected {}, found {content_version}",
                StoreContentChecksum::VERSION
            ),
        ));
    }
    Ok(PortableArchiveMetadata {
        version,
        source_generation: read_u64(bytes, 16),
        record_count: read_u64(bytes, 24),
        lsh_bands: read_u32(bytes, 32),
        lsh_rows: read_u32(bytes, 36),
        content_checksum: StoreContentChecksum {
            version: content_version,
            live_count: read_u64(bytes, 24),
            crc64_ecma: read_u64(bytes, 48),
        },
        payload_crc64_ecma: read_u64(bytes, 56),
    })
}

fn expected_archive_len(record_count: u64) -> Result<u64, HdcStoreError> {
    let record_size =
        u64::try_from(ARCHIVE_RECORD_SIZE).map_err(|_| HdcStoreError::ArithmeticOverflow {
            context: "portable archive record size conversion",
        })?;
    let payload =
        record_count
            .checked_mul(record_size)
            .ok_or(HdcStoreError::ArithmeticOverflow {
                context: "portable archive payload length",
            })?;
    (ARCHIVE_HEADER_SIZE as u64)
        .checked_add(payload)
        .ok_or(HdcStoreError::ArithmeticOverflow {
            context: "portable archive total length",
        })
}

fn read_record(file: &mut File, path: &Path) -> Result<(u64, BinaryHV), HdcStoreError> {
    let mut id_bytes = [0u8; 8];
    let mut hv_bytes = [0u8; 2048];
    file.read_exact(&mut id_bytes)
        .map_err(|error| archive_io(path, error))?;
    file.read_exact(&mut hv_bytes)
        .map_err(|error| archive_io(path, error))?;
    Ok((u64::from_le_bytes(id_bytes), BinaryHV(hv_bytes)))
}

fn create_staging_store(
    destination: &Path,
    config: StoreConfig,
) -> Result<(PathBuf, HdcStore), HdcStoreError> {
    for _ in 0..128 {
        let path = unique_temp_path(destination, "restore");
        match HdcStore::create_staging(&path, config) {
            Ok(store) => return Ok((path, store)),
            Err(HdcStoreError::Io(error)) if error.kind() == ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
    Err(archive_error(
        destination,
        "could not allocate a unique restore staging path",
    ))
}

fn create_temp_file(destination: &Path, purpose: &str) -> Result<(PathBuf, File), HdcStoreError> {
    for _ in 0..128 {
        let path = unique_temp_path(destination, purpose);
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => return Ok((path, file)),
            Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Err(archive_error(
        destination,
        "could not allocate a unique archive staging path",
    ))
}

fn unique_temp_path(destination: &Path, purpose: &str) -> PathBuf {
    let parent = destination.parent().unwrap_or_else(|| Path::new("."));
    let file_name = destination
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("hdc-archive");
    let sequence = ARCHIVE_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    parent.join(format!(
        ".{file_name}.{purpose}-{}-{sequence}.tmp",
        std::process::id()
    ))
}

fn install_without_overwrite(source: &Path, destination: &Path) -> Result<(), HdcStoreError> {
    match std::fs::hard_link(source, destination) {
        Ok(()) => {
            std::fs::remove_file(source)?;
            Ok(())
        }
        Err(error) if error.kind() == ErrorKind::AlreadyExists => {
            Err(HdcStoreError::ArchiveDestinationExists {
                path: destination.to_path_buf(),
            })
        }
        Err(error) => Err(error.into()),
    }
}

fn sync_parent_directory(path: &Path) -> Result<(), HdcStoreError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    File::open(parent)?.sync_all()?;
    Ok(())
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

fn archive_error(path: &Path, reason: impl Into<String>) -> HdcStoreError {
    HdcStoreError::InvalidPortableArchive {
        path: path.to_path_buf(),
        reason: reason.into(),
    }
}

fn archive_io(path: &Path, error: std::io::Error) -> HdcStoreError {
    archive_error(path, format!("archive record read failed: {error}"))
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
            let _ = std::fs::remove_file(batch_journal_path(&self.path));
            let _ = std::fs::remove_file(lsh_snapshot_path(&self.path));
            let _ = std::fs::remove_file(store_lock_path(&self.path));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Seek, SeekFrom, Write};

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn portable_round_trip_preserves_logical_content_and_lsh_config() {
        let dir = tempdir().unwrap();
        let source = dir.path().join("source.hdc");
        let archive = dir.path().join("backup.hdca");
        let destination = dir.path().join("restored.hdc");
        {
            let mut store = HdcStore::create(
                &source,
                StoreConfig {
                    initial_capacity: 8,
                    lsh_bands: 16,
                    lsh_rows: 6,
                },
            )
            .unwrap();
            store.append(30, &BinaryHV::random(30)).unwrap();
            store.append(10, &BinaryHV::random(10)).unwrap();
            store.append(20, &BinaryHV::random(20)).unwrap();
            store.delete(20).unwrap();
        }

        let exported = export_portable_archive(&source, &archive).unwrap();
        assert_eq!(exported.metadata.record_count, 2);
        assert_eq!(
            inspect_portable_archive(&archive).unwrap(),
            exported.metadata
        );
        let (restored, report) = restore_portable_archive(&archive, &destination).unwrap();
        assert_eq!(
            report.destination_checksum,
            exported.metadata.content_checksum
        );
        assert_eq!(restored.live_count(), 2);
        assert_eq!(restored.tombstone_count(), 0);
        assert_eq!(restored.get(10), Some(&BinaryHV::random(10)));
        assert_eq!(restored.get(30), Some(&BinaryHV::random(30)));
    }

    #[test]
    fn restore_uses_bounded_batch_generations() {
        let dir = tempdir().unwrap();
        let source = dir.path().join("source-many.hdc");
        let archive = dir.path().join("backup-many.hdca");
        let destination = dir.path().join("restored-many.hdc");
        let record_count = RESTORE_BATCH_RECORDS as u64 + 7;
        {
            let mut store = HdcStore::create(
                &source,
                StoreConfig {
                    initial_capacity: record_count,
                    ..StoreConfig::default()
                },
            )
            .unwrap();
            let mut batch = WriteBatch::new();
            for id in 0..record_count {
                batch.push_append(id, BinaryHV::random(id));
            }
            store.apply_batch(batch).unwrap();
        }
        export_portable_archive(&source, &archive).unwrap();
        let (restored, _) = restore_portable_archive(&archive, &destination).unwrap();
        assert_eq!(restored.live_count(), record_count);
        assert_eq!(restored.header_generation(), 3);
    }

    #[test]
    fn corrupt_payload_is_rejected_before_restore_publication() {
        let dir = tempdir().unwrap();
        let source = dir.path().join("source.hdc");
        let archive = dir.path().join("backup.hdca");
        let destination = dir.path().join("restored.hdc");
        {
            let mut store = HdcStore::create(&source, StoreConfig::default()).unwrap();
            store.append(1, &BinaryHV::random(1)).unwrap();
        }
        export_portable_archive(&source, &archive).unwrap();
        let mut file = OpenOptions::new().write(true).open(&archive).unwrap();
        file.seek(SeekFrom::Start((ARCHIVE_HEADER_SIZE + 12) as u64))
            .unwrap();
        file.write_all(&[0xAA]).unwrap();
        file.sync_all().unwrap();

        assert!(matches!(
            restore_portable_archive(&archive, &destination),
            Err(HdcStoreError::InvalidPortableArchive { .. })
        ));
        assert!(!destination.exists());
    }

    #[test]
    fn explicit_limits_reject_archive_before_restore_staging() {
        let dir = tempdir().unwrap();
        let source = dir.path().join("source.hdc");
        let archive = dir.path().join("backup.hdca");
        let destination = dir.path().join("restored.hdc");
        {
            let mut store = HdcStore::create(&source, StoreConfig::default()).unwrap();
            store.append(1, &BinaryHV::random(1)).unwrap();
            store.append(2, &BinaryHV::random(2)).unwrap();
        }
        export_portable_archive(&source, &archive).unwrap();
        let limits = PortableArchiveLimits {
            max_records: 1,
            max_archive_bytes: u64::MAX,
        };
        assert!(matches!(
            inspect_portable_archive_with_limits(&archive, limits),
            Err(HdcStoreError::ArchiveLimitExceeded { .. })
        ));
        assert!(matches!(
            restore_portable_archive_with_limits(&archive, &destination, limits),
            Err(HdcStoreError::ArchiveLimitExceeded { .. })
        ));
        assert!(!destination.exists());
    }

    #[test]
    fn export_limits_are_checked_before_staging() {
        let dir = tempdir().unwrap();
        let source = dir.path().join("source.hdc");
        let archive = dir.path().join("backup.hdca");
        {
            let mut store = HdcStore::create(&source, StoreConfig::default()).unwrap();
            store.append(1, &BinaryHV::random(1)).unwrap();
        }
        let limits = PortableArchiveLimits {
            max_records: 0,
            max_archive_bytes: u64::MAX,
        };
        assert!(matches!(
            export_portable_archive_with_limits(&source, &archive, limits),
            Err(HdcStoreError::ArchiveLimitExceeded { .. })
        ));
        assert!(!archive.exists());
    }

    #[test]
    fn export_and_restore_refuse_existing_destinations() {
        let dir = tempdir().unwrap();
        let source = dir.path().join("source.hdc");
        let archive = dir.path().join("backup.hdca");
        let destination = dir.path().join("restored.hdc");
        drop(HdcStore::create(&source, StoreConfig::default()).unwrap());
        export_portable_archive(&source, &archive).unwrap();
        assert!(matches!(
            export_portable_archive(&source, &archive),
            Err(HdcStoreError::ArchiveDestinationExists { .. })
        ));
        drop(HdcStore::create(&destination, StoreConfig::default()).unwrap());
        assert!(matches!(
            restore_portable_archive(&archive, &destination),
            Err(HdcStoreError::ArchiveDestinationExists { .. })
        ));
    }
}
