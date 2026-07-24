// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Atomic sidecar snapshots for deterministic LSH signatures.
//!
//! The canonical BinaryHV store remains independently valid. A snapshot is an
//! optional acceleration artifact: it is accepted only when its store counts,
//! generation, content fingerprint, LSH dimensions, seed, length, ordering,
//! and checksums all match the expected store state.

use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::HdcStoreError;
use crate::checksum::{Crc64Ecma, crc64_ecma};
use crate::lsh_persistent::{LshSignature, validate_lsh_config};

const SNAPSHOT_MAGIC: [u8; 8] = *b"HDCLSH01";
const SNAPSHOT_VERSION: u32 = 1;
const SNAPSHOT_HEADER_SIZE: usize = 96;
const HEADER_CHECKSUM_OFFSET: usize = 88;
const HEADER_CHECKSUM_END: usize = 96;
const RECORD_ID_SIZE: usize = 8;
static SNAPSHOT_TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Controls how opening a store treats its optional LSH sidecar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IndexOpenPolicy {
    /// Load a fully compatible snapshot when present; otherwise rebuild.
    #[default]
    PreferSnapshot,
    /// Ignore any sidecar and deterministically rebuild from canonical vectors.
    Rebuild,
    /// Fail opening unless a fully compatible snapshot can be loaded.
    RequireSnapshot,
}

/// How the current in-memory index was populated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexLoadSource {
    /// Newly created empty index.
    New,
    /// Loaded from a validated signature snapshot.
    Snapshot,
    /// Recomputed from canonical BinaryHV entries.
    Rebuilt,
}

/// Observable lifecycle state for the optional persisted index artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexStatus {
    pub source: IndexLoadSource,
    pub snapshot_path: PathBuf,
    pub snapshot_current: bool,
    pub ignored_snapshot_error: Option<String>,
}

impl IndexStatus {
    pub(crate) fn new(store_path: &Path) -> Self {
        Self {
            source: IndexLoadSource::New,
            snapshot_path: lsh_snapshot_path(store_path),
            snapshot_current: false,
            ignored_snapshot_error: None,
        }
    }

    pub(crate) fn loaded(store_path: &Path) -> Self {
        Self {
            source: IndexLoadSource::Snapshot,
            snapshot_path: lsh_snapshot_path(store_path),
            snapshot_current: true,
            ignored_snapshot_error: None,
        }
    }

    pub(crate) fn rebuilt(store_path: &Path, ignored_snapshot_error: Option<String>) -> Self {
        Self {
            source: IndexLoadSource::Rebuilt,
            snapshot_path: lsh_snapshot_path(store_path),
            snapshot_current: false,
            ignored_snapshot_error,
        }
    }

    pub(crate) fn mark_dirty(&mut self) {
        self.snapshot_current = false;
    }

    pub(crate) fn mark_checkpointed(&mut self) {
        self.snapshot_current = true;
        self.ignored_snapshot_error = None;
    }
}

/// Store and index identity required for a signature snapshot to be reusable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LshSnapshotMetadata {
    pub store_generation: u64,
    pub vector_count: u64,
    pub live_count: u64,
    pub tombstone_count: u64,
    pub lsh_bands: u32,
    pub lsh_rows: u32,
    pub lsh_seed: u64,
    pub store_fingerprint: u64,
}

impl LshSnapshotMetadata {
    pub fn validate(&self) -> Result<(), HdcStoreError> {
        let committed = self.live_count.checked_add(self.tombstone_count).ok_or(
            HdcStoreError::ArithmeticOverflow {
                context: "snapshot live_count + tombstone_count",
            },
        )?;
        if committed != self.vector_count {
            return Err(HdcStoreError::InvalidIndexSnapshot {
                path: PathBuf::new(),
                reason: format!(
                    "count invariant violated: vector_count={}, live_count={}, tombstone_count={}",
                    self.vector_count, self.live_count, self.tombstone_count
                ),
            });
        }
        validate_lsh_config(self.lsh_bands as usize, self.lsh_rows as usize).map_err(|error| {
            HdcStoreError::InvalidIndexSnapshot {
                path: PathBuf::new(),
                reason: format!("invalid LSH configuration: {error}"),
            }
        })
    }

    fn record_size(&self) -> Result<usize, HdcStoreError> {
        let hashes = (self.lsh_bands as usize)
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or(HdcStoreError::ArithmeticOverflow {
                context: "snapshot signature record size",
            })?;
        RECORD_ID_SIZE
            .checked_add(hashes)
            .ok_or(HdcStoreError::ArithmeticOverflow {
                context: "snapshot record size",
            })
    }

    fn expected_file_len(&self) -> Result<usize, HdcStoreError> {
        let entries =
            usize::try_from(self.live_count).map_err(|_| HdcStoreError::ArithmeticOverflow {
                context: "snapshot live_count conversion",
            })?;
        let payload =
            entries
                .checked_mul(self.record_size()?)
                .ok_or(HdcStoreError::ArithmeticOverflow {
                    context: "snapshot payload length",
                })?;
        SNAPSHOT_HEADER_SIZE
            .checked_add(payload)
            .ok_or(HdcStoreError::ArithmeticOverflow {
                context: "snapshot file length",
            })
    }
}

/// Fully validated snapshot contents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LshSnapshot {
    pub metadata: LshSnapshotMetadata,
    pub entries: Vec<(u64, LshSignature)>,
}

#[derive(Debug, Clone, Copy)]
struct SnapshotHeader {
    metadata: LshSnapshotMetadata,
    entry_count: u64,
    payload_checksum: u64,
    header_checksum: u64,
}

impl SnapshotHeader {
    fn new(metadata: LshSnapshotMetadata, entry_count: u64, payload_checksum: u64) -> Self {
        Self {
            metadata,
            entry_count,
            payload_checksum,
            header_checksum: 0,
        }
        .sealed()
    }

    fn sealed(mut self) -> Self {
        self.header_checksum = 0;
        self.header_checksum = crc64_ecma(&self.to_bytes_unchecked());
        self
    }

    fn to_bytes(self) -> [u8; SNAPSHOT_HEADER_SIZE] {
        self.sealed().to_bytes_unchecked()
    }

    fn to_bytes_unchecked(self) -> [u8; SNAPSHOT_HEADER_SIZE] {
        let mut bytes = [0u8; SNAPSHOT_HEADER_SIZE];
        bytes[0..8].copy_from_slice(&SNAPSHOT_MAGIC);
        bytes[8..12].copy_from_slice(&SNAPSHOT_VERSION.to_le_bytes());
        bytes[12..16].copy_from_slice(&0u32.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.metadata.store_generation.to_le_bytes());
        bytes[24..32].copy_from_slice(&self.metadata.vector_count.to_le_bytes());
        bytes[32..40].copy_from_slice(&self.metadata.live_count.to_le_bytes());
        bytes[40..48].copy_from_slice(&self.metadata.tombstone_count.to_le_bytes());
        bytes[48..52].copy_from_slice(&self.metadata.lsh_bands.to_le_bytes());
        bytes[52..56].copy_from_slice(&self.metadata.lsh_rows.to_le_bytes());
        bytes[56..64].copy_from_slice(&self.metadata.lsh_seed.to_le_bytes());
        bytes[64..72].copy_from_slice(&self.entry_count.to_le_bytes());
        bytes[72..80].copy_from_slice(&self.metadata.store_fingerprint.to_le_bytes());
        bytes[80..88].copy_from_slice(&self.payload_checksum.to_le_bytes());
        bytes[88..96].copy_from_slice(&self.header_checksum.to_le_bytes());
        bytes
    }

    fn parse(bytes: &[u8; SNAPSHOT_HEADER_SIZE], path: &Path) -> Result<Self, HdcStoreError> {
        if bytes[0..8] != SNAPSHOT_MAGIC {
            return Err(snapshot_error(path, "bad snapshot magic bytes"));
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().expect("fixed version slice"));
        if version != SNAPSHOT_VERSION {
            return Err(snapshot_error(
                path,
                format!("snapshot version mismatch: expected {SNAPSHOT_VERSION}, found {version}"),
            ));
        }
        let flags = u32::from_le_bytes(bytes[12..16].try_into().expect("fixed flags slice"));
        if flags != 0 {
            return Err(snapshot_error(
                path,
                format!("unsupported snapshot flags: {flags:#x}"),
            ));
        }

        let found_checksum = u64::from_le_bytes(
            bytes[HEADER_CHECKSUM_OFFSET..HEADER_CHECKSUM_END]
                .try_into()
                .expect("fixed checksum slice"),
        );
        let mut checksum_bytes = *bytes;
        checksum_bytes[HEADER_CHECKSUM_OFFSET..HEADER_CHECKSUM_END].fill(0);
        let expected_checksum = crc64_ecma(&checksum_bytes);
        if found_checksum != expected_checksum {
            return Err(snapshot_error(
                path,
                format!(
                    "header checksum mismatch: expected {expected_checksum:#018x}, found {found_checksum:#018x}"
                ),
            ));
        }

        let metadata = LshSnapshotMetadata {
            store_generation: read_u64(bytes, 16),
            vector_count: read_u64(bytes, 24),
            live_count: read_u64(bytes, 32),
            tombstone_count: read_u64(bytes, 40),
            lsh_bands: read_u32(bytes, 48),
            lsh_rows: read_u32(bytes, 52),
            lsh_seed: read_u64(bytes, 56),
            store_fingerprint: read_u64(bytes, 72),
        };
        metadata
            .validate()
            .map_err(|error| snapshot_error(path, error.to_string()))?;

        Ok(Self {
            metadata,
            entry_count: read_u64(bytes, 64),
            payload_checksum: read_u64(bytes, 80),
            header_checksum: found_checksum,
        })
    }
}

/// Compute the snapshot compatibility fingerprint over ascending live records.
pub(crate) fn fingerprint_ordered<'a>(
    records: impl IntoIterator<Item = (u64, &'a symthaea_core::hdc::BinaryHV)>,
    live_count: u64,
) -> u64 {
    let mut crc = Crc64Ecma::new();
    crc.update(b"HDCSTORE-LIVE-FINGERPRINT-V1");
    crc.update(&live_count.to_le_bytes());
    for (id, hv) in records {
        crc.update(&id.to_le_bytes());
        crc.update(&hv.0);
    }
    crc.finalize()
}

/// Deterministic sidecar path used for an HDC store.
pub fn lsh_snapshot_path(store_path: impl AsRef<Path>) -> PathBuf {
    let mut path = store_path.as_ref().as_os_str().to_os_string();
    path.push(".lsh");
    PathBuf::from(path)
}

/// Atomically write a complete, ID-sorted signature snapshot.
pub fn write_lsh_snapshot(
    store_path: impl AsRef<Path>,
    metadata: LshSnapshotMetadata,
    entries: &[(u64, LshSignature)],
) -> Result<PathBuf, HdcStoreError> {
    metadata
        .validate()
        .map_err(|error| snapshot_error(store_path.as_ref(), error.to_string()))?;
    if entries.len() as u64 != metadata.live_count {
        return Err(snapshot_error(
            store_path.as_ref(),
            format!(
                "snapshot contains {} signatures but metadata requires {}",
                entries.len(),
                metadata.live_count
            ),
        ));
    }

    let mut sorted = entries.to_vec();
    sorted.sort_unstable_by_key(|(id, _)| *id);
    validate_entries(&metadata, &sorted, store_path.as_ref())?;

    let destination = lsh_snapshot_path(store_path.as_ref());
    let (temporary_path, file) = create_unique_temp(&destination)?;
    let mut cleanup = TempSnapshotGuard::new(temporary_path.clone());
    let mut writer = BufWriter::new(file);
    writer.write_all(&[0u8; SNAPSHOT_HEADER_SIZE])?;

    let mut payload_crc = Crc64Ecma::new();
    for (id, signature) in &sorted {
        let id_bytes = id.to_le_bytes();
        writer.write_all(&id_bytes)?;
        payload_crc.update(&id_bytes);
        for &hash in signature.hashes() {
            let hash_bytes = hash.to_le_bytes();
            writer.write_all(&hash_bytes)?;
            payload_crc.update(&hash_bytes);
        }
    }

    let header = SnapshotHeader::new(metadata, sorted.len() as u64, payload_crc.finalize());
    writer.flush()?;
    writer.seek(SeekFrom::Start(0))?;
    writer.write_all(&header.to_bytes())?;
    writer.flush()?;
    let file = writer.into_inner().map_err(|error| error.into_error())?;
    file.sync_all()?;
    drop(file);

    replace_snapshot(&temporary_path, &destination)?;
    cleanup.disarm();
    sync_parent_directory(&destination)?;
    Ok(destination)
}

/// Load and validate an optional signature snapshot.
///
/// Missing sidecars return `Ok(None)`. Present but malformed or stale sidecars
/// return an error so callers can decide whether to rebuild or fail closed.
pub fn load_lsh_snapshot(
    store_path: impl AsRef<Path>,
    expected: LshSnapshotMetadata,
) -> Result<Option<LshSnapshot>, HdcStoreError> {
    let path = lsh_snapshot_path(store_path.as_ref());
    let file = match File::open(&path) {
        Ok(file) => file,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let file_len =
        usize::try_from(file.metadata()?.len()).map_err(|_| HdcStoreError::ArithmeticOverflow {
            context: "snapshot file length conversion",
        })?;
    if file_len < SNAPSHOT_HEADER_SIZE {
        return Err(snapshot_error(
            &path,
            format!(
                "snapshot is {file_len} bytes; at least {SNAPSHOT_HEADER_SIZE} bytes are required"
            ),
        ));
    }

    let mut reader = BufReader::new(file);
    let mut header_bytes = [0u8; SNAPSHOT_HEADER_SIZE];
    reader.read_exact(&mut header_bytes)?;
    let header = SnapshotHeader::parse(&header_bytes, &path)?;
    if header.metadata != expected {
        return Err(snapshot_error(
            &path,
            format!(
                "snapshot identity is stale or incompatible: expected {expected:?}, found {:?}",
                header.metadata
            ),
        ));
    }
    if header.entry_count != expected.live_count {
        return Err(snapshot_error(
            &path,
            format!(
                "snapshot entry_count {} does not match live_count {}",
                header.entry_count, expected.live_count
            ),
        ));
    }
    let expected_len = expected.expected_file_len()?;
    if file_len != expected_len {
        return Err(snapshot_error(
            &path,
            format!("snapshot length mismatch: expected {expected_len}, found {file_len}"),
        ));
    }

    let capacity =
        usize::try_from(header.entry_count).map_err(|_| HdcStoreError::ArithmeticOverflow {
            context: "snapshot entry_count conversion",
        })?;
    let mut entries = Vec::with_capacity(capacity);
    let mut payload_crc = Crc64Ecma::new();
    let mut previous_id = None;
    for _ in 0..header.entry_count {
        let mut id_bytes = [0u8; 8];
        reader.read_exact(&mut id_bytes)?;
        payload_crc.update(&id_bytes);
        let id = u64::from_le_bytes(id_bytes);
        if previous_id.is_some_and(|previous| id <= previous) {
            return Err(snapshot_error(
                &path,
                format!("snapshot IDs are not strictly increasing at {id}"),
            ));
        }
        previous_id = Some(id);

        let mut hashes = Vec::with_capacity(expected.lsh_bands as usize);
        for _ in 0..expected.lsh_bands {
            let mut hash_bytes = [0u8; 4];
            reader.read_exact(&mut hash_bytes)?;
            payload_crc.update(&hash_bytes);
            let hash = u32::from_le_bytes(hash_bytes);
            validate_band_hash(hash, expected.lsh_rows, &path)?;
            hashes.push(hash);
        }
        entries.push((id, LshSignature::from_hashes(hashes)));
    }

    let found_payload_checksum = payload_crc.finalize();
    if found_payload_checksum != header.payload_checksum {
        return Err(snapshot_error(
            &path,
            format!(
                "payload checksum mismatch: expected {:#018x}, found {found_payload_checksum:#018x}",
                header.payload_checksum
            ),
        ));
    }

    Ok(Some(LshSnapshot {
        metadata: header.metadata,
        entries,
    }))
}

fn validate_entries(
    metadata: &LshSnapshotMetadata,
    entries: &[(u64, LshSignature)],
    path: &Path,
) -> Result<(), HdcStoreError> {
    let mut previous_id = None;
    for (id, signature) in entries {
        if previous_id.is_some_and(|previous| *id <= previous) {
            return Err(snapshot_error(
                path,
                format!("snapshot IDs must be unique and strictly increasing; found {id}"),
            ));
        }
        previous_id = Some(*id);
        if signature.band_count() != metadata.lsh_bands as usize {
            return Err(snapshot_error(
                path,
                format!(
                    "signature for id {id} contains {} bands; expected {}",
                    signature.band_count(),
                    metadata.lsh_bands
                ),
            ));
        }
        for &hash in signature.hashes() {
            validate_band_hash(hash, metadata.lsh_rows, path)?;
        }
    }
    Ok(())
}

fn validate_band_hash(hash: u32, rows: u32, path: &Path) -> Result<(), HdcStoreError> {
    if rows < 32 && hash >> rows != 0 {
        return Err(snapshot_error(
            path,
            format!("band hash {hash:#010x} uses bits above configured row count {rows}"),
        ));
    }
    Ok(())
}

fn create_unique_temp(destination: &Path) -> Result<(PathBuf, File), HdcStoreError> {
    let parent = destination.parent().unwrap_or_else(|| Path::new("."));
    let file_name = destination
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("hdc-store.lsh");

    for _ in 0..128 {
        let sequence = SNAPSHOT_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(
            ".{file_name}.checkpoint-{}-{sequence}.tmp",
            std::process::id()
        ));
        match OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => return Ok((candidate, file)),
            Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }

    Err(snapshot_error(
        destination,
        "could not allocate a unique snapshot staging path after 128 attempts",
    ))
}

fn replace_snapshot(source: &Path, destination: &Path) -> Result<(), HdcStoreError> {
    std::fs::rename(source, destination).map_err(|error| {
        snapshot_error(
            destination,
            format!("same-directory snapshot replacement failed: {error}"),
        )
    })
}

#[cfg(unix)]
fn sync_parent_directory(path: &Path) -> Result<(), HdcStoreError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    File::open(parent)?.sync_all()?;
    Ok(())
}

#[cfg(not(unix))]
fn sync_parent_directory(_path: &Path) -> Result<(), HdcStoreError> {
    Ok(())
}

fn snapshot_error(path: impl AsRef<Path>, reason: impl Into<String>) -> HdcStoreError {
    HdcStoreError::InvalidIndexSnapshot {
        path: path.as_ref().to_path_buf(),
        reason: reason.into(),
    }
}

fn read_u32(bytes: &[u8; SNAPSHOT_HEADER_SIZE], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("fixed u32 snapshot field"),
    )
}

fn read_u64(bytes: &[u8; SNAPSHOT_HEADER_SIZE], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("fixed u64 snapshot field"),
    )
}

struct TempSnapshotGuard {
    path: PathBuf,
    armed: bool,
}

impl TempSnapshotGuard {
    fn new(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for TempSnapshotGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn metadata() -> LshSnapshotMetadata {
        LshSnapshotMetadata {
            store_generation: 7,
            vector_count: 3,
            live_count: 2,
            tombstone_count: 1,
            lsh_bands: 3,
            lsh_rows: 8,
            lsh_seed: 42,
            store_fingerprint: 0xA5A5_1234_9876_5A5A,
        }
    }

    fn entries() -> Vec<(u64, LshSignature)> {
        vec![
            (2, LshSignature::from_hashes(vec![1, 2, 3])),
            (9, LshSignature::from_hashes(vec![4, 5, 6])),
        ]
    }

    #[test]
    fn snapshot_roundtrip_is_deterministic() {
        let dir = tempdir().unwrap();
        let store_path = dir.path().join("memory.hdc");
        write_lsh_snapshot(&store_path, metadata(), &entries()).unwrap();
        let loaded = load_lsh_snapshot(&store_path, metadata()).unwrap().unwrap();
        assert_eq!(loaded.metadata, metadata());
        assert_eq!(loaded.entries, entries());
    }

    #[test]
    fn missing_snapshot_is_not_an_error() {
        let dir = tempdir().unwrap();
        let store_path = dir.path().join("memory.hdc");
        assert!(
            load_lsh_snapshot(&store_path, metadata())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn stale_snapshot_identity_is_rejected() {
        let dir = tempdir().unwrap();
        let store_path = dir.path().join("memory.hdc");
        write_lsh_snapshot(&store_path, metadata(), &entries()).unwrap();
        let mut stale = metadata();
        stale.store_generation += 1;
        assert!(matches!(
            load_lsh_snapshot(&store_path, stale),
            Err(HdcStoreError::InvalidIndexSnapshot { .. })
        ));
    }

    #[test]
    fn payload_corruption_is_detected() {
        use std::io::{Seek, SeekFrom, Write};

        let dir = tempdir().unwrap();
        let store_path = dir.path().join("memory.hdc");
        let snapshot_path = write_lsh_snapshot(&store_path, metadata(), &entries()).unwrap();
        let mut file = OpenOptions::new().write(true).open(snapshot_path).unwrap();
        file.seek(SeekFrom::Start((SNAPSHOT_HEADER_SIZE + 9) as u64))
            .unwrap();
        file.write_all(&[0xFE]).unwrap();
        file.sync_all().unwrap();

        assert!(matches!(
            load_lsh_snapshot(&store_path, metadata()),
            Err(HdcStoreError::InvalidIndexSnapshot { .. })
        ));
    }

    #[test]
    fn writer_rejects_wrong_signature_dimensions() {
        let dir = tempdir().unwrap();
        let store_path = dir.path().join("memory.hdc");
        let malformed = vec![
            (2, LshSignature::from_hashes(vec![1, 2])),
            (9, LshSignature::from_hashes(vec![4, 5, 6])),
        ];
        assert!(write_lsh_snapshot(&store_path, metadata(), &malformed).is_err());
    }
}
