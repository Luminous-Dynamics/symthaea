// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Explicit, validated migration from the legacy format-v1 layout.

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use memmap2::Mmap;
use symthaea_core::hdc::BinaryHV;

use crate::header::{
    ENTRY_HV_OFFSET, ENTRY_SIZE, HEADER_SIZE, LEGACY_DATA_OFFSET, LEGACY_VERSION, MAGIC,
    STATUS_LIVE, STATUS_TOMBSTONE, VERSION, checked_entry_offset, required_file_len,
};
use crate::locking::StoreLock;
use crate::lsh_persistent::validate_lsh_config;
use crate::{HdcStore, HdcStoreError, StoreConfig};

const BINARY_HV_BYTES: usize = 2048;
static MIGRATION_TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Summary of a completed v1-to-v2 migration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MigrationReport {
    /// Source format version.
    pub source_version: u32,
    /// Installed format version.
    pub target_version: u32,
    /// Total committed v1 entries before migration.
    pub source_vector_count: u64,
    /// Live entries copied into the v2 replacement.
    pub migrated_live_count: u64,
    /// Tombstones discarded as part of migration compaction.
    pub discarded_tombstones: u64,
    /// First committed generation in the installed v2 file after copying.
    pub target_generation: u64,
}

#[derive(Debug, Clone, Copy)]
struct LegacyHeaderV1 {
    vector_count: u64,
    live_count: u64,
    tombstone_count: u64,
    lsh_offset: u64,
    lsh_bands: u32,
    lsh_rows: u32,
}

impl LegacyHeaderV1 {
    fn parse(bytes: &[u8; HEADER_SIZE]) -> Result<Self, HdcStoreError> {
        let magic: [u8; 8] = bytes[0..8].try_into().expect("fixed legacy magic slice");
        if magic != MAGIC {
            return Err(HdcStoreError::InvalidHeader {
                reason: "legacy file has bad magic bytes".into(),
            });
        }
        let version =
            u32::from_le_bytes(bytes[8..12].try_into().expect("fixed legacy version slice"));
        if version == VERSION {
            return Err(HdcStoreError::MigrationNotRequired { version });
        }
        if version != LEGACY_VERSION {
            return Err(HdcStoreError::VersionMismatch {
                expected: LEGACY_VERSION,
                found: version,
            });
        }

        let header = Self {
            vector_count: u64::from_le_bytes(
                bytes[12..20]
                    .try_into()
                    .expect("fixed legacy vector_count slice"),
            ),
            live_count: u64::from_le_bytes(
                bytes[20..28]
                    .try_into()
                    .expect("fixed legacy live_count slice"),
            ),
            tombstone_count: u64::from_le_bytes(
                bytes[28..36]
                    .try_into()
                    .expect("fixed legacy tombstone_count slice"),
            ),
            lsh_offset: u64::from_le_bytes(
                bytes[36..44]
                    .try_into()
                    .expect("fixed legacy lsh_offset slice"),
            ),
            lsh_bands: u32::from_le_bytes(
                bytes[44..48]
                    .try_into()
                    .expect("fixed legacy lsh_bands slice"),
            ),
            lsh_rows: u32::from_le_bytes(
                bytes[48..52]
                    .try_into()
                    .expect("fixed legacy lsh_rows slice"),
            ),
        };
        header.validate()?;
        Ok(header)
    }

    fn validate(&self) -> Result<(), HdcStoreError> {
        if self.lsh_offset != 0 {
            return Err(HdcStoreError::InvalidHeader {
                reason: format!(
                    "legacy format does not support persisted LSH data (offset={})",
                    self.lsh_offset
                ),
            });
        }
        let total = self.live_count.checked_add(self.tombstone_count).ok_or(
            HdcStoreError::ArithmeticOverflow {
                context: "legacy live_count + tombstone_count",
            },
        )?;
        if total != self.vector_count {
            return Err(HdcStoreError::InvalidHeader {
                reason: format!(
                    "legacy count invariant violated: vector_count={}, live_count={}, tombstone_count={}",
                    self.vector_count, self.live_count, self.tombstone_count
                ),
            });
        }
        validate_lsh_config(self.lsh_bands as usize, self.lsh_rows as usize).map_err(|error| {
            HdcStoreError::InvalidHeader {
                reason: format!("invalid legacy LSH configuration: {error}"),
            }
        })
    }
}

/// Atomically migrate a validated format-v1 store into format v2.
///
/// Migration copies only live entries, so the installed v2 file is compacted.
/// The source path is not replaced until the complete v2 temporary file has
/// been flushed and synchronized.
pub fn migrate_v1(path: impl AsRef<Path>) -> Result<(HdcStore, MigrationReport), HdcStoreError> {
    let path = path.as_ref().to_path_buf();
    let coordination_lock = StoreLock::exclusive(&path)?;
    let source_file = OpenOptions::new().read(true).write(true).open(&path)?;
    lock_source(&source_file, &path)?;
    let source_permissions = source_file.metadata()?.permissions();
    let source_mmap = unsafe { Mmap::map(&source_file)? };

    if source_mmap.len() < HEADER_SIZE {
        return Err(HdcStoreError::InvalidHeader {
            reason: format!(
                "legacy file is {} bytes; at least {HEADER_SIZE} bytes are required",
                source_mmap.len()
            ),
        });
    }
    let header_bytes: [u8; HEADER_SIZE] = source_mmap[..HEADER_SIZE]
        .try_into()
        .expect("legacy header length checked before conversion");
    let header = LegacyHeaderV1::parse(&header_bytes)?;
    let required_len = required_file_len(LEGACY_DATA_OFFSET, header.vector_count)?;
    if required_len > source_mmap.len() {
        return Err(HdcStoreError::InvalidHeader {
            reason: format!(
                "legacy committed entries require {required_len} bytes, but file contains {} bytes",
                source_mmap.len()
            ),
        });
    }

    validate_legacy_entries(&source_mmap, &header)?;

    let config = StoreConfig {
        initial_capacity: header.live_count.max(64),
        lsh_bands: header.lsh_bands,
        lsh_rows: header.lsh_rows,
    };
    let (tmp_path, mut target) = create_migration_store(&path, config)?;
    let mut cleanup = TempPathGuard::new(tmp_path.clone());
    std::fs::set_permissions(&tmp_path, source_permissions)?;

    for index in 0..header.vector_count {
        let offset = checked_entry_offset(LEGACY_DATA_OFFSET, index)?;
        if source_mmap[offset] != STATUS_LIVE {
            continue;
        }
        let id = read_legacy_id(&source_mmap, offset, index)?;
        let hv = *legacy_hv_at(&source_mmap, offset, index)?;
        target.append(id, &hv)?;
    }

    target.sync_all()?;
    let target_generation = target.header_generation();
    drop(target);

    atomic_replace(&tmp_path, &path)?;
    cleanup.disarm();
    sync_parent_directory(&path)?;

    drop(source_mmap);
    drop(source_file);

    let migrated = HdcStore::open_after_replacement(&path, coordination_lock).map_err(|error| {
        HdcStoreError::MigrationFailed {
            reason: format!("v2 replacement was installed but could not be reopened: {error}"),
        }
    })?;
    let report = MigrationReport {
        source_version: LEGACY_VERSION,
        target_version: VERSION,
        source_vector_count: header.vector_count,
        migrated_live_count: header.live_count,
        discarded_tombstones: header.tombstone_count,
        target_generation,
    };
    Ok((migrated, report))
}

fn validate_legacy_entries(mmap: &Mmap, header: &LegacyHeaderV1) -> Result<(), HdcStoreError> {
    let mut live_ids = HashSet::new();
    let mut live_count = 0u64;
    let mut tombstone_count = 0u64;

    for index in 0..header.vector_count {
        let offset = checked_entry_offset(LEGACY_DATA_OFFSET, index)?;
        match mmap[offset] {
            STATUS_LIVE => {
                let id = read_legacy_id(mmap, offset, index)?;
                if !live_ids.insert(id) {
                    return Err(HdcStoreError::CorruptEntry {
                        index,
                        reason: format!("duplicate legacy live id {id}"),
                    });
                }
                legacy_hv_at(mmap, offset, index)?;
                live_count =
                    live_count
                        .checked_add(1)
                        .ok_or(HdcStoreError::ArithmeticOverflow {
                            context: "legacy scanned live count",
                        })?;
            }
            STATUS_TOMBSTONE => {
                tombstone_count =
                    tombstone_count
                        .checked_add(1)
                        .ok_or(HdcStoreError::ArithmeticOverflow {
                            context: "legacy scanned tombstone count",
                        })?;
            }
            status => {
                return Err(HdcStoreError::CorruptEntry {
                    index,
                    reason: format!("invalid legacy committed status byte {status}"),
                });
            }
        }
    }

    if live_count != header.live_count || tombstone_count != header.tombstone_count {
        return Err(HdcStoreError::InvalidHeader {
            reason: format!(
                "legacy entry scan disagrees with header: live {live_count}/{}, tombstones {tombstone_count}/{}",
                header.live_count, header.tombstone_count
            ),
        });
    }
    Ok(())
}

fn read_legacy_id(mmap: &Mmap, offset: usize, index: u64) -> Result<u64, HdcStoreError> {
    let end = offset
        .checked_add(9)
        .ok_or(HdcStoreError::ArithmeticOverflow {
            context: "legacy entry id end offset",
        })?;
    let bytes: [u8; 8] = mmap
        .get(offset + 1..end)
        .ok_or_else(|| HdcStoreError::CorruptEntry {
            index,
            reason: "legacy entry id extends beyond mapped file".into(),
        })?
        .try_into()
        .expect("legacy entry id slice is exactly eight bytes");
    Ok(u64::from_le_bytes(bytes))
}

fn legacy_hv_at<'a>(
    mmap: &'a Mmap,
    offset: usize,
    index: u64,
) -> Result<&'a BinaryHV, HdcStoreError> {
    let start = offset
        .checked_add(ENTRY_HV_OFFSET)
        .ok_or(HdcStoreError::ArithmeticOverflow {
            context: "legacy BinaryHV start offset",
        })?;
    let end = start
        .checked_add(BINARY_HV_BYTES)
        .ok_or(HdcStoreError::ArithmeticOverflow {
            context: "legacy BinaryHV end offset",
        })?;
    let bytes = mmap
        .get(start..end)
        .ok_or_else(|| HdcStoreError::CorruptEntry {
            index,
            reason: "legacy BinaryHV extends beyond mapped file".into(),
        })?;
    let ptr = bytes.as_ptr();
    if !(ptr as usize).is_multiple_of(32) {
        return Err(HdcStoreError::CorruptEntry {
            index,
            reason: format!("legacy BinaryHV is not 32-byte aligned at offset {start}"),
        });
    }

    // SAFETY: format-v1 used the same 32-byte aligned entry layout and exact
    // BinaryHV byte length as format v2. The reference is bounded by the mmap.
    Ok(unsafe { &*(ptr as *const BinaryHV) })
}

fn create_migration_store(
    destination: &Path,
    config: StoreConfig,
) -> Result<(PathBuf, HdcStore), HdcStoreError> {
    let parent = destination.parent().unwrap_or_else(|| Path::new("."));
    let file_name = destination
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("hdc-store");

    for _ in 0..128 {
        let sequence = MIGRATION_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(
            ".{file_name}.migrate-v1-{}-{sequence}.tmp",
            std::process::id()
        ));
        match HdcStore::create_staging(&candidate, config) {
            Ok(store) => return Ok((candidate, store)),
            Err(HdcStoreError::Io(error)) if error.kind() == ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }

    Err(HdcStoreError::MigrationFailed {
        reason: "could not allocate a unique migration path after 128 attempts".into(),
    })
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

fn lock_source(file: &File, path: &Path) -> Result<(), HdcStoreError> {
    match file.try_lock() {
        Ok(()) => Ok(()),
        Err(std::fs::TryLockError::WouldBlock) => Err(HdcStoreError::StoreLocked {
            path: path.to_path_buf(),
        }),
        Err(std::fs::TryLockError::Error(error)) => Err(HdcStoreError::Io(error)),
    }
}

fn atomic_replace(source: &Path, destination: &Path) -> Result<(), HdcStoreError> {
    std::fs::rename(source, destination).map_err(|error| HdcStoreError::MigrationFailed {
        reason: format!("same-directory migration rename failed: {error}"),
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

#[cfg(test)]
mod tests {
    use std::io::{Seek, SeekFrom, Write};

    use tempfile::tempdir;

    use super::*;

    fn legacy_header_bytes(vector_count: u64, live_count: u64, tombstones: u64) -> [u8; 128] {
        let mut bytes = [0u8; 128];
        bytes[0..8].copy_from_slice(&MAGIC);
        bytes[8..12].copy_from_slice(&LEGACY_VERSION.to_le_bytes());
        bytes[12..20].copy_from_slice(&vector_count.to_le_bytes());
        bytes[20..28].copy_from_slice(&live_count.to_le_bytes());
        bytes[28..36].copy_from_slice(&tombstones.to_le_bytes());
        bytes[44..48].copy_from_slice(&32u32.to_le_bytes());
        bytes[48..52].copy_from_slice(&8u32.to_le_bytes());
        bytes
    }

    fn write_legacy_entry(file: &mut File, index: u64, status: u8, id: u64, hv: &BinaryHV) {
        let offset = checked_entry_offset(LEGACY_DATA_OFFSET, index).unwrap();
        let mut entry = [0u8; ENTRY_SIZE];
        entry[0] = status;
        entry[1..9].copy_from_slice(&id.to_le_bytes());
        entry[ENTRY_HV_OFFSET..ENTRY_HV_OFFSET + BINARY_HV_BYTES].copy_from_slice(&hv.0);
        file.seek(SeekFrom::Start(offset as u64)).unwrap();
        file.write_all(&entry).unwrap();
    }

    fn create_legacy_fixture(path: &Path) {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap();
        file.set_len((LEGACY_DATA_OFFSET + ENTRY_SIZE * 8) as u64)
            .unwrap();
        file.write_all(&legacy_header_bytes(3, 2, 1)).unwrap();
        write_legacy_entry(&mut file, 0, STATUS_LIVE, 10, &BinaryHV::random(10));
        write_legacy_entry(&mut file, 1, STATUS_TOMBSTONE, 11, &BinaryHV::random(11));
        write_legacy_entry(&mut file, 2, STATUS_LIVE, 12, &BinaryHV::random(12));
        file.sync_all().unwrap();
    }

    #[test]
    fn migrates_live_entries_and_compacts_tombstones() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("legacy.hdc");
        create_legacy_fixture(&path);

        assert!(matches!(
            HdcStore::open(&path),
            Err(HdcStoreError::VersionMismatch {
                expected: VERSION,
                found: LEGACY_VERSION
            })
        ));

        let (store, report) = migrate_v1(&path).unwrap();
        assert_eq!(report.source_vector_count, 3);
        assert_eq!(report.migrated_live_count, 2);
        assert_eq!(report.discarded_tombstones, 1);
        assert_eq!(report.source_version, LEGACY_VERSION);
        assert_eq!(report.target_version, VERSION);
        assert_eq!(store.live_count(), 2);
        assert_eq!(store.tombstone_count(), 0);
        assert!(store.get(10).is_some());
        assert!(store.get(11).is_none());
        assert!(store.get(12).is_some());
    }

    #[test]
    fn current_format_is_not_migrated() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("current.hdc");
        drop(HdcStore::create(&path, StoreConfig::default()).unwrap());
        assert!(matches!(
            migrate_v1(&path),
            Err(HdcStoreError::MigrationNotRequired { version: VERSION })
        ));
    }

    #[test]
    fn invalid_legacy_entries_leave_source_untouched() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("legacy.hdc");
        create_legacy_fixture(&path);
        let before = std::fs::read(&path).unwrap();

        let mut file = OpenOptions::new().write(true).open(&path).unwrap();
        let offset = checked_entry_offset(LEGACY_DATA_OFFSET, 2).unwrap();
        file.seek(SeekFrom::Start(offset as u64)).unwrap();
        file.write_all(&[99]).unwrap();
        file.sync_all().unwrap();
        let corrupted = std::fs::read(&path).unwrap();
        assert_ne!(before, corrupted);

        assert!(matches!(
            migrate_v1(&path),
            Err(HdcStoreError::CorruptEntry { index: 2, .. })
        ));
        assert_eq!(std::fs::read(&path).unwrap(), corrupted);
    }

    #[cfg(unix)]
    #[test]
    fn migration_preserves_unix_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempdir().unwrap();
        let path = dir.path().join("legacy.hdc");
        create_legacy_fixture(&path);
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();

        let (store, _) = migrate_v1(&path).unwrap();
        drop(store);
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o640);
    }
}
