// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Non-mutating structural inspection for HdcStore files.

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use crate::HdcStoreError;
use crate::header::{
    DATA_OFFSET, ENTRY_SIZE, HEADER_SIZE, HeaderSlot, LEGACY_VERSION, MAGIC, STATUS_LIVE,
    STATUS_TOMBSTONE, StoreHeader, VERSION, checked_entry_offset,
};
use crate::locking::StoreLock;

/// Validation result for one redundant format-v2 header page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeaderSlotInspection {
    /// Header page inspected.
    pub slot: HeaderSlot,
    /// Whether semantic and checksum validation succeeded.
    pub valid: bool,
    /// Parsed generation when enough bytes were present.
    pub generation: Option<u64>,
    /// Validation error when the slot was invalid.
    pub error: Option<String>,
}

/// Structural issue discovered by [`inspect_store`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InspectionIssue {
    FileTooSmall {
        found: u64,
        required: u64,
    },
    LegacyFormatRequiresMigration {
        version: u32,
    },
    UnsupportedVersion {
        version: u32,
    },
    HeaderRedundancyDegraded {
        invalid_slot: HeaderSlot,
        reason: String,
    },
    NoValidHeader {
        primary: String,
        secondary: String,
    },
    HeaderConflict {
        generation: u64,
    },
    CommittedRegionTruncated {
        required: u64,
        found: u64,
    },
    InvalidEntryStatus {
        index: u64,
        status: u8,
    },
    DuplicateLiveId {
        index: u64,
        id: u64,
    },
    CountMismatch {
        declared_live: u64,
        scanned_live: u64,
        declared_tombstones: u64,
        scanned_tombstones: u64,
    },
    TrailingCommittedEntries {
        count: u64,
    },
}

/// Read-only structural report for a store file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoreInspection {
    /// File inspected.
    pub path: PathBuf,
    /// Current file length.
    pub file_len: u64,
    /// Detected format version when a recognizable header was available.
    pub detected_version: Option<u32>,
    /// Primary header validation result when the v2 metadata region exists.
    pub primary_header: Option<HeaderSlotInspection>,
    /// Secondary header validation result when the v2 metadata region exists.
    pub secondary_header: Option<HeaderSlotInspection>,
    /// Newest valid slot selected by normal v2 generation rules.
    pub selected_slot: Option<HeaderSlot>,
    /// Selected header generation.
    pub selected_generation: Option<u64>,
    /// Counts declared by the selected header.
    pub declared_vector_count: Option<u64>,
    pub declared_live_count: Option<u64>,
    pub declared_tombstone_count: Option<u64>,
    /// Counts observed by scanning the committed range.
    pub scanned_live_count: Option<u64>,
    pub scanned_tombstone_count: Option<u64>,
    /// Contiguous committed-looking entries after the visibility boundary.
    pub trailing_committed_entries: u64,
    /// All structural issues found without modifying the file.
    pub issues: Vec<InspectionIssue>,
}

impl StoreInspection {
    fn new(path: PathBuf, file_len: u64) -> Self {
        Self {
            path,
            file_len,
            detected_version: None,
            primary_header: None,
            secondary_header: None,
            selected_slot: None,
            selected_generation: None,
            declared_vector_count: None,
            declared_live_count: None,
            declared_tombstone_count: None,
            scanned_live_count: None,
            scanned_tombstone_count: None,
            trailing_committed_entries: 0,
            issues: Vec::new(),
        }
    }

    /// Whether no structural issue was found.
    pub fn is_clean(&self) -> bool {
        self.issues.is_empty()
    }

    /// Whether the report contains a condition handled by metadata recovery.
    pub fn metadata_recovery_may_help(&self) -> bool {
        let has_recoverable_issue = self.issues.iter().any(|issue| {
            matches!(
                issue,
                InspectionIssue::HeaderRedundancyDegraded { .. }
                    | InspectionIssue::CountMismatch { .. }
            )
        });
        let has_blocking_issue = self.issues.iter().any(|issue| {
            matches!(
                issue,
                InspectionIssue::FileTooSmall { .. }
                    | InspectionIssue::LegacyFormatRequiresMigration { .. }
                    | InspectionIssue::UnsupportedVersion { .. }
                    | InspectionIssue::NoValidHeader { .. }
                    | InspectionIssue::HeaderConflict { .. }
                    | InspectionIssue::CommittedRegionTruncated { .. }
                    | InspectionIssue::InvalidEntryStatus { .. }
                    | InspectionIssue::DuplicateLiveId { .. }
            )
        });
        has_recoverable_issue && !has_blocking_issue
    }
}

/// Inspect a store under a shared advisory lock without modifying it.
pub fn inspect_store(path: impl AsRef<Path>) -> Result<StoreInspection, HdcStoreError> {
    let path = path.as_ref().to_path_buf();
    let _coordination_lock = StoreLock::shared(&path)?;
    let file = OpenOptions::new().read(true).open(&path)?;
    lock_shared(&file, &path)?;
    let file_len = file.metadata()?.len();
    let mut report = StoreInspection::new(path, file_len);

    if file_len < 12 {
        report.issues.push(InspectionIssue::FileTooSmall {
            found: file_len,
            required: 12,
        });
        return Ok(report);
    }

    let mmap = unsafe { Mmap::map(&file)? };
    let raw_magic: [u8; 8] = mmap[0..8]
        .try_into()
        .expect("minimum inspection length checked");
    let raw_version = u32::from_le_bytes(
        mmap[8..12]
            .try_into()
            .expect("minimum inspection length checked"),
    );
    let legacy_signature = raw_magic == MAGIC && raw_version == LEGACY_VERSION;

    if mmap.len() < DATA_OFFSET {
        report.detected_version = (raw_magic == MAGIC).then_some(raw_version);
        if legacy_signature {
            report
                .issues
                .push(InspectionIssue::LegacyFormatRequiresMigration {
                    version: LEGACY_VERSION,
                });
        } else if raw_magic == MAGIC && raw_version != VERSION {
            report.issues.push(InspectionIssue::UnsupportedVersion {
                version: raw_version,
            });
        } else {
            report.issues.push(InspectionIssue::FileTooSmall {
                found: file_len,
                required: DATA_OFFSET as u64,
            });
        }
        return Ok(report);
    }

    let (primary_report, primary) = inspect_header_slot(&mmap, HeaderSlot::Primary);
    let (secondary_report, secondary) = inspect_header_slot(&mmap, HeaderSlot::Secondary);
    report.primary_header = Some(primary_report);
    report.secondary_header = Some(secondary_report);

    let selected = match (primary, secondary) {
        (Ok(primary), Ok(secondary)) => match primary.generation.cmp(&secondary.generation) {
            std::cmp::Ordering::Greater => Some((primary, HeaderSlot::Primary)),
            std::cmp::Ordering::Less => Some((secondary, HeaderSlot::Secondary)),
            std::cmp::Ordering::Equal if primary == secondary => {
                Some((primary, HeaderSlot::Primary))
            }
            std::cmp::Ordering::Equal => {
                report.issues.push(InspectionIssue::HeaderConflict {
                    generation: primary.generation,
                });
                None
            }
        },
        (Ok(primary), Err(error)) => {
            report
                .issues
                .push(InspectionIssue::HeaderRedundancyDegraded {
                    invalid_slot: HeaderSlot::Secondary,
                    reason: error,
                });
            Some((primary, HeaderSlot::Primary))
        }
        (Err(error), Ok(secondary)) => {
            report
                .issues
                .push(InspectionIssue::HeaderRedundancyDegraded {
                    invalid_slot: HeaderSlot::Primary,
                    reason: error,
                });
            Some((secondary, HeaderSlot::Secondary))
        }
        (Err(primary), Err(secondary)) => {
            if legacy_signature {
                report.detected_version = Some(LEGACY_VERSION);
                report
                    .issues
                    .push(InspectionIssue::LegacyFormatRequiresMigration {
                        version: LEGACY_VERSION,
                    });
            } else {
                report
                    .issues
                    .push(InspectionIssue::NoValidHeader { primary, secondary });
            }
            None
        }
    };

    let Some((header, slot)) = selected else {
        return Ok(report);
    };
    report.detected_version = Some(VERSION);
    report.selected_slot = Some(slot);
    report.selected_generation = Some(header.generation);
    report.declared_vector_count = Some(header.vector_count);
    report.declared_live_count = Some(header.live_count);
    report.declared_tombstone_count = Some(header.tombstone_count);

    let required_len = header.required_file_len()?;
    if required_len > mmap.len() {
        report
            .issues
            .push(InspectionIssue::CommittedRegionTruncated {
                required: required_len as u64,
                found: file_len,
            });
        return Ok(report);
    }

    let mut live_ids = HashSet::new();
    let mut scanned_live = 0u64;
    let mut scanned_tombstones = 0u64;
    for index in 0..header.vector_count {
        let offset = header.checked_entry_offset(index)?;
        match mmap[offset] {
            STATUS_LIVE => {
                scanned_live =
                    scanned_live
                        .checked_add(1)
                        .ok_or(HdcStoreError::ArithmeticOverflow {
                            context: "inspection live entry count",
                        })?;
                let id = read_entry_id(&mmap, offset, index)?;
                if !live_ids.insert(id) {
                    report
                        .issues
                        .push(InspectionIssue::DuplicateLiveId { index, id });
                }
            }
            STATUS_TOMBSTONE => {
                scanned_tombstones =
                    scanned_tombstones
                        .checked_add(1)
                        .ok_or(HdcStoreError::ArithmeticOverflow {
                            context: "inspection tombstone count",
                        })?;
            }
            status => report
                .issues
                .push(InspectionIssue::InvalidEntryStatus { index, status }),
        }
    }
    report.scanned_live_count = Some(scanned_live);
    report.scanned_tombstone_count = Some(scanned_tombstones);
    if scanned_live != header.live_count || scanned_tombstones != header.tombstone_count {
        report.issues.push(InspectionIssue::CountMismatch {
            declared_live: header.live_count,
            scanned_live,
            declared_tombstones: header.tombstone_count,
            scanned_tombstones,
        });
    }

    let trailing = count_trailing_entries(&mmap, header.vector_count)?;
    report.trailing_committed_entries = trailing;
    if trailing > 0 {
        report
            .issues
            .push(InspectionIssue::TrailingCommittedEntries { count: trailing });
    }
    Ok(report)
}

fn inspect_header_slot(
    mmap: &Mmap,
    slot: HeaderSlot,
) -> (HeaderSlotInspection, Result<StoreHeader, String>) {
    let offset = slot.page_offset();
    let bytes: [u8; HEADER_SIZE] = mmap[offset..offset + HEADER_SIZE]
        .try_into()
        .expect("format-v2 inspection region checked");
    let parsed = StoreHeader::from_bytes(&bytes);
    match StoreHeader::validate_serialized(&bytes) {
        Ok(header) => (
            HeaderSlotInspection {
                slot,
                valid: true,
                generation: Some(header.generation),
                error: None,
            },
            Ok(header),
        ),
        Err(error) => {
            let reason = error.to_string();
            (
                HeaderSlotInspection {
                    slot,
                    valid: false,
                    generation: (parsed.magic == MAGIC).then_some(parsed.generation),
                    error: Some(reason.clone()),
                },
                Err(reason),
            )
        }
    }
}

fn read_entry_id(mmap: &Mmap, offset: usize, index: u64) -> Result<u64, HdcStoreError> {
    let end = offset
        .checked_add(9)
        .ok_or(HdcStoreError::ArithmeticOverflow {
            context: "inspection entry id end offset",
        })?;
    let bytes: [u8; 8] = mmap
        .get(offset + 1..end)
        .ok_or_else(|| HdcStoreError::CorruptEntry {
            index,
            reason: "inspection entry id extends beyond mapped file".into(),
        })?
        .try_into()
        .expect("inspection entry id slice is exactly eight bytes");
    Ok(u64::from_le_bytes(bytes))
}

fn count_trailing_entries(mmap: &Mmap, vector_count: u64) -> Result<u64, HdcStoreError> {
    let mut count = 0u64;
    let mut index = vector_count;
    loop {
        let offset = checked_entry_offset(DATA_OFFSET, index)?;
        let end = offset
            .checked_add(ENTRY_SIZE)
            .ok_or(HdcStoreError::ArithmeticOverflow {
                context: "inspection trailing entry end offset",
            })?;
        if end > mmap.len() {
            return Ok(count);
        }
        match mmap[offset] {
            STATUS_LIVE | STATUS_TOMBSTONE => {
                count = count
                    .checked_add(1)
                    .ok_or(HdcStoreError::ArithmeticOverflow {
                        context: "inspection trailing entry count",
                    })?;
                index = index
                    .checked_add(1)
                    .ok_or(HdcStoreError::ArithmeticOverflow {
                        context: "inspection trailing entry index",
                    })?;
            }
            _ => return Ok(count),
        }
    }
}

fn lock_shared(file: &File, path: &Path) -> Result<(), HdcStoreError> {
    match file.try_lock_shared() {
        Ok(()) => Ok(()),
        Err(std::fs::TryLockError::WouldBlock) => Err(HdcStoreError::StoreLocked {
            path: path.to_path_buf(),
        }),
        Err(std::fs::TryLockError::Error(error)) => Err(HdcStoreError::Io(error)),
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Seek, SeekFrom, Write};

    use symthaea_core::hdc::BinaryHV;
    use tempfile::tempdir;

    use super::*;
    use crate::{HdcStore, StoreConfig};

    #[test]
    fn clean_store_inspection_is_clean() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        {
            let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
            store.append(1, &BinaryHV::random(1)).unwrap();
        }

        let report = inspect_store(&path).unwrap();
        assert!(report.is_clean());
        assert_eq!(report.detected_version, Some(VERSION));
        assert_eq!(report.declared_live_count, Some(1));
        assert_eq!(report.scanned_live_count, Some(1));
    }

    #[test]
    fn inspection_reports_degraded_header_without_repairing() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        drop(HdcStore::create(&path, StoreConfig::default()).unwrap());

        let before_generation = HdcStore::open(&path).unwrap().header_generation();
        let mut file = OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(
            (crate::header::HEADER_PAGE_SIZE + 100) as u64,
        ))
        .unwrap();
        file.write_all(&[0x91]).unwrap();
        file.sync_all().unwrap();

        let report = inspect_store(&path).unwrap();
        assert!(!report.is_clean());
        assert!(report.metadata_recovery_may_help());
        assert!(matches!(
            report.issues.first(),
            Some(InspectionIssue::HeaderRedundancyDegraded { .. })
        ));
        let store = HdcStore::open(&path).unwrap();
        assert_eq!(store.header_generation(), before_generation);
    }

    #[test]
    fn inspection_reports_count_mismatch_and_trailing_entry() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        {
            let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
            store.append(1, &BinaryHV::random(1)).unwrap();
        }

        let mut file = OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(DATA_OFFSET as u64)).unwrap();
        file.write_all(&[STATUS_TOMBSTONE]).unwrap();
        let trailing_offset = checked_entry_offset(DATA_OFFSET, 1).unwrap();
        let mut entry = [0u8; ENTRY_SIZE];
        entry[0] = STATUS_LIVE;
        entry[1..9].copy_from_slice(&2u64.to_le_bytes());
        file.seek(SeekFrom::Start(trailing_offset as u64)).unwrap();
        file.write_all(&entry).unwrap();
        file.sync_all().unwrap();

        let report = inspect_store(&path).unwrap();
        assert!(report.metadata_recovery_may_help());
        assert_eq!(report.trailing_committed_entries, 1);
        assert!(
            report
                .issues
                .iter()
                .any(|issue| matches!(issue, InspectionIssue::CountMismatch { .. }))
        );
        assert!(report.issues.iter().any(|issue| matches!(
            issue,
            InspectionIssue::TrailingCommittedEntries { count: 1 }
        )));
    }
    #[test]
    fn inspection_respects_path_stable_writer_lock() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        let _store = HdcStore::create(&path, StoreConfig::default()).unwrap();
        assert!(matches!(
            inspect_store(&path),
            Err(HdcStoreError::StoreLocked { .. })
        ));
    }
}
