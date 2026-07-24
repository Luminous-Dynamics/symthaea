// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shared-lock, read-only access to a committed HdcStore generation.

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use symthaea_core::hdc::BinaryHV;

use crate::HdcStoreError;
use crate::header::{
    DATA_OFFSET, ENTRY_HV_OFFSET, HEADER_SIZE, HeaderSlot, MAGIC, STATUS_LIVE, STATUS_TOMBSTONE,
    StoreHeader, VERSION,
};
use crate::locking::StoreLock;
use crate::lsh_persistent::{DEFAULT_LSH_SEED, LshIndex};
use crate::lsh_snapshot::{
    IndexOpenPolicy, IndexStatus, LshSnapshotMetadata, fingerprint_ordered, load_lsh_snapshot,
    lsh_snapshot_path,
};
use crate::search::{ApproximateSearchOptions, SearchOutcome};
use crate::transaction::batch_journal_path;

const BINARY_HV_BYTES: usize = 2048;

/// A validated, zero-copy, read-only store handle.
///
/// The handle owns shared advisory locks on both the path-stable coordination
/// inode and the current data inode. Mutable open, compaction, migration, and
/// recovery therefore remain excluded for the complete lifetime of the reader.
pub struct HdcStoreReader {
    mmap: Mmap,
    header: StoreHeader,
    entries: Vec<(u64, u64)>,
    path: PathBuf,
    active_header_slot: HeaderSlot,
    lsh: Option<LshIndex>,
    index_status: Option<IndexStatus>,
    #[allow(dead_code)]
    file: File,
    #[allow(dead_code)]
    coordination_lock: StoreLock,
}

impl HdcStoreReader {
    /// Open and strictly validate a committed format-v2 store for shared reads.
    ///
    /// Pending batch journals are rejected because only recovering mutable open
    /// is permitted to decide whether an interrupted batch must be rolled back
    /// or finalized.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, HdcStoreError> {
        Self::open_internal(path.as_ref(), None)
    }

    /// Open with an explicit policy for the optional LSH signature sidecar.
    ///
    /// Unlike [`Self::open`], this opt-in path constructs an ANN index.
    pub fn open_with_index_policy(
        path: impl AsRef<Path>,
        index_policy: IndexOpenPolicy,
    ) -> Result<Self, HdcStoreError> {
        Self::open_internal(path.as_ref(), Some(index_policy))
    }

    fn open_internal(
        path: &Path,
        index_policy: Option<IndexOpenPolicy>,
    ) -> Result<Self, HdcStoreError> {
        let path = path.to_path_buf();
        let coordination_lock = StoreLock::shared(&path)?;
        if batch_journal_path(&path).exists() {
            return Err(HdcStoreError::PendingBatchTransaction {
                path: batch_journal_path(&path),
            });
        }

        let file = OpenOptions::new().read(true).open(&path)?;
        lock_shared(&file, &path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        if mmap.len() < DATA_OFFSET {
            return Err(HdcStoreError::InvalidHeader {
                reason: format!(
                    "format-v2 file is {} bytes; at least {DATA_OFFSET} bytes are required",
                    mmap.len()
                ),
            });
        }

        let (header, active_header_slot) = select_header(&mmap)?;
        let required_len = header.required_file_len()?;
        if required_len > mmap.len() {
            return Err(HdcStoreError::InvalidHeader {
                reason: format!(
                    "committed entries require {required_len} bytes, but file contains {} bytes",
                    mmap.len()
                ),
            });
        }

        let entries = scan_live_entries(&mmap, &header)?;
        let scanned_live =
            u64::try_from(entries.len()).map_err(|_| HdcStoreError::ArithmeticOverflow {
                context: "read-only live count conversion",
            })?;
        let scanned_tombstones =
            header
                .vector_count
                .checked_sub(scanned_live)
                .ok_or(HdcStoreError::InvalidHeader {
                    reason: "read-only scan produced more live entries than committed entries"
                        .into(),
                })?;
        if scanned_live != header.live_count || scanned_tombstones != header.tombstone_count {
            return Err(HdcStoreError::InvalidHeader {
                reason: format!(
                    "entry scan disagrees with header: live {scanned_live}/{}, tombstones {scanned_tombstones}/{}",
                    header.live_count, header.tombstone_count
                ),
            });
        }

        let (lsh, index_status) = match index_policy {
            Some(policy) => {
                let (index, status) =
                    load_or_rebuild_index(&mmap, &header, &entries, &path, policy)?;
                (Some(index), Some(status))
            }
            None => (None, None),
        };

        Ok(Self {
            mmap,
            header,
            entries,
            path,
            active_header_slot,
            lsh,
            index_status,
            file,
            coordination_lock,
        })
    }

    /// Canonical path held by this reader.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Header generation selected when the reader opened.
    pub const fn generation(&self) -> u64 {
        self.header.generation
    }

    /// Header page that supplied the selected generation.
    pub const fn active_header_slot(&self) -> HeaderSlot {
        self.active_header_slot
    }

    /// Number of committed entries, including tombstones.
    pub const fn vector_count(&self) -> u64 {
        self.header.vector_count
    }

    /// Number of live entries exposed by this reader.
    pub const fn live_count(&self) -> u64 {
        self.header.live_count
    }

    /// Number of committed tombstones.
    pub const fn tombstone_count(&self) -> u64 {
        self.header.tombstone_count
    }

    /// LSH bands recorded in the canonical store header.
    pub const fn lsh_bands(&self) -> u32 {
        self.header.lsh_bands
    }

    /// LSH rows recorded in the canonical store header.
    pub const fn lsh_rows(&self) -> u32 {
        self.header.lsh_rows
    }

    /// Whether the committed live set is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Deterministic ascending live IDs.
    pub fn ids(&self) -> impl ExactSizeIterator<Item = u64> + '_ {
        self.entries.iter().map(|(id, _)| *id)
    }

    /// Get a zero-copy vector reference by ID.
    pub fn get(&self, id: u64) -> Option<&BinaryHV> {
        let position = self
            .entries
            .binary_search_by_key(&id, |(entry_id, _)| *entry_id)
            .ok()?;
        hv_at(&self.mmap, &self.header, self.entries[position].1).ok()
    }

    /// Iterate live vectors in ascending ID order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (u64, &BinaryHV)> + '_ {
        self.entries.iter().map(|(id, index)| {
            let hv = hv_at(&self.mmap, &self.header, *index)
                .expect("reader entries were validated during open");
            (*id, hv)
        })
    }

    /// Deterministic checksum of the logical live vector set.
    pub fn content_checksum(&self) -> crate::StoreContentChecksum {
        crate::content_checksum::checksum_ordered(self.iter(), self.live_count())
    }

    /// Current source and compatibility state of the in-memory ANN index.
    pub fn index_status(&self) -> Option<&IndexStatus> {
        self.index_status.as_ref()
    }

    /// Whether this reader was opened with an ANN index.
    pub fn has_approximate_index(&self) -> bool {
        self.lsh.is_some()
    }

    /// Estimated candidate probability for a target Hamming agreement.
    pub fn estimated_lsh_candidate_probability(&self, hamming_agreement: f64) -> Option<f64> {
        self.lsh
            .as_ref()
            .map(|index| index.estimated_candidate_probability(hamming_agreement))
    }

    /// Explicit approximate search using the validated or rebuilt LSH index.
    pub fn scan_similar_approx(
        &self,
        query: &BinaryHV,
        top_k: usize,
        options: ApproximateSearchOptions,
    ) -> SearchOutcome {
        let total_live = self.entries.len();
        if top_k == 0 || total_live == 0 {
            return SearchOutcome {
                neighbors: Vec::new(),
                examined: 0,
                total_live,
                exact: total_live == 0,
                fell_back_to_exact: false,
            };
        }

        let Some(lsh) = &self.lsh else {
            return SearchOutcome {
                neighbors: self.scan_similar(query, top_k),
                examined: total_live,
                total_live,
                exact: true,
                fell_back_to_exact: true,
            };
        };
        let candidates = lsh.query_candidates(query);
        let minimum_candidates = top_k.saturating_mul(options.candidate_multiplier);
        let should_fallback = (options.fallback_on_empty && candidates.is_empty())
            || (options.candidate_multiplier > 0
                && candidates.len() < minimum_candidates
                && candidates.len() < total_live);
        if should_fallback {
            return SearchOutcome {
                neighbors: self.scan_similar(query, top_k),
                examined: total_live,
                total_live,
                exact: true,
                fell_back_to_exact: true,
            };
        }

        let mut neighbors: Vec<(u64, f32)> = candidates
            .iter()
            .filter_map(|id| self.get(*id).map(|hv| (*id, query.similarity(hv))))
            .collect();
        let examined = neighbors.len();
        neighbors.sort_unstable_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        neighbors.truncate(top_k);
        SearchOutcome {
            neighbors,
            examined,
            total_live,
            exact: examined == total_live,
            fell_back_to_exact: false,
        }
    }

    /// Exact deterministic nearest-neighbor search over the pinned generation.
    pub fn scan_similar(&self, query: &BinaryHV, top_k: usize) -> Vec<(u64, f32)> {
        if top_k == 0 {
            return Vec::new();
        }
        let mut results: Vec<(u64, f32)> = self
            .iter()
            .map(|(id, hv)| (id, query.similarity(hv)))
            .collect();
        results.sort_unstable_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        results.truncate(top_k);
        results
    }
}

fn select_header(mmap: &Mmap) -> Result<(StoreHeader, HeaderSlot), HdcStoreError> {
    let primary = read_header_slot(mmap, HeaderSlot::Primary);
    let secondary = read_header_slot(mmap, HeaderSlot::Secondary);
    match (primary, secondary) {
        (Ok(primary), Ok(secondary)) => match primary.generation.cmp(&secondary.generation) {
            std::cmp::Ordering::Greater => Ok((primary, HeaderSlot::Primary)),
            std::cmp::Ordering::Less => Ok((secondary, HeaderSlot::Secondary)),
            std::cmp::Ordering::Equal if primary == secondary => Ok((primary, HeaderSlot::Primary)),
            std::cmp::Ordering::Equal => Err(HdcStoreError::HeaderConflict {
                generation: primary.generation,
            }),
        },
        (Ok(primary), Err(_)) => Ok((primary, HeaderSlot::Primary)),
        (Err(_), Ok(secondary)) => Ok((secondary, HeaderSlot::Secondary)),
        (Err(primary), Err(secondary)) => Err(HdcStoreError::NoValidHeader {
            primary: primary.to_string(),
            secondary: secondary.to_string(),
        }),
    }
}

fn read_header_slot(mmap: &Mmap, slot: HeaderSlot) -> Result<StoreHeader, HdcStoreError> {
    let offset = slot.page_offset();
    let bytes: [u8; HEADER_SIZE] = mmap[offset..offset + HEADER_SIZE]
        .try_into()
        .expect("format-v2 metadata region checked before header read");
    let parsed = StoreHeader::from_bytes(&bytes);
    if parsed.magic != MAGIC {
        return Err(HdcStoreError::InvalidHeader {
            reason: format!("{:?} header has bad magic bytes", slot),
        });
    }
    if parsed.version != VERSION {
        return Err(HdcStoreError::VersionMismatch {
            expected: VERSION,
            found: parsed.version,
        });
    }
    StoreHeader::validate_serialized(&bytes)
}

fn scan_live_entries(mmap: &Mmap, header: &StoreHeader) -> Result<Vec<(u64, u64)>, HdcStoreError> {
    let capacity =
        usize::try_from(header.live_count).map_err(|_| HdcStoreError::ArithmeticOverflow {
            context: "read-only live entry capacity",
        })?;
    let mut entries = Vec::with_capacity(capacity);
    let mut ids = HashSet::with_capacity(capacity);
    for index in 0..header.vector_count {
        let offset = header.checked_entry_offset(index)?;
        match mmap[offset] {
            STATUS_LIVE => {
                let id = read_entry_id(mmap, offset, index)?;
                if !ids.insert(id) {
                    return Err(HdcStoreError::CorruptEntry {
                        index,
                        reason: format!("duplicate live id {id}"),
                    });
                }
                entries.push((id, index));
            }
            STATUS_TOMBSTONE => {}
            status => {
                return Err(HdcStoreError::CorruptEntry {
                    index,
                    reason: format!("invalid committed status byte {status}"),
                });
            }
        }
    }
    entries.sort_unstable_by_key(|(id, _)| *id);
    Ok(entries)
}

fn read_entry_id(mmap: &Mmap, offset: usize, index: u64) -> Result<u64, HdcStoreError> {
    let end = offset
        .checked_add(9)
        .ok_or(HdcStoreError::ArithmeticOverflow {
            context: "read-only entry id end offset",
        })?;
    let bytes: [u8; 8] = mmap
        .get(offset + 1..end)
        .ok_or_else(|| HdcStoreError::CorruptEntry {
            index,
            reason: "entry id extends beyond mapped file".into(),
        })?
        .try_into()
        .expect("entry id slice is exactly eight bytes");
    Ok(u64::from_le_bytes(bytes))
}

fn hv_at<'a>(
    mmap: &'a Mmap,
    header: &StoreHeader,
    index: u64,
) -> Result<&'a BinaryHV, HdcStoreError> {
    let offset = header.checked_entry_offset(index)?;
    let hv_start =
        offset
            .checked_add(ENTRY_HV_OFFSET)
            .ok_or(HdcStoreError::ArithmeticOverflow {
                context: "read-only BinaryHV start offset",
            })?;
    let hv_end =
        hv_start
            .checked_add(BINARY_HV_BYTES)
            .ok_or(HdcStoreError::ArithmeticOverflow {
                context: "read-only BinaryHV end offset",
            })?;
    let bytes = mmap
        .get(hv_start..hv_end)
        .ok_or_else(|| HdcStoreError::CorruptEntry {
            index,
            reason: "BinaryHV extends beyond mapped file".into(),
        })?;
    let ptr = bytes.as_ptr();
    if !(ptr as usize).is_multiple_of(32) {
        return Err(HdcStoreError::CorruptEntry {
            index,
            reason: format!("BinaryHV pointer is not 32-byte aligned at offset {hv_start}"),
        });
    }

    // SAFETY: the mmap base is page-aligned, format-v2 entry payloads are
    // 32-byte aligned, the slice is exactly BinaryHV's byte length, and the
    // returned reference cannot outlive the immutable mmap borrow.
    Ok(unsafe { &*(ptr as *const BinaryHV) })
}

fn load_or_rebuild_index(
    mmap: &Mmap,
    header: &StoreHeader,
    entries: &[(u64, u64)],
    path: &Path,
    policy: IndexOpenPolicy,
) -> Result<(LshIndex, IndexStatus), HdcStoreError> {
    let records: Result<Vec<_>, HdcStoreError> = entries
        .iter()
        .map(|(id, index)| Ok((*id, hv_at(mmap, header, *index)?)))
        .collect();
    let metadata = LshSnapshotMetadata {
        store_generation: header.generation,
        vector_count: header.vector_count,
        live_count: header.live_count,
        tombstone_count: header.tombstone_count,
        lsh_bands: header.lsh_bands,
        lsh_rows: header.lsh_rows,
        lsh_seed: DEFAULT_LSH_SEED,
        store_fingerprint: fingerprint_ordered(records?, header.live_count),
    };
    let mut ignored_snapshot_error = None;

    if policy != IndexOpenPolicy::Rebuild {
        match load_lsh_snapshot(path, metadata) {
            Ok(Some(snapshot)) => {
                let mut index = LshIndex::new(
                    header.lsh_bands as usize,
                    header.lsh_rows as usize,
                    snapshot.metadata.lsh_seed,
                )?;
                if snapshot.entries.len() != entries.len() {
                    let error = HdcStoreError::InvalidIndexSnapshot {
                        path: lsh_snapshot_path(path),
                        reason: format!(
                            "snapshot contains {} IDs but reader contains {} live IDs",
                            snapshot.entries.len(),
                            entries.len()
                        ),
                    };
                    if policy == IndexOpenPolicy::RequireSnapshot {
                        return Err(error);
                    }
                    ignored_snapshot_error = Some(error.to_string());
                } else {
                    let mut valid = true;
                    for (id, signature) in snapshot.entries {
                        if entries
                            .binary_search_by_key(&id, |(entry_id, _)| *entry_id)
                            .is_err()
                        {
                            valid = false;
                            ignored_snapshot_error = Some(
                                HdcStoreError::InvalidIndexSnapshot {
                                    path: lsh_snapshot_path(path),
                                    reason: format!("snapshot contains non-live id {id}"),
                                }
                                .to_string(),
                            );
                            break;
                        }
                        if let Err(error) = index.insert_signature(id, signature) {
                            valid = false;
                            ignored_snapshot_error = Some(error.to_string());
                            break;
                        }
                    }
                    if valid {
                        return Ok((index, IndexStatus::loaded(path)));
                    }
                    if policy == IndexOpenPolicy::RequireSnapshot {
                        return Err(HdcStoreError::InvalidIndexSnapshot {
                            path: lsh_snapshot_path(path),
                            reason: ignored_snapshot_error
                                .unwrap_or_else(|| "snapshot validation failed".into()),
                        });
                    }
                }
            }
            Ok(None) if policy == IndexOpenPolicy::RequireSnapshot => {
                return Err(HdcStoreError::InvalidIndexSnapshot {
                    path: lsh_snapshot_path(path),
                    reason: "required LSH snapshot is missing".into(),
                });
            }
            Ok(None) => {}
            Err(error) if policy == IndexOpenPolicy::RequireSnapshot => return Err(error),
            Err(error) => ignored_snapshot_error = Some(error.to_string()),
        }
    }

    let mut index = LshIndex::new(
        header.lsh_bands as usize,
        header.lsh_rows as usize,
        DEFAULT_LSH_SEED,
    )?;
    for (id, entry_index) in entries {
        index.insert(*id, hv_at(mmap, header, *entry_index)?);
    }
    Ok((index, IndexStatus::rebuilt(path, ignored_snapshot_error)))
}

fn lock_shared(file: &File, path: &Path) -> Result<(), HdcStoreError> {
    match file.try_lock_shared() {
        Ok(()) => Ok(()),
        Err(std::fs::TryLockError::WouldBlock) => Err(HdcStoreError::StoreLocked {
            path: path.to_path_buf(),
        }),
        Err(std::fs::TryLockError::Error(error)) => Err(error.into()),
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;
    use crate::{HdcStore, StoreConfig};

    #[test]
    fn shared_reader_exposes_deterministic_zero_copy_data() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("reader.hdc");
        {
            let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
            store.append(9, &BinaryHV::random(9)).unwrap();
            store.append(3, &BinaryHV::random(3)).unwrap();
            store.append(7, &BinaryHV::random(7)).unwrap();
            store.delete(7).unwrap();
        }

        let reader = HdcStoreReader::open(&path).unwrap();
        assert_eq!(reader.ids().collect::<Vec<_>>(), vec![3, 9]);
        assert_eq!(reader.live_count(), 2);
        assert_eq!(reader.tombstone_count(), 1);
        assert_eq!(reader.get(3), Some(&BinaryHV::random(3)));
        assert!(reader.get(7).is_none());
        assert_eq!(reader.scan_similar(&BinaryHV::random(9), 1)[0].0, 9);
    }

    #[test]
    fn multiple_shared_readers_coexist_and_exclude_writers() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("reader-lock.hdc");
        drop(HdcStore::create(&path, StoreConfig::default()).unwrap());

        let first = HdcStoreReader::open(&path).unwrap();
        let second = HdcStoreReader::open(&path).unwrap();
        assert_eq!(first.generation(), second.generation());
        assert!(matches!(
            HdcStore::open(&path),
            Err(HdcStoreError::StoreLocked { .. })
        ));
        drop(first);
        assert!(matches!(
            HdcStore::open(&path),
            Err(HdcStoreError::StoreLocked { .. })
        ));
        drop(second);
        assert!(HdcStore::open(&path).is_ok());
    }

    #[test]
    fn ordinary_shared_open_does_not_build_an_ann_index() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("reader-exact-only.hdc");
        {
            let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
            store.append(1, &BinaryHV::random(1)).unwrap();
        }
        let reader = HdcStoreReader::open(&path).unwrap();
        assert!(!reader.has_approximate_index());
        assert!(reader.index_status().is_none());
        let outcome = reader.scan_similar_approx(
            &BinaryHV::random(1),
            1,
            ApproximateSearchOptions::default(),
        );
        assert!(outcome.exact);
        assert!(outcome.fell_back_to_exact);
        assert_eq!(outcome.neighbors[0].0, 1);
    }

    #[test]
    fn shared_reader_loads_a_checkpointed_snapshot() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("reader-snapshot.hdc");
        {
            let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
            for id in 0..16 {
                store.append(id, &BinaryHV::random(id)).unwrap();
            }
            store.checkpoint_lsh().unwrap();
        }
        let reader =
            HdcStoreReader::open_with_index_policy(&path, IndexOpenPolicy::RequireSnapshot)
                .unwrap();
        assert_eq!(
            reader.index_status().unwrap().source,
            crate::IndexLoadSource::Snapshot
        );
        let outcome = reader.scan_similar_approx(
            &BinaryHV::random(3),
            3,
            ApproximateSearchOptions::default(),
        );
        assert_eq!(outcome.neighbors[0].0, 3);
    }

    #[test]
    fn shared_reader_rebuilds_when_snapshot_is_absent() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("reader-rebuild.hdc");
        {
            let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
            store.append(1, &BinaryHV::random(1)).unwrap();
        }
        let reader =
            HdcStoreReader::open_with_index_policy(&path, IndexOpenPolicy::Rebuild).unwrap();
        assert_eq!(
            reader.index_status().unwrap().source,
            crate::IndexLoadSource::Rebuilt
        );
    }

    #[test]
    fn checksum_is_stable_across_compaction() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("checksum.hdc");
        {
            let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
            store.append(10, &BinaryHV::random(10)).unwrap();
            store.append(20, &BinaryHV::random(20)).unwrap();
            store.append(30, &BinaryHV::random(30)).unwrap();
            store.delete(20).unwrap();
        }
        let before = HdcStoreReader::open(&path).unwrap().content_checksum();
        {
            let mut store = HdcStore::open(&path).unwrap();
            store.compact().unwrap();
        }
        let after = HdcStoreReader::open(&path).unwrap().content_checksum();
        assert_eq!(before, after);
    }

    #[test]
    fn mutable_writer_excludes_shared_reader() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("writer-lock.hdc");
        let store = HdcStore::create(&path, StoreConfig::default()).unwrap();
        assert!(matches!(
            HdcStoreReader::open(&path),
            Err(HdcStoreError::StoreLocked { .. })
        ));
        drop(store);
        assert!(HdcStoreReader::open(&path).is_ok());
    }
}
