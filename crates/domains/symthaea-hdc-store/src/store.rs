// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HdcStore -- mmap'd contiguous BinaryHV storage.
//!
//! File layout for format version 2:
//! - primary checksummed header page: 4096 bytes
//! - secondary checksummed header page: 4096 bytes
//! - fixed-size entries: 32-byte metadata + 2048-byte BinaryHV
//!
//! `get()` returns a zero-copy reference directly into the mapped region.
//! Mutable stores hold path-stable and data-inode advisory locks for their lifetime.

use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use memmap2::MmapMut;
use symthaea_core::hdc::BinaryHV;

use crate::HdcStoreError;
use crate::batch::{BatchCommitReport, WriteBatch};
use crate::compaction::compaction_stats;
use crate::fault::{self, FailPoint};
use crate::header::{
    DATA_OFFSET, ENTRY_HV_OFFSET, ENTRY_SIZE, HEADER_PAGE_SIZE, HEADER_SIZE, HeaderSlot,
    LEGACY_VERSION, MAGIC, STATUS_LIVE, STATUS_TOMBSTONE, StoreHeader, VERSION,
};
use crate::health::StoreHealth;
use crate::locking::StoreLock;
use crate::lsh_persistent::{DEFAULT_LSH_SEED, LshIndex, validate_lsh_config};
use crate::lsh_snapshot::{
    IndexOpenPolicy, IndexStatus, LshSnapshot, LshSnapshotMetadata, fingerprint_ordered,
    load_lsh_snapshot, lsh_snapshot_path, write_lsh_snapshot,
};
use crate::read_view::HdcReadView;
use crate::recovery::{HeaderHealth, RecoveryReport};
use crate::search::{ApproximateSearchOptions, SearchOutcome};
use crate::transaction::{
    BatchJournal, JournalRecord, JournalRecordKind, batch_journal_path, recover_batch_journal,
    remove_batch_journal, write_batch_journal,
};

const BINARY_HV_BYTES: usize = 2048;
static COMPACTION_TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Configuration for creating a new HdcStore.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoreConfig {
    /// Initial capacity (number of entries to pre-allocate).
    pub initial_capacity: u64,
    /// LSH bands for similarity search.
    pub lsh_bands: u32,
    /// LSH rows per band.
    pub lsh_rows: u32,
}

impl StoreConfig {
    /// Validate dimensions and allocation arithmetic before creating a file.
    pub fn validate(&self) -> Result<(), HdcStoreError> {
        validate_lsh_config(self.lsh_bands as usize, self.lsh_rows as usize)?;
        self.file_len()?;
        Ok(())
    }

    fn file_len(&self) -> Result<usize, HdcStoreError> {
        let capacity = usize::try_from(self.initial_capacity).map_err(|_| {
            HdcStoreError::ArithmeticOverflow {
                context: "initial_capacity conversion",
            }
        })?;
        let entry_bytes =
            capacity
                .checked_mul(ENTRY_SIZE)
                .ok_or(HdcStoreError::ArithmeticOverflow {
                    context: "initial store allocation",
                })?;
        DATA_OFFSET
            .checked_add(entry_bytes)
            .ok_or(HdcStoreError::ArithmeticOverflow {
                context: "initial store file length",
            })
    }
}

impl Default for StoreConfig {
    fn default() -> Self {
        Self {
            initial_capacity: 1024,
            // Shorter bands substantially improve candidate recall for nearby
            // BinaryHVs while using fewer hyperplanes than the old 10x32 default.
            lsh_bands: 32,
            lsh_rows: 8,
        }
    }
}

/// Zero-copy mmap'd BinaryHV storage with one exclusive mutable opener.
pub struct HdcStore {
    /// Memory-mapped file.
    mmap: MmapMut,
    /// Parsed header cached after successful durable commits.
    header: StoreHeader,
    /// In-memory index: id -> entry index.
    id_to_index: HashMap<u64, u64>,
    /// File path used for diagnostics and compaction.
    path: PathBuf,
    /// Backing file handle; also owns the exclusive advisory lock.
    file: File,
    /// Rebuildable in-memory LSH index.
    lsh: LshIndex,
    /// Header slot containing the newest committed generation.
    active_header_slot: HeaderSlot,
    /// Validation state of the two redundant header pages.
    header_health: HeaderHealth,
    /// Origin and persistence state of the optional LSH sidecar.
    index_status: IndexStatus,
    /// Fail-stop state after any uncertain partial mutation.
    health: StoreHealth,
    /// Path-stable lock held across atomic data-file replacement.
    coordination_lock: Option<StoreLock>,
}

impl HdcStore {
    /// Create a new store. Fails if `path` already exists.
    pub fn create(path: impl AsRef<Path>, config: StoreConfig) -> Result<Self, HdcStoreError> {
        Self::create_inner(path.as_ref(), config, false)
    }

    /// Explicitly create or replace a store at `path`.
    ///
    /// Unlike [`Self::create`], this method intentionally destroys any existing
    /// store after obtaining its exclusive lock.
    pub fn create_or_replace(
        path: impl AsRef<Path>,
        config: StoreConfig,
    ) -> Result<Self, HdcStoreError> {
        Self::create_inner(path.as_ref(), config, true)
    }

    fn create_inner(
        path: &Path,
        config: StoreConfig,
        replace: bool,
    ) -> Result<Self, HdcStoreError> {
        config.validate()?;
        let coordination_lock = StoreLock::exclusive(path)?;
        let mut store = Self::create_uncoordinated(path, config, replace)?;
        store.coordination_lock = Some(coordination_lock);
        Ok(store)
    }

    pub(crate) fn create_staging(path: &Path, config: StoreConfig) -> Result<Self, HdcStoreError> {
        Self::create_uncoordinated(path, config, false)
    }

    fn create_uncoordinated(
        path: &Path,
        config: StoreConfig,
        replace: bool,
    ) -> Result<Self, HdcStoreError> {
        config.validate()?;
        let path = path.to_path_buf();
        let file_size = config.file_len()?;

        let file = if replace {
            OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .open(&path)?
        } else {
            OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&path)?
        };
        lock_exclusive(&file, &path)?;
        if replace {
            file.set_len(0)?;
        }
        file.set_len(usize_to_u64(file_size, "initial store file length")?)?;

        let mut header = StoreHeader::new();
        header.lsh_bands = config.lsh_bands;
        header.lsh_rows = config.lsh_rows;
        let header = header.sealed();

        let mut mmap = unsafe { MmapMut::map_mut(&file)? };
        mmap[..DATA_OFFSET].fill(0);
        write_header_slot_bytes(&mut mmap, HeaderSlot::Primary, &header);
        write_header_slot_bytes(&mut mmap, HeaderSlot::Secondary, &header);
        mmap.flush_range(0, DATA_OFFSET)?;
        file.sync_all()?;

        let lsh = LshIndex::new(
            config.lsh_bands as usize,
            config.lsh_rows as usize,
            DEFAULT_LSH_SEED,
        )?;

        Ok(Self {
            mmap,
            header,
            id_to_index: HashMap::new(),
            path: path.clone(),
            file,
            lsh,
            active_header_slot: HeaderSlot::Primary,
            header_health: HeaderHealth::Redundant,
            index_status: IndexStatus::new(&path),
            health: StoreHealth::Healthy,
            coordination_lock: None,
        })
    }

    /// Inspect a store under a shared lock without modifying it.
    pub fn inspect(path: impl AsRef<Path>) -> Result<crate::StoreInspection, HdcStoreError> {
        crate::inspection::inspect_store(path)
    }

    /// Migrate a validated legacy format-v1 store into format v2.
    ///
    /// The migration is compacting and installs a synchronized same-directory
    /// replacement only after every live entry has been copied successfully.
    pub fn migrate_v1(
        path: impl AsRef<Path>,
    ) -> Result<(Self, crate::MigrationReport), HdcStoreError> {
        crate::migration::migrate_v1(path)
    }

    /// Open an existing format-v2 store for exclusive mutable access.
    ///
    /// This strict path never rewrites metadata. Use [`Self::open_recovering`]
    /// when a crash between entry-state persistence and header commit may need
    /// to be reconciled explicitly. A compatible LSH snapshot is preferred
    /// but never required by this convenience method.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, HdcStoreError> {
        Self::open_with_index_policy(path, IndexOpenPolicy::PreferSnapshot)
    }

    /// Open with an explicit persisted-index policy.
    pub fn open_with_index_policy(
        path: impl AsRef<Path>,
        index_policy: IndexOpenPolicy,
    ) -> Result<Self, HdcStoreError> {
        Self::open_internal(path.as_ref(), false, index_policy).map(|(store, _)| store)
    }

    /// Open a store and repair only narrowly defined metadata failures.
    ///
    /// Recovery may reconstruct live/tombstone counts from the already
    /// committed entry range and may restore a damaged alternate header page.
    /// It never promotes trailing entries beyond `vector_count`; those are
    /// surfaced in the returned report for operator review.
    pub fn open_recovering(
        path: impl AsRef<Path>,
    ) -> Result<(Self, RecoveryReport), HdcStoreError> {
        Self::open_recovering_with_index_policy(path, IndexOpenPolicy::PreferSnapshot)
    }

    /// Recover metadata while applying an explicit persisted-index policy.
    pub fn open_recovering_with_index_policy(
        path: impl AsRef<Path>,
        index_policy: IndexOpenPolicy,
    ) -> Result<(Self, RecoveryReport), HdcStoreError> {
        Self::open_internal(path.as_ref(), true, index_policy)
    }

    fn open_internal(
        path: &Path,
        recover_metadata: bool,
        index_policy: IndexOpenPolicy,
    ) -> Result<(Self, RecoveryReport), HdcStoreError> {
        let coordination_lock = StoreLock::exclusive(path)?;
        let (mut store, report) =
            Self::open_internal_uncoordinated(path, recover_metadata, index_policy)?;
        store.coordination_lock = Some(coordination_lock);
        Ok((store, report))
    }

    pub(crate) fn open_after_replacement(
        path: &Path,
        coordination_lock: StoreLock,
    ) -> Result<Self, HdcStoreError> {
        let (mut store, _) =
            Self::open_internal_uncoordinated(path, false, IndexOpenPolicy::PreferSnapshot)?;
        store.coordination_lock = Some(coordination_lock);
        Ok(store)
    }

    fn open_internal_uncoordinated(
        path: &Path,
        recover_metadata: bool,
        index_policy: IndexOpenPolicy,
    ) -> Result<(Self, RecoveryReport), HdcStoreError> {
        let path = path.to_path_buf();
        let file = OpenOptions::new().read(true).write(true).open(&path)?;
        lock_exclusive(&file, &path)?;

        let mut mmap = unsafe { MmapMut::map_mut(&file)? };
        if mmap.len() < 12 {
            return Err(HdcStoreError::InvalidHeader {
                reason: format!(
                    "file is {} bytes; at least 12 bytes are required",
                    mmap.len()
                ),
            });
        }

        let first_magic: [u8; 8] = mmap[0..8]
            .try_into()
            .expect("minimum file length checked before conversion");
        let first_version = u32::from_le_bytes(
            mmap[8..12]
                .try_into()
                .expect("minimum file length checked before conversion"),
        );
        let legacy_signature = first_magic == MAGIC && first_version == LEGACY_VERSION;
        if mmap.len() < DATA_OFFSET {
            if legacy_signature {
                return Err(HdcStoreError::VersionMismatch {
                    expected: VERSION,
                    found: LEGACY_VERSION,
                });
            }
            return Err(HdcStoreError::InvalidHeader {
                reason: format!(
                    "format-v2 file is {} bytes; at least {DATA_OFFSET} bytes are required",
                    mmap.len()
                ),
            });
        }

        let selection = match select_header(&mmap) {
            Ok(selection) => selection,
            Err(_) if legacy_signature => {
                return Err(HdcStoreError::VersionMismatch {
                    expected: VERSION,
                    found: LEGACY_VERSION,
                });
            }
            Err(error) => return Err(error),
        };
        let mut header = selection.header;
        let journal_path = batch_journal_path(&path);
        let batch_recovery = if journal_path.exists() {
            if !recover_metadata {
                return Err(HdcStoreError::PendingBatchTransaction { path: journal_path });
            }
            recover_batch_journal(&path, &file, &mut mmap, &header)?
        } else {
            None
        };
        validate_lsh_config(header.lsh_bands as usize, header.lsh_rows as usize).map_err(
            |error| HdcStoreError::InvalidHeader {
                reason: format!("invalid LSH configuration: {error}"),
            },
        )?;

        let required_len = header.required_file_len()?;
        if required_len > mmap.len() {
            return Err(HdcStoreError::InvalidHeader {
                reason: format!(
                    "committed entries require {required_len} bytes, but file contains {} bytes",
                    mmap.len()
                ),
            });
        }

        let scan = scan_committed_entries(&mmap, &header)?;
        let count_mismatch =
            scan.live_count != header.live_count || scan.tombstone_count != header.tombstone_count;
        if count_mismatch && !recover_metadata {
            return Err(HdcStoreError::InvalidHeader {
                reason: format!(
                    "entry scan disagrees with header: live {}/{}, tombstones {}/{}",
                    scan.live_count,
                    header.live_count,
                    scan.tombstone_count,
                    header.tombstone_count
                ),
            });
        }

        let trailing_committed_entries =
            count_trailing_committed_entries(&mmap, header.vector_count)?;
        let lsh = LshIndex::new(
            header.lsh_bands as usize,
            header.lsh_rows as usize,
            DEFAULT_LSH_SEED,
        )?;

        let selected_slot = selection.slot;
        let selected_generation = header.generation;
        let header_health_before = selection.health.clone();
        let mut store = Self {
            mmap,
            header,
            id_to_index: scan.id_to_index,
            path: path.clone(),
            file,
            lsh,
            active_header_slot: selected_slot,
            header_health: selection.health,
            index_status: IndexStatus::new(&path),
            health: StoreHealth::Healthy,
            coordination_lock: None,
        };

        let mut repaired_entry_counts = false;
        let mut repaired_header_redundancy = false;
        if recover_metadata && count_mismatch {
            header.live_count = scan.live_count;
            header.tombstone_count = scan.tombstone_count;
            let committed = store.commit_header(header)?;
            store.header = committed;
            store.header_health = HeaderHealth::Redundant;
            repaired_entry_counts = true;
            repaired_header_redundancy = !header_health_before.is_redundant();
        } else if recover_metadata && !header_health_before.is_redundant() {
            let committed = store.commit_header(header)?;
            store.header = committed;
            store.header_health = HeaderHealth::Redundant;
            repaired_header_redundancy = true;
        }

        let (lsh, index_status) = load_or_rebuild_index(
            &store.mmap,
            &store.header,
            &store.id_to_index,
            &store.path,
            index_policy,
        )?;
        store.lsh = lsh;
        store.index_status = index_status;

        let report = RecoveryReport {
            selected_slot,
            selected_generation,
            header_health_before,
            repaired_entry_counts,
            repaired_header_redundancy,
            batch_recovery,
            trailing_committed_entries,
            final_generation: store.header.generation,
        };
        Ok((store, report))
    }

    /// Append a new vector to the store.
    pub fn append(&mut self, id: u64, hv: &BinaryHV) -> Result<(), HdcStoreError> {
        self.ensure_healthy()?;
        if self.id_to_index.contains_key(&id) {
            return Err(HdcStoreError::Duplicate { id });
        }

        let index = self.header.vector_count;
        let offset = self.header.checked_entry_offset(index)?;
        let needed = offset
            .checked_add(ENTRY_SIZE)
            .ok_or(HdcStoreError::ArithmeticOverflow {
                context: "append end offset",
            })?;

        // Complete every fallible arithmetic step before touching mapped bytes.
        let mut new_header = self.header;
        new_header.vector_count =
            new_header
                .vector_count
                .checked_add(1)
                .ok_or(HdcStoreError::ArithmeticOverflow {
                    context: "vector_count increment",
                })?;
        new_header.live_count =
            new_header
                .live_count
                .checked_add(1)
                .ok_or(HdcStoreError::ArithmeticOverflow {
                    context: "live_count increment",
                })?;
        self.ensure_capacity(needed)?;

        // Prepare the full entry, publishing STATUS_LIVE last within the entry.
        self.mmap[offset..offset + ENTRY_HV_OFFSET].fill(0);
        self.mmap[offset + 1..offset + 9].copy_from_slice(&id.to_le_bytes());
        let hv_start = offset + ENTRY_HV_OFFSET;
        self.mmap[hv_start..hv_start + BINARY_HV_BYTES].copy_from_slice(&hv.0);
        self.mmap[offset] = STATUS_LIVE;

        // From this point onward, any failure leaves durability uncertain.
        if let Err(error) = self.mmap.flush_range(offset, ENTRY_SIZE) {
            let error = HdcStoreError::Io(error);
            self.poison("append", &error);
            return Err(error);
        }
        if let Err(error) = fault::check(FailPoint::AfterAppendEntryFlush) {
            self.poison("append", &error);
            return Err(error);
        }
        let committed_header = match self.commit_header(new_header) {
            Ok(header) => header,
            Err(error) => {
                self.poison("append", &error);
                return Err(error);
            }
        };

        // Publish process-local indexes only after the durable header commit.
        self.header = committed_header;
        self.id_to_index.insert(id, index);
        self.lsh.insert(id, hv);
        self.index_status.mark_dirty();
        Ok(())
    }

    /// Get a zero-copy reference to a stored BinaryHV.
    pub fn get(&self, id: u64) -> Option<&BinaryHV> {
        let &index = self.id_to_index.get(&id)?;
        hv_at(&self.mmap, &self.header, index).ok()
    }

    /// Deterministic checksum of the logical live vector set.
    pub fn content_checksum(&self) -> crate::StoreContentChecksum {
        self.read_view().content_checksum()
    }

    /// Create a deterministic, generation-pinned zero-copy read view.
    pub fn read_view(&self) -> HdcReadView<'_> {
        HdcReadView::new(
            self,
            self.id_to_index
                .iter()
                .map(|(&id, &index)| (id, index))
                .collect(),
        )
    }

    pub(crate) fn get_by_index(&self, index: u64) -> Option<&BinaryHV> {
        hv_at(&self.mmap, &self.header, index).ok()
    }

    /// Durably mark an entry as deleted.
    ///
    /// Returns `Ok(false)` when `id` is not live. Storage failures are never
    /// collapsed into a successful deletion result.
    pub fn delete(&mut self, id: u64) -> Result<bool, HdcStoreError> {
        self.ensure_healthy()?;
        let Some(&index) = self.id_to_index.get(&id) else {
            return Ok(false);
        };

        let offset = self.header.checked_entry_offset(index)?;
        let mut new_header = self.header;
        new_header.live_count =
            new_header
                .live_count
                .checked_sub(1)
                .ok_or(HdcStoreError::InvalidHeader {
                    reason: "live_count underflow during delete".into(),
                })?;
        new_header.tombstone_count =
            new_header
                .tombstone_count
                .checked_add(1)
                .ok_or(HdcStoreError::ArithmeticOverflow {
                    context: "tombstone_count increment",
                })?;

        // Persist the entry state before publishing the corresponding counts.
        self.mmap[offset] = STATUS_TOMBSTONE;
        if let Err(error) = self.mmap.flush_range(offset, 1) {
            let error = HdcStoreError::Io(error);
            self.poison("delete", &error);
            return Err(error);
        }
        if let Err(error) = fault::check(FailPoint::AfterDeleteStatusFlush) {
            self.poison("delete", &error);
            return Err(error);
        }
        let committed_header = match self.commit_header(new_header) {
            Ok(header) => header,
            Err(error) => {
                self.poison("delete", &error);
                return Err(error);
            }
        };

        self.header = committed_header;
        self.id_to_index.remove(&id);
        self.lsh.remove_id(id);
        self.index_status.mark_dirty();
        Ok(true)
    }

    /// Validate and publish a multi-record mutation batch with one header commit.
    ///
    /// A checksummed intent journal is synchronized before any canonical entry
    /// is changed. If the process stops before publication, recovering open
    /// rolls the complete batch back. If the target header committed, recovery
    /// validates the complete mutation set and removes the stale journal.
    pub fn apply_batch(&mut self, batch: WriteBatch) -> Result<BatchCommitReport, HdcStoreError> {
        self.ensure_healthy()?;
        let generation_before = self.header.generation;
        if batch.is_empty() {
            return Ok(BatchCommitReport {
                generation_before,
                generation_after: generation_before,
                appended: 0,
                deleted: 0,
            });
        }

        let plan = self.plan_batch(batch)?;
        if let Some(needed) = plan.needed_file_len {
            self.ensure_capacity(needed)?;
        }
        write_batch_journal(&self.path, &plan.journal)?;
        if let Err(error) = fault::check(FailPoint::AfterBatchJournalSync) {
            self.poison("apply_batch", &error);
            return Err(error);
        }

        let mutation_result = self.apply_batch_bytes(&plan);
        if let Err(error) = mutation_result {
            self.poison("apply_batch", &error);
            return Err(error);
        }
        if let Err(error) = fault::check(FailPoint::AfterBatchDataFlush) {
            self.poison("apply_batch", &error);
            return Err(error);
        }
        let committed_header = match self.commit_header(plan.target_header) {
            Ok(header) => header,
            Err(error) => {
                self.poison("apply_batch", &error);
                return Err(error);
            }
        };

        self.header = committed_header;
        for append in &plan.appends {
            self.id_to_index.insert(append.id, append.index);
            self.lsh.insert(append.id, &append.hv);
        }
        for delete in &plan.deletes {
            self.id_to_index.remove(&delete.id);
            self.lsh.remove_id(delete.id);
        }
        self.index_status.mark_dirty();
        if let Err(error) = fault::check(FailPoint::AfterBatchHeaderCommit) {
            self.poison("apply_batch_cleanup", &error);
            return Err(error);
        }

        if let Err(error) = remove_batch_journal(&self.path) {
            self.poison("apply_batch_cleanup", &error);
            return Err(error);
        }

        Ok(BatchCommitReport {
            generation_before,
            generation_after: self.header.generation,
            appended: plan.journal.append_count,
            deleted: plan.journal.delete_count,
        })
    }

    fn plan_batch(&self, batch: WriteBatch) -> Result<BatchPlan, HdcStoreError> {
        let mut all_ids = HashSet::new();
        let mut appends = batch.appends;
        for (id, _) in &appends {
            if !all_ids.insert(*id) {
                return Err(HdcStoreError::InvalidBatch {
                    reason: format!("id {id} occurs more than once in the batch"),
                });
            }
            if self.id_to_index.contains_key(id) {
                return Err(HdcStoreError::Duplicate { id: *id });
            }
        }

        let mut deletes = Vec::with_capacity(batch.deletes.len());
        for id in batch.deletes {
            if !all_ids.insert(id) {
                return Err(HdcStoreError::InvalidBatch {
                    reason: format!("id {id} occurs more than once in the batch"),
                });
            }
            let index = self
                .id_to_index
                .get(&id)
                .copied()
                .ok_or(HdcStoreError::NotFound { id })?;
            deletes.push(PlannedDelete { id, index });
        }

        appends.sort_unstable_by_key(|(id, _)| *id);
        deletes.sort_unstable_by_key(|delete| delete.index);
        let append_count =
            u64::try_from(appends.len()).map_err(|_| HdcStoreError::ArithmeticOverflow {
                context: "batch append count conversion",
            })?;
        let delete_count =
            u64::try_from(deletes.len()).map_err(|_| HdcStoreError::ArithmeticOverflow {
                context: "batch delete count conversion",
            })?;

        let mut planned_appends = Vec::with_capacity(appends.len());
        let mut records = Vec::with_capacity(appends.len().saturating_add(deletes.len()));
        for (ordinal, (id, hv)) in appends.into_iter().enumerate() {
            let ordinal =
                u64::try_from(ordinal).map_err(|_| HdcStoreError::ArithmeticOverflow {
                    context: "batch append ordinal conversion",
                })?;
            let index = self.header.vector_count.checked_add(ordinal).ok_or(
                HdcStoreError::ArithmeticOverflow {
                    context: "batch append entry index",
                },
            )?;
            planned_appends.push(PlannedAppend { id, hv, index });
            records.push(JournalRecord {
                kind: JournalRecordKind::Append,
                id,
                index,
            });
        }
        for delete in &deletes {
            records.push(JournalRecord {
                kind: JournalRecordKind::Delete,
                id: delete.id,
                index: delete.index,
            });
        }

        let mut target_header = self.header;
        target_header.vector_count = target_header.vector_count.checked_add(append_count).ok_or(
            HdcStoreError::ArithmeticOverflow {
                context: "batch target vector_count",
            },
        )?;
        target_header.live_count = target_header.live_count.checked_add(append_count).ok_or(
            HdcStoreError::ArithmeticOverflow {
                context: "batch target live_count append",
            },
        )?;
        target_header.live_count = target_header.live_count.checked_sub(delete_count).ok_or(
            HdcStoreError::InvalidBatch {
                reason: "batch deletes exceed available live entries".into(),
            },
        )?;
        target_header.tombstone_count = target_header
            .tombstone_count
            .checked_add(delete_count)
            .ok_or(HdcStoreError::ArithmeticOverflow {
                context: "batch target tombstone_count",
            })?;

        let needed_file_len = planned_appends
            .last()
            .map(|append| {
                self.header
                    .checked_entry_offset(append.index)?
                    .checked_add(ENTRY_SIZE)
                    .ok_or(HdcStoreError::ArithmeticOverflow {
                        context: "batch append file length",
                    })
            })
            .transpose()?;
        let target_generation =
            self.header
                .generation
                .checked_add(1)
                .ok_or(HdcStoreError::ArithmeticOverflow {
                    context: "batch target generation",
                })?;
        let journal = BatchJournal {
            base_generation: self.header.generation,
            target_generation,
            base_vector_count: self.header.vector_count,
            base_live_count: self.header.live_count,
            base_tombstone_count: self.header.tombstone_count,
            append_count,
            delete_count,
            records,
        };
        journal.expected_target_counts()?;

        Ok(BatchPlan {
            appends: planned_appends,
            deletes,
            target_header,
            journal,
            needed_file_len,
        })
    }

    fn apply_batch_bytes(&mut self, plan: &BatchPlan) -> Result<(), HdcStoreError> {
        for append in &plan.appends {
            let offset = self.header.checked_entry_offset(append.index)?;
            self.mmap[offset..offset + ENTRY_HV_OFFSET].fill(0);
            self.mmap[offset + 1..offset + 9].copy_from_slice(&append.id.to_le_bytes());
            let hv_start = offset + ENTRY_HV_OFFSET;
            self.mmap[hv_start..hv_start + BINARY_HV_BYTES].copy_from_slice(&append.hv.0);
            self.mmap[offset] = STATUS_LIVE;
        }
        for delete in &plan.deletes {
            let offset = self.header.checked_entry_offset(delete.index)?;
            self.mmap[offset] = STATUS_TOMBSTONE;
        }

        if let (Some(first), Some(last)) = (plan.appends.first(), plan.appends.last()) {
            let start = self.header.checked_entry_offset(first.index)?;
            let end = self
                .header
                .checked_entry_offset(last.index)?
                .checked_add(ENTRY_SIZE)
                .ok_or(HdcStoreError::ArithmeticOverflow {
                    context: "batch append flush end",
                })?;
            self.mmap.flush_range(start, end - start)?;
        }
        for (start, len) in delete_flush_ranges(&self.header, &plan.deletes)? {
            self.mmap.flush_range(start, len)?;
        }
        Ok(())
    }

    /// Return exact top-k nearest neighbors by scanning every live vector.
    ///
    /// This is the default correctness contract. Call
    /// [`Self::scan_similar_approx`] to opt into LSH candidate filtering.
    pub fn scan_similar(&self, query: &BinaryHV, top_k: usize) -> Vec<(u64, f32)> {
        self.brute_force_scan(query, top_k)
    }

    /// Perform an explicitly approximate LSH search with diagnostics.
    ///
    /// The result reports whether a full exact fallback occurred and how many
    /// vectors were examined. Candidate count is never treated as proof of
    /// nearest-neighbor recall; callers can inspect `exact` and decide whether
    /// approximate results are acceptable for their workload.
    pub fn scan_similar_approx(
        &self,
        query: &BinaryHV,
        top_k: usize,
        options: ApproximateSearchOptions,
    ) -> SearchOutcome {
        let total_live = usize::try_from(self.header.live_count).unwrap_or(usize::MAX);
        if top_k == 0 || total_live == 0 {
            return SearchOutcome {
                neighbors: Vec::new(),
                examined: 0,
                total_live,
                exact: total_live == 0,
                fell_back_to_exact: false,
            };
        }

        let candidates = self.lsh.query_candidates(query);
        let minimum_candidates = top_k.saturating_mul(options.candidate_multiplier);
        let should_fallback = (options.fallback_on_empty && candidates.is_empty())
            || (options.candidate_multiplier > 0
                && candidates.len() < minimum_candidates
                && candidates.len() < total_live);

        if should_fallback {
            let neighbors = self.brute_force_scan(query, top_k);
            return SearchOutcome {
                neighbors,
                examined: total_live,
                total_live,
                exact: true,
                fell_back_to_exact: true,
            };
        }

        let mut neighbors: Vec<(u64, f32)> = candidates
            .iter()
            .filter_map(|&id| self.get(id).map(|hv| (id, query.similarity(hv))))
            .collect();
        let examined = neighbors.len();
        sort_and_truncate(&mut neighbors, top_k);
        SearchOutcome {
            neighbors,
            examined,
            total_live,
            exact: examined == total_live,
            fell_back_to_exact: false,
        }
    }

    /// Estimated LSH candidate probability for a target Hamming agreement.
    ///
    /// This is a model-based tuning aid, not an empirical recall guarantee.
    pub fn estimated_lsh_candidate_probability(&self, hamming_agreement: f64) -> f64 {
        self.lsh.estimated_candidate_probability(hamming_agreement)
    }

    /// Current source and persistence state of the in-memory ANN index.
    pub fn index_status(&self) -> &IndexStatus {
        &self.index_status
    }

    /// Fail-stop health of this mutable handle.
    pub fn health(&self) -> &StoreHealth {
        &self.health
    }

    /// Atomically checkpoint deterministic LSH signatures to the sidecar.
    ///
    /// The canonical store is synchronized first so the acceleration artifact
    /// is never published ahead of the data and header generation it names.
    pub fn checkpoint_lsh(&mut self) -> Result<PathBuf, HdcStoreError> {
        self.ensure_healthy()?;
        self.sync_all()?;
        if self.lsh.entry_count() != self.id_to_index.len() {
            return Err(HdcStoreError::InvalidIndexSnapshot {
                path: self.index_status.snapshot_path.clone(),
                reason: format!(
                    "in-memory index contains {} IDs but store contains {} live IDs",
                    self.lsh.entry_count(),
                    self.id_to_index.len()
                ),
            });
        }

        let metadata =
            lsh_snapshot_metadata(&self.mmap, &self.header, &self.id_to_index, self.lsh.seed())?;
        let entries = self.lsh.snapshot_entries();
        let path = write_lsh_snapshot(&self.path, metadata, &entries)?;
        self.index_status.mark_checkpointed();
        Ok(path)
    }

    /// Iterate over all live entries.
    pub fn iter_live(&self) -> impl Iterator<Item = (u64, &BinaryHV)> {
        self.id_to_index
            .keys()
            .filter_map(move |&id| self.get(id).map(|hv| (id, hv)))
    }

    /// Number of live entries.
    pub const fn live_count(&self) -> u64 {
        self.header.live_count
    }

    /// Number of tombstoned entries.
    pub const fn tombstone_count(&self) -> u64 {
        self.header.tombstone_count
    }

    /// Generation of the newest valid committed header.
    pub const fn header_generation(&self) -> u64 {
        self.header.generation
    }

    /// Header page that supplied the newest valid committed generation.
    pub const fn active_header_slot(&self) -> HeaderSlot {
        self.active_header_slot
    }

    /// Validation state of the redundant header pages at the last open/repair.
    pub fn header_health(&self) -> &HeaderHealth {
        &self.header_health
    }

    /// Restore a missing or corrupt alternate header page explicitly.
    ///
    /// Returns `Ok(false)` when both header pages were already valid.
    pub fn repair_header_redundancy(&mut self) -> Result<bool, HdcStoreError> {
        self.ensure_healthy()?;
        if self.header_health.is_redundant() {
            return Ok(false);
        }
        let committed = match self.commit_header(self.header) {
            Ok(header) => header,
            Err(error) => {
                self.poison("repair_header_redundancy", &error);
                return Err(error);
            }
        };
        self.header = committed;
        self.header_health = HeaderHealth::Redundant;
        self.index_status.mark_dirty();
        Ok(true)
    }

    /// Store path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Whether the shared compaction policy recommends compaction.
    pub fn needs_compaction(&self) -> bool {
        compaction_stats(self.header.live_count, self.header.tombstone_count).recommended
    }

    /// Compact the store through a synced, same-directory replacement file.
    ///
    /// On Unix, renaming within one directory atomically replaces the old path.
    /// The replacement file and parent directory are synced before success is
    /// reported. Temporary files are removed automatically on pre-rename errors.
    pub fn compact(&mut self) -> Result<(), HdcStoreError> {
        self.ensure_healthy()?;
        if self.coordination_lock.is_none() {
            return Err(HdcStoreError::CompactionFailed {
                reason: "canonical coordination lock is not held".into(),
            });
        }
        let config = StoreConfig {
            initial_capacity: self.header.live_count.max(64),
            lsh_bands: self.header.lsh_bands,
            lsh_rows: self.header.lsh_rows,
        };
        let (tmp_path, mut tmp) = self.create_compaction_store(config)?;
        let mut cleanup = TempPathGuard::new(tmp_path.clone());

        let original_permissions = self.file.metadata()?.permissions();
        std::fs::set_permissions(&tmp_path, original_permissions)?;

        let mut live_ids: Vec<u64> = self.id_to_index.keys().copied().collect();
        live_ids.sort_unstable();
        for id in live_ids {
            let hv_copy = *self.get(id).ok_or(HdcStoreError::CorruptEntry {
                index: *self.id_to_index.get(&id).expect("id collected from map"),
                reason: format!("live id {id} could not be read during compaction"),
            })?;
            tmp.append(id, &hv_copy)?;
        }

        tmp.sync_all()?;
        drop(tmp);

        atomic_replace(&tmp_path, &self.path)?;
        cleanup.disarm();

        let path = self.path.clone();
        let (mut replacement, _) =
            Self::open_internal_uncoordinated(&path, false, IndexOpenPolicy::PreferSnapshot)
                .map_err(|error| HdcStoreError::CompactionFailed {
                    reason: format!("replacement was installed but could not be reopened: {error}"),
                })?;
        replacement.coordination_lock = self.coordination_lock.take();
        *self = replacement;
        sync_parent_directory(&path)?;
        Ok(())
    }

    /// Flush mapped bytes and synchronize the backing file and required metadata.
    pub fn sync_all(&mut self) -> Result<(), HdcStoreError> {
        self.ensure_healthy()?;
        self.mmap.flush()?;
        self.file.sync_all()?;
        Ok(())
    }

    fn ensure_healthy(&self) -> Result<(), HdcStoreError> {
        match &self.health {
            StoreHealth::Healthy => Ok(()),
            StoreHealth::Poisoned { operation, cause } => Err(HdcStoreError::StorePoisoned {
                operation: *operation,
                cause: cause.clone(),
            }),
        }
    }

    fn poison(&mut self, operation: &'static str, error: &HdcStoreError) {
        if self.health.is_healthy() {
            self.health = StoreHealth::Poisoned {
                operation,
                cause: error.to_string(),
            };
        }
    }

    fn create_compaction_store(
        &self,
        config: StoreConfig,
    ) -> Result<(PathBuf, Self), HdcStoreError> {
        let parent = self.path.parent().unwrap_or_else(|| Path::new("."));
        let file_name = self
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("hdc-store");

        for _ in 0..128 {
            let sequence = COMPACTION_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let candidate = parent.join(format!(
                ".{file_name}.compact-{}-{sequence}.tmp",
                std::process::id()
            ));
            match Self::create_staging(&candidate, config) {
                Ok(store) => return Ok((candidate, store)),
                Err(HdcStoreError::Io(error)) if error.kind() == ErrorKind::AlreadyExists => {
                    continue;
                }
                Err(error) => return Err(error),
            }
        }

        Err(HdcStoreError::CompactionFailed {
            reason: "could not allocate a unique temporary path after 128 attempts".into(),
        })
    }

    fn ensure_capacity(&mut self, needed: usize) -> Result<(), HdcStoreError> {
        if needed <= self.mmap.len() {
            return Ok(());
        }

        let doubled = self.mmap.len().max(DATA_OFFSET).checked_mul(2).ok_or(
            HdcStoreError::ArithmeticOverflow {
                context: "store growth",
            },
        )?;
        let new_size = needed.max(doubled);
        self.file
            .set_len(usize_to_u64(new_size, "grown store file length")?)?;
        self.mmap = unsafe { MmapMut::map_mut(&self.file)? };
        Ok(())
    }

    fn collect_all_similarities(&self, query: &BinaryHV) -> Vec<(u64, f32)> {
        self.id_to_index
            .keys()
            .filter_map(|&id| self.get(id).map(|hv| (id, query.similarity(hv))))
            .collect()
    }

    fn brute_force_scan(&self, query: &BinaryHV, top_k: usize) -> Vec<(u64, f32)> {
        if top_k == 0 {
            return Vec::new();
        }
        let mut results = self.collect_all_similarities(query);
        sort_and_truncate(&mut results, top_k);
        results
    }

    fn commit_header(&mut self, mut header: StoreHeader) -> Result<StoreHeader, HdcStoreError> {
        header.generation =
            self.header
                .generation
                .checked_add(1)
                .ok_or(HdcStoreError::ArithmeticOverflow {
                    context: "header generation increment",
                })?;
        let header = header.sealed();
        let target_slot = self.active_header_slot.other();
        write_header_slot_bytes(&mut self.mmap, target_slot, &header);
        self.mmap
            .flush_range(target_slot.page_offset(), HEADER_PAGE_SIZE)?;
        self.file.sync_data()?;
        self.active_header_slot = target_slot;
        self.header_health = HeaderHealth::Redundant;
        Ok(header)
    }
}

fn write_header_slot_bytes(mmap: &mut MmapMut, slot: HeaderSlot, header: &StoreHeader) {
    let offset = slot.page_offset();
    mmap[offset..offset + HEADER_SIZE].copy_from_slice(&header.to_bytes());
}

fn read_header_slot(mmap: &MmapMut, slot: HeaderSlot) -> Result<StoreHeader, HdcStoreError> {
    let offset = slot.page_offset();
    let bytes: [u8; HEADER_SIZE] = mmap[offset..offset + HEADER_SIZE]
        .try_into()
        .expect("format-v2 metadata region length checked before header selection");
    StoreHeader::validate_serialized(&bytes)
}

struct PlannedAppend {
    id: u64,
    hv: BinaryHV,
    index: u64,
}

struct PlannedDelete {
    id: u64,
    index: u64,
}

struct BatchPlan {
    appends: Vec<PlannedAppend>,
    deletes: Vec<PlannedDelete>,
    target_header: StoreHeader,
    journal: BatchJournal,
    needed_file_len: Option<usize>,
}

fn delete_flush_ranges(
    header: &StoreHeader,
    deletes: &[PlannedDelete],
) -> Result<Vec<(usize, usize)>, HdcStoreError> {
    let mut ranges = Vec::new();
    let Some(first) = deletes.first() else {
        return Ok(ranges);
    };
    let mut range_start = header.checked_entry_offset(first.index)?;
    let mut previous_index = first.index;
    let mut range_end = range_start
        .checked_add(1)
        .ok_or(HdcStoreError::ArithmeticOverflow {
            context: "batch delete flush range",
        })?;

    for delete in &deletes[1..] {
        let offset = header.checked_entry_offset(delete.index)?;
        if delete.index == previous_index.saturating_add(1) {
            range_end = offset
                .checked_add(1)
                .ok_or(HdcStoreError::ArithmeticOverflow {
                    context: "batch delete flush range extension",
                })?;
        } else {
            ranges.push((range_start, range_end - range_start));
            range_start = offset;
            range_end = offset
                .checked_add(1)
                .ok_or(HdcStoreError::ArithmeticOverflow {
                    context: "batch delete flush range restart",
                })?;
        }
        previous_index = delete.index;
    }
    ranges.push((range_start, range_end - range_start));
    Ok(ranges)
}

struct HeaderSelection {
    header: StoreHeader,
    slot: HeaderSlot,
    health: HeaderHealth,
}

fn select_header(mmap: &MmapMut) -> Result<HeaderSelection, HdcStoreError> {
    let primary = read_header_slot(mmap, HeaderSlot::Primary);
    let secondary = read_header_slot(mmap, HeaderSlot::Secondary);

    match (primary, secondary) {
        (Ok(primary), Ok(secondary)) => {
            let (header, slot) = match primary.generation.cmp(&secondary.generation) {
                std::cmp::Ordering::Greater => (primary, HeaderSlot::Primary),
                std::cmp::Ordering::Less => (secondary, HeaderSlot::Secondary),
                std::cmp::Ordering::Equal if primary == secondary => (primary, HeaderSlot::Primary),
                std::cmp::Ordering::Equal => {
                    return Err(HdcStoreError::HeaderConflict {
                        generation: primary.generation,
                    });
                }
            };
            Ok(HeaderSelection {
                header,
                slot,
                health: HeaderHealth::Redundant,
            })
        }
        (Ok(primary), Err(error)) => Ok(HeaderSelection {
            header: primary,
            slot: HeaderSlot::Primary,
            health: HeaderHealth::Degraded {
                valid_slot: HeaderSlot::Primary,
                invalid_slot: HeaderSlot::Secondary,
                reason: error.to_string(),
            },
        }),
        (Err(error), Ok(secondary)) => Ok(HeaderSelection {
            header: secondary,
            slot: HeaderSlot::Secondary,
            health: HeaderHealth::Degraded {
                valid_slot: HeaderSlot::Secondary,
                invalid_slot: HeaderSlot::Primary,
                reason: error.to_string(),
            },
        }),
        (Err(primary), Err(secondary)) => Err(HdcStoreError::NoValidHeader {
            primary: primary.to_string(),
            secondary: secondary.to_string(),
        }),
    }
}

fn load_or_rebuild_index(
    mmap: &MmapMut,
    header: &StoreHeader,
    id_to_index: &HashMap<u64, u64>,
    store_path: &Path,
    policy: IndexOpenPolicy,
) -> Result<(LshIndex, IndexStatus), HdcStoreError> {
    let metadata = lsh_snapshot_metadata(mmap, header, id_to_index, DEFAULT_LSH_SEED)?;
    let mut ignored_snapshot_error = None;

    if policy != IndexOpenPolicy::Rebuild {
        match load_lsh_snapshot(store_path, metadata) {
            Ok(Some(snapshot)) => match restore_index_from_snapshot(
                snapshot,
                id_to_index,
                header.lsh_bands as usize,
                header.lsh_rows as usize,
                store_path,
            ) {
                Ok(index) => return Ok((index, IndexStatus::loaded(store_path))),
                Err(error) if policy == IndexOpenPolicy::RequireSnapshot => return Err(error),
                Err(error) => ignored_snapshot_error = Some(error.to_string()),
            },
            Ok(None) if policy == IndexOpenPolicy::RequireSnapshot => {
                return Err(HdcStoreError::InvalidIndexSnapshot {
                    path: lsh_snapshot_path(store_path),
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
    let mut ids: Vec<u64> = id_to_index.keys().copied().collect();
    ids.sort_unstable();
    for id in ids {
        let entry_index = id_to_index[&id];
        index.insert(id, hv_at(mmap, header, entry_index)?);
    }
    Ok((
        index,
        IndexStatus::rebuilt(store_path, ignored_snapshot_error),
    ))
}

fn restore_index_from_snapshot(
    snapshot: LshSnapshot,
    id_to_index: &HashMap<u64, u64>,
    bands: usize,
    rows: usize,
    store_path: &Path,
) -> Result<LshIndex, HdcStoreError> {
    let snapshot_path = lsh_snapshot_path(store_path);
    if snapshot.entries.len() != id_to_index.len() {
        return Err(HdcStoreError::InvalidIndexSnapshot {
            path: snapshot_path.clone(),
            reason: format!(
                "snapshot contains {} IDs but store contains {} live IDs",
                snapshot.entries.len(),
                id_to_index.len()
            ),
        });
    }

    let mut index = LshIndex::new(bands, rows, snapshot.metadata.lsh_seed)?;
    for (id, signature) in snapshot.entries {
        if !id_to_index.contains_key(&id) {
            return Err(HdcStoreError::InvalidIndexSnapshot {
                path: snapshot_path.clone(),
                reason: format!("snapshot contains non-live id {id}"),
            });
        }
        index.insert_signature(id, signature).map_err(|error| {
            HdcStoreError::InvalidIndexSnapshot {
                path: snapshot_path.clone(),
                reason: format!("invalid signature for id {id}: {error}"),
            }
        })?;
    }
    Ok(index)
}

fn lsh_snapshot_metadata(
    mmap: &MmapMut,
    header: &StoreHeader,
    id_to_index: &HashMap<u64, u64>,
    lsh_seed: u64,
) -> Result<LshSnapshotMetadata, HdcStoreError> {
    Ok(LshSnapshotMetadata {
        store_generation: header.generation,
        vector_count: header.vector_count,
        live_count: header.live_count,
        tombstone_count: header.tombstone_count,
        lsh_bands: header.lsh_bands,
        lsh_rows: header.lsh_rows,
        lsh_seed,
        store_fingerprint: fingerprint_live_entries(mmap, header, id_to_index)?,
    })
}

fn fingerprint_live_entries(
    mmap: &MmapMut,
    header: &StoreHeader,
    id_to_index: &HashMap<u64, u64>,
) -> Result<u64, HdcStoreError> {
    let mut ids: Vec<u64> = id_to_index.keys().copied().collect();
    ids.sort_unstable();
    let mut records = Vec::with_capacity(ids.len());
    for id in ids {
        let entry_index = id_to_index[&id];
        records.push((id, hv_at(mmap, header, entry_index)?));
    }
    Ok(fingerprint_ordered(records, header.live_count))
}

struct EntryScan {
    id_to_index: HashMap<u64, u64>,
    live_count: u64,
    tombstone_count: u64,
}

fn scan_committed_entries(
    mmap: &MmapMut,
    header: &StoreHeader,
) -> Result<EntryScan, HdcStoreError> {
    let mut id_to_index = HashMap::new();
    let mut live_count = 0u64;
    let mut tombstone_count = 0u64;

    for index in 0..header.vector_count {
        let offset = header.checked_entry_offset(index)?;
        match mmap[offset] {
            STATUS_LIVE => {
                let id = read_entry_id(mmap, offset, index)?;
                if id_to_index.insert(id, index).is_some() {
                    return Err(HdcStoreError::CorruptEntry {
                        index,
                        reason: format!("duplicate live id {id}"),
                    });
                }
                live_count =
                    live_count
                        .checked_add(1)
                        .ok_or(HdcStoreError::ArithmeticOverflow {
                            context: "scanned live entry count",
                        })?;
            }
            STATUS_TOMBSTONE => {
                tombstone_count =
                    tombstone_count
                        .checked_add(1)
                        .ok_or(HdcStoreError::ArithmeticOverflow {
                            context: "scanned tombstone count",
                        })?;
            }
            status => {
                return Err(HdcStoreError::CorruptEntry {
                    index,
                    reason: format!("invalid committed status byte {status}"),
                });
            }
        }
    }

    Ok(EntryScan {
        id_to_index,
        live_count,
        tombstone_count,
    })
}

fn count_trailing_committed_entries(
    mmap: &MmapMut,
    vector_count: u64,
) -> Result<u64, HdcStoreError> {
    let mut count = 0u64;
    let mut index = vector_count;
    loop {
        let offset = crate::header::checked_entry_offset(DATA_OFFSET, index)?;
        let end = offset
            .checked_add(ENTRY_SIZE)
            .ok_or(HdcStoreError::ArithmeticOverflow {
                context: "trailing entry end offset",
            })?;
        if end > mmap.len() {
            return Ok(count);
        }
        match mmap[offset] {
            STATUS_LIVE | STATUS_TOMBSTONE => {
                count = count
                    .checked_add(1)
                    .ok_or(HdcStoreError::ArithmeticOverflow {
                        context: "trailing committed entry count",
                    })?;
                index = index
                    .checked_add(1)
                    .ok_or(HdcStoreError::ArithmeticOverflow {
                        context: "trailing entry index",
                    })?;
            }
            _ => return Ok(count),
        }
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

fn atomic_replace(source: &Path, destination: &Path) -> Result<(), HdcStoreError> {
    std::fs::rename(source, destination).map_err(|error| HdcStoreError::CompactionFailed {
        reason: format!("same-directory replacement rename failed: {error}"),
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

fn lock_exclusive(file: &File, path: &Path) -> Result<(), HdcStoreError> {
    match file.try_lock() {
        Ok(()) => Ok(()),
        Err(std::fs::TryLockError::WouldBlock) => Err(HdcStoreError::StoreLocked {
            path: path.to_path_buf(),
        }),
        Err(std::fs::TryLockError::Error(error)) => Err(HdcStoreError::Io(error)),
    }
}

fn read_entry_id(mmap: &MmapMut, offset: usize, index: u64) -> Result<u64, HdcStoreError> {
    let id_end = offset
        .checked_add(9)
        .ok_or(HdcStoreError::ArithmeticOverflow {
            context: "entry id end offset",
        })?;
    let id_bytes: [u8; 8] = mmap
        .get(offset + 1..id_end)
        .ok_or_else(|| HdcStoreError::CorruptEntry {
            index,
            reason: "entry id extends beyond mapped file".into(),
        })?
        .try_into()
        .expect("entry id slice length is exactly eight bytes");
    Ok(u64::from_le_bytes(id_bytes))
}

fn hv_at<'a>(
    mmap: &'a MmapMut,
    header: &StoreHeader,
    index: u64,
) -> Result<&'a BinaryHV, HdcStoreError> {
    let offset = header.checked_entry_offset(index)?;
    let hv_start =
        offset
            .checked_add(ENTRY_HV_OFFSET)
            .ok_or(HdcStoreError::ArithmeticOverflow {
                context: "BinaryHV start offset",
            })?;
    let hv_end =
        hv_start
            .checked_add(BINARY_HV_BYTES)
            .ok_or(HdcStoreError::ArithmeticOverflow {
                context: "BinaryHV end offset",
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

    // SAFETY: the mmap base is page-aligned, all layout constants are
    // 32-byte aligned, the slice is exactly BinaryHV's byte length, and the
    // returned lifetime is bounded by the immutable mmap borrow.
    Ok(unsafe { &*(ptr as *const BinaryHV) })
}

fn sort_and_truncate(results: &mut Vec<(u64, f32)>, top_k: usize) {
    results.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    results.truncate(top_k);
}

fn usize_to_u64(value: usize, context: &'static str) -> Result<u64, HdcStoreError> {
    u64::try_from(value).map_err(|_| HdcStoreError::ArithmeticOverflow { context })
}

#[cfg(test)]
mod tests {
    use std::io::{Seek, SeekFrom, Write};

    use super::*;
    use tempfile::tempdir;

    fn random_hv(seed: u64) -> BinaryHV {
        BinaryHV::random(seed)
    }

    #[test]
    fn create_and_reopen() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");

        {
            let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
            store.append(1, &random_hv(1)).unwrap();
            store.append(2, &random_hv(2)).unwrap();
            assert_eq!(store.live_count(), 2);
        }

        let store = HdcStore::open(&path).unwrap();
        assert_eq!(store.live_count(), 2);
        assert_eq!(store.get(1).unwrap().similarity(&random_hv(1)), 1.0);
        assert_eq!(store.get(2).unwrap().similarity(&random_hv(2)), 1.0);
    }

    #[test]
    fn create_refuses_existing_path() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        let _store = HdcStore::create(&path, StoreConfig::default()).unwrap();
        assert!(HdcStore::create(&path, StoreConfig::default()).is_err());
    }

    #[test]
    fn explicit_replace_is_supported() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        {
            let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
            store.append(1, &random_hv(1)).unwrap();
        }
        let store = HdcStore::create_or_replace(&path, StoreConfig::default()).unwrap();
        assert_eq!(store.live_count(), 0);
    }

    #[test]
    fn second_mutable_open_is_rejected() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        let _first = HdcStore::create(&path, StoreConfig::default()).unwrap();
        assert!(matches!(
            HdcStore::open(&path),
            Err(HdcStoreError::StoreLocked { .. })
        ));
    }

    #[test]
    fn zero_copy_get() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
        let original = random_hv(42);
        store.append(42, &original).unwrap();
        assert_eq!(store.get(42).unwrap().similarity(&original), 1.0);
    }

    #[test]
    fn tombstone_and_compact() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();

        for i in 0..10 {
            store.append(i, &random_hv(i)).unwrap();
        }
        for i in 0..5 {
            assert!(store.delete(i).unwrap());
        }
        assert_eq!(store.live_count(), 5);
        assert_eq!(store.tombstone_count(), 5);
        assert!(store.needs_compaction());

        store.compact().unwrap();
        assert_eq!(store.live_count(), 5);
        assert_eq!(store.tombstone_count(), 0);
        for i in 5..10 {
            assert_eq!(store.get(i).unwrap().similarity(&random_hv(i)), 1.0);
        }
    }

    #[test]
    fn missing_delete_is_not_an_error() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
        assert!(!store.delete(99).unwrap());
    }

    #[test]
    fn duplicate_id_rejected() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
        store.append(1, &random_hv(1)).unwrap();
        assert!(matches!(
            store.append(1, &random_hv(2)),
            Err(HdcStoreError::Duplicate { id: 1 })
        ));
    }

    #[test]
    fn truncated_committed_region_is_rejected() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        {
            let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
            store.append(1, &random_hv(1)).unwrap();
        }
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(DATA_OFFSET as u64)
            .unwrap();
        assert!(matches!(
            HdcStore::open(&path),
            Err(HdcStoreError::InvalidHeader { .. })
        ));
    }

    #[test]
    fn invalid_committed_status_is_rejected() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        {
            let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
            store.append(1, &random_hv(1)).unwrap();
        }
        let mut file = OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(DATA_OFFSET as u64)).unwrap();
        file.write_all(&[99]).unwrap();
        file.sync_all().unwrap();
        assert!(matches!(
            HdcStore::open(&path),
            Err(HdcStoreError::CorruptEntry { index: 0, .. })
        ));
    }

    #[test]
    fn duplicate_live_ids_on_disk_are_rejected() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        {
            let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
            store.append(1, &random_hv(1)).unwrap();
            store.append(2, &random_hv(2)).unwrap();
        }
        let second_id_offset = DATA_OFFSET + ENTRY_SIZE + 1;
        let mut file = OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(second_id_offset as u64)).unwrap();
        file.write_all(&1u64.to_le_bytes()).unwrap();
        file.sync_all().unwrap();
        assert!(matches!(
            HdcStore::open(&path),
            Err(HdcStoreError::CorruptEntry { index: 1, .. })
        ));
    }

    #[test]
    fn invalid_lsh_config_is_rejected_before_create() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        let result = HdcStore::create(
            &path,
            StoreConfig {
                lsh_rows: 33,
                ..StoreConfig::default()
            },
        );
        assert!(matches!(result, Err(HdcStoreError::InvalidConfig { .. })));
        assert!(!path.exists());
    }

    #[test]
    fn scan_similar_returns_top_k_deterministically() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();

        let query = random_hv(100);
        store.append(100, &query).unwrap();
        for i in 0..50 {
            store.append(i, &random_hv(i)).unwrap();
        }

        let results = store.scan_similar(&query, 5);
        assert_eq!(results.len(), 5);
        assert_eq!(results[0].0, 100);
        assert!((results[0].1 - 1.0).abs() < 1e-6);
        for window in results.windows(2) {
            assert!(window[0].1 >= window[1].1);
        }
        assert!(store.scan_similar(&query, 0).is_empty());
    }

    #[test]
    fn approximate_search_reports_its_scope() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
        let query = random_hv(10_000);
        store.append(10_000, &query).unwrap();
        for id in 0..200 {
            store.append(id, &random_hv(id)).unwrap();
        }

        let outcome = store.scan_similar_approx(
            &query,
            5,
            ApproximateSearchOptions {
                candidate_multiplier: 0,
                fallback_on_empty: false,
            },
        );
        assert_eq!(outcome.total_live, 201);
        assert!(outcome.examined <= outcome.total_live);
        assert_eq!(outcome.exact, outcome.examined == outcome.total_live);
        assert!(!outcome.fell_back_to_exact);
        assert_eq!(outcome.neighbors.first().map(|entry| entry.0), Some(10_000));
    }

    #[test]
    fn approximate_search_can_require_exact_fallback() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        let mut store = HdcStore::create(
            &path,
            StoreConfig {
                lsh_bands: 1,
                lsh_rows: 32,
                ..StoreConfig::default()
            },
        )
        .unwrap();
        let query = random_hv(500);
        store.append(500, &query).unwrap();
        for id in 0..50 {
            store.append(id, &random_hv(id)).unwrap();
        }

        let outcome = store.scan_similar_approx(
            &query,
            5,
            ApproximateSearchOptions {
                candidate_multiplier: usize::MAX,
                fallback_on_empty: true,
            },
        );
        assert!(outcome.exact);
        assert!(outcome.fell_back_to_exact);
        assert_eq!(outcome.examined, outcome.total_live);
        assert_eq!(outcome.neighbors[0].0, 500);
    }

    #[test]
    fn default_lsh_probability_model_is_exposed() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        let store = HdcStore::create(&path, StoreConfig::default()).unwrap();
        let unrelated = store.estimated_lsh_candidate_probability(0.5);
        let nearby = store.estimated_lsh_candidate_probability(0.95);
        assert!(nearby > unrelated);
        assert!((0.0..=1.0).contains(&nearby));
    }

    #[test]
    fn iter_live_count() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
        for i in 0..20 {
            store.append(i, &random_hv(i)).unwrap();
        }
        store.delete(5).unwrap();
        store.delete(10).unwrap();
        assert_eq!(store.iter_live().count(), 18);
    }

    #[test]
    fn lsh_rebuilt_on_reopen() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        {
            let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
            let query = random_hv(999);
            store.append(999, &query).unwrap();
            for i in 0..150 {
                store.append(i, &random_hv(i)).unwrap();
            }
        }
        let store = HdcStore::open(&path).unwrap();
        let results = store.scan_similar(&random_hv(999), 5);
        assert_eq!(results[0].0, 999);
        assert!((results[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn auto_grow() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        let mut store = HdcStore::create(
            &path,
            StoreConfig {
                initial_capacity: 2,
                ..StoreConfig::default()
            },
        )
        .unwrap();
        for i in 0..10 {
            store.append(i, &random_hv(i)).unwrap();
        }
        assert_eq!(store.live_count(), 10);
        for i in 0..10 {
            assert!(store.get(i).is_some());
        }
    }
    #[test]
    fn compact_uses_unique_temp_and_cleans_it_up() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        let stale_legacy_tmp = path.with_extension("tmp");
        std::fs::write(&stale_legacy_tmp, b"do not overwrite").unwrap();

        let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
        for i in 0..8 {
            store.append(i, &random_hv(i)).unwrap();
        }
        store.delete(0).unwrap();
        store.delete(1).unwrap();
        store.compact().unwrap();

        assert_eq!(
            std::fs::read(&stale_legacy_tmp).unwrap(),
            b"do not overwrite"
        );
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_name().to_string_lossy().contains(".compact-"))
            .collect();
        assert!(leftovers.is_empty(), "temporary compaction files leaked");
    }

    #[test]
    fn all_tombstones_recommend_compaction() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
        store.append(1, &random_hv(1)).unwrap();
        store.delete(1).unwrap();
        assert!(store.needs_compaction());
    }

    #[cfg(unix)]
    #[test]
    fn compact_preserves_unix_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
        store.append(1, &random_hv(1)).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();

        store.compact().unwrap();
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o640);
    }

    #[test]
    fn newest_valid_header_generation_is_selected() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        {
            let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
            assert_eq!(store.header_generation(), 1);
            store.append(1, &random_hv(1)).unwrap();
            assert_eq!(store.header_generation(), 2);
            assert_eq!(store.active_header_slot(), HeaderSlot::Secondary);
            store.append(2, &random_hv(2)).unwrap();
            assert_eq!(store.header_generation(), 3);
            assert_eq!(store.active_header_slot(), HeaderSlot::Primary);
        }

        let store = HdcStore::open(&path).unwrap();
        assert_eq!(store.header_generation(), 3);
        assert_eq!(store.active_header_slot(), HeaderSlot::Primary);
        assert_eq!(store.live_count(), 2);
    }

    #[test]
    fn corrupt_header_page_falls_back_to_redundant_slot() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        {
            let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
            store.append(1, &random_hv(1)).unwrap();
        }

        let mut file = OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(100)).unwrap();
        file.write_all(&[0xA5]).unwrap();
        file.sync_all().unwrap();

        let store = HdcStore::open(&path).unwrap();
        assert_eq!(store.active_header_slot(), HeaderSlot::Secondary);
        assert_eq!(store.header_generation(), 2);
        assert!(store.get(1).is_some());
    }

    #[test]
    fn two_corrupt_header_pages_are_rejected() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        {
            let store = HdcStore::create(&path, StoreConfig::default()).unwrap();
            drop(store);
        }

        let mut file = OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(100)).unwrap();
        file.write_all(&[0xA5]).unwrap();
        file.seek(SeekFrom::Start((HEADER_PAGE_SIZE + 100) as u64))
            .unwrap();
        file.write_all(&[0x5A]).unwrap();
        file.sync_all().unwrap();

        assert!(matches!(
            HdcStore::open(&path),
            Err(HdcStoreError::NoValidHeader { .. })
        ));
    }

    #[test]
    fn recovering_open_repairs_delete_count_crash_window() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        {
            let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
            store.append(7, &random_hv(7)).unwrap();
        }

        let mut file = OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(DATA_OFFSET as u64)).unwrap();
        file.write_all(&[STATUS_TOMBSTONE]).unwrap();
        file.sync_all().unwrap();

        assert!(matches!(
            HdcStore::open(&path),
            Err(HdcStoreError::InvalidHeader { .. })
        ));

        let (store, report) = HdcStore::open_recovering(&path).unwrap();
        assert!(report.repaired_entry_counts);
        assert!(report.changed_store());
        assert_eq!(report.selected_generation, 2);
        assert_eq!(report.final_generation, 3);
        assert_eq!(store.live_count(), 0);
        assert_eq!(store.tombstone_count(), 1);
        assert!(store.header_health().is_redundant());
    }

    #[test]
    fn recovering_open_restores_corrupt_redundant_header() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        {
            let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
            store.append(1, &random_hv(1)).unwrap();
        }

        let mut file = OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(100)).unwrap();
        file.write_all(&[0xCC]).unwrap();
        file.sync_all().unwrap();

        let (store, report) = HdcStore::open_recovering(&path).unwrap();
        assert!(matches!(
            report.header_health_before,
            HeaderHealth::Degraded { .. }
        ));
        assert!(report.repaired_header_redundancy);
        assert!(!report.repaired_entry_counts);
        assert_eq!(store.header_generation(), 3);
        assert!(store.header_health().is_redundant());
        drop(store);

        let reopened = HdcStore::open(&path).unwrap();
        assert!(reopened.header_health().is_redundant());
        assert!(reopened.get(1).is_some());
    }

    #[test]
    fn trailing_entry_is_reported_but_not_resurrected() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        HdcStore::create(&path, StoreConfig::default()).unwrap();

        let hv = random_hv(44);
        let mut file = OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(DATA_OFFSET as u64)).unwrap();
        let mut entry = [0u8; ENTRY_SIZE];
        entry[0] = STATUS_LIVE;
        entry[1..9].copy_from_slice(&44u64.to_le_bytes());
        entry[ENTRY_HV_OFFSET..ENTRY_HV_OFFSET + BINARY_HV_BYTES].copy_from_slice(&hv.0);
        file.write_all(&entry).unwrap();
        file.sync_all().unwrap();

        let (store, report) = HdcStore::open_recovering(&path).unwrap();
        assert_eq!(report.trailing_committed_entries, 1);
        assert_eq!(store.live_count(), 0);
        assert!(store.get(44).is_none());
        assert!(!report.changed_store());
    }

    #[test]
    fn explicit_redundancy_repair_is_idempotent() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        HdcStore::create(&path, StoreConfig::default()).unwrap();

        let mut file = OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start((HEADER_PAGE_SIZE + 100) as u64))
            .unwrap();
        file.write_all(&[0x7E]).unwrap();
        file.sync_all().unwrap();

        let mut store = HdcStore::open(&path).unwrap();
        assert!(!store.header_health().is_redundant());
        assert!(store.repair_header_redundancy().unwrap());
        assert!(store.header_health().is_redundant());
        assert!(!store.repair_header_redundancy().unwrap());
    }

    #[test]
    fn create_or_replace_clears_stale_entry_region() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        {
            let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
            store.append(9, &random_hv(9)).unwrap();
        }

        drop(HdcStore::create_or_replace(&path, StoreConfig::default()).unwrap());
        let (_store, report) = HdcStore::open_recovering(&path).unwrap();
        assert_eq!(report.trailing_committed_entries, 0);
    }

    #[test]
    fn checkpointed_index_loads_under_required_policy() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        {
            let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
            for id in 0..16 {
                store.append(id, &random_hv(id)).unwrap();
            }
            let snapshot_path = store.checkpoint_lsh().unwrap();
            assert!(snapshot_path.exists());
            assert!(store.index_status().snapshot_current);
        }

        let store =
            HdcStore::open_with_index_policy(&path, IndexOpenPolicy::RequireSnapshot).unwrap();
        assert_eq!(
            store.index_status().source,
            crate::IndexLoadSource::Snapshot
        );
        assert!(store.index_status().snapshot_current);
        assert!(store.index_status().ignored_snapshot_error.is_none());
        assert_eq!(store.scan_similar(&random_hv(7), 1)[0].0, 7);
    }

    #[test]
    fn mutation_stales_snapshot_and_preferred_open_rebuilds() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        {
            let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
            store.append(1, &random_hv(1)).unwrap();
            store.checkpoint_lsh().unwrap();
            store.append(2, &random_hv(2)).unwrap();
            assert!(!store.index_status().snapshot_current);
        }

        assert!(matches!(
            HdcStore::open_with_index_policy(&path, IndexOpenPolicy::RequireSnapshot),
            Err(HdcStoreError::InvalidIndexSnapshot { .. })
        ));

        let store = HdcStore::open(&path).unwrap();
        assert_eq!(store.index_status().source, crate::IndexLoadSource::Rebuilt);
        assert!(!store.index_status().snapshot_current);
        assert!(store.index_status().ignored_snapshot_error.is_some());
        assert!(store.get(1).is_some());
        assert!(store.get(2).is_some());
    }

    #[test]
    fn rebuild_policy_ignores_valid_snapshot() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        {
            let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
            store.append(1, &random_hv(1)).unwrap();
            store.checkpoint_lsh().unwrap();
        }

        let store = HdcStore::open_with_index_policy(&path, IndexOpenPolicy::Rebuild).unwrap();
        assert_eq!(store.index_status().source, crate::IndexLoadSource::Rebuilt);
        assert!(!store.index_status().snapshot_current);
        assert!(store.index_status().ignored_snapshot_error.is_none());
    }

    #[test]
    fn corrupted_snapshot_falls_back_only_when_policy_allows_it() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        let snapshot_path;
        {
            let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
            store.append(1, &random_hv(1)).unwrap();
            snapshot_path = store.checkpoint_lsh().unwrap();
        }

        let mut snapshot = OpenOptions::new().write(true).open(&snapshot_path).unwrap();
        snapshot.seek(SeekFrom::Start(100)).unwrap();
        snapshot.write_all(&[0xFF]).unwrap();
        snapshot.sync_all().unwrap();
        drop(snapshot);

        assert!(matches!(
            HdcStore::open_with_index_policy(&path, IndexOpenPolicy::RequireSnapshot),
            Err(HdcStoreError::InvalidIndexSnapshot { .. })
        ));

        let store = HdcStore::open(&path).unwrap();
        assert_eq!(store.index_status().source, crate::IndexLoadSource::Rebuilt);
        assert!(store.index_status().ignored_snapshot_error.is_some());
    }

    #[test]
    fn checkpoint_after_mutation_restores_required_open() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        {
            let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
            store.append(1, &random_hv(1)).unwrap();
            store.checkpoint_lsh().unwrap();
            store.append(2, &random_hv(2)).unwrap();
            store.checkpoint_lsh().unwrap();
        }

        let store =
            HdcStore::open_with_index_policy(&path, IndexOpenPolicy::RequireSnapshot).unwrap();
        assert_eq!(store.live_count(), 2);
        assert_eq!(
            store.index_status().source,
            crate::IndexLoadSource::Snapshot
        );
    }

    #[test]
    fn poisoned_handle_rejects_further_mutations() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
        let synthetic = HdcStoreError::InvalidHeader {
            reason: "synthetic uncertain commit".into(),
        };
        store.poison("test_mutation", &synthetic);

        assert!(!store.health().is_healthy());
        assert_eq!(store.health().poisoned_operation(), Some("test_mutation"));
        assert!(matches!(
            store.append(1, &random_hv(1)),
            Err(HdcStoreError::StorePoisoned {
                operation: "test_mutation",
                ..
            })
        ));
        assert!(matches!(
            store.delete(1),
            Err(HdcStoreError::StorePoisoned { .. })
        ));
        assert!(matches!(
            store.checkpoint_lsh(),
            Err(HdcStoreError::StorePoisoned { .. })
        ));
    }

    #[test]
    fn batch_publishes_one_generation() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
        store.append(1, &random_hv(1)).unwrap();
        let before = store.header_generation();

        let report = store
            .apply_batch(
                WriteBatch::new()
                    .append(2, random_hv(2))
                    .append(3, random_hv(3))
                    .delete(1),
            )
            .unwrap();

        assert_eq!(report.generation_before, before);
        assert_eq!(report.generation_after, before + 1);
        assert_eq!(report.appended, 2);
        assert_eq!(report.deleted, 1);
        assert_eq!(store.live_count(), 2);
        assert_eq!(store.tombstone_count(), 1);
        assert!(store.get(1).is_none());
        assert!(store.get(2).is_some());
        assert!(store.get(3).is_some());
        assert!(!batch_journal_path(&path).exists());
    }

    #[test]
    fn invalid_batch_changes_nothing() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
        store.append(1, &random_hv(1)).unwrap();
        let generation = store.header_generation();

        let result = store.apply_batch(WriteBatch::new().append(2, random_hv(2)).delete(2));
        assert!(matches!(result, Err(HdcStoreError::InvalidBatch { .. })));
        assert_eq!(store.header_generation(), generation);
        assert_eq!(store.live_count(), 1);
        assert!(store.get(1).is_some());
        assert!(store.get(2).is_none());
        assert!(!batch_journal_path(&path).exists());
    }

    #[test]
    fn recovering_open_rolls_back_unpublished_batch() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        {
            let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
            store.append(1, &random_hv(1)).unwrap();
            let plan = store
                .plan_batch(WriteBatch::new().append(2, random_hv(2)).delete(1))
                .unwrap();
            if let Some(needed) = plan.needed_file_len {
                store.ensure_capacity(needed).unwrap();
            }
            write_batch_journal(&path, &plan.journal).unwrap();
            store.apply_batch_bytes(&plan).unwrap();
            // Simulated stop before commit_header.
        }

        assert!(matches!(
            HdcStore::open(&path),
            Err(HdcStoreError::PendingBatchTransaction { .. })
        ));
        let (store, report) = HdcStore::open_recovering(&path).unwrap();
        let batch = report.batch_recovery.unwrap();
        assert_eq!(
            batch.disposition,
            crate::BatchRecoveryDisposition::RolledBack
        );
        assert!(store.get(1).is_some());
        assert!(store.get(2).is_none());
        assert_eq!(store.live_count(), 1);
        assert_eq!(store.tombstone_count(), 0);
        assert!(!batch_journal_path(&path).exists());
    }

    #[test]
    fn recovering_open_finalizes_published_batch() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        {
            let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
            store.append(1, &random_hv(1)).unwrap();
            let plan = store
                .plan_batch(WriteBatch::new().append(2, random_hv(2)).delete(1))
                .unwrap();
            if let Some(needed) = plan.needed_file_len {
                store.ensure_capacity(needed).unwrap();
            }
            write_batch_journal(&path, &plan.journal).unwrap();
            store.apply_batch_bytes(&plan).unwrap();
            store.commit_header(plan.target_header).unwrap();
            // Simulated stop after durable publication but before journal cleanup.
        }

        assert!(matches!(
            HdcStore::open(&path),
            Err(HdcStoreError::PendingBatchTransaction { .. })
        ));
        let (store, report) = HdcStore::open_recovering(&path).unwrap();
        let batch = report.batch_recovery.unwrap();
        assert_eq!(
            batch.disposition,
            crate::BatchRecoveryDisposition::FinalizedCommitted
        );
        assert!(store.get(1).is_none());
        assert!(store.get(2).is_some());
        assert_eq!(store.live_count(), 1);
        assert_eq!(store.tombstone_count(), 1);
        assert!(!batch_journal_path(&path).exists());
    }

    #[test]
    fn read_view_is_generation_pinned_and_deterministic() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
        store.append(30, &random_hv(30)).unwrap();
        store.append(10, &random_hv(10)).unwrap();
        store.append(20, &random_hv(20)).unwrap();

        let view = store.read_view();
        assert_eq!(view.generation(), store.header_generation());
        assert_eq!(view.live_count(), 3);
        assert_eq!(view.ids().collect::<Vec<_>>(), vec![10, 20, 30]);
        assert_eq!(
            view.iter().map(|(id, _)| id).collect::<Vec<_>>(),
            vec![10, 20, 30]
        );
        assert_eq!(view.get(20).unwrap().similarity(&random_hv(20)), 1.0);
        assert_eq!(view.scan_similar(&random_hv(30), 1)[0].0, 30);
    }

    #[test]
    fn coordination_lock_survives_compaction_replacement() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        let lock_path = crate::store_lock_path(&path);
        let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
        for id in 0..8 {
            store.append(id, &random_hv(id)).unwrap();
        }
        for id in 0..4 {
            store.delete(id).unwrap();
        }
        assert!(lock_path.exists());

        store.compact().unwrap();
        assert!(lock_path.exists());
        assert!(matches!(
            HdcStore::open(&path),
            Err(HdcStoreError::StoreLocked { .. })
        ));
        assert_eq!(store.live_count(), 4);
        assert_eq!(store.tombstone_count(), 0);
    }

    #[test]
    fn append_fault_after_entry_flush_never_resurrects_entry() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        {
            let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
            let _guard =
                crate::fault::FailPointGuard::arm(crate::fault::FailPoint::AfterAppendEntryFlush);
            assert!(store.append(9, &random_hv(9)).is_err());
            assert!(!store.health().is_healthy());
        }

        let (store, report) = HdcStore::open_recovering(&path).unwrap();
        assert_eq!(store.live_count(), 0);
        assert!(store.get(9).is_none());
        assert_eq!(report.trailing_committed_entries, 1);
    }

    #[test]
    fn delete_fault_after_status_flush_is_reconciled() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        {
            let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
            store.append(9, &random_hv(9)).unwrap();
            let _guard =
                crate::fault::FailPointGuard::arm(crate::fault::FailPoint::AfterDeleteStatusFlush);
            assert!(store.delete(9).is_err());
            assert!(!store.health().is_healthy());
        }

        let (store, report) = HdcStore::open_recovering(&path).unwrap();
        assert!(report.repaired_entry_counts);
        assert_eq!(store.live_count(), 0);
        assert_eq!(store.tombstone_count(), 1);
        assert!(store.get(9).is_none());
    }

    #[test]
    fn batch_fault_after_journal_sync_rolls_back() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        {
            let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
            store.append(1, &random_hv(1)).unwrap();
            let _guard =
                crate::fault::FailPointGuard::arm(crate::fault::FailPoint::AfterBatchJournalSync);
            assert!(
                store
                    .apply_batch(WriteBatch::new().append(2, random_hv(2)).delete(1),)
                    .is_err()
            );
            assert!(!store.health().is_healthy());
        }

        let (store, report) = HdcStore::open_recovering(&path).unwrap();
        assert_eq!(
            report.batch_recovery.unwrap().disposition,
            crate::BatchRecoveryDisposition::RolledBack
        );
        assert!(store.get(1).is_some());
        assert!(store.get(2).is_none());
    }

    #[test]
    fn batch_fault_after_data_flush_rolls_back() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        {
            let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
            store.append(1, &random_hv(1)).unwrap();
            let _guard =
                crate::fault::FailPointGuard::arm(crate::fault::FailPoint::AfterBatchDataFlush);
            assert!(
                store
                    .apply_batch(WriteBatch::new().append(2, random_hv(2)).delete(1),)
                    .is_err()
            );
        }

        let (store, report) = HdcStore::open_recovering(&path).unwrap();
        assert_eq!(
            report.batch_recovery.unwrap().disposition,
            crate::BatchRecoveryDisposition::RolledBack
        );
        assert!(store.get(1).is_some());
        assert!(store.get(2).is_none());
        assert_eq!(store.live_count(), 1);
        assert_eq!(store.tombstone_count(), 0);
    }

    #[test]
    fn batch_fault_after_header_commit_finalizes_commit() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        {
            let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
            store.append(1, &random_hv(1)).unwrap();
            let _guard =
                crate::fault::FailPointGuard::arm(crate::fault::FailPoint::AfterBatchHeaderCommit);
            assert!(
                store
                    .apply_batch(WriteBatch::new().append(2, random_hv(2)).delete(1),)
                    .is_err()
            );
            assert!(!store.health().is_healthy());
        }

        let (store, report) = HdcStore::open_recovering(&path).unwrap();
        assert_eq!(
            report.batch_recovery.unwrap().disposition,
            crate::BatchRecoveryDisposition::FinalizedCommitted
        );
        assert!(store.get(1).is_none());
        assert!(store.get(2).is_some());
        assert_eq!(store.live_count(), 1);
        assert_eq!(store.tombstone_count(), 1);
    }
}
