// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HdcStore -- mmap'd contiguous BinaryHV storage.
//!
//! Each entry is 2064 bytes: 16-byte metadata prefix + 2048-byte BinaryHV.
//! The file layout is:
//! - Header (128 bytes)
//! - Entries (2064 bytes each, contiguous)
//!
//! Zero-copy reads: `get()` returns a reference directly into the mmap region.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

use memmap2::MmapMut;
use symthaea_core::hdc::BinaryHV;

use crate::HdcStoreError;
use crate::header::{
    ENTRY_HV_OFFSET, ENTRY_SIZE, HEADER_SIZE, STATUS_LIVE, STATUS_TOMBSTONE, StoreHeader,
};
use crate::lsh_persistent::LshIndex;

/// Configuration for creating a new HdcStore.
#[derive(Debug, Clone)]
pub struct StoreConfig {
    /// Initial capacity (number of entries to pre-allocate).
    pub initial_capacity: u64,
    /// LSH bands for similarity search.
    pub lsh_bands: u32,
    /// LSH rows per band.
    pub lsh_rows: u32,
}

impl Default for StoreConfig {
    fn default() -> Self {
        Self {
            initial_capacity: 1024,
            lsh_bands: 10,
            lsh_rows: 32,
        }
    }
}

/// Zero-copy mmap'd BinaryHV storage.
///
/// Stores 16,384-bit BinaryHV vectors with O(1) random access via mmap.
/// Each entry has a 16-byte metadata prefix (status, id_hash, padding)
/// followed by 2048 bytes of BinaryHV data.
pub struct HdcStore {
    /// Memory-mapped file.
    mmap: MmapMut,
    /// Parsed header (cached, flushed on mutation).
    header: StoreHeader,
    /// In-memory index: id -> entry index (not byte offset).
    id_to_index: HashMap<u64, u64>,
    /// File path for compaction.
    path: PathBuf,
    /// Backing file handle.
    file: File,
    /// LSH index for approximate nearest neighbor queries.
    lsh: LshIndex,
}

impl Clone for HdcStore {
    fn clone(&self) -> Self {
        Self::open(&self.path).expect("Failed to clone HdcStore (re-open at same path failed)")
    }
}

impl HdcStore {
    /// Create a new store at the given path.
    pub fn create(path: impl AsRef<Path>, config: StoreConfig) -> Result<Self, HdcStoreError> {
        let path = path.as_ref().to_path_buf();
        let mut header = StoreHeader::new();
        header.lsh_bands = config.lsh_bands;
        header.lsh_rows = config.lsh_rows;

        let file_size = HEADER_SIZE + (config.initial_capacity as usize) * ENTRY_SIZE;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;
        file.set_len(file_size as u64)?;

        let mut mmap = unsafe { MmapMut::map_mut(&file)? };

        // Write header
        mmap[..HEADER_SIZE].copy_from_slice(&header.to_bytes());
        mmap.flush()?;

        let lsh = LshIndex::new(config.lsh_bands as usize, config.lsh_rows as usize, 42);

        Ok(Self {
            mmap,
            header,
            id_to_index: HashMap::new(),
            path,
            file,
            lsh,
        })
    }

    /// Open an existing store.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, HdcStoreError> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new().read(true).write(true).open(&path)?;

        let mmap = unsafe { MmapMut::map_mut(&file)? };

        if mmap.len() < HEADER_SIZE {
            return Err(HdcStoreError::InvalidHeader {
                reason: "file too small for header".into(),
            });
        }

        let header_bytes: [u8; HEADER_SIZE] =
            mmap[..HEADER_SIZE]
                .try_into()
                .map_err(|_| HdcStoreError::InvalidHeader {
                    reason: "failed to read header bytes".into(),
                })?;
        let header = StoreHeader::from_bytes(&header_bytes);
        header.validate()?;

        // Rebuild in-memory index by scanning entries
        let mut id_to_index = HashMap::new();
        for i in 0..header.vector_count {
            let offset = header.entry_offset(i);
            if offset + ENTRY_SIZE > mmap.len() {
                break;
            }
            let status = mmap[offset];
            if status == STATUS_LIVE {
                let id =
                    u64::from_le_bytes(mmap[offset + 1..offset + 9].try_into().unwrap_or([0; 8]));
                id_to_index.insert(id, i);
            }
        }

        // Rebuild LSH from live entries
        let mut lsh = LshIndex::new(header.lsh_bands as usize, header.lsh_rows as usize, 42);
        for (&id, &index) in &id_to_index {
            let offset = header.entry_offset(index);
            let hv_start = offset + ENTRY_HV_OFFSET;
            let hv_end = hv_start + 2048;
            if hv_end <= mmap.len() {
                let ptr = mmap[hv_start..hv_end].as_ptr() as *const BinaryHV;
                let hv = unsafe { &*ptr };
                lsh.insert(id, hv);
            }
        }

        Ok(Self {
            mmap,
            header,
            id_to_index,
            path,
            file,
            lsh,
        })
    }

    /// Append a new vector to the store.
    pub fn append(&mut self, id: u64, hv: &BinaryHV) -> Result<(), HdcStoreError> {
        if self.id_to_index.contains_key(&id) {
            return Err(HdcStoreError::Duplicate { id });
        }

        let index = self.header.vector_count;
        let offset = self.header.entry_offset(index);
        let needed = offset + ENTRY_SIZE;

        // Grow file if necessary
        if needed > self.mmap.len() {
            let new_size = (needed * 2).max(self.mmap.len() * 2);
            self.file.set_len(new_size as u64)?;
            self.mmap = unsafe { MmapMut::map_mut(&self.file)? };
        }

        // Write entry: [status: u8] [id: u64 LE] [padding: 23 bytes] [hv: 2048 bytes]
        // The 32-byte metadata prefix ensures HV data is 32-byte aligned.
        self.mmap[offset] = STATUS_LIVE;
        self.mmap[offset + 1..offset + 9].copy_from_slice(&id.to_le_bytes());
        // 23 bytes padding (already zeroed)
        let hv_start = offset + ENTRY_HV_OFFSET;
        self.mmap[hv_start..hv_start + 2048].copy_from_slice(&hv.0);

        // Crash safety: flush entry data before updating header count.
        // On crash between data write and header update, the orphaned entry
        // is invisible (scanner reads up to `vector_count`).
        self.mmap.flush_range(offset, ENTRY_SIZE)?;

        self.id_to_index.insert(id, index);
        self.header.vector_count += 1;
        self.header.live_count += 1;
        self.lsh.insert(id, hv);
        self.flush_header()?;

        Ok(())
    }

    /// Get a zero-copy reference to a stored BinaryHV.
    pub fn get(&self, id: u64) -> Option<&BinaryHV> {
        let &index = self.id_to_index.get(&id)?;
        let offset = self.header.entry_offset(index);
        let hv_start = offset + ENTRY_HV_OFFSET;
        let hv_end = hv_start + 2048;

        if hv_end > self.mmap.len() {
            return None;
        }

        // SAFETY: BinaryHV is #[repr(align(32))] wrapping [u8; 2048].
        // The entry layout guarantees HV data starts at a 32-byte aligned
        // offset (HEADER_SIZE=128 is 32-aligned, ENTRY_HV_OFFSET=32 is
        // 32-aligned, ENTRY_SIZE=2080 is 32-aligned). The mmap base is
        // page-aligned (4096), so all HV pointers are 32-byte aligned.
        // The lifetime is tied to &self which keeps the mmap alive.
        let ptr = self.mmap[hv_start..hv_end].as_ptr();
        // Alignment is guaranteed by construction:
        // - mmap base is page-aligned (4096-byte)
        // - HEADER_SIZE (128), ENTRY_HV_OFFSET (32), ENTRY_SIZE (2080) are all 32-aligned
        // So every HV pointer is 32-byte aligned. Assert rather than silently leaking.
        assert!(
            (ptr as usize) % 32 == 0,
            "BinaryHV pointer not 32-byte aligned at offset {hv_start} — \
             this indicates a layout invariant violation (bug)"
        );
        Some(unsafe { &*(ptr as *const BinaryHV) })
    }

    /// Mark an entry as deleted (tombstone).
    pub fn delete(&mut self, id: u64) -> bool {
        if let Some(&index) = self.id_to_index.get(&id) {
            // Remove from LSH before tombstoning
            let offset = self.header.entry_offset(index);
            let hv_start = offset + ENTRY_HV_OFFSET;
            let hv_end = hv_start + 2048;
            if hv_end <= self.mmap.len() {
                let ptr = self.mmap[hv_start..hv_end].as_ptr() as *const BinaryHV;
                let hv = unsafe { &*ptr };
                let hv_copy = *hv; // BinaryHV is Copy
                self.lsh.remove(id, &hv_copy);
            }

            self.mmap[offset] = STATUS_TOMBSTONE;
            self.id_to_index.remove(&id);
            self.header.tombstone_count += 1;
            self.header.live_count -= 1;
            let _ = self.flush_header();
            true
        } else {
            false
        }
    }

    /// Similarity scan with LSH candidate filtering. Returns top-k (id, similarity) pairs.
    ///
    /// For stores with <= 100 live entries, performs brute-force SIMD scan.
    /// For larger stores, uses LSH to narrow candidates before verification.
    /// Falls back to brute-force if LSH returns too few candidates.
    pub fn scan_similar(&self, query: &BinaryHV, top_k: usize) -> Vec<(u64, f32)> {
        // For small stores, brute-force is fast enough
        if self.header.live_count <= 100 {
            let mut results: Vec<(u64, f32)> = self
                .id_to_index
                .iter()
                .filter_map(|(&id, _)| {
                    let hv = self.get(id)?;
                    Some((id, query.similarity(hv)))
                })
                .collect();
            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            results.truncate(top_k);
            return results;
        }

        // For larger stores, use LSH candidates first
        let candidates = self.lsh.query_candidates(query);

        let mut results: Vec<(u64, f32)> = if candidates.len() >= top_k * 2 {
            // LSH provided enough candidates — verify on just those
            candidates
                .iter()
                .filter_map(|&id| {
                    let hv = self.get(id)?;
                    Some((id, query.similarity(hv)))
                })
                .collect()
        } else {
            // Not enough LSH candidates — brute-force
            self.id_to_index
                .iter()
                .filter_map(|(&id, _)| {
                    let hv = self.get(id)?;
                    Some((id, query.similarity(hv)))
                })
                .collect()
        };

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    /// Iterate over all live entries.
    pub fn iter_live(&self) -> impl Iterator<Item = (u64, &BinaryHV)> {
        self.id_to_index
            .iter()
            .filter_map(move |(&id, _)| Some((id, self.get(id)?)))
    }

    /// Number of live entries.
    pub fn live_count(&self) -> u64 {
        self.header.live_count
    }

    /// Number of tombstoned entries.
    pub fn tombstone_count(&self) -> u64 {
        self.header.tombstone_count
    }

    /// Whether compaction is recommended (tombstones > 25% of live).
    pub fn needs_compaction(&self) -> bool {
        self.header.live_count > 0 && self.header.tombstone_count > self.header.live_count / 4
    }

    /// Compact the store: copy live entries to a temp file, then atomic rename.
    pub fn compact(&mut self) -> Result<(), HdcStoreError> {
        let tmp_path = self.path.with_extension("tmp");
        let config = StoreConfig {
            initial_capacity: self.header.live_count.max(64),
            lsh_bands: self.header.lsh_bands,
            lsh_rows: self.header.lsh_rows,
        };

        let mut tmp = Self::create(&tmp_path, config)?;

        // Copy all live entries
        let live_ids: Vec<u64> = self.id_to_index.keys().copied().collect();
        for id in live_ids {
            if let Some(hv) = self.get(id) {
                let hv_copy = *hv; // BinaryHV is Copy
                tmp.append(id, &hv_copy)?;
            }
        }

        tmp.mmap.flush()?;
        drop(tmp);

        // Atomic rename
        std::fs::rename(&tmp_path, &self.path).map_err(|e| HdcStoreError::CompactionFailed {
            reason: format!("rename failed: {e}"),
        })?;

        // Reopen
        *self = Self::open(&self.path)?;
        Ok(())
    }

    fn flush_header(&mut self) -> Result<(), HdcStoreError> {
        self.mmap[..HEADER_SIZE].copy_from_slice(&self.header.to_bytes());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::BinaryHV;
    use tempfile::tempdir;

    fn random_hv(seed: u64) -> BinaryHV {
        BinaryHV::random(seed)
    }

    #[test]
    fn create_and_reopen() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");

        // Create and insert
        {
            let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
            store.append(1, &random_hv(1)).unwrap();
            store.append(2, &random_hv(2)).unwrap();
            assert_eq!(store.live_count(), 2);
        }

        // Reopen and verify
        {
            let store = HdcStore::open(&path).unwrap();
            assert_eq!(store.live_count(), 2);
            let hv1 = store.get(1).unwrap();
            assert_eq!(hv1.similarity(&random_hv(1)), 1.0);
            let hv2 = store.get(2).unwrap();
            assert_eq!(hv2.similarity(&random_hv(2)), 1.0);
        }
    }

    #[test]
    fn zero_copy_get() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();

        let original = random_hv(42);
        store.append(42, &original).unwrap();

        let retrieved = store.get(42).unwrap();
        assert_eq!(retrieved.similarity(&original), 1.0);
    }

    #[test]
    fn tombstone_and_compact() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();

        for i in 0..10 {
            store.append(i, &random_hv(i)).unwrap();
        }
        assert_eq!(store.live_count(), 10);

        // Delete some
        for i in 0..5 {
            assert!(store.delete(i));
        }
        assert_eq!(store.live_count(), 5);
        assert_eq!(store.tombstone_count(), 5);
        assert!(store.needs_compaction());

        // Compact
        store.compact().unwrap();
        assert_eq!(store.live_count(), 5);
        assert_eq!(store.tombstone_count(), 0);
        assert!(!store.needs_compaction());

        // Verify surviving entries
        for i in 5..10 {
            let hv = store.get(i).unwrap();
            assert_eq!(hv.similarity(&random_hv(i)), 1.0);
        }
    }

    #[test]
    fn duplicate_id_rejected() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();

        store.append(1, &random_hv(1)).unwrap();
        assert!(store.append(1, &random_hv(2)).is_err());
    }

    #[test]
    fn scan_similar_returns_top_k() {
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
        // The query itself should be the most similar (similarity = 1.0)
        assert_eq!(results[0].0, 100);
        assert!((results[0].1 - 1.0).abs() < 1e-6);
        // Results should be sorted descending
        for w in results.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    }

    #[test]
    fn iter_live_count() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();

        for i in 0..20 {
            store.append(i, &random_hv(i)).unwrap();
        }
        store.delete(5);
        store.delete(10);

        let count = store.iter_live().count();
        assert_eq!(count, 18);
    }

    #[test]
    fn lsh_accelerated_scan() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();

        // Insert enough entries to trigger LSH path (> 100)
        let query = random_hv(999);
        store.append(999, &query).unwrap();
        for i in 0..150 {
            store.append(i, &random_hv(i)).unwrap();
        }

        let results = store.scan_similar(&query, 5);
        assert_eq!(results.len(), 5);
        // The query itself should be top result
        assert_eq!(results[0].0, 999);
        assert!((results[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn lsh_survives_delete() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();

        for i in 0..10 {
            store.append(i, &random_hv(i)).unwrap();
        }

        // Delete an entry and verify it doesn't appear in scan results
        store.delete(5);
        let results = store.scan_similar(&random_hv(5), 10);
        assert!(results.iter().all(|(id, _)| *id != 5));
    }

    #[test]
    fn lsh_rebuilt_on_reopen() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");

        // Create store with enough entries to trigger LSH path
        {
            let mut store = HdcStore::create(&path, StoreConfig::default()).unwrap();
            let query = random_hv(999);
            store.append(999, &query).unwrap();
            for i in 0..150 {
                store.append(i, &random_hv(i)).unwrap();
            }
        }

        // Reopen and verify LSH works
        {
            let store = HdcStore::open(&path).unwrap();
            let query = random_hv(999);
            let results = store.scan_similar(&query, 5);
            assert_eq!(results[0].0, 999);
            assert!((results[0].1 - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn auto_grow() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hdc");
        let mut store = HdcStore::create(
            &path,
            StoreConfig {
                initial_capacity: 2,
                ..Default::default()
            },
        )
        .unwrap();

        // Insert more than initial capacity
        for i in 0..10 {
            store.append(i, &random_hv(i)).unwrap();
        }
        assert_eq!(store.live_count(), 10);

        // Verify all retrievable
        for i in 0..10 {
            assert!(store.get(i).is_some());
        }
    }
}
