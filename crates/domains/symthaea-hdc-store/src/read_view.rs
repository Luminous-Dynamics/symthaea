// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Generation-pinned, zero-copy read views over an open mutable store.

use symthaea_core::hdc::BinaryHV;

use crate::HdcStore;

/// A deterministic read-only view pinned to one process-local store generation.
///
/// The view immutably borrows its `HdcStore`, so Rust prevents append, delete,
/// batch mutation, recovery repair, or compaction while it is alive. Vector
/// payloads remain zero-copy references into the same mmap.
pub struct HdcReadView<'a> {
    store: &'a HdcStore,
    entries: Vec<(u64, u64)>,
    generation: u64,
    live_count: u64,
    tombstone_count: u64,
}

impl<'a> HdcReadView<'a> {
    pub(crate) fn new(store: &'a HdcStore, mut entries: Vec<(u64, u64)>) -> Self {
        entries.sort_unstable_by_key(|(id, _)| *id);
        Self {
            store,
            entries,
            generation: store.header_generation(),
            live_count: store.live_count(),
            tombstone_count: store.tombstone_count(),
        }
    }

    /// Header generation pinned by this view.
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Live-entry count observed when the view was created.
    pub const fn live_count(&self) -> u64 {
        self.live_count
    }

    /// Tombstone count observed when the view was created.
    pub const fn tombstone_count(&self) -> u64 {
        self.tombstone_count
    }

    /// Whether the view contains no live vectors.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Deterministic ascending live IDs.
    pub fn ids(&self) -> impl ExactSizeIterator<Item = u64> + '_ {
        self.entries.iter().map(|(id, _)| *id)
    }

    /// Get a zero-copy vector reference by ID.
    pub fn get(&self, id: u64) -> Option<&'a BinaryHV> {
        let position = self
            .entries
            .binary_search_by_key(&id, |(entry_id, _)| *entry_id)
            .ok()?;
        let index = self.entries[position].1;
        self.store.get_by_index(index)
    }

    /// Iterate live vectors in ascending ID order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (u64, &'a BinaryHV)> + '_ {
        self.entries.iter().map(|(id, index)| {
            let hv = self
                .store
                .get_by_index(*index)
                .expect("read view was built from validated live indexes");
            (*id, hv)
        })
    }

    /// Deterministic checksum of the logical live vector set.
    pub fn content_checksum(&self) -> crate::StoreContentChecksum {
        crate::content_checksum::checksum_ordered(self.iter(), self.live_count())
    }

    /// Exact deterministic nearest-neighbor search over the pinned live set.
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
