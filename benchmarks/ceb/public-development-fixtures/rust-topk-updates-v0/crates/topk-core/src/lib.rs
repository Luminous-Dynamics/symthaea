//! Binary-heap baseline with an intentional stale-update defect.
//!
//! The ordering is deterministic, but every update is pushed as a new entry.
//! No key index or generation check invalidates older entries.

use std::collections::BinaryHeap;

use ceb_topk_rank_contract::RankedItem;

#[derive(Debug)]
pub struct TopK {
    capacity: usize,
    heap: BinaryHeap<RankedItem>,
}

impl TopK {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            heap: BinaryHeap::new(),
        }
    }

    /// Insert or update a key.
    ///
    /// Intentional defect: an update appends a new heap entry without removing
    /// or marking the previous version stale.
    pub fn upsert(&mut self, item: RankedItem) {
        self.heap.push(item);
    }

    /// Return the highest-ranked entries under the declared total ordering.
    ///
    /// Intentional defect: stale entries remain eligible and may cause the
    /// same key to appear more than once.
    pub fn top_k(&self) -> Vec<RankedItem> {
        self.heap
            .clone()
            .into_sorted_vec()
            .into_iter()
            .rev()
            .take(self.capacity)
            .collect()
    }
}
