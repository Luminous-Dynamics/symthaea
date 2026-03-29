// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Persistent LSH (Locality-Sensitive Hashing) index for approximate nearest neighbor search.
//!
//! Uses random hyperplane LSH: each hash function is a random BinaryHV,
//! and the hash bit is 1 if similarity > 0.5. Bands of hash bits form bucket keys.

use std::collections::HashMap;
use symthaea_core::hdc::BinaryHV;

/// Persistent LSH index for BinaryHV similarity search.
#[derive(Debug)]
pub struct LshIndex {
    /// Random hyperplanes for hashing (bands x rows).
    hyperplanes: Vec<BinaryHV>,
    /// Number of bands.
    bands: usize,
    /// Number of rows (hash functions) per band.
    rows: usize,
    /// Bucket -> entry IDs mapping, keyed by (band, bucket_hash).
    buckets: HashMap<(u16, u32), Vec<u64>>,
}

impl LshIndex {
    /// Create a new LSH index with the given band/row configuration.
    pub fn new(bands: usize, rows: usize, seed: u64) -> Self {
        let hyperplanes: Vec<BinaryHV> = (0..bands * rows)
            .map(|i| BinaryHV::random(seed.wrapping_add(i as u64)))
            .collect();

        Self {
            hyperplanes,
            bands,
            rows,
            buckets: HashMap::new(),
        }
    }

    /// Insert an entry into the index.
    pub fn insert(&mut self, id: u64, hv: &BinaryHV) {
        for band in 0..self.bands {
            let hash = self.band_hash(band, hv);
            self.buckets
                .entry((band as u16, hash))
                .or_default()
                .push(id);
        }
    }

    /// Remove an entry from the index.
    pub fn remove(&mut self, id: u64, hv: &BinaryHV) {
        for band in 0..self.bands {
            let hash = self.band_hash(band, hv);
            if let Some(bucket) = self.buckets.get_mut(&(band as u16, hash)) {
                bucket.retain(|&x| x != id);
            }
        }
    }

    /// Query for candidate IDs that may be similar to the query HV.
    ///
    /// Returns deduplicated candidate IDs. The caller must verify similarity
    /// with the actual vectors (LSH provides candidates, not exact matches).
    pub fn query_candidates(&self, query: &BinaryHV) -> Vec<u64> {
        let mut candidates = Vec::new();
        for band in 0..self.bands {
            let hash = self.band_hash(band, query);
            if let Some(bucket) = self.buckets.get(&(band as u16, hash)) {
                candidates.extend_from_slice(bucket);
            }
        }
        candidates.sort_unstable();
        candidates.dedup();
        candidates
    }

    /// Clear the index.
    pub fn clear(&mut self) {
        self.buckets.clear();
    }

    /// Number of entries in the index.
    pub fn entry_count(&self) -> usize {
        let mut ids = Vec::new();
        for bucket in self.buckets.values() {
            ids.extend_from_slice(bucket);
        }
        ids.sort_unstable();
        ids.dedup();
        ids.len()
    }

    fn band_hash(&self, band: usize, hv: &BinaryHV) -> u32 {
        let mut hash: u32 = 0;
        let base = band * self.rows;
        for row in 0..self.rows.min(32) {
            let hp = &self.hyperplanes[base + row];
            if hv.similarity(hp) > 0.5 {
                hash |= 1 << row;
            }
        }
        hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_query() {
        let mut lsh = LshIndex::new(5, 8, 42);
        let hv1 = BinaryHV::random(1);
        let hv2 = BinaryHV::random(2);

        lsh.insert(1, &hv1);
        lsh.insert(2, &hv2);

        // Query with hv1 itself should return 1 as candidate
        let candidates = lsh.query_candidates(&hv1);
        assert!(candidates.contains(&1));
    }

    #[test]
    fn remove_entry() {
        let mut lsh = LshIndex::new(5, 8, 42);
        let hv = BinaryHV::random(1);

        lsh.insert(1, &hv);
        assert_eq!(lsh.entry_count(), 1);

        lsh.remove(1, &hv);
        // After removal, shouldn't find it
        let candidates = lsh.query_candidates(&hv);
        assert!(!candidates.contains(&1));
    }

    #[test]
    fn similar_vectors_share_buckets() {
        let mut lsh = LshIndex::new(10, 16, 42);
        let hv1 = BinaryHV::random(1);
        // Create a similar vector by adding small noise
        let hv_similar = hv1.add_noise(0.05, 99);

        lsh.insert(1, &hv1);
        lsh.insert(2, &hv_similar);

        // Similar vectors should co-occur in at least some buckets
        let candidates_1 = lsh.query_candidates(&hv1);
        let candidates_2 = lsh.query_candidates(&hv_similar);

        // hv1 query should find hv_similar as candidate (high probability with 10 bands)
        // Note: this is probabilistic but with 95%+ probability at sim > 0.9
        assert!(
            candidates_1.contains(&2) || candidates_2.contains(&1),
            "Similar vectors should share at least one LSH bucket"
        );
    }

    #[test]
    fn clear_empties_index() {
        let mut lsh = LshIndex::new(5, 8, 42);
        for i in 0..10 {
            lsh.insert(i, &BinaryHV::random(i));
        }
        assert!(lsh.entry_count() > 0);
        lsh.clear();
        assert_eq!(lsh.entry_count(), 0);
    }
}
