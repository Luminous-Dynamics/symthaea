// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Revolutionary Incremental Computation for HDC Operations
//!
//! **Paradigm Shift**: Don't recompute everything - update only what changed!
//!
//! # The Problem
//!
//! Traditional HDC operations recompute EVERYTHING every cycle:
//! - Rebundle ALL concept vectors even if only 1 changed
//! - Recompute ALL similarities even if query is the same
//! - Rebuild entire semantic spaces from scratch
//!
//! This is like repainting your entire screen for every pixel change!
//!
//! # The Revolutionary Solution
//!
//! **Incremental Computation**: Track what changed, update only affected parts
//!
//! ## Performance Impact
//!
//! For typical consciousness operations where <10% changes per cycle:
//! - Bundle update: O(n) → O(k) where k=changed vectors (**10x+** speedup)
//! - Similarity cache: O(n²) → O(n×k) where k=queries (**100x+** for stable queries)
//! - Semantic drift: O(n) → O(1) with incremental statistics (**n×** speedup)
//!
//! # Usage
//!
//! ```rust,ignore
//! use symthaea::hdc::incremental_hv::*;
//! use symthaea::hdc::binary_hv::BinaryHV;
//!
//! // Create incremental bundle tracker
//! let mut bundle = IncrementalBundle::new();
//!
//! // Add initial vectors
//! bundle.add(vec![BinaryHV::random(0), BinaryHV::random(1), BinaryHV::random(2)]);
//!
//! // Get bundled result - O(n) first time
//! let result1 = bundle.get_bundle();
//!
//! // Update one vector - O(1) incremental update!
//! bundle.update(1, BinaryHV::random(10));
//!
//! // Get updated bundle - O(k) where k=1, not O(n)!
//! let result2 = bundle.get_bundle();  // ~100x faster for n=100!
//! ```

use super::binary_hv::BinaryHV;
use std::collections::HashMap;

// Helper functions using BinaryHV's native methods (replacement for incompatible simd_hv)
#[inline]
fn simd_bind(a: &BinaryHV, b: &BinaryHV) -> BinaryHV {
    a.bind(b)
}

#[inline]
fn simd_bundle(vectors: &[BinaryHV]) -> BinaryHV {
    BinaryHV::bundle(vectors)
}

#[inline]
fn simd_similarity(a: &BinaryHV, b: &BinaryHV) -> f32 {
    a.similarity(b)
}

// ============================================================================
// INCREMENTAL BUNDLE - Revolutionary Update-Only Bundling
// ============================================================================

/// **REVOLUTIONARY**: Incremental bundle that updates in O(k) instead of O(n)
///
/// Traditional bundling recomputes everything:
/// ```text
/// bundle([v1, v2, v3, v4, v5]) = count_bits([v1, v2, v3, v4, v5])  // O(n)
/// ```
///
/// Incremental bundling tracks bit counts:
/// ```text
/// Initial:  bit_counts = count_bits([v1, v2, v3, v4, v5])  // O(n) once
/// Update:   bit_counts -= v3; bit_counts += v3_new         // O(1)!
/// Bundle:   majority_vote(bit_counts)                      // O(1)
/// ```
///
/// # Performance
///
/// - Initial bundle: O(n) - same as traditional
/// - Update k vectors: O(k) - **n/k times faster** than rebundling
/// - Get bundle: O(1) - just majority vote on cached counts
///
/// For n=1000, k=10 changes: **100x faster** than full rebundle!
pub struct IncrementalBundle {
    /// Current vectors in the bundle
    vectors: Vec<BinaryHV>,

    /// Cached bit counts: bit_counts[byte_idx][bit_idx] = count
    /// Positive = more 1s, negative = more 0s
    /// BinaryHV is 2048 bytes (16,384 bits), so we need 2048 byte positions
    bit_counts: Vec<[i32; 8]>,

    /// Cached bundle result (invalidated on updates)
    cached_bundle: Option<BinaryHV>,

    /// Dirty flag - true if counts changed since last bundle
    dirty: bool,
}

/// BinaryHV byte size (2048 bytes = 16,384 bits)
const HV16_BYTES: usize = 2048;

impl IncrementalBundle {
    /// Create new incremental bundle
    pub fn new() -> Self {
        Self {
            vectors: Vec::new(),
            bit_counts: vec![[0; 8]; HV16_BYTES],
            cached_bundle: None,
            dirty: false,
        }
    }

    /// Add vectors to the bundle - O(n) for n new vectors
    pub fn add(&mut self, new_vectors: Vec<BinaryHV>) {
        for vector in &new_vectors {
            self.increment_counts(vector);
        }
        self.vectors.extend(new_vectors);
        self.dirty = true;
    }

    /// Update single vector - O(1) incremental update!
    ///
    /// This is the REVOLUTIONARY part: change ONE vector without rebundling ALL!
    pub fn update(&mut self, index: usize, new_vector: BinaryHV) {
        if index >= self.vectors.len() {
            return;
        }

        // Clone old vector to avoid borrow conflict
        // (BinaryHV is 2048 bytes - small enough that clone is fine)
        let old_vector = self.vectors[index];

        // Decrement old vector's contribution
        self.decrement_counts(&old_vector);

        // Increment new vector's contribution
        self.increment_counts(&new_vector);

        // Update storage
        self.vectors[index] = new_vector;
        self.dirty = true;
    }

    /// Remove vector - O(1) incremental update
    pub fn remove(&mut self, index: usize) -> Option<BinaryHV> {
        if index >= self.vectors.len() {
            return None;
        }

        let removed = self.vectors.remove(index);
        self.decrement_counts(&removed);
        self.dirty = true;
        Some(removed)
    }

    /// Get current bundled result - O(1) if cached, O(2048) if dirty
    ///
    /// Much faster than O(n) rebundling!
    pub fn get_bundle(&mut self) -> BinaryHV {
        if !self.dirty {
            if let Some(cached) = &self.cached_bundle {
                return *cached;
            }
        }

        // Rebuild from counts - O(2048) = O(1) constant time!
        let mut result = [0u8; HV16_BYTES];
        for byte_idx in 0..HV16_BYTES {
            let mut byte_val = 0u8;
            for bit_idx in 0..8 {
                if self.bit_counts[byte_idx][bit_idx] > 0 {
                    byte_val |= 1 << bit_idx;
                }
            }
            result[byte_idx] = byte_val;
        }

        let bundle = BinaryHV(result);
        self.cached_bundle = Some(bundle);
        self.dirty = false;
        bundle
    }

    /// Get current vectors
    pub fn vectors(&self) -> &[BinaryHV] {
        &self.vectors
    }

    /// Number of vectors in bundle
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if bundle is empty
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    // Helper: Increment bit counts for a vector
    #[inline]
    fn increment_counts(&mut self, vector: &BinaryHV) {
        for byte_idx in 0..HV16_BYTES {
            let byte = vector.0[byte_idx];
            for bit_idx in 0..8 {
                if (byte >> bit_idx) & 1 == 1 {
                    self.bit_counts[byte_idx][bit_idx] += 1;
                } else {
                    self.bit_counts[byte_idx][bit_idx] -= 1;
                }
            }
        }
    }

    // Helper: Decrement bit counts for a vector
    #[inline]
    fn decrement_counts(&mut self, vector: &BinaryHV) {
        for byte_idx in 0..HV16_BYTES {
            let byte = vector.0[byte_idx];
            for bit_idx in 0..8 {
                if (byte >> bit_idx) & 1 == 1 {
                    self.bit_counts[byte_idx][bit_idx] -= 1;
                } else {
                    self.bit_counts[byte_idx][bit_idx] += 1;
                }
            }
        }
    }
}

impl Default for IncrementalBundle {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// INCREMENTAL SIMILARITY CACHE - Revolutionary Query Caching
// ============================================================================

/// **REVOLUTIONARY**: Cache similarity computations for stable queries
///
/// Most consciousness operations use the SAME queries repeatedly:
/// - Concept retrieval: Same concept vectors across cycles
/// - Pattern matching: Same pattern templates
/// - Memory recall: Same recall cues
///
/// Traditional approach: Recompute EVERY similarity EVERY time = O(q×m)
/// Incremental approach: Cache results, invalidate on change = O(changed)
///
/// # Performance
///
/// For stable queries (90% query reuse):
/// - Uncached: q×m similarities per cycle
/// - Cached: 0.1×q×m similarities per cycle
/// - **Speedup**: **10x** for 90% cache hit rate!
///
/// # Example
///
/// ```rust,ignore
/// let mut cache = SimilarityCache::new();
/// let query = BinaryHV::random(42);
/// let memory = vec![BinaryHV::random(1), BinaryHV::random(2)];
///
/// // First call - O(m) compute and cache
/// let sim1 = cache.get_similarity(&query, 0, &memory[0]);
///
/// // Second call - O(1) cache hit!
/// let sim2 = cache.get_similarity(&query, 0, &memory[0]);  // Instant!
/// ```
pub struct SimilarityCache {
    /// Cache: (query_id, target_id) -> similarity
    cache: HashMap<(u64, u64), f32>,

    /// Query vectors: query_id -> vector
    queries: HashMap<u64, BinaryHV>,

    /// Next query ID
    next_id: u64,

    /// Cache hit statistics
    hits: usize,
    misses: usize,
}

impl SimilarityCache {
    /// Create new similarity cache
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            queries: HashMap::new(),
            next_id: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// Register a query vector, returns query_id for future lookups
    pub fn register_query(&mut self, query: BinaryHV) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.queries.insert(id, query);
        id
    }

    /// Get similarity with caching - O(1) on cache hit!
    pub fn get_similarity(&mut self, query_id: u64, target_id: u64, target: &BinaryHV) -> f32 {
        let key = (query_id, target_id);

        if let Some(&similarity) = self.cache.get(&key) {
            self.hits += 1;
            return similarity;
        }

        // Cache miss - compute and store
        self.misses += 1;
        if let Some(query) = self.queries.get(&query_id) {
            let similarity = simd_similarity(query, target);
            self.cache.insert(key, similarity);
            similarity
        } else {
            0.0 // Unknown query
        }
    }

    /// Invalidate cache for a specific query (when query vector changes)
    pub fn invalidate_query(&mut self, query_id: u64) {
        self.cache.retain(|(qid, _), _| *qid != query_id);
    }

    /// Invalidate cache for a specific target (when target vector changes)
    pub fn invalidate_target(&mut self, target_id: u64) {
        self.cache.retain(|(_, tid), _| *tid != target_id);
    }

    /// Clear entire cache
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits,
            misses: self.misses,
            hit_rate: if self.hits + self.misses > 0 {
                self.hits as f64 / (self.hits + self.misses) as f64
            } else {
                0.0
            },
            cache_size: self.cache.len(),
        }
    }
}

impl Default for SimilarityCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub hits: usize,
    pub misses: usize,
    pub hit_rate: f64,
    pub cache_size: usize,
}

// ============================================================================
// INCREMENTAL BIND - Smart Batch Binding with Change Tracking
// ============================================================================

/// **REVOLUTIONARY**: Track which vectors changed, only rebind those
///
/// Traditional batch bind: Rebind ALL queries with new key
/// Incremental bind: Track which queries changed, bind only those
///
/// # Performance
///
/// For k changed queries out of n total:
/// - Traditional: O(n) bind operations
/// - Incremental: O(k) bind operations
/// - **Speedup**: n/k (e.g., 100x for k=1, n=100)
pub struct IncrementalBind {
    /// Current queries
    queries: Vec<BinaryHV>,

    /// Current key
    key: BinaryHV,

    /// Cached bound results: query_idx -> bound vector
    cached_results: HashMap<usize, BinaryHV>,

    /// Dirty flags: which queries changed since last bind
    dirty: Vec<bool>,
}

impl IncrementalBind {
    /// Create new incremental bind tracker
    pub fn new(initial_key: BinaryHV) -> Self {
        Self {
            queries: Vec::new(),
            key: initial_key,
            cached_results: HashMap::new(),
            dirty: Vec::new(),
        }
    }

    /// Add queries
    pub fn add_queries(&mut self, queries: Vec<BinaryHV>) {
        let start_idx = self.queries.len();
        self.queries.extend(queries);
        self.dirty.resize(self.queries.len(), true);

        // Mark new queries as dirty
        for i in start_idx..self.queries.len() {
            self.dirty[i] = true;
        }
    }

    /// Update single query - marks as dirty
    pub fn update_query(&mut self, index: usize, new_query: BinaryHV) {
        if index < self.queries.len() {
            self.queries[index] = new_query;
            self.dirty[index] = true;
            self.cached_results.remove(&index);
        }
    }

    /// Update key - marks ALL as dirty
    pub fn update_key(&mut self, new_key: BinaryHV) {
        self.key = new_key;
        for dirty_flag in &mut self.dirty {
            *dirty_flag = true;
        }
        self.cached_results.clear();
    }

    /// Get bound results - O(k) for k dirty queries
    pub fn get_bound_results(&mut self) -> Vec<BinaryHV> {
        // Bind dirty queries only
        for (idx, is_dirty) in self.dirty.iter().enumerate() {
            if *is_dirty {
                let bound = simd_bind(&self.queries[idx], &self.key);
                self.cached_results.insert(idx, bound);
            }
        }

        // Clear dirty flags
        self.dirty.fill(false);

        // Return cached results in order
        (0..self.queries.len())
            .map(|idx| {
                *self
                    .cached_results
                    .get(&idx)
                    .expect("all query indices populated in loop above")
            })
            .collect()
    }

    /// Number of queries
    pub fn len(&self) -> usize {
        self.queries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.queries.is_empty()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ===== IncrementalBundle Construction =====

    #[test]
    fn test_incremental_bundle_new() {
        let bundle = IncrementalBundle::new();
        assert!(bundle.is_empty());
        assert_eq!(bundle.len(), 0);
    }

    #[test]
    fn test_incremental_bundle_default() {
        let bundle = IncrementalBundle::default();
        assert!(bundle.is_empty());
    }

    // ===== IncrementalBundle Correctness =====

    #[test]
    fn test_incremental_bundle_correctness() {
        let vectors = vec![
            BinaryHV::random(0),
            BinaryHV::random(1),
            BinaryHV::random(2),
        ];

        let mut inc_bundle = IncrementalBundle::new();
        inc_bundle.add(vectors.clone());

        let traditional = simd_bundle(&vectors);
        let incremental = inc_bundle.get_bundle();

        assert_eq!(
            traditional.0, incremental.0,
            "Incremental bundle must match traditional"
        );
    }

    #[test]
    fn test_incremental_bundle_single_vector() {
        let v = BinaryHV::random(42);
        let mut bundle = IncrementalBundle::new();
        bundle.add(vec![v]);

        // Single vector bundle: majority vote means bit=1 if count>0 (count=1 for each 1-bit, -1 for each 0-bit)
        let result = bundle.get_bundle();
        assert_eq!(
            result.0, v.0,
            "Bundle of single vector should return that vector"
        );
    }

    #[test]
    fn test_incremental_bundle_add_multiple_batches() {
        let mut bundle = IncrementalBundle::new();
        bundle.add(vec![BinaryHV::random(0), BinaryHV::random(1)]);
        bundle.add(vec![BinaryHV::random(2)]);

        assert_eq!(bundle.len(), 3);

        let all_vectors = vec![
            BinaryHV::random(0),
            BinaryHV::random(1),
            BinaryHV::random(2),
        ];
        let expected = simd_bundle(&all_vectors);
        let actual = bundle.get_bundle();
        assert_eq!(
            actual.0, expected.0,
            "Incremental add should match batch bundle"
        );
    }

    // ===== IncrementalBundle Update =====

    #[test]
    fn test_incremental_bundle_update() {
        let mut bundle = IncrementalBundle::new();
        bundle.add(vec![
            BinaryHV::random(0),
            BinaryHV::random(1),
            BinaryHV::random(2),
        ]);

        let before = bundle.get_bundle();
        bundle.update(1, BinaryHV::random(10));
        let after = bundle.get_bundle();

        assert_ne!(before.0, after.0, "Bundle should change after update");

        let expected = simd_bundle(bundle.vectors());
        assert_eq!(expected.0, after.0, "Updated bundle must be correct");
    }

    #[test]
    fn test_incremental_bundle_update_out_of_bounds() {
        let mut bundle = IncrementalBundle::new();
        bundle.add(vec![BinaryHV::random(0)]);

        let before = bundle.get_bundle();
        bundle.update(999, BinaryHV::random(42)); // Out of bounds, should be no-op
        let after = bundle.get_bundle();

        assert_eq!(before.0, after.0, "Out-of-bounds update should be no-op");
    }

    #[test]
    fn test_incremental_bundle_update_preserves_len() {
        let mut bundle = IncrementalBundle::new();
        bundle.add(vec![BinaryHV::random(0), BinaryHV::random(1)]);
        assert_eq!(bundle.len(), 2);

        bundle.update(0, BinaryHV::random(99));
        assert_eq!(bundle.len(), 2, "Update should not change length");
    }

    // ===== IncrementalBundle Remove =====

    #[test]
    fn test_incremental_bundle_remove() {
        let v0 = BinaryHV::random(0);
        let v1 = BinaryHV::random(1);
        let v2 = BinaryHV::random(2);

        let mut bundle = IncrementalBundle::new();
        bundle.add(vec![v0, v1, v2]);

        let removed = bundle.remove(1);
        assert!(removed.is_some());
        assert_eq!(bundle.len(), 2);

        let expected = simd_bundle(bundle.vectors());
        let actual = bundle.get_bundle();
        assert_eq!(
            expected.0, actual.0,
            "Bundle after removal should be correct"
        );
    }

    #[test]
    fn test_incremental_bundle_remove_out_of_bounds() {
        let mut bundle = IncrementalBundle::new();
        bundle.add(vec![BinaryHV::random(0)]);

        let removed = bundle.remove(5);
        assert!(removed.is_none());
        assert_eq!(bundle.len(), 1);
    }

    // ===== IncrementalBundle Caching =====

    #[test]
    fn test_incremental_bundle_caching() {
        let mut bundle = IncrementalBundle::new();
        bundle.add(vec![BinaryHV::random(0), BinaryHV::random(1)]);

        let first = bundle.get_bundle();
        let cached = bundle.get_bundle(); // Should return cached
        assert_eq!(first.0, cached.0, "Cached result should be identical");
    }

    // ===== SimilarityCache =====

    #[test]
    fn test_similarity_cache_basic() {
        let mut cache = SimilarityCache::new();

        let query = BinaryHV::random(42);
        let target = BinaryHV::random(43);

        let qid = cache.register_query(query);

        let sim1 = cache.get_similarity(qid, 0, &target);
        let sim2 = cache.get_similarity(qid, 0, &target);

        assert_eq!(sim1, sim2, "Cached similarity must match");

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_similarity_cache_multiple_queries() {
        let mut cache = SimilarityCache::new();
        let target = BinaryHV::random(100);

        let q1 = cache.register_query(BinaryHV::random(1));
        let q2 = cache.register_query(BinaryHV::random(2));

        let sim1 = cache.get_similarity(q1, 0, &target);
        let sim2 = cache.get_similarity(q2, 0, &target);

        // Different queries should (likely) have different similarities
        // But both should be valid
        assert!(sim1 >= 0.0 && sim1 <= 1.0);
        assert!(sim2 >= 0.0 && sim2 <= 1.0);
    }

    #[test]
    fn test_similarity_cache_invalidate_query() {
        let mut cache = SimilarityCache::new();
        let target = BinaryHV::random(100);

        let qid = cache.register_query(BinaryHV::random(1));
        let _sim = cache.get_similarity(qid, 0, &target);

        assert_eq!(cache.stats().cache_size, 1);
        cache.invalidate_query(qid);
        assert_eq!(
            cache.stats().cache_size,
            0,
            "Invalidation should clear entries"
        );
    }

    #[test]
    fn test_similarity_cache_invalidate_target() {
        let mut cache = SimilarityCache::new();
        let target = BinaryHV::random(100);

        let qid = cache.register_query(BinaryHV::random(1));
        let _sim = cache.get_similarity(qid, 0, &target);

        cache.invalidate_target(0);
        assert_eq!(
            cache.stats().cache_size,
            0,
            "Target invalidation should clear entries"
        );
    }

    #[test]
    fn test_similarity_cache_clear() {
        let mut cache = SimilarityCache::new();
        let target = BinaryHV::random(100);

        let qid = cache.register_query(BinaryHV::random(1));
        let _ = cache.get_similarity(qid, 0, &target);
        let _ = cache.get_similarity(qid, 1, &target);

        assert_eq!(cache.stats().cache_size, 2);
        cache.clear();
        assert_eq!(cache.stats().cache_size, 0);
    }

    #[test]
    fn test_similarity_cache_unknown_query() {
        let mut cache = SimilarityCache::new();
        let target = BinaryHV::random(100);

        // Query ID 999 not registered
        let sim = cache.get_similarity(999, 0, &target);
        assert_eq!(sim, 0.0, "Unknown query should return 0.0");
    }

    // ===== IncrementalBind =====

    #[test]
    fn test_incremental_bind_correctness() {
        let queries = vec![
            BinaryHV::random(0),
            BinaryHV::random(1),
            BinaryHV::random(2),
        ];
        let key = BinaryHV::random(999);

        let mut inc_bind = IncrementalBind::new(key);
        inc_bind.add_queries(queries.clone());

        let results = inc_bind.get_bound_results();

        for (i, result) in results.iter().enumerate() {
            let expected = simd_bind(&queries[i], &key);
            assert_eq!(
                expected.0, result.0,
                "Incremental bind must match traditional"
            );
        }
    }

    #[test]
    fn test_incremental_bind_update_query() {
        let queries = vec![BinaryHV::random(0), BinaryHV::random(1)];
        let key = BinaryHV::random(999);

        let mut inc_bind = IncrementalBind::new(key);
        inc_bind.add_queries(queries);

        let before = inc_bind.get_bound_results();

        let new_query = BinaryHV::random(50);
        inc_bind.update_query(0, new_query);
        let after = inc_bind.get_bound_results();

        assert_ne!(
            before[0].0, after[0].0,
            "Updated query should produce different result"
        );
        assert_eq!(
            before[1].0, after[1].0,
            "Unchanged query should produce same result"
        );
        assert_eq!(after[0].0, simd_bind(&new_query, &key).0);
    }

    #[test]
    fn test_incremental_bind_update_key() {
        let queries = vec![BinaryHV::random(0), BinaryHV::random(1)];
        let key1 = BinaryHV::random(100);
        let key2 = BinaryHV::random(200);

        let mut inc_bind = IncrementalBind::new(key1);
        inc_bind.add_queries(queries.clone());
        let _before = inc_bind.get_bound_results();

        inc_bind.update_key(key2);
        let after = inc_bind.get_bound_results();

        for (i, result) in after.iter().enumerate() {
            let expected = simd_bind(&queries[i], &key2);
            assert_eq!(expected.0, result.0, "All bindings should use new key");
        }
    }

    #[test]
    fn test_incremental_bind_empty() {
        let mut inc_bind = IncrementalBind::new(BinaryHV::random(42));
        assert!(inc_bind.is_empty());
        assert_eq!(inc_bind.len(), 0);
        let results = inc_bind.get_bound_results();
        assert!(results.is_empty());
    }
}
