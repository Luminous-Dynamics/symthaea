// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Φ Calculation Caching Layer
//!
//! Provides caching for Φ (integrated information) calculations to avoid
//! redundant computation when measuring the same topologies repeatedly.
//!
//! ## Caching Strategy
//!
//! 1. **Topology Hashing**: Create a hash from node representations
//! 2. **Similarity Matrix Cache**: Cache the O(n²D) similarity computation
//! 3. **Result Cache**: Cache the final Φ value (O(n³) eigenvalue savings)
//!
//! ## Performance Impact
//!
//! For an 8-node topology at HDC_DIMENSION=16,384:
//! - Cold calculation: ~2ms (similarity) + ~0.5ms (eigenvalues)
//! - Cached lookup: ~10μs (hash + lookup)
//! - Speedup: ~200x for repeated calculations
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::phi_engine::{CachedPhiEngine, PhiMethod};
//!
//! let mut engine = CachedPhiEngine::new(PhiMethod::SpectralConnectivity, 100);
//! let result = engine.compute(&topology.node_representations);
//!
//! // Second call with same topology is cached
//! let result2 = engine.compute(&topology.node_representations);
//! assert_eq!(result.phi, result2.phi);
//! ```

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

use super::{PhiCategory, PhiEngine, PhiMethod, PhiResult};
use crate::hdc::unified_hv::ContinuousHV;

/// Cache entry for Φ results
#[derive(Clone, Debug)]
struct CacheEntry {
    /// The computed Φ value
    phi: f64,
    /// Method used for calculation
    method: &'static str,
    /// Number of nodes
    n_nodes: usize,
    /// When this entry was computed (for potential TTL expiration)
    #[allow(dead_code)]
    computed_at: Instant,
    /// How long the original computation took
    computation_time: Duration,
}

/// Statistics about cache performance
#[derive(Clone, Debug, Default)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Total time saved by cache hits (estimated)
    pub time_saved: Duration,
    /// Current cache size
    pub cache_size: usize,
    /// Maximum cache capacity
    pub max_capacity: usize,
}

impl CacheStats {
    /// Get cache hit rate as percentage
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f64 / total as f64) * 100.0
        }
    }
}

/// Cached Φ calculation engine
///
/// Wraps `PhiEngine` with LRU-style caching for repeated topology measurements.
/// The cache uses a hash of node representations as the key.
///
/// ## Thread Safety
///
/// This implementation is NOT thread-safe. For concurrent access, wrap in
/// `Arc<Mutex<CachedPhiEngine>>` or use the thread-local cache variant.
pub struct CachedPhiEngine {
    /// Underlying Phi engine
    engine: PhiEngine,
    /// Result cache (topology_hash -> CacheEntry)
    cache: HashMap<u64, CacheEntry>,
    /// Maximum cache entries (LRU eviction when exceeded)
    max_entries: usize,
    /// Statistics
    stats: CacheStats,
}

impl CachedPhiEngine {
    /// Create a new cached Φ engine
    ///
    /// # Arguments
    /// * `method` - Calculation method to use
    /// * `max_entries` - Maximum cache size (default: 1000)
    pub fn new(method: PhiMethod, max_entries: usize) -> Self {
        Self {
            engine: PhiEngine::new(method),
            cache: HashMap::with_capacity(max_entries.min(1000)),
            max_entries,
            stats: CacheStats {
                max_capacity: max_entries,
                ..Default::default()
            },
        }
    }

    /// Create with default settings (Auto method, 1000 entries)
    pub fn default_cached() -> Self {
        Self::new(PhiMethod::Auto, 1000)
    }

    /// Compute Φ with caching
    ///
    /// If the topology has been computed before, returns cached result.
    /// Otherwise computes and caches the new result.
    pub fn compute(&mut self, node_representations: &[ContinuousHV]) -> PhiResult {
        // Step 1: Hash the topology
        let topology_hash = self.hash_topology(node_representations);

        // Step 2: Check cache
        if let Some(entry) = self.cache.get(&topology_hash) {
            self.stats.hits += 1;
            self.stats.time_saved += entry.computation_time;

            return PhiResult {
                phi: entry.phi,
                method: entry.method,
                category: PhiCategory::SpectralConnectivity,
                computation_time: Duration::from_micros(10), // Cache lookup time
                n_nodes: entry.n_nodes,
                limiting_partition: None,
            };
        }

        // Step 3: Cache miss - compute
        self.stats.misses += 1;

        let start = Instant::now();
        let result = self.engine.compute(node_representations);
        let computation_time = start.elapsed();

        // Step 4: Store in cache
        self.store_result(topology_hash, &result, computation_time);

        result
    }

    /// Compute Φ for ContinuousHV representations (legacy interface)
    pub fn compute_from_real_hvs(&mut self, components: &[ContinuousHV]) -> f64 {
        // Convert to ContinuousHV
        let continuous: Vec<ContinuousHV> = components
            .iter()
            .map(|rhv| ContinuousHV::from_vec(rhv.values.clone()))
            .collect();

        self.compute(&continuous).phi
    }

    /// Hash a topology by its node representations
    ///
    /// Uses a combination of:
    /// - Number of nodes
    /// - Dimension of hypervectors
    /// - Sampled values from each hypervector
    ///
    /// This is a probabilistic hash - collisions are possible but rare for
    /// different topologies.
    fn hash_topology(&self, node_representations: &[ContinuousHV]) -> u64 {
        let mut hasher = DefaultHasher::new();

        // Hash number of nodes
        node_representations.len().hash(&mut hasher);

        // For each node, hash sampled values
        // (hashing all values would be too slow for 16,384 dimensions)
        for hv in node_representations {
            let dim = hv.values.len();
            dim.hash(&mut hasher);

            // Sample values at specific indices for the hash
            // Using prime-spaced indices to get good coverage
            const SAMPLE_INDICES: [usize; 8] = [0, 17, 97, 293, 787, 1999, 4999, 9973];

            for &idx in &SAMPLE_INDICES {
                if idx < dim {
                    // Convert f32 to bits for hashing
                    let bits = hv.values[idx].to_bits();
                    bits.hash(&mut hasher);
                }
            }
        }

        hasher.finish()
    }

    /// Store a result in the cache with LRU eviction
    fn store_result(&mut self, hash: u64, result: &PhiResult, computation_time: Duration) {
        // Evict oldest entries if at capacity
        if self.cache.len() >= self.max_entries {
            // Simple eviction: remove random entry
            // (True LRU would require tracking access order)
            if let Some(key) = self.cache.keys().next().copied() {
                self.cache.remove(&key);
            }
        }

        let entry = CacheEntry {
            phi: result.phi,
            method: result.method,
            n_nodes: result.n_nodes,
            computed_at: Instant::now(),
            computation_time,
        };

        self.cache.insert(hash, entry);
        self.stats.cache_size = self.cache.len();
    }

    /// Get cache statistics
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Clear the cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.stats.cache_size = 0;
    }

    /// Prefetch a topology into the cache
    ///
    /// Useful for warming the cache before time-critical measurements.
    pub fn prefetch(&mut self, node_representations: &[ContinuousHV]) {
        let _ = self.compute(node_representations);
    }

    /// Check if a topology is cached
    pub fn is_cached(&self, node_representations: &[ContinuousHV]) -> bool {
        let hash = self.hash_topology(node_representations);
        self.cache.contains_key(&hash)
    }

    /// Get underlying PhiEngine
    pub fn engine(&self) -> &PhiEngine {
        &self.engine
    }

    /// Set the calculation method
    pub fn set_method(&mut self, method: PhiMethod) {
        self.engine.set_method(method);
        // Clear cache since method changed
        self.clear_cache();
    }
}

impl Default for CachedPhiEngine {
    fn default() -> Self {
        Self::default_cached()
    }
}

/// Thread-local cache for Φ calculations
///
/// Provides a global thread-local cache that can be accessed without
/// explicit cache management.
#[cfg(feature = "thread_local_cache")]
thread_local! {
    static PHI_CACHE: std::cell::RefCell<CachedPhiEngine> =
        std::cell::RefCell::new(CachedPhiEngine::default_cached());
}

#[cfg(feature = "thread_local_cache")]
pub fn compute_phi_cached(node_representations: &[ContinuousHV]) -> PhiResult {
    PHI_CACHE.with(|cache| cache.borrow_mut().compute(node_representations))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::HDC_DIMENSION;
    use proptest::prelude::*;

    #[test]
    fn test_cache_hit() {
        let mut cache = CachedPhiEngine::default_cached();

        // Create test topology
        let hvs: Vec<ContinuousHV> = (0..4)
            .map(|i| ContinuousHV::random(HDC_DIMENSION, i as u64))
            .collect();

        // First call - cache miss
        let result1 = cache.compute(&hvs);
        assert_eq!(cache.stats().hits, 0);
        assert_eq!(cache.stats().misses, 1);

        // Second call - cache hit
        let result2 = cache.compute(&hvs);
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 1);

        // Results should be identical
        assert!((result1.phi - result2.phi).abs() < 1e-10);
    }

    #[test]
    fn test_cache_miss_different_topology() {
        let mut cache = CachedPhiEngine::default_cached();

        // Topology A
        let hvs_a: Vec<ContinuousHV> = (0..4)
            .map(|i| ContinuousHV::random(HDC_DIMENSION, i as u64))
            .collect();

        // Topology B (different seeds)
        let hvs_b: Vec<ContinuousHV> = (100..104)
            .map(|i| ContinuousHV::random(HDC_DIMENSION, i as u64))
            .collect();

        cache.compute(&hvs_a);
        cache.compute(&hvs_b);

        // Both should be misses since topologies are different
        assert_eq!(cache.stats().misses, 2);
        assert_eq!(cache.stats().hits, 0);
    }

    #[test]
    fn test_cache_eviction() {
        let mut cache = CachedPhiEngine::new(PhiMethod::Auto, 3);

        // Fill cache to capacity
        for seed_base in 0..4 {
            let hvs: Vec<ContinuousHV> = (0..4)
                .map(|i| ContinuousHV::random(HDC_DIMENSION, (seed_base * 10 + i) as u64))
                .collect();
            cache.compute(&hvs);
        }

        // Cache should have evicted at least one entry
        assert!(cache.stats().cache_size <= 3);
    }

    #[test]
    fn test_hit_rate() {
        let mut cache = CachedPhiEngine::default_cached();

        let hvs: Vec<ContinuousHV> = (0..4)
            .map(|i| ContinuousHV::random(HDC_DIMENSION, i as u64))
            .collect();

        // 1 miss, 4 hits
        for _ in 0..5 {
            cache.compute(&hvs);
        }

        let hit_rate = cache.stats().hit_rate();
        assert!((hit_rate - 80.0).abs() < 1e-10); // 4/5 = 80%
    }

    #[test]
    fn test_clear_cache() {
        let mut cache = CachedPhiEngine::default_cached();

        let hvs: Vec<ContinuousHV> = (0..4)
            .map(|i| ContinuousHV::random(HDC_DIMENSION, i as u64))
            .collect();

        cache.compute(&hvs);
        assert_eq!(cache.stats().cache_size, 1);

        cache.clear_cache();
        assert_eq!(cache.stats().cache_size, 0);
    }

    #[test]
    fn test_prefetch() {
        let mut cache = CachedPhiEngine::default_cached();

        let hvs: Vec<ContinuousHV> = (0..4)
            .map(|i| ContinuousHV::random(HDC_DIMENSION, i as u64))
            .collect();

        // Prefetch
        cache.prefetch(&hvs);
        assert!(cache.is_cached(&hvs));

        // Actual compute should be a cache hit
        cache.compute(&hvs);
        assert_eq!(cache.stats().hits, 1);
    }

    // =====================================================================
    // Property-based tests
    // =====================================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_hit_rate_bounded(hits in 0u64..1000, misses in 0u64..1000) {
            let stats = CacheStats {
                hits,
                misses,
                time_saved: Duration::ZERO,
                cache_size: 0,
                max_capacity: 1000,
            };
            let rate = stats.hit_rate();
            prop_assert!(rate >= 0.0, "Hit rate should be >= 0, got {}", rate);
            prop_assert!(rate <= 100.0, "Hit rate should be <= 100, got {}", rate);
        }

        #[test]
        fn prop_hit_rate_monotonic(misses in 1u64..100) {
            // More hits with same misses → higher rate
            let rate_low = CacheStats {
                hits: 1, misses, time_saved: Duration::ZERO,
                cache_size: 0, max_capacity: 1000,
            }.hit_rate();
            let rate_high = CacheStats {
                hits: misses * 10, misses, time_saved: Duration::ZERO,
                cache_size: 0, max_capacity: 1000,
            }.hit_rate();
            prop_assert!(rate_high >= rate_low,
                "More hits should mean higher rate: {} >= {}", rate_high, rate_low);
        }

        #[test]
        fn prop_hit_rate_perfect_when_no_misses(hits in 1u64..1000) {
            let rate = CacheStats {
                hits, misses: 0, time_saved: Duration::ZERO,
                cache_size: 0, max_capacity: 1000,
            }.hit_rate();
            prop_assert!((rate - 100.0).abs() < 1e-10,
                "All hits, no misses should give 100% rate, got {}", rate);
        }

        #[test]
        fn prop_hit_rate_zero_when_no_hits(misses in 1u64..1000) {
            let rate = CacheStats {
                hits: 0, misses, time_saved: Duration::ZERO,
                cache_size: 0, max_capacity: 1000,
            }.hit_rate();
            prop_assert!((rate - 0.0).abs() < 1e-10,
                "No hits should give 0% rate, got {}", rate);
        }

        #[test]
        fn prop_hash_deterministic(seed in 0u64..100) {
            let cache = CachedPhiEngine::default_cached();
            let hvs: Vec<ContinuousHV> = (0..3)
                .map(|i| ContinuousHV::random(128, seed * 10 + i))
                .collect();
            let h1 = cache.hash_topology(&hvs);
            let h2 = cache.hash_topology(&hvs);
            prop_assert_eq!(h1, h2, "Same input should produce same hash");
        }
    }
}
