// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use super::{PrimitiveError, PrimitiveResult, PrimitiveSystem};
use std::collections::HashMap;

/// Cache for memoizing primitive composition operations.
///
/// Stores results of bind, bundle, sequence, and other operations to avoid
/// redundant computation. Uses operation strings as keys.
///
/// # Example
/// ```ignore
/// let mut cache = CompositionCache::new(1000);
/// let system = PrimitiveSystem::global();
///
/// // First call computes and caches
/// let result1 = cache.bind_cached(system, "CAUSE", "EFFECT").unwrap();
///
/// // Second call retrieves from cache
/// let result2 = cache.bind_cached(system, "CAUSE", "EFFECT").unwrap();
/// assert_eq!(result1.encoding, result2.encoding);
/// ```
#[derive(Debug)]
pub struct CompositionCache {
    /// Cached results: operation_key -> PrimitiveResult
    cache: HashMap<String, PrimitiveResult>,
    /// Maximum cache size (LRU eviction when exceeded)
    max_size: usize,
    /// Access order for LRU (operation_key -> access_count)
    access_order: HashMap<String, u64>,
    /// Global access counter
    access_counter: u64,
    /// Cache statistics
    hits: u64,
    misses: u64,
}

impl CompositionCache {
    /// Create a new composition cache with the given maximum size.
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(max_size),
            max_size,
            access_order: HashMap::with_capacity(max_size),
            access_counter: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// Create a cache with default size (1000 entries).
    pub fn default_size() -> Self {
        Self::new(1000)
    }

    /// Bind two primitives with caching.
    pub fn bind_cached(
        &mut self,
        system: &PrimitiveSystem,
        a: &str,
        b: &str,
    ) -> Result<PrimitiveResult, PrimitiveError> {
        let key = format!("bind:{a}:{b}");

        if let Some(result) = self.get(&key) {
            return Ok(result.clone());
        }

        let result = system.bind_primitives(a, b)?;
        self.put(key, result.clone());
        Ok(result)
    }

    /// Bundle primitives with caching.
    pub fn bundle_cached(
        &mut self,
        system: &PrimitiveSystem,
        names: &[&str],
    ) -> Result<PrimitiveResult, PrimitiveError> {
        let key = format!("bundle:{}", names.join(":"));

        if let Some(result) = self.get(&key) {
            return Ok(result.clone());
        }

        let result = system.bundle_primitives(names)?;
        self.put(key, result.clone());
        Ok(result)
    }

    /// Weighted bundle with caching.
    pub fn bundle_weighted_cached(
        &mut self,
        system: &PrimitiveSystem,
        weighted: &[(&str, f32)],
    ) -> Result<PrimitiveResult, PrimitiveError> {
        // Create a key that includes weights (rounded for cache friendliness)
        let key = format!(
            "bundle_weighted:{}",
            weighted
                .iter()
                .map(|(n, w)| format!("{n}:{w:.2}"))
                .collect::<Vec<_>>()
                .join(":")
        );

        if let Some(result) = self.get(&key) {
            return Ok(result.clone());
        }

        let result = system.bundle_weighted(weighted)?;
        self.put(key, result.clone());
        Ok(result)
    }

    /// Encode sequence with caching.
    pub fn sequence_cached(
        &mut self,
        system: &PrimitiveSystem,
        names: &[&str],
    ) -> Result<PrimitiveResult, PrimitiveError> {
        let key = format!("sequence:{}", names.join(":"));

        if let Some(result) = self.get(&key) {
            return Ok(result.clone());
        }

        let result = system.encode_sequence(names)?;
        self.put(key, result.clone());
        Ok(result)
    }

    /// Analogy with caching.
    pub fn analogy_cached(
        &mut self,
        system: &PrimitiveSystem,
        a: &str,
        b: &str,
        c: &str,
    ) -> Result<PrimitiveResult, PrimitiveError> {
        let key = format!("analogy:{a}:{b}:{c}");

        if let Some(result) = self.get(&key) {
            return Ok(result.clone());
        }

        let result = system.analogy(a, b, c)?;
        self.put(key, result.clone());
        Ok(result)
    }

    /// Permute primitive with caching.
    pub fn permute_cached(
        &mut self,
        system: &PrimitiveSystem,
        name: &str,
        steps: usize,
    ) -> Result<PrimitiveResult, PrimitiveError> {
        let key = format!("permute:{name}:{steps}");

        if let Some(result) = self.get(&key) {
            return Ok(result.clone());
        }

        let result = system.permute_primitive(name, steps)?;
        self.put(key, result.clone());
        Ok(result)
    }

    /// Get an entry from the cache, updating access order.
    fn get(&mut self, key: &str) -> Option<&PrimitiveResult> {
        if self.cache.contains_key(key) {
            self.hits += 1;
            self.access_counter += 1;
            self.access_order
                .insert(key.to_string(), self.access_counter);
            self.cache.get(key)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Put an entry in the cache, evicting LRU if necessary.
    fn put(&mut self, key: String, value: PrimitiveResult) {
        // Evict if at capacity
        if self.cache.len() >= self.max_size {
            self.evict_lru();
        }

        self.access_counter += 1;
        self.access_order.insert(key.clone(), self.access_counter);
        self.cache.insert(key, value);
    }

    /// Evict the least recently used entry.
    fn evict_lru(&mut self) {
        if let Some((lru_key, _)) = self
            .access_order
            .iter()
            .min_by_key(|&(_, &access_time)| access_time)
            .map(|(k, v)| (k.clone(), *v))
        {
            self.cache.remove(&lru_key);
            self.access_order.remove(&lru_key);
        }
    }

    /// Clear the cache and reset statistics.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.access_order.clear();
        self.access_counter = 0;
        self.hits = 0;
        self.misses = 0;
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        let total = self.hits + self.misses;
        CacheStats {
            size: self.cache.len(),
            max_size: self.max_size,
            hits: self.hits,
            misses: self.misses,
            hit_rate: if total > 0 {
                self.hits as f32 / total as f32
            } else {
                0.0
            },
        }
    }
}

impl Default for CompositionCache {
    fn default() -> Self {
        Self::default_size()
    }
}

/// Statistics about the composition cache
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub size: usize,
    pub max_size: usize,
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f32,
}
