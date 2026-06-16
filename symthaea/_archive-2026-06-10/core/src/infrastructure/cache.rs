// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! F2: LRU Cache for HDC Encodings
//!
//! Provides efficient caching of semantic vectors to avoid recomputation.
//! Uses an LRU (Least Recently Used) eviction policy.

use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::time::{Duration, Instant};

/// Generic LRU cache with TTL support
#[derive(Debug)]
pub struct LruCache<K, V> {
    /// Cached entries
    entries: HashMap<K, CacheEntry<V>>,
    /// Access order for LRU eviction (most recent at end)
    access_order: VecDeque<K>,
    /// Maximum cache size
    max_size: usize,
    /// Time-to-live for entries
    ttl: Option<Duration>,
    /// Cache statistics
    stats: CacheStats,
}

#[derive(Debug, Clone)]
struct CacheEntry<V> {
    value: V,
    created_at: Instant,
    last_accessed: Instant,
    access_count: u64,
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub expirations: u64,
    pub total_size: usize,
}

impl CacheStats {
    /// Hit rate as percentage
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f64 / total as f64) * 100.0
        }
    }
}

impl<K: Eq + Hash + Clone, V: Clone> LruCache<K, V> {
    /// Create a new LRU cache with given capacity
    pub fn new(max_size: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(max_size),
            access_order: VecDeque::with_capacity(max_size),
            max_size,
            ttl: None,
            stats: CacheStats::default(),
        }
    }

    /// Create with TTL
    pub fn with_ttl(max_size: usize, ttl: Duration) -> Self {
        Self {
            entries: HashMap::with_capacity(max_size),
            access_order: VecDeque::with_capacity(max_size),
            max_size,
            ttl: Some(ttl),
            stats: CacheStats::default(),
        }
    }

    /// Get a value from cache
    pub fn get(&mut self, key: &K) -> Option<V> {
        // First check if entry exists and if it's expired (without holding borrow)
        let is_expired = self.entries.get(key).map(|entry| {
            if let Some(ttl) = self.ttl {
                entry.created_at.elapsed() > ttl
            } else {
                false
            }
        });

        match is_expired {
            Some(true) => {
                // Entry exists but is expired
                self.remove(key);
                self.stats.expirations += 1;
                self.stats.misses += 1;
                None
            }
            Some(false) => {
                // Entry exists and is valid - update access order first
                self.touch(key);

                // Now update entry and get value
                if let Some(entry) = self.entries.get_mut(key) {
                    entry.last_accessed = Instant::now();
                    entry.access_count += 1;
                    self.stats.hits += 1;
                    Some(entry.value.clone())
                } else {
                    self.stats.misses += 1;
                    None
                }
            }
            None => {
                // Entry doesn't exist
                self.stats.misses += 1;
                None
            }
        }
    }

    /// Insert a value into cache
    pub fn insert(&mut self, key: K, value: V) {
        // Evict if at capacity
        while self.entries.len() >= self.max_size {
            self.evict_lru();
        }

        let now = Instant::now();
        let entry = CacheEntry {
            value,
            created_at: now,
            last_accessed: now,
            access_count: 1,
        };

        // Remove from access order if exists
        self.access_order.retain(|k| k != &key);
        self.access_order.push_back(key.clone());

        self.entries.insert(key, entry);
        self.stats.total_size = self.entries.len();
    }

    /// Remove a key from cache
    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.access_order.retain(|k| k != key);
        let removed = self.entries.remove(key).map(|e| e.value);
        self.stats.total_size = self.entries.len();
        removed
    }

    /// Check if key exists (without updating access order)
    pub fn contains(&self, key: &K) -> bool {
        self.entries.contains_key(key)
    }

    /// Get current size
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.entries.clear();
        self.access_order.clear();
        self.stats.total_size = 0;
    }

    /// Get cache statistics
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Move key to end of access order (most recently used)
    fn touch(&mut self, key: &K) {
        self.access_order.retain(|k| k != key);
        self.access_order.push_back(key.clone());
    }

    /// Evict least recently used entry
    fn evict_lru(&mut self) {
        if let Some(key) = self.access_order.front().cloned() {
            self.entries.remove(&key);
            self.access_order.pop_front();
            self.stats.evictions += 1;
            self.stats.total_size = self.entries.len();
        }
    }

    /// Prune expired entries
    pub fn prune_expired(&mut self) {
        if let Some(ttl) = self.ttl {
            let expired: Vec<K> = self
                .entries
                .iter()
                .filter(|(_, e)| e.created_at.elapsed() > ttl)
                .map(|(k, _)| k.clone())
                .collect();

            for key in expired {
                self.remove(&key);
                self.stats.expirations += 1;
            }
        }
    }
}

/// Specialized cache for HDC semantic vectors
pub struct HdcCache {
    /// Cache for text -> BinaryHV encodings
    text_cache: LruCache<String, Vec<u16>>,
    /// Cache for command -> completion list
    completion_cache: LruCache<String, Vec<String>>,
    /// Cache for option path -> semantic similarity scores
    similarity_cache: LruCache<(String, String), f32>,
}

impl HdcCache {
    /// Create new HDC cache with default sizes
    pub fn new() -> Self {
        Self {
            text_cache: LruCache::with_ttl(10000, Duration::from_secs(3600)), // 1 hour
            completion_cache: LruCache::with_ttl(1000, Duration::from_secs(300)), // 5 min
            similarity_cache: LruCache::with_ttl(50000, Duration::from_secs(1800)), // 30 min
        }
    }

    /// Create with custom sizes
    pub fn with_sizes(text_size: usize, completion_size: usize, similarity_size: usize) -> Self {
        Self {
            text_cache: LruCache::with_ttl(text_size, Duration::from_secs(3600)),
            completion_cache: LruCache::with_ttl(completion_size, Duration::from_secs(300)),
            similarity_cache: LruCache::with_ttl(similarity_size, Duration::from_secs(1800)),
        }
    }

    /// Cache a text encoding
    pub fn cache_encoding(&mut self, text: &str, hv: Vec<u16>) {
        self.text_cache.insert(text.to_string(), hv);
    }

    /// Get cached encoding
    pub fn get_encoding(&mut self, text: &str) -> Option<Vec<u16>> {
        self.text_cache.get(&text.to_string())
    }

    /// Cache completions for a prefix
    pub fn cache_completions(&mut self, prefix: &str, completions: Vec<String>) {
        self.completion_cache
            .insert(prefix.to_string(), completions);
    }

    /// Get cached completions
    pub fn get_completions(&mut self, prefix: &str) -> Option<Vec<String>> {
        self.completion_cache.get(&prefix.to_string())
    }

    /// Cache similarity score
    pub fn cache_similarity(&mut self, a: &str, b: &str, score: f32) {
        let key = if a < b {
            (a.to_string(), b.to_string())
        } else {
            (b.to_string(), a.to_string())
        };
        self.similarity_cache.insert(key, score);
    }

    /// Get cached similarity
    pub fn get_similarity(&mut self, a: &str, b: &str) -> Option<f32> {
        let key = if a < b {
            (a.to_string(), b.to_string())
        } else {
            (b.to_string(), a.to_string())
        };
        self.similarity_cache.get(&key)
    }

    /// Get combined stats
    pub fn stats(&self) -> CombinedCacheStats {
        CombinedCacheStats {
            text: self.text_cache.stats().clone(),
            completion: self.completion_cache.stats().clone(),
            similarity: self.similarity_cache.stats().clone(),
        }
    }

    /// Clear all caches
    pub fn clear(&mut self) {
        self.text_cache.clear();
        self.completion_cache.clear();
        self.similarity_cache.clear();
    }

    /// Prune expired entries
    pub fn prune(&mut self) {
        self.text_cache.prune_expired();
        self.completion_cache.prune_expired();
        self.similarity_cache.prune_expired();
    }
}

impl Default for HdcCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Combined cache statistics
#[derive(Debug, Clone)]
pub struct CombinedCacheStats {
    pub text: CacheStats,
    pub completion: CacheStats,
    pub similarity: CacheStats,
}

impl CombinedCacheStats {
    /// Overall hit rate
    pub fn overall_hit_rate(&self) -> f64 {
        let total_hits = self.text.hits + self.completion.hits + self.similarity.hits;
        let total_misses = self.text.misses + self.completion.misses + self.similarity.misses;
        let total = total_hits + total_misses;
        if total == 0 {
            0.0
        } else {
            (total_hits as f64 / total as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru_basic() {
        let mut cache: LruCache<String, i32> = LruCache::new(3);

        cache.insert("a".to_string(), 1);
        cache.insert("b".to_string(), 2);
        cache.insert("c".to_string(), 3);

        assert_eq!(cache.get(&"a".to_string()), Some(1));
        assert_eq!(cache.get(&"b".to_string()), Some(2));
        assert_eq!(cache.get(&"c".to_string()), Some(3));
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache: LruCache<String, i32> = LruCache::new(2);

        cache.insert("a".to_string(), 1);
        cache.insert("b".to_string(), 2);
        cache.insert("c".to_string(), 3); // Should evict "a"

        assert_eq!(cache.get(&"a".to_string()), None);
        assert_eq!(cache.get(&"b".to_string()), Some(2));
        assert_eq!(cache.get(&"c".to_string()), Some(3));
    }

    #[test]
    fn test_lru_access_order() {
        let mut cache: LruCache<String, i32> = LruCache::new(2);

        cache.insert("a".to_string(), 1);
        cache.insert("b".to_string(), 2);

        // Access "a" to make it more recent
        cache.get(&"a".to_string());

        cache.insert("c".to_string(), 3); // Should evict "b"

        assert_eq!(cache.get(&"a".to_string()), Some(1));
        assert_eq!(cache.get(&"b".to_string()), None);
        assert_eq!(cache.get(&"c".to_string()), Some(3));
    }

    #[test]
    fn test_cache_stats() {
        let mut cache: LruCache<String, i32> = LruCache::new(10);

        cache.insert("a".to_string(), 1);
        cache.get(&"a".to_string()); // Hit
        cache.get(&"b".to_string()); // Miss

        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hit_rate(), 50.0);
    }

    #[test]
    fn test_hdc_cache() {
        let mut cache = HdcCache::new();

        cache.cache_encoding("test", vec![1, 2, 3]);
        assert_eq!(cache.get_encoding("test"), Some(vec![1, 2, 3]));
        assert_eq!(cache.get_encoding("missing"), None);

        cache.cache_similarity("a", "b", 0.85);
        assert_eq!(cache.get_similarity("a", "b"), Some(0.85));
        assert_eq!(cache.get_similarity("b", "a"), Some(0.85)); // Symmetric
    }
}
