//! Proof Caching Layer
//!
//! In-memory cache for frequently accessed proofs to avoid regeneration.
//!
//! ## Features
//!
//! - LRU eviction policy
//! - TTL-based expiration
//! - Thread-safe access
//! - Hit/miss statistics
//!
//! ## Usage
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::integration::{ProofCache, CacheConfig};
//!
//! let cache = ProofCache::new(CacheConfig {
//!     max_entries: 1000,
//!     ttl_seconds: 3600,
//! });
//!
//! // Store a proof
//! cache.put("key", proof_envelope);
//!
//! // Retrieve
//! if let Some(proof) = cache.get("key") {
//!     // Use cached proof
//! }
//! ```

use crate::proofs::ProofType;
use super::serialization::ProofEnvelope;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Cache configuration
#[derive(Clone, Debug)]
pub struct CacheConfig {
    /// Maximum number of entries
    pub max_entries: usize,

    /// Time-to-live in seconds (0 = no expiration)
    pub ttl_seconds: u64,

    /// Enable statistics tracking
    pub track_stats: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            ttl_seconds: 3600, // 1 hour
            track_stats: true,
        }
    }
}

/// Cache entry with metadata
#[derive(Clone)]
struct CacheEntry {
    envelope: ProofEnvelope,
    created_at: Instant,
    last_accessed: Instant,
    access_count: u64,
}

impl CacheEntry {
    fn new(envelope: ProofEnvelope) -> Self {
        let now = Instant::now();
        Self {
            envelope,
            created_at: now,
            last_accessed: now,
            access_count: 1,
        }
    }

    fn is_expired(&self, ttl: Duration) -> bool {
        if ttl.is_zero() {
            return false;
        }
        self.created_at.elapsed() > ttl
    }

    fn touch(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }
}

/// Cache statistics
#[derive(Clone, Debug, Default)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: u64,

    /// Number of cache misses
    pub misses: u64,

    /// Number of evictions
    pub evictions: u64,

    /// Number of expirations
    pub expirations: u64,

    /// Current entry count
    pub entry_count: usize,

    /// Total proof bytes stored
    pub total_bytes: usize,
}

impl CacheStats {
    /// Calculate hit rate (0.0 to 1.0)
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Thread-safe proof cache
pub struct ProofCache {
    entries: Arc<RwLock<HashMap<String, CacheEntry>>>,
    config: CacheConfig,
    stats: Arc<RwLock<CacheStats>>,
}

impl ProofCache {
    /// Create a new cache with configuration
    pub fn new(config: CacheConfig) -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            config,
            stats: Arc::new(RwLock::new(CacheStats::default())),
        }
    }

    /// Create a cache with default configuration
    pub fn with_defaults() -> Self {
        Self::new(CacheConfig::default())
    }

    /// Store a proof in the cache
    pub fn put(&self, key: impl Into<String>, envelope: ProofEnvelope) {
        let key = key.into();
        let entry_size = envelope.size();

        // Recover from poisoned lock (another thread panicked while holding it)
        let mut entries = self.entries.write().unwrap_or_else(|e| e.into_inner());

        // Check if we need to evict
        if entries.len() >= self.config.max_entries && !entries.contains_key(&key) {
            self.evict_lru(&mut entries);
        }

        entries.insert(key, CacheEntry::new(envelope));

        if self.config.track_stats {
            let mut stats = self.stats.write().unwrap_or_else(|e| e.into_inner());
            stats.entry_count = entries.len();
            stats.total_bytes += entry_size;
        }
    }

    /// Retrieve a proof from the cache
    pub fn get(&self, key: &str) -> Option<ProofEnvelope> {
        let ttl = Duration::from_secs(self.config.ttl_seconds);

        // Recover from poisoned lock
        let mut entries = self.entries.write().unwrap_or_else(|e| e.into_inner());

        if let Some(entry) = entries.get_mut(key) {
            // Check expiration
            if entry.is_expired(ttl) {
                let size = entry.envelope.size();
                entries.remove(key);

                if self.config.track_stats {
                    let mut stats = self.stats.write().unwrap_or_else(|e| e.into_inner());
                    stats.expirations += 1;
                    stats.misses += 1;
                    stats.entry_count = entries.len();
                    stats.total_bytes = stats.total_bytes.saturating_sub(size);
                }
                return None;
            }

            entry.touch();

            if self.config.track_stats {
                let mut stats = self.stats.write().unwrap_or_else(|e| e.into_inner());
                stats.hits += 1;
            }

            return Some(entry.envelope.clone());
        }

        if self.config.track_stats {
            let mut stats = self.stats.write().unwrap_or_else(|e| e.into_inner());
            stats.misses += 1;
        }

        None
    }

    /// Check if a key exists (without updating access time)
    pub fn contains(&self, key: &str) -> bool {
        let entries = self.entries.read().unwrap_or_else(|e| e.into_inner());
        entries.contains_key(key)
    }

    /// Remove a specific entry
    pub fn remove(&self, key: &str) -> Option<ProofEnvelope> {
        let mut entries = self.entries.write().unwrap_or_else(|e| e.into_inner());

        if let Some(entry) = entries.remove(key) {
            if self.config.track_stats {
                let mut stats = self.stats.write().unwrap_or_else(|e| e.into_inner());
                stats.entry_count = entries.len();
                stats.total_bytes = stats.total_bytes.saturating_sub(entry.envelope.size());
            }
            Some(entry.envelope)
        } else {
            None
        }
    }

    /// Clear all entries
    pub fn clear(&self) {
        let mut entries = self.entries.write().unwrap_or_else(|e| e.into_inner());
        entries.clear();

        if self.config.track_stats {
            let mut stats = self.stats.write().unwrap_or_else(|e| e.into_inner());
            stats.entry_count = 0;
            stats.total_bytes = 0;
        }
    }

    /// Get current statistics
    pub fn stats(&self) -> CacheStats {
        self.stats.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Get the number of entries
    pub fn len(&self) -> usize {
        self.entries.read().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.entries.read().unwrap_or_else(|e| e.into_inner()).is_empty()
    }

    /// Evict least recently used entry
    fn evict_lru(&self, entries: &mut HashMap<String, CacheEntry>) {
        let lru_key = entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_accessed)
            .map(|(key, _)| key.clone());

        if let Some(key) = lru_key {
            if let Some(entry) = entries.remove(&key) {
                if self.config.track_stats {
                    let mut stats = self.stats.write().unwrap_or_else(|e| e.into_inner());
                    stats.evictions += 1;
                    stats.total_bytes = stats.total_bytes.saturating_sub(entry.envelope.size());
                }
            }
        }
    }

    /// Purge expired entries
    pub fn purge_expired(&self) -> usize {
        let ttl = Duration::from_secs(self.config.ttl_seconds);
        if ttl.is_zero() {
            return 0;
        }

        let mut entries = self.entries.write().unwrap_or_else(|e| e.into_inner());
        let initial_count = entries.len();

        let expired_keys: Vec<String> = entries
            .iter()
            .filter(|(_, entry)| entry.is_expired(ttl))
            .map(|(key, _)| key.clone())
            .collect();

        let mut total_freed = 0;
        for key in &expired_keys {
            if let Some(entry) = entries.remove(key) {
                total_freed += entry.envelope.size();
            }
        }

        let purged = initial_count - entries.len();

        if self.config.track_stats && purged > 0 {
            let mut stats = self.stats.write().unwrap_or_else(|e| e.into_inner());
            stats.expirations += purged as u64;
            stats.entry_count = entries.len();
            stats.total_bytes = stats.total_bytes.saturating_sub(total_freed);
        }

        purged
    }
}

impl Default for ProofCache {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Cache key builder for consistent key generation
pub struct CacheKeyBuilder {
    parts: Vec<String>,
}

impl CacheKeyBuilder {
    /// Create a new key builder
    pub fn new(proof_type: ProofType) -> Self {
        Self {
            parts: vec![format!("{:?}", proof_type)],
        }
    }

    /// Add a string component
    pub fn with_str(mut self, s: &str) -> Self {
        self.parts.push(s.to_string());
        self
    }

    /// Add a numeric component
    pub fn with_num<N: std::fmt::Display>(mut self, n: N) -> Self {
        self.parts.push(n.to_string());
        self
    }

    /// Add bytes (as hex)
    pub fn with_bytes(mut self, bytes: &[u8]) -> Self {
        let hex: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
        self.parts.push(hex);
        self
    }

    /// Build the final key
    pub fn build(self) -> String {
        self.parts.join(":")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proofs::{RangeProof, ProofConfig, SecurityLevel};

    fn test_config() -> ProofConfig {
        ProofConfig {
            security_level: SecurityLevel::Standard96,
            parallel: false,
            max_proof_size: 0,
        }
    }

    fn make_envelope() -> ProofEnvelope {
        let proof = RangeProof::generate(50, 0, 100, test_config()).unwrap();
        ProofEnvelope::from_range_proof(&proof).unwrap()
    }

    #[test]
    fn test_cache_put_get() {
        let cache = ProofCache::with_defaults();

        let envelope = make_envelope();
        cache.put("test_key", envelope.clone());

        let retrieved = cache.get("test_key");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().proof_type, envelope.proof_type);
    }

    #[test]
    fn test_cache_miss() {
        let cache = ProofCache::with_defaults();

        let result = cache.get("nonexistent");
        assert!(result.is_none());

        let stats = cache.stats();
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_cache_eviction() {
        let config = CacheConfig {
            max_entries: 2,
            ttl_seconds: 0,
            track_stats: true,
        };
        let cache = ProofCache::new(config);

        // Fill cache
        cache.put("key1", make_envelope());
        cache.put("key2", make_envelope());

        // Access key1 to make it more recent
        cache.get("key1");

        // Add third entry, should evict key2 (LRU)
        cache.put("key3", make_envelope());

        assert!(cache.contains("key1"));
        assert!(!cache.contains("key2"));
        assert!(cache.contains("key3"));

        let stats = cache.stats();
        assert_eq!(stats.evictions, 1);
    }

    #[test]
    fn test_cache_stats() {
        let cache = ProofCache::with_defaults();

        cache.put("key1", make_envelope());
        cache.get("key1"); // Hit
        cache.get("key1"); // Hit
        cache.get("missing"); // Miss

        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_cache_clear() {
        let cache = ProofCache::with_defaults();

        cache.put("key1", make_envelope());
        cache.put("key2", make_envelope());
        assert_eq!(cache.len(), 2);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_key_builder() {
        let key = CacheKeyBuilder::new(ProofType::Range)
            .with_str("test")
            .with_num(42)
            .with_bytes(&[0xab, 0xcd])
            .build();

        assert_eq!(key, "Range:test:42:abcd");
    }

    #[test]
    fn test_cache_remove() {
        let cache = ProofCache::with_defaults();

        cache.put("key1", make_envelope());
        assert!(cache.contains("key1"));

        let removed = cache.remove("key1");
        assert!(removed.is_some());
        assert!(!cache.contains("key1"));
    }
}
