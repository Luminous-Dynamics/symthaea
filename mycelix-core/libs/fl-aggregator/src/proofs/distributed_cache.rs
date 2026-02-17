//! Distributed Proof Cache
//!
//! High-performance caching layer for proofs with support for distributed deployments.
//!
//! ## Features
//!
//! - Multi-tier caching (L1 memory, L2 distributed)
//! - TTL-based expiration
//! - LRU eviction policy
//! - Cache warming and preloading
//! - Statistics and monitoring
//!
//! ## Usage
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::distributed_cache::{
//!     DistributedCache, CacheConfig, CacheTier,
//! };
//!
//! // Create cache with configuration
//! let cache = DistributedCache::new(CacheConfig::default());
//!
//! // Store proof
//! cache.put("proof-123", proof_bytes, Duration::from_secs(3600)).await?;
//!
//! // Retrieve proof
//! if let Some(bytes) = cache.get("proof-123").await? {
//!     println!("Cache hit!");
//! }
//!
//! // Get statistics
//! let stats = cache.stats().await;
//! println!("Hit rate: {:.2}%", stats.hit_rate() * 100.0);
//! ```

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::proofs::{ProofError, ProofResult, ProofType};

// ============================================================================
// Configuration
// ============================================================================

/// Cache configuration
#[derive(Debug, Clone)]
pub struct DistributedCacheConfig {
    /// L1 (memory) cache size in entries
    pub l1_max_entries: usize,
    /// L1 max memory in bytes
    pub l1_max_memory_bytes: usize,
    /// L2 (distributed) enabled
    pub l2_enabled: bool,
    /// L2 connection string (e.g., redis://localhost:6379)
    pub l2_connection: Option<String>,
    /// Default TTL for entries
    pub default_ttl_secs: u64,
    /// Maximum TTL
    pub max_ttl_secs: u64,
    /// Enable compression
    pub compression: bool,
    /// Compression threshold (bytes)
    pub compression_threshold: usize,
    /// Key prefix for namespacing
    pub key_prefix: String,
    /// Enable async writes
    pub async_writes: bool,
}

impl Default for DistributedCacheConfig {
    fn default() -> Self {
        Self {
            l1_max_entries: 10_000,
            l1_max_memory_bytes: 100 * 1024 * 1024, // 100MB
            l2_enabled: false,
            l2_connection: None,
            default_ttl_secs: 3600,
            max_ttl_secs: 86400,
            compression: true,
            compression_threshold: 1024,
            key_prefix: "proof:".to_string(),
            async_writes: false,
        }
    }
}

// ============================================================================
// Cache Entry
// ============================================================================

/// Cached entry
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct CacheEntry {
    /// Cached data
    data: Vec<u8>,
    /// Creation time
    created_at: Instant,
    /// Expiration time
    expires_at: Instant,
    /// Access count
    access_count: u64,
    /// Last access time
    last_accessed: Instant,
    /// Size in bytes
    size_bytes: usize,
    /// Metadata
    metadata: EntryMetadata,
}

impl CacheEntry {
    fn new(data: Vec<u8>, ttl: Duration) -> Self {
        let now = Instant::now();
        let size = data.len();
        Self {
            data,
            created_at: now,
            expires_at: now + ttl,
            access_count: 0,
            last_accessed: now,
            size_bytes: size,
            metadata: EntryMetadata::default(),
        }
    }

    fn is_expired(&self) -> bool {
        Instant::now() >= self.expires_at
    }

    fn touch(&mut self) {
        self.access_count += 1;
        self.last_accessed = Instant::now();
    }

    #[allow(dead_code)]
    fn remaining_ttl(&self) -> Duration {
        let now = Instant::now();
        if now >= self.expires_at {
            Duration::ZERO
        } else {
            self.expires_at - now
        }
    }
}

/// Entry metadata
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
struct EntryMetadata {
    proof_type: Option<ProofType>,
    compressed: bool,
    original_size: Option<usize>,
}

// ============================================================================
// LRU Cache (L1)
// ============================================================================

/// LRU cache implementation
struct LruCache {
    entries: HashMap<String, CacheEntry>,
    order: VecDeque<String>,
    max_entries: usize,
    max_memory: usize,
    current_memory: usize,
}

impl LruCache {
    fn new(max_entries: usize, max_memory: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            max_entries,
            max_memory,
            current_memory: 0,
        }
    }

    fn get(&mut self, key: &str) -> Option<Vec<u8>> {
        // First check if entry exists and is expired (separate scope to avoid borrow conflict)
        let is_expired = self.entries.get(key).map(|e| e.is_expired()).unwrap_or(false);

        if is_expired {
            self.remove(key);
            return None;
        }

        if let Some(entry) = self.entries.get_mut(key) {
            entry.touch();
            // Move to back (most recently used)
            self.order.retain(|k| k != key);
            self.order.push_back(key.to_string());
            Some(entry.data.clone())
        } else {
            None
        }
    }

    fn put(&mut self, key: String, data: Vec<u8>, ttl: Duration) {
        let entry = CacheEntry::new(data, ttl);
        let size = entry.size_bytes;

        // Remove existing entry if present
        if let Some(old) = self.entries.remove(&key) {
            self.current_memory -= old.size_bytes;
            self.order.retain(|k| k != &key);
        }

        // Evict if necessary
        while self.entries.len() >= self.max_entries
            || self.current_memory + size > self.max_memory
        {
            if let Some(evict_key) = self.order.pop_front() {
                if let Some(evicted) = self.entries.remove(&evict_key) {
                    self.current_memory -= evicted.size_bytes;
                }
            } else {
                break;
            }
        }

        self.current_memory += size;
        self.entries.insert(key.clone(), entry);
        self.order.push_back(key);
    }

    fn remove(&mut self, key: &str) -> bool {
        if let Some(entry) = self.entries.remove(key) {
            self.current_memory -= entry.size_bytes;
            self.order.retain(|k| k != key);
            true
        } else {
            false
        }
    }

    fn contains(&self, key: &str) -> bool {
        self.entries
            .get(key)
            .map(|e| !e.is_expired())
            .unwrap_or(false)
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn memory_used(&self) -> usize {
        self.current_memory
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.current_memory = 0;
    }

    fn cleanup_expired(&mut self) -> usize {
        let expired: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, v)| v.is_expired())
            .map(|(k, _)| k.clone())
            .collect();

        let count = expired.len();
        for key in expired {
            self.remove(&key);
        }
        count
    }
}

// ============================================================================
// Distributed Cache
// ============================================================================

/// Distributed proof cache
pub struct DistributedCache {
    config: DistributedCacheConfig,
    l1: Arc<RwLock<LruCache>>,
    stats: Arc<RwLock<CacheStats>>,
}

impl DistributedCache {
    /// Create a new distributed cache
    pub fn new(config: DistributedCacheConfig) -> Self {
        let l1 = LruCache::new(config.l1_max_entries, config.l1_max_memory_bytes);

        Self {
            config,
            l1: Arc::new(RwLock::new(l1)),
            stats: Arc::new(RwLock::new(CacheStats::default())),
        }
    }

    /// Get an entry from cache
    pub async fn get(&self, key: &str) -> ProofResult<Option<Vec<u8>>> {
        let full_key = self.make_key(key);

        // Try L1 first
        {
            let mut l1 = self.l1.write().await;
            if let Some(data) = l1.get(&full_key) {
                self.record_hit(CacheTier::L1).await;
                return Ok(Some(self.decompress(&data)?));
            }
        }

        // L2 would go here in a full implementation
        // For now, just record miss
        self.record_miss().await;
        Ok(None)
    }

    /// Put an entry into cache
    pub async fn put(&self, key: &str, data: Vec<u8>, ttl: Duration) -> ProofResult<()> {
        self.put_with_metadata(key, data, ttl, None).await
    }

    /// Put with metadata
    pub async fn put_with_metadata(
        &self,
        key: &str,
        data: Vec<u8>,
        ttl: Duration,
        _proof_type: Option<ProofType>,
    ) -> ProofResult<()> {
        let full_key = self.make_key(key);
        let ttl = ttl.min(Duration::from_secs(self.config.max_ttl_secs));

        // Add compression marker and optionally compress
        let stored_data = if self.config.compression && data.len() > self.config.compression_threshold
        {
            self.compress(&data)?
        } else {
            // Add uncompressed marker (0) before data
            let mut result = vec![0u8];
            result.extend_from_slice(&data);
            result
        };

        // Store in L1
        {
            let mut l1 = self.l1.write().await;
            l1.put(full_key.clone(), stored_data, ttl);
        }

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.writes += 1;
        }

        Ok(())
    }

    /// Remove an entry
    pub async fn remove(&self, key: &str) -> ProofResult<bool> {
        let full_key = self.make_key(key);

        let removed = {
            let mut l1 = self.l1.write().await;
            l1.remove(&full_key)
        };

        if removed {
            let mut stats = self.stats.write().await;
            stats.evictions += 1;
        }

        Ok(removed)
    }

    /// Check if key exists
    pub async fn contains(&self, key: &str) -> bool {
        let full_key = self.make_key(key);
        let l1 = self.l1.read().await;
        l1.contains(&full_key)
    }

    /// Get or compute
    pub async fn get_or_insert<F, Fut>(
        &self,
        key: &str,
        ttl: Duration,
        compute: F,
    ) -> ProofResult<Vec<u8>>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = ProofResult<Vec<u8>>>,
    {
        // Try to get existing
        if let Some(data) = self.get(key).await? {
            return Ok(data);
        }

        // Compute and store
        let data = compute().await?;
        self.put(key, data.clone(), ttl).await?;
        Ok(data)
    }

    /// Clear all entries
    pub async fn clear(&self) {
        let mut l1 = self.l1.write().await;
        l1.clear();
    }

    /// Cleanup expired entries
    pub async fn cleanup(&self) -> usize {
        let mut l1 = self.l1.write().await;
        l1.cleanup_expired()
    }

    /// Get cache statistics
    pub async fn stats(&self) -> CacheStats {
        let l1 = self.l1.read().await;
        let mut stats = self.stats.read().await.clone();
        stats.l1_entries = l1.len();
        stats.l1_memory_bytes = l1.memory_used();
        stats
    }

    /// Get multiple entries
    pub async fn get_many(&self, keys: &[&str]) -> HashMap<String, Vec<u8>> {
        let mut results = HashMap::new();
        for key in keys {
            if let Ok(Some(data)) = self.get(key).await {
                results.insert(key.to_string(), data);
            }
        }
        results
    }

    /// Put multiple entries
    pub async fn put_many(&self, entries: &[(&str, Vec<u8>)], ttl: Duration) -> ProofResult<()> {
        for (key, data) in entries {
            self.put(key, data.clone(), ttl).await?;
        }
        Ok(())
    }

    // ========================================================================
    // Private helpers
    // ========================================================================

    fn make_key(&self, key: &str) -> String {
        format!("{}{}", self.config.key_prefix, key)
    }

    fn compress(&self, data: &[u8]) -> ProofResult<Vec<u8>> {
        #[cfg(feature = "proofs-compressed")]
        {
            // Use real zstd compression
            let compressed = zstd::encode_all(data, self.config.compression_level.unwrap_or(3))
                .map_err(|e| ProofError::SerializationError(format!("Compression failed: {}", e)))?;

            // Format: [1 byte marker][4 bytes original len][compressed data]
            let mut result = vec![1u8];
            result.extend_from_slice(&(data.len() as u32).to_le_bytes());
            result.extend_from_slice(&compressed);
            Ok(result)
        }

        #[cfg(not(feature = "proofs-compressed"))]
        {
            // Fallback: just add marker and store uncompressed
            let mut result = vec![1u8];
            result.extend_from_slice(&(data.len() as u32).to_le_bytes());
            result.extend_from_slice(data);
            Ok(result)
        }
    }

    fn decompress(&self, data: &[u8]) -> ProofResult<Vec<u8>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }

        match data[0] {
            0 => {
                // Not compressed - skip the marker byte
                Ok(data[1..].to_vec())
            }
            1 => {
                // Compressed
                if data.len() < 5 {
                    return Err(ProofError::SerializationError("Invalid compressed data".to_string()));
                }
                let original_len = u32::from_le_bytes(data[1..5].try_into().unwrap()) as usize;
                let compressed_data = &data[5..];

                #[cfg(feature = "proofs-compressed")]
                {
                    // Use real zstd decompression
                    let decompressed = zstd::decode_all(compressed_data)
                        .map_err(|e| ProofError::SerializationError(format!("Decompression failed: {}", e)))?;

                    if decompressed.len() != original_len {
                        return Err(ProofError::SerializationError(
                            format!("Decompressed size mismatch: expected {}, got {}", original_len, decompressed.len())
                        ));
                    }
                    Ok(decompressed)
                }

                #[cfg(not(feature = "proofs-compressed"))]
                {
                    // Fallback: data is stored uncompressed after header
                    let _ = original_len;
                    Ok(compressed_data.to_vec())
                }
            }
            _ => {
                // Unknown marker - treat as raw data for backwards compatibility
                Ok(data.to_vec())
            }
        }
    }

    async fn record_hit(&self, tier: CacheTier) {
        let mut stats = self.stats.write().await;
        stats.hits += 1;
        match tier {
            CacheTier::L1 => stats.l1_hits += 1,
            CacheTier::L2 => stats.l2_hits += 1,
        }
    }

    async fn record_miss(&self) {
        let mut stats = self.stats.write().await;
        stats.misses += 1;
    }
}

/// Cache tier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheTier {
    L1,
    L2,
}

/// Cache statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub writes: u64,
    pub evictions: u64,
    pub l1_hits: u64,
    pub l2_hits: u64,
    pub l1_entries: usize,
    pub l1_memory_bytes: usize,
    pub l2_entries: usize,
}

impl CacheStats {
    /// Calculate hit rate
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Calculate L1 hit rate
    pub fn l1_hit_rate(&self) -> f64 {
        if self.hits == 0 {
            0.0
        } else {
            self.l1_hits as f64 / self.hits as f64
        }
    }
}

// ============================================================================
// Cache Key Builder
// ============================================================================

/// Helper for building cache keys
pub struct CacheKey {
    parts: Vec<String>,
}

impl CacheKey {
    pub fn new() -> Self {
        Self { parts: Vec::new() }
    }

    pub fn with_type(mut self, proof_type: ProofType) -> Self {
        self.parts.push(format!("{:?}", proof_type).to_lowercase());
        self
    }

    pub fn with_part(mut self, part: &str) -> Self {
        self.parts.push(part.to_string());
        self
    }

    pub fn with_hash(mut self, data: &[u8]) -> Self {
        let hash = blake3::hash(data);
        self.parts.push(hex::encode(&hash.as_bytes()[..16]));
        self
    }

    pub fn build(self) -> String {
        self.parts.join(":")
    }
}

impl Default for CacheKey {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_config_default() {
        let config = DistributedCacheConfig::default();
        assert_eq!(config.l1_max_entries, 10_000);
        assert!(config.compression);
    }

    #[test]
    fn test_cache_entry_expiration() {
        let entry = CacheEntry::new(vec![1, 2, 3], Duration::from_millis(1));
        std::thread::sleep(Duration::from_millis(10));
        assert!(entry.is_expired());
    }

    #[test]
    fn test_lru_cache_basic() {
        let mut cache = LruCache::new(10, 1024 * 1024);

        cache.put("key1".to_string(), vec![1, 2, 3], Duration::from_secs(60));
        cache.put("key2".to_string(), vec![4, 5, 6], Duration::from_secs(60));

        assert!(cache.contains("key1"));
        assert!(cache.contains("key2"));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_lru_cache_eviction() {
        let mut cache = LruCache::new(2, 1024 * 1024);

        cache.put("key1".to_string(), vec![1], Duration::from_secs(60));
        cache.put("key2".to_string(), vec![2], Duration::from_secs(60));
        cache.put("key3".to_string(), vec![3], Duration::from_secs(60));

        // key1 should be evicted (LRU)
        assert!(!cache.contains("key1"));
        assert!(cache.contains("key2"));
        assert!(cache.contains("key3"));
    }

    #[test]
    fn test_lru_cache_access_order() {
        let mut cache = LruCache::new(2, 1024 * 1024);

        cache.put("key1".to_string(), vec![1], Duration::from_secs(60));
        cache.put("key2".to_string(), vec![2], Duration::from_secs(60));

        // Access key1, making key2 the LRU
        cache.get("key1");

        cache.put("key3".to_string(), vec![3], Duration::from_secs(60));

        // key2 should be evicted (LRU)
        assert!(cache.contains("key1"));
        assert!(!cache.contains("key2"));
        assert!(cache.contains("key3"));
    }

    #[tokio::test]
    async fn test_distributed_cache_basic() {
        let cache = DistributedCache::new(DistributedCacheConfig::default());

        cache
            .put("test", vec![1, 2, 3], Duration::from_secs(60))
            .await
            .unwrap();

        let result = cache.get("test").await.unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap(), vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn test_distributed_cache_miss() {
        let cache = DistributedCache::new(DistributedCacheConfig::default());

        let result = cache.get("nonexistent").await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_distributed_cache_remove() {
        let cache = DistributedCache::new(DistributedCacheConfig::default());

        cache
            .put("test", vec![1, 2, 3], Duration::from_secs(60))
            .await
            .unwrap();

        let removed = cache.remove("test").await.unwrap();
        assert!(removed);

        let result = cache.get("test").await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_distributed_cache_get_or_insert() {
        let cache = DistributedCache::new(DistributedCacheConfig::default());

        let result = cache
            .get_or_insert("computed", Duration::from_secs(60), || async {
                Ok(vec![42, 43, 44])
            })
            .await
            .unwrap();

        assert_eq!(result, vec![42, 43, 44]);

        // Second call should hit cache
        let result2 = cache
            .get_or_insert("computed", Duration::from_secs(60), || async {
                Ok(vec![0, 0, 0]) // Different value
            })
            .await
            .unwrap();

        assert_eq!(result2, vec![42, 43, 44]); // Should return cached value
    }

    #[tokio::test]
    async fn test_distributed_cache_stats() {
        let cache = DistributedCache::new(DistributedCacheConfig::default());

        cache
            .put("test", vec![1, 2, 3], Duration::from_secs(60))
            .await
            .unwrap();

        cache.get("test").await.unwrap(); // Hit
        cache.get("missing").await.unwrap(); // Miss

        let stats = cache.stats().await;
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.writes, 1);
    }

    #[tokio::test]
    async fn test_distributed_cache_get_many() {
        let cache = DistributedCache::new(DistributedCacheConfig::default());

        cache
            .put("key1", vec![1], Duration::from_secs(60))
            .await
            .unwrap();
        cache
            .put("key2", vec![2], Duration::from_secs(60))
            .await
            .unwrap();

        let results = cache.get_many(&["key1", "key2", "key3"]).await;

        assert_eq!(results.len(), 2);
        assert!(results.contains_key("key1"));
        assert!(results.contains_key("key2"));
    }

    #[test]
    fn test_cache_key_builder() {
        let key = CacheKey::new()
            .with_type(ProofType::Range)
            .with_part("test")
            .with_hash(&[1, 2, 3])
            .build();

        assert!(key.starts_with("range:test:"));
    }

    #[test]
    fn test_cache_stats_hit_rate() {
        let stats = CacheStats {
            hits: 80,
            misses: 20,
            ..Default::default()
        };

        assert!((stats.hit_rate() - 0.8).abs() < 0.001);
    }
}
