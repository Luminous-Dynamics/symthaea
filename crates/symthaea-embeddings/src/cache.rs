// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Persistent embedding cache using redb (pure Rust KV store).
//!
//! Survives process restarts. Auto-invalidates when model name,
//! embedding dimension, or storage format changes. Uses blake3
//! content hashing for keys.
//!
//! ## Quantized Storage (f16)
//!
//! When `use_f16_storage` is enabled, embeddings are stored as
//! half-precision (f16) floats, halving disk usage at the cost of
//! ~0.01 absolute precision loss. Useful for large caches where
//! disk space matters more than exact fidelity.

use anyhow::Result;
use redb::{Database, ReadableTableMetadata, TableDefinition};
use std::path::PathBuf;

const EMBEDDINGS_TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("embeddings");
const METADATA_TABLE: TableDefinition<&str, &str> = TableDefinition::new("metadata");

/// Cache hit/miss/put statistics.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub puts: u64,
}

impl CacheStats {
    /// Fraction of lookups that were hits (0.0–1.0).
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Configuration for the persistent cache.
pub struct PersistentCacheConfig {
    /// Path to the database file. Defaults to `~/.cache/symthaea/embeddings.redb`.
    pub path: Option<PathBuf>,
    /// Model name used to validate cache coherence.
    pub model_name: String,
    /// Embedding dimension used to validate cache coherence.
    pub dimension: usize,
    /// Store embeddings as f16 (half precision) to halve disk usage.
    pub use_f16_storage: bool,
}

/// On-disk embedding cache backed by redb.
pub struct PersistentCache {
    db: Database,
    path: PathBuf,
    model_name: String,
    dimension: usize,
    use_f16_storage: bool,
    stats: CacheStats,
}

impl PersistentCache {
    /// Open (or create) the persistent cache.
    ///
    /// If the stored metadata (model name, dimension, storage format)
    /// doesn't match the provided config, the cache is cleared to avoid
    /// serving stale or incompatible vectors.
    pub fn open(config: PersistentCacheConfig) -> Result<Self> {
        let path = match config.path {
            Some(p) => p,
            None => {
                let cache_dir = dirs::cache_dir()
                    .unwrap_or_else(|| PathBuf::from("/tmp"))
                    .join("symthaea");
                std::fs::create_dir_all(&cache_dir)?;
                cache_dir.join("embeddings.redb")
            }
        };

        let db = Database::create(&path)?;
        let format_str = if config.use_f16_storage { "f16" } else { "f32" };

        // Validate metadata — clear if model/dim/format changed
        let needs_clear = {
            let read_txn = db.begin_read()?;
            match read_txn.open_table(METADATA_TABLE) {
                Ok(table) => {
                    let stored_model = table
                        .get("model_name")
                        .ok()
                        .flatten()
                        .map(|v| v.value().to_string());
                    let stored_dim = table
                        .get("dimension")
                        .ok()
                        .flatten()
                        .map(|v| v.value().to_string());
                    let stored_format = table
                        .get("storage_format")
                        .ok()
                        .flatten()
                        .map(|v| v.value().to_string());

                    stored_model.as_deref() != Some(&config.model_name)
                        || stored_dim.as_deref() != Some(&config.dimension.to_string())
                        || stored_format.as_deref() != Some(format_str)
                }
                Err(_) => true, // Table doesn't exist yet
            }
        };

        if needs_clear {
            let write_txn = db.begin_write()?;
            {
                // Drop and recreate tables
                let _ = write_txn.delete_table(EMBEDDINGS_TABLE);
                let _ = write_txn.delete_table(METADATA_TABLE);
                let mut meta = write_txn.open_table(METADATA_TABLE)?;
                meta.insert("model_name", config.model_name.as_str())?;
                meta.insert("dimension", config.dimension.to_string().as_str())?;
                meta.insert("storage_format", format_str)?;
                // Ensure embeddings table exists
                drop(meta);
                let _ = write_txn.open_table(EMBEDDINGS_TABLE)?;
            }
            write_txn.commit()?;
        }

        Ok(Self {
            db,
            path,
            model_name: config.model_name,
            dimension: config.dimension,
            use_f16_storage: config.use_f16_storage,
            stats: CacheStats::default(),
        })
    }

    /// Look up a cached embedding by text.
    pub fn get(&mut self, text: &str) -> Result<Option<Vec<f32>>> {
        let key = Self::hash_key(text);
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(EMBEDDINGS_TABLE)?;

        match table.get(key.as_slice())? {
            Some(value) => {
                let bytes = value.value();
                let floats = if self.use_f16_storage {
                    Self::f16_bytes_to_f32(bytes)
                } else {
                    Self::bytes_to_f32(bytes)
                };
                self.stats.hits += 1;
                Ok(Some(floats))
            }
            None => {
                self.stats.misses += 1;
                Ok(None)
            }
        }
    }

    /// Store an embedding for the given text.
    pub fn put(&mut self, text: &str, embedding: &[f32]) -> Result<()> {
        let key = Self::hash_key(text);
        let bytes = if self.use_f16_storage {
            Self::f32_to_f16_bytes(embedding)
        } else {
            Self::f32_to_bytes(embedding)
        };

        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(EMBEDDINGS_TABLE)?;
            table.insert(key.as_slice(), bytes.as_slice())?;
        }
        write_txn.commit()?;
        self.stats.puts += 1;
        Ok(())
    }

    /// Count cached entries.
    pub fn len(&self) -> Result<usize> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(EMBEDDINGS_TABLE)?;
        Ok(table.len()? as usize)
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Model name this cache was opened with.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Embedding dimension this cache was opened with.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Whether this cache stores embeddings in f16 format.
    pub fn is_f16(&self) -> bool {
        self.use_f16_storage
    }

    /// Current hit/miss/put statistics.
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Approximate on-disk size in bytes.
    pub fn disk_size_bytes(&self) -> Result<u64> {
        Ok(std::fs::metadata(&self.path)?.len())
    }

    // ── Hashing ──────────────────────────────────────────────────────

    /// Compute blake3 hash of the text for use as a cache key.
    fn hash_key(text: &str) -> Vec<u8> {
        let hash = blake3::hash(text.as_bytes());
        hash.as_bytes().to_vec()
    }

    // ── Serialization (f32) ──────────────────────────────────────────

    /// Serialize f32 slice to bytes (little-endian).
    fn f32_to_bytes(floats: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(floats.len() * 4);
        for &f in floats {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        bytes
    }

    /// Deserialize bytes to f32 vec (little-endian).
    fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| {
                let arr: [u8; 4] = chunk
                    .try_into()
                    .expect("chunks_exact(4) guarantees 4-byte slices");
                f32::from_le_bytes(arr)
            })
            .collect()
    }

    // ── Serialization (f16) ──────────────────────────────────────────

    /// Convert f32 slice to f16 bytes (little-endian).
    fn f32_to_f16_bytes(floats: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(floats.len() * 2);
        for &f in floats {
            let h = half::f16::from_f32(f);
            bytes.extend_from_slice(&h.to_le_bytes());
        }
        bytes
    }

    /// Convert f16 bytes (little-endian) back to f32 vec.
    fn f16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(2)
            .map(|chunk| {
                let arr: [u8; 2] = chunk
                    .try_into()
                    .expect("chunks_exact(2) guarantees 2-byte slices");
                half::f16::from_le_bytes(arr).to_f32()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_db_path() -> PathBuf {
        let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let pid = std::process::id();
        std::env::temp_dir().join(format!("symthaea_cache_test_{pid}_{id}.redb"))
    }

    fn cleanup(path: &PathBuf) {
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_persistent_cache_roundtrip() {
        let path = temp_db_path();
        let mut cache = PersistentCache::open(PersistentCacheConfig {
            path: Some(path.clone()),
            model_name: "test-model".into(),
            dimension: 4,
            use_f16_storage: false,
        })
        .unwrap();

        let embedding = vec![1.0f32, 2.0, 3.0, 4.0];
        cache.put("hello", &embedding).unwrap();

        let retrieved = cache.get("hello").unwrap().unwrap();
        assert_eq!(retrieved, embedding);

        cleanup(&path);
    }

    #[test]
    fn test_persistent_cache_miss() {
        let path = temp_db_path();
        let mut cache = PersistentCache::open(PersistentCacheConfig {
            path: Some(path.clone()),
            model_name: "test-model".into(),
            dimension: 4,
            use_f16_storage: false,
        })
        .unwrap();

        let result = cache.get("nonexistent").unwrap();
        assert!(result.is_none());

        cleanup(&path);
    }

    #[test]
    fn test_persistent_cache_invalidation() {
        let path = temp_db_path();

        // First cache with model-a
        {
            let mut cache = PersistentCache::open(PersistentCacheConfig {
                path: Some(path.clone()),
                model_name: "model-a".into(),
                dimension: 4,
                use_f16_storage: false,
            })
            .unwrap();
            cache.put("key1", &[1.0, 2.0, 3.0, 4.0]).unwrap();
            assert_eq!(cache.len().unwrap(), 1);
        }

        // Reopen with different model → cache should be cleared
        {
            let mut cache = PersistentCache::open(PersistentCacheConfig {
                path: Some(path.clone()),
                model_name: "model-b".into(),
                dimension: 4,
                use_f16_storage: false,
            })
            .unwrap();
            assert_eq!(
                cache.len().unwrap(),
                0,
                "Cache should be cleared on model change"
            );
            assert!(cache.get("key1").unwrap().is_none());
        }

        cleanup(&path);
    }

    #[test]
    fn test_persistent_cache_integration() {
        let path = temp_db_path();

        // Embed and cache
        {
            let mut cache = PersistentCache::open(PersistentCacheConfig {
                path: Some(path.clone()),
                model_name: "qwen3-sim".into(),
                dimension: 1024,
                use_f16_storage: false,
            })
            .unwrap();
            let embedding: Vec<f32> = (0..1024).map(|i| i as f32 * 0.001).collect();
            cache.put("test sentence", &embedding).unwrap();
        }

        // New process / new cache handle — should find the embedding
        {
            let mut cache = PersistentCache::open(PersistentCacheConfig {
                path: Some(path.clone()),
                model_name: "qwen3-sim".into(),
                dimension: 1024,
                use_f16_storage: false,
            })
            .unwrap();
            let retrieved = cache.get("test sentence").unwrap();
            assert!(
                retrieved.is_some(),
                "Should find embedding from previous session"
            );
            let emb = retrieved.unwrap();
            assert_eq!(emb.len(), 1024);
            assert!((emb[0] - 0.0).abs() < 1e-6);
            assert!((emb[1] - 0.001).abs() < 1e-6);
        }

        cleanup(&path);
    }

    #[test]
    fn test_persistent_cache_len() {
        let path = temp_db_path();
        let mut cache = PersistentCache::open(PersistentCacheConfig {
            path: Some(path.clone()),
            model_name: "test".into(),
            dimension: 4,
            use_f16_storage: false,
        })
        .unwrap();

        assert_eq!(cache.len().unwrap(), 0);
        assert!(cache.is_empty().unwrap());

        cache.put("a", &[1.0, 2.0, 3.0, 4.0]).unwrap();
        cache.put("b", &[5.0, 6.0, 7.0, 8.0]).unwrap();
        assert_eq!(cache.len().unwrap(), 2);
        assert!(!cache.is_empty().unwrap());

        cleanup(&path);
    }

    // ── Stats tests ──────────────────────────────────────────────────

    #[test]
    fn test_persistent_cache_stats() {
        let path = temp_db_path();
        let mut cache = PersistentCache::open(PersistentCacheConfig {
            path: Some(path.clone()),
            model_name: "test".into(),
            dimension: 4,
            use_f16_storage: false,
        })
        .unwrap();

        assert_eq!(cache.stats().hits, 0);
        assert_eq!(cache.stats().misses, 0);
        assert_eq!(cache.stats().puts, 0);
        assert_eq!(cache.stats().hit_rate(), 0.0);

        // Miss
        let _ = cache.get("a").unwrap();
        assert_eq!(cache.stats().misses, 1);

        // Put
        cache.put("a", &[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(cache.stats().puts, 1);

        // Hit
        let _ = cache.get("a").unwrap();
        assert_eq!(cache.stats().hits, 1);

        assert!((cache.stats().hit_rate() - 0.5).abs() < 1e-6); // 1 hit / (1 hit + 1 miss)

        cleanup(&path);
    }

    #[test]
    fn test_persistent_cache_disk_size() {
        let path = temp_db_path();
        let mut cache = PersistentCache::open(PersistentCacheConfig {
            path: Some(path.clone()),
            model_name: "test".into(),
            dimension: 128,
            use_f16_storage: false,
        })
        .unwrap();

        let embedding: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
        cache.put("hello", &embedding).unwrap();

        let size = cache.disk_size_bytes().unwrap();
        assert!(size > 0, "Database file should have non-zero size");

        cleanup(&path);
    }

    // ── f16 tests ────────────────────────────────────────────────────

    #[test]
    fn test_f16_roundtrip() {
        let path = temp_db_path();
        let mut cache = PersistentCache::open(PersistentCacheConfig {
            path: Some(path.clone()),
            model_name: "test".into(),
            dimension: 4,
            use_f16_storage: true,
        })
        .unwrap();

        let embedding = vec![1.0f32, 0.5, -0.25, 3.14];
        cache.put("hello", &embedding).unwrap();

        let retrieved = cache.get("hello").unwrap().unwrap();
        assert_eq!(retrieved.len(), 4);
        for (orig, got) in embedding.iter().zip(retrieved.iter()) {
            assert!(
                (orig - got).abs() < 0.01,
                "f16 roundtrip: {orig} vs {got} exceeds tolerance"
            );
        }

        cleanup(&path);
    }

    #[test]
    fn test_f16_size_reduction() {
        // Verify that f16 serialization uses exactly half the bytes of f32.
        // (redb's page allocator doesn't scale linearly with value size,
        //  so we test the serialization layer directly.)
        let embedding: Vec<f32> = (0..1024).map(|i| i as f32 * 0.001).collect();

        let f32_bytes = PersistentCache::f32_to_bytes(&embedding);
        let f16_bytes = PersistentCache::f32_to_f16_bytes(&embedding);

        assert_eq!(
            f32_bytes.len(),
            1024 * 4,
            "f32 should be 4 bytes per element"
        );
        assert_eq!(
            f16_bytes.len(),
            1024 * 2,
            "f16 should be 2 bytes per element"
        );
        assert_eq!(
            f16_bytes.len() * 2,
            f32_bytes.len(),
            "f16 storage should be exactly half the size of f32"
        );

        // Verify roundtrip fidelity through f16
        let recovered = PersistentCache::f16_bytes_to_f32(&f16_bytes);
        assert_eq!(recovered.len(), 1024);
        for (orig, got) in embedding.iter().zip(recovered.iter()) {
            assert!(
                (orig - got).abs() < 0.01,
                "f16 roundtrip precision: {orig} vs {got}"
            );
        }
    }

    #[test]
    fn test_f16_format_mismatch_clears_cache() {
        let path = temp_db_path();

        // Open as f32 and insert
        {
            let mut cache = PersistentCache::open(PersistentCacheConfig {
                path: Some(path.clone()),
                model_name: "test".into(),
                dimension: 4,
                use_f16_storage: false,
            })
            .unwrap();
            cache.put("key1", &[1.0, 2.0, 3.0, 4.0]).unwrap();
            assert_eq!(cache.len().unwrap(), 1);
        }

        // Reopen as f16 → should clear
        {
            let cache = PersistentCache::open(PersistentCacheConfig {
                path: Some(path.clone()),
                model_name: "test".into(),
                dimension: 4,
                use_f16_storage: true,
            })
            .unwrap();
            assert_eq!(
                cache.len().unwrap(),
                0,
                "Cache should be cleared when storage format changes"
            );
        }

        cleanup(&path);
    }
}
