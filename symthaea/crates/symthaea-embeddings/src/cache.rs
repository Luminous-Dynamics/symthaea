//! Persistent embedding cache using redb (pure Rust KV store).
//!
//! Survives process restarts. Auto-invalidates when model name or
//! embedding dimension changes. Uses blake3 content hashing for keys.

use anyhow::Result;
use redb::{Database, ReadableTable, TableDefinition};
use std::path::PathBuf;

const EMBEDDINGS_TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("embeddings");
const METADATA_TABLE: TableDefinition<&str, &str> = TableDefinition::new("metadata");

/// Configuration for the persistent cache.
pub struct PersistentCacheConfig {
    /// Path to the database file. Defaults to `~/.cache/symthaea/embeddings.redb`.
    pub path: Option<PathBuf>,
    /// Model name used to validate cache coherence.
    pub model_name: String,
    /// Embedding dimension used to validate cache coherence.
    pub dimension: usize,
}

/// On-disk embedding cache backed by redb.
pub struct PersistentCache {
    db: Database,
    model_name: String,
    dimension: usize,
}

impl PersistentCache {
    /// Open (or create) the persistent cache.
    ///
    /// If the stored metadata (model name, dimension) doesn't match the
    /// provided config, the cache is cleared to avoid serving stale vectors.
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

        // Validate metadata — clear if model/dim changed
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

                    stored_model.as_deref() != Some(&config.model_name)
                        || stored_dim.as_deref() != Some(&config.dimension.to_string())
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
                // Ensure embeddings table exists
                drop(meta);
                let _ = write_txn.open_table(EMBEDDINGS_TABLE)?;
            }
            write_txn.commit()?;
        }

        Ok(Self {
            db,
            model_name: config.model_name,
            dimension: config.dimension,
        })
    }

    /// Look up a cached embedding by text.
    pub fn get(&self, text: &str) -> Result<Option<Vec<f32>>> {
        let key = Self::hash_key(text);
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(EMBEDDINGS_TABLE)?;

        match table.get(key.as_slice())? {
            Some(value) => {
                let bytes = value.value();
                let floats = Self::bytes_to_f32(bytes);
                Ok(Some(floats))
            }
            None => Ok(None),
        }
    }

    /// Store an embedding for the given text.
    pub fn put(&self, text: &str, embedding: &[f32]) -> Result<()> {
        let key = Self::hash_key(text);
        let bytes = Self::f32_to_bytes(embedding);

        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(EMBEDDINGS_TABLE)?;
            table.insert(key.as_slice(), bytes.as_slice())?;
        }
        write_txn.commit()?;
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

    /// Compute blake3 hash of the text for use as a cache key.
    fn hash_key(text: &str) -> Vec<u8> {
        let hash = blake3::hash(text.as_bytes());
        hash.as_bytes().to_vec()
    }

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
                let arr: [u8; 4] = chunk.try_into().unwrap();
                f32::from_le_bytes(arr)
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
        let cache = PersistentCache::open(PersistentCacheConfig {
            path: Some(path.clone()),
            model_name: "test-model".into(),
            dimension: 4,
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
        let cache = PersistentCache::open(PersistentCacheConfig {
            path: Some(path.clone()),
            model_name: "test-model".into(),
            dimension: 4,
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
            let cache = PersistentCache::open(PersistentCacheConfig {
                path: Some(path.clone()),
                model_name: "model-a".into(),
                dimension: 4,
            })
            .unwrap();
            cache.put("key1", &[1.0, 2.0, 3.0, 4.0]).unwrap();
            assert_eq!(cache.len().unwrap(), 1);
        }

        // Reopen with different model → cache should be cleared
        {
            let cache = PersistentCache::open(PersistentCacheConfig {
                path: Some(path.clone()),
                model_name: "model-b".into(),
                dimension: 4,
            })
            .unwrap();
            assert_eq!(cache.len().unwrap(), 0, "Cache should be cleared on model change");
            assert!(cache.get("key1").unwrap().is_none());
        }

        cleanup(&path);
    }

    #[test]
    fn test_persistent_cache_integration() {
        let path = temp_db_path();

        // Embed and cache
        {
            let cache = PersistentCache::open(PersistentCacheConfig {
                path: Some(path.clone()),
                model_name: "qwen3-sim".into(),
                dimension: 1024,
            })
            .unwrap();
            let embedding: Vec<f32> = (0..1024).map(|i| i as f32 * 0.001).collect();
            cache.put("test sentence", &embedding).unwrap();
        }

        // New process / new cache handle — should find the embedding
        {
            let cache = PersistentCache::open(PersistentCacheConfig {
                path: Some(path.clone()),
                model_name: "qwen3-sim".into(),
                dimension: 1024,
            })
            .unwrap();
            let retrieved = cache.get("test sentence").unwrap();
            assert!(retrieved.is_some(), "Should find embedding from previous session");
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
        let cache = PersistentCache::open(PersistentCacheConfig {
            path: Some(path.clone()),
            model_name: "test".into(),
            dimension: 4,
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
}
