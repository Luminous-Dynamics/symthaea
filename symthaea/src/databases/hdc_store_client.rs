// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HdcStore-backed ConsciousnessDatabase implementation.
//!
//! Vectors are stored in the zero-copy mmap'd HdcStore for O(1) access.
//! Metadata (content, valence, arousal, psi, topics) is stored in an
//! in-memory HashMap keyed by the same u64 ID hash.
//!
//! This hybrid approach gives us:
//! - Zero-copy vector reads (no deserialization)
//! - SIMD-accelerated similarity search
//! - Persistent vector storage across restarts
//! - LSH-accelerated approximate nearest neighbor

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

use symthaea_core::hdc::BinaryHV;
use symthaea_hdc_store::{HdcStore, LshIndex, StoreConfig};

use super::{DatabaseError, DatabaseStats, DbResult, MemoryRecord, MemoryType, SearchResult};

/// Hash a string ID to u64 for HdcStore keying.
///
/// Uses BLAKE3 (deterministic across Rust versions) truncated to u64.
/// DefaultHasher uses SipHash which is NOT stable across Rust releases.
fn id_to_u64(id: &str) -> u64 {
    let hash = blake3::hash(id.as_bytes());
    let bytes: [u8; 8] = hash.as_bytes()[..8].try_into().unwrap();
    u64::from_le_bytes(bytes)
}

/// Metadata sidecar for a stored memory record (everything except the BinaryHV).
#[derive(Debug, Clone)]
struct RecordMetadata {
    id: String,
    memory_type: MemoryType,
    content: String,
    timestamp_ms: u64,
    valence: f32,
    arousal: f32,
    psi: f64,
    topics: Vec<String>,
    metadata: String,
    consolidation_strength: f64,
    retrieval_count: u32,
}

impl RecordMetadata {
    fn from_record(record: &MemoryRecord) -> Self {
        Self {
            id: record.id.clone(),
            memory_type: record.memory_type,
            content: record.content.clone(),
            timestamp_ms: record.timestamp_ms,
            valence: record.valence,
            arousal: record.arousal,
            psi: record.psi,
            topics: record.topics.clone(),
            metadata: record.metadata.clone(),
            consolidation_strength: record.consolidation_strength,
            retrieval_count: record.retrieval_count,
        }
    }

    fn to_record(&self, encoding: BinaryHV) -> MemoryRecord {
        MemoryRecord {
            id: self.id.clone(),
            memory_type: self.memory_type,
            encoding,
            content: self.content.clone(),
            timestamp_ms: self.timestamp_ms,
            valence: self.valence,
            arousal: self.arousal,
            psi: self.psi,
            topics: self.topics.clone(),
            metadata: self.metadata.clone(),
            consolidation_strength: self.consolidation_strength,
            retrieval_count: self.retrieval_count,
        }
    }
}

/// HdcStore-backed consciousness database.
///
/// Uses mmap'd BinaryHV storage with LSH indexing for fast similarity search.
pub struct HdcStoreDatabase {
    /// Vector storage (mmap'd, zero-copy reads).
    store: RwLock<HdcStore>,
    /// LSH index for approximate nearest neighbor.
    lsh: RwLock<LshIndex>,
    /// Metadata sidecar: id_hash -> metadata.
    metadata: RwLock<HashMap<u64, RecordMetadata>>,
    /// Reverse map: string ID -> u64 hash.
    id_map: RwLock<HashMap<String, u64>>,
    /// Store file path.
    path: PathBuf,
    /// Query counter for stats.
    total_queries: AtomicU64,
}

impl HdcStoreDatabase {
    /// Create a new HdcStore database at the given path.
    pub fn new(path: impl AsRef<Path>) -> DbResult<Self> {
        let path = path.as_ref().to_path_buf();
        let store_path = path.with_extension("hdc");

        let store = if store_path.exists() {
            HdcStore::open(&store_path)
                .map_err(|e| DatabaseError::ConnectionFailed(e.to_string()))?
        } else {
            HdcStore::create(&store_path, StoreConfig::default())
                .map_err(|e| DatabaseError::ConnectionFailed(e.to_string()))?
        };

        let lsh = LshIndex::new(10, 32, 42);

        Ok(Self {
            store: RwLock::new(store),
            lsh: RwLock::new(lsh),
            metadata: RwLock::new(HashMap::new()),
            id_map: RwLock::new(HashMap::new()),
            path: store_path,
            total_queries: AtomicU64::new(0),
        })
    }

    /// Create an in-memory HdcStore database (for testing).
    pub fn in_memory() -> DbResult<Self> {
        let tmp = std::env::temp_dir().join(format!(
            "hdc_store_{}_{}.hdc",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        Self::new(&tmp)
    }
}

#[async_trait::async_trait]
impl super::ConsciousnessDatabase for HdcStoreDatabase {
    async fn store(&self, record: MemoryRecord) -> DbResult<()> {
        let id_hash = id_to_u64(&record.id);
        let meta = RecordMetadata::from_record(&record);
        let hv = record.encoding;
        let id_string = record.id;

        {
            let mut store = self
                .store
                .write()
                .map_err(|e| DatabaseError::Other(e.to_string()))?;
            store
                .append(id_hash, &hv)
                .map_err(|e| DatabaseError::InsertFailed(e.to_string()))?;
        }
        {
            let mut lsh = self
                .lsh
                .write()
                .map_err(|e| DatabaseError::Other(e.to_string()))?;
            lsh.insert(id_hash, &hv);
        }
        {
            let mut metadata = self
                .metadata
                .write()
                .map_err(|e| DatabaseError::Other(e.to_string()))?;
            metadata.insert(id_hash, meta);
        }
        {
            let mut id_map = self
                .id_map
                .write()
                .map_err(|e| DatabaseError::Other(e.to_string()))?;
            id_map.insert(id_string, id_hash);
        }

        Ok(())
    }

    async fn search_similar(&self, query: &BinaryHV, top_k: usize) -> DbResult<Vec<SearchResult>> {
        self.total_queries.fetch_add(1, Ordering::Relaxed);

        let store = self
            .store
            .read()
            .map_err(|e| DatabaseError::Other(e.to_string()))?;
        let metadata = self
            .metadata
            .read()
            .map_err(|e| DatabaseError::Other(e.to_string()))?;

        // Use LSH candidates first, then verify with actual similarity
        let candidates = {
            let lsh = self
                .lsh
                .read()
                .map_err(|e| DatabaseError::Other(e.to_string()))?;
            lsh.query_candidates(query)
        };

        let mut results: Vec<(u64, f32)> = if candidates.len() >= top_k {
            // LSH provided enough candidates -- verify similarity on just those
            candidates
                .iter()
                .filter_map(|&id| {
                    let hv = store.get(id)?;
                    Some((id, query.similarity(hv)))
                })
                .collect()
        } else {
            // Not enough LSH candidates -- fall back to brute-force scan
            store.scan_similar(query, top_k)
        };

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        let search_results = results
            .into_iter()
            .filter_map(|(id_hash, similarity)| {
                let meta = metadata.get(&id_hash)?;
                let hv = store.get(id_hash)?;
                Some(SearchResult {
                    record: meta.to_record(*hv),
                    similarity,
                })
            })
            .collect();

        Ok(search_results)
    }

    async fn search_similar_filtered(
        &self,
        query: &BinaryHV,
        top_k: usize,
        _filter: Option<&str>,
    ) -> DbResult<Vec<SearchResult>> {
        // Filter not supported for HdcStore -- fall back to unfiltered
        self.search_similar(query, top_k).await
    }

    async fn get(&self, id: &str) -> DbResult<Option<MemoryRecord>> {
        let id_hash = id_to_u64(id);
        let store = self
            .store
            .read()
            .map_err(|e| DatabaseError::Other(e.to_string()))?;
        let metadata = self
            .metadata
            .read()
            .map_err(|e| DatabaseError::Other(e.to_string()))?;

        match (store.get(id_hash), metadata.get(&id_hash)) {
            (Some(hv), Some(meta)) => Ok(Some(meta.to_record(*hv))),
            _ => Ok(None),
        }
    }

    async fn delete(&self, id: &str) -> DbResult<bool> {
        let id_hash = id_to_u64(id);

        // Get the HV copy for LSH removal before mutating the store
        let hv_copy = {
            let store = self
                .store
                .read()
                .map_err(|e| DatabaseError::Other(e.to_string()))?;
            store.get(id_hash).copied()
        };

        // Remove from LSH index
        if let Some(hv) = &hv_copy {
            let mut lsh = self
                .lsh
                .write()
                .map_err(|e| DatabaseError::Other(e.to_string()))?;
            lsh.remove(id_hash, hv);
        }

        // Delete from store
        let deleted = {
            let mut store = self
                .store
                .write()
                .map_err(|e| DatabaseError::Other(e.to_string()))?;
            store.delete(id_hash)
        };

        if deleted {
            let mut metadata = self
                .metadata
                .write()
                .map_err(|e| DatabaseError::Other(e.to_string()))?;
            metadata.remove(&id_hash);
            drop(metadata);

            let mut id_map = self
                .id_map
                .write()
                .map_err(|e| DatabaseError::Other(e.to_string()))?;
            id_map.remove(id);
        }

        // Auto-compact if needed
        {
            let mut store = self
                .store
                .write()
                .map_err(|e| DatabaseError::Other(e.to_string()))?;
            if store.needs_compaction() {
                let _ = store.compact();
            }
        }

        Ok(deleted)
    }

    async fn count(&self) -> DbResult<usize> {
        let store = self
            .store
            .read()
            .map_err(|e| DatabaseError::Other(e.to_string()))?;
        Ok(store.live_count() as usize)
    }

    async fn health_check(&self) -> DbResult<bool> {
        // Healthy if we can acquire the lock and read the count
        let _store = self
            .store
            .read()
            .map_err(|e| DatabaseError::Other(e.to_string()))?;
        Ok(true)
    }

    async fn stats(&self) -> DbResult<DatabaseStats> {
        let store = self
            .store
            .read()
            .map_err(|e| DatabaseError::Other(e.to_string()))?;
        let metadata = self
            .metadata
            .read()
            .map_err(|e| DatabaseError::Other(e.to_string()))?;
        let total_queries = self.total_queries.load(Ordering::Relaxed);

        // Count by memory type
        let mut type_counts: HashMap<String, usize> = HashMap::new();
        let mut psi_sum = 0.0;
        let mut oldest = u64::MAX;
        let mut newest = 0u64;
        for meta in metadata.values() {
            *type_counts
                .entry(format!("{:?}", meta.memory_type))
                .or_default() += 1;
            psi_sum += meta.psi;
            oldest = oldest.min(meta.timestamp_ms);
            newest = newest.max(meta.timestamp_ms);
        }

        let count = store.live_count() as usize;
        let file_size = std::fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0);

        Ok(DatabaseStats {
            total_records: count,
            database_size_bytes: file_size,
            page_count: 0,
            page_size: 2080,
            freelist_count: store.tombstone_count(),
            cache_hit_ratio: 1.0, // mmap is always "cached"
            cache_hits: total_queries,
            cache_misses: 0,
            avg_query_latency_us: 0,
            total_queries,
            memory_type_counts: type_counts.into_iter().collect(),
            avg_psi: if count > 0 {
                psi_sum / count as f64
            } else {
                0.0
            },
            oldest_timestamp_ms: if oldest == u64::MAX { 0 } else { oldest },
            newest_timestamp_ms: newest,
            backend_status: format!(
                "HdcStore: {} live, {} tombstones, {:.1}KB",
                store.live_count(),
                store.tombstone_count(),
                file_size as f64 / 1024.0
            ),
        })
    }

    async fn list_all(&self) -> DbResult<Vec<MemoryRecord>> {
        let store = self
            .store
            .read()
            .map_err(|e| DatabaseError::Other(e.to_string()))?;
        let metadata = self
            .metadata
            .read()
            .map_err(|e| DatabaseError::Other(e.to_string()))?;

        let records: Vec<MemoryRecord> = store
            .iter_live()
            .filter_map(|(id_hash, hv)| {
                let meta = metadata.get(&id_hash)?;
                Some(meta.to_record(*hv))
            })
            .collect();

        Ok(records)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(id: &str, seed: u64) -> MemoryRecord {
        MemoryRecord {
            id: id.to_string(),
            memory_type: MemoryType::Semantic,
            encoding: BinaryHV::random(seed),
            content: format!("test content {id}"),
            timestamp_ms: 1000 + seed,
            valence: 0.5,
            arousal: 0.3,
            psi: 0.7,
            topics: vec!["test".to_string()],
            metadata: "{}".to_string(),
            consolidation_strength: 1.0,
            retrieval_count: 0,
        }
    }

    #[tokio::test]
    async fn store_and_retrieve() {
        let db = HdcStoreDatabase::in_memory().unwrap();
        let record = make_record("rec1", 42);
        db.store(record).await.unwrap();

        let retrieved = db.get("rec1").await.unwrap().unwrap();
        assert_eq!(retrieved.id, "rec1");
        assert_eq!(retrieved.content, "test content rec1");
        assert_eq!(retrieved.encoding.similarity(&BinaryHV::random(42)), 1.0);
    }

    #[tokio::test]
    async fn search_similar_finds_matching() {
        let db = HdcStoreDatabase::in_memory().unwrap();

        let query_hv = BinaryHV::random(100);
        db.store(make_record("exact", 100)).await.unwrap();
        for i in 0..20 {
            db.store(make_record(&format!("other_{i}"), i))
                .await
                .unwrap();
        }

        let results = db.search_similar(&query_hv, 5).await.unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].record.id, "exact");
        assert!((results[0].similarity - 1.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn delete_removes_record() {
        let db = HdcStoreDatabase::in_memory().unwrap();
        db.store(make_record("del", 1)).await.unwrap();
        assert_eq!(db.count().await.unwrap(), 1);

        let deleted = db.delete("del").await.unwrap();
        assert!(deleted);
        assert_eq!(db.count().await.unwrap(), 0);
        assert!(db.get("del").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn stats_populated() {
        let db = HdcStoreDatabase::in_memory().unwrap();
        for i in 0..5 {
            db.store(make_record(&format!("s{i}"), i)).await.unwrap();
        }

        let stats = db.stats().await.unwrap();
        assert_eq!(stats.total_records, 5);
        assert!(stats.backend_status.contains("5 live"));
    }

    #[test]
    fn blake3_hash_deterministic() {
        let h1 = id_to_u64("did:mycelix:alice");
        let h2 = id_to_u64("did:mycelix:alice");
        assert_eq!(h1, h2, "BLAKE3 hash must be deterministic");
    }

    #[test]
    fn blake3_hash_collision_resistance() {
        let h1 = id_to_u64("did:mycelix:alice");
        let h2 = id_to_u64("did:mycelix:bob");
        let h3 = id_to_u64("did:mycelix:alice2");
        let h4 = id_to_u64("");
        assert_ne!(h1, h2);
        assert_ne!(h1, h3);
        assert_ne!(h2, h3);
        assert_ne!(h1, h4);
    }

    #[tokio::test]
    async fn list_all_returns_all() {
        let db = HdcStoreDatabase::in_memory().unwrap();
        for i in 0..3 {
            db.store(make_record(&format!("l{i}"), i)).await.unwrap();
        }

        let all = db.list_all().await.unwrap();
        assert_eq!(all.len(), 3);
    }
}
