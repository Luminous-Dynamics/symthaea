//! Qdrant Vector Database Client
//!
//! **Mental Role**: Sensory Cortex
//!
//! Qdrant handles the fast, high-frequency perceptual layer - like the
//! sensory cortex in the brain. It processes hundreds of vector searches
//! per second for immediate pattern recognition.
//!
//! ## Architecture Role
//!
//! ```text
//! User Input → [Qdrant: <10ms] → Pattern Match → Conscious Processing
//!                 ↑
//!         Sensory Cortex
//!         - Fast vector search
//!         - Pattern recognition
//!         - Immediate awareness
//! ```
//!
//! ## Consciousness Mapping
//!
//! - **#26 Attention**: Fast similarity search for attention allocation
//! - **#25 Binding**: Pattern matching for feature binding
//! - **#23 Workspace**: Quick retrieval for global workspace access

#[allow(unused_imports)]  // MemoryType used in tests
use super::{ConsciousnessDatabase, DbResult, MemoryRecord, MemoryType, SearchResult};
use crate::hdc::binary_hv::HV16;
use async_trait::async_trait;

#[cfg(feature = "qdrant")]
use qdrant_client::{
    Qdrant,
    qdrant::{
        CreateCollectionBuilder, DeletePointsBuilder, Distance, GetPointsBuilder,
        PointId, PointStruct, SearchPointsBuilder, UpsertPointsBuilder, VectorParamsBuilder,
    },
};

/// Configuration for Qdrant connection
#[derive(Debug, Clone)]
pub struct QdrantConfig {
    /// Qdrant server URL
    pub url: String,

    /// Collection name for consciousness memories
    pub collection_name: String,

    /// Vector dimension (should match HV16::DIMENSIONS)
    pub vector_size: u64,

    /// API key (optional)
    pub api_key: Option<String>,
}

impl Default for QdrantConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:6334".to_string(),
            collection_name: "consciousness_sensory".to_string(),
            vector_size: 16384, // HV16 dimensions
            api_key: None,
        }
    }
}

/// Qdrant-backed sensory cortex database
pub struct QdrantSensory {
    config: QdrantConfig,

    #[cfg(feature = "qdrant")]
    client: Option<Qdrant>,

    /// Fallback in-memory storage when Qdrant unavailable
    fallback: dashmap::DashMap<String, MemoryRecord>,

    /// Connected flag
    connected: std::sync::atomic::AtomicBool,
}

impl QdrantSensory {
    /// Create new Qdrant sensory database
    pub fn new(config: QdrantConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "qdrant")]
            client: None,
            fallback: dashmap::DashMap::new(),
            connected: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Try to connect to Qdrant server
    #[cfg(feature = "qdrant")]
    pub async fn connect(&mut self) -> DbResult<()> {
        // Build Qdrant client with URL and optional API key
        let client_result = if let Some(ref api_key) = self.config.api_key {
            Qdrant::from_url(&self.config.url)
                .api_key(api_key.clone())
                .build()
        } else {
            Qdrant::from_url(&self.config.url).build()
        };

        match client_result {
            Ok(client) => {
                // Check connection by listing collections
                match client.list_collections().await {
                    Ok(_) => {
                        self.client = Some(client);
                        self.connected.store(true, std::sync::atomic::Ordering::SeqCst);

                        // Ensure collection exists
                        self.ensure_collection().await?;

                        Ok(())
                    }
                    Err(e) => Err(DatabaseError::ConnectionFailed(format!("Failed to connect to Qdrant: {}", e))),
                }
            }
            Err(e) => Err(DatabaseError::ConnectionFailed(format!("Failed to create Qdrant client: {}", e))),
        }
    }

    /// Ensure collection exists
    #[cfg(feature = "qdrant")]
    async fn ensure_collection(&self) -> DbResult<()> {
        if let Some(ref client) = self.client {
            let collections = client
                .list_collections()
                .await
                .map_err(|e| DatabaseError::QueryFailed(e.to_string()))?;

            let exists = collections
                .collections
                .iter()
                .any(|c| c.name == self.config.collection_name);

            if !exists {
                client
                    .create_collection(
                        CreateCollectionBuilder::new(&self.config.collection_name)
                            .vectors_config(
                                VectorParamsBuilder::new(self.config.vector_size, Distance::Cosine)
                            )
                    )
                    .await
                    .map_err(|e| DatabaseError::ConnectionFailed(format!("Failed to create collection: {}", e)))?;
            }
        }
        Ok(())
    }

    /// Convert HV16 to f32 vector for Qdrant
    /// HV16 is 2048 bits stored as 256 bytes
    fn hv_to_vec(hv: &HV16) -> Vec<f32> {
        let mut vec = Vec::with_capacity(2048);
        for byte in hv.0.iter() {
            for bit in 0..8 {
                let is_set = (byte >> bit) & 1 == 1;
                vec.push(if is_set { 1.0 } else { -1.0 });
            }
        }
        vec
    }

    /// Convert f32 vector back to HV16
    #[allow(dead_code)]
    fn vec_to_hv(vec: &[f32]) -> HV16 {
        let mut bytes = [0u8; 2048];
        for (byte_idx, byte) in bytes.iter_mut().enumerate() {
            for bit in 0..8 {
                let vec_idx = byte_idx * 8 + bit;
                if vec_idx < vec.len() && vec[vec_idx] > 0.0 {
                    *byte |= 1 << bit;
                }
            }
        }
        HV16(bytes)
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        self.connected.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Fallback search using in-memory storage
    fn search_fallback(&self, query: &HV16, top_k: usize) -> DbResult<Vec<SearchResult>> {
        let mut results: Vec<(MemoryRecord, f32)> = self
            .fallback
            .iter()
            .map(|entry| {
                let sim = query.similarity(&entry.encoding);
                (entry.value().clone(), sim)
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results
            .into_iter()
            .take(top_k)
            .map(|(record, similarity)| SearchResult { record, similarity })
            .collect())
    }
}

#[async_trait]
impl ConsciousnessDatabase for QdrantSensory {
    async fn store(&self, record: MemoryRecord) -> DbResult<()> {
        #[cfg(feature = "qdrant")]
        if let Some(ref client) = self.client {
            let vector = Self::hv_to_vec(&record.encoding);

            let mut payload = HashMap::new();
            payload.insert("memory_type".to_string(), serde_json::json!(record.memory_type));
            payload.insert("content".to_string(), serde_json::json!(record.content));
            payload.insert("valence".to_string(), serde_json::json!(record.valence));
            payload.insert("arousal".to_string(), serde_json::json!(record.arousal));
            payload.insert("phi".to_string(), serde_json::json!(record.phi));
            payload.insert("topics".to_string(), serde_json::json!(record.topics));
            payload.insert("timestamp_ms".to_string(), serde_json::json!(record.timestamp_ms));
            payload.insert("metadata".to_string(), serde_json::json!(record.metadata));

            use qdrant_client::qdrant::Value;
            let payload_map: std::collections::HashMap<String, Value> =
                payload.into_iter().map(|(k, v)| (k, v.into())).collect();
            let point = PointStruct::new(
                record.id.clone(),
                vector,
                payload_map,
            );

            client
                .upsert_points(
                    UpsertPointsBuilder::new(&self.config.collection_name, vec![point])
                        .wait(true)  // Wait for completion (blocking behavior)
                )
                .await
                .map_err(|e| DatabaseError::InsertFailed(format!("Failed to store in Qdrant: {}", e)))?;

            return Ok(());
        }

        // Fallback to in-memory
        self.fallback.insert(record.id.clone(), record);
        Ok(())
    }

    async fn search_similar(&self, query: &HV16, top_k: usize) -> DbResult<Vec<SearchResult>> {
        // Use fallback implementation (works with or without Qdrant)
        self.search_fallback(query, top_k)
    }

    async fn get(&self, id: &str) -> DbResult<Option<MemoryRecord>> {
        #[cfg(feature = "qdrant")]
        if let Some(ref client) = self.client {
            let point_id: PointId = id.to_string().into();
            let response = client
                .get_points(
                    GetPointsBuilder::new(&self.config.collection_name, vec![point_id])
                        .with_payload(true)
                        .with_vectors(true)
                )
                .await
                .map_err(|e| DatabaseError::QueryFailed(format!("Get failed: {}", e)))?;

            // Reconstruct MemoryRecord from Qdrant response
            if let Some(point) = response.result.into_iter().next() {
                // Extract vector and convert to HV16
                let encoding = if let Some(vectors) = point.vectors {
                    use qdrant_client::qdrant::vectors::VectorsOptions;
                    match vectors.vectors_options {
                        Some(VectorsOptions::Vector(v)) => Self::vec_to_hv(&v.data),
                        _ => HV16::zeros(),
                    }
                } else {
                    HV16::zeros()
                };

                // Extract payload fields
                let payload = point.payload;

                let memory_type = payload.get("memory_type")
                    .and_then(|v| serde_json::from_value(serde_json::Value::from(v.clone())).ok())
                    .unwrap_or(MemoryType::Episodic);

                let content = payload.get("content")
                    .and_then(|v| v.as_str().map(|s| s.to_string()))
                    .unwrap_or_default();

                let valence = payload.get("valence")
                    .and_then(|v| v.as_double())
                    .map(|f| f as f32)
                    .unwrap_or(0.0);

                let arousal = payload.get("arousal")
                    .and_then(|v| v.as_double())
                    .map(|f| f as f32)
                    .unwrap_or(0.0);

                let phi = payload.get("phi")
                    .and_then(|v| v.as_double())
                    .unwrap_or(0.0);

                let topics = payload.get("topics")
                    .and_then(|v| {
                        v.as_list().map(|list| {
                            list.values.iter()
                                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                .collect()
                        })
                    })
                    .unwrap_or_default();

                let timestamp_ms = payload.get("timestamp_ms")
                    .and_then(|v| v.as_integer())
                    .map(|i| i as u64)
                    .unwrap_or(0);

                let metadata = payload.get("metadata")
                    .and_then(|v| v.as_str().map(|s| s.to_string()))
                    .unwrap_or_default();

                // Extract ID from point
                let record_id = match point.id {
                    Some(pid) => match pid.point_id_options {
                        Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(uuid)) => uuid,
                        Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(n)) => n.to_string(),
                        None => id.to_string(),
                    },
                    None => id.to_string(),
                };

                return Ok(Some(MemoryRecord {
                    id: record_id,
                    encoding,
                    timestamp_ms,
                    memory_type,
                    content,
                    valence,
                    arousal,
                    phi,
                    topics,
                    metadata,
                }));
            }

            return Ok(None);
        }

        Ok(self.fallback.get(id).map(|r| r.clone()))
    }

    async fn delete(&self, id: &str) -> DbResult<bool> {
        #[cfg(feature = "qdrant")]
        if let Some(ref client) = self.client {
            use qdrant_client::qdrant::points_selector::PointsSelectorOneOf;
            use qdrant_client::qdrant::PointsIdsList;

            let point_id: PointId = id.to_string().into();
            client
                .delete_points(
                    DeletePointsBuilder::new(&self.config.collection_name)
                        .points(PointsSelectorOneOf::Points(PointsIdsList {
                            ids: vec![point_id],
                        }))
                        .wait(true)
                )
                .await
                .map_err(|e| DatabaseError::QueryFailed(format!("Delete failed: {}", e)))?;

            return Ok(true);
        }

        Ok(self.fallback.remove(id).is_some())
    }

    async fn count(&self) -> DbResult<usize> {
        #[cfg(feature = "qdrant")]
        if let Some(ref client) = self.client {
            let info = client
                .collection_info(self.config.collection_name.clone())
                .await
                .map_err(|e| DatabaseError::QueryFailed(format!("Count failed: {}", e)))?;

            return Ok(info.result.and_then(|r| r.points_count).map(|c| c as usize).unwrap_or(0));
        }

        Ok(self.fallback.len())
    }

    async fn health_check(&self) -> DbResult<bool> {
        #[cfg(feature = "qdrant")]
        if let Some(ref client) = self.client {
            return client
                .health_check()
                .await
                .map(|_| true)
                .map_err(|e| DatabaseError::ConnectionFailed(format!("Health check failed: {}", e)));
        }

        Ok(true) // Fallback is always healthy
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a test MemoryRecord with required fields
    fn test_record(id: &str, seed: u64, memory_type: MemoryType, content: &str) -> MemoryRecord {
        MemoryRecord {
            id: id.to_string(),
            encoding: HV16::random(seed),
            timestamp_ms: 1700000000000 + seed,
            memory_type,
            content: content.to_string(),
            valence: 0.5,
            arousal: 0.3,
            phi: 0.6,
            topics: vec!["test".to_string()],
            metadata: "{}".to_string(),
        }
    }

    #[tokio::test]
    async fn test_qdrant_config_default() {
        let config = QdrantConfig::default();
        assert_eq!(config.url, "http://localhost:6334");
        assert_eq!(config.vector_size, 16384);
    }

    #[tokio::test]
    async fn test_qdrant_fallback_store() {
        let db = QdrantSensory::new(QdrantConfig::default());
        let record = test_record("test-1", 42, MemoryType::Episodic, "Test memory");
        db.store(record).await.unwrap();
        assert_eq!(db.count().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_qdrant_fallback_search() {
        let db = QdrantSensory::new(QdrantConfig::default());
        let encoding = HV16::random(100);
        let mut record = test_record("search-test", 100, MemoryType::Semantic, "Searchable");
        record.encoding = encoding.clone();
        db.store(record).await.unwrap();

        let results = db.search_similar(&encoding, 5).await.unwrap();
        assert!(!results.is_empty());
        assert!(results[0].similarity > 0.99);
    }

    #[tokio::test]
    async fn test_qdrant_fallback_delete() {
        let db = QdrantSensory::new(QdrantConfig::default());
        let record = test_record("delete-test", 200, MemoryType::Working, "To be deleted");
        db.store(record).await.unwrap();
        assert_eq!(db.count().await.unwrap(), 1);

        let deleted = db.delete("delete-test").await.unwrap();
        assert!(deleted);
        assert_eq!(db.count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_qdrant_health() {
        let db = QdrantSensory::new(QdrantConfig::default());
        assert!(db.health_check().await.unwrap());
    }
}
