//! Mock Database Implementation
//!
//! In-memory implementation for testing and development without
//! requiring actual database connections.

use super::{ConsciousnessDatabase, DbResult, MemoryRecord, SearchResult};
use crate::hdc::HV16;
use dashmap::DashMap;
use std::sync::Arc;

/// Mock in-memory database for testing
pub struct MockDatabase {
    records: Arc<DashMap<String, MemoryRecord>>,
}

impl MockDatabase {
    pub fn new() -> Self {
        Self {
            records: Arc::new(DashMap::new()),
        }
    }

    /// Get number of records
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

impl Default for MockDatabase {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl ConsciousnessDatabase for MockDatabase {
    async fn store(&self, record: MemoryRecord) -> DbResult<()> {
        self.records.insert(record.id.clone(), record);
        Ok(())
    }

    async fn search_similar(&self, query: &HV16, top_k: usize) -> DbResult<Vec<SearchResult>> {
        let mut results: Vec<SearchResult> = self.records
            .iter()
            .map(|entry| {
                let record = entry.value().clone();
                let similarity = query.similarity(&record.encoding);
                SearchResult { record, similarity }
            })
            .collect();

        // Sort by similarity (descending)
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        Ok(results)
    }

    async fn get(&self, id: &str) -> DbResult<Option<MemoryRecord>> {
        Ok(self.records.get(id).map(|r| r.value().clone()))
    }

    async fn delete(&self, id: &str) -> DbResult<bool> {
        Ok(self.records.remove(id).is_some())
    }

    async fn count(&self) -> DbResult<usize> {
        Ok(self.records.len())
    }

    async fn health_check(&self) -> DbResult<bool> {
        Ok(true) // Mock is always healthy
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::databases::MemoryType;

    fn create_test_record(id: &str, seed: u64) -> MemoryRecord {
        MemoryRecord {
            id: id.to_string(),
            encoding: HV16::random(seed),
            timestamp_ms: 1700000000000,
            memory_type: MemoryType::Episodic,
            content: format!("Memory {}", id),
            valence: 0.5,
            arousal: 0.3,
            phi: 0.65,
            topics: vec!["test".to_string()],
            metadata: "{}".to_string(),
        }
    }

    #[tokio::test]
    async fn test_mock_store_and_get() {
        let db = MockDatabase::new();
        let record = create_test_record("test-1", 42);

        db.store(record.clone()).await.unwrap();

        let retrieved = db.get("test-1").await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, "test-1");
    }

    #[tokio::test]
    async fn test_mock_search_similar() {
        let db = MockDatabase::new();

        // Store some records
        for i in 0..5 {
            let record = create_test_record(&format!("test-{}", i), i as u64 * 100);
            db.store(record).await.unwrap();
        }

        // Search for similar
        let query = HV16::random(100); // Similar to test-1
        let results = db.search_similar(&query, 3).await.unwrap();

        assert_eq!(results.len(), 3);
        // First result should have highest similarity
        assert!(results[0].similarity >= results[1].similarity);
    }

    #[tokio::test]
    async fn test_mock_delete() {
        let db = MockDatabase::new();
        let record = create_test_record("to-delete", 42);

        db.store(record).await.unwrap();
        assert!(db.get("to-delete").await.unwrap().is_some());

        let deleted = db.delete("to-delete").await.unwrap();
        assert!(deleted);
        assert!(db.get("to-delete").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_mock_count() {
        let db = MockDatabase::new();

        assert_eq!(db.count().await.unwrap(), 0);

        for i in 0..10 {
            let record = create_test_record(&format!("test-{}", i), i as u64);
            db.store(record).await.unwrap();
        }

        assert_eq!(db.count().await.unwrap(), 10);
    }

    #[tokio::test]
    async fn test_mock_health_check() {
        let db = MockDatabase::new();
        assert!(db.health_check().await.unwrap());
    }
}
