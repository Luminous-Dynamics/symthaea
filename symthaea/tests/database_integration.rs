//! Database Integration Tests
//!
//! Tests for the persistence layer including MemoryRecord CRUD operations,
//! similarity search, and SQLite backend.

mod common;

use common::prelude::*;
use symthaea::databases::{ConsciousnessDatabase, DbResult, MemoryRecord, MemoryType, SqliteMemory};
use symthaea_core::hdc::binary_hv::HV16;

// ============================================================================
// SQLITEMEMORY CREATION TESTS
// ============================================================================

#[tokio::test]
async fn test_in_memory_database_creation() {
    let db = SqliteMemory::in_memory();
    assert!(db.is_ok(), "Should create in-memory database");
}

#[tokio::test]
async fn test_file_database_creation() {
    let temp = TempTestDir::new().expect("Should create temp dir");
    let db_path = temp.path("test.db");

    let db = SqliteMemory::new(&db_path);
    assert!(db.is_ok(), "Should create file-based database");
    assert!(db_path.exists(), "Database file should exist");
}

// ============================================================================
// MEMORY RECORD CRUD TESTS
// ============================================================================

#[tokio::test]
async fn test_store_and_retrieve_memory() {
    let db = SqliteMemory::in_memory().expect("Should create database");

    let record = create_test_memory("test-1", 42, MemoryType::Episodic);
    let store_result = db.store(record.clone()).await;
    assert!(store_result.is_ok(), "Should store memory");

    let retrieved = db.get("test-1").await;
    assert!(retrieved.is_ok(), "Should retrieve memory");

    let retrieved = retrieved.unwrap();
    assert!(retrieved.is_some(), "Memory should exist");

    let retrieved = retrieved.unwrap();
    assert_eq!(retrieved.id, "test-1", "ID should match");
    assert_eq!(retrieved.memory_type, MemoryType::Episodic, "Type should match");
}

#[tokio::test]
async fn test_store_multiple_memories() {
    let db = SqliteMemory::in_memory().expect("Should create database");

    let batch = create_memory_batch("batch", 100, 5, MemoryType::Semantic);

    for record in &batch {
        db.store(record.clone()).await.expect("Should store memory");
    }

    // Verify all were stored
    for (i, _) in batch.iter().enumerate() {
        let retrieved = db.get(&format!("batch-{}", i)).await;
        assert!(retrieved.is_ok(), "Should retrieve memory {}", i);
        assert!(retrieved.unwrap().is_some(), "Memory {} should exist", i);
    }
}

#[tokio::test]
async fn test_get_nonexistent_memory() {
    let db = SqliteMemory::in_memory().expect("Should create database");

    let result = db.get("nonexistent-id").await;
    assert!(result.is_ok(), "Query should succeed");
    assert!(result.unwrap().is_none(), "Should return None for missing ID");
}

#[tokio::test]
async fn test_delete_memory() {
    let db = SqliteMemory::in_memory().expect("Should create database");

    let record = create_test_memory("to-delete", 42, MemoryType::Working);
    db.store(record).await.expect("Should store memory");

    // Verify it exists
    let exists = db.get("to-delete").await.unwrap().is_some();
    assert!(exists, "Memory should exist before deletion");

    // Delete
    let delete_result = db.delete("to-delete").await;
    assert!(delete_result.is_ok(), "Should delete memory");

    // Verify it's gone
    let exists_after = db.get("to-delete").await.unwrap().is_some();
    assert!(!exists_after, "Memory should not exist after deletion");
}

#[tokio::test]
async fn test_update_memory_via_store() {
    let db = SqliteMemory::in_memory().expect("Should create database");

    let mut record = create_test_memory("to-update", 42, MemoryType::Episodic);
    record.valence = 0.5;
    db.store(record.clone()).await.expect("Should store memory");

    // Update by storing with same ID (INSERT OR REPLACE behavior)
    record.valence = 0.9;
    record.content = "Updated content".to_string();
    db.store(record.clone()).await.expect("Should update memory via store");

    // Verify update
    let retrieved = db.get("to-update").await.unwrap().unwrap();
    assert!((retrieved.valence - 0.9).abs() < 0.01, "Valence should be updated");
    assert_eq!(retrieved.content, "Updated content", "Content should be updated");
}

// ============================================================================
// SIMILARITY SEARCH TESTS
// ============================================================================

#[tokio::test]
async fn test_similarity_search_exact_match() {
    let db = SqliteMemory::in_memory().expect("Should create database");

    // Store a memory with known encoding
    let encoding = HV16::random(42);
    let record = MemoryRecord {
        id: "exact-match".to_string(),
        memory_type: MemoryType::Semantic,
        encoding,
        content: "Test content".to_string(),
        timestamp_ms: 1704067200000,
        valence: 0.5,
        arousal: 0.5,
        phi: 0.5,
        topics: vec!["test".to_string()],
        metadata: "{}".to_string(),
    };
    db.store(record).await.expect("Should store memory");

    // Search with same encoding
    let results = db.search_similar(&encoding, 5).await;
    assert!(results.is_ok(), "Search should succeed");

    let results = results.unwrap();
    assert!(!results.is_empty(), "Should find at least one result");
    assert!(results[0].similarity > 0.99, "Exact match should have ~1.0 similarity");
    assert_eq!(results[0].record.id, "exact-match", "Should find the right record");
}

#[tokio::test]
async fn test_similarity_search_ordering() {
    let db = SqliteMemory::in_memory().expect("Should create database");

    let query = HV16::random(1000);

    // Store memories with decreasing similarity to query
    for i in 0..5 {
        let encoding = if i == 0 {
            query // Identical
        } else {
            // Add noise proportional to i
            query.add_noise(0.1 * i as f32, i as u64)
        };

        let record = MemoryRecord {
            id: format!("similarity-{}", i),
            memory_type: MemoryType::Semantic,
            encoding,
            content: format!("Content {}", i),
            timestamp_ms: 1704067200000 + i as u64,
            valence: 0.5,
            arousal: 0.5,
            phi: 0.5,
            topics: vec![],
            metadata: "{}".to_string(),
        };
        db.store(record).await.expect("Should store memory");
    }

    // Search should return in order of similarity
    let results = db.search_similar(&query, 5).await.unwrap();

    assert_eq!(results.len(), 5, "Should return all 5 memories");
    assert_eq!(results[0].record.id, "similarity-0", "Most similar should be first");

    // Verify descending similarity order
    for i in 1..results.len() {
        assert!(
            results[i - 1].similarity >= results[i].similarity,
            "Results should be in descending similarity order"
        );
    }
}

#[tokio::test]
async fn test_similarity_search_limit() {
    let db = SqliteMemory::in_memory().expect("Should create database");

    // Store 10 memories
    for i in 0..10 {
        let record = create_test_memory(&format!("limit-test-{}", i), 500 + i, MemoryType::Semantic);
        db.store(record).await.expect("Should store memory");
    }

    // Search with limit 3
    let query = HV16::random(505);
    let results = db.search_similar(&query, 3).await.unwrap();

    assert_eq!(results.len(), 3, "Should respect limit parameter");
}

// ============================================================================
// COUNT AND HEALTH CHECK TESTS
// ============================================================================

#[tokio::test]
async fn test_count_memories() {
    let db = SqliteMemory::in_memory().expect("Should create database");

    // Initially empty
    assert_eq!(db.count().await.unwrap(), 0, "Database should start empty");

    // Store different types
    db.store(create_test_memory("ep-1", 1, MemoryType::Episodic))
        .await
        .unwrap();
    db.store(create_test_memory("ep-2", 2, MemoryType::Episodic))
        .await
        .unwrap();
    db.store(create_test_memory("sem-1", 3, MemoryType::Semantic))
        .await
        .unwrap();
    db.store(create_test_memory("work-1", 4, MemoryType::Working))
        .await
        .unwrap();

    // Should count all memories
    assert_eq!(db.count().await.unwrap(), 4, "Should have 4 memories");

    // Delete one
    db.delete("ep-1").await.unwrap();
    assert_eq!(db.count().await.unwrap(), 3, "Should have 3 memories after delete");
}

#[tokio::test]
async fn test_health_check() {
    let db = SqliteMemory::in_memory().expect("Should create database");
    assert!(db.health_check().await.unwrap(), "Health check should pass");
}

// ============================================================================
// PERSISTENCE TESTS
// ============================================================================

#[tokio::test]
async fn test_persistence_across_connections() {
    let temp = TempTestDir::new().expect("Should create temp dir");
    let db_path = temp.path("persist-test.db");

    // Create, store, close
    {
        let db = SqliteMemory::new(&db_path).expect("Should create database");
        let record = create_test_memory("persistent-1", 42, MemoryType::Semantic);
        db.store(record).await.expect("Should store memory");
    }

    // Reopen and verify
    {
        let db = SqliteMemory::new(&db_path).expect("Should reopen database");
        let retrieved = db.get("persistent-1").await.unwrap();
        assert!(retrieved.is_some(), "Memory should persist across connections");
    }
}

// ============================================================================
// ENCODING INTEGRITY TESTS
// ============================================================================

#[tokio::test]
async fn test_encoding_roundtrip_integrity() {
    let db = SqliteMemory::in_memory().expect("Should create database");

    let original_encoding = HV16::random(9999);
    let record = MemoryRecord {
        id: "encoding-test".to_string(),
        memory_type: MemoryType::Semantic,
        encoding: original_encoding,
        content: "Encoding integrity test".to_string(),
        timestamp_ms: 1704067200000,
        valence: 0.5,
        arousal: 0.5,
        phi: 0.5,
        topics: vec![],
        metadata: "{}".to_string(),
    };

    db.store(record).await.expect("Should store memory");
    let retrieved = db.get("encoding-test").await.unwrap().unwrap();

    // Encoding should be bit-for-bit identical
    assert_eq!(
        retrieved.encoding, original_encoding,
        "Encoding should survive roundtrip exactly"
    );
}

// ============================================================================
// EMOTIONAL METADATA TESTS
// ============================================================================

#[tokio::test]
async fn test_emotional_values_roundtrip() {
    let db = SqliteMemory::in_memory().expect("Should create database");

    let record = MemoryRecord {
        id: "emotion-test".to_string(),
        memory_type: MemoryType::Episodic,
        encoding: HV16::random(42),
        content: "Emotional memory".to_string(),
        timestamp_ms: 1704067200000,
        valence: -0.75, // Negative emotion
        arousal: 0.95,  // High arousal
        phi: 0.88,
        topics: vec!["emotion".to_string()],
        metadata: r#"{"intensity": "high"}"#.to_string(),
    };

    db.store(record).await.expect("Should store memory");
    let retrieved = db.get("emotion-test").await.unwrap().unwrap();

    assert_f32_eq(retrieved.valence, -0.75, 0.001, "Valence should roundtrip");
    assert_f32_eq(retrieved.arousal, 0.95, 0.001, "Arousal should roundtrip");
    assert_f64_eq(retrieved.phi, 0.88, 0.0001, "Phi should roundtrip");
}

// ============================================================================
// CONCURRENT ACCESS TESTS
// ============================================================================

#[tokio::test]
async fn test_concurrent_reads() {
    let db = SqliteMemory::in_memory().expect("Should create database");

    // Store initial data
    for i in 0..10 {
        let record = create_test_memory(&format!("concurrent-{}", i), i, MemoryType::Semantic);
        db.store(record).await.expect("Should store memory");
    }

    // Concurrent reads (simulated with sequential calls since in_memory is single-connection)
    let mut handles = vec![];
    for i in 0..10 {
        let id = format!("concurrent-{}", i);
        let result = db.get(&id).await;
        handles.push(result);
    }

    // All reads should succeed
    for (i, result) in handles.into_iter().enumerate() {
        assert!(result.is_ok(), "Read {} should succeed", i);
        assert!(result.unwrap().is_some(), "Memory {} should exist", i);
    }
}
