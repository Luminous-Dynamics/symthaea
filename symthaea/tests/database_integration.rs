// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Database Integration Tests
//!
//! Tests for the persistence layer including MemoryRecord CRUD operations,
//! similarity search, and SQLite backend.

mod common;

use common::prelude::*;
use symthaea::databases::{ConsciousnessDatabase, MemoryRecord, MemoryType, SqliteMemory};
use symthaea_core::hdc::binary_hv::BinaryHV;

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
    assert_eq!(
        retrieved.memory_type,
        MemoryType::Episodic,
        "Type should match"
    );
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
    assert!(
        result.unwrap().is_none(),
        "Should return None for missing ID"
    );
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
    db.store(record.clone())
        .await
        .expect("Should update memory via store");

    // Verify update
    let retrieved = db.get("to-update").await.unwrap().unwrap();
    assert!(
        (retrieved.valence - 0.9).abs() < 0.01,
        "Valence should be updated"
    );
    assert_eq!(
        retrieved.content, "Updated content",
        "Content should be updated"
    );
}

// ============================================================================
// SIMILARITY SEARCH TESTS
// ============================================================================

#[tokio::test]
async fn test_similarity_search_exact_match() {
    let db = SqliteMemory::in_memory().expect("Should create database");

    // Store a memory with known encoding
    let encoding = BinaryHV::random(42);
    let record = MemoryRecord {
        id: "exact-match".to_string(),
        memory_type: MemoryType::Semantic,
        encoding,
        content: "Test content".to_string(),
        timestamp_ms: 1704067200000,
        valence: 0.5,
        arousal: 0.5,
        psi: 0.5,
        topics: vec!["test".to_string()],
        metadata: "{}".to_string(),
        consolidation_strength: 0.0,
        retrieval_count: 0,
    };
    db.store(record).await.expect("Should store memory");

    // Search with same encoding
    let results = db.search_similar(&encoding, 5).await;
    assert!(results.is_ok(), "Search should succeed");

    let results = results.unwrap();
    assert!(!results.is_empty(), "Should find at least one result");
    assert!(
        results[0].similarity > 0.99,
        "Exact match should have ~1.0 similarity"
    );
    assert_eq!(
        results[0].record.id, "exact-match",
        "Should find the right record"
    );
}

#[tokio::test]
async fn test_similarity_search_ordering() {
    let db = SqliteMemory::in_memory().expect("Should create database");

    let query = BinaryHV::random(1000);

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
            psi: 0.5,
            topics: vec![],
            metadata: "{}".to_string(),
            consolidation_strength: 0.0,
            retrieval_count: 0,
        };
        db.store(record).await.expect("Should store memory");
    }

    // Search should return in order of similarity
    let results = db.search_similar(&query, 5).await.unwrap();

    assert_eq!(results.len(), 5, "Should return all 5 memories");
    assert_eq!(
        results[0].record.id, "similarity-0",
        "Most similar should be first"
    );

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
        let record =
            create_test_memory(&format!("limit-test-{}", i), 500 + i, MemoryType::Semantic);
        db.store(record).await.expect("Should store memory");
    }

    // Search with limit 3
    let query = BinaryHV::random(505);
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
    assert_eq!(
        db.count().await.unwrap(),
        3,
        "Should have 3 memories after delete"
    );
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
        assert!(
            retrieved.is_some(),
            "Memory should persist across connections"
        );
    }
}

// ============================================================================
// ENCODING INTEGRITY TESTS
// ============================================================================

#[tokio::test]
async fn test_encoding_roundtrip_integrity() {
    let db = SqliteMemory::in_memory().expect("Should create database");

    let original_encoding = BinaryHV::random(9999);
    let record = MemoryRecord {
        id: "encoding-test".to_string(),
        memory_type: MemoryType::Semantic,
        encoding: original_encoding,
        content: "Encoding integrity test".to_string(),
        timestamp_ms: 1704067200000,
        valence: 0.5,
        arousal: 0.5,
        psi: 0.5,
        topics: vec![],
        metadata: "{}".to_string(),
        consolidation_strength: 0.0,
        retrieval_count: 0,
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
        encoding: BinaryHV::random(42),
        content: "Emotional memory".to_string(),
        timestamp_ms: 1704067200000,
        valence: -0.75, // Negative emotion
        arousal: 0.95,  // High arousal
        psi: 0.88,
        topics: vec!["emotion".to_string()],
        metadata: r#"{"intensity": "high"}"#.to_string(),
        consolidation_strength: 0.0,
        retrieval_count: 0,
    };

    db.store(record).await.expect("Should store memory");
    let retrieved = db.get("emotion-test").await.unwrap().unwrap();

    assert_f32_eq(retrieved.valence, -0.75, 0.001, "Valence should roundtrip");
    assert_f32_eq(retrieved.arousal, 0.95, 0.001, "Arousal should roundtrip");
    assert_f64_eq(retrieved.psi, 0.88, 0.0001, "Phi should roundtrip");
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

// ============================================================================
// LIST ALL TESTS
// ============================================================================

#[tokio::test]
async fn test_sqlite_list_all() {
    let db = SqliteMemory::in_memory().expect("Should create database");

    let batch = create_memory_batch("list-all", 200, 50, MemoryType::Semantic);
    for record in &batch {
        db.store(record.clone()).await.expect("Should store memory");
    }

    let all = db.list_all().await.expect("list_all should succeed");
    assert_eq!(all.len(), 50, "Should return all 50 records");

    // Verify ascending timestamp order
    for i in 1..all.len() {
        assert!(
            all[i].timestamp_ms >= all[i - 1].timestamp_ms,
            "list_all should return records in ascending timestamp order"
        );
    }
}

// ============================================================================
// DEFAULT CONFIG TESTS
// ============================================================================

#[tokio::test]
async fn test_default_config_creates_sqlite() {
    use symthaea::databases::{DatabaseConfig, create_database};

    let db = create_database(&DatabaseConfig::default())
        .await
        .expect("Default config should create a database");

    assert!(db.health_check().await.unwrap(), "Health check should pass");
    assert_eq!(
        db.count().await.unwrap(),
        0,
        "Fresh database should be empty"
    );
}

// ============================================================================
// SEARCH SIMILAR FILTERED (SQLITE — NOW SUPPORTS FILTERING)
// ============================================================================

#[tokio::test]
async fn test_sqlite_search_similar_filtered_by_memory_type() {
    let db = SqliteMemory::in_memory().expect("Should create database");

    // Store episodic and semantic records
    for i in 0..5 {
        db.store(create_test_memory(
            &format!("ep-filter-{i}"),
            100 + i,
            MemoryType::Episodic,
        ))
        .await
        .unwrap();
        db.store(create_test_memory(
            &format!("sem-filter-{i}"),
            200 + i,
            MemoryType::Semantic,
        ))
        .await
        .unwrap();
    }

    let query = BinaryHV::random(150);

    // Filter to episodic only
    let results = db
        .search_similar_filtered(&query, 10, Some("memory_type = 'episodic'"))
        .await
        .expect("Filtered search should succeed");

    assert_eq!(results.len(), 5, "Should return only 5 episodic results");
    for r in &results {
        assert_eq!(r.record.memory_type, MemoryType::Episodic);
    }

    // Filter to semantic only
    let results = db
        .search_similar_filtered(&query, 10, Some("memory_type = 'semantic'"))
        .await
        .expect("Filtered search should succeed");

    assert_eq!(results.len(), 5, "Should return only 5 semantic results");
    for r in &results {
        assert_eq!(r.record.memory_type, MemoryType::Semantic);
    }

    // No filter — returns all
    let results = db
        .search_similar_filtered(&query, 20, None)
        .await
        .expect("Unfiltered search should succeed");

    assert_eq!(results.len(), 10, "No filter should return all 10 records");
}

#[tokio::test]
async fn test_sqlite_search_similar_filtered_by_phi() {
    let db = SqliteMemory::in_memory().expect("Should create database");

    // Store records with varying phi values
    for i in 0..10u64 {
        let mut record = create_test_memory(&format!("phi-{i}"), i, MemoryType::Episodic);
        record.psi = i as f64 * 0.1; // 0.0, 0.1, ..., 0.9
        db.store(record).await.unwrap();
    }

    let query = BinaryHV::random(42);

    // Filter to high-phi records only
    let results = db
        .search_similar_filtered(&query, 20, Some("phi > 0.5"))
        .await
        .expect("Phi-filtered search should succeed");

    // phi > 0.5 matches 0.6, 0.7, 0.8, 0.9 → 4 records
    assert_eq!(results.len(), 4);
    for r in &results {
        assert!(r.record.psi > 0.5, "All results should have phi > 0.5");
    }
}

#[tokio::test]
async fn test_sqlite_search_similar_filtered_rejects_invalid_column() {
    let db = SqliteMemory::in_memory().expect("Should create database");
    let query = BinaryHV::random(42);

    // Attempt SQL injection via disallowed column name
    let result = db
        .search_similar_filtered(&query, 10, Some("1=1; DROP TABLE memories; --"))
        .await;

    assert!(result.is_err(), "Should reject non-allowlisted column");
}

// ============================================================================
// RECONSOLIDATION TRACKING TESTS
// ============================================================================

#[tokio::test]
async fn test_sqlite_reconsolidation_tracking() {
    let db = SqliteMemory::in_memory().expect("Should create database");

    // Store a record
    let encoding = BinaryHV::random(7777);
    let record = MemoryRecord {
        id: "recon-test".to_string(),
        memory_type: MemoryType::Semantic,
        encoding,
        content: "Reconsolidation test".to_string(),
        timestamp_ms: 1704067200000,
        valence: 0.5,
        arousal: 0.5,
        psi: 0.5,
        topics: vec![],
        metadata: "{}".to_string(),
        consolidation_strength: 0.0,
        retrieval_count: 0,
    };
    db.store(record).await.expect("Should store memory");

    // Search triggers reconsolidation
    let results = db
        .search_similar(&encoding, 5)
        .await
        .expect("Search should succeed");
    assert!(!results.is_empty(), "Should find the record");
    assert_eq!(results[0].record.id, "recon-test");

    // Verify retrieval_count was bumped
    let retrieved = db.get("recon-test").await.unwrap().unwrap();
    assert_eq!(
        retrieved.retrieval_count, 1,
        "retrieval_count should be bumped after search"
    );
    assert!(
        retrieved.consolidation_strength > 0.0,
        "consolidation_strength should increase after search"
    );

    // Search again — should bump further
    let _ = db.search_similar(&encoding, 5).await.unwrap();
    let retrieved2 = db.get("recon-test").await.unwrap().unwrap();
    assert_eq!(
        retrieved2.retrieval_count, 2,
        "retrieval_count should increment on each search"
    );
}