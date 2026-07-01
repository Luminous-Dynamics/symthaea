// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Database Persistence Integration Tests
//!
//! Comprehensive tests for SQLite database layer including:
//! - Roundtrip tests (store and retrieve data integrity)
//! - ACID transaction properties
//! - Concurrent access patterns
//! - Recovery from corruption/edge cases
//!
//! Run with: `cargo test --test database_persistence`

use std::sync::Arc;
use std::time::{Duration, Instant};
use symthaea::databases::{ConsciousnessDatabase, MemoryRecord, MemoryType, SqliteMemory};
use symthaea::hdc::binary_hv::BinaryHV;
use tempfile::TempDir;

// ============================================================================
// Test Utilities
// ============================================================================

/// Create a test memory record with specified parameters
fn create_memory(id: &str, seed: u64, memory_type: MemoryType, content: &str) -> MemoryRecord {
    MemoryRecord {
        id: id.to_string(),
        encoding: BinaryHV::random(seed),
        timestamp_ms: 1700000000000 + seed,
        memory_type,
        content: content.to_string(),
        valence: ((seed % 200) as f32 / 100.0) - 1.0, // -1.0 to 1.0
        arousal: (seed % 100) as f32 / 100.0,         // 0.0 to 1.0
        psi: (seed % 100) as f64 / 100.0,             // 0.0 to 1.0
        topics: vec![format!("topic_{}", seed % 10)],
        metadata: format!(r#"{{"seed": {}}}"#, seed),
        consolidation_strength: 0.0,
        retrieval_count: 0,
    }
}

/// Create a temporary database for testing
fn temp_db() -> (TempDir, SqliteMemory) {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let db_path = temp_dir.path().join("test.db");
    let db = SqliteMemory::new(&db_path).expect("Failed to create database");
    (temp_dir, db)
}

// ============================================================================
// Section 1: Roundtrip Tests
// ============================================================================

mod roundtrip_tests {
    use super::*;

    #[tokio::test]
    async fn test_basic_store_and_retrieve() {
        let db = SqliteMemory::in_memory().expect("Failed to create in-memory database");

        let original = create_memory("roundtrip-1", 42, MemoryType::Episodic, "Test content");
        db.store(original.clone()).await.expect("Store failed");

        let retrieved = db.get("roundtrip-1").await.expect("Get failed");
        assert!(retrieved.is_some(), "Memory should exist");

        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.id, original.id);
        assert_eq!(retrieved.content, original.content);
        assert_eq!(retrieved.memory_type, original.memory_type);
        assert!((retrieved.valence - original.valence).abs() < 0.001);
        assert!((retrieved.arousal - original.arousal).abs() < 0.001);
        assert!((retrieved.psi - original.psi).abs() < 0.0001);
    }

    #[tokio::test]
    async fn test_encoding_roundtrip_integrity() {
        let db = SqliteMemory::in_memory().expect("Failed to create database");

        // Test with specific seed to ensure deterministic encoding
        let seed = 12345u64;
        let original_encoding = BinaryHV::random(seed);
        let record = MemoryRecord {
            id: "encoding-test".to_string(),
            encoding: original_encoding,
            timestamp_ms: 1700000000000,
            memory_type: MemoryType::Semantic,
            content: "Testing encoding integrity".to_string(),
            valence: 0.0,
            arousal: 0.5,
            psi: 0.75,
            topics: vec![],
            metadata: "{}".to_string(),
            consolidation_strength: 0.0,
            retrieval_count: 0,
        };

        db.store(record).await.expect("Store failed");
        let retrieved = db.get("encoding-test").await.expect("Get failed").unwrap();

        // Verify encoding matches exactly (bit-for-bit)
        let similarity = original_encoding.similarity(&retrieved.encoding);
        assert!(
            (similarity - 1.0).abs() < 0.0001,
            "Encoding should be identical, got similarity: {}",
            similarity
        );
    }

    #[tokio::test]
    async fn test_all_memory_types_roundtrip() {
        let db = SqliteMemory::in_memory().expect("Failed to create database");

        let types = [
            (MemoryType::Episodic, "episodic-1"),
            (MemoryType::Semantic, "semantic-1"),
            (MemoryType::Procedural, "procedural-1"),
            (MemoryType::Working, "working-1"),
        ];

        // Store all types
        for (i, (mem_type, id)) in types.iter().enumerate() {
            let record = create_memory(
                id,
                i as u64,
                *mem_type,
                &format!("Content for {:?}", mem_type),
            );
            db.store(record).await.expect("Store failed");
        }

        // Verify all types
        for (mem_type, id) in types.iter() {
            let retrieved = db.get(id).await.expect("Get failed").unwrap();
            assert_eq!(
                retrieved.memory_type, *mem_type,
                "Memory type mismatch for {}",
                id
            );
        }
    }

    #[tokio::test]
    async fn test_unicode_content_roundtrip() {
        let db = SqliteMemory::in_memory().expect("Failed to create database");

        let unicode_content = "Hello 世界! 🌍 مرحبا العالم Привет мир こんにちは";
        let record = create_memory("unicode-test", 1, MemoryType::Semantic, unicode_content);

        db.store(record).await.expect("Store failed");
        let retrieved = db.get("unicode-test").await.expect("Get failed").unwrap();

        assert_eq!(
            retrieved.content, unicode_content,
            "Unicode content should be preserved"
        );
    }

    #[tokio::test]
    async fn test_large_metadata_roundtrip() {
        let db = SqliteMemory::in_memory().expect("Failed to create database");

        // Create large metadata JSON
        let large_metadata = serde_json::json!({
            "nested": {
                "deeply": {
                    "nested": {
                        "value": "test",
                        "array": (0..100).collect::<Vec<_>>()
                    }
                }
            },
            "large_string": "x".repeat(10000)
        })
        .to_string();

        let mut record =
            create_memory("large-meta", 1, MemoryType::Semantic, "Large metadata test");
        record.metadata = large_metadata.clone();

        db.store(record).await.expect("Store failed");
        let retrieved = db.get("large-meta").await.expect("Get failed").unwrap();

        assert_eq!(retrieved.metadata, large_metadata);
    }

    #[tokio::test]
    async fn test_topics_array_roundtrip() {
        let db = SqliteMemory::in_memory().expect("Failed to create database");

        let topics = vec![
            "consciousness".to_string(),
            "φ-integration".to_string(),
            "memory-consolidation".to_string(),
            "topic with spaces".to_string(),
            "unicode-话题".to_string(),
        ];

        let mut record = create_memory("topics-test", 1, MemoryType::Episodic, "Topics test");
        record.topics = topics.clone();

        db.store(record).await.expect("Store failed");
        let retrieved = db.get("topics-test").await.expect("Get failed").unwrap();

        assert_eq!(retrieved.topics, topics, "Topics array should be preserved");
    }

    #[tokio::test]
    async fn test_extreme_values_roundtrip() {
        let db = SqliteMemory::in_memory().expect("Failed to create database");

        let record = MemoryRecord {
            id: "extreme-values".to_string(),
            encoding: BinaryHV::random(0),
            timestamp_ms: u64::MAX,
            memory_type: MemoryType::Working,
            content: "".to_string(), // Empty content
            valence: 1.0,            // Max positive
            arousal: 0.0,            // Min arousal
            psi: 1.0,                // Max phi
            topics: vec![],          // Empty topics
            metadata: "{}".to_string(),
            consolidation_strength: 0.0,
            retrieval_count: 0,
        };

        db.store(record).await.expect("Store failed");
        let retrieved = db.get("extreme-values").await.expect("Get failed").unwrap();

        // timestamp_ms is stored as i64, so MAX u64 will overflow
        // This tests how the system handles edge cases
        assert!(retrieved.content.is_empty());
        assert!(retrieved.topics.is_empty());
        assert!((retrieved.valence - 1.0).abs() < 0.001);
        assert!((retrieved.psi - 1.0).abs() < 0.0001);
    }
}

// ============================================================================
// Section 2: ACID Transaction Tests
// ============================================================================

mod acid_tests {
    use super::*;

    #[tokio::test]
    async fn test_atomicity_single_insert() {
        let db = SqliteMemory::in_memory().expect("Failed to create database");

        let record = create_memory("atomic-1", 1, MemoryType::Episodic, "Atomic insert test");
        db.store(record).await.expect("Store should succeed");

        // Verify the record exists
        let count = db.count().await.expect("Count failed");
        assert_eq!(count, 1, "Should have exactly one record");
    }

    #[tokio::test]
    async fn test_atomicity_update_via_replace() {
        let db = SqliteMemory::in_memory().expect("Failed to create database");

        // Insert initial record
        let record = create_memory("update-test", 1, MemoryType::Episodic, "Original content");
        db.store(record).await.expect("Initial store failed");

        // Update with same ID (INSERT OR REPLACE)
        let updated = create_memory("update-test", 2, MemoryType::Semantic, "Updated content");
        db.store(updated).await.expect("Update store failed");

        // Should still be 1 record, with updated content
        let count = db.count().await.expect("Count failed");
        assert_eq!(count, 1, "Should still have exactly one record");

        let retrieved = db.get("update-test").await.expect("Get failed").unwrap();
        assert_eq!(retrieved.content, "Updated content");
        assert_eq!(retrieved.memory_type, MemoryType::Semantic);
    }

    #[tokio::test]
    async fn test_isolation_read_committed() {
        // SQLite uses serializable isolation by default, which is stronger than read committed
        let (_temp, db) = temp_db();

        // Store a record
        let record = create_memory("isolation-1", 1, MemoryType::Episodic, "Isolation test");
        db.store(record).await.expect("Store failed");

        // Multiple reads should see consistent data
        let read1 = db.get("isolation-1").await.expect("Read 1 failed");
        let read2 = db.get("isolation-1").await.expect("Read 2 failed");

        assert_eq!(read1.unwrap().content, read2.unwrap().content);
    }

    #[tokio::test]
    async fn test_durability_persist_to_disk() {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let db_path = temp_dir.path().join("durability_test.db");

        // Create database and store record
        {
            let db = SqliteMemory::new(&db_path).expect("Failed to create database");
            let record = create_memory("durable-1", 1, MemoryType::Episodic, "Durability test");
            db.store(record).await.expect("Store failed");
        }
        // Database dropped here, should be flushed to disk

        // Reopen and verify
        {
            let db = SqliteMemory::new(&db_path).expect("Failed to reopen database");
            let retrieved = db.get("durable-1").await.expect("Get failed");
            assert!(retrieved.is_some(), "Record should persist after reopen");
            assert_eq!(retrieved.unwrap().content, "Durability test");
        }
    }

    #[tokio::test]
    async fn test_consistency_referential_integrity() {
        let db = SqliteMemory::in_memory().expect("Failed to create database");

        // Store multiple related records
        for i in 0..10 {
            let record = create_memory(
                &format!("consistent-{}", i),
                i as u64,
                MemoryType::Episodic,
                "Consistency test",
            );
            db.store(record).await.expect("Store failed");
        }

        // Count should match
        let count = db.count().await.expect("Count failed");
        assert_eq!(count, 10);

        // Delete some
        for i in 0..5 {
            db.delete(&format!("consistent-{}", i))
                .await
                .expect("Delete failed");
        }

        // Count should be updated
        let count = db.count().await.expect("Count failed");
        assert_eq!(count, 5);

        // Remaining records should be intact
        for i in 5..10 {
            let record = db
                .get(&format!("consistent-{}", i))
                .await
                .expect("Get failed");
            assert!(record.is_some(), "Record {} should exist", i);
        }
    }
}

// ============================================================================
// Section 3: Concurrent Access Tests
// ============================================================================

mod concurrent_tests {
    use super::*;

    #[tokio::test]
    async fn test_concurrent_reads() {
        let db = Arc::new(SqliteMemory::in_memory().expect("Failed to create database"));

        // Store initial data
        for i in 0..10 {
            let record = create_memory(
                &format!("concurrent-read-{}", i),
                i as u64,
                MemoryType::Episodic,
                "Concurrent read test",
            );
            db.store(record).await.expect("Store failed");
        }

        // Spawn multiple concurrent read tasks
        let mut handles = vec![];
        for _ in 0..10 {
            let db_clone = Arc::clone(&db);
            handles.push(tokio::spawn(async move {
                for i in 0..10 {
                    let result = db_clone.get(&format!("concurrent-read-{}", i)).await;
                    assert!(result.is_ok(), "Concurrent read should succeed");
                }
            }));
        }

        // Wait for all to complete
        for handle in handles {
            handle.await.expect("Task panicked");
        }
    }

    #[tokio::test]
    async fn test_concurrent_writes() {
        let db = Arc::new(SqliteMemory::in_memory().expect("Failed to create database"));

        // Spawn multiple concurrent write tasks
        let mut handles = vec![];
        for writer_id in 0..5 {
            let db_clone = Arc::clone(&db);
            handles.push(tokio::spawn(async move {
                for i in 0..20 {
                    let id = format!("writer-{}-item-{}", writer_id, i);
                    let record = create_memory(
                        &id,
                        (writer_id * 100 + i) as u64,
                        MemoryType::Working,
                        "Concurrent write",
                    );
                    db_clone
                        .store(record)
                        .await
                        .expect("Concurrent write should succeed");
                }
            }));
        }

        // Wait for all writes
        for handle in handles {
            handle.await.expect("Task panicked");
        }

        // Verify all records were written
        let count = db.count().await.expect("Count failed");
        assert_eq!(count, 100, "All 100 records should be written");
    }

    #[tokio::test]
    async fn test_concurrent_read_write() {
        let db = Arc::new(SqliteMemory::in_memory().expect("Failed to create database"));

        // Pre-populate
        for i in 0..50 {
            let record = create_memory(
                &format!("pre-{}", i),
                i as u64,
                MemoryType::Episodic,
                "Pre-populated",
            );
            db.store(record).await.expect("Pre-store failed");
        }

        // Mix of readers and writers
        let mut handles = vec![];

        // Readers
        for _ in 0..5 {
            let db_clone = Arc::clone(&db);
            handles.push(tokio::spawn(async move {
                for _ in 0..100 {
                    let query = BinaryHV::random(42);
                    let _ = db_clone.search_similar(&query, 5).await;
                }
            }));
        }

        // Writers
        for writer_id in 0..3 {
            let db_clone = Arc::clone(&db);
            handles.push(tokio::spawn(async move {
                for i in 0..30 {
                    let id = format!("concurrent-rw-{}-{}", writer_id, i);
                    let record = create_memory(
                        &id,
                        (writer_id * 100 + i) as u64,
                        MemoryType::Working,
                        "Concurrent RW",
                    );
                    let _ = db_clone.store(record).await;
                }
            }));
        }

        // Wait for all
        for handle in handles {
            handle.await.expect("Task panicked");
        }

        // Database should still be consistent
        let health = db.health_check().await.expect("Health check failed");
        assert!(health, "Database should be healthy after concurrent access");
    }

    #[tokio::test]
    async fn test_concurrent_deletes() {
        let db = Arc::new(SqliteMemory::in_memory().expect("Failed to create database"));

        // Pre-populate
        for i in 0..100 {
            let record = create_memory(
                &format!("delete-me-{}", i),
                i as u64,
                MemoryType::Working,
                "To be deleted",
            );
            db.store(record).await.expect("Store failed");
        }

        // Concurrent deletes
        let mut handles = vec![];
        for deleter_id in 0..5 {
            let db_clone = Arc::clone(&db);
            handles.push(tokio::spawn(async move {
                for i in (deleter_id * 20)..((deleter_id + 1) * 20) {
                    let _ = db_clone.delete(&format!("delete-me-{}", i)).await;
                }
            }));
        }

        for handle in handles {
            handle.await.expect("Task panicked");
        }

        // All should be deleted
        let count = db.count().await.expect("Count failed");
        assert_eq!(count, 0, "All records should be deleted");
    }
}

// ============================================================================
// Section 4: Recovery and Edge Cases
// ============================================================================

mod recovery_tests {
    use super::*;

    #[tokio::test]
    async fn test_get_nonexistent_record() {
        let db = SqliteMemory::in_memory().expect("Failed to create database");

        let result = db
            .get("nonexistent-id")
            .await
            .expect("Get should not error");
        assert!(result.is_none(), "Nonexistent record should return None");
    }

    #[tokio::test]
    async fn test_delete_nonexistent_record() {
        let db = SqliteMemory::in_memory().expect("Failed to create database");

        let deleted = db
            .delete("nonexistent-id")
            .await
            .expect("Delete should not error");
        assert!(!deleted, "Deleting nonexistent record should return false");
    }

    #[tokio::test]
    async fn test_empty_database_operations() {
        let db = SqliteMemory::in_memory().expect("Failed to create database");

        // Count on empty
        let count = db.count().await.expect("Count failed");
        assert_eq!(count, 0);

        // Search on empty
        let query = BinaryHV::random(42);
        let results = db.search_similar(&query, 10).await.expect("Search failed");
        assert!(results.is_empty());

        // Health check on empty
        let health = db.health_check().await.expect("Health check failed");
        assert!(health);
    }

    #[tokio::test]
    async fn test_search_with_no_matches() {
        let db = SqliteMemory::in_memory().expect("Failed to create database");

        // Store some records
        for i in 0..5 {
            let record = create_memory(
                &format!("search-test-{}", i),
                i as u64,
                MemoryType::Semantic,
                "Search test",
            );
            db.store(record).await.expect("Store failed");
        }

        // Search with very different encoding
        let query = BinaryHV::random(99999);
        let results = db.search_similar(&query, 3).await.expect("Search failed");

        // Should return results (even if not very similar)
        assert!(
            !results.is_empty(),
            "Search should return results even if not highly similar"
        );
    }

    #[tokio::test]
    async fn test_duplicate_id_handling() {
        let db = SqliteMemory::in_memory().expect("Failed to create database");

        // Store initial record
        let record1 = create_memory("duplicate-id", 1, MemoryType::Episodic, "First version");
        db.store(record1).await.expect("First store failed");

        // Store with same ID (should replace)
        let record2 = create_memory("duplicate-id", 2, MemoryType::Semantic, "Second version");
        db.store(record2).await.expect("Second store failed");

        // Should have only one record with second content
        let count = db.count().await.expect("Count failed");
        assert_eq!(count, 1, "Should have exactly one record");

        let retrieved = db.get("duplicate-id").await.expect("Get failed").unwrap();
        assert_eq!(retrieved.content, "Second version");
    }

    #[tokio::test]
    async fn test_special_characters_in_content() {
        let db = SqliteMemory::in_memory().expect("Failed to create database");

        let special_content =
            r#"Test with "quotes", 'apostrophes', \backslashes\, and <xml>tags</xml>"#;
        let record = create_memory("special-chars", 1, MemoryType::Semantic, special_content);

        db.store(record).await.expect("Store failed");
        let retrieved = db.get("special-chars").await.expect("Get failed").unwrap();

        assert_eq!(
            retrieved.content, special_content,
            "Special characters should be preserved"
        );
    }

    #[tokio::test]
    async fn test_very_long_content() {
        let db = SqliteMemory::in_memory().expect("Failed to create database");

        // Create 1MB of content
        let long_content = "x".repeat(1_000_000);
        let record = create_memory("long-content", 1, MemoryType::Semantic, &long_content);

        db.store(record).await.expect("Store failed");
        let retrieved = db.get("long-content").await.expect("Get failed").unwrap();

        assert_eq!(
            retrieved.content.len(),
            1_000_000,
            "Long content should be preserved"
        );
    }

    #[tokio::test]
    async fn test_rapid_store_delete_cycles() {
        let db = SqliteMemory::in_memory().expect("Failed to create database");

        for cycle in 0..100 {
            let id = format!("cycle-{}", cycle);
            let record = create_memory(&id, cycle as u64, MemoryType::Working, "Rapid cycle");

            db.store(record).await.expect("Store failed");
            let deleted = db.delete(&id).await.expect("Delete failed");
            assert!(deleted, "Delete should succeed for cycle {}", cycle);
        }

        let count = db.count().await.expect("Count failed");
        assert_eq!(count, 0, "All records should be deleted after cycles");
    }

    #[tokio::test]
    async fn test_reopen_after_many_operations() {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let db_path = temp_dir.path().join("reopen_test.db");

        // Many operations
        {
            let db = SqliteMemory::new(&db_path).expect("Failed to create database");

            for i in 0..1000 {
                let record = create_memory(
                    &format!("op-{}", i),
                    i as u64,
                    MemoryType::Episodic,
                    "Many ops test",
                );
                db.store(record).await.expect("Store failed");
            }

            // Delete half
            for i in 0..500 {
                db.delete(&format!("op-{}", i))
                    .await
                    .expect("Delete failed");
            }

            // Update some
            for i in 500..600 {
                let record = create_memory(
                    &format!("op-{}", i),
                    (i + 10000) as u64,
                    MemoryType::Semantic,
                    "Updated",
                );
                db.store(record).await.expect("Update failed");
            }
        }

        // Reopen and verify
        {
            let db = SqliteMemory::new(&db_path).expect("Failed to reopen database");

            let count = db.count().await.expect("Count failed");
            assert_eq!(count, 500, "Should have 500 records after operations");

            // Check an updated record
            let retrieved = db.get("op-550").await.expect("Get failed").unwrap();
            assert_eq!(retrieved.content, "Updated");
            assert_eq!(retrieved.memory_type, MemoryType::Semantic);
        }
    }
}

// ============================================================================
// Section 5: Performance Sanity Tests
// ============================================================================

mod performance_tests {
    use super::*;

    #[tokio::test]
    async fn test_bulk_insert_performance() {
        let db = SqliteMemory::in_memory().expect("Failed to create database");

        let start = Instant::now();
        for i in 0..1000 {
            let record = create_memory(
                &format!("perf-{}", i),
                i as u64,
                MemoryType::Working,
                "Performance test",
            );
            db.store(record).await.expect("Store failed");
        }
        let elapsed = start.elapsed();

        // Should complete in reasonable time (less than 10 seconds for 1000 inserts)
        assert!(
            elapsed < Duration::from_secs(10),
            "Bulk insert took too long: {:?}",
            elapsed
        );

        eprintln!("[Performance] 1000 inserts completed in {:?}", elapsed);
    }

    #[tokio::test]
    async fn test_search_performance() {
        let db = SqliteMemory::in_memory().expect("Failed to create database");

        // Populate
        for i in 0..1000 {
            let record = create_memory(
                &format!("search-perf-{}", i),
                i as u64,
                MemoryType::Semantic,
                "Search performance test",
            );
            db.store(record).await.expect("Store failed");
        }

        // Time searches
        let start = Instant::now();
        for i in 0..100 {
            let query = BinaryHV::random(i);
            let _ = db.search_similar(&query, 10).await.expect("Search failed");
        }
        let elapsed = start.elapsed();

        // 100 searches in under 30 seconds
        assert!(
            elapsed < Duration::from_secs(30),
            "Search performance too slow: {:?}",
            elapsed
        );

        eprintln!(
            "[Performance] 100 searches over 1000 records completed in {:?}",
            elapsed
        );
    }
}