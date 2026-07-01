// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! LanceDB Consciousness Pipeline Integration Tests
//!
//! Tests that LanceDB storage works correctly within the consciousness pipeline:
//! memory accumulation, retrieval-based context influence, and persistence
//! across reinitialization.
//!
//! Gated behind `lancedb-backend` since these tests exercise the Lance storage backend.
#![cfg(feature = "lancedb-backend")]

mod common;

use common::prelude::*;
use symthaea::databases::{ConsciousnessDatabase, LanceMemory, MemoryRecord, MemoryType};
use symthaea::symthaea_core::hdc::binary_hv::BinaryHV;

// ============================================================================
// Test 1: Memory Accumulation
// ============================================================================

#[tokio::test]
async fn test_lancedb_memory_accumulation() {
    // Create a LanceDB instance in a temp directory so we have a real persistent store.
    let temp = TempTestDir::new().expect("Should create temp dir");
    let db_path = temp.path("lance_accumulation");
    let db = LanceMemory::new(db_path.to_str().unwrap())
        .await
        .expect("Should create LanceDB database");

    // Database should start empty.
    let initial_count = db.count().await.expect("count should succeed");
    assert_eq!(initial_count, 0, "Fresh LanceDB should be empty");

    // Store 100+ records with varied seeds, types, and psi values to simulate
    // a consciousness pipeline accumulating memories over many cycles.
    let num_records: usize = 120;
    for i in 0..num_records {
        let memory_type = match i % 4 {
            0 => MemoryType::Episodic,
            1 => MemoryType::Semantic,
            2 => MemoryType::Procedural,
            _ => MemoryType::Working,
        };
        let record = MemoryRecord {
            id: format!("accum-{i}"),
            encoding: BinaryHV::random(1000 + i as u64),
            timestamp_ms: 1700000000000 + i as u64 * 100,
            memory_type,
            content: format!("Accumulated memory content #{i}"),
            valence: ((i as f32 / num_records as f32) * 2.0) - 1.0, // spans -1..1
            arousal: (i as f32 / num_records as f32),
            psi: 0.3 + (i as f64 / num_records as f64) * 0.6, // spans 0.3..0.9
            topics: vec!["accumulation_test".to_string()],
            metadata: "{}".to_string(),
            consolidation_strength: 0.0,
            retrieval_count: 0,
        };
        db.store(record).await.expect("Should store memory");
    }

    // Verify count grew to the expected total.
    let final_count = db.count().await.expect("count should succeed");
    assert_eq!(
        final_count, num_records,
        "LanceDB should contain all {num_records} stored records"
    );

    // Verify individual records can be retrieved by ID.
    for sample_idx in [0, 50, 99, 119] {
        let id = format!("accum-{sample_idx}");
        let retrieved = db.get(&id).await.expect("get should succeed");
        assert!(
            retrieved.is_some(),
            "Record '{id}' should be retrievable by ID"
        );
        let rec = retrieved.unwrap();
        assert_eq!(rec.id, id);
        assert!(
            rec.content.contains(&format!("#{sample_idx}")),
            "Content should match for record {id}"
        );
    }

    // Verify similarity search returns results ordered by similarity.
    let query_encoding = BinaryHV::random(1042); // seed 1042 = same as record accum-42
    let results = db
        .search_similar(&query_encoding, 10)
        .await
        .expect("search_similar should succeed");
    assert!(
        !results.is_empty(),
        "Similarity search should return results"
    );
    // The top result should be the exact-match record (seed 1042 = accum-42).
    assert_eq!(
        results[0].record.id, "accum-42",
        "Top similarity hit should be the record with identical encoding"
    );
    assert!(
        results[0].similarity > 0.99,
        "Exact-match similarity should be near 1.0, got {}",
        results[0].similarity
    );

    // Verify descending similarity ordering.
    for i in 1..results.len() {
        assert!(
            results[i - 1].similarity >= results[i].similarity,
            "Results should be in descending similarity order"
        );
    }

    // Verify stats report the correct record count.
    let stats = db.stats().await.expect("stats should succeed");
    assert_eq!(
        stats.total_records, num_records,
        "Stats should reflect the correct total"
    );
}

// ============================================================================
// Test 2: Retrieval Affects Context (familiar vs novel)
// ============================================================================

#[tokio::test]
async fn test_lancedb_retrieval_affects_context() {
    let db = LanceMemory::in_memory()
        .await
        .expect("Should create in-memory LanceDB");

    // Seed the database with a cluster of "familiar" memories that share
    // a common base encoding (with small noise perturbations).
    let base_encoding = BinaryHV::random(7777);
    for i in 0..20 {
        let encoding = if i == 0 {
            base_encoding
        } else {
            // Small noise — these should remain highly similar to the base.
            base_encoding.add_noise(0.05, i as u64)
        };
        let record = MemoryRecord {
            id: format!("familiar-{i}"),
            encoding,
            timestamp_ms: 1700000000000 + i as u64,
            memory_type: MemoryType::Semantic,
            content: format!("Familiar concept variant {i}"),
            valence: 0.6,
            arousal: 0.3,
            psi: 0.7,
            topics: vec!["familiar".to_string()],
            metadata: "{}".to_string(),
            consolidation_strength: 0.0,
            retrieval_count: 0,
        };
        db.store(record)
            .await
            .expect("Should store familiar memory");
    }

    // Also store some "distant" memories with unrelated encodings.
    for i in 0..20 {
        let record = MemoryRecord {
            id: format!("distant-{i}"),
            encoding: BinaryHV::random(90000 + i as u64),
            timestamp_ms: 1700000000000 + 100 + i as u64,
            memory_type: MemoryType::Episodic,
            content: format!("Distant unrelated memory {i}"),
            valence: -0.2,
            arousal: 0.5,
            psi: 0.4,
            topics: vec!["distant".to_string()],
            metadata: "{}".to_string(),
            consolidation_strength: 0.0,
            retrieval_count: 0,
        };
        db.store(record).await.expect("Should store distant memory");
    }

    // Query with the base encoding — should retrieve familiar memories first.
    let familiar_results = db
        .search_similar(&base_encoding, 10)
        .await
        .expect("Familiar search should succeed");
    assert!(
        !familiar_results.is_empty(),
        "Should get results for familiar query"
    );
    // All top-10 hits should be from the familiar cluster (similarity > 0.85).
    let familiar_count = familiar_results
        .iter()
        .filter(|r| r.record.id.starts_with("familiar-"))
        .count();
    assert!(
        familiar_count >= 8,
        "At least 8 of top-10 results should be familiar memories, got {familiar_count}"
    );
    let avg_familiar_sim: f32 =
        familiar_results.iter().map(|r| r.similarity).sum::<f32>() / familiar_results.len() as f32;

    // Query with a completely novel encoding (unrelated to anything stored).
    let novel_encoding = BinaryHV::random(55555);
    let novel_results = db
        .search_similar(&novel_encoding, 10)
        .await
        .expect("Novel search should succeed");
    let avg_novel_sim: f32 =
        novel_results.iter().map(|r| r.similarity).sum::<f32>() / novel_results.len() as f32;

    // Familiar query should yield significantly higher average similarity than novel.
    assert!(
        avg_familiar_sim > avg_novel_sim + 0.1,
        "Familiar content should have higher avg similarity ({avg_familiar_sim:.3}) \
         than novel content ({avg_novel_sim:.3}) by at least 0.1"
    );

    // Verify filtered search works: only semantic memories for familiar query.
    let filtered = db
        .search_similar_filtered(&base_encoding, 40, Some("memory_type = 'semantic'"))
        .await
        .expect("Filtered search should succeed");
    for r in &filtered {
        assert_eq!(
            r.record.memory_type,
            MemoryType::Semantic,
            "Filtered results should only contain semantic memories"
        );
    }
}

// ============================================================================
// Test 3: Persistence Across Reinitialization
// ============================================================================

#[tokio::test]
async fn test_lancedb_persistence_across_reinit() {
    let temp = TempTestDir::new().expect("Should create temp dir");
    let db_path = temp.path("lance_persist");
    let db_path_str = db_path.to_str().unwrap();

    let stored_encoding = BinaryHV::random(31415);

    // Phase 1: Create database, store records, drop it.
    {
        let db = LanceMemory::new(db_path_str)
            .await
            .expect("Should create LanceDB (phase 1)");

        for i in 0..25u64 {
            let record = MemoryRecord {
                id: format!("persist-{i}"),
                encoding: BinaryHV::random(31415 + i),
                timestamp_ms: 1700000000000 + i * 1000,
                memory_type: MemoryType::Episodic,
                content: format!("Persistent memory #{i}"),
                valence: 0.5,
                arousal: 0.4,
                psi: 0.6 + (i as f64 * 0.01),
                topics: vec!["persistence".to_string()],
                metadata: format!(r#"{{"phase": 1, "index": {i}}}"#),
                consolidation_strength: 0.0,
                retrieval_count: 0,
            };
            db.store(record).await.expect("Should store in phase 1");
        }

        let count = db.count().await.expect("count should succeed");
        assert_eq!(count, 25, "Phase 1 should have 25 records");
        // db is dropped here
    }

    // Phase 2: Reopen the same path, verify all records survived.
    {
        let db = LanceMemory::new(db_path_str)
            .await
            .expect("Should reopen LanceDB (phase 2)");

        let count = db.count().await.expect("count should succeed");
        assert_eq!(
            count, 25,
            "Phase 2 should still have 25 records after reinit"
        );

        // Spot-check specific records by ID.
        for check_idx in [0, 12, 24] {
            let id = format!("persist-{check_idx}");
            let retrieved = db.get(&id).await.expect("get should succeed");
            assert!(
                retrieved.is_some(),
                "Record '{id}' should persist across reinit"
            );
            let rec = retrieved.unwrap();
            assert_eq!(rec.id, id);
            assert_eq!(rec.memory_type, MemoryType::Episodic);
            assert!(rec.content.contains(&format!("#{check_idx}")));
        }

        // Verify encoding integrity — the encoding for persist-0 should match
        // BinaryHV::random(31415) exactly.
        let first = db
            .get("persist-0")
            .await
            .expect("get should succeed")
            .expect("persist-0 should exist");
        assert_eq!(
            first.encoding, stored_encoding,
            "Encoding should survive persistence roundtrip bit-for-bit"
        );

        // Verify similarity search still works on the reopened database.
        let results = db
            .search_similar(&stored_encoding, 5)
            .await
            .expect("search_similar should succeed after reinit");
        assert!(!results.is_empty(), "Search should return results");
        assert_eq!(
            results[0].record.id, "persist-0",
            "Exact-match search should find persist-0"
        );
        assert!(
            results[0].similarity > 0.99,
            "Exact match similarity should be near 1.0"
        );

        // Add more records to the reopened database to confirm it is writable.
        let extra = MemoryRecord {
            id: "persist-extra".to_string(),
            encoding: BinaryHV::random(99999),
            timestamp_ms: 1700000099000,
            memory_type: MemoryType::Semantic,
            content: "Extra record added after reinit".to_string(),
            valence: 0.8,
            arousal: 0.2,
            psi: 0.75,
            topics: vec!["persistence".to_string(), "phase2".to_string()],
            metadata: r#"{"phase": 2}"#.to_string(),
            consolidation_strength: 0.0,
            retrieval_count: 0,
        };
        db.store(extra).await.expect("Should store in phase 2");

        let final_count = db.count().await.expect("count should succeed");
        assert_eq!(
            final_count, 26,
            "Should have 26 records after adding one in phase 2"
        );
    }
}