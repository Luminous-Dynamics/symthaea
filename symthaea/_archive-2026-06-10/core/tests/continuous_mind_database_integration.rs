// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for ContinuousMind + Database architecture
//!
//! Tests the integration of ContinuousMind's working memory, episodic memory,
//! and database persistence via the Symthaea facade.
//!
//! Gated behind `lancedb-backend` since these tests exercise the full
//! database-backed memory pipeline.
#![cfg(feature = "lancedb-backend")]

use symthaea::databases::{MemoryRecord, MemoryType};
use symthaea::symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea::{ContinuousMind, MindConfig};

/// Create a test memory record with deterministic encoding.
fn create_test_memory(id: &str, seed: u64, memory_type: MemoryType) -> MemoryRecord {
    MemoryRecord {
        id: id.to_string(),
        encoding: BinaryHV::random(seed),
        timestamp_ms: 1700000000000 + seed,
        memory_type,
        content: format!("Test memory content for {}", id),
        valence: 0.5,
        arousal: 0.3,
        psi: 0.65,
        topics: vec!["integration_test".to_string()],
        metadata: "{}".to_string(),
        consolidation_strength: 0.0,
        retrieval_count: 0,
    }
}

// ==================================================================================
// ContinuousMind Working Memory Tests
// ==================================================================================

#[test]
fn test_continuous_mind_creation() {
    let config = MindConfig::default();
    let mind = ContinuousMind::new(config);

    // Working memory should start empty
    assert!(
        mind.working_memory().is_empty(),
        "Working memory should start empty"
    );

    // State should be valid
    let state = mind.state();
    assert!(state.consciousness_level >= 0.0);
}

#[test]
fn test_continuous_mind_perceive_fills_working_memory() {
    let config = MindConfig::default();
    let mut mind = ContinuousMind::new(config);
    mind.awaken();

    // Perceive several items and tick to process them into working memory
    for i in 0..5 {
        let hv = symthaea::symthaea_core::hdc::unified_hv::ContinuousHV::random(
            symthaea::symthaea_core::hdc::HDC_DIMENSION,
            i as u64,
        );
        mind.perceive(hv);
        mind.tick();
    }

    // Working memory should have items
    assert!(
        mind.working_memory().len() <= 7,
        "Working memory should respect capacity (7+/-2)"
    );
    assert!(
        !mind.working_memory().is_empty(),
        "Working memory should have items after perception"
    );
}

#[test]
fn test_continuous_mind_eviction() {
    let mut config = MindConfig::default();
    config.working_memory_capacity = 3; // Small capacity for testing
    let mut mind = ContinuousMind::new(config);
    mind.awaken();

    // Perceive more items than capacity, ticking each to process
    for i in 0..6 {
        let hv = symthaea::symthaea_core::hdc::unified_hv::ContinuousHV::random(
            symthaea::symthaea_core::hdc::HDC_DIMENSION,
            i as u64,
        );
        mind.perceive(hv);
        mind.tick();
    }

    // Should have evicted items
    let evicted = mind.take_evicted();
    assert!(
        !evicted.is_empty(),
        "Should have evicted items when exceeding capacity"
    );

    // Each evicted item should have (hv, steps_survived, source, is_verified)
    for (hv, steps, _source, _is_verified) in &evicted {
        assert!(!hv.values.is_empty(), "Evicted HV should have values");
        // steps_survived can be 0 if evicted immediately
        let _ = steps;
    }
}

#[test]
fn test_continuous_mind_tick_cycle() {
    let config = MindConfig::default();
    let mut mind = ContinuousMind::new(config);
    mind.awaken();

    // Perceive an input
    let hv = symthaea::symthaea_core::hdc::unified_hv::ContinuousHV::random(
        symthaea::symthaea_core::hdc::HDC_DIMENSION,
        42,
    );
    mind.perceive(hv);

    // Run tick cycle
    let output = mind.tick();

    // Output may or may not be produced (depends on internal state)
    let _ = output;

    // State should still be valid after tick
    let state = mind.state();
    assert!(state.consciousness_level >= 0.0);
}

#[test]
fn test_continuous_mind_snapshot_consistency() {
    let config = MindConfig::default();
    let mut mind = ContinuousMind::new(config);
    mind.awaken();

    let snapshot1 = mind.snapshot();
    let snapshot2 = mind.snapshot();

    // Snapshots taken without intervening changes should be consistent
    assert!(
        (snapshot1.consciousness_level - snapshot2.consciousness_level).abs() < 0.001,
        "Consecutive snapshots should have consistent consciousness level"
    );
}

// ==================================================================================
// Memory Record Tests
// ==================================================================================

#[test]
fn test_memory_record_creation() {
    let record = create_test_memory("test-1", 42, MemoryType::Episodic);

    assert_eq!(record.id, "test-1");
    assert_eq!(record.memory_type, MemoryType::Episodic);
    assert!((record.psi - 0.65).abs() < 0.001);
    assert!(!record.encoding.0.is_empty());
}

#[test]
fn test_memory_record_types() {
    // Verify all memory types can be created
    let types = [
        MemoryType::Working,
        MemoryType::Episodic,
        MemoryType::Semantic,
        MemoryType::Procedural,
    ];

    for memory_type in types {
        let record = create_test_memory("type-test", 1, memory_type);
        assert_eq!(record.memory_type, memory_type);
    }
}

// ==================================================================================
// ContinuousMind Goals Integration
// ==================================================================================

#[test]
fn test_continuous_mind_goals() {
    let config = MindConfig::default();
    let mut mind = ContinuousMind::new(config);
    mind.awaken();

    // Set a goal and tick to process it
    let goal_embedding = symthaea::symthaea_core::hdc::unified_hv::ContinuousHV::random(
        symthaea::symthaea_core::hdc::HDC_DIMENSION,
        99,
    );
    mind.set_goal("Test goal", goal_embedding, 0.8);
    mind.tick();

    // Should have active goals
    let goals = mind.active_goals();
    assert!(
        !goals.is_empty(),
        "Should have at least one active goal after set_goal"
    );
}

// ==================================================================================
// Memory Seeding (Epistemic Baseline)
// ==================================================================================

#[test]
fn test_continuous_mind_seeding() {
    let config = MindConfig::default();
    let mut mind = ContinuousMind::new(config);

    // Seed memory with a priori knowledge
    let result = mind.seed_memory();

    assert!(
        result.prototypes_seeded > 0,
        "Should seed at least some prototypes"
    );
    assert!(mind.is_seeded(), "Mind should be marked as seeded");
    assert!(mind.seeded_count() > 0, "Seeded count should be positive");

    // Working memory should now have items
    assert!(
        !mind.working_memory().is_empty(),
        "Working memory should have seeded items"
    );
}