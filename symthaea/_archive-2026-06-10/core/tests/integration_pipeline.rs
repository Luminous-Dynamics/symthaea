// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for the Symthaea consciousness pipeline.
//!
//! Tests the full process() pipeline, tick() loop, persistence
//! lifecycle, and multi-query state evolution.
//!
//! NOTE: Tests that call `Symthaea::process()` require an Ollama backend
//! (or the default LLM fallback). If Ollama is not running, these tests
//! may produce fallback responses but should still pass -- the assertions
//! are written against the structural guarantees of the pipeline rather
//! than the quality of LLM output.

use symthaea::Symthaea;
use symthaea::mind::{ContinuousMind, MindConfig};
use symthaea_core::hdc::real_hv::RealHV;

// ---------------------------------------------------------------------------
// Full pipeline tests (async, require Symthaea::new)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_symthaea_process_basic() {
    let mut sym = Symthaea::new(256, 16).await.unwrap();

    // Process a simple query
    let response = sym.process("What is 2 + 2?").await.unwrap();

    // Basic structural assertions
    assert!(!response.content.is_empty(), "Response should have content");
    assert!(
        response.confidence >= 0.0 && response.confidence <= 1.0,
        "Confidence {} should be in [0, 1]",
        response.confidence
    );
    assert!(response.safe, "Response should be safe");
    assert!(
        response.structured_thought.is_some(),
        "Should have structured thought"
    );
}

#[tokio::test]
async fn test_symthaea_consciousness_grows() {
    let mut sym = Symthaea::new(256, 16).await.unwrap();

    let queries = [
        "Hello",
        "How does memory work?",
        "Tell me about consciousness",
        "What is integration?",
    ];

    // Process multiple queries to build up working memory
    for query in &queries {
        sym.process(query).await.unwrap();
    }

    // Memory should grow as a result of processing
    let final_intro = sym.introspect();
    assert!(
        final_intro.memory_stats.short_term_count > 0,
        "Should have items in working memory after processing {} queries, got {}",
        queries.len(),
        final_intro.memory_stats.short_term_count
    );
}

#[tokio::test]
async fn test_symthaea_pause_resume() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("symthaea_test_state.json");
    let path_str = path.to_str().unwrap();

    // Create and process
    let mut sym = Symthaea::new(256, 16).await.unwrap();
    sym.process("Remember this context").await.unwrap();
    let before = sym.partnership_state();

    // Pause (persists partnership state)
    sym.pause(path_str).unwrap();

    // Resume
    let sym2 = Symthaea::resume(path_str).unwrap();
    let after = sym2.partnership_state();

    // Partnership interaction count should be preserved
    assert_eq!(
        before.interactions, after.interactions,
        "Interaction count should survive pause/resume: before={}, after={}",
        before.interactions, after.interactions
    );
}

#[tokio::test]
async fn test_symthaea_sleep_consolidation() {
    let mut sym = Symthaea::new(256, 16).await.unwrap();

    // Build up memory with several queries
    for i in 0..5 {
        sym.process(&format!("Topic {} about science", i))
            .await
            .unwrap();
    }

    let before_mem = sym.introspect().memory_stats.short_term_count;

    // Sleep runs consolidation ticks
    let report = sym.sleep().await.unwrap();

    // Sleep should have done something -- either scaled, consolidated, or
    // the memory was already small enough that no pruning occurred.
    assert!(
        report.scaled > 0 || report.consolidated > 0 || before_mem == 0,
        "Sleep should process memory (scaled={}, consolidated={}, before_mem={})",
        report.scaled,
        report.consolidated,
        before_mem
    );
}

#[tokio::test]
async fn test_multiple_queries_epistemic_status() {
    let mut sym = Symthaea::new(256, 16).await.unwrap();

    // Different query types
    let r1 = sym.process("What is 2 + 2?").await.unwrap();
    let r2 = sym.process("What will happen tomorrow?").await.unwrap();

    // Both should have structured thoughts
    assert!(
        r1.structured_thought.is_some(),
        "r1 should have structured thought"
    );
    assert!(
        r2.structured_thought.is_some(),
        "r2 should have structured thought"
    );

    // Both should be safe
    assert!(r1.safe, "r1 should be safe");
    assert!(r2.safe, "r2 should be safe");
}

#[tokio::test]
async fn test_partnership_evolves() {
    let mut sym = Symthaea::new(256, 16).await.unwrap();

    let initial = sym.partnership_state();
    assert_eq!(initial.interactions, 0, "Should start with 0 interactions");

    // Multiple interactions should evolve partnership
    for _ in 0..5 {
        sym.process("Let's explore this together").await.unwrap();
    }

    let evolved = sym.partnership_state();
    assert!(
        evolved.interactions >= 5,
        "Should have at least 5 interactions, got {}",
        evolved.interactions
    );
}

#[tokio::test]
async fn test_introspection_consistency() {
    let mut sym = Symthaea::new(256, 16).await.unwrap();

    sym.process("Hello world").await.unwrap();

    let intro = sym.introspect();
    assert!(
        intro.consciousness_level >= 0.0 && intro.consciousness_level <= 1.0,
        "Consciousness level {} should be in [0, 1]",
        intro.consciousness_level
    );
    assert!(
        intro.complexity >= 0.0,
        "Complexity {} should be non-negative",
        intro.complexity
    );
    // graph_size is usize so always >= 0; just verify it's a reasonable value
    assert!(
        intro.graph_size < 1_000_000,
        "Graph size should be reasonable, got {}",
        intro.graph_size
    );
}

// ---------------------------------------------------------------------------
// Mind-level tests (synchronous, no LLM dependency)
// ---------------------------------------------------------------------------

#[test]
fn test_mind_tick_lifecycle() {
    let config = MindConfig {
        dimension: 256,
        ..MindConfig::default()
    };
    let mut mind = ContinuousMind::new(config);
    mind.awaken();

    // Initial state
    let state = mind.snapshot();
    assert_eq!(state.tick, 0, "Tick should start at 0");
    assert!(state.is_active, "Mind should be active after awaken()");

    // Feed input and tick
    let hv = RealHV::random(256, 42);
    mind.perceive_text("test input", hv);

    // Run several ticks
    for _ in 0..20 {
        mind.tick();
    }

    let state = mind.snapshot();
    assert!(
        state.tick >= 20,
        "Should have at least 20 ticks, got {}",
        state.tick
    );
    assert!(
        !mind.working_memory().is_empty(),
        "Working memory should have at least one entry after perceive + tick"
    );
}

#[test]
fn test_mind_structured_thought_extraction() {
    let config = MindConfig {
        dimension: 256,
        ..MindConfig::default()
    };
    let mut mind = ContinuousMind::new(config);
    mind.awaken();
    mind.seed_memory();

    // Perceive and tick
    let hv = RealHV::random(256, 42);
    mind.perceive_text("What is quantum computing?", hv);
    mind.tick();

    // Extract structured thought
    let thought = mind.extract_structured_thought();

    // Thought should have valid phi and coherence
    assert!(
        thought.psi >= 0.0,
        "Phi {} should be non-negative",
        thought.psi
    );
    assert!(
        thought.coherence >= 0.0,
        "Coherence {} should be non-negative",
        thought.coherence
    );
}

#[test]
fn test_mind_working_memory_seeding() {
    let config = MindConfig {
        dimension: 256,
        ..MindConfig::default()
    };
    let mut mind = ContinuousMind::new(config);
    mind.awaken();

    assert!(
        mind.working_memory().is_empty(),
        "Working memory should be empty before seeding"
    );

    let result = mind.seed_memory();
    assert!(
        result.prototypes_seeded > 0,
        "Should seed at least one prototype"
    );
    assert!(
        mind.is_seeded(),
        "Mind should report as seeded after seed_memory()"
    );
    assert_eq!(
        mind.working_memory().len(),
        result.prototypes_seeded,
        "Working memory length should match seeded count"
    );
}

#[test]
fn test_mind_snapshot_fields() {
    let config = MindConfig {
        dimension: 256,
        ..MindConfig::default()
    };
    let mut mind = ContinuousMind::new(config);
    mind.awaken();
    mind.tick();

    let state = mind.snapshot();

    // Verify key fields are populated
    assert!(state.is_active, "Should be active");
    assert_eq!(state.tick, 1, "Should have one tick");
    assert!(
        state.time_awake_ms < 60_000,
        "Should not have been awake for a minute"
    );
    // phi and consciousness_level should both be set
    assert_eq!(
        state.psi, state.consciousness_level,
        "psi should equal consciousness_level in snapshot"
    );
}

#[test]
fn test_mind_multiple_perceptions() {
    let config = MindConfig {
        dimension: 256,
        ..MindConfig::default()
    };
    let mut mind = ContinuousMind::new(config);
    mind.awaken();

    // Feed multiple perceptions
    for seed in 0..5u64 {
        let hv = RealHV::random(256, seed);
        mind.perceive(hv);
        mind.tick();
    }

    // Should have entries in working memory (capped by capacity)
    assert!(
        !mind.working_memory().is_empty(),
        "Should have working memory after 5 perceptions"
    );
    assert!(
        mind.working_memory().len() <= mind.config().working_memory_capacity + 5,
        "Working memory should be bounded"
    );
}

// ---------------------------------------------------------------------------
// Edge case and robustness tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_symthaea_empty_query() {
    let mut sym = Symthaea::new(256, 16).await.unwrap();

    // Empty string should not crash
    let response = sym.process("").await.unwrap();
    assert!(
        response.structured_thought.is_some(),
        "Even empty input should produce a structured thought"
    );
}

#[tokio::test]
async fn test_symthaea_long_query() {
    let mut sym = Symthaea::new(256, 16).await.unwrap();

    // Very long input should not crash
    let long_input = "a ".repeat(5000);
    let response = sym.process(&long_input).await.unwrap();
    assert!(
        !response.content.is_empty(),
        "Should produce some response for long input"
    );
}

#[tokio::test]
async fn test_symthaea_pause_resume_round_trip_preserves_config() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("config_round_trip.json");
    let path_str = path.to_str().unwrap();

    let mut sym = Symthaea::new(256, 16).await.unwrap();
    assert_eq!(sym.dimension(), 256, "Dimension should be 256");

    sym.pause(path_str).unwrap();
    let sym2 = Symthaea::resume(path_str).unwrap();

    assert_eq!(
        sym2.dimension(),
        256,
        "Dimension should survive pause/resume"
    );
}

#[test]
fn test_mind_shutdown_lifecycle() {
    let config = MindConfig {
        dimension: 256,
        ..MindConfig::default()
    };
    let mut mind = ContinuousMind::new(config);
    mind.awaken();
    assert!(mind.snapshot().is_active, "Should be active after awaken");

    mind.request_shutdown();
    assert!(mind.is_shutdown_requested(), "Shutdown should be requested");
    assert!(
        !mind.snapshot().is_active,
        "Should not be active after shutdown request"
    );
}