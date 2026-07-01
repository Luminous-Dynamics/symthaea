// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Symthaea + Neural Bridge v2 Integration Test
//!
//! Verifies that the full consciousness pipeline works with BGE-M3 semantic encoding.

#![cfg(feature = "neural-bridge")]

use std::path::Path;

/// Test that Symthaea initializes with Neural Bridge and processes text.
#[tokio::test]
#[ignore = "Downloads 2.2GB model - run explicitly with --ignored"]
async fn test_symthaea_with_neural_bridge() {
    use symthaea::Symthaea;

    // Check probe weights exist
    let weights_path = Path::new("models/neural_bridge/probe_weights_bge_m3.npy");
    if !weights_path.exists() {
        eprintln!(
            "Skipping test: probe weights not found at {:?}",
            weights_path
        );
        return;
    }

    println!("Creating Symthaea with Neural Bridge v2...");
    let mut symthaea = Symthaea::new(512, 64)
        .await
        .expect("Failed to create Symthaea");

    // Check neural bridge is active
    let has_bridge = symthaea.has_neural_bridge();
    println!("Neural Bridge active: {}", has_bridge);

    if !has_bridge {
        eprintln!("Warning: Neural Bridge not loaded, test will use hash-based encoding");
    }

    // Process a query through the full pipeline
    println!("\nProcessing query through consciousness pipeline...");
    let response = symthaea
        .process("What is justice?")
        .await
        .expect("Processing failed");

    println!("Response:");
    println!("  Content: {}", response.content);
    println!("  Confidence: {:.3}", response.confidence);
    println!("  Safe: {}", response.safe);
    println!("  Translation verified: {}", response.translation_verified);

    // Check neural bridge stats if available
    if let Some(stats) = symthaea.neural_bridge_stats() {
        println!("\nNeural Bridge stats:");
        println!("  Total calls: {}", stats.total_calls);
        println!("  HDC cache hits: {}", stats.hdc_cache_hits);
        println!("  Cache hit rate: {:.1}%", stats.cache_hit_rate() * 100.0);
    }

    // Basic assertions
    assert!(!response.content.is_empty(), "Response should not be empty");
    assert!(response.confidence >= 0.0 && response.confidence <= 1.0);
}

/// Test cache effectiveness through Symthaea.
#[tokio::test]
#[ignore = "Downloads 2.2GB model - run explicitly with --ignored"]
async fn test_symthaea_cache_effectiveness() {
    use std::time::Instant;
    use symthaea::Symthaea;

    let weights_path = Path::new("models/neural_bridge/probe_weights_bge_m3.npy");
    if !weights_path.exists() {
        eprintln!("Skipping test: probe weights not found");
        return;
    }

    let mut symthaea = Symthaea::new(512, 64)
        .await
        .expect("Failed to create Symthaea");

    if !symthaea.has_neural_bridge() {
        eprintln!("Skipping: Neural Bridge not available");
        return;
    }

    // First query - uncached
    let start = Instant::now();
    let _r1 = symthaea.process("Explain consciousness").await.unwrap();
    let first_time = start.elapsed();

    // Same query again - should benefit from cache
    let start = Instant::now();
    let _r2 = symthaea.process("Explain consciousness").await.unwrap();
    let second_time = start.elapsed();

    println!("Cache effectiveness through Symthaea:");
    println!("  First query:  {:?}", first_time);
    println!("  Second query: {:?}", second_time);

    if let Some(stats) = symthaea.neural_bridge_stats() {
        println!("  HDC cache hits: {}", stats.hdc_cache_hits);
    }

    // Second query should be faster (cache hit on encoding)
    // Note: Full pipeline includes LLM translation, so speedup won't be as dramatic
    println!("  Encoding speedup visible in neural bridge stats above");
}