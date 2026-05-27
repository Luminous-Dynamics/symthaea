// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Neural Bridge Integration Tests
//!
//! Tests the complete pipeline from loading weights to projecting activations.
//!
//! ## Test Categories
//!
//! - **Phase 1 Tests**: EmbeddingGemma (768-dim) → HDC (legacy)
//! - **Phase 2 Tests**: BGE-M3 (1024-dim) → HDC via NeuralBridgeV2
//!
//! Run with: `cargo test --test neural_bridge_integration --features neural-bridge -- --nocapture`

use std::path::Path;

/// Test loading actual probe weights file.
///
/// Run with: cargo test --test neural_bridge_integration -- --nocapture
#[test]
fn test_load_real_weights() {
    use symthaea::perception::NeuralBridge;

    let weights_path = Path::new("models/neural_bridge/probe_weights_embeddinggemma.npy");

    if !weights_path.exists() {
        eprintln!(
            "Skipping test: weights file not found at {:?}",
            weights_path
        );
        eprintln!("Run the Python training pipeline first to generate weights.");
        return;
    }

    let bridge = NeuralBridge::load(weights_path).expect("Failed to load probe weights");

    // EmbeddingGemma produces 768-dimensional embeddings
    assert_eq!(
        bridge.input_dim(),
        768,
        "Expected input_dim=768 for embeddinggemma"
    );
    assert_eq!(
        bridge.output_dim(),
        16384,
        "Expected output_dim=16384 (HDC_DIMENSION)"
    );

    println!("Loaded NeuralBridge:");
    println!("  Input dim:  {}", bridge.input_dim());
    println!("  Output dim: {}", bridge.output_dim());
    println!("  Weights:    {} floats", bridge.weights().len());
}

/// Test projection with synthetic activation.
#[test]
fn test_projection_with_synthetic_activation() {
    use symthaea::perception::NeuralBridge;

    let weights_path = Path::new("models/neural_bridge/probe_weights_embeddinggemma.npy");

    if !weights_path.exists() {
        eprintln!("Skipping test: weights file not found");
        return;
    }

    let bridge = NeuralBridge::load(weights_path).unwrap();

    // Create synthetic activation (768 dimensions)
    let activation: Vec<f32> = (0..768).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();

    // Project to continuous
    let continuous = bridge.project(&activation).expect("Projection failed");

    assert_eq!(continuous.len(), 16384);

    // Check values are reasonable (not NaN or Inf)
    assert!(continuous.iter().all(|&v| v.is_finite()));

    // Project to bipolar
    let bipolar = bridge
        .project_to_bipolar(&activation)
        .expect("Bipolar projection failed");

    assert_eq!(bipolar.len(), 16384);

    // All values should be -1 or +1
    assert!(bipolar.iter().all(|&v| v == 1 || v == -1));

    // Check distribution is roughly balanced
    let positive = bipolar.iter().filter(|&&v| v == 1).count();
    let negative = bipolar.iter().filter(|&&v| v == -1).count();

    println!(
        "Bipolar distribution: {} positive, {} negative",
        positive, negative
    );

    // Should be somewhat balanced (allow 60-40 split)
    assert!(positive > 6000, "Too few positive values");
    assert!(negative > 6000, "Too few negative values");
}

/// Test that packed bipolar produces correct results.
#[test]
fn test_packed_bipolar_projection() {
    use symthaea::perception::NeuralBridge;

    let weights_path = Path::new("models/neural_bridge/probe_weights_embeddinggemma.npy");

    if !weights_path.exists() {
        eprintln!("Skipping test: weights file not found");
        return;
    }

    let bridge = NeuralBridge::load(weights_path).unwrap();

    // Create activation
    let activation: Vec<f32> = (0..768).map(|i| (i as f32 / 768.0 - 0.5) * 2.0).collect();

    // Project to packed
    let packed = bridge
        .project_to_packed(&activation)
        .expect("Packed projection failed");

    // Get bipolar for comparison
    let bipolar = bridge.project_to_bipolar(&activation).unwrap();

    // Verify packed matches bipolar by converting back
    let unpacked = packed.to_bipolar();

    for (i, (&original, &unpacked_val)) in bipolar.iter().zip(unpacked.iter()).enumerate() {
        assert_eq!(original, unpacked_val, "Mismatch at index {}", i);
    }

    println!("PackedBipolar matches bipolar representation");
}

/// Test that different activations produce different HDC vectors.
#[test]
fn test_activation_differentiation() {
    use symthaea::perception::NeuralBridge;

    let weights_path = Path::new("models/neural_bridge/probe_weights_embeddinggemma.npy");

    if !weights_path.exists() {
        eprintln!("Skipping test: weights file not found");
        return;
    }

    let bridge = NeuralBridge::load(weights_path).unwrap();

    // Create two different activations
    let activation1: Vec<f32> = (0..768).map(|i| (i as f32 * 0.01).sin()).collect();

    let activation2: Vec<f32> = (0..768).map(|i| (i as f32 * 0.02).cos()).collect();

    let bipolar1 = bridge.project_to_bipolar(&activation1).unwrap();
    let bipolar2 = bridge.project_to_bipolar(&activation2).unwrap();

    // Count matching bits
    let matching: usize = bipolar1
        .iter()
        .zip(bipolar2.iter())
        .filter(|(&a, &b)| a == b)
        .count();

    let similarity = matching as f64 / 16384.0;

    println!(
        "Similarity between different activations: {:.4}",
        similarity
    );

    // Different activations should produce somewhat different vectors
    // (not identical, not completely opposite)
    assert!(similarity > 0.3, "Vectors too different");
    assert!(similarity < 0.9, "Vectors too similar");
}

/// Test error handling for wrong dimensions.
#[test]
fn test_dimension_mismatch_error() {
    use symthaea::perception::NeuralBridge;

    let weights_path = Path::new("models/neural_bridge/probe_weights_embeddinggemma.npy");

    if !weights_path.exists() {
        eprintln!("Skipping test: weights file not found");
        return;
    }

    let bridge = NeuralBridge::load(weights_path).unwrap();

    // Wrong dimension (512 instead of 768)
    let wrong_activation: Vec<f32> = vec![0.0; 512];

    let result = bridge.project(&wrong_activation);
    assert!(result.is_err(), "Should fail with wrong dimension");

    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("mismatch"),
        "Error should mention dimension mismatch"
    );

    println!("Correctly rejects wrong dimensions: {}", err);
}

// =============================================================================
// Neural Bridge v2 Tests (BGE-M3, 1024-dim)
// =============================================================================

/// Test loading BGE-M3 probe weights.
#[test]
fn test_load_bge_m3_weights() {
    use symthaea::perception::NeuralBridge;

    let weights_path = Path::new("models/neural_bridge/probe_weights_bge_m3.npy");

    if !weights_path.exists() {
        eprintln!(
            "Skipping test: BGE-M3 weights not found at {:?}",
            weights_path
        );
        eprintln!("Run: python scripts/neural_bridge/train_probe_bge_m3.py");
        return;
    }

    let bridge = NeuralBridge::load(weights_path).expect("Failed to load BGE-M3 probe weights");

    // BGE-M3 produces 1024-dimensional embeddings
    assert_eq!(
        bridge.input_dim(),
        1024,
        "Expected input_dim=1024 for BGE-M3"
    );
    assert_eq!(
        bridge.output_dim(),
        16384,
        "Expected output_dim=16384 (HDC_DIMENSION)"
    );

    println!("Loaded BGE-M3 NeuralBridge:");
    println!("  Input dim:  {}", bridge.input_dim());
    println!("  Output dim: {}", bridge.output_dim());
    println!(
        "  Weights:    {} floats ({:.1} MB)",
        bridge.weights().len(),
        bridge.weights().len() as f64 * 4.0 / 1_000_000.0
    );
}

/// Test BGE-M3 projection with synthetic embedding.
#[test]
fn test_bge_m3_projection() {
    use symthaea::perception::NeuralBridge;

    let weights_path = Path::new("models/neural_bridge/probe_weights_bge_m3.npy");

    if !weights_path.exists() {
        eprintln!("Skipping test: BGE-M3 weights not found");
        return;
    }

    let bridge = NeuralBridge::load(weights_path).unwrap();

    // Create synthetic BGE-M3-like embedding (1024 dimensions, L2-normalized)
    let mut embedding: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();

    // Normalize to unit length (BGE-M3 outputs normalized embeddings)
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    for v in &mut embedding {
        *v /= norm;
    }

    // Project to HDC
    let packed = bridge
        .project_to_packed(&embedding)
        .expect("Packed projection failed");

    // Verify dimension
    let bipolar = packed.to_bipolar();
    assert_eq!(bipolar.len(), 16384);

    // Check distribution
    let positive = bipolar.iter().filter(|&&v| v == 1).count();
    let negative = bipolar.iter().filter(|&&v| v == -1).count();

    println!(
        "BGE-M3 projection - Bipolar: {} positive, {} negative",
        positive, negative
    );

    // Should be roughly balanced
    assert!(
        positive > 5000 && negative > 5000,
        "Distribution too skewed"
    );
}

/// Test that trained concepts have high similarity to their targets.
///
/// This validates the probe was trained correctly by checking that
/// synthetic embeddings similar to trained concepts produce the expected
/// HDC vectors.
#[test]
fn test_trained_concept_similarity() {
    use symthaea::perception::NeuralBridge;

    let weights_path = Path::new("models/neural_bridge/probe_weights_bge_m3.npy");

    if !weights_path.exists() {
        eprintln!("Skipping test: BGE-M3 weights not found");
        return;
    }

    let bridge = NeuralBridge::load(weights_path).unwrap();

    // Create two similar embeddings
    let embedding1: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.001).sin()).collect();

    let embedding2: Vec<f32> = (0..1024)
        .map(|i| (i as f32 * 0.001 + 0.001).sin()) // Slightly perturbed
        .collect();

    // Create a very different embedding
    let embedding3: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.05).cos()).collect();

    let packed1 = bridge.project_to_packed(&embedding1).unwrap();
    let packed2 = bridge.project_to_packed(&embedding2).unwrap();
    let packed3 = bridge.project_to_packed(&embedding3).unwrap();

    let sim_12 = packed1.xor_similarity(&packed2);
    let sim_13 = packed1.xor_similarity(&packed3);

    println!("Similarity (similar embeddings): {:.4}", sim_12);
    println!("Similarity (different embeddings): {:.4}", sim_13);

    // Similar embeddings should have higher similarity
    assert!(
        sim_12 > sim_13,
        "Similar embeddings should produce more similar HDC vectors"
    );
}

/// Test epistemic vector uncertainty tracking.
#[test]
fn test_epistemic_metadata() {
    use symthaea::perception::epistemic_vector::EpistemicSemanticVector;
    use symthaea_core::hdc::{HDC_DIMENSION, PackedBipolar};

    // Create a vector with high margin (high confidence)
    let high_margin: Vec<f32> = (0..HDC_DIMENSION)
        .map(|i| if i % 2 == 0 { 0.9 } else { -0.9 })
        .collect();

    // Create a vector with low margin (low confidence)
    let low_margin: Vec<f32> = (0..HDC_DIMENSION)
        .map(|i| if i % 2 == 0 { 0.05 } else { -0.05 })
        .collect();

    let packed_high = PackedBipolar::from_continuous(&high_margin);
    let packed_low = PackedBipolar::from_continuous(&low_margin);

    // Calculate margin-based uncertainty for low margin case
    let low_margin_ratio =
        low_margin.iter().filter(|&&m| m.abs() < 0.1).count() as f32 / HDC_DIMENSION as f32;

    println!("High margin case: margins ~0.9");
    println!(
        "Low margin case: {:.1}% dimensions near boundary",
        low_margin_ratio * 100.0
    );

    // Verify the low margin case has high uncertainty
    assert!(
        low_margin_ratio > 0.9,
        "Low margin embeddings should have >90% dimensions near boundary"
    );

    // Build epistemic vectors
    let esv_high = EpistemicSemanticVector::with_confidence(packed_high, 0.95);
    let esv_low = EpistemicSemanticVector::with_confidence(packed_low, 0.1);

    assert!(esv_high.is_high_confidence());
    assert!(esv_low.is_low_confidence());

    println!("High confidence ESV: {}", esv_high.confidence);
    println!("Low confidence ESV: {}", esv_low.confidence);
}

// =============================================================================
// End-to-End Pipeline Tests (Full BGE-M3 Model)
// Requires: --features neural-bridge
// =============================================================================

/// End-to-end test: Text → HDC via full NeuralBridgeV2 pipeline.
///
/// This test downloads BGE-M3 from HuggingFace Hub (~2.2GB first run),
/// loads the probe weights, and verifies the complete pipeline.
///
/// Run with: cargo test --test neural_bridge_integration --features neural-bridge test_e2e_text_to_hdc -- --nocapture --ignored
#[test]
#[ignore = "Downloads 2.2GB model - run explicitly with --ignored"]
#[cfg(feature = "neural-bridge")]
fn test_e2e_text_to_hdc() {
    use symthaea::perception::NeuralBridgeV2;

    let weights_path = Path::new("models/neural_bridge/probe_weights_bge_m3.npy");
    if !weights_path.exists() {
        eprintln!("Skipping test: BGE-M3 probe weights not found");
        eprintln!("Run: python scripts/neural_bridge/train_probe_bge_m3.py");
        return;
    }

    println!("Loading NeuralBridgeV2 (downloading BGE-M3 on first run)...");
    let start = std::time::Instant::now();

    let mut bridge = NeuralBridgeV2::load_default().expect("Failed to load NeuralBridgeV2");

    println!("Model loaded in {:.2}s", start.elapsed().as_secs_f64());
    println!("  Encoder dim: {}", bridge.encoder_dim());
    println!("  Probe input:  {}", bridge.probe_input_dim());
    println!("  Probe output: {}", bridge.probe_output_dim());
    println!("  Using CUDA: {}", bridge.is_cuda());

    // Test concepts
    let concepts = [
        "justice",
        "fairness",
        "What is consciousness?",
        "The neural correlates of awareness",
        "Quantum mechanics and decoherence",
    ];

    println!("\nEncoding {} concepts:", concepts.len());
    let mut hdcs = Vec::new();

    for concept in &concepts {
        let encode_start = std::time::Instant::now();
        let hdc = bridge.encode_to_hdc(concept).expect("Encoding failed");
        let elapsed = encode_start.elapsed();

        println!(
            "  {:40} → {} bits in {:.1}ms",
            concept,
            hdc.to_bipolar().len(),
            elapsed.as_secs_f64() * 1000.0
        );

        hdcs.push(hdc);
    }

    // Test semantic similarity
    println!("\nSemantic similarity matrix:");
    print!("{:20}", "");
    for (i, _) in concepts.iter().enumerate().take(5) {
        print!("{:>8}", format!("[{}]", i));
    }
    println!();

    for (i, concept) in concepts.iter().enumerate() {
        let label: String = concept.chars().take(18).collect();
        print!("[{}] {:17}", i, label);
        for j in 0..concepts.len().min(5) {
            let sim = hdcs[i].xor_similarity(&hdcs[j]);
            print!("{:8.3}", sim);
        }
        println!();
    }

    // Verify statistics
    let stats = bridge.stats();
    println!("\nPipeline statistics:");
    println!("  Total calls: {}", stats.total_calls);
    println!(
        "  HDC cache:   {} ({:.1}%)",
        stats.hdc_cache_hits,
        stats.hdc_cache_hit_rate() * 100.0
    );
    println!(
        "  Emb cache:   {} ({:.1}%)",
        stats.embedding_cache_hits,
        stats.embedding_cache_hit_rate() * 100.0
    );
    println!("  Avg encode:  {:.2}ms", stats.avg_encode_ms());
    println!("  Avg probe:   {:.2}ms", stats.avg_probe_ms());

    // Basic assertions
    assert!(stats.total_calls >= concepts.len() as u64);
    // avg_encode_ms might be 0 if all were cache hits
}

/// Test batch encoding performance.
#[test]
#[ignore = "Downloads 2.2GB model - run explicitly with --ignored"]
#[cfg(feature = "neural-bridge")]
fn test_e2e_batch_encoding() {
    use symthaea::perception::NeuralBridgeV2;

    let weights_path = Path::new("models/neural_bridge/probe_weights_bge_m3.npy");
    if !weights_path.exists() {
        eprintln!("Skipping test: probe weights not found");
        return;
    }

    let mut bridge = NeuralBridgeV2::load_default().expect("Failed to load NeuralBridgeV2");

    let texts: Vec<&str> = vec![
        "consciousness",
        "awareness",
        "sentience",
        "qualia",
        "phenomenal experience",
    ];

    println!("Batch encoding {} texts...", texts.len());
    let start = std::time::Instant::now();

    let hdcs = bridge
        .encode_batch_to_hdc(&texts)
        .expect("Batch encoding failed");

    let elapsed = start.elapsed();
    let per_text = elapsed.as_secs_f64() * 1000.0 / texts.len() as f64;

    println!(
        "Batch complete in {:.1}ms ({:.1}ms/text)",
        elapsed.as_secs_f64() * 1000.0,
        per_text
    );

    assert_eq!(hdcs.len(), texts.len());

    // All concepts should have some similarity (related to consciousness)
    for i in 0..hdcs.len() {
        for j in (i + 1)..hdcs.len() {
            let sim = hdcs[i].xor_similarity(&hdcs[j]);
            println!("  {} <-> {}: {:.3}", texts[i], texts[j], sim);
            // Related concepts should have positive similarity
            assert!(
                sim > 0.0,
                "Expected positive similarity between related concepts"
            );
        }
    }
}

/// Test epistemic metadata generation.
#[test]
#[ignore = "Downloads 2.2GB model - run explicitly with --ignored"]
#[cfg(feature = "neural-bridge")]
fn test_e2e_epistemic_encoding() {
    use symthaea::perception::NeuralBridgeV2;

    let weights_path = Path::new("models/neural_bridge/probe_weights_bge_m3.npy");
    if !weights_path.exists() {
        eprintln!("Skipping test: probe weights not found");
        return;
    }

    let mut bridge = NeuralBridgeV2::load_default().expect("Failed to load NeuralBridgeV2");

    // Test well-defined concept
    let esv_clear = bridge
        .encode_epistemic("Justice is treating people fairly.")
        .expect("Encoding failed");

    // Test ambiguous/unusual text
    let esv_unusual = bridge
        .encode_epistemic("xyzzy plugh zyqqx quantum entanglement maybe")
        .expect("Encoding failed");

    println!("Epistemic encoding results:");
    println!("  Clear concept:");
    println!("    Confidence: {:.3}", esv_clear.confidence);
    println!("    Stability:  {:?}", esv_clear.stability);
    println!(
        "    Uncertainty sources: {}",
        esv_clear.uncertainty_sources.len()
    );

    println!("  Unusual text:");
    println!("    Confidence: {:.3}", esv_unusual.confidence);
    println!("    Stability:  {:?}", esv_unusual.stability);
    println!(
        "    Uncertainty sources: {}",
        esv_unusual.uncertainty_sources.len()
    );

    // The unusual text should have more uncertainty sources
    // (though this depends on the specific text and model behavior)
    println!("\nUncertainty details for unusual text:");
    for source in &esv_unusual.uncertainty_sources {
        println!("  {:?}", source);
    }
}

/// Benchmark-style test for probe projection performance.
///
/// Note: Debug builds without BLAS optimization are slow (~500ms per projection).
/// Release builds with ndarray BLAS should achieve <1ms per projection.
#[test]
#[ignore = "Performance test - run explicitly with --ignored"]
fn test_projection_performance() {
    use std::time::Instant;
    use symthaea::perception::NeuralBridge;

    let weights_path = Path::new("models/neural_bridge/probe_weights_bge_m3.npy");

    if !weights_path.exists() {
        eprintln!("Skipping test: BGE-M3 weights not found");
        return;
    }

    let bridge = NeuralBridge::load(weights_path).unwrap();

    // Create test embedding
    let embedding: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();

    // Warm up
    for _ in 0..5 {
        let _ = bridge.project_to_packed(&embedding);
    }

    // Benchmark (fewer iterations for debug builds)
    let iterations = 10;
    let start = Instant::now();

    for _ in 0..iterations {
        let _ = bridge.project_to_packed(&embedding);
    }

    let elapsed = start.elapsed();
    let avg_us = elapsed.as_micros() as f64 / iterations as f64;

    println!("Probe projection performance:");
    println!("  Iterations: {}", iterations);
    println!("  Total time: {:?}", elapsed);
    println!(
        "  Average: {:.2} µs per projection ({:.2} ms)",
        avg_us,
        avg_us / 1000.0
    );
    println!("  Throughput: {:.1} projections/sec", 1_000_000.0 / avg_us);

    // Relaxed threshold for debug builds
    // Release builds with BLAS should achieve <1ms (1000 µs)
    // Debug builds without BLAS may take 500ms+
    #[cfg(debug_assertions)]
    let threshold_us = 1_000_000.0; // 1 second for debug

    #[cfg(not(debug_assertions))]
    let threshold_us = 10_000.0; // 10ms for release

    assert!(
        avg_us < threshold_us,
        "Projection too slow: {:.2} µs (threshold: {:.0} µs)",
        avg_us,
        threshold_us
    );
}

/// Test HDC cache effectiveness.
///
/// Verifies that repeated encoding of the same text returns in <1ms
/// vs ~380ms for uncached encoding.
#[test]
#[ignore = "Downloads 2.2GB model - run explicitly with --ignored"]
#[cfg(feature = "neural-bridge")]
fn test_e2e_cache_performance() {
    use std::time::Instant;
    use symthaea::perception::NeuralBridgeV2;

    let weights_path = Path::new("models/neural_bridge/probe_weights_bge_m3.npy");
    if !weights_path.exists() {
        eprintln!("Skipping test: probe weights not found");
        return;
    }

    let mut bridge = NeuralBridgeV2::load_default().expect("Failed to load NeuralBridgeV2");

    let test_text = "What is the meaning of consciousness?";

    // First encoding - should be slow (~350-400ms)
    let start = Instant::now();
    let hdc1 = bridge
        .encode_to_hdc(test_text)
        .expect("First encoding failed");
    let first_time = start.elapsed();

    // Second encoding - should be fast (<1ms, from cache)
    let start = Instant::now();
    let hdc2 = bridge
        .encode_to_hdc(test_text)
        .expect("Second encoding failed");
    let cached_time = start.elapsed();

    // Third encoding - also cached
    let start = Instant::now();
    let hdc3 = bridge
        .encode_to_hdc(test_text)
        .expect("Third encoding failed");
    let cached_time2 = start.elapsed();

    println!("Cache performance test:");
    println!("  First encoding:  {:?}", first_time);
    println!("  Cached (2nd):    {:?}", cached_time);
    println!("  Cached (3rd):    {:?}", cached_time2);
    println!(
        "  Speedup:         {:.0}x",
        first_time.as_secs_f64() / cached_time.as_secs_f64()
    );

    // Verify cache returns identical results
    assert_eq!(
        hdc1.to_bipolar(),
        hdc2.to_bipolar(),
        "Cached result differs from original!"
    );
    assert_eq!(
        hdc2.to_bipolar(),
        hdc3.to_bipolar(),
        "Cached results differ!"
    );

    // Verify cache stats
    let stats = bridge.stats();
    println!("\nCache statistics:");
    println!("  Total calls:     {}", stats.total_calls);
    println!("  HDC cache hits:  {}", stats.hdc_cache_hits);
    println!(
        "  HDC hit rate:    {:.1}%",
        stats.hdc_cache_hit_rate() * 100.0
    );

    // Should have 2 HDC cache hits (calls 2 and 3)
    assert_eq!(stats.hdc_cache_hits, 2, "Expected 2 HDC cache hits");
    assert_eq!(stats.total_calls, 3, "Expected 3 total calls");

    // Cached retrieval should be at least 100x faster
    let speedup = first_time.as_secs_f64() / cached_time.as_secs_f64();
    assert!(
        speedup > 100.0,
        "Cache speedup only {:.0}x (expected >100x)",
        speedup
    );

    // Cached retrieval should be under 1ms
    assert!(
        cached_time.as_millis() < 1,
        "Cached retrieval took {:?} (expected <1ms)",
        cached_time
    );

    println!(
        "\n✓ Cache is working: {:.0}x speedup, <1ms retrieval",
        speedup
    );
}