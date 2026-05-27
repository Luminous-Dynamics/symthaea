// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Neural Bridge Phase 2 End-to-End Pipeline Test
//!
//! Tests the semantic -> HDC projection pipeline using synthetic embeddings
//! (simulating BGE-M3 1024-dim output) through a random projection matrix
//! to produce 16384-dim HDC vectors.
//!
//! This test is self-contained: no model downloads or weight files needed.
//!
//! Run with:
//!   CARGO_TARGET_DIR=/tmp/symthaea-target cargo test -p symthaea --test neural_bridge_e2e -- --nocapture

use symthaea::perception::NeuralBridge;
use symthaea_core::hdc::HDC_DIMENSION;

const INPUT_DIM: usize = 1024; // BGE-M3 output dimension

/// Helper: generate a deterministic pseudo-random projection matrix.
/// Uses a simple LCG seeded by `seed` so results are reproducible.
fn make_random_weights(seed: u64) -> Vec<f32> {
    let total = HDC_DIMENSION * INPUT_DIM;
    let mut weights = Vec::with_capacity(total);
    let mut state = seed;
    for _ in 0..total {
        // Simple LCG: state = state * 6364136223846793005 + 1
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        // Map to [-1, 1) centered at 0, then scale for JL-like projection
        let bits = (state >> 33) as i64 - (1i64 << 30); // center around 0
        let val = (bits as f32) / (1u32 << 30) as f32;
        weights.push(val / (INPUT_DIM as f32).sqrt());
    }
    weights
}

/// Helper: create a mock 1024-dim embedding from a simple pattern.
/// `base_freq` controls the frequency of the sine pattern.
/// `offset` adds a phase shift.
/// The result is L2-normalized to simulate BGE-M3 output.
fn make_mock_embedding(base_freq: f32, offset: f32) -> Vec<f32> {
    let mut emb: Vec<f32> = (0..INPUT_DIM)
        .map(|i| (i as f32 * base_freq + offset).sin())
        .collect();
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    for v in &mut emb {
        *v /= norm;
    }
    emb
}

/// Cosine similarity between two packed bipolar vectors (via their bipolar expansions).
fn cosine_similarity(a: &[i8], b: &[i8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let dot: i64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| x as i64 * y as i64)
        .sum();
    let norm_a: f64 = (a.iter().map(|&x| (x as i64) * (x as i64)).sum::<i64>() as f64).sqrt();
    let norm_b: f64 = (b.iter().map(|&x| (x as i64) * (x as i64)).sum::<i64>() as f64).sqrt();
    dot as f64 / (norm_a * norm_b)
}

#[test]
fn test_e2e_mock_embedding_to_hdc() {
    println!("=== Neural Bridge Phase 2 E2E Test ===");
    println!("Pipeline: 1024-dim mock embedding -> NeuralBridge -> 16384-dim HDC vector\n");

    // Step 1: Create a random projection matrix (deterministic seed)
    let weights = make_random_weights(42);
    let bridge = NeuralBridge::from_weights(weights, INPUT_DIM, HDC_DIMENSION)
        .expect("Failed to create NeuralBridge from synthetic weights");

    assert_eq!(bridge.input_dim(), INPUT_DIM);
    assert_eq!(bridge.output_dim(), HDC_DIMENSION);
    println!(
        "Bridge created: {}->dim -> {}-dim",
        bridge.input_dim(),
        bridge.output_dim()
    );

    // Step 2: Create a mock 1024-dim embedding (simulating BGE-M3 output)
    let embedding = make_mock_embedding(0.01, 0.0);
    assert_eq!(embedding.len(), INPUT_DIM);

    // Verify it's L2-normalized
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-5,
        "Embedding should be L2-normalized, got norm={}",
        norm
    );
    println!(
        "Mock embedding: {} dims, L2 norm = {:.6}",
        embedding.len(),
        norm
    );

    // Step 3: Project through neural bridge to continuous HDC space
    let continuous = bridge
        .project(&embedding)
        .expect("Projection to continuous failed");
    assert_eq!(continuous.len(), HDC_DIMENSION);
    assert!(
        continuous.iter().all(|v| v.is_finite()),
        "All output values must be finite"
    );

    let magnitude: f32 = continuous.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!(
        "Continuous output: {} dims, magnitude = {:.4}",
        continuous.len(),
        magnitude
    );
    assert!(magnitude > 0.0, "Output magnitude must be positive");

    // Step 4: Project to packed bipolar HDC
    let packed = bridge
        .project_to_packed(&embedding)
        .expect("Packed projection failed");
    let bipolar = packed.to_bipolar();
    assert_eq!(bipolar.len(), HDC_DIMENSION);

    // All values must be -1 or +1
    assert!(
        bipolar.iter().all(|&v| v == 1 || v == -1),
        "Bipolar values must be -1 or +1"
    );

    let positive = bipolar.iter().filter(|&&v| v == 1).count();
    let negative = bipolar.iter().filter(|&&v| v == -1).count();
    println!(
        "Bipolar output: {} positive, {} negative (total {})",
        positive,
        negative,
        bipolar.len()
    );

    // Distribution should be roughly balanced (within 60/40 split)
    let balance_ratio = positive.min(negative) as f64 / (positive + negative) as f64;
    assert!(
        balance_ratio > 0.35,
        "Bipolar distribution too skewed: {:.1}% minority",
        balance_ratio * 100.0
    );

    println!("\n--- Basic pipeline: PASS ---\n");
}

#[test]
fn test_e2e_similar_inputs_produce_similar_hdcs() {
    println!("=== Similar Input Similarity Test ===\n");

    let weights = make_random_weights(42);
    let bridge = NeuralBridge::from_weights(weights, INPUT_DIM, HDC_DIMENSION)
        .expect("Failed to create NeuralBridge");

    // Create two very similar embeddings (tiny perturbation)
    let emb_a = make_mock_embedding(0.01, 0.0);
    let emb_b = make_mock_embedding(0.01, 0.0001); // Very small phase shift

    // Verify the input embeddings are indeed similar
    let input_cos: f64 = emb_a
        .iter()
        .zip(emb_b.iter())
        .map(|(a, b)| (*a as f64) * (*b as f64))
        .sum();
    println!("Input embedding cosine similarity: {:.6}", input_cos);
    assert!(input_cos > 0.99, "Test inputs should be very similar");

    let packed_a = bridge.project_to_packed(&emb_a).unwrap();
    let packed_b = bridge.project_to_packed(&emb_b).unwrap();

    let bip_a = packed_a.to_bipolar();
    let bip_b = packed_b.to_bipolar();

    let cos_sim = cosine_similarity(&bip_a, &bip_b);
    println!("HDC output cosine similarity: {:.6}", cos_sim);

    // XOR similarity (Hamming-based)
    let xor_sim = packed_a.xor_similarity(&packed_b);
    println!("HDC XOR similarity: {:.6}", xor_sim);

    assert!(
        cos_sim > 0.8,
        "Similar inputs should produce similar HDC vectors (cosine > 0.8), got {:.4}",
        cos_sim
    );
    println!(
        "\n--- Similar inputs test: PASS (cosine={:.4}) ---\n",
        cos_sim
    );
}

#[test]
fn test_e2e_dissimilar_inputs_produce_dissimilar_hdcs() {
    println!("=== Dissimilar Input Similarity Test ===\n");

    let weights = make_random_weights(42);
    let bridge = NeuralBridge::from_weights(weights, INPUT_DIM, HDC_DIMENSION)
        .expect("Failed to create NeuralBridge");

    // Create two very different embeddings
    let emb_a = make_mock_embedding(0.01, 0.0);
    let emb_c = make_mock_embedding(0.07, 2.5); // Very different frequency and phase

    // Verify inputs are dissimilar
    let input_cos: f64 = emb_a
        .iter()
        .zip(emb_c.iter())
        .map(|(a, b)| (*a as f64) * (*b as f64))
        .sum();
    println!("Input embedding cosine similarity: {:.6}", input_cos);

    let packed_a = bridge.project_to_packed(&emb_a).unwrap();
    let packed_c = bridge.project_to_packed(&emb_c).unwrap();

    let bip_a = packed_a.to_bipolar();
    let bip_c = packed_c.to_bipolar();

    let cos_sim = cosine_similarity(&bip_a, &bip_c);
    println!("HDC output cosine similarity: {:.6}", cos_sim);

    let xor_sim = packed_a.xor_similarity(&packed_c);
    println!("HDC XOR similarity: {:.6}", xor_sim);

    // For sign-binarized random projections, the JL guarantee is approximate.
    // Dissimilar inputs (input cosine near 0 or negative) should produce
    // HDC vectors with cosine significantly lower than similar pairs.
    // We use a relaxed threshold: |cosine| < 0.5 for dissimilar inputs,
    // acknowledging that sign() binarization can amplify small correlations.
    assert!(
        cos_sim.abs() < 0.5,
        "Dissimilar inputs should produce dissimilar HDC vectors (|cosine| < 0.5), got {:.4}",
        cos_sim
    );
    println!(
        "\n--- Dissimilar inputs test: PASS (cosine={:.4}) ---\n",
        cos_sim
    );
}

#[test]
fn test_e2e_similarity_ordering_preserved() {
    println!("=== Similarity Ordering Preservation Test ===\n");

    let weights = make_random_weights(42);
    let bridge = NeuralBridge::from_weights(weights, INPUT_DIM, HDC_DIMENSION)
        .expect("Failed to create NeuralBridge");

    // Three embeddings: A is very similar to B, A is dissimilar to C
    let emb_a = make_mock_embedding(0.01, 0.0);
    let emb_b = make_mock_embedding(0.01, 0.0001); // Near-identical to A
    let emb_c = make_mock_embedding(0.07, 2.5); // Very different from A

    let packed_a = bridge.project_to_packed(&emb_a).unwrap();
    let packed_b = bridge.project_to_packed(&emb_b).unwrap();
    let packed_c = bridge.project_to_packed(&emb_c).unwrap();

    let sim_ab = packed_a.xor_similarity(&packed_b);
    let sim_ac = packed_a.xor_similarity(&packed_c);

    println!("XOR similarity A<->B (similar): {:.4}", sim_ab);
    println!("XOR similarity A<->C (different): {:.4}", sim_ac);

    assert!(
        sim_ab > sim_ac,
        "Similar pairs must have higher similarity than dissimilar pairs: AB={:.4} vs AC={:.4}",
        sim_ab,
        sim_ac
    );

    println!(
        "\n--- Ordering preservation: PASS (AB={:.4} > AC={:.4}) ---\n",
        sim_ab, sim_ac
    );
}