// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! SIMD Integration Tests
//!
//! Tests BinaryHV operations which use SIMD acceleration internally.
//! SimdHV16 has been removed — all methods absorbed into BinaryHV.
//! Run with: cargo test --test simd_test -- --nocapture

use symthaea::hdc::binary_hv::BinaryHV;

#[test]
fn test_binary_hv_bind_correctness() {
    let a = BinaryHV::random(42);
    let b = BinaryHV::random(43);
    let bound = a.bind(&b);

    // XOR is self-inverse
    let recovered = bound.bind(&b);
    assert_eq!(a, recovered, "bind should be self-inverse");
}

#[test]
fn test_binary_hv_similarity_correctness() {
    let a = BinaryHV::random(42);
    let b = BinaryHV::random(43);

    let sim = a.similarity(&b);
    // Random vectors should be near 0.5
    assert!((sim - 0.5).abs() < 0.05, "Expected ~0.5, got {}", sim);
}

#[test]
fn test_binary_hv_bundle_correctness() {
    let vectors: Vec<BinaryHV> = (0..10).map(|i| BinaryHV::random(i + 100)).collect();
    let bundled = BinaryHV::bundle(&vectors);

    // Bundled should be similar to each component
    for v in &vectors {
        let sim = bundled.similarity(v);
        assert!(sim > 0.4, "Bundle should be similar to component");
    }
}

#[test]
fn test_binary_hv_batch_similarity() {
    let query = BinaryHV::random(999);
    let targets: Vec<BinaryHV> = (0..100).map(|i| BinaryHV::random(i + 300)).collect();

    let batch = BinaryHV::batch_similarity(&query, &targets);
    assert_eq!(batch.len(), 100);

    // Verify batch matches individual calls
    for (i, &sim) in batch.iter().enumerate() {
        let expected = query.similarity(&targets[i]);
        assert!(
            (sim - expected).abs() < 1e-6,
            "Batch similarity mismatch at {}",
            i
        );
    }
}

#[test]
fn test_binary_hv_top_k() {
    let query = BinaryHV::random(999);
    let targets: Vec<BinaryHV> = (0..100).map(|i| BinaryHV::random(i + 400)).collect();

    let top5 = BinaryHV::top_k_similar_in(&query, &targets, 5);
    assert_eq!(top5.len(), 5);

    // Results should be sorted descending by similarity
    for i in 1..top5.len() {
        assert!(
            top5[i - 1].1 >= top5[i].1,
            "Top-K should be sorted descending"
        );
    }
}