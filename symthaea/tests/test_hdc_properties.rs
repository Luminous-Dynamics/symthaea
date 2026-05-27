// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # HDC Property Tests
//!
//! Property-based tests for hyperdimensional computing operations to ensure
//! mathematical invariants hold across random inputs.

use symthaea::hdc::{BinaryHV, HDC_DIMENSION, HdcContext};

/// Test: Binding is its own inverse (XOR property)
/// bind(bind(a, b), b) ≈ a (within similarity threshold)
#[test]
fn test_bind_is_self_inverse() {
    for seed_base in 0..10 {
        let a = BinaryHV::random(seed_base);
        let b = BinaryHV::random(seed_base + 100);

        // bind(a, b) then bind with b again should approximate a
        let bound = a.bind(&b);
        let unbound = bound.bind(&b);

        // For binary XOR, this should be exactly equal
        // Use similarity - should be 1.0 for identical vectors
        let sim = a.similarity(&unbound);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Bind self-inverse property failed: similarity={} (seed_base={})",
            sim,
            seed_base
        );
    }
}

/// Test: Binding is commutative
/// bind(a, b) == bind(b, a)
#[test]
fn test_bind_is_commutative() {
    for seed_base in 0..10 {
        let a = BinaryHV::random(seed_base * 7);
        let b = BinaryHV::random(seed_base * 7 + 50);

        let ab = a.bind(&b);
        let ba = b.bind(&a);

        // Should be exactly equal for XOR binding
        let sim = ab.similarity(&ba);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Bind commutativity failed: similarity={} (seed_base={})",
            sim,
            seed_base
        );
    }
}

/// Test: Bundling preserves similarity to components
/// bundle([a, b, c]) should be similar to each component
#[test]
fn test_bundle_preserves_component_similarity() {
    for seed_base in 0..5 {
        let a = BinaryHV::random(seed_base * 11);
        let b = BinaryHV::random(seed_base * 11 + 10);
        let c = BinaryHV::random(seed_base * 11 + 20);

        // Bundle all three
        let abc = BinaryHV::bundle(&[a, b, c]);

        // Bundle should be more similar to each component than random
        // Random similarity is ~0.5, bundled should be > 0.5
        let sim_a = abc.similarity(&a);
        let sim_b = abc.similarity(&b);
        let sim_c = abc.similarity(&c);

        assert!(
            sim_a > 0.55,
            "Bundle should be similar to component a: sim={} (seed_base={})",
            sim_a,
            seed_base
        );
        assert!(
            sim_b > 0.55,
            "Bundle should be similar to component b: sim={} (seed_base={})",
            sim_b,
            seed_base
        );
        assert!(
            sim_c > 0.55,
            "Bundle should be similar to component c: sim={} (seed_base={})",
            sim_c,
            seed_base
        );
    }
}

/// Test: Self-similarity is 1.0
#[test]
fn test_self_similarity_is_one() {
    for seed in 0..20 {
        let hv = BinaryHV::random(seed);
        let sim = hv.similarity(&hv);

        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Self-similarity should be 1.0, got {} (seed={})",
            sim,
            seed
        );
    }
}

/// Test: Different random vectors have similarity around 0.5
/// (half the bits match by chance in high dimensions)
#[test]
fn test_random_vectors_half_similar() {
    let mut total_similarity: f64 = 0.0;
    let mut count = 0;

    for i in 0..20u64 {
        for j in (i + 1)..20u64 {
            let a = BinaryHV::random(i);
            let b = BinaryHV::random(j);
            total_similarity += a.similarity(&b) as f64;
            count += 1;
        }
    }

    let avg_similarity = total_similarity / count as f64;

    // Random high-dimensional binary vectors should have similarity ~0.5
    // (half the bits match by chance)
    assert!(
        (avg_similarity - 0.5).abs() < 0.05,
        "Random vectors should have ~0.5 similarity, got {}",
        avg_similarity
    );
}

/// Test: Permutation is cyclic
/// permute(v, n); permute(_, n) cycles through positions
#[test]
fn test_permute_cyclic() {
    let original = BinaryHV::random(42);
    let mut current = original;

    // Permute by dimension should return to original
    for _ in 0..HDC_DIMENSION {
        current = current.permute(1);
    }

    // Should be back to original
    let sim = original.similarity(&current);
    assert!(
        (sim - 1.0).abs() < 1e-6,
        "Full cycle permutation should return original, similarity = {}",
        sim
    );
}

/// Test: Single permutation changes the vector
#[test]
fn test_permute_changes_vector() {
    let original = BinaryHV::random(42);
    let permuted = original.permute(1);

    // Permuted vector should be different (similarity < 1.0)
    let sim = original.similarity(&permuted);
    assert!(
        sim < 0.99,
        "Permuted vector should be different from original, similarity = {}",
        sim
    );
}

/// Test: Arena allocation provides correct results
#[test]
fn test_arena_bind_correctness() {
    let ctx = HdcContext::new();

    // Create test vectors
    let a: Vec<i8> = (0..HDC_DIMENSION)
        .map(|i| if i % 2 == 0 { 1 } else { -1 })
        .collect();
    let b: Vec<i8> = (0..HDC_DIMENSION)
        .map(|i| if i % 3 == 0 { 1 } else { -1 })
        .collect();

    let result = ctx.bind(&a, &b);

    // Verify element-wise multiplication
    for i in 0..HDC_DIMENSION {
        let expected = a[i] * b[i];
        assert_eq!(
            result[i], expected,
            "Arena bind mismatch at index {}: expected {}, got {}",
            i, expected, result[i]
        );
    }
}

/// Test: Arena bundle produces correct threshold
#[test]
fn test_arena_bundle_threshold() {
    let ctx = HdcContext::new();

    // Create vectors that when bundled should produce predictable results
    let all_positive: Vec<i8> = vec![1; HDC_DIMENSION];
    let all_negative: Vec<i8> = vec![-1; HDC_DIMENSION];

    // Bundle: 2 positive + 1 negative = positive majority
    let result = ctx.bundle(&[&all_positive, &all_positive, &all_negative]);

    // All should be positive (2 votes for 1, 1 vote for -1)
    assert!(
        result.iter().all(|&x| x == 1),
        "Bundle with positive majority should yield all 1s"
    );
}

/// Test: Phi is non-negative
#[test]
fn test_phi_is_nonnegative() {
    use symthaea::hdc::unified_hv::ContinuousHV;
    use symthaea::phi_engine::PhiEngine;

    let engine = PhiEngine::auto();

    // Test with various topology sizes
    for n in 2..8 {
        let hvs: Vec<ContinuousHV> = (0..n)
            .map(|i| ContinuousHV::random(HDC_DIMENSION, i as u64))
            .collect();

        let result = engine.compute(&hvs);

        assert!(
            result.phi >= 0.0,
            "Phi should be non-negative, got {} for n={}",
            result.phi,
            n
        );
    }
}

/// Test: Phi symmetry
/// phi(union(A, B)) == phi(union(B, A))
#[test]
fn test_phi_is_symmetric() {
    use symthaea::hdc::unified_hv::ContinuousHV;
    use symthaea::phi_engine::PhiEngine;

    let engine = PhiEngine::auto();

    let a: Vec<ContinuousHV> = (0..3)
        .map(|i| ContinuousHV::random(HDC_DIMENSION, i))
        .collect();

    let b: Vec<ContinuousHV> = (10..13)
        .map(|i| ContinuousHV::random(HDC_DIMENSION, i))
        .collect();

    // A then B
    let mut ab: Vec<ContinuousHV> = a.clone();
    ab.extend(b.clone());

    // B then A
    let mut ba: Vec<ContinuousHV> = b.clone();
    ba.extend(a.clone());

    let phi_ab = engine.compute(&ab);
    let phi_ba = engine.compute(&ba);

    // Order shouldn't matter for Phi
    assert!(
        (phi_ab.phi - phi_ba.phi).abs() < 1e-6,
        "Phi should be symmetric: phi(A+B)={}, phi(B+A)={}",
        phi_ab.phi,
        phi_ba.phi
    );
}

/// Test: Consistency - same input produces same output
#[test]
fn test_phi_is_deterministic() {
    use symthaea::hdc::unified_hv::ContinuousHV;
    use symthaea::phi_engine::PhiEngine;

    let engine = PhiEngine::auto();

    let hvs: Vec<ContinuousHV> = (0..5)
        .map(|i| ContinuousHV::random(HDC_DIMENSION, i))
        .collect();

    let phi1 = engine.compute(&hvs);
    let phi2 = engine.compute(&hvs);

    assert!(
        (phi1.phi - phi2.phi).abs() < 1e-10,
        "Phi should be deterministic: first={}, second={}",
        phi1.phi,
        phi2.phi
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// ContinuousHV Property Tests
// ═══════════════════════════════════════════════════════════════════════════════

/// Test: ContinuousHV self-similarity is 1.0 (cosine similarity identity)
#[test]
fn test_continuous_self_similarity_is_one() {
    use symthaea::hdc::unified_hv::ContinuousHV;

    for seed in 0..20 {
        let hv = ContinuousHV::random(HDC_DIMENSION, seed);
        let sim = hv.similarity(&hv);

        assert!(
            (sim - 1.0).abs() < 1e-5,
            "ContinuousHV self-similarity should be 1.0, got {} (seed={})",
            sim,
            seed
        );
    }
}

/// Test: Random ContinuousHV vectors are nearly orthogonal in high dimensions
/// (cosine similarity ≈ 0 by concentration of measure)
#[test]
fn test_continuous_random_near_orthogonal() {
    use symthaea::hdc::unified_hv::ContinuousHV;

    let mut total_sim: f64 = 0.0;
    let mut count = 0;

    for i in 0..15u64 {
        for j in (i + 1)..15u64 {
            let a = ContinuousHV::random(HDC_DIMENSION, i);
            let b = ContinuousHV::random(HDC_DIMENSION, j);
            total_sim += a.similarity(&b).abs() as f64;
            count += 1;
        }
    }

    let avg_abs_sim = total_sim / count as f64;
    // In 16,384 dimensions, |cos(θ)| ≈ O(1/√d) ≈ 0.0078
    assert!(
        avg_abs_sim < 0.05,
        "Random ContinuousHV should be near-orthogonal, avg |sim| = {}",
        avg_abs_sim
    );
}

/// Test: ContinuousHV bind is commutative (A⊗B = B⊗A)
#[test]
fn test_continuous_bind_commutative() {
    use symthaea::hdc::unified_hv::ContinuousHV;

    for seed in 0..10 {
        let a = ContinuousHV::random(HDC_DIMENSION, seed * 7);
        let b = ContinuousHV::random(HDC_DIMENSION, seed * 7 + 50);

        let ab = a.bind(&b);
        let ba = b.bind(&a);

        let sim = ab.similarity(&ba);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "ContinuousHV bind should be commutative: sim(A⊗B, B⊗A) = {} (seed={})",
            sim,
            seed
        );
    }
}

/// Test: ContinuousHV bind produces vectors dissimilar to both inputs
/// (for random vectors, A⊗B is ~orthogonal to A and B)
#[test]
fn test_continuous_bind_dissimilarity() {
    use symthaea::hdc::unified_hv::ContinuousHV;

    for seed in 0..10 {
        let a = ContinuousHV::random(HDC_DIMENSION, seed * 3);
        let b = ContinuousHV::random(HDC_DIMENSION, seed * 3 + 1);
        let bound = a.bind(&b);

        let sim_a = bound.similarity(&a).abs();
        let sim_b = bound.similarity(&b).abs();

        assert!(
            sim_a < 0.1,
            "Bound vector should be dissimilar to A: |sim| = {} (seed={})",
            sim_a,
            seed
        );
        assert!(
            sim_b < 0.1,
            "Bound vector should be dissimilar to B: |sim| = {} (seed={})",
            sim_b,
            seed
        );
    }
}

/// Test: ContinuousHV bind preserves similarity structure
/// sim(A⊗C, B⊗C) ≈ sim(A, B)
#[test]
fn test_continuous_bind_preserves_similarity() {
    use symthaea::hdc::unified_hv::ContinuousHV;

    // Create two vectors with known similarity (partial overlap via bundling)
    let base = ContinuousHV::random(HDC_DIMENSION, 42);
    let noise = ContinuousHV::random(HDC_DIMENSION, 43);
    let similar = ContinuousHV::bundle(&[&base, &base, &noise]); // ~2/3 similar to base

    let c = ContinuousHV::random(HDC_DIMENSION, 99);

    let sim_original = base.similarity(&similar);
    let sim_bound = base.bind(&c).similarity(&similar.bind(&c));

    // Binding with same key should preserve relative similarity direction
    // (both positive or both negative; magnitude may vary with element-wise multiplication)
    assert!(
        sim_original > 0.3,
        "Precondition: base and similar should be correlated: {}",
        sim_original
    );
    assert!(
        sim_bound > 0.0,
        "Binding should preserve positive similarity direction: original={}, bound={}",
        sim_original,
        sim_bound
    );
}

/// Test: ContinuousHV bundle preserves component similarity
/// sim(bundle(A,B,C), A) > random similarity
#[test]
fn test_continuous_bundle_preserves_components() {
    use symthaea::hdc::unified_hv::ContinuousHV;

    for seed in 0..5 {
        let a = ContinuousHV::random(HDC_DIMENSION, seed * 11);
        let b = ContinuousHV::random(HDC_DIMENSION, seed * 11 + 10);
        let c = ContinuousHV::random(HDC_DIMENSION, seed * 11 + 20);

        let bundled = ContinuousHV::bundle(&[&a, &b, &c]);

        let sim_a = bundled.similarity(&a);
        let sim_b = bundled.similarity(&b);
        let sim_c = bundled.similarity(&c);

        // Bundle should be positively correlated with all components
        assert!(
            sim_a > 0.3,
            "Bundle should be similar to component A: sim={} (seed={})",
            sim_a,
            seed
        );
        assert!(
            sim_b > 0.3,
            "Bundle should be similar to component B: sim={} (seed={})",
            sim_b,
            seed
        );
        assert!(
            sim_c > 0.3,
            "Bundle should be similar to component C: sim={} (seed={})",
            sim_c,
            seed
        );
    }
}

/// Test: ContinuousHV similarity is bounded in [-1, 1]
#[test]
fn test_continuous_similarity_bounded() {
    use symthaea::hdc::unified_hv::ContinuousHV;

    for seed in 0..20 {
        let a = ContinuousHV::random(HDC_DIMENSION, seed);
        let b = ContinuousHV::random(HDC_DIMENSION, seed + 100);

        let sim = a.similarity(&b);
        assert!(
            (-1.0..=1.0).contains(&sim),
            "Similarity should be in [-1, 1], got {} (seed={})",
            sim,
            seed
        );

        // Also test with non-random vectors
        let negated = a.scale(-1.0);
        let sim_neg = a.similarity(&negated);
        assert!(
            (-1.0..=1.0).contains(&sim_neg),
            "Negated similarity should be in [-1, 1], got {}",
            sim_neg
        );
    }
}

/// Test: ContinuousHV normalize produces unit norm
#[test]
fn test_continuous_normalize_unit_norm() {
    use symthaea::hdc::unified_hv::ContinuousHV;

    for seed in 0..10 {
        let hv = ContinuousHV::random(HDC_DIMENSION, seed);
        let normalized = hv.normalize();

        let norm = normalized.norm();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "Normalized vector should have unit norm, got {} (seed={})",
            norm,
            seed
        );
    }
}

/// Test: ContinuousHV permute is cyclic (P^dim(A) = A)
#[test]
fn test_continuous_permute_cyclic() {
    use symthaea::hdc::unified_hv::ContinuousHV;

    // Use a smaller dimension for speed (full cycle over 16,384 would be slow)
    let dim = 256;
    let original = ContinuousHV::random(dim, 42);

    let mut current = original.clone();
    for _ in 0..dim {
        current = current.permute(1);
    }

    let sim = original.similarity(&current);
    assert!(
        (sim - 1.0).abs() < 1e-5,
        "Full cycle permutation should return original, similarity = {}",
        sim
    );
}

/// Test: ContinuousHV inverse_permute undoes permute
#[test]
fn test_continuous_inverse_permute() {
    use symthaea::hdc::unified_hv::ContinuousHV;

    for shift in [1, 7, 100, 1000, HDC_DIMENSION - 1] {
        let original = ContinuousHV::random(HDC_DIMENSION, 42);
        let permuted = original.permute(shift);
        let restored = permuted.inverse_permute(shift);

        let sim = original.similarity(&restored);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "inverse_permute should undo permute: sim = {} (shift={})",
            sim,
            shift
        );
    }
}

/// Test: ContinuousHV permutation changes the vector
#[test]
fn test_continuous_permute_changes_vector() {
    use symthaea::hdc::unified_hv::ContinuousHV;

    let original = ContinuousHV::random(HDC_DIMENSION, 42);
    let permuted = original.permute(1);

    let sim = original.similarity(&permuted);
    // For random vectors, single-position rotation should yield ~0 similarity
    assert!(
        sim.abs() < 0.1,
        "Permuted vector should be dissimilar from original, sim = {}",
        sim
    );
}

/// Test: ContinuousHV negation yields similarity = -1
#[test]
fn test_continuous_negation_similarity() {
    use symthaea::hdc::unified_hv::ContinuousHV;

    for seed in 0..10 {
        let hv = ContinuousHV::random(HDC_DIMENSION, seed);
        let negated = hv.scale(-1.0);

        let sim = hv.similarity(&negated);
        assert!(
            (sim - (-1.0)).abs() < 1e-5,
            "Negated vector should have similarity -1.0, got {} (seed={})",
            sim,
            seed
        );
    }
}

/// Test: ContinuousHV bundle of identical vectors equals that vector
#[test]
fn test_continuous_bundle_identical_is_identity() {
    use symthaea::hdc::unified_hv::ContinuousHV;

    let hv = ContinuousHV::random(HDC_DIMENSION, 42);
    let bundled = ContinuousHV::bundle(&[&hv, &hv, &hv]);

    let sim = hv.similarity(&bundled);
    assert!(
        (sim - 1.0).abs() < 1e-5,
        "Bundle of identical vectors should equal that vector: sim = {}",
        sim
    );
}

/// Test: ContinuousHV zero vector has similarity 0 with everything
#[test]
fn test_continuous_zero_similarity() {
    use symthaea::hdc::unified_hv::ContinuousHV;

    let zero = ContinuousHV::zero(HDC_DIMENSION);
    let random = ContinuousHV::random(HDC_DIMENSION, 42);

    let sim = zero.similarity(&random);
    assert!(
        sim.abs() < 1e-6,
        "Zero vector should have similarity 0, got {}",
        sim
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// BinaryHV Property Tests (original)
// ═══════════════════════════════════════════════════════════════════════════════

/// Test: Hamming distance is consistent with similarity
#[test]
fn test_hamming_similarity_relationship() {
    for seed in 0..10 {
        let a = BinaryHV::random(seed);
        let b = BinaryHV::random(seed + 100);

        let sim = a.similarity(&b);
        let hamming = a.hamming_distance(&b);

        // similarity = 1 - hamming/dimension
        let expected_sim = 1.0 - (hamming as f32 / HDC_DIMENSION as f32);

        assert!(
            (sim - expected_sim).abs() < 1e-6,
            "Similarity and Hamming should be consistent: sim={}, expected={}",
            sim,
            expected_sim
        );
    }
}