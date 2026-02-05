//! # HDC Property Tests
//!
//! Property-based tests for hyperdimensional computing operations to ensure
//! mathematical invariants hold across random inputs.

use symthaea::hdc::{HV16, HdcContext, HDC_DIMENSION};

/// Test: Binding is its own inverse (XOR property)
/// bind(bind(a, b), b) ≈ a (within similarity threshold)
#[test]
fn test_bind_is_self_inverse() {
    for seed_base in 0..10 {
        let a = HV16::random(seed_base);
        let b = HV16::random(seed_base + 100);

        // bind(a, b) then bind with b again should approximate a
        let bound = a.bind(&b);
        let unbound = bound.bind(&b);

        // For binary XOR, this should be exactly equal
        // Use similarity - should be 1.0 for identical vectors
        let sim = a.similarity(&unbound);
        assert!((sim - 1.0).abs() < 1e-6,
            "Bind self-inverse property failed: similarity={} (seed_base={})",
            sim, seed_base);
    }
}

/// Test: Binding is commutative
/// bind(a, b) == bind(b, a)
#[test]
fn test_bind_is_commutative() {
    for seed_base in 0..10 {
        let a = HV16::random(seed_base * 7);
        let b = HV16::random(seed_base * 7 + 50);

        let ab = a.bind(&b);
        let ba = b.bind(&a);

        // Should be exactly equal for XOR binding
        let sim = ab.similarity(&ba);
        assert!((sim - 1.0).abs() < 1e-6,
            "Bind commutativity failed: similarity={} (seed_base={})", sim, seed_base);
    }
}

/// Test: Bundling preserves similarity to components
/// bundle([a, b, c]) should be similar to each component
#[test]
fn test_bundle_preserves_component_similarity() {
    for seed_base in 0..5 {
        let a = HV16::random(seed_base * 11);
        let b = HV16::random(seed_base * 11 + 10);
        let c = HV16::random(seed_base * 11 + 20);

        // Bundle all three
        let abc = HV16::bundle(&[a.clone(), b.clone(), c.clone()]);

        // Bundle should be more similar to each component than random
        // Random similarity is ~0.5, bundled should be > 0.5
        let sim_a = abc.similarity(&a);
        let sim_b = abc.similarity(&b);
        let sim_c = abc.similarity(&c);

        assert!(sim_a > 0.55,
            "Bundle should be similar to component a: sim={} (seed_base={})",
            sim_a, seed_base);
        assert!(sim_b > 0.55,
            "Bundle should be similar to component b: sim={} (seed_base={})",
            sim_b, seed_base);
        assert!(sim_c > 0.55,
            "Bundle should be similar to component c: sim={} (seed_base={})",
            sim_c, seed_base);
    }
}

/// Test: Self-similarity is 1.0
#[test]
fn test_self_similarity_is_one() {
    for seed in 0..20 {
        let hv = HV16::random(seed);
        let sim = hv.similarity(&hv);

        assert!((sim - 1.0).abs() < 1e-6,
            "Self-similarity should be 1.0, got {} (seed={})", sim, seed);
    }
}

/// Test: Different random vectors have similarity around 0.5
/// (half the bits match by chance in high dimensions)
#[test]
fn test_random_vectors_half_similar() {
    let mut total_similarity: f64 = 0.0;
    let mut count = 0;

    for i in 0..20u64 {
        for j in (i+1)..20u64 {
            let a = HV16::random(i);
            let b = HV16::random(j);
            total_similarity += a.similarity(&b) as f64;
            count += 1;
        }
    }

    let avg_similarity = total_similarity / count as f64;

    // Random high-dimensional binary vectors should have similarity ~0.5
    // (half the bits match by chance)
    assert!((avg_similarity - 0.5).abs() < 0.05,
        "Random vectors should have ~0.5 similarity, got {}",
        avg_similarity);
}

/// Test: Permutation is cyclic
/// permute(v, n); permute(_, n) cycles through positions
#[test]
fn test_permute_cyclic() {
    let original = HV16::random(42);
    let mut current = original.clone();

    // Permute by dimension should return to original
    for _ in 0..HDC_DIMENSION {
        current = current.permute(1);
    }

    // Should be back to original
    let sim = original.similarity(&current);
    assert!((sim - 1.0).abs() < 1e-6,
        "Full cycle permutation should return original, similarity = {}", sim);
}

/// Test: Single permutation changes the vector
#[test]
fn test_permute_changes_vector() {
    let original = HV16::random(42);
    let permuted = original.permute(1);

    // Permuted vector should be different (similarity < 1.0)
    let sim = original.similarity(&permuted);
    assert!(sim < 0.99,
        "Permuted vector should be different from original, similarity = {}", sim);
}

/// Test: Arena allocation provides correct results
#[test]
fn test_arena_bind_correctness() {
    let ctx = HdcContext::new();

    // Create test vectors
    let a: Vec<i8> = (0..HDC_DIMENSION).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
    let b: Vec<i8> = (0..HDC_DIMENSION).map(|i| if i % 3 == 0 { 1 } else { -1 }).collect();

    let result = ctx.bind(&a, &b);

    // Verify element-wise multiplication
    for i in 0..HDC_DIMENSION {
        let expected = a[i] * b[i];
        assert_eq!(result[i], expected,
            "Arena bind mismatch at index {}: expected {}, got {}",
            i, expected, result[i]);
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
    assert!(result.iter().all(|&x| x == 1),
        "Bundle with positive majority should yield all 1s");
}

/// Test: Phi is non-negative
#[test]
fn test_phi_is_nonnegative() {
    use symthaea::phi_engine::PhiEngine;
    use symthaea::hdc::unified_hv::ContinuousHV;

    let engine = PhiEngine::auto();

    // Test with various topology sizes
    for n in 2..8 {
        let hvs: Vec<ContinuousHV> = (0..n)
            .map(|i| ContinuousHV::random(HDC_DIMENSION, i as u64))
            .collect();

        let result = engine.compute(&hvs);

        assert!(result.phi >= 0.0,
            "Phi should be non-negative, got {} for n={}", result.phi, n);
    }
}

/// Test: Phi symmetry
/// phi(union(A, B)) == phi(union(B, A))
#[test]
fn test_phi_is_symmetric() {
    use symthaea::phi_engine::PhiEngine;
    use symthaea::hdc::unified_hv::ContinuousHV;

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
    assert!((phi_ab.phi - phi_ba.phi).abs() < 1e-6,
        "Phi should be symmetric: phi(A+B)={}, phi(B+A)={}", phi_ab.phi, phi_ba.phi);
}

/// Test: Consistency - same input produces same output
#[test]
fn test_phi_is_deterministic() {
    use symthaea::phi_engine::PhiEngine;
    use symthaea::hdc::unified_hv::ContinuousHV;

    let engine = PhiEngine::auto();

    let hvs: Vec<ContinuousHV> = (0..5)
        .map(|i| ContinuousHV::random(HDC_DIMENSION, i))
        .collect();

    let phi1 = engine.compute(&hvs);
    let phi2 = engine.compute(&hvs);

    assert!((phi1.phi - phi2.phi).abs() < 1e-10,
        "Phi should be deterministic: first={}, second={}", phi1.phi, phi2.phi);
}

/// Test: Hamming distance is consistent with similarity
#[test]
fn test_hamming_similarity_relationship() {
    for seed in 0..10 {
        let a = HV16::random(seed);
        let b = HV16::random(seed + 100);

        let sim = a.similarity(&b);
        let hamming = a.hamming_distance(&b);

        // similarity = 1 - hamming/dimension
        let expected_sim = 1.0 - (hamming as f32 / HDC_DIMENSION as f32);

        assert!((sim - expected_sim).abs() < 1e-6,
            "Similarity and Hamming should be consistent: sim={}, expected={}",
            sim, expected_sim);
    }
}
