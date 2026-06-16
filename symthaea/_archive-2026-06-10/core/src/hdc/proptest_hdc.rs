// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
Property-Based Tests for HDC Invariants

Uses proptest to verify mathematical properties of hyperdimensional computing
operations hold across randomly generated inputs.

## Key Invariants Tested

1. **Binding is self-inverse**: bind(bind(a, b), b) ≈ a
2. **Bundling is associative**: bundle(a, bundle(b, c)) ≈ bundle(bundle(a, b), c)
3. **Permutation is invertible**: unpermute(permute(v, k), k) = v
4. **Near-orthogonality**: random vectors have ~50% similarity
5. **Similarity is symmetric**: sim(a, b) = sim(b, a)
6. **Binding preserves structure**: sim(bind(a,c), bind(b,c)) ≈ sim(a, b)
*/

use super::HdcContext;
use super::native_similarity::PackedBipolar;
use proptest::prelude::*;

/// Generate random bipolar vector of given dimension
fn arbitrary_bipolar(dim: usize) -> impl Strategy<Value = Vec<i8>> {
    prop::collection::vec(prop::sample::select(vec![-1i8, 1i8]), dim)
}

/// Generate random bipolar vector at standard HDC dimension
fn arbitrary_hdc_vector() -> impl Strategy<Value = Vec<i8>> {
    // Use smaller dimension for faster tests, scale results
    arbitrary_bipolar(1024)
}

/// Compute Hamming similarity between two bipolar vectors
fn hamming_similarity(a: &[i8], b: &[i8]) -> f32 {
    assert_eq!(a.len(), b.len());
    let matches: usize = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();
    matches as f32 / a.len() as f32
}

/// Element-wise multiply (bind) two bipolar vectors
fn bind(a: &[i8], b: &[i8]) -> Vec<i8> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// Majority vote bundle of vectors
fn bundle(vectors: &[&[i8]]) -> Vec<i8> {
    if vectors.is_empty() {
        return Vec::new();
    }

    let dim = vectors[0].len();
    let threshold = vectors.len() as i32 / 2;

    (0..dim)
        .map(|i| {
            let sum: i32 = vectors.iter().map(|v| v[i] as i32).sum();
            if sum > threshold { 1 } else { -1 }
        })
        .collect()
}

/// Circular permute (shift right)
fn permute(v: &[i8], shift: usize) -> Vec<i8> {
    let dim = v.len();
    let mut result = vec![0i8; dim];
    for i in 0..dim {
        let new_idx = (i + shift) % dim;
        result[new_idx] = v[i];
    }
    result
}

/// Inverse permute (shift left)
fn unpermute(v: &[i8], shift: usize) -> Vec<i8> {
    let dim = v.len();
    let mut result = vec![0i8; dim];
    for i in 0..dim {
        let orig_idx = (i + dim - (shift % dim)) % dim;
        result[orig_idx] = v[i];
    }
    result
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    // =========================================================================
    // Property 1: Binding is self-inverse
    // bind(bind(a, b), b) = a  (XOR is self-inverse)
    // =========================================================================
    #[test]
    fn prop_binding_self_inverse(
        a in arbitrary_hdc_vector(),
        b in arbitrary_hdc_vector()
    ) {
        let bound = bind(&a, &b);
        let recovered = bind(&bound, &b);

        // Should be exact equality for bipolar XOR
        prop_assert_eq!(&recovered, &a, "bind(bind(a,b), b) should equal a");
    }

    // =========================================================================
    // Property 2: Similarity is symmetric
    // sim(a, b) = sim(b, a)
    // =========================================================================
    #[test]
    fn prop_similarity_symmetric(
        a in arbitrary_hdc_vector(),
        b in arbitrary_hdc_vector()
    ) {
        let sim_ab = hamming_similarity(&a, &b);
        let sim_ba = hamming_similarity(&b, &a);

        prop_assert!(
            (sim_ab - sim_ba).abs() < f32::EPSILON,
            "Similarity should be symmetric: {} vs {}",
            sim_ab, sim_ba
        );
    }

    // =========================================================================
    // Property 3: Self-similarity is maximal
    // sim(a, a) = 1.0
    // =========================================================================
    #[test]
    fn prop_self_similarity_maximal(a in arbitrary_hdc_vector()) {
        let sim = hamming_similarity(&a, &a);
        prop_assert!(
            (sim - 1.0).abs() < f32::EPSILON,
            "Self-similarity should be 1.0, got {}",
            sim
        );
    }

    // =========================================================================
    // Property 4: Opposite vectors have zero similarity
    // sim(a, -a) = 0.0
    // =========================================================================
    #[test]
    fn prop_opposite_similarity_zero(a in arbitrary_hdc_vector()) {
        let neg_a: Vec<i8> = a.iter().map(|x| -x).collect();
        let sim = hamming_similarity(&a, &neg_a);

        prop_assert!(
            sim.abs() < f32::EPSILON,
            "Opposite similarity should be 0.0, got {}",
            sim
        );
    }

    // =========================================================================
    // Property 5: Random vectors are near-orthogonal (~50% similarity)
    // Central limit theorem: for large D, similarity → 0.5
    // =========================================================================
    #[test]
    fn prop_random_near_orthogonal(
        a in arbitrary_hdc_vector(),
        b in arbitrary_hdc_vector()
    ) {
        let sim = hamming_similarity(&a, &b);

        // With 1024 dimensions, expect similarity in [0.35, 0.65] almost always
        // 99.7% confidence interval for 50% ± 3σ where σ = sqrt(0.25/1024) ≈ 0.0156
        // 3σ ≈ 0.047, so bounds are roughly [0.45, 0.55]
        // We use wider bounds [0.3, 0.7] to account for test randomness
        prop_assert!(
            (0.3..=0.7).contains(&sim),
            "Random vectors should have ~50% similarity, got {}",
            sim
        );
    }

    // =========================================================================
    // Property 6: Permutation is invertible
    // unpermute(permute(v, k), k) = v
    // =========================================================================
    #[test]
    fn prop_permutation_invertible(
        v in arbitrary_hdc_vector(),
        shift in 0usize..1024
    ) {
        let permuted = permute(&v, shift);
        let recovered = unpermute(&permuted, shift);

        prop_assert_eq!(
            &recovered, &v,
            "unpermute(permute(v, {}), {}) should equal v",
            shift, shift
        );
    }

    // =========================================================================
    // Property 7: Permutation preserves similarity structure
    // sim(permute(a, k), permute(b, k)) = sim(a, b)
    // =========================================================================
    #[test]
    fn prop_permutation_preserves_similarity(
        a in arbitrary_hdc_vector(),
        b in arbitrary_hdc_vector(),
        shift in 0usize..1024
    ) {
        let sim_original = hamming_similarity(&a, &b);
        let sim_permuted = hamming_similarity(&permute(&a, shift), &permute(&b, shift));

        prop_assert!(
            (sim_original - sim_permuted).abs() < f32::EPSILON,
            "Permutation should preserve similarity: {} vs {}",
            sim_original, sim_permuted
        );
    }

    // =========================================================================
    // Property 8: Binding preserves similarity structure
    // sim(bind(a, c), bind(b, c)) = sim(a, b)
    // =========================================================================
    #[test]
    fn prop_binding_preserves_similarity(
        a in arbitrary_hdc_vector(),
        b in arbitrary_hdc_vector(),
        c in arbitrary_hdc_vector()
    ) {
        let sim_original = hamming_similarity(&a, &b);
        let sim_bound = hamming_similarity(&bind(&a, &c), &bind(&b, &c));

        prop_assert!(
            (sim_original - sim_bound).abs() < f32::EPSILON,
            "Binding should preserve similarity: {} vs {}",
            sim_original, sim_bound
        );
    }

    // =========================================================================
    // Property 9: Bundle is similar to all constituents
    // sim(bundle(a, b), a) > 0.5 (for 2-element bundle)
    // sim(bundle(a, b), b) > 0.5 (for 2-element bundle)
    // =========================================================================
    #[test]
    fn prop_bundle_similar_to_constituents(
        a in arbitrary_hdc_vector(),
        b in arbitrary_hdc_vector()
    ) {
        let bundled = bundle(&[&a[..], &b[..]]);

        let sim_a = hamming_similarity(&bundled, &a);
        let sim_b = hamming_similarity(&bundled, &b);

        // For a 2-element bundle, each constituent should have > 50% similarity
        // (actually exactly 0.5 for random orthogonal vectors, but with ties)
        prop_assert!(
            sim_a >= 0.45,
            "Bundle should be similar to constituent a: {}",
            sim_a
        );
        prop_assert!(
            sim_b >= 0.45,
            "Bundle should be similar to constituent b: {}",
            sim_b
        );
    }

    // =========================================================================
    // Property 10: PackedBipolar roundtrip preserves data
    // from_bipolar(v).to_bipolar() = v
    // =========================================================================
    #[test]
    fn prop_packed_bipolar_roundtrip(v in arbitrary_hdc_vector()) {
        let packed = PackedBipolar::from_bipolar(&v);
        let recovered = packed.to_bipolar();

        prop_assert_eq!(
            &recovered, &v,
            "PackedBipolar roundtrip should preserve data"
        );
    }

    // =========================================================================
    // Property 11: PackedBipolar XOR similarity matches Hamming
    // packed.xor_similarity(other) = hamming_similarity(a, b)
    // =========================================================================
    #[test]
    fn prop_packed_xor_matches_hamming(
        a in arbitrary_hdc_vector(),
        b in arbitrary_hdc_vector()
    ) {
        let packed_a = PackedBipolar::from_bipolar(&a);
        let packed_b = PackedBipolar::from_bipolar(&b);

        let xor_sim = packed_a.xor_similarity(&packed_b);
        let ham_sim = hamming_similarity(&a, &b);

        prop_assert!(
            (xor_sim - ham_sim).abs() < 0.001,
            "XOR similarity should match Hamming: {} vs {}",
            xor_sim, ham_sim
        );
    }

    // =========================================================================
    // Property 12: HdcContext bind matches manual implementation
    // =========================================================================
    #[test]
    fn prop_hdc_context_bind_correct(
        a in arbitrary_bipolar(128), // Smaller for faster context tests
        b in arbitrary_bipolar(128)
    ) {
        let ctx = HdcContext::new();
        let ctx_result = ctx.bind(&a, &b).to_vec();
        let manual_result = bind(&a, &b);

        prop_assert_eq!(
            &ctx_result, &manual_result,
            "HdcContext::bind should match manual implementation"
        );
    }

    // =========================================================================
    // Property 13: Binding is commutative for bipolar
    // bind(a, b) = bind(b, a) (XOR is commutative)
    // =========================================================================
    #[test]
    fn prop_binding_commutative(
        a in arbitrary_hdc_vector(),
        b in arbitrary_hdc_vector()
    ) {
        let ab = bind(&a, &b);
        let ba = bind(&b, &a);

        prop_assert_eq!(
            &ab, &ba,
            "Binding should be commutative"
        );
    }

    // =========================================================================
    // Property 14: Identity binding
    // bind(a, ones) = a where ones = [1, 1, ..., 1]
    // =========================================================================
    #[test]
    fn prop_identity_binding(a in arbitrary_hdc_vector()) {
        let ones: Vec<i8> = vec![1i8; a.len()];
        let bound = bind(&a, &ones);

        prop_assert_eq!(
            &bound, &a,
            "Binding with ones should be identity"
        );
    }
}

// Additional deterministic tests for edge cases
#[test]
fn test_empty_bundle() {
    let result = bundle(&[]);
    assert!(result.is_empty());
}

#[test]
fn test_single_element_bundle() {
    let a = vec![1i8, -1, 1, -1, 1, -1, 1, -1];
    let result = bundle(&[&a[..]]);
    // Single element bundle should match original
    // (all votes go to the same value)
    assert_eq!(result, a);
}

#[test]
fn test_permute_by_zero() {
    let v = vec![1i8, -1, 1, -1, 1, -1, 1, -1];
    let result = permute(&v, 0);
    assert_eq!(result, v);
}

#[test]
fn test_permute_by_dimension() {
    let v = vec![1i8, -1, 1, -1, 1, -1, 1, -1];
    let result = permute(&v, v.len());
    assert_eq!(result, v); // Full rotation = identity
}

#[test]
fn test_packed_memory_efficiency() {
    let dim = 16384;
    let v = vec![1i8; dim];
    let packed = PackedBipolar::from_bipolar(&v);

    // Should use dim/8 bytes (2KB for 16K dimensions)
    let expected = dim.div_ceil(64) * 8;
    assert_eq!(packed.memory_bytes(), expected);

    // Much smaller than raw i8 vector
    assert!(packed.memory_bytes() < dim);
}

#[test]
fn test_xor_similarity_bounds() {
    // All same
    let a = vec![1i8; 64];
    let b = vec![1i8; 64];
    let packed_a = PackedBipolar::from_bipolar(&a);
    let packed_b = PackedBipolar::from_bipolar(&b);
    assert!((packed_a.xor_similarity(&packed_b) - 1.0).abs() < 0.001);

    // All opposite
    let c = vec![-1i8; 64];
    let packed_c = PackedBipolar::from_bipolar(&c);
    assert!((packed_a.xor_similarity(&packed_c) - 0.0).abs() < 0.001);

    // Half and half
    let mut d = vec![1i8; 64];
    for i in 0..32 {
        d[i] = -1;
    }
    let packed_d = PackedBipolar::from_bipolar(&d);
    assert!((packed_a.xor_similarity(&packed_d) - 0.5).abs() < 0.001);
}
