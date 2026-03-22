// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
Property-Based Tests for ContinuousHV Invariants

Verifies mathematical properties of ContinuousHV (continuous hypervector)
operations hold across randomly generated inputs using proptest.

## Key Invariants Tested

1. **Similarity is symmetric**: sim(a, b) = sim(b, a)
2. **Self-similarity is maximal**: sim(a, a) ≈ 1.0 for normalized vectors
3. **Binding preserves distances**: sim(bind(a,c), bind(b,c)) ≈ sim(a, b)
4. **Bundle is commutative**: bundle([a,b]) ≈ bundle([b,a])
5. **Permutation is invertible**: unpermute(permute(v, k), k) = v
6. **Normalization is idempotent**: normalize(normalize(v)) = normalize(v)
7. **Random vectors are near-orthogonal**: sim(random_a, random_b) ≈ 0
*/

#![cfg(test)]

use super::unified_hv::ContinuousHV;
use proptest::prelude::*;

/// Use smaller dimension for faster proptest runs.
const TEST_DIM: usize = 256;

/// Generate a random ContinuousHV with values in [-1, 1].
fn arbitrary_continuous_hv() -> impl Strategy<Value = ContinuousHV> {
    prop::collection::vec(-1.0f32..1.0f32, TEST_DIM)
        .prop_map(|values| ContinuousHV::from_vec(values))
}

/// Generate a non-zero ContinuousHV (for normalization tests).
fn arbitrary_nonzero_hv() -> impl Strategy<Value = ContinuousHV> {
    (0u64..100_000u64).prop_map(|seed| ContinuousHV::random(TEST_DIM, seed))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    // =================================================================
    // Similarity properties
    // =================================================================

    #[test]
    fn similarity_is_symmetric(seed_a in 0u64..10000, seed_b in 0u64..10000) {
        let a = ContinuousHV::random(TEST_DIM, seed_a);
        let b = ContinuousHV::random(TEST_DIM, seed_b);
        let sim_ab = a.similarity(&b);
        let sim_ba = b.similarity(&a);
        prop_assert!((sim_ab - sim_ba).abs() < 1e-6,
            "sim(a,b)={} != sim(b,a)={}", sim_ab, sim_ba);
    }

    #[test]
    fn self_similarity_is_maximal(seed in 0u64..10000) {
        let v = ContinuousHV::random(TEST_DIM, seed);
        let n = v.normalize();
        let sim = n.similarity(&n);
        prop_assert!((sim - 1.0).abs() < 1e-5,
            "Self-similarity of normalized vector should be ~1.0, got {}", sim);
    }

    #[test]
    fn random_vectors_near_orthogonal(seed_a in 0u64..10000, seed_b in 10000u64..20000) {
        let a = ContinuousHV::random(TEST_DIM, seed_a);
        let b = ContinuousHV::random(TEST_DIM, seed_b);
        let sim = a.similarity(&b);
        // With dim=256, random cosine similarity should be close to 0
        // but with some variance: |sim| < 0.3 is a reasonable bound
        prop_assert!(sim.abs() < 0.3,
            "Random vectors should be near-orthogonal, got sim={}", sim);
    }

    #[test]
    fn similarity_bounded(a in arbitrary_continuous_hv(), b in arbitrary_continuous_hv()) {
        let sim = a.similarity(&b);
        prop_assert!(sim >= -1.01 && sim <= 1.01,
            "Cosine similarity should be in [-1, 1], got {}", sim);
    }

    // =================================================================
    // Binding properties
    // =================================================================

    #[test]
    fn bind_preserves_distance(
        seed_a in 0u64..10000,
        seed_b in 10000u64..20000,
        seed_c in 20000u64..30000,
    ) {
        let a = ContinuousHV::random(TEST_DIM, seed_a);
        let b = ContinuousHV::random(TEST_DIM, seed_b);
        let c = ContinuousHV::random(TEST_DIM, seed_c);

        let sim_ab = a.similarity(&b);
        let ac = a.bind(&c);
        let bc = b.bind(&c);
        let sim_ac_bc = ac.similarity(&bc);

        // Binding (element-wise multiply) with same vector preserves similarity
        // approximately, especially for random vectors
        prop_assert!((sim_ab - sim_ac_bc).abs() < 0.3,
            "Binding should approximately preserve distance: sim(a,b)={}, sim(a*c,b*c)={}",
            sim_ab, sim_ac_bc);
    }

    #[test]
    fn bind_is_commutative(seed_a in 0u64..10000, seed_b in 10000u64..20000) {
        let a = ContinuousHV::random(TEST_DIM, seed_a);
        let b = ContinuousHV::random(TEST_DIM, seed_b);
        let ab = a.bind(&b);
        let ba = b.bind(&a);
        let sim = ab.similarity(&ba);
        prop_assert!((sim - 1.0).abs() < 1e-5,
            "Binding should be commutative, got sim(a*b, b*a)={}", sim);
    }

    // =================================================================
    // Bundle properties
    // =================================================================

    #[test]
    fn bundle_is_commutative(seed_a in 0u64..10000, seed_b in 10000u64..20000) {
        let a = ContinuousHV::random(TEST_DIM, seed_a);
        let b = ContinuousHV::random(TEST_DIM, seed_b);
        let ab = ContinuousHV::bundle(&[&a, &b]);
        let ba = ContinuousHV::bundle(&[&b, &a]);
        let sim = ab.similarity(&ba);
        prop_assert!((sim - 1.0).abs() < 1e-5,
            "Bundle should be commutative, got sim={}", sim);
    }

    #[test]
    fn bundle_similar_to_inputs(seed_a in 0u64..10000, seed_b in 10000u64..20000) {
        let a = ContinuousHV::random(TEST_DIM, seed_a);
        let b = ContinuousHV::random(TEST_DIM, seed_b);
        let bundled = ContinuousHV::bundle(&[&a, &b]);

        let sim_a = bundled.similarity(&a);
        let sim_b = bundled.similarity(&b);

        // Bundle should be somewhat similar to both inputs
        prop_assert!(sim_a > 0.2, "Bundle should be similar to input a, got {}", sim_a);
        prop_assert!(sim_b > 0.2, "Bundle should be similar to input b, got {}", sim_b);
    }

    // =================================================================
    // Permutation properties
    // =================================================================

    #[test]
    fn permute_invertible(seed in 0u64..10000, shift in 1usize..256) {
        let v = ContinuousHV::random(TEST_DIM, seed);
        let permuted = v.permute(shift);
        let unpermuted = permuted.permute(TEST_DIM - shift);

        // After permute then unpermute, should get back the original
        let sim = v.similarity(&unpermuted);
        prop_assert!((sim - 1.0).abs() < 1e-5,
            "Permute should be invertible, got sim={} for shift={}", sim, shift);
    }

    #[test]
    fn permute_changes_vector(seed in 0u64..10000, shift in 1usize..256) {
        let v = ContinuousHV::random(TEST_DIM, seed);
        let permuted = v.permute(shift);
        let sim = v.similarity(&permuted);

        // Permuted vector should be quite different from original
        // (for random vectors, similarity should be near 0)
        prop_assert!(sim.abs() < 0.5,
            "Permutation should produce a different vector, got sim={}", sim);
    }

    // =================================================================
    // Normalization properties
    // =================================================================

    #[test]
    fn normalize_is_idempotent(seed in 0u64..10000) {
        let v = ContinuousHV::random(TEST_DIM, seed);
        let n1 = v.normalize();
        let n2 = n1.normalize();
        let sim = n1.similarity(&n2);
        prop_assert!((sim - 1.0).abs() < 1e-5,
            "Normalization should be idempotent, got sim={}", sim);
    }

    #[test]
    fn normalized_has_unit_norm(seed in 0u64..10000) {
        let v = ContinuousHV::random(TEST_DIM, seed);
        let n = v.normalize();
        let norm: f32 = n.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        prop_assert!((norm - 1.0).abs() < 1e-4,
            "Normalized vector should have unit norm, got {}", norm);
    }

    // =================================================================
    // Algebraic composition properties
    // =================================================================

    #[test]
    fn bind_then_bundle_retrieval(
        seed_k in 0u64..10000,
        seed_v in 10000u64..20000,
        seed_noise in 20000u64..30000,
    ) {
        // Store a key-value pair using bind, bundle with noise, retrieve with key
        let key = ContinuousHV::random(TEST_DIM, seed_k);
        let value = ContinuousHV::random(TEST_DIM, seed_v);
        let noise = ContinuousHV::random(TEST_DIM, seed_noise);

        // Memory = bind(key, value) + noise
        let kv = key.bind(&value);
        let memory = ContinuousHV::bundle(&[&kv, &noise]);

        // Retrieve: bind(memory, key) should be similar to value
        let retrieved = memory.bind(&key);
        let sim = retrieved.similarity(&value);

        // With one item and one noise term, should have decent retrieval
        prop_assert!(sim > 0.1,
            "Retrieved value should be similar to stored value, got sim={}", sim);
    }
}
