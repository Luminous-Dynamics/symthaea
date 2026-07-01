// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Property-based tests for the ContinuousHV (unified hypervector) module.
//!
//! Tests fundamental HDC algebraic invariants for continuous-valued hypervectors:
//! commutativity, orthogonality, symmetry, permutation invertibility, normalization
//! idempotence, and bundle retrieval properties.

use super::unified_hv::ContinuousHV;
use proptest::prelude::*;

const TEST_DIM: usize = 512;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    // =========================================================================
    // 1. Bind commutativity: a.bind(b) == b.bind(a) (element-wise multiply)
    // =========================================================================

    #[test]
    fn prop_bind_commutative(s1 in 1u64..100_000, s2 in 100_001u64..200_000) {
        let a = ContinuousHV::random(TEST_DIM, s1);
        let b = ContinuousHV::random(TEST_DIM, s2);
        let ab = a.bind(&b);
        let ba = b.bind(&a);
        let sim = ab.similarity(&ba);
        prop_assert!(
            (sim - 1.0).abs() < 0.001,
            "bind should be commutative, got sim={}",
            sim
        );
    }

    // =========================================================================
    // 2. Bind produces vectors near-orthogonal to both inputs
    // =========================================================================

    #[test]
    fn prop_bind_random_near_orthogonal(s1 in 1u64..100_000, s2 in 100_001u64..200_000) {
        let a = ContinuousHV::random(TEST_DIM, s1);
        let b = ContinuousHV::random(TEST_DIM, s2);
        let ab = a.bind(&b);

        let sim_a = ab.similarity(&a);
        let sim_b = ab.similarity(&b);

        prop_assert!(
            sim_a.abs() < 0.3,
            "bind(a,b) should be near-orthogonal to a, got sim={}",
            sim_a
        );
        prop_assert!(
            sim_b.abs() < 0.3,
            "bind(a,b) should be near-orthogonal to b, got sim={}",
            sim_b
        );
    }

    // =========================================================================
    // 3. Similarity is symmetric (bitwise exact for identical f32 ops)
    // =========================================================================

    #[test]
    fn prop_similarity_symmetric(s1 in 1u64..100_000, s2 in 100_001u64..200_000) {
        let a = ContinuousHV::random(TEST_DIM, s1);
        let b = ContinuousHV::random(TEST_DIM, s2);
        let sim_ab = a.similarity(&b);
        let sim_ba = b.similarity(&a);

        // Cosine similarity uses the same dot product and norms regardless of
        // argument order, so the result should be bitwise identical.
        prop_assert!(
            sim_ab == sim_ba,
            "similarity should be symmetric: sim(a,b)={} vs sim(b,a)={}",
            sim_ab,
            sim_ba
        );
    }

    // =========================================================================
    // 4. Self-similarity of a normalized vector is 1.0
    // =========================================================================

    #[test]
    fn prop_self_similarity_is_one(seed in 1u64..200_000) {
        let a = ContinuousHV::random(TEST_DIM, seed);
        let n = a.normalize();
        let sim = n.similarity(&n);

        prop_assert!(
            (sim - 1.0).abs() < 1e-4,
            "self-similarity of normalized vector should be ~1.0, got {}",
            sim
        );
    }

    // =========================================================================
    // 5. Random vectors with different seeds are near-orthogonal
    // =========================================================================

    #[test]
    fn prop_random_near_orthogonal(s1 in 1u64..100_000, s2 in 100_001u64..200_000) {
        let a = ContinuousHV::random(TEST_DIM, s1);
        let b = ContinuousHV::random(TEST_DIM, s2);
        let sim = a.similarity(&b);

        // At dim=512, random cosine similarity has stddev ~ 1/sqrt(512) ~ 0.044.
        // A bound of 0.25 is ~5.7 sigma -- extremely unlikely to fail.
        prop_assert!(
            sim.abs() < 0.25,
            "random vectors should be near-orthogonal, got sim={}",
            sim
        );
    }

    // =========================================================================
    // 6. Permute is invertible: a.permute(k).inverse_permute(k) == a (exact)
    // =========================================================================

    #[test]
    fn prop_permute_invertible(seed in 1u64..200_000, shift in 0usize..1024) {
        let a = ContinuousHV::random(TEST_DIM, seed);
        let restored = a.permute(shift).inverse_permute(shift);

        // Element-wise exact equality (no floating-point arithmetic, just moves)
        prop_assert!(
            a.values == restored.values,
            "permute({}).inverse_permute({}) should be identity; first diff at index {}",
            shift,
            shift,
            a.values
                .iter()
                .zip(restored.values.iter())
                .position(|(x, y)| x != y)
                .unwrap_or(usize::MAX)
        );
    }

    // =========================================================================
    // 7. Permute by 0 is the identity
    // =========================================================================

    #[test]
    fn prop_permute_zero_is_identity(seed in 1u64..200_000) {
        let a = ContinuousHV::random(TEST_DIM, seed);
        let permuted = a.permute(0);

        prop_assert!(
            a.values == permuted.values,
            "permute(0) should be the identity"
        );
    }

    // =========================================================================
    // 8. Permute by dim is the identity (cyclic wrap)
    // =========================================================================

    #[test]
    fn prop_permute_dim_is_identity(seed in 1u64..200_000) {
        let a = ContinuousHV::random(TEST_DIM, seed);
        let permuted = a.permute(TEST_DIM);

        prop_assert!(
            a.values == permuted.values,
            "permute(dim) should be the identity"
        );
    }

    // =========================================================================
    // 9. Normalize is idempotent: normalize(normalize(v)) == normalize(v)
    // =========================================================================

    #[test]
    fn prop_normalize_idempotent(seed in 1u64..200_000) {
        let a = ContinuousHV::random(TEST_DIM, seed);
        let n1 = a.normalize();
        let n2 = n1.normalize();
        let sim = n1.similarity(&n2);

        prop_assert!(
            (sim - 1.0).abs() < 1e-5,
            "normalize should be idempotent, got sim(n1, n2)={}",
            sim
        );

        // Also check element-wise: values should be nearly identical
        let max_diff = n1
            .values
            .iter()
            .zip(n2.values.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);

        prop_assert!(
            max_diff < 1e-5,
            "normalize idempotence: max element diff={}, expected < 1e-5",
            max_diff
        );
    }

    // =========================================================================
    // 10. Bundle is more similar to its inputs than to a random outsider
    // =========================================================================

    #[test]
    fn prop_bundle_similar_to_inputs(
        s1 in 1u64..100_000,
        s2 in 100_001u64..200_000,
        s3 in 200_001u64..300_000,
    ) {
        let a = ContinuousHV::random(TEST_DIM, s1);
        let b = ContinuousHV::random(TEST_DIM, s2);
        let c = ContinuousHV::random(TEST_DIM, s3); // random outsider

        let bundled = ContinuousHV::bundle(&[&a, &b]);

        let sim_a = bundled.similarity(&a);
        let sim_b = bundled.similarity(&b);
        let sim_c = bundled.similarity(&c);

        // Bundle should be positively correlated with both inputs
        prop_assert!(
            sim_a > 0.0,
            "bundle([a,b]) should have positive similarity with a, got {}",
            sim_a
        );
        prop_assert!(
            sim_b > 0.0,
            "bundle([a,b]) should have positive similarity with b, got {}",
            sim_b
        );

        // Bundle should be more similar to its inputs than to a random outsider
        prop_assert!(
            sim_a > sim_c,
            "bundle should be more similar to input a ({}) than to random c ({})",
            sim_a,
            sim_c
        );
        prop_assert!(
            sim_b > sim_c,
            "bundle should be more similar to input b ({}) than to random c ({})",
            sim_b,
            sim_c
        );
    }
}
