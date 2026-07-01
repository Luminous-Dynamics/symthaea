// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
Property-Based Tests for BinaryHV Invariants

Verifies mathematical properties of BinaryHV (16,384-bit packed binary hypervector)
operations hold across randomly generated inputs using proptest.

## Key Invariants Tested

1. **Bind (XOR) algebra**: self-inverse, commutative, associative, identity, self-annihilation
2. **Similarity**: symmetric, bounded, self = 1.0, inverse = 0.0, cosine relationship
3. **Hamming distance**: consistency with similarity
4. **Permute**: popcount preservation, identity shifts, invertibility
5. **Bundle**: idempotent for identical vectors, majority preservation
6. **Invert**: involution (double-invert = original)
7. **Statistical**: random vectors near-orthogonal, balanced density
*/

#![cfg(test)]

use super::binary_hv::BinaryHV;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    // =========================================================================
    // 1. Bind is self-inverse: a.bind(&b).bind(&b) == a
    // =========================================================================
    #[test]
    fn prop_bind_self_inverse(seed_a in 0u64..100_000, seed_b in 100_000u64..200_000) {
        let a = BinaryHV::random(seed_a);
        let b = BinaryHV::random(seed_b);
        let recovered = a.bind(&b).bind(&b);
        prop_assert_eq!(recovered, a, "bind(bind(a, b), b) should equal a");
    }

    // =========================================================================
    // 2. Bind is commutative: a.bind(&b) == b.bind(&a)
    // =========================================================================
    #[test]
    fn prop_bind_commutative(seed_a in 0u64..100_000, seed_b in 100_000u64..200_000) {
        let a = BinaryHV::random(seed_a);
        let b = BinaryHV::random(seed_b);
        prop_assert_eq!(a.bind(&b), b.bind(&a), "bind should be commutative");
    }

    // =========================================================================
    // 3. Bind with zero is identity: a.bind(&zero()) == a
    // =========================================================================
    #[test]
    fn prop_bind_zero_identity(seed in 0u64..100_000) {
        let a = BinaryHV::random(seed);
        prop_assert_eq!(a.bind(&BinaryHV::zero()), a, "bind with zero should be identity");
    }

    // =========================================================================
    // 4. Bind with self is zero: a.bind(&a) == zero()
    // =========================================================================
    #[test]
    fn prop_bind_self_is_zero(seed in 0u64..100_000) {
        let a = BinaryHV::random(seed);
        prop_assert_eq!(a.bind(&a), BinaryHV::zero(), "bind with self should produce zero");
    }

    // =========================================================================
    // 5. Bind is associative: (a.bind(&b)).bind(&c) == a.bind(&(b.bind(&c)))
    // =========================================================================
    #[test]
    fn prop_bind_associative(
        seed_a in 0u64..100_000,
        seed_b in 100_000u64..200_000,
        seed_c in 200_000u64..300_000,
    ) {
        let a = BinaryHV::random(seed_a);
        let b = BinaryHV::random(seed_b);
        let c = BinaryHV::random(seed_c);
        let lhs = a.bind(&b).bind(&c);
        let rhs = a.bind(&b.bind(&c));
        prop_assert_eq!(lhs, rhs, "bind should be associative");
    }

    // =========================================================================
    // 6. Similarity is symmetric: a.similarity(&b) == b.similarity(&a)
    // =========================================================================
    #[test]
    fn prop_similarity_symmetric(seed_a in 0u64..100_000, seed_b in 100_000u64..200_000) {
        let a = BinaryHV::random(seed_a);
        let b = BinaryHV::random(seed_b);
        let sim_ab = a.similarity(&b);
        let sim_ba = b.similarity(&a);
        prop_assert_eq!(sim_ab, sim_ba, "similarity should be exactly symmetric");
    }

    // =========================================================================
    // 7. Self-similarity is 1.0: a.similarity(&a) == 1.0
    // =========================================================================
    #[test]
    fn prop_self_similarity_is_one(seed in 0u64..100_000) {
        let a = BinaryHV::random(seed);
        let sim = a.similarity(&a);
        prop_assert_eq!(sim, 1.0f32, "self-similarity should be exactly 1.0, got {}", sim);
    }

    // =========================================================================
    // 8. Opposite similarity is 0.0: a.similarity(&a.invert()) == 0.0
    // =========================================================================
    #[test]
    fn prop_opposite_similarity_is_zero(seed in 0u64..100_000) {
        let a = BinaryHV::random(seed);
        let sim = a.similarity(&a.invert());
        prop_assert_eq!(sim, 0.0f32, "similarity with inverse should be exactly 0.0, got {}", sim);
    }

    // =========================================================================
    // 9. Similarity bounded: 0.0 <= sim <= 1.0
    // =========================================================================
    #[test]
    fn prop_similarity_bounded(seed_a in 0u64..100_000, seed_b in 100_000u64..200_000) {
        let a = BinaryHV::random(seed_a);
        let b = BinaryHV::random(seed_b);
        let sim = a.similarity(&b);
        prop_assert!(sim >= 0.0 && sim <= 1.0,
            "similarity should be in [0.0, 1.0], got {}", sim);
    }

    // =========================================================================
    // 10. Cosine similarity bounded: -1.0 <= cos_sim <= 1.0
    // =========================================================================
    #[test]
    fn prop_cosine_similarity_bounded(seed_a in 0u64..100_000, seed_b in 100_000u64..200_000) {
        let a = BinaryHV::random(seed_a);
        let b = BinaryHV::random(seed_b);
        let cos_sim = a.cosine_similarity(&b);
        prop_assert!(cos_sim >= -1.0 && cos_sim <= 1.0,
            "cosine_similarity should be in [-1.0, 1.0], got {}", cos_sim);
    }

    // =========================================================================
    // 11. Cosine similarity matches 2*sim - 1
    // =========================================================================
    #[test]
    fn prop_cosine_equals_two_sim_minus_one(seed_a in 0u64..100_000, seed_b in 100_000u64..200_000) {
        let a = BinaryHV::random(seed_a);
        let b = BinaryHV::random(seed_b);
        let cos_sim = a.cosine_similarity(&b);
        let expected = 2.0 * a.similarity(&b) - 1.0;
        prop_assert!(
            (cos_sim - expected).abs() < f32::EPSILON,
            "cosine_similarity ({}) should equal 2*similarity-1 ({})", cos_sim, expected
        );
    }

    // =========================================================================
    // 12. Hamming distance + matching bits = DIM
    // =========================================================================
    #[test]
    fn prop_hamming_plus_matching_equals_dim(seed_a in 0u64..100_000, seed_b in 100_000u64..200_000) {
        let a = BinaryHV::random(seed_a);
        let b = BinaryHV::random(seed_b);
        let hamming = a.hamming_distance(&b);
        let sim = a.similarity(&b);
        let matching = (sim * BinaryHV::DIM as f32).round() as u32;
        prop_assert_eq!(
            hamming + matching, BinaryHV::DIM as u32,
            "hamming_distance ({}) + matching_bits ({}) should equal DIM ({})",
            hamming, matching, BinaryHV::DIM
        );
    }

    // =========================================================================
    // 13. Permute preserves popcount
    // =========================================================================
    #[test]
    fn prop_permute_preserves_popcount(seed in 0u64..100_000, shift in 0usize..16384) {
        let a = BinaryHV::random(seed);
        let permuted = a.permute(shift);
        prop_assert_eq!(
            a.popcount(), permuted.popcount(),
            "permute({}) should preserve popcount: {} vs {}", shift, a.popcount(), permuted.popcount()
        );
    }

    // =========================================================================
    // 14. Permute by 0 is identity
    // =========================================================================
    #[test]
    fn prop_permute_zero_is_identity(seed in 0u64..100_000) {
        let a = BinaryHV::random(seed);
        prop_assert_eq!(a.permute(0), a, "permute(0) should be identity");
    }

    // =========================================================================
    // 15. Permute by DIM is identity
    // =========================================================================
    #[test]
    fn prop_permute_dim_is_identity(seed in 0u64..100_000) {
        let a = BinaryHV::random(seed);
        prop_assert_eq!(
            a.permute(BinaryHV::DIM), a,
            "permute(DIM) should be identity (full rotation)"
        );
    }

    // =========================================================================
    // 16. Permute is invertible: a.permute(s).permute(DIM - s) == a
    // =========================================================================
    #[test]
    fn prop_permute_invertible(seed in 0u64..100_000, shift in 1usize..16384) {
        let a = BinaryHV::random(seed);
        let recovered = a.permute(shift).permute(BinaryHV::DIM - shift);
        prop_assert_eq!(recovered, a,
            "permute({}).permute({}) should recover original", shift, BinaryHV::DIM - shift);
    }

    // =========================================================================
    // 17. Bundle of identical vectors is that vector
    // =========================================================================
    #[test]
    fn prop_bundle_identical_is_identity(seed in 0u64..100_000) {
        let a = BinaryHV::random(seed);
        let bundled = BinaryHV::bundle(&[a, a, a]);
        prop_assert_eq!(bundled, a, "bundle([a, a, a]) should equal a");
    }

    // =========================================================================
    // 18. Bundle preserves majority: bundle([a, a, b]) == a
    // =========================================================================
    #[test]
    fn prop_bundle_preserves_majority(seed_a in 0u64..100_000, seed_b in 100_000u64..200_000) {
        let a = BinaryHV::random(seed_a);
        let b = BinaryHV::random(seed_b);
        let bundled = BinaryHV::bundle(&[a, a, b]);
        prop_assert_eq!(bundled, a,
            "bundle([a, a, b]) should equal a (majority vote with 2-of-3)");
    }

    // =========================================================================
    // 19. Random vectors near orthogonal: |sim - 0.5| < 0.1
    // =========================================================================
    #[test]
    fn prop_random_near_orthogonal(seed_a in 0u64..100_000, seed_b in 100_000u64..200_000) {
        let a = BinaryHV::random(seed_a);
        let b = BinaryHV::random(seed_b);
        let sim = a.similarity(&b);
        prop_assert!(
            (sim - 0.5).abs() < 0.1,
            "random vectors should be near-orthogonal (sim ~0.5), got {}", sim
        );
    }

    // =========================================================================
    // 20. Invert is involution: a.invert().invert() == a
    // =========================================================================
    #[test]
    fn prop_invert_is_involution(seed in 0u64..100_000) {
        let a = BinaryHV::random(seed);
        prop_assert_eq!(a.invert().invert(), a, "double-invert should be identity");
    }

    // =========================================================================
    // 21. Density of random is ~0.5: (density - 0.5).abs() < 0.05
    // =========================================================================
    #[test]
    fn prop_random_density_near_half(seed in 0u64..100_000) {
        let a = BinaryHV::random(seed);
        let density = a.density();
        prop_assert!(
            (density - 0.5).abs() < 0.05,
            "random vector density should be ~0.5, got {}", density
        );
    }
}
