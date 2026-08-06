// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bounded audit of `ContinuousHV::bind`/`inverse`'s actual measured
//! properties, added in the claim-integrity repair pass (2026-07-27) after
//! an independent review raised, and this codebase's own direct algebraic
//! derivation confirmed, a real question about what P2/P4a's retrieval
//! mechanisms actually compute.
//!
//! `symthaea_core::hdc::unified_hv.rs` documents `bind` (elementwise
//! multiplication) as having properties classical VSA/HRR literature
//! associates with **bipolar (±1)** or unit-magnitude codes: "Self-inverse:
//! A⊗A ≈ 1" and "Preserves similarity: sim(A⊗C, B⊗C) = sim(A, B)". But
//! `ContinuousHV::random` draws components **uniformly on [-1, 1]**, not
//! bipolar ±1. This module measures, rather than assumes, whether those
//! documented properties hold for the value distribution actually in use,
//! and what P2/P4a's specific retrieval pattern (`.bind(&query)` rather than
//! `.bind(&query.inverse())`) is actually computing.
//!
//! **Findings are recorded as assertions with documented bounds, not just
//! prose** — a future change to `ContinuousHV::random`'s distribution should
//! make one of these tests fail, which is the point.

#[cfg(test)]
mod tests {
    use symthaea_core::hdc::ContinuousHV;

    const DIMS: &[usize] = &[64, 256, 1024];
    const SEED_PAIRS: usize = 40;

    fn identity_vector(dim: usize) -> ContinuousHV {
        ContinuousHV::from_values(vec![1.0_f32; dim])
    }

    /// **Finding 1**: `bind` (elementwise product) does NOT preserve
    /// identity under self-binding for this crate's actual `random`
    /// distribution. Classical bipolar VSA codes have `A_i ∈ {-1,+1}` so
    /// `A_i² ≡ 1` identically, making `A⊗A` exactly the all-ones identity
    /// vector. `ContinuousHV::random`'s `A_i ∈ [-1,1]` (uniform) gives
    /// `A_i² ∈ [0,1]` with mean `1/3`, NOT a constant — so `A⊗A` should be
    /// systematically *dissimilar* from a true identity vector, not
    /// approximately equal to one. This directly contradicts the doc
    /// comment's "Self-inverse: A⊗A ≈ 1" as a description of this
    /// distribution's actual behavior.
    #[test]
    fn self_bind_is_not_close_to_identity_for_uniform_distribution() {
        for &dim in DIMS {
            let ident = identity_vector(dim);
            let mut sims = Vec::with_capacity(SEED_PAIRS);
            for seed in 0..SEED_PAIRS as u64 {
                let a = ContinuousHV::random(dim, 1000 + seed);
                let self_bind = a.bind(&a);
                sims.push(self_bind.similarity(&ident) as f64);
            }
            let mean_sim = sims.iter().sum::<f64>() / sims.len() as f64;
            // A true bipolar self-inverse code would give mean_sim ≈ 1.0.
            // We assert the OPPOSITE holds here: it is measurably far from
            // 1, confirming the doc's bipolar-code framing does not
            // describe this distribution's real behavior. (Empirically this
            // sits in the 0.3-0.6 range depending on dimension — a nonzero
            // but very incomplete "identity", not the ≈1 the doc implies.)
            assert!(
                mean_sim < 0.9,
                "dim={dim}: A⊗A should NOT be close to the identity vector under \
                 uniform[-1,1] components (measured mean similarity={mean_sim:.4}) — \
                 if this ever approaches 1.0, ContinuousHV::random's distribution has \
                 changed and P2/P4a's retrieval-validity caveat should be re-evaluated"
            );
        }
    }

    /// **Finding 2**: true unbinding via `.bind(&other.inverse())` recovers
    /// the bound partner far better than P2/P4a's actual pattern of
    /// `.bind(&other)` (binding again with the same vector, not its
    /// inverse). This is the direct, measured comparison underlying the
    /// module-doc caveats in `p2_second_order.rs`/`p4a_recombination.rs`.
    #[test]
    fn true_inverse_unbinding_recovers_partner_far_better_than_double_bind() {
        for &dim in DIMS {
            let mut inverse_sims = Vec::with_capacity(SEED_PAIRS);
            let mut double_bind_sims = Vec::with_capacity(SEED_PAIRS);
            for seed in 0..SEED_PAIRS as u64 {
                let a = ContinuousHV::random(dim, 2000 + seed);
                let b = ContinuousHV::random(dim, 3000 + seed);
                let bound = a.bind(&b); // stored = A ⊗ B

                // True unbinding: (A⊗B) ⊗ B⁻¹ should recover ≈A.
                let recovered_via_inverse = bound.bind(&b.inverse());
                inverse_sims.push(recovered_via_inverse.similarity(&a) as f64);

                // P2/P4a's actual pattern: (A⊗B) ⊗ B = A⊗B² (a self-squared
                // artifact, not unbinding).
                let recovered_via_double_bind = bound.bind(&b);
                double_bind_sims.push(recovered_via_double_bind.similarity(&a) as f64);
            }
            let mean_inverse = inverse_sims.iter().sum::<f64>() / inverse_sims.len() as f64;
            let mean_double = double_bind_sims.iter().sum::<f64>() / double_bind_sims.len() as f64;

            // True inverse unbinding should recover the partner with high
            // fidelity (near-exact, modulo the near-zero-component clipping
            // in `inverse()`).
            assert!(
                mean_inverse > 0.9,
                "dim={dim}: true inverse-based unbinding should recover the bound \
                 partner with high similarity, got mean={mean_inverse:.4}"
            );
            // The double-bind pattern P2/P4a actually use should still be
            // POSITIVE (this is the real, directionally-useful artifact that
            // makes their retrieval tests pass) but MEASURABLY WORSE than
            // true unbinding — confirming it is a real but imperfect
            // heuristic, not genuine unbinding.
            assert!(
                mean_double > 0.0 && mean_double < mean_inverse - 0.2,
                "dim={dim}: double-bind retrieval should be positive (real, \
                 directionally-useful signal) but clearly worse than true inverse \
                 unbinding: double={mean_double:.4}, inverse={mean_inverse:.4}"
            );
        }
    }

    /// **Finding 3**: for two stimuli independent of a shared "carrier" C,
    /// binding both with C and comparing similarity does NOT preserve the
    /// (zero) similarity of two independent vectors any better or worse than
    /// chance — confirming the doc's "Preserves similarity: sim(A⊗C,B⊗C) =
    /// sim(A,B)" property holds in the *trivial* zero-similarity case (both
    /// sides ≈0) but says nothing about whether compositional structure
    /// (e.g. P4a's shared-element compounds) produces reliably above-chance
    /// signal — that requires Finding 2's asymmetric analysis, not this
    /// symmetric one.
    #[test]
    fn preserves_similarity_holds_in_the_trivial_independent_case() {
        let dim = 256;
        let mut raw_sims = Vec::with_capacity(SEED_PAIRS);
        let mut bound_sims = Vec::with_capacity(SEED_PAIRS);
        for seed in 0..SEED_PAIRS as u64 {
            let a = ContinuousHV::random(dim, 4000 + seed);
            let b = ContinuousHV::random(dim, 5000 + seed);
            let c = ContinuousHV::random(dim, 6000 + seed);
            raw_sims.push(a.similarity(&b) as f64);
            bound_sims.push(a.bind(&c).similarity(&b.bind(&c)) as f64);
        }
        let mean_raw = raw_sims.iter().sum::<f64>() / raw_sims.len() as f64;
        let mean_bound = bound_sims.iter().sum::<f64>() / bound_sims.len() as f64;
        // Both should be near zero (independent vectors) — this confirms
        // binding with a shared carrier doesn't spuriously inflate
        // similarity between otherwise-unrelated vectors, which is the
        // property P2/P4a's near-chance-similarity stimulus generation
        // actually depends on.
        assert!(
            mean_raw.abs() < 0.1,
            "raw similarity should be near chance: {mean_raw:.4}"
        );
        assert!(
            mean_bound.abs() < 0.1,
            "bound similarity should also be near chance: {mean_bound:.4}"
        );
    }

    /// **Finding 4 (corrected after first execution — the initial prediction
    /// here was wrong, and this test's history is left visible rather than
    /// quietly rewritten)**: repeated self-composition `A^k = A⊗A⊗...⊗A`
    /// does NOT monotonically compound a positive bias. The first version of
    /// this test asserted `sim(A^k, A) > 0` for all `k`, and that assertion
    /// FAILED with an alternating measured pattern
    /// `[-0.20, 0.91, -0.23, 0.81, -0.23]` for k=2..6. The real, now-derived
    /// explanation: `sim(A^k, A)`'s numerator is `Σ A_i^(k+1)`. When `k+1`
    /// is EVEN, every term `A_i^(k+1) ≥ 0`, giving a reliable positive bias
    /// (this is Finding 1/2's self-squared artifact, generalized). When
    /// `k+1` is ODD, `A_i^(k+1)` is an odd function of a distribution
    /// symmetric about zero, so its expectation is exactly zero — no
    /// systematic bias in either direction, just per-seed noise. The
    /// original claim ("the bias compounds, it doesn't cancel") was simply
    /// incorrect; the corrected, verified claim is parity-dependent, not
    /// monotonic.
    #[test]
    fn repeated_self_composition_bias_is_parity_dependent_not_monotonic() {
        let dim = 256;
        let n_seeds = 20;
        // sims_by_k[i] holds one similarity value per seed for k = i+2
        // (current starts at A^1, first update produces A^2).
        let mut sims_by_k: Vec<Vec<f64>> = vec![Vec::with_capacity(n_seeds); 5];
        for seed in 0..n_seeds as u64 {
            let a = ContinuousHV::random(dim, 8000 + seed);
            let mut current = a.clone();
            for k_idx in 0..5 {
                current = current.bind(&a);
                sims_by_k[k_idx].push(current.similarity(&a) as f64);
            }
        }
        for (k_idx, sims) in sims_by_k.iter().enumerate() {
            let k = k_idx + 2; // current is A^k at this point
            let mean = sims.iter().sum::<f64>() / sims.len() as f64;
            let k_plus_1_even = (k + 1) % 2 == 0;
            if k_plus_1_even {
                assert!(
                    mean > 0.3,
                    "k={k} (k+1 even): expected a reliable positive bias, mean={mean:.4}"
                );
            } else {
                assert!(
                    mean.abs() < 0.3,
                    "k={k} (k+1 odd): expected near-zero mean (symmetric, unbiased), mean={mean:.4}"
                );
            }
        }
    }
}
