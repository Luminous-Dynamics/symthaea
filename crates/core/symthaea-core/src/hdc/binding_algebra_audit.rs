// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Binding-algebra contract tests for `ContinuousHV` and `BinaryHV`.
//!
//! Added as Phase 1 / Commit B of the "HDC Binding Algebra Qualification and
//! Migration Plan" (2026-07-27), prompted by a real defect found downstream:
//! `symthaea-psych-bench`'s UAL benchmark suite (`crates/domains/
//! symthaea-psych-bench/src/benchmarks/ual/`) found that `ContinuousHV`'s own
//! doc comments claimed bipolar-VSA properties ("Self-inverse: A⊗A ≈ 1",
//! "Preserves similarity: sim(A⊗C, B⊗C) = sim(A, B)") that do not actually
//! hold for `ContinuousHV::random`'s real uniform-`[-1,1]` distribution — see
//! `crates/domains/symthaea-psych-bench/src/benchmarks/ual/hdc_binding_properties.rs`
//! for the original benchmark-local audit and
//! `symthaea/docs/SYMTHAEA_UAL_FROZEN_EVIDENCE_2026-07-27.md` for the frozen
//! before-state this work is measured against.
//!
//! This module holds only the **hard, deterministic contract tests** —
//! properties that must hold exactly (`BinaryHV`'s XOR-based recovery) or
//! structurally (dimension checks, finiteness) regardless of random seed.
//! **Distributional characterization** (recovery-similarity distributions,
//! self-bind-vs-identity distance, inverse-based numerical stability, etc.)
//! deliberately does NOT live here as asserted unit tests — arbitrary
//! sampled-mean thresholds (e.g. "mean similarity > 0.73") make flaky or
//! meaningless regressions. That distributional work is a generated report,
//! not a test suite: see `examples/binding_algebra_characterization.rs` and
//! its output, `docs/BINDING_ALGEBRA_CHARACTERIZATION_REPORT.md`.

#[cfg(test)]
mod contract_tests {
    use crate::hdc::binary_hv::BinaryHV;
    use crate::hdc::unified_hv::ContinuousHV;

    // ---- BinaryHV: exact, deterministic algebraic guarantees ----

    /// `BinaryHV`'s bind is XOR: binding twice with the same operand is
    /// EXACTLY self-inverse, not approximately. This is the real, known-good
    /// case the `ContinuousHV` audit below is contrasted against.
    #[test]
    fn binary_hv_double_bind_exact_recovery() {
        for seed in 0..20u64 {
            let a = BinaryHV::random(seed);
            let b = BinaryHV::random(seed + 1000);
            let bound = a.bind(&b);
            let recovered = bound.bind(&b);
            assert_eq!(
                recovered, a,
                "seed={seed}: XOR double-bind must recover the original exactly"
            );
        }
    }

    /// XOR binding preserves Hamming distance exactly: for any fixed key K,
    /// hamming(A⊕K, B⊕K) == hamming(A, B). This is the algebraic property
    /// that makes `BinaryHV` a genuine bipolar-isomorphic code (see module
    /// doc on `symthaea-psych-bench`'s side for why this doesn't transfer to
    /// `ContinuousHV`'s Hadamard-product binding on non-constant-magnitude
    /// components).
    #[test]
    fn binary_hv_bind_preserves_hamming_distance_exactly() {
        for seed in 0..20u64 {
            let a = BinaryHV::random(seed);
            let b = BinaryHV::random(seed + 1000);
            let k = BinaryHV::random(seed + 2000);
            let base_distance = a.hamming_distance(&b);
            let bound_distance = a.bind(&k).hamming_distance(&b.bind(&k));
            assert_eq!(
                base_distance, bound_distance,
                "seed={seed}: XOR binding with a shared key must preserve Hamming distance exactly"
            );
        }
    }

    // ---- ContinuousHV: structural contracts (finiteness, dimension) ----

    /// `bind` on mismatched dimensions must panic (documented behavior via
    /// `assert_eq!` inside `bind` itself) — pinned here so a future
    /// refactor that silently truncates/pads instead of rejecting would be
    /// caught immediately.
    #[test]
    #[should_panic(expected = "Dimension mismatch")]
    fn continuous_hv_bind_rejects_dimension_mismatch() {
        let a = ContinuousHV::random(64, 1);
        let b = ContinuousHV::random(128, 2);
        let _ = a.bind(&b);
    }

    /// `normalize()` must always return a finite vector for finite input,
    /// across a range of magnitudes including the near-zero-norm guard case.
    #[test]
    fn continuous_hv_normalize_is_always_finite() {
        for seed in 0..20u64 {
            let v = ContinuousHV::random(256, seed);
            let normalized = v.normalize();
            assert!(
                normalized.values.iter().all(|x| x.is_finite()),
                "seed={seed}: normalize() produced a non-finite component"
            );
        }
        // The documented near-zero-norm guard: normalize() on a literal zero
        // vector must return the zero vector unchanged (not NaN from 0/0).
        let zero = ContinuousHV::zero(256);
        let normalized_zero = zero.normalize();
        assert!(
            normalized_zero.values.iter().all(|x| x.is_finite()),
            "normalize() of the zero vector must stay finite, not produce NaN"
        );
    }

    /// `inverse()`'s documented near-zero epsilon (`1e-7`, per its doc
    /// comment) must match what the implementation actually does: components
    /// with `|v| < 1e-7` map to `0.0`, not to a reciprocal.
    #[test]
    fn continuous_hv_inverse_epsilon_matches_documented_threshold() {
        const DOCUMENTED_EPSILON: f32 = 1e-7;
        let below_threshold = ContinuousHV::from_values(vec![
            DOCUMENTED_EPSILON * 0.5,
            -DOCUMENTED_EPSILON * 0.5,
            0.0,
        ]);
        let inv = below_threshold.inverse();
        assert_eq!(
            inv.values,
            vec![0.0, 0.0, 0.0],
            "components below the documented epsilon must map to exactly 0.0, not a reciprocal"
        );

        let above_threshold = ContinuousHV::from_values(vec![DOCUMENTED_EPSILON * 2.0]);
        let inv_above = above_threshold.inverse();
        assert!(
            inv_above.values[0].is_finite() && inv_above.values[0] != 0.0,
            "a component just above the documented epsilon must be inverted, not zeroed"
        );
    }

    /// No unexpected NaN/Infinity propagation through the core operation
    /// chain (`bind` -> `weighted_bundle` -> `normalize` -> `bind` again)
    /// under adversarial near-zero and large-magnitude inputs.
    #[test]
    fn no_nan_or_infinity_propagation_through_core_operation_chain() {
        let dim = 64;
        let near_zero = ContinuousHV::from_values(vec![1e-30_f32; dim]);
        let large = ContinuousHV::from_values(vec![1e30_f32; dim]);
        let normal = ContinuousHV::random(dim, 42);

        let bound = near_zero.bind(&large);
        assert!(
            bound.values.iter().all(|x| x.is_finite()),
            "bind(near_zero, large) produced non-finite components"
        );

        let bundled = ContinuousHV::weighted_bundle(&[&bound, &normal], &[0.5, 0.5]);
        assert!(
            bundled.values.iter().all(|x| x.is_finite()),
            "weighted_bundle of an adversarial pair produced non-finite components"
        );

        let normalized = bundled.normalize();
        assert!(
            normalized.values.iter().all(|x| x.is_finite()),
            "normalize() of an adversarial bundle produced non-finite components"
        );

        let rebound = normalized.bind(&normal);
        assert!(
            rebound.values.iter().all(|x| x.is_finite()),
            "re-binding a normalized adversarial result produced non-finite components"
        );
    }
}
