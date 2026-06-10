// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Property-based tests for HDC encoding and causal learning.
//!
//! Uses proptest to verify invariants that must hold for any input:
//! - Encoding determinism: same input → same vector
//! - Hebbian convergence: repeated co-occurrence strengthens edges
//! - Drift monotonicity: more dissimilar observations → higher drift
//! - Rollback completeness: all critical steps rolled back on failure

use proptest::prelude::*;

use symthaea_core::hdc::ContinuousHV;
use symthaea_nix::encoding::{NixCodebook, UserInputEncoder};
use symthaea_nix::mind::HdcWorldModel;
use symthaea_nix::mind::causal_graph::NixCausalGraph;

// ── Encoding Determinism ─────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Same token always produces the same vector in a codebook.
    #[test]
    fn encoding_determinism(token in "[a-z]{2,12}") {
        let mut cb1 = NixCodebook::with_dim(256);
        let mut cb2 = NixCodebook::with_dim(256);

        let hv1 = cb1.get_or_create(&token).clone();
        let hv2 = cb2.get_or_create(&token).clone();

        let sim = hv1.similarity(&hv2);
        prop_assert!((sim - 1.0).abs() < 1e-5,
            "Same token '{}' should produce identical vectors, got similarity {}", token, sim);
    }

    /// Different tokens produce quasi-orthogonal vectors.
    #[test]
    fn different_tokens_quasi_orthogonal(
        a in "[a-z]{3,10}",
        b in "[a-z]{3,10}",
    ) {
        prop_assume!(a != b);

        let mut cb = NixCodebook::with_dim(256);
        let hv_a = cb.get_or_create(&a).clone();
        let hv_b = cb.get_or_create(&b).clone();

        let sim = hv_a.similarity(&hv_b).abs();
        // In 256-d space, random vectors have ~0.06 expected similarity
        prop_assert!(sim < 0.5,
            "Different tokens '{}' and '{}' should be quasi-orthogonal, got {}", a, b, sim);
    }

    /// Option path encoding is level-sensitive: same segment at different
    /// levels produces different vectors.
    #[test]
    fn level_sensitivity(
        segment in "[a-z]{2,8}",
        level_a in 0u8..7,
        level_b in 0u8..7,
    ) {
        prop_assume!(level_a != level_b);

        let mut cb = NixCodebook::with_dim(256);
        let hv_a = cb.encode_segment(&segment, level_a as usize);
        let hv_b = cb.encode_segment(&segment, level_b as usize);

        let sim = hv_a.similarity(&hv_b).abs();
        prop_assert!(sim < 0.5,
            "Segment '{}' at levels {} and {} should differ, got similarity {}",
            segment, level_a, level_b, sim);
    }

    /// User input encoding is deterministic.
    #[test]
    fn input_encoding_determinism(input in "[a-z]{3,8}( [a-z]{3,8}){1,4}") {
        let mut cb1 = NixCodebook::with_dim(256);
        let mut cb2 = NixCodebook::with_dim(256);

        let mut enc1 = UserInputEncoder::new(&mut cb1);
        let hv1 = enc1.encode_input(&input);

        let mut enc2 = UserInputEncoder::new(&mut cb2);
        let hv2 = enc2.encode_input(&input);

        let sim = hv1.similarity(&hv2);
        prop_assert!((sim - 1.0).abs() < 1e-4,
            "Input '{}' should encode deterministically, got similarity {}", input, sim);
    }
}

// ── Hebbian Learning ─────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    /// Repeated co-occurrence monotonically strengthens an edge.
    #[test]
    fn hebbian_strengthening_monotonic(
        initial_confidence in 0.1f64..0.8,
        repetitions in 2u32..10,
    ) {
        let mut graph = NixCausalGraph::new(42);
        graph.add_structural_edge("A", "B", initial_confidence);

        let mut prev_confidence = initial_confidence;
        for _ in 0..repetitions {
            graph.observe_outcome("A", &["B"], &["B"]);
            let conf = graph.edge_confidence("A", "B").unwrap();
            prop_assert!(conf >= prev_confidence,
                "Confidence should monotonically increase: {} -> {}", prev_confidence, conf);
            prev_confidence = conf;
        }

        // After repeated strengthening, confidence should exceed initial
        prop_assert!(prev_confidence > initial_confidence,
            "After {} repetitions, confidence {} should exceed initial {}",
            repetitions, prev_confidence, initial_confidence);
    }

    /// Repeated non-occurrence monotonically weakens an edge.
    #[test]
    fn hebbian_weakening_monotonic(
        initial_confidence in 0.2f64..0.9,
        repetitions in 2u32..10,
    ) {
        let mut graph = NixCausalGraph::new(42);
        graph.add_structural_edge("A", "B", initial_confidence);

        let mut prev_confidence = initial_confidence;
        for _ in 0..repetitions {
            graph.observe_outcome("A", &[], &["B"]);
            // Edge may have been pruned
            if let Some(conf) = graph.edge_confidence("A", "B") {
                prop_assert!(conf <= prev_confidence,
                    "Confidence should monotonically decrease: {} -> {}", prev_confidence, conf);
                prev_confidence = conf;
            } else {
                break; // Pruned — definitely decreased
            }
        }
    }
}

// ── World Model Drift ────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    /// Observing the expected state should not trigger drift.
    #[test]
    fn no_drift_when_at_expected(seed in 1u64..1000) {
        let dim = 256;
        let mut wm = HdcWorldModel::new(dim);
        let expected = ContinuousHV::random(dim, seed);
        wm.set_expected_state(expected.clone());
        wm.observe(&expected);

        let report = wm.detect_drift(0.8);
        prop_assert!(!report.drifted,
            "Should not drift when at expected state: similarity={}", report.similarity);
    }

    /// More observations of the expected state keep drift low.
    #[test]
    fn repeated_expected_observations_low_drift(
        seed in 1u64..1000,
        n_obs in 2u32..10,
    ) {
        let dim = 256;
        let mut wm = HdcWorldModel::new(dim);
        let expected = ContinuousHV::random(dim, seed);
        wm.set_expected_state(expected.clone());

        for _ in 0..n_obs {
            wm.observe(&expected);
        }

        let report = wm.detect_drift(0.8);
        prop_assert!(!report.drifted,
            "Should not drift after {} observations of expected state: similarity={}",
            n_obs, report.similarity);
    }

    /// EMA alpha in (0, 1] is respected — observation count always increments.
    #[test]
    fn observation_count_increments(
        alpha in 0.01f32..1.0,
        n_obs in 1u32..20,
    ) {
        let dim = 128;
        let mut wm = HdcWorldModel::new(dim).with_ema_alpha(alpha);

        for i in 0..n_obs {
            wm.observe(&ContinuousHV::random(dim, i as u64 + 1));
        }

        prop_assert_eq!(wm.observation_count(), n_obs as usize);
    }
}
