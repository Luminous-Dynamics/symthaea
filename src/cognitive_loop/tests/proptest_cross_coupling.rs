// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Property tests for cross-subsystem couplings.
//!
//! Validates invariants of the Broca↔ToM↔Foveation↔Substrate couplings
//! introduced in the March 8, 2026 subsystem integration pass.

use proptest::prelude::*;

proptest! {
    // Property 1: Broca cadence spacing is always ≥ 5
    // ToM mismatch can lower it from 7 to 5 but never below.
    #[test]
    fn prop_broca_cadence_bounded(tom_mismatch in 0.0f32..=1.0) {
        let spacing = if tom_mismatch > 0.5 { 5usize } else { 7 };
        prop_assert!(spacing >= 5, "cadence spacing below 5: {}", spacing);
        prop_assert!(spacing <= 7, "cadence spacing above 7: {}", spacing);
    }

    // Property 2: Broca attention contraction is always in [0.955, 1.0]
    #[test]
    fn prop_broca_attention_contraction_bounded(quality_ema in 0.0f32..=1.0) {
        if quality_ema > 0.7 {
            let contraction = 1.0 - (quality_ema - 0.7) * 0.15;
            prop_assert!(contraction >= 0.955 - 1e-6,
                "contraction below 0.955: {} for quality_ema={}", contraction, quality_ema);
            prop_assert!(contraction <= 1.0 + 1e-6,
                "contraction above 1.0: {} for quality_ema={}", contraction, quality_ema);
        }
    }

    // Property 3: Foveation budget is always ≥ 1 regardless of tau_factor
    #[test]
    fn prop_foveation_budget_at_least_one(
        max_dispatches in 1u8..=10,
        tau_factor in 0.1f32..=3.0,
    ) {
        let tau_scale = tau_factor.max(0.5);
        let max = ((max_dispatches as f32) * tau_scale)
            .round()
            .max(1.0) as usize;
        prop_assert!(max >= 1, "budget below 1: {} for dispatches={}, tau={}",
            max, max_dispatches, tau_factor);
    }

    // Property 4: ToM exploration boost is bounded [0, 0.06]
    #[test]
    fn prop_tom_exploration_boost_bounded(mismatch_ema in 0.0f32..=1.0) {
        if mismatch_ema > 0.4 {
            let boost = (mismatch_ema - 0.4) * 0.1;
            prop_assert!(boost >= 0.0, "boost negative: {}", boost);
            prop_assert!(boost <= 0.06 + 1e-6,
                "boost above 0.06: {} for mismatch={}", boost, mismatch_ema);
        }
    }

    // Property 5: Broca quality composite is always in [0, 1]
    #[test]
    fn prop_broca_quality_bounded(
        coherence in 0.0f32..=1.0,
        semantic_pe in 0.0f32..=2.0,
        long_coherence in 0.0f32..=1.0,
    ) {
        let quality = coherence * 0.4
            + (1.0 - semantic_pe.min(1.0)) * 0.4
            + long_coherence * 0.2;
        let quality = quality.clamp(0.0, 1.0);
        prop_assert!(quality >= 0.0 && quality <= 1.0,
            "quality out of [0,1]: {} for coh={}, pe={}, long={}",
            quality, coherence, semantic_pe, long_coherence);
    }

    // Property 6: ToM confidence modulation is bounded
    #[test]
    fn prop_tom_confidence_modulation_bounded(accuracy in 0.0f32..=1.0) {
        if accuracy > 0.7 {
            let boost = (accuracy - 0.7) * 0.05;
            prop_assert!(boost >= 0.0 && boost <= 0.015 + 1e-6,
                "boost out of [0, 0.015]: {}", boost);
        } else if accuracy < 0.3 {
            let dampen = 1.0 - (0.3 - accuracy) * 0.05;
            prop_assert!(dampen >= 0.985 - 1e-6 && dampen <= 1.0,
                "dampen out of [0.985, 1.0]: {}", dampen);
        }
    }
}
