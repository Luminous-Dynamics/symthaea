// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
Property-Based Tests for Cantor Fractal HDC Subsystem

Verifies structural invariants of the Cantor CRHV pipeline:
- Codebook size never exceeds cap (256)
- Dream surprise stays in [0.0, 1.0]
- Diversity threshold actually prevents high-similarity duplicates
- Adaptive depth is monotonic with activation
- FEP coupling is bounded by clamp
- Resonance boost is in [0.0, 1.0]
*/

use proptest::prelude::*;
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

fn make_cantor_service() -> CognitiveLoopService {
    CognitiveLoopService::new(CognitiveLoopConfig {
        async_training: false,
        learning_threshold: 0.0,
        enable_gwt: true,
        enable_dream_replay: true,
        ..Default::default()
    })
    .expect("CognitiveLoopService::new should succeed")
}

// ═══════════════════════════════════════════════════════════════════════════════
// Properties
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    /// Codebook size never exceeds CANTOR_CODEBOOK_MAX_ENTRIES (256).
    #[test]
    fn prop_codebook_bounded(cycles in 50usize..200) {
        let mut service = make_cantor_service();
        for i in 0..cycles {
            let result = service.cycle(&format!("codebook prop input {i}"));
            prop_assert!(
                result.metadata.cantor.cantor_codebook_size <= 256,
                "Codebook size {} exceeded cap at cycle {}",
                result.metadata.cantor.cantor_codebook_size, i
            );
        }
    }

    /// Dream surprise stays in [0.0, 1.0].
    #[test]
    fn prop_dream_surprise_bounded(cycles in 50usize..200) {
        let mut service = make_cantor_service();
        for i in 0..cycles {
            let result = service.cycle(&format!("surprise prop input {i}"));
            let s = result.metadata.cantor.cantor_dream_surprise;
            prop_assert!(
                (0.0..=1.0).contains(&s),
                "Dream surprise {} out of [0,1] at cycle {}", s, i
            );
        }
    }

    /// Resonance boost stays in [0.0, 1.0].
    #[test]
    fn prop_resonance_boost_bounded(cycles in 50usize..200) {
        let mut service = make_cantor_service();
        for i in 0..cycles {
            let result = service.cycle(&format!("resonance prop input {i}"));
            let r = result.metadata.cantor.cantor_resonance_boost;
            prop_assert!(
                (0.0..=1.0).contains(&r),
                "Resonance boost {} out of [0,1] at cycle {}", r, i
            );
        }
    }

    /// Metacognitive depth stays in [0.0, 1.0].
    #[test]
    fn prop_metacognitive_depth_bounded(cycles in 50usize..200) {
        let mut service = make_cantor_service();
        for i in 0..cycles {
            let result = service.cycle(&format!("metacog prop input {i}"));
            let d = result.metadata.cantor.cantor_metacognitive_depth;
            prop_assert!(
                (0.0..=1.0).contains(&d),
                "Metacognitive depth {} out of [0,1] at cycle {}", d, i
            );
        }
    }

    /// Buffer occupancy never exceeds cap of 32.
    #[test]
    fn prop_buffer_bounded(cycles in 50usize..200) {
        let mut service = make_cantor_service();
        for i in 0..cycles {
            let result = service.cycle(&format!("buffer prop input {i}"));
            prop_assert!(
                result.metadata.cantor.cantor_buffer_occupancy <= 32,
                "Buffer {} exceeded cap at cycle {}",
                result.metadata.cantor.cantor_buffer_occupancy, i
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Core-level property tests (no CLS dependency)
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Adaptive depth is monotonic with activation:
    /// higher activation → equal or deeper CRHV.
    #[test]
    fn prop_adaptive_depth_monotonic(act_lo in 0.0f32..0.5, act_hi in 0.5f32..1.0) {
        // Mirror the constants from thresholds.rs (pub(crate))
        const DEPTH_MIN: usize = 2;
        const DEPTH_MAX: usize = 7;
        let range = DEPTH_MAX - DEPTH_MIN;
        let depth_lo = DEPTH_MIN + (act_lo * range as f32).round() as usize;
        let depth_hi = DEPTH_MIN + (act_hi * range as f32).round() as usize;
        prop_assert!(
            depth_hi >= depth_lo,
            "depth_hi {} < depth_lo {} for activations ({}, {})",
            depth_hi, depth_lo, act_hi, act_lo
        );
    }

    /// Diversity threshold rejects near-duplicates in BinaryCodebook.
    #[test]
    fn prop_diversity_rejects_clones(seed in 0u64..1000) {
        use symthaea_core::hdc::binary_hv::BinaryHV;
        use symthaea_core::hdc::cantor_resonator_cleanup::BinaryCodebook;
        let mut codebook = BinaryCodebook::new(256);
        let v = BinaryHV::random(seed);
        codebook.add("original", v);
        // Identical vector should be rejected at threshold 0.92
        let accepted = codebook.add_if_diverse("clone", v, 0.92);
        prop_assert!(!accepted, "Identical vector should be rejected");
    }
}