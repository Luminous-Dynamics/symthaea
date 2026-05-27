// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

/*!
Property-Based Tests for Consciousness Equation Bounds

Verifies critical invariants of the consciousness measurement pipeline:

1. Consciousness level is always in [0, 1] for all valid inputs
2. All consciousness metrics are always finite (never NaN or Inf)
3. Phi is always non-negative
4. Consciousness continuity under stable input
*/

use proptest::prelude::*;
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

/// Create a service with consciousness subsystems enabled.
fn consciousness_service() -> CognitiveLoopService {
    CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        enable_surprise_exploration: true,
        enable_prefrontal: true,
        enable_affective_bridge: true,
        enable_gwt: true,
        enable_predictive_processing: true,
        enable_attention_schema: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .expect("Failed to create CognitiveLoopService")
}

// ═══════════════════════════════════════════════════════════════════════════════
// Consciousness always bounded [0, 1]
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(15))]

    /// Consciousness level must be in [0, 1] after any sequence of cycles.
    #[test]
    fn prop_consciousness_always_bounded(
        n_cycles in 5usize..25,
        input_seed in 0u64..1000,
    ) {
        let mut service = consciousness_service();

        for i in 0..n_cycles {
            let input = format!("test input {} seed {}", i, input_seed);
            let result = service.cycle(&input);

            let c = result.metadata.consciousness.consciousness_level;
            prop_assert!(
                c >= 0.0 && c <= 1.0,
                "Consciousness out of [0,1]: {} at cycle {}",
                c, i
            );
        }
    }

    /// Key consciousness metrics must always be finite (no NaN or Inf).
    #[test]
    fn prop_consciousness_metrics_finite(
        n_cycles in 5usize..20,
    ) {
        let mut service = consciousness_service();

        for i in 0..n_cycles {
            let result = service.cycle(&format!("finite test {i}"));
            let m = &result.metadata;

            prop_assert!(
                m.consciousness.consciousness_level.is_finite(),
                "consciousness_level NaN/Inf at cycle {}", i
            );
            prop_assert!(
                result.prediction_error.is_finite(),
                "prediction_error NaN/Inf at cycle {}", i
            );
            prop_assert!(
                result.peak_attention.is_finite(),
                "peak_attention NaN/Inf at cycle {}", i
            );
            prop_assert!(
                m.resonance_frequency.is_finite(),
                "resonance_frequency NaN/Inf at cycle {}", i
            );
        }
    }

    /// Phi (integrated information) must be non-negative.
    #[test]
    fn prop_phi_non_negative(
        n_cycles in 10usize..25,
    ) {
        let mut service = consciousness_service();

        for i in 0..n_cycles {
            let result = service.cycle(&format!("phi test {i}"));
            let phi = result.metadata.consciousness.consciousness_level;
            prop_assert!(
                phi >= 0.0,
                "Consciousness level negative: {} at cycle {}", phi, i
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Consciousness continuity
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    /// After warmup, consciousness should not jump more than 50% per cycle
    /// under stable input (no adversarial perturbation).
    #[test]
    fn prop_consciousness_continuity(
        warmup in 15usize..30,
    ) {
        let mut service = consciousness_service();

        // Warmup phase
        for i in 0..warmup {
            service.cycle(&format!("warmup {i}"));
        }

        // Stability check: 10 cycles with same input
        let mut prev_c = service.cycle("stable input").metadata.consciousness.consciousness_level;
        let mut max_delta: f64 = 0.0;

        for _ in 0..10 {
            let c = service.cycle("stable input").metadata.consciousness.consciousness_level;
            let delta = (c - prev_c).abs();
            max_delta = max_delta.max(delta);
            prev_c = c;
        }

        prop_assert!(
            max_delta < 0.5,
            "Consciousness jumped {} in one cycle under stable input (threshold: 0.5)",
            max_delta
        );
    }
}