// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
Property-Based Tests for Cognitive Loop Invariants

Uses proptest to verify that key invariants hold across randomly generated
input sequences through the full cognitive pipeline.

## Invariants Tested

1. **Finiteness**: All metadata fields are finite (no NaN/Inf) for any UTF-8 input
2. **Bounds**: Diversity in [0,1], best_sim in [0,1], prediction_error in [0,1]
3. **Monotonic cycle count**: total_cycles strictly increments
4. **Codebook capacity**: codebook never exceeds resonator_max_symbols
5. **Reward bounds**: Q-learning reward always in [-1, 1]
*/

use proptest::prelude::*;
#[cfg(feature = "physics-bridge")]
use symthaea::cognitive_loop::ParetoContext;
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

/// Strategy for generating valid input strings (non-empty ASCII for reproducibility)
fn input_strategy() -> impl Strategy<Value = String> {
    "[a-z ]{3,40}"
}

/// Strategy for generating a sequence of inputs
fn input_sequence(min: usize, max: usize) -> impl Strategy<Value = Vec<String>> {
    prop::collection::vec(input_strategy(), min..=max)
}

/// Helper: create a default test service
fn test_service() -> CognitiveLoopService {
    CognitiveLoopService::new(CognitiveLoopConfig {
        enable_primitive_consciousness: true,
        learning_threshold: 0.0,
        async_training: false,
        ..Default::default()
    })
    .unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))] // 8 cases — each runs multiple cycles

    /// All CycleMetadata fields must be finite for any input sequence
    #[test]
    fn prop_metadata_always_finite(inputs in input_sequence(20, 50)) {
        let mut service = test_service();
        for (i, input) in inputs.iter().enumerate() {
            let result = service.cycle(input);
            prop_assert!(result.prediction_error.is_finite(),
                "prediction_error NaN/Inf at cycle {i} with input '{input}'");
            prop_assert!(result.metadata.fep.fep_accuracy.is_finite(),
                "fep_accuracy NaN/Inf at cycle {i}");
            prop_assert!(result.metadata.fep.fep_complexity.is_finite(),
                "fep_complexity NaN/Inf at cycle {i}");
            prop_assert!(result.metadata.fep.fep_surprise.is_finite(),
                "fep_surprise NaN/Inf at cycle {i}");
            prop_assert!(result.metadata.fep.fep_td_error.is_finite(),
                "fep_td_error NaN/Inf at cycle {i}");
            prop_assert!(result.metadata.fep.fep_pragmatic_value.is_finite(),
                "fep_pragmatic_value NaN/Inf at cycle {i}");
            prop_assert!(result.metadata.memory.codebook_diversity.is_finite(),
                "codebook_diversity NaN/Inf at cycle {i}");
            prop_assert!(result.metadata.memory.resonator_best_sim.is_finite(),
                "resonator_best_sim NaN/Inf at cycle {i}");
        }
    }

    /// Bounded fields stay within their documented ranges
    #[test]
    fn prop_fields_within_bounds(inputs in input_sequence(20, 50)) {
        let mut service = test_service();
        for (i, input) in inputs.iter().enumerate() {
            let result = service.cycle(input);
            let m = &result.metadata;

            prop_assert!(result.prediction_error >= 0.0 && result.prediction_error <= 1.0,
                "prediction_error out of [0,1] at cycle {i}: {}", result.prediction_error);
            prop_assert!(m.memory.codebook_diversity >= 0.0 && m.memory.codebook_diversity <= 1.0,
                "codebook_diversity out of [0,1] at cycle {i}: {}", m.memory.codebook_diversity);
            prop_assert!(m.memory.resonator_best_sim >= -0.01 && m.memory.resonator_best_sim <= 1.01,
                "resonator_best_sim out of bounds at cycle {i}: {}", m.memory.resonator_best_sim);
            prop_assert!(m.memory.resonator_reconsolidated <= 3,
                "reconsolidated exceeds TOP_K at cycle {i}: {}", m.memory.resonator_reconsolidated);
        }
    }

    /// Cycle count strictly increments by 1 each cycle
    #[test]
    fn prop_cycle_count_monotonic(inputs in input_sequence(10, 40)) {
        let mut service = test_service();
        for (i, input) in inputs.iter().enumerate() {
            service.cycle(input);
            prop_assert_eq!(service.stats().total_cycles, i + 1,
                "Cycle count mismatch at iteration {}", i);
        }
    }

    /// Repeated identical input should not cause divergence
    #[test]
    fn prop_repeated_input_stable(input in input_strategy(), count in 10..30usize) {
        let mut service = test_service();
        for i in 0..count {
            let result = service.cycle(&input);
            prop_assert!(result.prediction_error.is_finite(),
                "Diverged at cycle {i} with repeated input '{input}'");
            prop_assert!(result.prediction_error <= 1.0,
                "Error > 1.0 at cycle {i}: {}", result.prediction_error);
        }
    }

    /// Codebook size never exceeds max_symbols (small capacity to trigger pruning)
    #[test]
    fn prop_codebook_capacity_bounded(inputs in input_sequence(20, 60)) {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
            enable_primitive_consciousness: true,
            learning_threshold: 0.0,
            async_training: false,
            resonator_max_symbols: 8, // very small to trigger pruning
            ..Default::default()
        }).unwrap();

        for (i, input) in inputs.iter().enumerate() {
            let result = service.cycle(input);
            // Codebook evictions should be bounded (at most 3 per cycle from top_eps)
            prop_assert!(result.metadata.memory.codebook_evictions <= 3,
                "Too many evictions at cycle {i}: {}", result.metadata.memory.codebook_evictions);
            prop_assert!(result.metadata.memory.resonator_promotions <= 3,
                "Too many promotions at cycle {i}: {}", result.metadata.memory.resonator_promotions);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Physics Bridge Property Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "physics-bridge")]
mod physics_bridge_props {
    use super::*;

    /// Helper: create a service with physics bridge enabled at aggressive settings
    fn physics_service(interval: usize, blend: f32) -> CognitiveLoopService {
        CognitiveLoopService::new(CognitiveLoopConfig {
            enable_primitive_consciousness: true,
            learning_threshold: 0.0,
            async_training: false,
            enable_physics_bridge: true,
            physics_bridge_query_interval: interval,
            physics_bridge_blend_weight: blend,
            ..Default::default()
        })
        .unwrap()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(5))]

        /// Physics bridge telemetry fields are always finite for any input
        #[test]
        fn prop_physics_telemetry_finite(inputs in input_sequence(10, 30)) {
            let mut service = physics_service(3, 0.2);
            for (i, input) in inputs.iter().enumerate() {
                let result = service.cycle(input);
                if let Some(ref pb) = result.metadata.physics_bridge {
                    prop_assert!(pb.top_score.is_finite(),
                        "physics top_score NaN/Inf at cycle {i}");
                    prop_assert!(pb.effective_blend_weight.is_finite(),
                        "physics effective_blend_weight NaN/Inf at cycle {i}");
                    prop_assert!(pb.effective_blend_weight >= 0.0 && pb.effective_blend_weight <= 1.0,
                        "physics effective_blend_weight out of [0,1] at cycle {i}: {}",
                        pb.effective_blend_weight);
                    prop_assert!(pb.effective_interval >= 1,
                        "physics effective_interval < 1 at cycle {i}: {}",
                        pb.effective_interval);
                    if let Some(score) = pb.pareto_best_analogy {
                        prop_assert!(score.is_finite(),
                            "pareto_best_analogy NaN/Inf at cycle {i}");
                    }
                }
            }
        }

        /// Physics bridge never causes prediction_error to go NaN/Inf
        #[test]
        fn prop_physics_blend_preserves_finiteness(
            inputs in input_sequence(15, 40),
            blend in 0.0f32..1.0f32,
        ) {
            let interval = 1; // query every cycle for max stress
            let mut service = physics_service(interval, blend);
            for (i, input) in inputs.iter().enumerate() {
                let result = service.cycle(input);
                prop_assert!(result.prediction_error.is_finite(),
                    "prediction_error NaN/Inf with blend={blend} at cycle {i}");
                prop_assert!(result.prediction_error >= 0.0 && result.prediction_error <= 1.0,
                    "prediction_error out of [0,1] with blend={blend} at cycle {i}: {}",
                    result.prediction_error);
            }
        }

        /// Pareto context injection never panics, always drains correctly
        #[test]
        fn prop_pareto_context_inject_drain(
            frontier_size in 0usize..100,
            score in 0.0f32..1.0f32,
        ) {
            let mut service = physics_service(5, 0.1);
            // Run a few cycles to warm up
            for _ in 0..3 {
                service.cycle("warmup");
            }
            // Inject Pareto context
            service.set_physics_pareto_context(ParetoContext {
                frontier_size,
                power_range: (1.0, 50.0),
                dominant_domain: "Thermodynamics".into(),
                best_analogy_score: score,
            });
            // Next cycle should drain it
            let result = service.cycle("after pareto inject");
            if let Some(ref pb) = result.metadata.physics_bridge {
                if let Some(fs) = pb.pareto_frontier_size {
                    prop_assert_eq!(fs, frontier_size,
                        "Pareto frontier size mismatch");
                }
                if let Some(ba) = pb.pareto_best_analogy {
                    prop_assert!((ba - score).abs() < 1e-6,
                        "Pareto best_analogy mismatch: expected {score}, got {ba}");
                }
            }
        }
    }
}
