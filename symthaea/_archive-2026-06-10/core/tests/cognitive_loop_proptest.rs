// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
Property-Based Tests for Cognitive Loop Invariants

Uses proptest to verify that the cognitive loop maintains its invariants
across randomly generated inputs and configurations.

## Key Invariants Tested

1. **Error bounded**: prediction_error is always in [0.0, 1.0]
2. **No panics**: cycle() never panics regardless of input
3. **Stats monotonic**: total_cycles always increases
4. **Output dimensions**: output Vec is always 256 elements (num_neurons)
5. **Timing valid**: cycle_time_us > 0
6. **Metadata consistent**: all metadata fields are valid
*/

use proptest::prelude::*;
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

/// Strategy for generating random ASCII text inputs
fn arbitrary_input() -> impl Strategy<Value = String> {
    prop::string::string_regex("[a-zA-Z0-9 .,!?]{1,200}")
        .unwrap()
        .prop_filter("non-empty", |s| !s.is_empty())
}

/// Strategy for generating valid CognitiveLoopConfig
fn arbitrary_config() -> impl Strategy<Value = CognitiveLoopConfig> {
    (
        0.0f32..0.5,     // learning_threshold
        prop::bool::ANY, // async_training
        prop::bool::ANY, // enable_surprise_exploration
        prop::bool::ANY, // enable_prefrontal
    )
        .prop_map(
            |(learning_threshold, async_training, surprise, prefrontal)| CognitiveLoopConfig {
                learning_threshold,
                async_training,
                enable_surprise_exploration: surprise,
                enable_prefrontal: prefrontal,
                ..Default::default()
            },
        )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    /// INV: prediction_error is always in [0.0, 1.0]
    #[test]
    fn error_bounded(input in arbitrary_input()) {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
        let result = service.cycle(&input);
        prop_assert!(result.prediction_error >= 0.0, "error < 0: {}", result.prediction_error);
        prop_assert!(result.prediction_error <= 1.0, "error > 1: {}", result.prediction_error);
    }

    /// INV: cycle never panics on arbitrary text
    #[test]
    fn no_panics(input in arbitrary_input()) {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
        let _result = service.cycle(&input);
        // If we reach here, no panic occurred
    }

    /// INV: stats.total_cycles increments by 1 per cycle
    #[test]
    fn stats_monotonic(inputs in prop::collection::vec(arbitrary_input(), 1..10)) {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
        for (i, input) in inputs.iter().enumerate() {
            service.cycle(input);
            prop_assert_eq!(
                service.stats().total_cycles,
                i + 1,
                "total_cycles mismatch after cycle {}",
                i
            );
        }
    }

    /// INV: output dimension matches config.cfc_config.num_neurons (256)
    #[test]
    fn output_dimensions(input in arbitrary_input()) {
        let config = CognitiveLoopConfig::default();
        let expected_dim = config.cfc_config.num_neurons;
        let mut service = CognitiveLoopService::new(config).unwrap();
        let result = service.cycle(&input);
        prop_assert_eq!(
            result.output.len(),
            expected_dim,
            "output dimension mismatch"
        );
    }

    /// INV: cycle_time_us is always > 0
    #[test]
    fn timing_valid(input in arbitrary_input()) {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
        let result = service.cycle(&input);
        prop_assert!(result.cycle_time_us > 0, "cycle_time_us must be > 0");
    }

    /// INV: reasoning_confidence and plan_confidence are non-negative
    #[test]
    fn metadata_consistent(input in arbitrary_input()) {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
        let result = service.cycle(&input);
        prop_assert!(
            result.metadata.reasoning_confidence >= 0.0,
            "reasoning_confidence < 0"
        );
        prop_assert!(
            result.metadata.reasoning_plan_confidence >= 0.0,
            "plan_confidence < 0"
        );
    }

    /// INV: service works with random config combinations
    #[test]
    fn arbitrary_config_no_panic(
        config in arbitrary_config(),
        input in arbitrary_input(),
    ) {
        let mut service = CognitiveLoopService::new(config).unwrap();
        let result = service.cycle(&input);
        prop_assert!(result.prediction_error >= 0.0);
        prop_assert!(result.prediction_error <= 1.0);
    }

    /// INV: output contains only finite values (no NaN, no Inf)
    #[test]
    fn output_finite(input in arbitrary_input()) {
        let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
        let result = service.cycle(&input);
        for (i, val) in result.output.iter().enumerate() {
            prop_assert!(val.is_finite(), "output[{}] = {} (not finite)", i, val);
        }
    }
}