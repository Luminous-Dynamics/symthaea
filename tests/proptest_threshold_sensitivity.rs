/*!
Threshold Sensitivity Analysis via Property-Based Testing

Verifies that the cognitive loop remains stable when key thresholds are
perturbed within ±wide ranges. This catches brittleness: if a change
causes NaN, panic, or unbounded divergence, the threshold needs tighter
documentation or a wider stability margin.
*/

use proptest::prelude::*;
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

fn perturbed_config() -> impl Strategy<Value = CognitiveLoopConfig> {
    (
        0.01f32..0.3,    // learning_threshold (default ~0.1, ±wide)
        prop::bool::ANY, // enable_surprise_exploration
        prop::bool::ANY, // enable_prefrontal
    )
        .prop_map(|(lt, surprise, prefrontal)| CognitiveLoopConfig {
            learning_threshold: lt,
            async_training: false,
            enable_surprise_exploration: surprise,
            enable_prefrontal: prefrontal,
            enable_primitive_consciousness: true,
            ..Default::default()
        })
}

fn input_sequence(n: usize) -> impl Strategy<Value = Vec<String>> {
    prop::collection::vec("[a-z ]{3,30}", 1..=n)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(6))]

    /// Metadata stays finite and bounded across threshold perturbations.
    #[test]
    fn threshold_perturbation_finiteness(
        config in perturbed_config(),
        inputs in input_sequence(40),
    ) {
        let mut service = CognitiveLoopService::new(config).unwrap();
        for input in &inputs {
            let result = service.cycle(input);
            let m = &result.metadata;

            // Core invariants on CycleResult
            prop_assert!(result.prediction_error.is_finite(), "prediction_error NaN/Inf");
            prop_assert!(result.prediction_error >= 0.0, "prediction_error negative");

            // Metadata invariants
            prop_assert!(m.temporal_coherence_score.is_finite(), "coherence NaN/Inf");
            prop_assert!(m.actual_effective_lr.is_finite(), "learning_rate NaN/Inf");
            prop_assert!(m.cycle_reward.is_finite(), "cycle_reward NaN/Inf");

            // Bounded checks
            prop_assert!(
                m.actual_effective_lr >= 0.0 && m.actual_effective_lr <= 10.0,
                "learning_rate diverged: {}", m.actual_effective_lr
            );
        }
    }

    /// Learning rate stays bounded even with extreme learning_threshold.
    #[test]
    fn lr_bounded_across_thresholds(
        lt in 0.001f32..0.5,
        inputs in input_sequence(30),
    ) {
        let config = CognitiveLoopConfig {
            learning_threshold: lt,
            async_training: false,
            enable_primitive_consciousness: true,
            ..Default::default()
        };
        let mut service = CognitiveLoopService::new(config).unwrap();

        for input in &inputs {
            let result = service.cycle(input);
            let lr = result.metadata.actual_effective_lr;
            prop_assert!(lr.is_finite(), "LR is NaN/Inf at lt={lt}");
            prop_assert!(lr >= 0.0, "LR went negative: {lr}");
            prop_assert!(lr < 5.0, "LR grew too large: {lr}");
        }
    }

    /// Reward signal stays bounded in [-1, 1] regardless of threshold.
    #[test]
    fn reward_bounded(
        lt in 0.001f32..0.5,
        inputs in input_sequence(30),
    ) {
        let config = CognitiveLoopConfig {
            learning_threshold: lt,
            async_training: false,
            enable_primitive_consciousness: true,
            ..Default::default()
        };
        let mut service = CognitiveLoopService::new(config).unwrap();

        for input in &inputs {
            let result = service.cycle(input);
            let reward = result.metadata.cycle_reward;
            prop_assert!(
                reward >= -1.0 && reward <= 1.0,
                "reward out of [-1,1]: {reward}"
            );
        }
    }
}
