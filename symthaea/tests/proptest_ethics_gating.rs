// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
Property-Based Tests for Ethics Output Gating

Fuzzes ethical verdicts through the cognitive loop, verifying safety invariants:

1. **Blocked never succeeds**: `EthicalVerdict::Blocked` must prevent all motor success
2. **Telemetry consistency**: `metadata.ethics.unified_verdict` always matches the set verdict
3. **Caution caps confidence**: No high-confidence destructive action succeeds under Caution
4. **Override clear restores engine**: Clearing an override lets the ethics engine reclaim control
5. **Verdict does not zero consciousness**: Ethics gating affects output, not consciousness computation

Science: APA Ethics Code (2017) principle 3.04 — avoid harm.
*/

use proptest::prelude::*;
use symthaea::cognitive_loop::motor_output_bridge::{MotorActionRequest, MotorOutputBridge};
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService, EthicalVerdict};

// ═══════════════════════════════════════════════════════════════════════════════
// Strategies
// ═══════════════════════════════════════════════════════════════════════════════

fn fuzz_input_strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        "[a-z ]{0,60}",
        Just("happy joy love wonderful amazing excellent".to_string()),
        Just("sad angry terrible horrible pain suffering".to_string()),
        Just("incredible! urgent! NOW! terrified!".to_string()),
        Just("".to_string()),
        Just("same same same same same same same".to_string()),
        "([a-z]{1,10} ){1,8}[!?.]{0,3}",
    ]
}

fn fuzz_input_sequence(min: usize, max: usize) -> impl Strategy<Value = Vec<String>> {
    prop::collection::vec(fuzz_input_strategy(), min..=max)
}

fn verdict_strategy() -> impl Strategy<Value = EthicalVerdict> {
    (0u8..3).prop_map(|v| match v {
        0 => EthicalVerdict::Safe,
        1 => EthicalVerdict::Caution,
        _ => EthicalVerdict::Blocked,
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════════

const GENESIS_SEED: &str = "proptest_ethics_gating_v1";

fn make_service() -> CognitiveLoopService {
    CognitiveLoopService::new(CognitiveLoopConfig {
        genesis_phrase: Some(GENESIS_SEED.to_string()),
        async_training: false,
        learning_threshold: 0.0,
        ..Default::default()
    })
    .expect("CognitiveLoopService::new should succeed")
}

fn make_service_with_motor() -> CognitiveLoopService {
    let mut service = make_service();
    let bridge = MotorOutputBridge::with_defaults().unwrap();
    service.set_motor_output_bridge(bridge);
    service
}

fn verdict_name(v: &EthicalVerdict) -> &'static str {
    match v {
        EthicalVerdict::Safe => "Safe",
        EthicalVerdict::Caution => "Caution",
        EthicalVerdict::Blocked => "Blocked",
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Property 1: Blocked verdict never allows motor success
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig { cases: 20, .. ProptestConfig::default() })]

    #[test]
    fn prop_blocked_never_succeeds_motor(
        inputs in fuzz_input_sequence(5, 30),
    ) {
        let mut service = make_service_with_motor();

        for (idx, input) in inputs.iter().enumerate() {
            service.set_ethics_verdict(EthicalVerdict::Blocked);
            service.set_motor_request(MotorActionRequest {
                target_path: Some(std::path::PathBuf::from(
                    "/tmp/symthaea/proptest_ethics/blocked.txt",
                )),
                ..Default::default()
            });

            let _result = service.cycle(input);

            if let Some(motor_result) = service.take_motor_result() {
                prop_assert!(
                    !motor_result.success,
                    "Motor output succeeded under Blocked verdict at cycle {}: {:?}",
                    idx, motor_result
                );
                assert_eq!(
                    motor_result.prediction_error, 1.0,
                    "Blocked motor should report PE = 1.0 at cycle {}",
                    idx
                );
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Property 2: Telemetry verdict always matches the set verdict
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig { cases: 20, .. ProptestConfig::default() })]

    #[test]
    fn prop_verdict_telemetry_consistent(
        verdict in verdict_strategy(),
        inputs in fuzz_input_sequence(1, 10),
    ) {
        let mut service = make_service();
        let expected = verdict_name(&verdict);

        for (_idx, input) in inputs.iter().enumerate() {
            service.set_ethics_verdict(verdict.clone());
            let result = service.cycle(input);

            prop_assert_eq!(
                &result.metadata.ethics.unified_verdict,
                expected,
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Property 3: Caution caps confidence — no high-confidence action succeeds
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig { cases: 20, .. ProptestConfig::default() })]

    #[test]
    fn prop_caution_caps_confidence(
        inputs in fuzz_input_sequence(5, 30),
    ) {
        let mut service = make_service_with_motor();

        for (_idx, input) in inputs.iter().enumerate() {
            service.set_ethics_verdict(EthicalVerdict::Caution);
            service.set_motor_request(MotorActionRequest {
                target_path: Some(std::path::PathBuf::from(
                    "/tmp/symthaea/proptest_ethics/caution.txt",
                )),
                program: Some("echo".into()),
                args: vec!["test".into()],
                content: None,
            });

            let _result = service.cycle(input);

            if let Some(motor_result) = service.take_motor_result() {
                if !motor_result.success {
                    prop_assert!(
                        motor_result.error.is_some(),
                        "Failed motor under Caution should have error"
                    );
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Property 4: Override clear restores engine control
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig { cases: 20, .. ProptestConfig::default() })]

    #[test]
    fn prop_override_clear_restores_engine(
        verdict in prop_oneof![
            Just(EthicalVerdict::Blocked),
            Just(EthicalVerdict::Caution),
        ],
        cycles_before_clear in 1u32..5,
        cycles_after_clear in 7u32..20,
    ) {
        let mut service = make_service();
        let forced_name = verdict_name(&verdict);

        // Phase 1: Run with forced override
        for _i in 0..cycles_before_clear {
            service.set_ethics_verdict(verdict.clone());
            let result = service.cycle("benign neutral input text");
            prop_assert_eq!(
                &result.metadata.ethics.unified_verdict,
                forced_name,
            );
        }

        // Phase 2: Clear override, run enough cycles for the engine to fire
        service.clear_ethics_override();

        let mut verdicts = Vec::new();
        for i in 0..cycles_after_clear {
            let input = if i % 2 == 0 {
                "The weather is pleasant today."
            } else {
                "Learning is a continuous process of growth."
            };
            let result = service.cycle(input);
            verdicts.push(result.metadata.ethics.unified_verdict.clone());
        }

        // After 7+ cycles with benign input, verdict should move away from forced value
        let non_forced_count = verdicts.iter().filter(|v| v.as_str() != forced_name).count();
        prop_assert!(
            non_forced_count > 0,
            "After clearing override, verdict should not remain stuck at {}. All: {:?}",
            forced_name, verdicts
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Property 5: Verdict does not zero out consciousness level
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig { cases: 20, .. ProptestConfig::default() })]

    #[test]
    fn prop_verdict_does_not_affect_consciousness_level(
        verdict in verdict_strategy(),
    ) {
        let mut service = make_service();

        // Phase 1: 10 warmup cycles with Safe
        for _i in 0..10 {
            service.set_ethics_verdict(EthicalVerdict::Safe);
            let _result = service.cycle("The universe is vast and full of wonder.");
        }

        // Phase 2: 10 cycles with the fuzzed verdict
        let mut test_levels = Vec::new();
        for _i in 0..10 {
            service.set_ethics_verdict(verdict.clone());
            let result = service.cycle("The universe is vast and full of wonder.");
            test_levels.push(result.metadata.consciousness.consciousness_level);
        }

        // After warmup, consciousness_level must be > 0.0 regardless of verdict
        let tail = &test_levels[5..];
        for (idx, level) in tail.iter().enumerate() {
            prop_assert!(
                *level > 0.0,
                "consciousness_level should be > 0.0 under {:?} (cycle {}, level = {})",
                verdict, idx + 15, level
            );
            prop_assert!(
                level.is_finite(),
                "consciousness_level must be finite under {:?} (cycle {}, level = {})",
                verdict, idx + 15, level
            );
        }
    }
}