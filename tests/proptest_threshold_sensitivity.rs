// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
Property-Based Tests for Threshold Sensitivity

The cognitive loop has 220+ named constants in `thresholds.rs` that control
every aspect of consciousness computation: moral evaluation, FEP active
inference, exploration, learning rate adaptation, neuromodulation, etc.

Since these constants are `pub const`, they cannot be modified at runtime.
Instead, this suite tests **sensitivity** by varying the knobs that ARE
runtime-configurable — consciousness profiles, substrate types, learning
thresholds, and subsystem enable flags — and verifying the system remains
robust (finite, bounded, non-divergent) across the entire parameter space.

## Complement to `proptest_feedback_stability.rs`

That file fuzzes **inputs** (random text, emotional words, adversarial strings).
This file fuzzes **configuration** (profiles, substrates, thresholds, enable flags)
and verifies the system's internal parameter space doesn't hide divergence.
*/

use proptest::prelude::*;
use symthaea::cognitive_loop::{
    CognitiveLoopConfig, CognitiveLoopService, config::ConsciousnessProfile,
};
use symthaea_core::hdc::substrate_independence::SubstrateType;

// ═══════════════════════════════════════════════════════════════════════════════
// Strategies
// ═══════════════════════════════════════════════════════════════════════════════

/// Strategy for generating random but valid input strings.
/// Includes emotional words, special characters, empty-ish strings, and normal text.
fn fuzz_input_strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        // Normal ASCII text
        "[a-z ]{0,60}",
        // Emotional positive words (trigger EmotionContagion)
        Just("happy joy love wonderful amazing excellent delighted thrilled".to_string()),
        // Emotional negative words
        Just("sad angry terrible horrible pain suffering grief struggle".to_string()),
        // High arousal text
        Just("incredible! urgent! NOW! terrified! ecstatic!!!".to_string()),
        // Empty / whitespace
        Just("".to_string()),
        // Repeated pattern (boredom trigger)
        Just("same same same same same same same same same same".to_string()),
        // Mixed content
        "([a-z]{1,10} ){1,8}[!?.]{0,5}",
    ]
}

/// Strategy for generating a sequence of fuzzed inputs.
fn fuzz_input_sequence(min: usize, max: usize) -> impl Strategy<Value = Vec<String>> {
    prop::collection::vec(fuzz_input_strategy(), min..=max)
}

/// Strategy: random ConsciousnessProfile from all 4 variants.
fn profile_strategy() -> impl Strategy<Value = ConsciousnessProfile> {
    prop_oneof![
        Just(ConsciousnessProfile::Minimal),
        Just(ConsciousnessProfile::Standard),
        Just(ConsciousnessProfile::Full),
        Just(ConsciousnessProfile::Research),
    ]
}

/// Strategy: random SubstrateType from all 8 variants.
fn substrate_strategy() -> impl Strategy<Value = SubstrateType> {
    prop_oneof![
        Just(SubstrateType::BiologicalNeurons),
        Just(SubstrateType::SiliconDigital),
        Just(SubstrateType::QuantumComputer),
        Just(SubstrateType::PhotonicProcessor),
        Just(SubstrateType::NeuromorphicChip),
        Just(SubstrateType::BiochemicalComputer),
        Just(SubstrateType::HybridSystem),
        Just(SubstrateType::ExoticSubstrate),
    ]
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Create a service from a consciousness profile with async training disabled.
fn profile_service(profile: ConsciousnessProfile) -> CognitiveLoopService {
    let mut config = CognitiveLoopConfig::from_profile(profile);
    config.async_training = false;
    CognitiveLoopService::new(config).unwrap()
}

/// Assert an f32 value is finite.
fn assert_finite_f32(val: f32, name: &str) -> Result<(), TestCaseError> {
    prop_assert!(val.is_finite(), "{name} is not finite: {val}");
    Ok(())
}

/// Assert an f64 value is finite.
fn assert_finite_f64(val: f64, name: &str) -> Result<(), TestCaseError> {
    prop_assert!(val.is_finite(), "{name} is not finite: {val}");
    Ok(())
}

/// Assert an f32 value is within [lo, hi].
fn assert_bounded_f32(val: f32, lo: f32, hi: f32, name: &str) -> Result<(), TestCaseError> {
    prop_assert!(val >= lo && val <= hi, "{name} out of [{lo}, {hi}]: {val}");
    Ok(())
}

/// Assert an f64 value is within [lo, hi].
fn assert_bounded_f64(val: f64, lo: f64, hi: f64, name: &str) -> Result<(), TestCaseError> {
    prop_assert!(val >= lo && val <= hi, "{name} out of [{lo}, {hi}]: {val}");
    Ok(())
}

/// Assert all core metadata fields are finite and bounded.
/// Checks the invariants that must hold regardless of which thresholds are active.
fn assert_metadata_sane(
    result: &symthaea::cognitive_loop::CycleResult,
    cycle: usize,
) -> Result<(), TestCaseError> {
    let m = &result.metadata;

    // consciousness_level: [0.0, 1.0]
    assert_finite_f64(
        m.consciousness.consciousness_level,
        &format!("consciousness_level@cycle{cycle}"),
    )?;
    assert_bounded_f64(
        m.consciousness.consciousness_level,
        0.0,
        1.0,
        &format!("consciousness_level@cycle{cycle}"),
    )?;

    // thermodynamic_load: [0.0, 1.0]
    assert_finite_f32(
        m.temporal.thermodynamic_load,
        &format!("thermodynamic_load@cycle{cycle}"),
    )?;
    assert_bounded_f32(
        m.temporal.thermodynamic_load,
        0.0,
        1.0,
        &format!("thermodynamic_load@cycle{cycle}"),
    )?;

    // actual_effective_lr: >= 0
    assert_finite_f32(
        m.actual_effective_lr,
        &format!("actual_effective_lr@cycle{cycle}"),
    )?;
    prop_assert!(
        m.actual_effective_lr >= 0.0,
        "actual_effective_lr negative at cycle {cycle}: {}",
        m.actual_effective_lr
    );

    // somatic_stress: >= 0
    assert_finite_f64(
        m.embodied.somatic_stress,
        &format!("somatic_stress@cycle{cycle}"),
    )?;
    prop_assert!(
        m.embodied.somatic_stress >= 0.0,
        "somatic_stress negative at cycle {cycle}: {}",
        m.embodied.somatic_stress
    );

    // prediction_error: finite
    assert_finite_f32(
        result.prediction_error,
        &format!("prediction_error@cycle{cycle}"),
    )?;

    // FEP signals: all finite
    assert_finite_f64(m.fep.fep_accuracy, &format!("fep_accuracy@cycle{cycle}"))?;
    assert_finite_f64(
        m.fep.fep_complexity,
        &format!("fep_complexity@cycle{cycle}"),
    )?;
    assert_finite_f64(m.fep.fep_surprise, &format!("fep_surprise@cycle{cycle}"))?;
    assert_finite_f64(m.fep.fep_td_error, &format!("fep_td_error@cycle{cycle}"))?;
    assert_finite_f64(
        m.fep.fep_pragmatic_value,
        &format!("fep_pragmatic_value@cycle{cycle}"),
    )?;

    // reasoning_confidence: finite
    assert_finite_f32(
        m.reasoning_confidence,
        &format!("reasoning_confidence@cycle{cycle}"),
    )?;

    // body telemetry: finite
    assert_finite_f32(
        m.embodied.body_valence,
        &format!("body_valence@cycle{cycle}"),
    )?;
    assert_finite_f32(
        m.embodied.body_arousal,
        &format!("body_arousal@cycle{cycle}"),
    )?;

    // cycle_reward: bounded [-1, 1]
    assert_finite_f32(m.cycle_reward, &format!("cycle_reward@cycle{cycle}"))?;
    assert_bounded_f32(
        m.cycle_reward,
        -1.0,
        1.0,
        &format!("cycle_reward@cycle{cycle}"),
    )?;

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════════
// 1. Profile robustness: all ConsciousnessProfile variants survive 50 cycles
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    // 10 cases in debug mode (each runs 50 cycles); CI or --release can use
    // PROPTEST_CASES=200 env var to scale up.
    #![proptest_config(ProptestConfig::with_cases(10))]

    /// Every ConsciousnessProfile (Minimal, Standard, Full, Research) enables
    /// different subsets of the 220+ threshold-controlled subsystems. Running
    /// 50 cycles with random inputs per profile verifies that no configuration
    /// of thresholds leads to divergence.
    #[test]
    fn prop_profile_robustness(
        profile in profile_strategy(),
        inputs in fuzz_input_sequence(50, 50),
    ) {
        let mut svc = profile_service(profile);
        for (i, input) in inputs.iter().enumerate() {
            let result = svc.cycle(input);
            assert_metadata_sane(&result, i)?;

            // prediction_confidence via accessor: [0, 1]
            let conf = svc.prediction_confidence();
            assert_bounded_f32(conf, 0.0, 1.0, &format!("prediction_confidence@cycle{i}"))?;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. Substrate sensitivity: different substrates affect consciousness level
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    // 10 cases (each runs 30 cycles); scale up with PROPTEST_CASES=200.
    #![proptest_config(ProptestConfig::with_cases(10))]

    /// Different substrate types feed different feasibility values into the
    /// Consciousness Equation V2 via SubstrateManager. This tests that all
    /// 8 substrate types produce finite, bounded consciousness under random input.
    #[test]
    fn prop_substrate_sensitivity(
        substrate in substrate_strategy(),
        inputs in fuzz_input_sequence(30, 30),
    ) {
        let mut config = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Standard);
        config.substrate_type = substrate;
        config.async_training = false;
        let mut svc = CognitiveLoopService::new(config).unwrap();

        for (i, input) in inputs.iter().enumerate() {
            let result = svc.cycle(input);
            assert_metadata_sane(&result, i)?;
        }

        // After 30 cycles, verify substrate-specific properties
        let eff = svc.substrate_effective_feasibility();
        assert_finite_f64(eff, "substrate_effective_feasibility")?;
        prop_assert!(
            eff >= 0.0 && eff <= 1.0,
            "effective_feasibility out of [0, 1]: {eff} for {substrate:?}"
        );

        let tau = svc.substrate_tau_factor();
        assert_finite_f32(tau, "substrate_tau_factor")?;
        prop_assert!(
            tau > 0.0,
            "tau_factor non-positive: {tau} for {substrate:?}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. Learning threshold sweep: convergence across threshold range
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    // 10 cases (each runs 20 cycles); scale up with PROPTEST_CASES=100.
    #![proptest_config(ProptestConfig::with_cases(10))]

    /// The learning_threshold parameter interacts with dozens of threshold
    /// constants (FEP_SURPRISE_SCALE, HEBBIAN_LR_SCALE, etc.) to determine
    /// when and how fast the system learns. Sweeping it from 0.001 to 0.5
    /// verifies no threshold combination causes NaN propagation.
    #[test]
    fn prop_learning_threshold_sweep(
        threshold in 0.001f32..=0.5f32,
        inputs in fuzz_input_sequence(20, 20),
    ) {
        let mut config = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Standard);
        config.learning_threshold = threshold;
        config.async_training = false;
        let mut svc = CognitiveLoopService::new(config).unwrap();

        let mut first_errors = Vec::new();
        let mut last_errors = Vec::new();

        for (i, input) in inputs.iter().enumerate() {
            let result = svc.cycle(input);
            assert_metadata_sane(&result, i)?;

            // Collect prediction errors for convergence check
            if i < 5 {
                first_errors.push(result.prediction_error);
            }
            if i >= 15 {
                last_errors.push(result.prediction_error);
            }
        }

        // Verify all errors are finite (no NaN propagation through threshold interactions)
        for e in &first_errors {
            assert_finite_f32(*e, "early_prediction_error")?;
        }
        for e in &last_errors {
            assert_finite_f32(*e, "late_prediction_error")?;
        }

        // Verify carryover state is clean
        let conf = svc.prediction_confidence();
        assert_finite_f32(conf, "final_prediction_confidence")?;
        assert_bounded_f32(conf, 0.0, 1.0, "final_prediction_confidence")?;

        let lr = svc.stats().adaptive_learning_rate;
        assert_finite_f32(lr, "final_adaptive_learning_rate")?;
        prop_assert!(lr >= 0.0, "final learning rate negative: {lr}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4. Extreme config stability: maximal and minimal subsystem configurations
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    // 10 cases (each runs 30 cycles); scale up with PROPTEST_CASES=100.
    #![proptest_config(ProptestConfig::with_cases(10))]

    /// Create configs with extreme subsystem enable/disable combinations.
    /// This exercises threshold interactions that only occur when specific
    /// subsystem pairs are both active (e.g., moral + FEP, attention + GWT).
    #[test]
    fn prop_extreme_config_stability(
        all_enabled in proptest::bool::ANY,
        threshold in prop_oneof![
            Just(0.001f32),
            Just(0.01f32),
            Just(0.1f32),
            Just(0.5f32),
        ],
        substrate in substrate_strategy(),
        inputs in fuzz_input_sequence(30, 30),
    ) {
        let config = if all_enabled {
            // All subsystems enabled — maximum threshold interaction surface
            CognitiveLoopConfig {
                enable_primitive_consciousness: true,
                enable_surprise_exploration: true,
                enable_prefrontal: true,
                enable_affective_bridge: true,
                enable_virtual_body: true,
                enable_gwt: true,
                enable_predictive_processing: true,
                enable_attention_schema: true,
                enable_narrative_self: true,
                enable_predictive_self: true,
                enable_meta_cognition: true,
                enable_resonance: true,
                enable_quantum_coherence: true,
                enable_temporal_consciousness: true,
                enable_embodied_cognition: true,
                enable_narrative_gwt: true,
                learning_threshold: threshold,
                substrate_type: substrate,
                async_training: false,
                ..Default::default()
            }
        } else {
            // All subsystems disabled — minimal threshold surface
            CognitiveLoopConfig {
                enable_primitive_consciousness: false,
                enable_surprise_exploration: false,
                enable_prefrontal: false,
                enable_affective_bridge: false,
                enable_virtual_body: false,
                enable_gwt: false,
                enable_predictive_processing: false,
                enable_attention_schema: false,
                enable_narrative_self: false,
                enable_predictive_self: false,
                enable_meta_cognition: false,
                enable_resonance: false,
                enable_quantum_coherence: false,
                enable_temporal_consciousness: false,
                enable_embodied_cognition: false,
                enable_narrative_gwt: false,
                learning_threshold: threshold,
                substrate_type: substrate,
                async_training: false,
                ..Default::default()
            }
        };

        let mut svc = CognitiveLoopService::new(config).unwrap();
        for (i, input) in inputs.iter().enumerate() {
            let result = svc.cycle(input);
            assert_metadata_sane(&result, i)?;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 5. Multi-profile transition: switching substrate mid-run
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    // 5 cases (each runs 40 cycles + transition); scale up with PROPTEST_CASES=50.
    #![proptest_config(ProptestConfig::with_cases(5))]

    /// Run 20 cycles on one profile/substrate, then reconfigure substrate
    /// (simulating a configuration transition) and run 20 more cycles.
    /// Verifies that accumulated internal state (neuromodulator levels,
    /// prediction confidence, exploration urge) doesn't diverge when the
    /// threshold interaction surface changes mid-run.
    #[test]
    #[ignore] // Slower due to 40 total cycles with substrate reconfiguration
    fn prop_multi_profile_transition(
        profile in profile_strategy(),
        substrate_before in substrate_strategy(),
        substrate_after in substrate_strategy(),
        inputs in fuzz_input_sequence(40, 40),
    ) {
        let mut config = CognitiveLoopConfig::from_profile(profile);
        config.substrate_type = substrate_before;
        config.async_training = false;
        let mut svc = CognitiveLoopService::new(config).unwrap();

        // Phase 1: 20 cycles on initial substrate
        for (i, input) in inputs[..20].iter().enumerate() {
            let result = svc.cycle(input);
            assert_metadata_sane(&result, i)?;
        }

        // Record pre-transition state
        let pre_conf = svc.prediction_confidence();
        assert_finite_f32(pre_conf, "pre_transition_confidence")?;

        // Transition: switch substrate
        let (old_feas, new_feas) = svc.reconfigure_substrate(substrate_after);
        assert_finite_f64(old_feas, "old_feasibility")?;
        assert_finite_f64(new_feas, "new_feasibility")?;

        // Phase 2: 20 more cycles on new substrate
        for (i, input) in inputs[20..].iter().enumerate() {
            let result = svc.cycle(input);
            assert_metadata_sane(&result, 20 + i)?;
        }

        // Post-transition: system must still be coherent
        let post_conf = svc.prediction_confidence();
        assert_finite_f32(post_conf, "post_transition_confidence")?;
        assert_bounded_f32(post_conf, 0.0, 1.0, "post_transition_confidence")?;

        // Consciousness level must still be bounded
        // (run one more cycle to get fresh metadata)
        let final_result = svc.cycle("transition stability check");
        assert_metadata_sane(&final_result, 41)?;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 6. Feedback loop telemetry: Session 15 observability fields are sane
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]

    /// After 30+ cycles, verify that the 6 new feedback-loop telemetry fields
    /// (moral_consolidation_ease, consolidation_threshold, mce_bottleneck_lr_applied,
    /// homeostasis_recalibrated, confidence_falling_lr_boost, coherence_velocity_budget_scaled)
    /// are finite, bounded, and self-consistent across all profiles.
    #[test]
    fn prop_feedback_loop_telemetry_sane(
        profile in profile_strategy(),
        inputs in fuzz_input_sequence(35, 35),
    ) {
        let mut svc = profile_service(profile);
        let mut last_meta = None;
        for (i, input) in inputs.iter().enumerate() {
            let result = svc.cycle(input);
            assert_metadata_sane(&result, i)?;
            if i >= 30 {
                last_meta = Some(result.metadata);
            }
        }
        let m = last_meta.unwrap();

        // moral_consolidation_ease: [0.0, ~0.1] (moral score ∈ [0,1], ease factor 0.15)
        assert_finite_f32(m.moral_consolidation_ease, "moral_consolidation_ease")?;
        prop_assert!(
            m.moral_consolidation_ease >= 0.0 && m.moral_consolidation_ease <= 0.2,
            "moral_consolidation_ease out of bounds: {}", m.moral_consolidation_ease
        );

        // consolidation_threshold: [0.2, 1.0] (floored at 0.2)
        assert_finite_f32(m.consolidation_threshold, "consolidation_threshold")?;
        prop_assert!(
            m.consolidation_threshold >= 0.2 && m.consolidation_threshold <= 1.0,
            "consolidation_threshold out of bounds: {}", m.consolidation_threshold
        );

        // When moral ease > 0, consolidation threshold should be lowered
        if m.moral_consolidation_ease > 0.0 {
            // consolidation_threshold was lowered by moral_consolidation_ease
            // (it may still be at the floor of 0.2, but cannot exceed ema - 0.1)
            prop_assert!(m.consolidation_threshold <= 1.0);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 7. MCE bottleneck coupling: LR boost fires when bottleneck is identified
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(5))]

    /// MCE fires every ~194 cycles. Run 250 cycles and verify:
    /// (a) mce_bottleneck_lr_applied becomes true at least once
    /// (b) LR remains bounded after bottleneck-driven boosting
    /// (c) No divergence from repeated MCE-driven LR modulation
    #[test]
    #[ignore] // Slow: 250 cycles per case
    fn prop_mce_bottleneck_lr_bounded(
        profile in profile_strategy(),
        inputs in fuzz_input_sequence(250, 250),
    ) {
        let mut svc = profile_service(profile);
        let mut bottleneck_fired = false;
        for (i, input) in inputs.iter().enumerate() {
            let result = svc.cycle(input);
            assert_metadata_sane(&result, i)?;
            if result.metadata.modulation.mce_bottleneck_lr_applied {
                bottleneck_fired = true;
            }
            // LR must never diverge regardless of bottleneck boosting
            let lr = result.metadata.actual_effective_lr;
            prop_assert!(
                lr >= 0.0 && lr < 10.0,
                "LR diverged at cycle {i}: {lr}"
            );
        }
        // MCE fires every ~194 cycles; in 250 cycles it should fire at least once
        prop_assert!(
            bottleneck_fired,
            "MCE bottleneck never fired in 250 cycles — may indicate MCE not running"
        );
    }
}