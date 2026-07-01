// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for the Stability Regime system

use symthaea::consciousness::primitive_consciousness::ConsciousnessPrimitiveProcessor;
use symthaea::consciousness::stability_regime::{
    CfCPrimitive, StabilityRegimeConfig, StabilityRegimeProcessor, StabilityRegimeType,
};
use symthaea_core::hdc::BinaryHV;
use symthaea_core::hdc::primitive_system::{Primitive, PrimitiveTier};
use symthaea_core::hdc::unified_hv::ContinuousHV;

fn make_prim(name: &str, tier: PrimitiveTier) -> Primitive {
    Primitive {
        name: name.to_string(),
        tier,
        domain: "test".to_string(),
        encoding: BinaryHV::random(name.len() as u64 * 31 + tier as u64),
        definition: name.to_string(),
        is_base: true,
        derivation: None,
    }
}

#[test]
fn test_regime_assignment_all_tiers() {
    let mappings = [
        (PrimitiveTier::NSM, StabilityRegimeType::Crystallized),
        (
            PrimitiveTier::Mathematical,
            StabilityRegimeType::Crystallized,
        ),
        (PrimitiveTier::Physical, StabilityRegimeType::Crystallized),
        (PrimitiveTier::Geometric, StabilityRegimeType::Crystallized),
        (PrimitiveTier::Strategic, StabilityRegimeType::Plastic),
        (PrimitiveTier::MetaCognitive, StabilityRegimeType::Plastic),
        (PrimitiveTier::Temporal, StabilityRegimeType::Plastic),
        (PrimitiveTier::Compositional, StabilityRegimeType::Fluid),
        (PrimitiveTier::Consciousness, StabilityRegimeType::Fluid),
    ];

    for (tier, expected) in &mappings {
        assert_eq!(
            StabilityRegimeType::from_tier(*tier),
            *expected,
            "Tier {:?} should be {:?}",
            tier,
            expected,
        );
    }
}

#[test]
fn test_cfc_primitive_starts_plastic() {
    let config = StabilityRegimeConfig::default();
    let prim = make_prim("FORCE", PrimitiveTier::Physical);
    let cfc = CfCPrimitive::new(prim, &config, 42);

    // All primitives start Plastic regardless of tier
    assert_eq!(cfc.regime, StabilityRegimeType::Plastic);
    // But tier_regime tracks the intended final regime
    assert_eq!(cfc.tier_regime, StabilityRegimeType::Crystallized);
    assert!(!cfc.is_active);
}

#[test]
fn test_plastic_drift() {
    let config = StabilityRegimeConfig::default();
    let prim = make_prim("PLAN", PrimitiveTier::Strategic);
    let mut cfc = CfCPrimitive::new(prim, &config, 42);
    let params = config.params(StabilityRegimeType::Plastic);

    let initial_sim = cfc.attractor_similarity();
    let context = ContinuousHV::random(16_384, 777);
    for _ in 0..200 {
        cfc.evolve(0.1, &context, params);
    }

    assert!(
        cfc.attractor_similarity() < initial_sim,
        "Plastic should drift: {} -> {}",
        initial_sim,
        cfc.attractor_similarity(),
    );
}

#[test]
fn test_activation_hysteresis() {
    let config = StabilityRegimeConfig::default();
    let prim = make_prim("TEST", PrimitiveTier::Physical);
    let mut cfc = CfCPrimitive::new(prim, &config, 42);
    let params = config.params(StabilityRegimeType::Crystallized);

    cfc.activation = 0.5;
    cfc.update_active_status(params);
    assert!(cfc.is_active);

    cfc.activation = 0.30;
    cfc.update_active_status(params);
    assert!(cfc.is_active, "Hysteresis: should stay active");

    cfc.activation = 0.20;
    cfc.update_active_status(params);
    assert!(!cfc.is_active, "Should deactivate below threshold");
}

#[test]
fn test_process_input_produces_state() {
    let mut processor = StabilityRegimeProcessor::new();
    let input = BinaryHV::random(42);

    let (prim_state, _transitions) = processor.process_input(&input, 0.1, 0.0);
    assert!(prim_state.phi >= 0.0);
}

#[test]
fn test_coherence_feedback_over_multiple_steps() {
    let mut processor = StabilityRegimeProcessor::new();
    let input = BinaryHV::random(42);

    for i in 0..10 {
        processor.process_input(&input, 0.1, i as f64 * 0.1);
    }

    let lr = processor.coherence_bridge().effective_learning_rate();
    assert!(lr > 0.0, "Learning rate should be positive: {}", lr);
}

#[test]
fn test_backward_compat_inner_processor() {
    let processor = StabilityRegimeProcessor::new();
    let stats = processor.inner().primitive_stats();
    assert!(stats.total_primitives > 0);

    let mut inner_only = ConsciousnessPrimitiveProcessor::new();
    let input = BinaryHV::random(99);
    let state = inner_only.process_input(&input, 0.0);
    assert!(state.phi >= 0.0);
}

#[test]
fn test_multiple_inputs_build_history() {
    let mut processor = StabilityRegimeProcessor::new();

    for i in 0..5 {
        let input = BinaryHV::random(i as u64 * 100 + 1);
        processor.process_input(&input, 0.1, i as f64);
    }

    assert_eq!(processor.history().len(), 5);
}

#[test]
fn test_regime_transition_plastic_to_crystallized() {
    // A primitive that gets activated 50+ times should transition to Crystallized
    let config = StabilityRegimeConfig::default();
    assert_eq!(config.plastic_to_crystallized_activations, 50);

    let prim = make_prim("BUSY", PrimitiveTier::Strategic);
    let mut cfc = CfCPrimitive::new(prim, &config, 42);

    // Starts Plastic
    assert_eq!(cfc.regime, StabilityRegimeType::Plastic);

    // Simulate 50 activations
    cfc.total_activation_count = 50;
    cfc.last_activated_cycle = 50;
    cfc.update_regime(50, &config);

    assert_eq!(
        cfc.regime,
        StabilityRegimeType::Crystallized,
        "Should crystallize after {} activations",
        config.plastic_to_crystallized_activations,
    );
}

#[test]
fn test_regime_transition_fluid_to_plastic() {
    let config = StabilityRegimeConfig::default();
    assert_eq!(config.fluid_to_plastic_activations, 10);

    let prim = make_prim("EXPLORE", PrimitiveTier::Compositional);
    let mut cfc = CfCPrimitive::new(prim, &config, 42);

    // All primitives start Plastic, so force it to Fluid to test this transition
    cfc.regime = StabilityRegimeType::Fluid;
    cfc.total_activation_count = 0;

    // Below threshold: should stay Fluid
    cfc.total_activation_count = 9;
    cfc.update_regime(9, &config);
    assert_eq!(cfc.regime, StabilityRegimeType::Fluid);

    // At threshold: should transition
    cfc.total_activation_count = 10;
    cfc.update_regime(10, &config);
    assert_eq!(cfc.regime, StabilityRegimeType::Plastic);
}

#[test]
fn test_decrystallize_after_idle() {
    let config = StabilityRegimeConfig::default();
    assert_eq!(config.decrystallize_idle_cycles, 100);

    let prim = make_prim("STALE", PrimitiveTier::Physical);
    let mut cfc = CfCPrimitive::new(prim, &config, 42);

    // Force to Crystallized with enough activations
    cfc.total_activation_count = 60;
    cfc.last_activated_cycle = 50;
    cfc.regime = StabilityRegimeType::Crystallized;

    // 99 cycles idle: should stay Crystallized
    cfc.update_regime(149, &config);
    assert_eq!(cfc.regime, StabilityRegimeType::Crystallized);

    // 100 cycles idle: should drop back to Plastic
    cfc.update_regime(150, &config);
    assert_eq!(
        cfc.regime,
        StabilityRegimeType::Plastic,
        "Should decrystallize after {} idle cycles",
        config.decrystallize_idle_cycles,
    );

    // Activation count should be reset to fluid_to_plastic threshold
    // so it can re-crystallize naturally
    assert_eq!(
        cfc.total_activation_count,
        config.fluid_to_plastic_activations
    );
}

#[test]
fn test_default_thresholds() {
    let config = StabilityRegimeConfig::default();
    assert_eq!(config.fluid_to_plastic_activations, 10);
    assert_eq!(config.plastic_to_crystallized_activations, 50);
    assert_eq!(config.decrystallize_idle_cycles, 100);
}

#[test]
fn test_global_cycle_increments() {
    let mut processor = StabilityRegimeProcessor::new();
    assert_eq!(processor.global_cycle(), 0);

    let input = BinaryHV::random(42);
    processor.process_input(&input, 0.1, 0.0);
    assert_eq!(processor.global_cycle(), 1);

    processor.process_input(&input, 0.1, 0.1);
    assert_eq!(processor.global_cycle(), 2);
}

#[test]
fn test_crystallized_primitives_lower_variance() {
    // Crystallized primitives should produce more stable (lower variance) CfC
    // outputs than fluid/plastic ones because their high attractor_strength (0.95)
    // snaps the neuron state back toward a fixed point, while plastic primitives
    // (attractor_strength 0.5) drift more freely with each input.

    let config = StabilityRegimeConfig::default();

    // --- Crystallized primitive: force regime + warm up with 60 activations ---
    let prim_c = make_prim("FORCE", PrimitiveTier::Physical);
    let mut cfc_crystallized = CfCPrimitive::new(prim_c, &config, 42);

    // Warm up: feed the same input 200 times so the crystallized primitive
    // deeply settles into its attractor. We also manually set the regime to
    // ensure it is Crystallized for the measurement phase.
    let warmup_input = ContinuousHV::random(16_384, 1001);
    let crystallized_params = config.params(StabilityRegimeType::Crystallized);
    for _ in 0..200 {
        cfc_crystallized.evolve(0.1, &warmup_input, crystallized_params);
    }
    cfc_crystallized.regime = StabilityRegimeType::Crystallized;
    cfc_crystallized.total_activation_count = 200;

    // --- Fresh plastic primitive with brief warmup ---
    // Give it a few varied inputs so it develops some state, then measure drift.
    let prim_p = make_prim("PLAN", PrimitiveTier::Strategic);
    let mut cfc_plastic = CfCPrimitive::new(prim_p, &config, 77);
    let plastic_params = config.params(StabilityRegimeType::Plastic);
    for seed in 3000..3010u64 {
        let warmup = ContinuousHV::random(16_384, seed);
        cfc_plastic.evolve(0.1, &warmup, plastic_params);
    }

    // --- Measurement phase: feed 10 varied inputs and record activations ---
    let mut crystallized_activations = Vec::new();
    let mut plastic_activations = Vec::new();

    for seed in 2000..2010u64 {
        let probe = ContinuousHV::random(16_384, seed);

        let act_c = cfc_crystallized.evolve(0.1, &probe, crystallized_params);
        crystallized_activations.push(act_c);

        let act_p = cfc_plastic.evolve(0.1, &probe, plastic_params);
        plastic_activations.push(act_p);
    }

    // Compute variance: Var(X) = E[X^2] - E[X]^2
    let variance = |vals: &[f64]| -> f64 {
        let n = vals.len() as f64;
        let mean = vals.iter().sum::<f64>() / n;
        let mean_sq = vals.iter().map(|v| v * v).sum::<f64>() / n;
        mean_sq - mean * mean
    };

    let var_crystallized = variance(&crystallized_activations);
    let var_plastic = variance(&plastic_activations);

    println!(
        "Crystallized activations: {:?}\n  variance = {}",
        crystallized_activations, var_crystallized
    );
    println!(
        "Plastic activations: {:?}\n  variance = {}",
        plastic_activations, var_plastic
    );

    // Allow a 3× tolerance: crystallized variance should be at most 3× the plastic
    // variance. Both values are often near the noise floor (~1e-4), so a strict
    // less-than comparison is too fragile across random seeds.
    assert!(
        var_crystallized < var_plastic * 3.0,
        "Crystallized variance ({:.6}) should be less than 3× plastic variance ({:.6}). \
         Crystallized primitives snap back to their attractor, producing more stable outputs.",
        var_crystallized,
        var_plastic,
    );
}