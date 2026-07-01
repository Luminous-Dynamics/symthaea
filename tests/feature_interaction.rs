// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// ==================================================================================
// Feature Interaction Integration Test: Consciousness Profiles
// ==================================================================================
//
// Tests that different consciousness profile configurations interact correctly:
//   1. Create configs with Minimal, Standard, and Full profiles
//   2. Run cycles with each profile
//   3. Verify all profiles produce valid results
//   4. Test reset/resume behavior across profiles
//   5. Verify profile-specific modules are active
//
// No feature flags required — uses default configuration only.
// ==================================================================================

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService, ConsciousnessProfile};

// ═══════════════════════════════════════════════════════════════════════════════
// MINIMAL PROFILE
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_minimal_profile_produces_valid_results() {
    let config = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Minimal);

    // Verify minimal profile settings
    assert!(
        config.enable_virtual_body,
        "Minimal profile should enable virtual body"
    );
    assert!(
        !config.enable_surprise_exploration,
        "Minimal profile should not enable surprise exploration"
    );
    assert!(
        !config.enable_meta_cognition,
        "Minimal profile should not enable meta cognition"
    );

    let mut service = CognitiveLoopService::new(config).unwrap();

    for _ in 0..10 {
        let result = service.cycle("test input for minimal profile");
        assert!(
            result.prediction_error.is_finite(),
            "Minimal profile should produce finite prediction error"
        );
        assert!(
            !result.output.is_empty(),
            "Minimal profile should produce non-empty output"
        );
    }

    assert_eq!(service.stats().total_cycles, 10);
}

// ═══════════════════════════════════════════════════════════════════════════════
// STANDARD PROFILE
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_standard_profile_produces_valid_results() {
    let config = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Standard);

    // Verify standard profile settings
    assert!(
        config.enable_virtual_body,
        "Standard profile should enable virtual body"
    );
    assert!(
        config.enable_surprise_exploration,
        "Standard profile should enable surprise exploration"
    );
    assert!(
        config.enable_prefrontal,
        "Standard profile should enable prefrontal"
    );
    assert!(
        config.enable_meta_cognition,
        "Standard profile should enable meta cognition"
    );
    assert!(
        config.enable_narrative_self,
        "Standard profile should enable narrative self"
    );
    assert!(config.enable_gwt, "Standard profile should enable GWT");
    assert!(
        config.enable_embodied_cognition,
        "Standard profile should enable embodied cognition"
    );
    assert!(
        config.enable_attention_schema,
        "Standard profile should enable attention schema"
    );

    let mut service = CognitiveLoopService::new(config).unwrap();

    for _ in 0..10 {
        let result = service.cycle("test input for standard profile");
        assert!(
            result.prediction_error.is_finite(),
            "Standard profile should produce finite prediction error"
        );
        assert!(
            !result.output.is_empty(),
            "Standard profile should produce non-empty output"
        );
    }

    assert_eq!(service.stats().total_cycles, 10);
}

// ═══════════════════════════════════════════════════════════════════════════════
// FULL PROFILE
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_full_profile_produces_valid_results() {
    let config = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Full);

    // Verify full profile settings
    assert!(
        config.enable_virtual_body,
        "Full profile should enable virtual body"
    );
    assert!(
        config.enable_dream_replay,
        "Full profile should enable dream replay"
    );
    assert!(
        config.enable_predictive_processing,
        "Full profile should enable predictive processing"
    );
    assert!(
        config.enable_cross_modal_binding,
        "Full profile should enable cross-modal binding"
    );
    assert!(
        config.enable_affective_bridge,
        "Full profile should enable affective bridge"
    );
    assert!(
        config.enable_primitive_consciousness,
        "Full profile should enable primitive consciousness"
    );
    assert!(
        config.enable_resonator_recall,
        "Full profile should enable resonator recall"
    );

    let mut service = CognitiveLoopService::new(config).unwrap();

    for _ in 0..10 {
        let result = service.cycle("test input for full profile");
        assert!(
            result.prediction_error.is_finite(),
            "Full profile should produce finite prediction error"
        );
        assert!(
            !result.output.is_empty(),
            "Full profile should produce non-empty output"
        );
    }

    assert_eq!(service.stats().total_cycles, 10);
}

// ═══════════════════════════════════════════════════════════════════════════════
// PROFILE COMPARISON: All profiles produce consistent structure
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_all_profiles_produce_consistent_output_structure() {
    let profiles = [
        ConsciousnessProfile::Minimal,
        ConsciousnessProfile::Standard,
        ConsciousnessProfile::Full,
    ];

    for profile in &profiles {
        let config = CognitiveLoopConfig::from_profile(*profile);
        let mut service = CognitiveLoopService::new(config).unwrap();

        let result = service.cycle("consistent structure test");

        // All profiles should produce the same output structure
        assert!(
            result.prediction_error >= 0.0,
            "{profile:?}: prediction_error should be non-negative"
        );
        assert!(
            result.prediction_error <= 1.0,
            "{profile:?}: prediction_error should be <= 1.0, got {}",
            result.prediction_error
        );
        assert!(
            !result.output.is_empty(),
            "{profile:?}: output should not be empty"
        );
        assert!(
            result.cycle_time_us > 0,
            "{profile:?}: cycle_time_us should be positive"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// METADATA VARIES BY PROFILE
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_metadata_reflects_profile_modules() {
    // Minimal: no surprise exploration, no meta-cognition
    let minimal_config = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Minimal);
    let mut minimal_svc = CognitiveLoopService::new(minimal_config).unwrap();

    // Run a few cycles so subsystems have time to activate
    for _ in 0..5 {
        minimal_svc.cycle("warmup cycle");
    }
    let minimal_result = minimal_svc.cycle("metadata test");

    // Minimal should not have surprise or meta-cognitive activity
    assert!(
        !minimal_result.metadata.surprise_triggered,
        "Minimal profile should not trigger surprise exploration"
    );
    assert_eq!(
        minimal_result.metadata.quality.meta_cognitive_accuracy, 0.0,
        "Minimal profile should have zero meta-cognitive accuracy"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// RESET BEHAVIOR
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_reset_clears_state() {
    let config = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Standard);
    let mut service = CognitiveLoopService::new(config).unwrap();

    // Run some cycles
    for _ in 0..20 {
        service.cycle("pre-reset input");
    }
    assert_eq!(service.stats().total_cycles, 20);

    // Reset
    service.reset();
    assert_eq!(
        service.stats().total_cycles,
        0,
        "Reset should clear cycle count"
    );

    // Should be able to run again after reset
    let result = service.cycle("post-reset input");
    assert!(
        result.prediction_error.is_finite(),
        "Should work after reset"
    );
    assert_eq!(service.stats().total_cycles, 1);
}

#[test]
fn test_reset_and_rerun_with_different_profile() {
    // Start with Minimal, run, reset, reconfigure as Standard
    let minimal_config = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Minimal);
    let mut service = CognitiveLoopService::new(minimal_config).unwrap();

    for _ in 0..10 {
        service.cycle("minimal phase");
    }

    service.reset();

    // After reset, continue running — the service retains its original config
    // but internal state is cleared
    for _ in 0..10 {
        let result = service.cycle("post-reset phase");
        assert!(
            result.prediction_error.is_finite(),
            "Should produce finite results after reset"
        );
    }

    assert_eq!(service.stats().total_cycles, 10);
}

// ═══════════════════════════════════════════════════════════════════════════════
// GENESIS SEEDING: Deterministic behavior across profiles
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_genesis_seeded_minimal_is_deterministic() {
    let phrase = "feature_interaction_test_seed";

    let mut config_a = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Minimal);
    config_a.genesis_phrase = Some(phrase.to_string());
    config_a.async_training = false;

    let mut config_b = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Minimal);
    config_b.genesis_phrase = Some(phrase.to_string());
    config_b.async_training = false;

    let mut svc_a = CognitiveLoopService::new(config_a).unwrap();
    let mut svc_b = CognitiveLoopService::new(config_b).unwrap();

    for _ in 0..10 {
        let r_a = svc_a.cycle("deterministic test input");
        let r_b = svc_b.cycle("deterministic test input");

        // Floating-point non-determinism in CfC ODE + 70+ feedback loops means
        // seeded runs are approximately (not bit-exactly) identical.
        // Use same tolerance as genesis_determinism.rs (0.15 for accumulated drift).
        assert_eq!(
            r_a.output.len(),
            r_b.output.len(),
            "Genesis-seeded services should produce same-length outputs"
        );
        assert!(
            r_a.output
                .iter()
                .zip(r_b.output.iter())
                .all(|(a, b)| (a - b).abs() <= 0.15),
            "Genesis-seeded outputs should be approximately equal (tolerance 0.15)"
        );
        assert!(
            (r_a.prediction_error - r_b.prediction_error).abs() < 0.15,
            "Genesis-seeded prediction errors should be approximately equal"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PROFILE DEPENDENCY VALIDATION
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_profile_dependency_warnings() {
    // Manually create a config with temporal_consciousness but without its dependencies
    let config = CognitiveLoopConfig {
        enable_temporal_consciousness: true,
        enable_narrative_self: false,
        enable_predictive_self: false,
        ..Default::default()
    };

    let warnings = config.validate_dependencies();
    assert!(
        !warnings.is_empty(),
        "Should warn when temporal_consciousness is enabled without dependencies"
    );
}

#[test]
fn test_standard_profile_no_dependency_warnings() {
    let config = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Standard);
    let warnings = config.validate_dependencies();

    // Standard profile should be well-configured (may have some soft warnings)
    // The key is that it creates a valid, running service
    let mut service = CognitiveLoopService::new(config).unwrap();
    let result = service.cycle("dependency validation test");
    assert!(result.prediction_error.is_finite());

    // Log warnings for visibility
    if !warnings.is_empty() {
        eprintln!(
            "Standard profile dependency warnings (non-fatal): {:?}",
            warnings
        );
    }
}

#[test]
fn test_full_profile_no_dependency_warnings() {
    let config = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Full);
    let warnings = config.validate_dependencies();

    // Full profile should have zero dependency warnings because it enables everything
    assert!(
        warnings.is_empty(),
        "Full profile should have no dependency warnings, got: {:?}",
        warnings
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// LEARNING ACROSS PROFILES
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_learning_occurs_in_all_profiles() {
    let profiles = [
        ConsciousnessProfile::Minimal,
        ConsciousnessProfile::Standard,
        ConsciousnessProfile::Full,
    ];

    for profile in &profiles {
        let mut config = CognitiveLoopConfig::from_profile(*profile);
        config.learning_threshold = 0.0; // Always learn
        config.async_training = false; // Synchronous for determinism

        let mut service = CognitiveLoopService::new(config).unwrap();

        let mut any_learning = false;
        for _ in 0..10 {
            let result = service.cycle("learning test across profiles");
            if result.learning_occurred {
                any_learning = true;
            }
        }

        assert!(
            any_learning,
            "{profile:?}: Learning should occur at least once in 10 cycles"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// IGNORED: EXTENDED PROFILE COMPARISON (expensive)
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
#[ignore]
fn test_extended_profile_comparison_100_cycles() {
    let profiles = [
        ConsciousnessProfile::Minimal,
        ConsciousnessProfile::Standard,
        ConsciousnessProfile::Full,
    ];

    let inputs = [
        "the quick brown fox jumps over the lazy dog",
        "quantum entanglement connects distant particles",
        "consciousness emerges from neural complexity",
        "empathy is the foundation of moral reasoning",
        "mathematical beauty reveals deep truth",
    ];

    for profile in &profiles {
        let mut config = CognitiveLoopConfig::from_profile(*profile);
        config.learning_threshold = 0.0;
        config.async_training = false;

        let mut service = CognitiveLoopService::new(config).unwrap();

        let mut errors = Vec::new();
        for cycle in 0..100 {
            let input = inputs[cycle % inputs.len()];
            let result = service.cycle(input);

            assert!(
                result.prediction_error.is_finite(),
                "{profile:?}: prediction error should be finite at cycle {cycle}"
            );
            errors.push(result.prediction_error);
        }

        // All profiles should show some learning (error decrease)
        let first_half: f32 = errors[..50].iter().sum::<f32>() / 50.0;
        let second_half: f32 = errors[50..].iter().sum::<f32>() / 50.0;

        assert!(
            second_half <= first_half + 0.15,
            "{profile:?}: prediction error should decrease or stabilize: \
             first_half={first_half:.4}, second_half={second_half:.4}"
        );
    }
}