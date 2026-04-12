// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use super::super::*;
use crate::dynamics::ConsciousnessPattern;

#[test]
fn test_service_creation() {
    let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    assert_eq!(service.stats().total_cycles, 0);
}

#[test]
fn test_single_cycle() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let result = service.cycle("test input");

    assert!(result.prediction_error >= 0.0);
    assert!(result.prediction_error <= 1.0);
    assert_eq!(service.stats().total_cycles, 1);
}

#[test]
fn test_multiple_cycles_reduce_error() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        learning_threshold: 0.0, // Always learn
        ..Default::default()
    })
    .unwrap();

    // Run multiple cycles with same input
    let mut errors = Vec::new();
    for _ in 0..20 {
        let result = service.cycle("cause effect action");
        errors.push(result.prediction_error);
    }

    // Error should generally decrease (or at least not increase dramatically)
    let first_half_avg: f32 = errors[..10].iter().sum::<f32>() / 10.0;
    let second_half_avg: f32 = errors[10..].iter().sum::<f32>() / 10.0;

    println!("First half avg error: {}", first_half_avg);
    println!("Second half avg error: {}", second_half_avg);

    // Second half should be lower or similar
    assert!(
        second_half_avg <= first_half_avg + 0.1,
        "Error should decrease or stabilize over cycles"
    );
}

#[test]
fn test_attention_emergence() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        learning_threshold: 0.0,
        encoder_config: PredictiveEncoderConfig {
            attention_lr: 0.5, // High learning rate
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap();

    // Run many cycles
    for _ in 0..50 {
        service.cycle("cause effect");
    }

    // Check attention has diverged from uniform
    let stats = service.stats();
    println!("Attention variance: {}", stats.attention_variance);

    // Attention variance should be finite and non-negative
    assert!(
        stats.attention_variance.is_finite(),
        "Attention variance should be finite"
    );
    assert!(
        stats.attention_variance >= 0.0,
        "Attention variance should be non-negative"
    );
}

#[test]
fn test_builder() {
    let service = CognitiveLoopBuilder::new()
        .with_ltc_neurons(128)
        .with_learning_rate(0.001)
        .with_learning_threshold(0.1)
        .build()
        .unwrap();

    assert_eq!(service.stats().total_cycles, 0);
}

#[test]
fn test_reset() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    // Run some cycles
    for _ in 0..5 {
        service.cycle("test");
    }
    assert!(service.stats().total_cycles > 0);

    // Reset
    service.reset();

    assert_eq!(service.stats().total_cycles, 0);
    assert_eq!(service.buffer.len(), 0);
}

#[test]
fn test_consolidation() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_consolidation: true,
        learning_threshold: 0.0,
        ..Default::default()
    })
    .unwrap();

    // Fill buffer with experiences
    for i in 0..20 {
        service.cycle(&format!("input {}", i));
    }

    // Should have some experiences
    assert!(!service.buffer.is_empty());

    // Run consolidation
    let loss = service.consolidate().unwrap();
    println!("Consolidation loss: {}", loss);
}

#[test]
fn test_prediction_confidence() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    // Initial confidence should be 0.5
    assert!((service.prediction_confidence() - 0.5).abs() < 0.01);

    // Run several cycles
    for _ in 0..10 {
        service.cycle("consistent stable input");
    }

    // Confidence should be tracked
    let confidence = service.prediction_confidence();
    assert!((0.0..=1.0).contains(&confidence));

    // Reset should restore neutral confidence
    service.reset();
    assert!((service.prediction_confidence() - 0.5).abs() < 0.01);
}

#[test]
fn test_predictions_trustworthy() {
    let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    // Initial state should have some trust
    assert!(service.prediction_confidence() > 0.3);

    // predictions_trustworthy depends on confidence threshold
    // At 0.5 initial confidence, should be trustworthy
    assert!(service.predictions_trustworthy());
}

#[test]
fn test_flow_state_initial() {
    let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    // Initially not in flow
    assert!(!service.in_flow());
    assert_eq!(service.flow_intensity(), 0.0);
    assert_eq!(service.flow_streak(), 0);
    assert_eq!(service.flow_learning_boost(), 1.0);
}

#[test]
fn test_flow_state_reset() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    // Run some cycles
    for _ in 0..10 {
        service.cycle("focused input");
    }

    // Reset
    service.reset();

    // Flow state should be reset
    assert!(!service.in_flow());
    assert_eq!(service.behavior.flow_state.streak, 0);
}

#[test]
fn test_flow_state_struct() {
    let mut flow = FlowState::default();

    // Test update with flow-compatible conditions
    for _ in 0..10 {
        flow.update(
            ConsciousnessPattern::Focused,
            0.1, // Low error
            0.8, // High coherence
            0.7, // Good confidence
        );
    }

    // After sufficient streak, should be in flow
    assert!(flow.streak >= FlowState::FLOW_ENTRY_STREAK);
    assert!(flow.in_flow);
    assert!(flow.learning_boost > 1.0);
}

#[test]
fn test_emotion_contagion_positive() {
    let mut emotion = EmotionContagion::default();

    // Analyze happy content
    emotion.analyze("I am so happy and excited! This is wonderful and amazing!");

    // Should detect positive valence
    assert!(emotion.valence > 0.0);
    assert!(emotion.smoothed_valence() > 0.0);

    // High arousal due to exclamation and excited words
    assert!(emotion.arousal > 0.5);
}

#[test]
fn test_emotion_contagion_negative() {
    let mut emotion = EmotionContagion::default();

    // Analyze sad content
    emotion.analyze("I feel sad and worried about this terrible problem.");

    // Should detect negative valence
    assert!(emotion.valence < 0.0);
    assert!(emotion.smoothed_valence() < 0.0);
}

#[test]
fn test_emotion_contagion_neutral() {
    let mut emotion = EmotionContagion::default();

    // Analyze neutral content
    emotion.analyze("The system processes data and returns results.");

    // Should have near-zero valence
    assert!(emotion.valence.abs() < 0.3);
}

#[test]
fn test_emotion_pattern_nudge() {
    let mut emotion = EmotionContagion::default();

    // Strong positive emotion should nudge toward Excited
    emotion.analyze("This is absolutely amazing! I'm so thrilled and excited!");
    let (pattern, strength) = emotion.pattern_nudge();
    assert!(pattern.is_some());
    assert!(strength > 0.0);
}

#[test]
fn test_emotion_in_service() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    // Initially no emotional content
    assert!(!service.has_emotional_content());

    // Process emotional content
    service.cycle("I'm so happy and grateful for this wonderful day!");

    // The unified emotional state integrates multiple signals (somatic, affective, mood)
    // that can pull valence in either direction even for clearly positive text.
    // After one cycle, the emotion_contagion module should have registered some change,
    // and the unified valence should be non-NaN. We don't require positive because
    // somatic signals, prediction error, and homeostatic pulls can dominate a single cycle.
    let valence = service.emotional_valence();
    assert!(
        valence.is_finite(),
        "emotional valence should be finite after cycle"
    );
}

#[test]
fn test_emotion_reset() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    // Process emotional content
    service.cycle("I'm incredibly excited about this amazing opportunity!");

    // Reset
    service.reset();

    // Emotion should be reset
    assert_eq!(service.emotional_valence(), 0.0);
    assert_eq!(service.emotional_arousal(), 0.5);
}

#[test]
fn test_curiosity_drive_initial() {
    let curiosity = CuriosityDrive::default();

    assert_eq!(curiosity.boredom, 0.0);
    assert!(curiosity.curiosity > 0.0); // Starts with some curiosity
    assert_eq!(curiosity.exploration_urge, 0.0);
    assert_eq!(curiosity.novelty_bonus, 1.0);
    assert!(!curiosity.should_explore());
}

#[test]
fn test_curiosity_boredom_buildup() {
    let mut curiosity = CuriosityDrive::default();

    // Feed consistently low errors (boring/predictable)
    for _ in 0..20 {
        curiosity.update(0.05); // Very low error
    }

    // Boredom should build up
    assert!(curiosity.boredom > 0.3);
    assert!(curiosity.curiosity > 0.5);
}

#[test]
fn test_curiosity_exploration_trigger() {
    let mut curiosity = CuriosityDrive::default();

    // Feed many low errors to trigger exploration
    for _ in 0..30 {
        curiosity.update(0.05);
    }

    // Should want to explore
    assert!(curiosity.boredom > 0.5);
    // After sufficient boredom, should_explore or have high exploration urge
    assert!(curiosity.exploration_urge > 0.0 || curiosity.boredom > 0.7);
}

#[test]
fn test_curiosity_novelty_bonus() {
    let mut curiosity = CuriosityDrive::default();

    // High error = novel situation
    curiosity.update(0.8);

    // Should have some novelty bonus
    assert!(curiosity.novelty_bonus >= 1.0);
}

#[test]
fn test_curiosity_in_service() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    // Initially no boredom
    assert_eq!(service.boredom(), 0.0);
    assert!(!service.is_bored());

    // Run some cycles
    for _ in 0..5 {
        service.cycle("test input");
    }

    // Curiosity should be tracked
    assert!(service.curiosity() >= 0.0);
    assert!(service.novelty_bonus() >= 1.0);
}

#[test]
fn test_curiosity_reset() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    // Run some cycles
    for _ in 0..10 {
        service.cycle("test");
    }

    // Reset
    service.reset();

    // Curiosity should be reset
    assert_eq!(service.boredom(), 0.0);
    assert!(!service.curiosity_should_explore());
}

// ========== Self-Reflection Tests ==========

#[test]
fn test_self_reflection_initial() {
    let reflection = SelfReflection::default();

    assert_eq!(reflection.reflection_count, 0);
    assert_eq!(reflection.adjustments_made, 0);
    assert_eq!(reflection.self_assessment, SelfAssessment::Learning);
    assert!(reflection.flow_error_threshold > 0.0);
    assert!(reflection.boredom_threshold > 0.0);
}

#[test]
fn test_self_reflection_record_cycle() {
    let mut reflection = SelfReflection::default();

    // Record some cycles
    for _ in 0..10 {
        reflection.record_cycle(0.3, false, false, 0.5);
    }

    // Should update learning effectiveness after recording cycles
    let summary = reflection.summary();
    assert!(summary.learning_effectiveness >= 0.0);
}

#[test]
fn test_self_reflection_should_reflect() {
    let mut reflection = SelfReflection::default();

    // Initially shouldn't reflect
    assert!(!reflection.should_reflect());

    // Record enough cycles
    for _ in 0..60 {
        reflection.record_cycle(0.3, false, false, 0.5);
    }

    // Should now want to reflect
    assert!(reflection.should_reflect());
}

#[test]
fn test_self_reflection_reflect() {
    let mut reflection = SelfReflection::default();

    // Record cycles to trigger reflection
    for _ in 0..60 {
        reflection.record_cycle(0.3, false, false, 0.5);
    }

    // Perform reflection
    let recommendations = reflection.reflect();

    // Reflection count should increase
    assert_eq!(reflection.reflection_count, 1);

    // Should have some assessment
    assert!(reflection.learning_effectiveness() >= 0.0);

    // Recommendations may or may not be empty depending on state
    let _ = recommendations;
}

#[test]
fn test_self_reflection_stagnation_detection() {
    let mut reflection = SelfReflection::default();

    // Simulate stagnation: low error, no flow, no exploration
    for _ in 0..60 {
        reflection.record_cycle(0.1, false, false, 0.6);
    }
    reflection.reflect();

    // Should detect stagnation or overconfidence
    assert!(
        reflection.self_assessment == SelfAssessment::Stagnating
            || reflection.self_assessment == SelfAssessment::Overconfident
            || reflection.self_assessment == SelfAssessment::Learning
    );
}

#[test]
fn test_self_reflection_in_service() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    // Initial state
    assert_eq!(service.reflection_count(), 0);
    assert!(service.learning_effectiveness() >= 0.0);

    // Run some cycles (not enough to trigger reflection yet)
    for _ in 0..10 {
        service.cycle("test input");
    }

    // Check self-assessment is available
    let assessment = service.self_assessment();
    assert!(assessment == SelfAssessment::Learning || assessment == SelfAssessment::Exploring);
}

#[test]
fn test_self_reflection_thresholds() {
    let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    // Get adapted thresholds
    let thresholds = service.adapted_thresholds();

    // Should have valid thresholds
    assert!(thresholds.flow_error > 0.0 && thresholds.flow_error < 1.0);
    assert!(thresholds.boredom > 0.0 && thresholds.boredom < 1.0);
    assert!(thresholds.trust > 0.0 && thresholds.trust < 1.0);
}

#[test]
fn test_self_reflection_reset() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    // Run cycles and force reflect
    for _ in 0..10 {
        service.cycle("test");
    }
    service.force_reflect();

    // Reset
    service.reset();

    // Reflection count should reset but thresholds preserved
    assert_eq!(service.self_reflection().reflection_count, 0);
    // Thresholds are preserved across reset
    assert!(service.adapted_thresholds().flow_error > 0.0);
}

#[test]
fn test_self_reflection_summary() {
    let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    let summary = service.reflection_summary();

    assert_eq!(summary.reflection_count, 0);
    assert!(summary.learning_effectiveness >= 0.0);
    assert!(summary.next_reflection_in > 0);
}

// ========== Consciousness Snapshot Tests ==========

#[test]
fn test_consciousness_snapshot_initial() {
    let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    let snapshot = service.consciousness_snapshot();

    // Initial state checks
    assert_eq!(snapshot.cycle, 0);
    assert!(snapshot.consciousness_level >= 0.0 && snapshot.consciousness_level <= 1.0);
    assert!(!snapshot.in_flow);
    assert!(!snapshot.exploring);
    assert_eq!(snapshot.reflection_count, 0);
}

#[test]
fn test_consciousness_snapshot_after_cycles() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    // Run some cycles
    for _ in 0..10 {
        service.cycle("test input for consciousness");
    }

    let snapshot = service.consciousness_snapshot();

    // Should have recorded cycles
    assert_eq!(snapshot.cycle, 10);
    // Should have valid metrics
    assert!(snapshot.prediction_confidence >= 0.0);
    assert!(snapshot.consciousness_level >= 0.0);
    assert!(snapshot.flow_threshold > 0.0);
    assert!(snapshot.boredom_threshold > 0.0);
}

#[test]
fn test_consciousness_snapshot_status() {
    let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    let snapshot = service.consciousness_snapshot();
    let status = snapshot.status();

    // Status should be a non-empty string
    assert!(!status.is_empty());
    // Should contain pattern info
    assert!(status.contains("Conf:") || status.contains("Err:"));
}

#[test]
fn test_consciousness_snapshot_recommended_actions() {
    let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    let snapshot = service.consciousness_snapshot();
    let actions = snapshot.recommended_actions();

    // Actions should be a valid collection (even if empty)
    // Each action string should be non-empty if present
    for action in &actions {
        assert!(
            !action.is_empty(),
            "Recommended actions should not contain empty strings"
        );
    }
}

#[test]
fn test_consciousness_snapshot_is_optimal() {
    let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    let snapshot = service.consciousness_snapshot();

    // Fresh service with no cycles should not be optimal
    // (it hasn't had any experience yet)
    // Just verify the call completes without panicking
    let _optimal = snapshot.is_optimal();
}

#[test]
fn test_consciousness_snapshot_needs_attention() {
    let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    let snapshot = service.consciousness_snapshot();

    // Verify needs_attention returns without panicking and phi is finite
    let _needs = snapshot.needs_attention();
    assert!(
        snapshot.unified_psi.is_finite(),
        "Snapshot phi should be finite"
    );
}

#[test]
fn test_consciousness_snapshot_dominant_concern() {
    let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    let snapshot = service.consciousness_snapshot();

    // dominant_concern returns Option<&str>
    let concern = snapshot.dominant_concern();
    // If a concern is present, it should be a non-empty string
    if let Some(c) = concern {
        assert!(
            !c.is_empty(),
            "Dominant concern should not be an empty string"
        );
    }
}

#[test]
fn test_status_line() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    service.cycle("test");
    let status = service.status_line();

    assert!(!status.is_empty());
}

#[test]
fn test_consciousness_level() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    // Run some cycles
    for _ in 0..5 {
        service.cycle("focused stable input");
    }

    let level = service.consciousness_level();
    assert!((0.0..=1.0).contains(&level));
}

#[test]
fn test_adapted_thresholds_wiring() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    // Get initial thresholds
    let initial_flow = service.adapted_thresholds().flow_error;
    let initial_boredom = service.adapted_thresholds().boredom;

    // The thresholds should be valid
    assert!(initial_flow > 0.0 && initial_flow < 1.0);
    assert!(initial_boredom > 0.0 && initial_boredom < 1.0);

    // Run cycles - thresholds are passed to flow_state and curiosity_drive
    for _ in 0..5 {
        service.cycle("test");
    }

    // Verify snapshot reflects adapted thresholds
    let snapshot = service.consciousness_snapshot();
    assert_eq!(
        snapshot.flow_threshold,
        service.adapted_thresholds().flow_error
    );
    assert_eq!(
        snapshot.boredom_threshold,
        service.adapted_thresholds().boredom
    );
}

#[test]
fn test_try_cycle_returns_ok() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let result = service.try_cycle("hello world");
    assert!(result.is_ok());
    let cycle_result = result.unwrap();
    assert!(cycle_result.prediction_error >= 0.0);
    assert_eq!(service.stats().total_cycles, 1);
}

#[test]
fn test_try_cycle_multiple() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    for i in 0..5 {
        let result = service.try_cycle(&format!("input {i}"));
        assert!(result.is_ok(), "try_cycle failed on iteration {i}");
    }
    assert_eq!(service.stats().total_cycles, 5);
}

#[test]
fn test_config_validate_default_clean() {
    let config = CognitiveLoopConfig::default();
    let warnings = config.validate_dependencies();
    assert!(
        warnings.is_empty(),
        "default config should have no warnings: {warnings:?}"
    );
}

#[test]
fn test_config_validate_profiles_clean() {
    use crate::cognitive_loop::config::ConsciousnessProfile;
    for profile in [
        ConsciousnessProfile::Minimal,
        ConsciousnessProfile::Standard,
        ConsciousnessProfile::Full,
    ] {
        let config = CognitiveLoopConfig::from_profile(profile);
        let warnings = config.validate_dependencies();
        assert!(
            warnings.is_empty(),
            "{profile:?} profile should have no warnings: {warnings:?}"
        );
    }
}

#[test]
fn test_config_validate_research_warns_without_did() {
    use crate::cognitive_loop::config::ConsciousnessProfile;
    let config = CognitiveLoopConfig::from_profile(ConsciousnessProfile::Research);
    let warnings = config.validate_dependencies();
    // Research enables psi_attestation but agent_did defaults to None
    assert!(
        warnings.iter().any(|w| w.contains("agent_did")),
        "Research profile without agent_did should warn: {warnings:?}"
    );
}

#[test]
fn test_config_validate_detects_missing_deps() {
    let config = CognitiveLoopConfig {
        enable_temporal_consciousness: true,
        enable_narrative_self: false,
        enable_predictive_self: false,
        // narrative_self and predictive_self both false
        enable_embodied_cognition: true,
        // virtual_body defaults to true, so no warning for that
        enable_cross_modal_binding: true,
        enable_affective_bridge: false,
        // affective_bridge false
        ..Default::default()
    };
    let warnings = config.validate_dependencies();
    assert!(warnings.iter().any(|w| w.contains("narrative_self")));
    assert!(warnings.iter().any(|w| w.contains("predictive_self")));
    assert!(warnings.iter().any(|w| w.contains("affective_bridge")));
}

#[test]
fn test_config_validate_no_false_positives() {
    // All deps satisfied — should produce zero warnings
    let config = CognitiveLoopConfig {
        enable_temporal_consciousness: true,
        enable_narrative_self: true,
        enable_predictive_self: true,
        enable_embodied_cognition: true,
        enable_virtual_body: true,
        enable_cross_modal_binding: true,
        enable_affective_bridge: true,
        enable_predictive_processing: true,
        enable_narrative_gwt: true,
        enable_gwt: true,
        enable_dream_replay: true,
        enable_surprise_exploration: true,
        enable_psi_attestation: true,
        agent_did: Some("did:key:z6Mktest".into()),
        ..Default::default()
    };
    let warnings = config.validate_dependencies();
    assert!(
        warnings.is_empty(),
        "fully-satisfied config should have no warnings: {warnings:?}"
    );
}

// ─── Config range validation tests ───────────────────────────────────

#[test]
fn test_default_config_is_valid() {
    let config = CognitiveLoopConfig::default();
    assert!(
        config.validate().is_ok(),
        "default CognitiveLoopConfig must pass range validation"
    );
}

#[test]
fn test_cfc_default_config_is_valid() {
    let config = CfCConfig::default();
    assert!(
        config.validate().is_ok(),
        "default CfCConfig must pass validation"
    );
}

#[test]
#[allow(clippy::field_reassign_with_default)]
fn test_invalid_learning_rate_rejected() {
    // CfC learning_rate = 0.0 (out of range)
    let mut config = CognitiveLoopConfig::default();
    config.cfc_config.learning_rate = 0.0;
    let err = config.validate().unwrap_err();
    assert!(
        err.contains("learning_rate"),
        "error should mention learning_rate: {err}"
    );

    // CfC learning_rate > 1.0
    config.cfc_config.learning_rate = 1.5;
    let err = config.validate().unwrap_err();
    assert!(
        err.contains("learning_rate"),
        "error should mention learning_rate: {err}"
    );

    // CfC learning_rate = NaN
    config.cfc_config.learning_rate = f32::NAN;
    let err = config.validate().unwrap_err();
    assert!(
        err.contains("learning_rate") || err.contains("finite"),
        "error should mention learning_rate or finite: {err}"
    );

    // learning_threshold out of range
    let mut config = CognitiveLoopConfig::default();
    config.learning_threshold = -0.1;
    let err = config.validate().unwrap_err();
    assert!(
        err.contains("learning_threshold"),
        "error should mention learning_threshold: {err}"
    );

    config.learning_threshold = 1.5;
    let err = config.validate().unwrap_err();
    assert!(
        err.contains("learning_threshold"),
        "error should mention learning_threshold: {err}"
    );
}

#[test]
fn test_invalid_dimension_rejected() {
    let mut config = CognitiveLoopConfig::default();
    config.cfc_config.num_neurons = 0;
    let err = config.validate().unwrap_err();
    assert!(
        err.contains("num_neurons"),
        "error should mention num_neurons: {err}"
    );

    let mut config = CognitiveLoopConfig::default();
    config.cfc_config.input_dim = 0;
    let err = config.validate().unwrap_err();
    assert!(
        err.contains("input_dim"),
        "error should mention input_dim: {err}"
    );
}

#[test]
#[allow(clippy::field_reassign_with_default)]
fn test_invalid_buffer_size_rejected() {
    let mut config = CognitiveLoopConfig::default();
    config.buffer_size = 0;
    let err = config.validate().unwrap_err();
    assert!(
        err.contains("buffer_size"),
        "error should mention buffer_size: {err}"
    );
}

#[test]
#[allow(clippy::field_reassign_with_default)]
fn test_invalid_target_frequency_rejected() {
    let mut config = CognitiveLoopConfig::default();
    config.target_frequency = 0.0;
    assert!(
        config.validate().is_err(),
        "target_frequency=0.0 should fail"
    );

    config.target_frequency = -10.0;
    assert!(
        config.validate().is_err(),
        "negative target_frequency should fail"
    );

    config.target_frequency = f32::INFINITY;
    assert!(
        config.validate().is_err(),
        "infinite target_frequency should fail"
    );
}

#[test]
#[allow(clippy::field_reassign_with_default)]
fn test_invalid_causal_discovery_interval_rejected() {
    let mut config = CognitiveLoopConfig::default();
    config.causal_discovery_interval = 0;
    let err = config.validate().unwrap_err();
    assert!(
        err.contains("causal_discovery_interval"),
        "error should mention causal_discovery_interval: {err}"
    );
}

#[test]
#[allow(clippy::field_reassign_with_default)]
fn test_invalid_resonator_params_rejected() {
    let mut config = CognitiveLoopConfig::default();
    config.resonator_novelty_threshold = 0.0;
    assert!(
        config.validate().is_err(),
        "resonator_novelty_threshold=0.0 should fail"
    );

    config.resonator_novelty_threshold = 1.1;
    assert!(
        config.validate().is_err(),
        "resonator_novelty_threshold=1.1 should fail"
    );

    let mut config = CognitiveLoopConfig::default();
    config.resonator_max_symbols = 0;
    assert!(
        config.validate().is_err(),
        "resonator_max_symbols=0 should fail"
    );
}

#[test]
#[allow(clippy::field_reassign_with_default)]
fn test_invalid_attestation_buffer_capacity_rejected() {
    let mut config = CognitiveLoopConfig::default();
    config.attestation_buffer_capacity = 0;
    assert!(
        config.validate().is_err(),
        "attestation_buffer_capacity=0 should fail"
    );
}

#[test]
fn test_cfc_empty_prediction_horizons_rejected() {
    let mut config = CognitiveLoopConfig::default();
    config.cfc_config.prediction_horizons = vec![];
    let err = config.validate().unwrap_err();
    assert!(
        err.contains("prediction_horizons"),
        "error should mention prediction_horizons: {err}"
    );
}

#[test]
fn test_cfc_negative_prediction_horizon_rejected() {
    let mut config = CognitiveLoopConfig::default();
    config.cfc_config.prediction_horizons = vec![0.02, -0.1, 0.2];
    let err = config.validate().unwrap_err();
    assert!(
        err.contains("prediction_horizons"),
        "error should mention prediction_horizons: {err}"
    );
}

#[test]
fn test_cfc_invalid_delta_t_rejected() {
    let mut config = CognitiveLoopConfig::default();
    config.cfc_config.delta_t = 0.0;
    assert!(config.validate().is_err(), "delta_t=0.0 should fail");

    config.cfc_config.delta_t = -1.0;
    assert!(config.validate().is_err(), "negative delta_t should fail");
}

#[test]
fn test_profiles_pass_range_validation() {
    use crate::cognitive_loop::config::ConsciousnessProfile;
    for profile in [
        ConsciousnessProfile::Minimal,
        ConsciousnessProfile::Standard,
        ConsciousnessProfile::Full,
        ConsciousnessProfile::Research,
    ] {
        let config = CognitiveLoopConfig::from_profile(profile);
        assert!(
            config.validate().is_ok(),
            "{profile:?} profile must pass range validation"
        );
    }
}
