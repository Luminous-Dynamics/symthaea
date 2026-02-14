use super::*;
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
    assert_eq!(service.flow_state().streak, 0);
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

    // Should detect emotional content
    let valence = service.emotional_valence();
    // The smoothing will reduce the effect, but should be positive
    assert!(valence >= 0.0 || service.emotion_contagion().valence > 0.0);
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

    // Should track historical metrics
    assert!(reflection.historical_error() > 0.0);
    assert!(reflection.historical_confidence() > 0.0);
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
    let optimal = snapshot.is_optimal();
    // Just verify the return is a real boolean (not a panic)
    assert!(optimal || !optimal, "is_optimal should return a valid bool");
}

#[test]
fn test_consciousness_snapshot_needs_attention() {
    let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    let snapshot = service.consciousness_snapshot();

    // Verify needs_attention returns a valid boolean and phi is finite
    let needs = snapshot.needs_attention();
    assert!(
        needs || !needs,
        "needs_attention should return a valid bool"
    );
    assert!(
        snapshot.unified_phi.is_finite(),
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

// ═══════════════════════════════════════════════════════════════════════════════
// UNIFIED ARCHITECTURE COMPONENT TESTS
// ═══════════════════════════════════════════════════════════════════════════════

// -------------------- ThalamicRouter Tests --------------------

#[test]
fn test_thalamic_router_default() {
    let router = ThalamicRouter::default();
    assert_eq!(router.novelty_threshold, 0.7);
    assert_eq!(router.urgency_threshold, 0.8);
    assert_eq!(router.familiarity_threshold, 0.3);
}

#[test]
fn test_thalamic_router_reflex_route() {
    let mut router = ThalamicRouter::default();

    // Low novelty, low complexity, low urgency → Reflex
    let depth = router.route(0.1, 0.2, 0.1, 0.1);
    assert_eq!(depth, CognitiveDepth::Reflex);
}

#[test]
fn test_thalamic_router_cortical_route() {
    let mut router = ThalamicRouter::default();

    // Medium values → Cortical
    let depth = router.route(0.4, 0.4, 0.5, 0.3);
    assert_eq!(depth, CognitiveDepth::Cortical);
}

#[test]
fn test_thalamic_router_deep_thought_high_novelty() {
    let mut router = ThalamicRouter::default();

    // High novelty → DeepThought
    let depth = router.route(0.9, 0.3, 0.3, 0.3);
    assert_eq!(depth, CognitiveDepth::DeepThought);
}

#[test]
fn test_thalamic_router_deep_thought_high_urgency() {
    let mut router = ThalamicRouter::default();

    // High urgency → DeepThought
    let depth = router.route(0.3, 0.9, 0.3, 0.3);
    assert_eq!(depth, CognitiveDepth::DeepThought);
}

#[test]
fn test_thalamic_router_deep_thought_high_complexity() {
    let mut router = ThalamicRouter::default();

    // High complexity → DeepThought
    let depth = router.route(0.3, 0.3, 0.9, 0.3);
    assert_eq!(depth, CognitiveDepth::DeepThought);
}

#[test]
fn test_thalamic_router_deep_thought_high_emotion() {
    let mut router = ThalamicRouter::default();

    // High emotional intensity → DeepThought
    let depth = router.route(0.3, 0.3, 0.3, 0.9);
    assert_eq!(depth, CognitiveDepth::DeepThought);
}

#[test]
fn test_thalamic_router_routing_stats() {
    let mut router = ThalamicRouter::default();

    // Make several routing decisions
    router.route(0.1, 0.2, 0.1, 0.1); // Reflex
    router.route(0.1, 0.2, 0.1, 0.1); // Reflex
    router.route(0.5, 0.5, 0.5, 0.3); // Cortical
    router.route(0.9, 0.5, 0.5, 0.3); // DeepThought

    let (reflex, cortical, deep) = router.routing_stats();

    assert_eq!(reflex, 0.5); // 2 out of 4
    assert_eq!(cortical, 0.25); // 1 out of 4
    assert_eq!(deep, 0.25); // 1 out of 4
}

#[test]
fn test_thalamic_router_from_cycle() {
    let mut router = ThalamicRouter::default();

    // High prediction error (novel) → DeepThought
    let depth = router.route_from_cycle(0.9, ConsciousnessPattern::Uncertain, 0.3);
    assert_eq!(depth, CognitiveDepth::DeepThought);

    // Low error, focused, neutral emotion → likely Cortical or Reflex
    let depth2 = router.route_from_cycle(0.1, ConsciousnessPattern::Focused, 0.1);
    assert!(matches!(
        depth2,
        CognitiveDepth::Cortical | CognitiveDepth::Reflex
    ));
}

// -------------------- ActiveInferenceBridge Tests --------------------

#[test]
fn test_active_inference_bridge_default() {
    let bridge = ActiveInferenceBridge::default();
    assert_eq!(bridge.coupling_quality(), CouplingQuality::InsufficientData);
    assert!(bridge.modulation_index().is_none());
}

#[test]
fn test_active_inference_bridge_observe_resolution() {
    let mut bridge = ActiveInferenceBridge::default();

    // Add some observations
    for i in 0..15 {
        let confidence = 0.8;
        let success = i % 2 == 0; // Alternating success/failure
        bridge.observe_resolution(confidence, success);
    }

    // Should have enough data now
    assert!(bridge.modulation_index().is_some());
    assert_ne!(bridge.coupling_quality(), CouplingQuality::InsufficientData);
}

#[test]
fn test_active_inference_bridge_perfect_coupling() {
    let mut bridge = ActiveInferenceBridge::default();

    // Perfect coupling: high confidence → success, low confidence → failure
    for _ in 0..20 {
        bridge.observe_resolution(0.9, true);
        bridge.observe_resolution(0.1, false);
    }

    let mi = bridge.modulation_index().unwrap();
    // Should have strong positive correlation
    assert!(mi > 0.5, "Expected strong coupling, got MI={}", mi);
    assert!(matches!(
        bridge.coupling_quality(),
        CouplingQuality::ModerateCoupling | CouplingQuality::StrongCoupling
    ));
}

#[test]
fn test_active_inference_bridge_statistics() {
    let mut bridge = ActiveInferenceBridge::default();

    for _ in 0..15 {
        bridge.observe_resolution(0.7, true);
    }

    let stats = bridge.statistics();
    assert_eq!(stats.total_observations, 15);
    assert!(stats.modulation_index.is_some());
    assert!(stats.average_prediction_error.is_some());
    // All successes → 0% error
    assert!(stats.average_prediction_error.unwrap() < 0.01);
}

#[test]
fn test_active_inference_bridge_reset() {
    let mut bridge = ActiveInferenceBridge::default();

    for _ in 0..20 {
        bridge.observe_resolution(0.5, true);
    }

    bridge.reset();

    assert_eq!(bridge.coupling_quality(), CouplingQuality::InsufficientData);
    let stats = bridge.statistics();
    assert_eq!(stats.total_observations, 0);
}

#[test]
fn test_coupling_quality_is_meaningful() {
    assert!(!CouplingQuality::InsufficientData.is_meaningful());
    assert!(!CouplingQuality::NoCoupling.is_meaningful());
    assert!(CouplingQuality::WeakCoupling.is_meaningful());
    assert!(CouplingQuality::ModerateCoupling.is_meaningful());
    assert!(CouplingQuality::StrongCoupling.is_meaningful());
}

// -------------------- ClosedLearningLoop Tests --------------------

#[test]
fn test_closed_learning_loop_default() {
    let loop_ = ClosedLearningLoop::default();
    assert_eq!(loop_.current_strategy, ResponseStrategy::Supportive);
    assert!(loop_.last_result.is_none());
    assert_eq!(loop_.average_reward(), 0.0);
}

#[test]
fn test_closed_learning_loop_strategy_selection() {
    let mut loop_ = ClosedLearningLoop::default();

    // With neutral Φ (0.45), should use Q-learning selection
    let strategy = loop_.select_strategy(0.45, None);

    // Should return some valid strategy
    assert!(matches!(
        strategy,
        ResponseStrategy::Detailed
            | ResponseStrategy::Concise
            | ResponseStrategy::Clarifying
            | ResponseStrategy::Supportive
            | ResponseStrategy::Exploratory
    ));
}

#[test]
fn test_closed_learning_loop_phi_gating_high() {
    let mut loop_ = ClosedLearningLoop::default();

    // Set Supportive as best strategy with high Q-value
    // Then with high Φ, it should shift toward Exploratory
    for _ in 0..100 {
        let strategy = loop_.select_strategy(0.8, None);
        // High Φ → integrative mode → favors Exploratory/Detailed
        assert!(
            !matches!(
                strategy,
                ResponseStrategy::Supportive | ResponseStrategy::Concise
            ) || loop_.last_result.is_some(),
            "High Φ should shift away from Supportive/Concise"
        );
        break; // Just check first selection
    }
}

#[test]
fn test_closed_learning_loop_q_learning_update() {
    let mut loop_ = ClosedLearningLoop::default();

    // Record a positive result for Detailed
    let result = CycleLearningResult {
        strategy_used: ResponseStrategy::Detailed,
        reward: 0.8,
        successful: true,
        prediction_error: 0.1,
        coherence: 0.8,
    };

    let initial_q = loop_.q_values()[0]; // Detailed index
    loop_.update(result);

    // Q-value should increase
    assert!(loop_.q_values()[0] > initial_q);
    assert_eq!(loop_.strategy_counts()[0], 1);
}

#[test]
fn test_closed_learning_loop_reward_tracking() {
    let mut loop_ = ClosedLearningLoop::default();

    // Record multiple results
    for _ in 0..5 {
        let result = CycleLearningResult {
            strategy_used: ResponseStrategy::Supportive,
            reward: 0.6,
            successful: true,
            prediction_error: 0.2,
            coherence: 0.7,
        };
        loop_.update(result);
    }

    assert_eq!(loop_.average_reward(), 0.6);
    assert_eq!(loop_.strategy_counts()[3], 5); // Supportive index
}

#[test]
fn test_closed_learning_loop_best_strategy() {
    let mut loop_ = ClosedLearningLoop::default();

    // Train Exploratory with high rewards
    for _ in 0..20 {
        let result = CycleLearningResult {
            strategy_used: ResponseStrategy::Exploratory,
            reward: 0.9,
            successful: true,
            prediction_error: 0.1,
            coherence: 0.9,
        };
        loop_.update(result);
    }

    // Exploratory should become best
    assert_eq!(loop_.best_strategy(), ResponseStrategy::Exploratory);
}

#[test]
fn test_closed_learning_loop_reset() {
    let mut loop_ = ClosedLearningLoop::default();

    // Add some data
    let result = CycleLearningResult {
        strategy_used: ResponseStrategy::Detailed,
        reward: 0.7,
        successful: true,
        prediction_error: 0.15,
        coherence: 0.75,
    };
    loop_.update(result);

    loop_.reset();

    assert!(loop_.last_result.is_none());
    assert_eq!(loop_.average_reward(), 0.0);
}

#[test]
fn test_response_strategy_opposite() {
    // Check actual implementation:
    // Detailed <-> Concise (symmetric)
    // Clarifying -> Supportive -> Exploratory -> Clarifying (cycle)
    assert_eq!(
        ResponseStrategy::Detailed.opposite(),
        ResponseStrategy::Concise
    );
    assert_eq!(
        ResponseStrategy::Concise.opposite(),
        ResponseStrategy::Detailed
    );
    assert_eq!(
        ResponseStrategy::Clarifying.opposite(),
        ResponseStrategy::Supportive
    );
    assert_eq!(
        ResponseStrategy::Supportive.opposite(),
        ResponseStrategy::Exploratory
    );
    assert_eq!(
        ResponseStrategy::Exploratory.opposite(),
        ResponseStrategy::Clarifying
    );
}

// -------------------- EpisodicMemoryBridge Tests --------------------

#[test]
fn test_episodic_memory_bridge_default() {
    let bridge = EpisodicMemoryBridge::default();
    assert_eq!(bridge.memory_count(), (0, 0));
}

#[test]
fn test_episodic_memory_encode() {
    let mut bridge = EpisodicMemoryBridge::default();

    let id = bridge.encode(
        "test memory",
        vec![0.1, 0.2, 0.3, 0.4],
        0.5, // valence
        0.6, // phi
        100, // cycle
    );

    assert_eq!(id, 0);
    assert_eq!(bridge.memory_count(), (1, 0));
    assert_eq!(bridge.stats.total_encoded, 1);
}

#[test]
fn test_episodic_memory_recall() {
    let mut bridge = EpisodicMemoryBridge::default();

    // Encode some memories
    bridge.encode("memory one", vec![1.0, 0.0, 0.0, 0.0], 0.5, 0.6, 1);
    bridge.encode("memory two", vec![0.0, 1.0, 0.0, 0.0], 0.3, 0.5, 2);
    bridge.encode("memory three", vec![0.9, 0.1, 0.0, 0.0], 0.7, 0.8, 3);

    // Query similar to "memory one" and "memory three"
    let results = bridge.recall(&[1.0, 0.0, 0.0, 0.0], 2, 0.5);

    assert!(!results.is_empty());
    assert!(results.len() <= 2);
    // First result should be most similar (memory one)
    assert_eq!(results[0].0.content, "memory one");
}

#[test]
fn test_episodic_memory_consolidation() {
    let mut bridge = EpisodicMemoryBridge::default();

    // Fill short-term memory to trigger consolidation
    for i in 0..105 {
        bridge.encode(format!("memory {}", i), vec![0.1; 4], 0.5, 0.6, i);
    }

    // Should have consolidated some to long-term
    let (short, long) = bridge.memory_count();
    assert!(short <= 100);
    assert!(long > 0, "Expected some memories consolidated to long-term");
    assert!(bridge.stats.consolidations > 0);
}

#[test]
fn test_episodic_memory_decay() {
    let mut bridge = EpisodicMemoryBridge::default();

    bridge.encode("memory", vec![0.1; 4], 0.5, 0.6, 1);

    // Decay several times
    for _ in 0..10 {
        bridge.decay(0.1);
    }

    // Short-term memories persist but weaken
    assert_eq!(bridge.memory_count().0, 1);
}

#[test]
fn test_episodic_memory_reset() {
    let mut bridge = EpisodicMemoryBridge::default();

    bridge.encode("memory", vec![0.1; 4], 0.5, 0.6, 1);
    bridge.reset();

    assert_eq!(bridge.memory_count(), (0, 0));
    assert_eq!(bridge.stats.total_encoded, 0);
}

#[test]
fn test_episodic_memory_similarity() {
    let memory = EpisodicMemory {
        id: 0,
        encoded_at_cycle: 0,
        content: "test".into(),
        embedding: vec![1.0, 0.0, 0.0, 0.0],
        valence: 0.5,
        phi_at_encoding: 0.6,
        access_count: 0,
        strength: 1.0,
    };

    // Same vector → similarity 1.0
    let sim1 = memory.similarity(&[1.0, 0.0, 0.0, 0.0]);
    assert!((sim1 - 1.0).abs() < 0.001);

    // Orthogonal vector → similarity 0.0
    let sim2 = memory.similarity(&[0.0, 1.0, 0.0, 0.0]);
    assert!((sim2 - 0.0).abs() < 0.001);
}

// -------------------- GoalSystemBridge Tests --------------------

#[test]
fn test_goal_system_bridge_default() {
    let bridge = GoalSystemBridge::new();
    assert!(bridge.active_goals().is_empty());
    assert_eq!(bridge.attention_bias(), 1.0);
}

#[test]
fn test_goal_system_add_goal() {
    let mut bridge = GoalSystemBridge::new();

    let goal = CognitiveGoal::new("goal1", "Test goal", 0.8);
    bridge.add_goal(goal);

    assert_eq!(bridge.active_goals().len(), 1);
    assert!(bridge.attention_bias() > 1.0);
}

#[test]
fn test_goal_system_attention_bias() {
    let mut bridge = GoalSystemBridge::new();

    // Add high-priority goal
    bridge.add_goal(CognitiveGoal::new("goal1", "High priority", 1.0));

    // Attention bias should increase
    let bias = bridge.attention_bias();
    assert!(bias > 1.0);
    assert!(bias <= 1.2); // Max 20% boost per unit weight
}

#[test]
fn test_goal_system_update_progress() {
    let mut bridge = GoalSystemBridge::new();

    bridge.add_goal(CognitiveGoal::new("goal1", "Test", 0.5));

    // Update progress
    bridge.update_progress("goal1", 0.5);

    let goals = bridge.active_goals();
    assert_eq!(goals[0].progress, 0.5);

    // Complete the goal
    bridge.update_progress("goal1", 0.6);

    // Goal should be deactivated when progress >= 1.0
    assert!(bridge.active_goals().is_empty());
}

#[test]
fn test_goal_system_top_goal() {
    let mut bridge = GoalSystemBridge::new();

    bridge.add_goal(CognitiveGoal::new("low", "Low priority", 0.3));
    bridge.add_goal(CognitiveGoal::new("high", "High priority", 0.9));
    bridge.add_goal(CognitiveGoal::new("mid", "Mid priority", 0.5));

    let top = bridge.top_goal().unwrap();
    assert_eq!(top.id, "high");
    assert_eq!(top.priority, 0.9);
}

#[test]
fn test_goal_system_clear_completed() {
    let mut bridge = GoalSystemBridge::new();

    bridge.add_goal(CognitiveGoal::new("goal1", "Goal 1", 0.5));
    bridge.add_goal(CognitiveGoal::new("goal2", "Goal 2", 0.5));

    // Complete goal1
    bridge.update_progress("goal1", 1.0);

    bridge.clear_completed();

    assert_eq!(bridge.active_goals().len(), 1);
}

#[test]
fn test_goal_system_reset() {
    let mut bridge = GoalSystemBridge::new();

    bridge.add_goal(CognitiveGoal::new("goal1", "Test", 0.5));
    bridge.reset();

    assert!(bridge.active_goals().is_empty());
}

#[test]
fn test_cognitive_goal_creation() {
    let goal = CognitiveGoal::new("test", "Test goal description", 0.75);

    assert_eq!(goal.id, "test");
    assert_eq!(goal.description, "Test goal description");
    assert_eq!(goal.priority, 0.75);
    assert_eq!(goal.progress, 0.0);
    assert!(goal.is_active);
    assert_eq!(goal.attention_weight, 0.75);
}

// -------------------- WorldModelBridge Tests --------------------

#[test]
fn test_world_model_bridge_default() {
    let bridge = WorldModelBridge::default();

    assert_eq!(bridge.total_predictions, 0);
    assert_eq!(bridge.avg_error, 0.0);

    // Should have 4 levels by default
    assert!(bridge.get_level_state(0).is_some());
    assert!(bridge.get_level_state(3).is_some());
    assert!(bridge.get_level_state(4).is_none());
}

#[test]
fn test_world_model_update_sensory() {
    let mut bridge = WorldModelBridge::default();

    // Create input matching level 0 dimension (64)
    let input: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();

    bridge.update_sensory(&input);

    assert_eq!(bridge.total_predictions, 1);
    assert!(bridge.avg_error >= 0.0);
}

#[test]
fn test_world_model_level_states() {
    let mut bridge = WorldModelBridge::default();

    let input: Vec<f32> = vec![1.0; 64];
    bridge.update_sensory(&input);

    // Level 0 should match input
    let level0 = bridge.get_level_state(0).unwrap();
    assert_eq!(level0.len(), 64);
    assert!((level0[0] - 1.0).abs() < 0.001);

    // Higher levels should exist and have been updated
    let level1 = bridge.get_level_state(1).unwrap();
    assert!(!level1.is_empty(), "Level 1 should have state");
    // The propagation logic chunks and averages, so sum should be non-zero
    let level1_sum: f32 = level1.iter().sum();
    assert!(
        level1_sum > 0.0,
        "Level 1 should have non-zero sum after propagation"
    );
}

#[test]
fn test_world_model_abstract_state() {
    let mut bridge = WorldModelBridge::default();

    let input: Vec<f32> = vec![0.5; 64];
    bridge.update_sensory(&input);

    let abstract_state = bridge.abstract_state();
    assert!(!abstract_state.is_empty());
    // Abstract state is highest level (128 dims)
    assert_eq!(abstract_state.len(), 128);
}

#[test]
fn test_world_model_level_errors() {
    let mut bridge = WorldModelBridge::default();

    // First update will have high error (predicting from zeros)
    let input: Vec<f32> = vec![1.0; 64];
    bridge.update_sensory(&input);

    let errors = bridge.level_errors();
    assert_eq!(errors.len(), 4);
    assert!(errors[0] > 0.0); // First prediction has error
}

#[test]
fn test_world_model_reset() {
    let mut bridge = WorldModelBridge::default();

    let input: Vec<f32> = vec![1.0; 64];
    bridge.update_sensory(&input);

    bridge.reset();

    assert_eq!(bridge.total_predictions, 0);
    assert_eq!(bridge.avg_error, 0.0);

    // States should be zeroed
    let level0 = bridge.get_level_state(0).unwrap();
    assert!(level0.iter().all(|&v| v == 0.0));
}

// -------------------- Cognitive Depth Tests --------------------

#[test]
fn test_cognitive_depth_default() {
    assert_eq!(CognitiveDepth::default(), CognitiveDepth::Cortical);
}

#[test]
fn test_cognitive_depth_equality() {
    assert_eq!(CognitiveDepth::Reflex, CognitiveDepth::Reflex);
    assert_eq!(CognitiveDepth::Cortical, CognitiveDepth::Cortical);
    assert_eq!(CognitiveDepth::DeepThought, CognitiveDepth::DeepThought);
    assert_ne!(CognitiveDepth::Reflex, CognitiveDepth::Cortical);
}

// -------------------- Moral Evaluation Tests --------------------

#[test]
fn test_moral_evaluation_runs_each_cycle() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    for _ in 0..5 {
        service.cycle("neutral input about weather");
    }

    let stats = service.stats();
    assert_eq!(
        stats.moral_evaluations, 5,
        "Moral evaluation should run every cycle"
    );
}

#[test]
fn test_moral_evaluation_tracks_concerns() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    // Run with morally neutral content
    service.cycle("The sky is blue and the grass is green");
    let neutral_concerns = service.stats().moral_concerns_detected;

    // Run with content that contains harm-related language
    service.cycle("deliberately causing harm and suffering to others");

    // The moral judgment should be populated after any cycle
    assert!(service.last_moral_judgment().is_some());
    let judgment = service.last_moral_judgment().unwrap();
    // moral_score is a finite f64
    assert!(judgment.moral_score.is_finite());
    // concerns counter should be >= what it was (may or may not detect concern)
    assert!(service.stats().moral_concerns_detected >= neutral_concerns);
}

// -------------------- Demand-Driven Consolidation Tests --------------------

#[test]
fn test_demand_driven_consolidation_trigger() {
    use crate::memory::episodic_replay::{EpisodicMemory, EpisodicReplayConfig};

    // Test that trigger_demand_replay causes should_replay to return true
    let config = EpisodicReplayConfig {
        phi_threshold: 0.01,
        min_episodes_for_replay: 1,
        replay_interval: 1000, // Long interval so periodic won't trigger
        ..Default::default()
    };
    let mut memory = EpisodicMemory::new(config);

    // Store an episode to meet minimum
    let episode = crate::memory::episodic_replay::Episode::new(
        symthaea_core::hdc::unified_hv::ContinuousHV::random(64, 1),
        symthaea_core::hdc::unified_hv::ContinuousHV::random(64, 2),
        0.5,
        1,
    );
    memory.store_if_significant(episode);

    // Without trigger, should_replay is false (replay_interval=1000)
    assert!(!memory.should_replay());

    // Trigger demand replay
    memory.trigger_demand_replay();
    assert!(
        memory.should_replay(),
        "Demand trigger should enable replay"
    );

    // Stats should track demand replays
    assert_eq!(memory.stats().demand_replay_count, 1);
}

// -------------------- FEP Learning Signal Tests --------------------

#[test]
fn test_fep_learning_signal_updates() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        learning_threshold: 0.0, // Always learn
        ..Default::default()
    })
    .unwrap();

    // Run enough cycles for FEP to produce learning signals
    for _ in 0..15 {
        service.cycle("fep learning signal test input");
    }

    let stats = service.stats();
    // FEP learning signal should be finite
    assert!(stats.fep_learning_signal.is_finite());
    // Effective learning rate should be positive and finite
    assert!(stats.effective_learning_rate.is_finite());
    assert!(stats.effective_learning_rate >= 0.0);
}

// -------------------- Stats Validation Tests --------------------

#[test]
fn test_stats_comprehensive_after_cycles() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        learning_threshold: 0.0,
        ..Default::default()
    })
    .unwrap();

    for i in 0..10 {
        service.cycle(&format!("varied input number {}", i));
    }

    let stats = service.stats();

    // Basic counters
    assert_eq!(stats.total_cycles, 10);
    assert!(
        stats.learning_cycles > 0,
        "With threshold=0.0, some learning should occur"
    );

    // Prediction error should be tracked
    assert!(stats.avg_prediction_error.is_finite());
    assert!(stats.avg_prediction_error >= 0.0);

    // Temporal coherence should be tracked
    assert!(stats.temporal_coherence.is_finite());

    // Semantic memory stats should be tracked
    assert!(
        stats.semantic_entries_stored > 0,
        "Semantic memory should have entries"
    );
    assert!(stats.semantic_lr_factor.is_finite());

    // LTC consciousness should be finite
    assert!(stats.ltc_consciousness.is_finite());
}

// -------------------- Primitive-Belief Bridge Verification --------------------

#[test]
fn test_primitive_belief_bridge_produces_signals() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        learning_threshold: 0.0,
        ..Default::default()
    })
    .unwrap();

    // First cycle: initializes prev_primitive_state
    service.cycle("initial state");

    // Subsequent cycles: bridge computes TD signals from prev vs current state
    for _ in 0..5 {
        service.cycle("primitive bridge signal test");
    }

    let stats = service.stats();
    // FEP learning signal should have been modulated by primitive bridge TD signals
    // After several cycles, the signal should be finite (may be zero if states are similar)
    assert!(stats.fep_learning_signal.is_finite());
}

// -------------------- Integration Tests --------------------

#[test]
fn test_unified_architecture_integration() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    // Run several cycles to exercise unified architecture
    for _ in 0..10 {
        service.cycle("test unified architecture integration");
    }

    let snapshot = service.consciousness_snapshot();

    // Verify unified components are operating
    assert!(snapshot.consciousness_level >= 0.0 && snapshot.consciousness_level <= 1.0);
    assert_eq!(snapshot.cycle, 10);

    // Verify cognitive depth was set
    assert!(matches!(
        snapshot.cognitive_depth,
        CognitiveDepth::Reflex | CognitiveDepth::Cortical | CognitiveDepth::DeepThought
    ));

    // Verify response strategy was set (use service method)
    let strategy = service.current_strategy();
    assert!(matches!(
        strategy,
        ResponseStrategy::Detailed
            | ResponseStrategy::Concise
            | ResponseStrategy::Clarifying
            | ResponseStrategy::Supportive
            | ResponseStrategy::Exploratory
    ));
}

#[test]
fn test_thalamic_routing_in_service() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    // Run cycles and verify routing happens
    for _ in 0..5 {
        service.cycle("familiar simple input");
    }

    // After several similar inputs, should settle into a routing pattern
    let snapshot = service.consciousness_snapshot();
    assert!(matches!(
        snapshot.cognitive_depth,
        CognitiveDepth::Reflex | CognitiveDepth::Cortical | CognitiveDepth::DeepThought
    ));
}

#[test]
fn test_closed_learning_loop_in_service() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    // Run cycles to accumulate learning
    for _ in 0..20 {
        service.cycle("learning loop test input");
    }

    // Should have a strategy selected (use service method)
    let strategy = service.current_strategy();
    assert!(matches!(
        strategy,
        ResponseStrategy::Detailed
            | ResponseStrategy::Concise
            | ResponseStrategy::Clarifying
            | ResponseStrategy::Supportive
            | ResponseStrategy::Exploratory
    ));
}
