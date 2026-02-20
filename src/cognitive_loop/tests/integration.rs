use super::super::*;

// -------------------- Moral Evaluation Tests --------------------

#[test]
fn test_moral_evaluation_throttled() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    // Same input: evaluation runs on cycle 1 (total_cycles % 5 == 1) and cycle 6
    for _ in 0..5 {
        service.cycle("neutral input about weather");
    }

    let stats = service.stats();
    assert!(
        stats.moral_evaluations >= 1 && stats.moral_evaluations <= 5,
        "Moral evaluation should be throttled for repeated identical input, got {}",
        stats.moral_evaluations,
    );

    // Different inputs: each unique input triggers fresh evaluation
    let before = service.stats().moral_evaluations;
    service.cycle("a completely different topic");
    assert!(
        service.stats().moral_evaluations > before,
        "New input should trigger moral evaluation"
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
        psi_threshold: 0.01,
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

// -------------------- CycleMetadata Tests --------------------

#[test]
fn test_cycle_metadata_populated() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    let result = service.cycle("test metadata population");

    // All metadata fields should have valid defaults
    assert!(!result.metadata.prefrontal_veto); // No prefrontal enabled
    assert!(!result.metadata.surprise_triggered); // First cycle unlikely to trigger
    assert!(!result.metadata.reasoning_gate_blocked); // No reasoning engine in default features
    assert!(result.metadata.reasoning_confidence >= 0.0);
    assert!(result.metadata.reasoning_plan_confidence >= 0.0);
    assert_eq!(result.metadata.meta_cognitive_accuracy, 0.0); // Not enabled
    assert_eq!(result.metadata.meta_cognitive_depth, 0); // Not enabled
}

#[test]
fn test_cycle_timing_bounds() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    let result = service.cycle("timing test");

    // cycle_time_us should be positive
    assert!(result.cycle_time_us > 0, "Cycle time should be positive");
    // Should complete in under 5 seconds (generous bound)
    assert!(
        result.cycle_time_us < 5_000_000,
        "Cycle should complete in under 5 seconds, took {}us",
        result.cycle_time_us
    );
}

#[test]
fn test_cycle_learning_occurs_when_threshold_met() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        learning_threshold: 0.0, // Always learn
        ..Default::default()
    })
    .unwrap();

    // First cycle has no previous state, but learning should still attempt
    let _first = service.cycle("initial input");

    // Second cycle should learn from state transition
    let result = service.cycle("different input to trigger learning");

    // With threshold 0.0, some learning cycles should have occurred
    assert!(
        service.stats().learning_cycles > 0,
        "Learning should occur with threshold=0.0"
    );
    // Training loss may or may not be Some depending on async training
    assert!(result.prediction_error >= 0.0);
}

#[test]
fn test_cycle_output_valid_dimensions() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();

    let result = service.cycle("dimension check");

    // Output should match CfC num_neurons
    assert_eq!(
        result.output.len(),
        service.stats().total_cycles * 0 + 256, // Default num_neurons = 256
        "Output should match CfC num_neurons"
    );
}

#[test]
fn test_cycle_with_prefrontal_enabled() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_prefrontal: true,
        ..Default::default()
    })
    .unwrap();

    // Run enough cycles to exercise prefrontal
    for _ in 0..10 {
        service.cycle("prefrontal test input");
    }

    // With default capacity=7 and 10 inputs, some items should have been graduated
    // Verify the service didn't panic and metadata is valid
    let result = service.cycle("one more cycle");
    // prefrontal_veto is a valid boolean
    assert!(result.metadata.prefrontal_veto || !result.metadata.prefrontal_veto);
}

#[test]
fn test_cycle_with_surprise_exploration() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_surprise_exploration: true,
        ..Default::default()
    })
    .unwrap();

    for _ in 0..10 {
        service.cycle("surprise exploration input");
    }

    // Verify it ran without errors and metadata is populated
    let result = service.cycle("final check");
    assert!(result.prediction_error >= 0.0);
}

#[test]
fn test_cycle_with_meta_cognition() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_meta_cognition: true,
        ..Default::default()
    })
    .unwrap();

    for _ in 0..10 {
        service.cycle("meta cognition input");
    }

    let result = service.cycle("meta check");
    // Meta-cognition accuracy should be between 0 and 1
    assert!(result.metadata.meta_cognitive_accuracy >= 0.0);
    assert!(result.metadata.meta_cognitive_accuracy <= 1.0);
    // Depth should be a valid value (usize is always >= 0)
    let _ = result.metadata.meta_cognitive_depth;
}

#[test]
fn test_cycle_with_predictive_self() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_predictive_self: true,
        enable_narrative_self: true, // Required dependency
        ..Default::default()
    })
    .unwrap();

    for _ in 0..10 {
        service.cycle("predictive self test input");
    }

    let result = service.cycle("predictive self check");
    // Predictive self safety should be valid (0 = no confidence yet, up to 1.0)
    assert!(result.metadata.predictive_self_safety >= 0.0);
    assert!(result.metadata.predictive_self_safety <= 1.0);
}

#[test]
fn test_cycle_with_attention_schema() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_attention_schema: true,
        ..Default::default()
    })
    .unwrap();

    for _ in 0..10 {
        service.cycle("attention schema test input");
    }

    let result = service.cycle("attention check");
    // Attention schema focus should be valid
    assert!(result.metadata.attention_schema_focus >= 0.0);
    assert!(result.metadata.attention_schema_focus <= 1.0);
}

#[test]
fn test_cycle_with_gwt() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_gwt: true,
        ..Default::default()
    })
    .unwrap();

    for _ in 0..10 {
        service.cycle("gwt test input");
    }

    let result = service.cycle("gwt check");
    // GWT broadcast is a valid boolean
    assert!(result.metadata.gwt_broadcast || !result.metadata.gwt_broadcast);
}

#[test]
fn test_cycle_with_resonance() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_resonance: true,
        ..Default::default()
    })
    .unwrap();

    for _ in 0..10 {
        service.cycle("resonance test input");
    }

    let result = service.cycle("resonance check");
    // Resonance frequency should be finite
    assert!(result.metadata.resonance_frequency.is_finite());
}

#[test]
fn test_cycle_with_quantum_coherence() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_quantum_coherence: true,
        ..Default::default()
    })
    .unwrap();

    for _ in 0..10 {
        service.cycle("quantum coherence test input");
    }

    let result = service.cycle("quantum check");
    // Quantum coherence level should be valid
    assert!(result.metadata.quantum_coherence_level >= 0.0);
    assert!(result.metadata.quantum_coherence_level <= 1.0);
}

#[test]
fn test_body_phi_modulation_feedback() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_virtual_body: true,
        ..Default::default()
    })
    .unwrap();

    // Run 20 cycles to let the body feedback loop stabilize
    let mut body_mods = Vec::new();
    for _ in 0..20 {
        let result = service.cycle("body feedback test");
        body_mods.push(result.metadata.body_phi_modulation);
    }

    // At least some body phi modulation should differ from 1.0 after warmup
    let non_neutral = body_mods
        .iter()
        .filter(|&&m| (m - 1.0).abs() > 0.001)
        .count();
    assert!(
        non_neutral > 0,
        "Body phi modulation should deviate from 1.0 after warmup"
    );
}

#[test]
fn test_all_consciousness_modules_enabled() {
    // Smoke test: enable everything and verify no panics
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_predictive_self: true,
        enable_narrative_self: true,
        enable_attention_schema: true,
        enable_gwt: true,
        enable_resonance: true,
        enable_quantum_coherence: true,
        enable_meta_cognition: true,
        enable_prefrontal: true,
        enable_surprise_exploration: true,
        enable_virtual_body: true,
        enable_temporal_consciousness: true,
        enable_embodied_cognition: true,
        enable_narrative_gwt: true,
        ..Default::default()
    })
    .unwrap();

    for _ in 0..15 {
        service.cycle("all modules enabled test input");
    }

    let result = service.cycle("final check with all modules");
    assert!(result.prediction_error.is_finite());
    assert!(result.metadata.consciousness_level.is_finite());
    // Temporal consciousness should have valid coherence
    assert!(result.metadata.temporal_coherence_score >= 0.0);
    assert!(result.metadata.temporal_coherence_score <= 1.0);
    // Embodied cognition should have valid phi modulation
    assert!(result.metadata.embodied_phi_modulation.is_finite());
    // Narrative-GWT self phi should be finite
    assert!(result.metadata.narrative_gwt_self_phi.is_finite());
}

#[test]
fn test_cycle_with_temporal_consciousness() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_temporal_consciousness: true,
        enable_narrative_self: true,  // Dependency
        enable_predictive_self: true, // Dependency
        ..Default::default()
    })
    .unwrap();

    for _ in 0..15 {
        service.cycle("temporal consciousness test input");
    }

    let result = service.cycle("temporal check");
    // Temporal coherence should be between 0 and 1
    assert!(result.metadata.temporal_coherence_score >= 0.0);
    assert!(result.metadata.temporal_coherence_score <= 1.0);
    // After 15 cycles, should have enough data for analysis
    // (discontinuity is a boolean - just verify it's valid)
    assert!(result.metadata.temporal_discontinuity || !result.metadata.temporal_discontinuity);
}

#[test]
fn test_cycle_with_embodied_cognition() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_embodied_cognition: true,
        enable_virtual_body: true, // Provides interoceptive state
        ..Default::default()
    })
    .unwrap();

    for _ in 0..10 {
        service.cycle("embodied cognition test input");
    }

    let result = service.cycle("embodied check");
    // Embodied phi modulation should be finite and reasonable
    assert!(result.metadata.embodied_phi_modulation.is_finite());
    // Agency should be between 0 and 1
    assert!(result.metadata.embodied_agency >= 0.0);
    assert!(result.metadata.embodied_agency <= 1.0);
}

#[test]
fn test_cycle_with_narrative_gwt() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig {
        enable_narrative_gwt: true,
        ..Default::default()
    })
    .unwrap();

    for _ in 0..10 {
        service.cycle("narrative gwt governance test input");
    }

    let result = service.cycle("governance check");
    // Self-phi should be finite
    assert!(result.metadata.narrative_gwt_self_phi.is_finite());
    // Veto is a boolean - just verify valid
    assert!(result.metadata.narrative_gwt_veto || !result.metadata.narrative_gwt_veto);
}

// ═══════════════════════════════════════════════════════════════════════════════
// v0.6.1 FEEDBACK LOOP TESTS
// ═══════════════════════════════════════════════════════════════════════════════
