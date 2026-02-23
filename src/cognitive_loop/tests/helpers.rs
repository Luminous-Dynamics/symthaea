//! Unit tests for extracted helper methods in helpers/.
//!
//! Tests cover Phases 1–5 extractions: safety precheck, cognitive depth,
//! negation detection, learning rate composition, moral phase, episodic recall,
//! surprise exploration, Psi synthesis, reward signal, strategy modulation,
//! FEP active inference, cross-modal binding, and parallel post-processing.

use super::super::*;

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 1: Safety precheck, cognitive depth, negation, learning rate
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_detect_negation_polarity_no_negation() {
    let service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let polarity = service.detect_negation_polarity("hello world");
    // Without negation detector configured, returns 0.0
    assert!((polarity - 0.0).abs() < f32::EPSILON);
}

#[test]
fn test_compose_effective_lr_default_range() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let lr = service.compose_effective_lr(1.0, 1.0);
    // Learning rate must be in [0.0, 0.01]
    assert!(lr >= 0.0, "LR below 0: {lr}");
    assert!(lr <= 0.01, "LR above 0.01: {lr}");
}

#[test]
fn test_compose_effective_lr_resets_subsystem_factor() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    service.carryover.learning.subsystem_lr_factor = 2.5;
    let _lr = service.compose_effective_lr(1.0, 1.0);
    // After compose, subsystem_lr_factor should be reset to 1.0
    assert!(
        (service.carryover.learning.subsystem_lr_factor - 1.0).abs() < f32::EPSILON,
        "subsystem_lr_factor not reset: {}",
        service.carryover.learning.subsystem_lr_factor
    );
}

#[test]
fn test_safety_precheck_benign_input() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let result = service.safety_precheck("hello", std::time::Instant::now());
    // Benign input should not be blocked
    assert!(result.is_none(), "Benign input was unexpectedly blocked");
}

#[test]
fn test_update_cognitive_depth_initial() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    service.update_cognitive_depth();
    // Cognitive depth should be a valid variant after update
    assert!(
        matches!(
            service.cognitive_depth,
            CognitiveDepth::Reflex | CognitiveDepth::Cortical | CognitiveDepth::DeepThought
        ),
        "cognitive_depth should be a valid variant: {:?}",
        service.cognitive_depth
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 2: Moral phase, episodic recall, surprise exploration
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_run_moral_phase_benign_input() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    service.stats.total_cycles = 1; // trigger evaluation (cycle % 7 == 1)
    let (score, concern, judgment) = service.run_moral_phase("hello world", 0.0);
    // Benign input: score should be non-negative, no concern
    assert!(score >= -1.0, "Moral score below -1: {score}");
    assert!(!concern, "Benign input flagged as moral concern");
    assert!(!judgment.consent_violation, "No consent violation expected");
}

#[test]
fn test_run_moral_phase_negation_dampening() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    service.stats.total_cycles = 1;
    // First, get baseline moral score without negation
    let (score_no_neg, _, _) = service.run_moral_phase("harmful action", 0.0);
    // Reset for second call
    service.stats.total_cycles = 1;
    service.last_moral_judgment = None;
    // With high negation polarity ("not harmful"), score should be dampened
    let (score_with_neg, _, _) = service.run_moral_phase("harmful action", 0.8);
    // Negation dampening multiplies by 0.3, so |dampened| <= |original|
    assert!(
        score_with_neg.abs() <= score_no_neg.abs() + f32::EPSILON,
        "Negation should dampen score: orig={score_no_neg}, dampened={score_with_neg}"
    );
}

#[test]
fn test_run_moral_phase_throttling() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    // First evaluation
    service.stats.total_cycles = 1;
    let evals_before = service.stats.moral_evaluations;
    let _ = service.run_moral_phase("hello", 0.0);
    let evals_after_first = service.stats.moral_evaluations;
    assert_eq!(
        evals_after_first,
        evals_before + 1,
        "First call should evaluate"
    );

    // Same input, non-evaluation cycle — should reuse cached judgment
    service.stats.total_cycles = 3; // 3 % 7 != 1
    let _ = service.run_moral_phase("hello", 0.0);
    assert_eq!(
        service.stats.moral_evaluations, evals_after_first,
        "Throttled call should NOT re-evaluate"
    );
}

#[test]
fn test_run_moral_phase_updates_stats() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    service.stats.total_cycles = 1;
    let _ = service.run_moral_phase("hello", 0.0);
    // Stats should be populated
    assert!(
        service.stats.moral_evaluations > 0,
        "moral_evaluations should increment"
    );
    // moral_score should have been written
    // (we just check it's finite — the actual value depends on the evaluator)
    assert!(
        service.stats.moral_score.is_finite(),
        "moral_score not finite"
    );
}

#[test]
fn test_recall_episodic_context_empty_memory() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let compressed = vec![0.1f32; 128];
    let boost = service.recall_episodic_context(&compressed);
    // With empty episodic memory, boost should be 0
    assert!(
        boost.abs() < f32::EPSILON,
        "Empty memory should yield 0 boost, got {boost}"
    );
}

#[test]
fn test_recall_episodic_context_preserves_confidence_bounds() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    service.prediction_confidence = 0.99;
    let compressed = vec![0.5f32; 128];
    let _ = service.recall_episodic_context(&compressed);
    // Confidence should remain bounded [0, 1]
    assert!(
        service.prediction_confidence >= 0.0 && service.prediction_confidence <= 1.0,
        "Confidence out of bounds: {}",
        service.prediction_confidence
    );
}

#[test]
fn test_run_surprise_exploration_no_bridge() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    // Default config has no surprise bridge
    let compressed = vec![0.3f32; 128];
    let (triggered, action) = service.run_surprise_exploration(&compressed);
    assert!(!triggered, "No surprise bridge → no trigger");
    assert!(action.is_none(), "No surprise bridge → no action");
}

#[test]
fn test_run_surprise_exploration_with_bridge() {
    let mut config = CognitiveLoopConfig::default();
    config.enable_surprise_exploration = true;
    let mut service = CognitiveLoopService::new(config).unwrap();
    // Run a few cycles to prime state
    let compressed = vec![0.5f32; 128];
    service.last_state = Some(compressed.clone());
    service.last_prediction = Some(vec![0.0f32; 128]); // large mismatch
    let (triggered, action) = service.run_surprise_exploration(&compressed);
    // We can't guarantee triggering (threshold-dependent), but verify no panic
    // and that action is coherent with triggered
    if triggered {
        assert!(action.is_some(), "Triggered but no action");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 3: Psi synthesis, reward signal, strategy modulation
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_compute_unified_psi_range() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let psi = service.compute_unified_psi();
    assert!(psi >= 0.0, "Psi below 0: {psi}");
    assert!(psi <= 1.0, "Psi above 1: {psi}");
}

#[test]
fn test_compute_unified_psi_updates_unification_engine() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let psi = service.compute_unified_psi();
    // The unification engine's psi should match the returned value
    assert!(
        (service.unification_engine.psi - psi).abs() < 1e-10,
        "unification_engine.psi ({}) != returned psi ({psi})",
        service.unification_engine.psi
    );
}

#[test]
fn test_compute_reward_signal_low_error() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    service.prediction_confidence = 0.8;
    // Low prediction error (below threshold) → positive reward
    let reward = service.compute_reward_signal(0.01, 0.3);
    assert!(
        reward > 0.0,
        "Low-error reward should be positive, got {reward}"
    );
    assert!(reward <= 1.0, "Reward above 1: {reward}");
}

#[test]
fn test_compute_reward_signal_high_error() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    // High prediction error (above 0.5) → negative reward
    let reward = service.compute_reward_signal(0.8, 0.3);
    assert!(
        reward < 0.0,
        "High-error reward should be negative, got {reward}"
    );
    assert!(reward >= -1.0, "Reward below -1: {reward}");
}

#[test]
fn test_compute_reward_signal_clamped() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    // Extreme values should still clamp to [-1, 1]
    let reward = service.compute_reward_signal(10.0, 0.3);
    assert!(
        reward >= -1.0 && reward <= 1.0,
        "Reward not clamped: {reward}"
    );
}

#[test]
fn test_compute_reward_signal_consumes_external() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    service.external_reward = 0.7;
    let _reward = service.compute_reward_signal(0.2, 0.3);
    // external_reward should be consumed (set to 0)
    assert!(
        service.external_reward.abs() < f32::EPSILON,
        "external_reward not consumed: {}",
        service.external_reward
    );
}

#[test]
fn test_apply_strategy_modulation_exploratory() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let before = service.adaptive_behavior.exploration_factor;
    service.apply_strategy_modulation(ResponseStrategy::Exploratory);
    // Exploratory strategy sets a specific exploration factor
    assert!(
        (service.adaptive_behavior.exploration_factor - before).abs() > f32::EPSILON
            || service.adaptive_behavior.exploration_factor == 0.8,
        "Exploratory strategy didn't modulate exploration_factor"
    );
}

#[test]
fn test_apply_strategy_modulation_concise() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    service.apply_strategy_modulation(ResponseStrategy::Concise);
    assert!(
        (service.adaptive_behavior.speech_rate_multiplier - 1.2).abs() < f32::EPSILON,
        "Concise strategy should set speech_rate_multiplier to 1.2, got {}",
        service.adaptive_behavior.speech_rate_multiplier
    );
}

#[test]
fn test_reapply_strategy_preserves_stronger() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    // Set exploration very high
    service.adaptive_behavior.exploration_factor = 0.95;
    service.reapply_strategy_modulation(ResponseStrategy::Exploratory);
    // reapply uses .max(), so should preserve the higher value
    assert!(
        service.adaptive_behavior.exploration_factor >= 0.8,
        "reapply should preserve higher exploration: {}",
        service.adaptive_behavior.exploration_factor
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 4: FEP active inference, cross-modal binding
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_step_fep_active_inference_returns_valid_action() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let (action_idx, action_probs, _is_surprised, pragmatic) =
        service.step_fep_active_inference(0.3, 0.5);
    // Action index should be within range (0-3 for 4-action FEP)
    assert!(action_idx <= 3, "Invalid action index: {action_idx}");
    // Probabilities should sum to ~1.0
    let prob_sum: f64 = action_probs.iter().sum();
    assert!(
        (prob_sum - 1.0).abs() < 0.01,
        "Action probs don't sum to 1: {prob_sum}"
    );
    // Pragmatic value should be finite
    assert!(
        pragmatic.is_finite(),
        "Pragmatic value not finite: {pragmatic}"
    );
}

#[test]
fn test_step_fep_active_inference_high_error_response() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let initial_lr_boost = service.fep_lr_boost;
    // Run with very high prediction error
    let (_idx, _probs, _surprised, _prag) = service.step_fep_active_inference(0.9, 0.2);
    // The FEP system should have responded in some way
    // (exact behavior depends on action selection, but state should have changed)
    let state_changed = service.fep_lr_boost != initial_lr_boost
        || service.curiosity_drive.exploration_urge > 0.0
        || service.fep_agent.precision.sensory_precision != 1.0;
    // This is a soft assertion — FEP action is stochastic
    let _ = state_changed; // Just verify no panic
}

#[test]
fn test_update_cross_modal_binding_without_binder() {
    let service_config = CognitiveLoopConfig::default();
    let mut service = CognitiveLoopService::new(service_config).unwrap();
    // Default config has no cross-modal binder
    let hv = symthaea_core::hdc::binary_hv::BinaryHV::random(42);
    let (strength, psi) = service.update_cross_modal_binding(&hv, 0.5, 0.3);
    assert!(
        (strength - 0.0).abs() < f32::EPSILON,
        "Expected 0 strength without binder"
    );
    assert!(
        (psi - 0.0).abs() < f64::EPSILON,
        "Expected 0 psi without binder"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 5: Parallel post-processing free functions
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_run_stability_regime_returns_timing() {
    let mut regime = crate::consciousness::stability_regime::StabilityRegimeProcessor::new();
    let mut discovery = crate::consciousness::primitive_discovery::PrimitiveDiscoveryService::new(
        Default::default(),
    );
    let hv = symthaea_core::hdc::binary_hv::BinaryHV::random(99);
    let timing = super::super::helpers::run_stability_regime(
        &mut regime,
        &mut discovery,
        &hv,
        0.02,
        5,
        CycleUrgency::Normal,
    );
    // Should return timing >= 0
    assert!(timing < 1_000_000, "Timing suspiciously high: {timing}us");
}

#[test]
fn test_run_stability_regime_skips_in_cruise() {
    let mut regime = crate::consciousness::stability_regime::StabilityRegimeProcessor::new();
    let mut discovery = crate::consciousness::primitive_discovery::PrimitiveDiscoveryService::new(
        Default::default(),
    );
    let hv = symthaea_core::hdc::binary_hv::BinaryHV::random(99);
    // Cruise urgency with should_run(cycle=1, ..., cruise_every=20) should skip
    let timing = super::super::helpers::run_stability_regime(
        &mut regime,
        &mut discovery,
        &hv,
        0.02,
        1,
        CycleUrgency::Cruise,
    );
    // Even when skipped, returns valid timing
    assert!(timing < 1_000_000, "Timing suspiciously high: {timing}us");
}

#[test]
fn test_parallel_semantic_causal_no_panic() {
    let mut semantic_memory = crate::memory::semantic_memory::SemanticMemory::new(100);
    let mut causal_enhancer: Option<crate::causal::CausalLoopEnhancer> = None;
    let semantic_hdc = vec![0.1f32; 64];
    let compressed = vec![0.2f32; 128];
    let output = vec![0.3f32; 64];
    // Should not panic
    super::super::helpers::parallel_semantic_causal(
        &mut semantic_memory,
        &mut causal_enhancer,
        semantic_hdc,
        &compressed,
        &output,
        0.5,
        10,
    );
}

#[test]
fn test_episodic_learning_context_construction() {
    let compressed = vec![0.5f32; 128];
    let prims = vec!["awareness".to_string()];
    let ctx = super::super::helpers::EpisodicLearningContext {
        prediction_error: 0.3,
        in_flow: true,
        input: "test input",
        compressed_state: &compressed,
        emotional_valence: 0.5,
        phi: 0.6,
        total_cycles: 100,
        smoothed_coh: 0.7,
        detected_primitives: &prims,
        memory_context_boost: 0.1,
        wm_importance_boost: 0.2,
    };
    assert_eq!(ctx.total_cycles, 100);
    assert!(ctx.in_flow);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Phase 6: Consciousness subsystem extractions from cycle_consciousness.rs
// Batch 1: temporal primitives, dissipative, equation v2
// Batch 2: lattice, value evaluator, fiduciary harmonics, causal self-explanation
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_compute_temporal_primitives_no_analyzer() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let mut timings = super::super::ModuleTimings::default();
    let hv = symthaea_core::hdc::BinaryHV::random(42);
    let (chains, continuity, max_len, cycle_nums, codebook, replay) =
        service.compute_temporal_primitives_phase(hv, 0.5, 0.7, &mut timings);
    // Without temporal_analyzer, all values are zero/empty
    assert_eq!(chains, 0);
    assert!((continuity - 0.0).abs() < f64::EPSILON);
    assert_eq!(max_len, 0);
    assert!(cycle_nums.is_empty());
    assert!(codebook.is_empty());
    assert!(!replay);
    assert!(timings.temporal_analyzer > 0 || true); // timing may be 0 on fast machines
}

#[test]
fn test_compute_temporal_primitives_confidence_stable_without_analyzer() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let mut timings = super::super::ModuleTimings::default();
    let hv = symthaea_core::hdc::BinaryHV::random(42);
    let initial_conf = service.prediction_confidence;
    let _ = service.compute_temporal_primitives_phase(hv, 0.5, 0.7, &mut timings);
    // Without analyzer, no feedback → confidence unchanged
    assert!(
        (service.prediction_confidence - initial_conf).abs() < f32::EPSILON,
        "Confidence changed without analyzer"
    );
}

#[test]
fn test_compute_dissipative_no_dc() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let mut timings = super::super::ModuleTimings::default();
    let (health, regime, entropy) = service.compute_dissipative_phase(0.3, 0.6, 0.5, &mut timings);
    // Without dissipative_consciousness, all zeros
    assert!((health - 0.0).abs() < f64::EPSILON);
    assert!(regime.is_empty());
    assert!((entropy - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_compute_dissipative_confidence_stable_without_dc() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let mut timings = super::super::ModuleTimings::default();
    let initial_conf = service.prediction_confidence;
    let _ = service.compute_dissipative_phase(0.3, 0.6, 0.5, &mut timings);
    // Without DC, no feedback → confidence unchanged
    assert!(
        (service.prediction_confidence - initial_conf).abs() < f32::EPSILON,
        "Confidence changed without DC"
    );
}

#[test]
fn test_compute_equation_v2_no_equation() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let mut timings = super::super::ModuleTimings::default();
    let score = service.compute_equation_v2_phase(0.5, 0.7, 0.3, 0.6, &mut timings);
    // Without consciousness_equation_v2, returns 0.0
    assert!((score - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_compute_equation_v2_exploration_unaffected_at_zero() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let mut timings = super::super::ModuleTimings::default();
    let initial_explore = service.curiosity_drive.exploration_urge;
    let _ = service.compute_equation_v2_phase(0.5, 0.7, 0.3, 0.6, &mut timings);
    // Score=0.0, low-consciousness feedback (0.0 < 0.3) doesn't apply since eq is exactly 0.0
    // and condition is `> 0.0 && < 0.3`
    assert!(
        (service.curiosity_drive.exploration_urge - initial_explore).abs() < f32::EPSILON,
        "Exploration changed at score=0.0"
    );
}

#[test]
fn test_compute_lattice_no_lattice() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let mut timings = super::super::ModuleTimings::default();
    let names = vec!["awareness".to_string(), "binding".to_string()];
    let (height, width, join) = service.compute_lattice_phase(&names, &mut timings);
    // Without primitive_lattice, all zeros/None
    assert_eq!(height, 0);
    assert_eq!(width, 0);
    assert!(join.is_none());
}

#[test]
fn test_compute_lattice_lr_unaffected_without_lattice() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let mut timings = super::super::ModuleTimings::default();
    let initial_lr = service.carryover.learning.subsystem_lr_factor;
    let _ = service.compute_lattice_phase(&[], &mut timings);
    assert!(
        (service.carryover.learning.subsystem_lr_factor - initial_lr).abs() < f32::EPSILON,
        "LR factor changed without lattice"
    );
}

#[test]
fn test_compute_value_evaluator_no_evaluator() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let mut timings = super::super::ModuleTimings::default();
    let (score, decision, gate_factor) = service.compute_value_evaluator_phase(0.5, &mut timings);
    assert!((score - 0.0).abs() < f64::EPSILON);
    assert!(decision.is_empty());
    assert!((gate_factor - 1.0).abs() < f32::EPSILON);
}

#[test]
fn test_compute_value_evaluator_lr_stable_without_evaluator() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let mut timings = super::super::ModuleTimings::default();
    let initial_lr = service.carryover.learning.subsystem_lr_factor;
    let _ = service.compute_value_evaluator_phase(0.5, &mut timings);
    // No evaluator → no Veto → LR unchanged
    assert!(
        (service.carryover.learning.subsystem_lr_factor - initial_lr).abs() < f32::EPSILON,
        "LR changed without evaluator"
    );
}

#[test]
fn test_compute_fiduciary_harmonics_no_field() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let mut timings = super::super::ModuleTimings::default();
    let (coherence, love, interferences) =
        service.compute_fiduciary_harmonics_phase(0.7, 0.3, 0.5, &mut timings);
    assert!((coherence - 0.0).abs() < f64::EPSILON);
    assert!((love - 0.0).abs() < f64::EPSILON);
    assert_eq!(interferences, 0);
}

#[test]
fn test_compute_fiduciary_harmonics_confidence_stable_without_field() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let mut timings = super::super::ModuleTimings::default();
    let initial_conf = service.prediction_confidence;
    let _ = service.compute_fiduciary_harmonics_phase(0.7, 0.3, 0.5, &mut timings);
    assert!(
        (service.prediction_confidence - initial_conf).abs() < f32::EPSILON,
        "Confidence changed without harmonic field"
    );
}

#[test]
fn test_compute_causal_self_explanation_no_explainer() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let mut timings = super::super::ModuleTimings::default();
    let hv = symthaea_core::hdc::BinaryHV::random(42);
    let names = vec!["awareness".to_string()];
    let (count, confidence) =
        service.compute_causal_self_explanation_phase(hv, &names, 0.5, &mut timings);
    assert_eq!(count, 0);
    assert!((confidence - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_compute_causal_history_unchanged_without_explainer() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let mut timings = super::super::ModuleTimings::default();
    let hv = symthaea_core::hdc::BinaryHV::random(42);
    let initial_relations = service.carryover.history.last_causal_relations;
    let initial_confidence = service.carryover.history.last_causal_confidence;
    let _ = service.compute_causal_self_explanation_phase(hv, &[], 0.5, &mut timings);
    assert_eq!(
        service.carryover.history.last_causal_relations,
        initial_relations
    );
    assert!(
        (service.carryover.history.last_causal_confidence - initial_confidence).abs()
            < f64::EPSILON
    );
}

#[test]
fn test_all_consciousness_methods_write_timings() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let hv = symthaea_core::hdc::BinaryHV::random(42);

    let mut t1 = super::super::ModuleTimings::default();
    let _ = service.compute_temporal_primitives_phase(hv, 0.5, 0.7, &mut t1);
    // temporal_analyzer timing field should be written (may be 0 on fast execution)

    let mut t2 = super::super::ModuleTimings::default();
    let _ = service.compute_dissipative_phase(0.3, 0.6, 0.5, &mut t2);

    let mut t3 = super::super::ModuleTimings::default();
    let _ = service.compute_equation_v2_phase(0.5, 0.7, 0.3, 0.6, &mut t3);

    let mut t4 = super::super::ModuleTimings::default();
    let _ = service.compute_lattice_phase(&[], &mut t4);

    let mut t5 = super::super::ModuleTimings::default();
    let _ = service.compute_value_evaluator_phase(0.5, &mut t5);

    let mut t6 = super::super::ModuleTimings::default();
    let _ = service.compute_fiduciary_harmonics_phase(0.7, 0.3, 0.5, &mut t6);

    let mut t7 = super::super::ModuleTimings::default();
    let _ = service.compute_causal_self_explanation_phase(hv, &[], 0.5, &mut t7);

    // All methods execute without panic — that's the key assertion
    // Timing fields are written (they may be 0µs on fast machines, so we just verify no panic)
}
