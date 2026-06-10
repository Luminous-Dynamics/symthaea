// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use super::super::*;
use crate::dynamics::ConsciousnessPattern;

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
    assert!(stats.modulation_index.is_none());
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
    let strategy = loop_.select_strategy(0.8, None);
    // High Φ → integrative mode → favors Exploratory/Detailed
    assert!(
        !matches!(
            strategy,
            ResponseStrategy::Supportive | ResponseStrategy::Concise
        ) || loop_.last_result.is_some(),
        "High Φ should shift away from Supportive/Concise"
    );
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
