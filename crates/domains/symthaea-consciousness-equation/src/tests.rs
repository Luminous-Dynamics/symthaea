// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use crate::*;

#[test]
fn test_default_config() {
    let config = MasterEquationConfig::default();
    assert!(config.softmin_tau > 0.0);
    assert!(config.enable_embodiment_factor);
    assert!(config.enable_narrative_factor);
    assert!(config.enable_social_factor);
}

#[test]
fn test_component_weights_normalize() {
    let mut weights = ComponentWeights::default();
    weights.normalize();
    let total = weights.total();
    assert!((total - 1.0).abs() < 0.001);
}

#[test]
fn test_embodiment_factor() {
    let mut ef = EmbodimentFactor::new();

    // Record some predictions
    ef.record_prediction(0.5, 0.5); // Perfect prediction
    ef.record_prediction(0.5, 0.6); // Small error
    ef.record_prediction(0.5, 0.4); // Small error

    ef.update_interoceptive(0.7, 0.8); // Good coherence

    let m = ef.compute();
    assert!(m > 0.0 && m <= 1.0);
    assert!(ef.sensorimotor_accuracy() > 0.5);
}

#[test]
fn test_narrative_coherence() {
    let mut nc = NarrativeCoherence::new();

    // Add some episodes
    nc.add_episode("Started the day".to_string(), 0.5);
    nc.add_episode("Had a good conversation".to_string(), 0.7);
    nc.add_episode("Learned something new".to_string(), 0.6);

    // Add future scenario
    nc.add_future_scenario("Complete the project".to_string(), 5, 0.8, 0.9);

    let n = nc.compute();
    assert!(n > 0.0 && n <= 1.0);
    assert_eq!(nc.episode_count(), 3);
    assert_eq!(nc.scenario_count(), 1);
}

#[test]
fn test_social_embedding() {
    let mut se = SocialEmbedding::new();

    // Update self model
    se.update_self_model(
        vec!["help users".to_string(), "learn".to_string()],
        vec!["AI is helpful".to_string()],
        0.5,
    );

    // Add agent model
    se.update_agent_model(
        "user1",
        vec!["user belief".to_string()],
        vec!["get help".to_string()],
        0.6,
        0.8,
    );

    // Record ToM prediction
    se.record_tom_prediction("user1", 0.7);
    se.provide_tom_feedback("user1", 0.6);

    let soc = se.compute();
    assert!(soc > 0.0 && soc <= 1.0);
    assert_eq!(se.agent_count(), 1);
}

#[test]
fn test_master_equation_basic() {
    let mut equation = MasterConsciousnessEquation::default();

    let inputs = ConsciousnessInputs {
        phi: 0.6,
        broadcast: 0.7,
        working_memory: 0.5,
        attention: 0.8,
        recurrence: 0.6,
        embodiment: 0.5,
        knowledge: 0.7,
        synchrony: 0.8,
    };

    let result = equation.compute(&inputs);

    assert!(result.consciousness_level >= 0.0);
    assert!(result.consciousness_level <= 1.0);
    assert!(!result.bottleneck_name.is_empty());
}

#[test]
fn test_bottleneck_detection() {
    let mut equation = MasterConsciousnessEquation::default();

    // Create inputs with one clear bottleneck (low attention)
    let inputs = ConsciousnessInputs {
        phi: 0.8,
        broadcast: 0.8,
        working_memory: 0.8,
        attention: 0.1, // Bottleneck
        recurrence: 0.8,
        embodiment: 0.8,
        knowledge: 0.8,
        synchrony: 0.8,
    };

    let result = equation.compute(&inputs);

    // Attention should be identified as bottleneck
    assert!(result.bottleneck_name.contains("Attention"));

    // Consciousness should be limited by low attention
    assert!(result.consciousness_level < 0.5);
}

#[test]
fn test_temporal_stability() {
    let mut equation = MasterConsciousnessEquation::default();

    let inputs = ConsciousnessInputs {
        phi: 0.6,
        broadcast: 0.6,
        working_memory: 0.6,
        attention: 0.6,
        recurrence: 0.6,
        embodiment: 0.6,
        knowledge: 0.6,
        synchrony: 0.8,
    };

    // Compute multiple times with same inputs (should be stable)
    for _ in 0..10 {
        equation.compute(&inputs);
    }

    let result = equation.compute(&inputs);

    // High stability when inputs are consistent
    assert!(result.temporal_stability > 0.7);
}

#[test]
fn test_gating_factors() {
    let mut equation = MasterConsciousnessEquation::default();

    // Gate off attention
    equation.set_gating("attention", 0.0);

    let inputs = ConsciousnessInputs {
        phi: 0.8,
        broadcast: 0.8,
        working_memory: 0.8,
        attention: 0.8,
        recurrence: 0.8,
        embodiment: 0.8,
        knowledge: 0.8,
        synchrony: 0.8,
    };

    let result = equation.compute(&inputs);

    // Should still compute (attention gated out of weighted sum)
    assert!(result.consciousness_level > 0.0);
}

#[test]
fn test_new_factors_integration() {
    let mut equation = MasterConsciousnessEquation::default();

    // Build up the new factors
    equation.embodiment_factor.record_prediction(0.5, 0.5);
    equation.embodiment_factor.update_interoceptive(0.6, 0.6);

    equation
        .narrative_coherence
        .add_episode("test".to_string(), 0.5);
    equation
        .narrative_coherence
        .add_future_scenario("future".to_string(), 3, 0.7, 0.8);

    equation.social_embedding.update_self_model(
        vec!["goal".to_string()],
        vec!["belief".to_string()],
        0.5,
    );

    let inputs = ConsciousnessInputs {
        phi: 0.6,
        broadcast: 0.6,
        working_memory: 0.6,
        attention: 0.6,
        recurrence: 0.6,
        embodiment: 0.6,
        knowledge: 0.6,
        synchrony: 0.8,
    };

    let result = equation.compute(&inputs);

    // All new factors should be computed
    assert!(result.embodiment_factor > 0.0);
    assert!(result.narrative_coherence > 0.0);
    assert!(result.social_embedding > 0.0);
}

#[test]
fn test_describe_state() {
    let mut equation = MasterConsciousnessEquation::default();

    let inputs = ConsciousnessInputs {
        phi: 0.7,
        broadcast: 0.7,
        working_memory: 0.7,
        attention: 0.7,
        recurrence: 0.7,
        embodiment: 0.7,
        knowledge: 0.7,
        synchrony: 0.8,
    };

    let result = equation.compute(&inputs);
    let description = equation.describe_state(&result);

    assert!(!description.is_empty());
    assert!(description.contains("Consciousness"));
    assert!(description.contains("bottleneck"));
}

#[test]
fn test_consciousness_trend() {
    let mut equation = MasterConsciousnessEquation::default();

    // First compute with low values
    for _ in 0..5 {
        let inputs = ConsciousnessInputs {
            phi: 0.3,
            broadcast: 0.3,
            working_memory: 0.3,
            attention: 0.3,
            recurrence: 0.3,
            embodiment: 0.3,
            knowledge: 0.3,
            synchrony: 0.5,
        };
        equation.compute(&inputs);
    }

    // Then compute with high values
    for _ in 0..5 {
        let inputs = ConsciousnessInputs {
            phi: 0.8,
            broadcast: 0.8,
            working_memory: 0.8,
            attention: 0.8,
            recurrence: 0.8,
            embodiment: 0.8,
            knowledge: 0.8,
            synchrony: 0.9,
        };
        equation.compute(&inputs);
    }

    // Trend should be positive
    let trend = equation.consciousness_trend();
    assert!(trend > 0.0);
}

#[test]
#[allow(clippy::field_reassign_with_default)]
fn test_disabled_factors() {
    let mut config = MasterEquationConfig::default();
    config.enable_embodiment_factor = false;
    config.enable_narrative_factor = false;
    config.enable_social_factor = false;

    let mut equation = MasterConsciousnessEquation::new(config);

    let inputs = ConsciousnessInputs {
        phi: 0.6,
        broadcast: 0.6,
        working_memory: 0.6,
        attention: 0.6,
        recurrence: 0.6,
        embodiment: 0.6,
        knowledge: 0.6,
        synchrony: 0.8,
    };

    let result = equation.compute(&inputs);

    // Disabled factors should be 1.0 (no effect)
    assert!((result.embodiment_factor - 1.0).abs() < 0.001);
    assert!((result.narrative_coherence - 1.0).abs() < 0.001);
    assert!((result.social_embedding - 1.0).abs() < 0.001);
}

// ========================================================================
// ENHANCED EMBODIMENT FACTOR TESTS
// ========================================================================

#[test]
fn test_sensorimotor_prediction_extended() {
    let mut ef = EmbodimentFactor::new();

    // Record multi-dimensional predictions
    ef.record_sensorimotor_prediction_extended(
        0.5,                // motor command
        &[0.5, 0.6, 0.4],   // predicted sensory
        &[0.5, 0.55, 0.45], // actual sensory (close to predicted)
        Some(0.9),          // good proprioceptive feedback
    );

    let m = ef.compute();
    assert!(m > 0.0 && m <= 1.0);

    // Sensorimotor accuracy should be high after good prediction
    assert!(ef.sensorimotor_accuracy() > 0.5);
}

#[test]
fn test_sensorimotor_contingency() {
    let mut ef = EmbodimentFactor::new();

    // Record a sequence of predictions with consistent pattern
    for i in 0..20 {
        let motor = (i as f64) / 20.0;
        let outcome = motor * 0.9; // Strong contingency
        ef.record_prediction(motor, outcome);
    }

    let contingency = ef.compute_sensorimotor_contingency();
    assert!(contingency > 0.5); // Should show high contingency
}

#[test]
fn test_interoceptive_multisystem() {
    let mut ef = EmbodimentFactor::new();

    let expected = InteroceptiveState::new(0.6, 0.7, 0.5, 0.8);
    let actual = InteroceptiveState::new(0.65, 0.68, 0.52, 0.78);

    // Run multiple updates to allow EMA to converge (smoothing factor is 0.1)
    for _ in 0..20 {
        ef.update_interoceptive_multisystem(&expected, &actual);
    }

    // Coherence should be high since states are close
    assert!(ef.interoceptive_coherence() > 0.7);
}

#[test]
fn test_allostatic_error() {
    let mut ef = EmbodimentFactor::new();

    // Record consistent predictions (low variance)
    for _ in 0..10 {
        ef.record_prediction(0.5, 0.52); // Small consistent error
    }

    let allostatic = ef.compute_allostatic_error();
    assert!(allostatic > 0.5); // Low variance = good allostatic prediction
}

#[test]
fn test_embodiment_diagnostics() {
    let mut ef = EmbodimentFactor::new();

    ef.record_prediction(0.5, 0.5);
    ef.update_interoceptive(0.6, 0.6);

    let diag = ef.diagnostics();
    assert!(diag.embodiment_factor > 0.0);
    assert_eq!(diag.prediction_count, 1);
}

// ========================================================================
// ENHANCED NARRATIVE COHERENCE TESTS
// ========================================================================

#[test]
fn test_episode_with_embedding() {
    let mut nc = NarrativeCoherence::new();

    nc.add_episode_with_embedding(
        "First experience".to_string(),
        0.7,
        &[0.1, 0.2, 0.3, 0.4],
        vec!["learning".to_string(), "growth".to_string()],
    );

    nc.add_episode_with_embedding(
        "Related experience".to_string(),
        0.6,
        &[0.15, 0.25, 0.28, 0.38],
        vec!["learning".to_string()],
    );

    assert_eq!(nc.episode_count(), 2);
    assert!(nc.autobiographical_integration() > 0.0);
}

#[test]
fn test_narrative_arc_coherence() {
    let mut nc = NarrativeCoherence::new();

    // Add episodes with emotional arc (starts positive, dips, recovers)
    nc.add_episode("Beginning".to_string(), 0.6);
    nc.add_episode("Challenge".to_string(), 0.2);
    nc.add_episode("Struggle".to_string(), -0.1);
    nc.add_episode("Turning point".to_string(), 0.3);
    nc.add_episode("Resolution".to_string(), 0.7);
    nc.add_episode("Growth".to_string(), 0.8);

    let arc_coherence = nc.compute_narrative_arc_coherence();
    assert!(arc_coherence > 0.0);
}

#[test]
fn test_salient_episodes() {
    let mut nc = NarrativeCoherence::new();

    nc.add_episode("Low relevance".to_string(), 0.1);
    nc.add_episode("Medium relevance".to_string(), 0.5);
    nc.add_episode("High relevance".to_string(), 0.8);

    let salient = nc.salient_episodes(2);
    assert_eq!(salient.len(), 2);
}

#[test]
fn test_branching_scenario() {
    let mut nc = NarrativeCoherence::new();

    let branches = vec![
        SimulationBranch::new("Success path".to_string(), 5, 0.6, 0.9),
        SimulationBranch::new("Alternative path".to_string(), 7, 0.3, 0.5),
        SimulationBranch::new("Unlikely path".to_string(), 10, 0.1, 0.2),
    ];

    nc.add_branching_scenario("Main goal".to_string(), 5, branches);

    assert!(nc.scenario_count() >= 2); // Should have main + at least one alternative
}

#[test]
fn test_prospective_capability() {
    let mut nc = NarrativeCoherence::new();

    nc.add_future_scenario("Near term".to_string(), 2, 0.7, 0.6);
    nc.add_future_scenario("Medium term".to_string(), 5, 0.5, 0.8);
    nc.add_future_scenario("Long term".to_string(), 10, 0.3, 0.9);

    let capability = nc.compute_prospective_capability();
    assert!(capability > 0.0 && capability <= 1.0);
}

#[test]
fn test_mental_time_travel() {
    let mut nc = NarrativeCoherence::new();

    // Add past episodes with causal links
    nc.add_episode("Past 1".to_string(), 0.5);
    nc.add_episode("Past 2".to_string(), 0.5);
    nc.add_episode("Past 3".to_string(), 0.5);

    // Add future scenarios
    nc.add_future_scenario("Future 1".to_string(), 5, 0.7, 0.8);
    nc.add_future_scenario("Future 2".to_string(), 8, 0.5, 0.6);

    let mtt = nc.compute_mental_time_travel();
    assert!(mtt > 0.0);
}

#[test]
fn test_narrative_diagnostics() {
    let mut nc = NarrativeCoherence::new();

    nc.add_episode("Event 1".to_string(), 0.5);
    nc.add_future_scenario("Plan 1".to_string(), 3, 0.7, 0.8);

    let diag = nc.diagnostics();
    assert_eq!(diag.episode_count, 1);
    assert_eq!(diag.scenario_count, 1);
    assert!(diag.narrative_coherence > 0.0);
}
